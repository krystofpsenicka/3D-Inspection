"""ExperimentRunner base class with seed management, timing, and skip logic."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator

import cupy as cp
import numpy as np

from .persistence import save_run_result

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Memory instrumentation (process-wide singletons)
#
# The previous implementation only logged CuPy's own pool, which is near-zero
# whenever cuGraph/cuOpt have installed rmm_cupy_allocator — that routes every
# CuPy allocation through RMM, so CuPy's pool metrics no longer reflect real
# usage.  We now report all allocators side-by-side and wrap RMM's current MR
# with StatisticsResourceAdaptor so we can see RMM's live/peak bytes directly.
# ─────────────────────────────────────────────────────────────────────────────

_RMM_STATS_MR = None   # StatisticsResourceAdaptor, set by _install_memory_instrumentation
_NVML_READY = False    # pynvml initialized?
_MY_PID = os.getpid()


def _ensure_rmm_stats_wrapped() -> None:
    """(Re)wrap RMM's current device MR with a statistics adaptor if needed.

    cuGraph / cuOpt call ``rmm.reinitialize`` on import, which replaces the
    device MR and silently discards any adaptor we previously installed.  We
    therefore check on every snapshot that the current MR is still our adaptor
    and rewrap if not.  Already-outstanding allocations keep their original MR
    (that's fine — they'll be freed through it); any allocations made *after*
    the rewrap are counted.
    """
    global _RMM_STATS_MR
    try:
        import rmm
        current = rmm.mr.get_current_device_resource()
        if isinstance(current, rmm.mr.StatisticsResourceAdaptor):
            _RMM_STATS_MR = current
            return
        _RMM_STATS_MR = rmm.mr.StatisticsResourceAdaptor(current)
        rmm.mr.set_current_device_resource(_RMM_STATS_MR)
    except Exception as e:
        logger.debug("RMM stats adaptor not installed: %s", e)


def _ensure_nvml_ready() -> None:
    global _NVML_READY
    if _NVML_READY:
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        _NVML_READY = True
    except Exception as e:
        logger.debug("pynvml not available: %s", e)


_ensure_rmm_stats_wrapped()
_ensure_nvml_ready()


class Timer:
    """Wall-clock timer with GPU sync."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        cp.cuda.Stream.null.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        cp.cuda.Stream.null.synchronize()
        self.elapsed = time.perf_counter() - self._start


@contextmanager
def timed() -> Iterator[Timer]:
    """Context manager that records elapsed wall-clock time (GPU-synced)."""
    t = Timer()
    t.__enter__()
    try:
        yield t
    finally:
        t.__exit__(None, None, None)


def set_seed(seed: int) -> None:
    """Set numpy + cupy random seeds for reproducibility."""
    np.random.seed(seed)
    cp.random.seed(seed)


def _log_memory_snapshot(tag: str) -> None:
    """Emit one multi-allocator memory line.  Safe to call whenever.

    Layout (per allocator, MB):
      nvml:  free / used (proc) / total
      cupy:  used / total       (CuPy's own pool — ~0 when RMM owns CuPy)
      rmm:   current / peak     (from StatisticsResourceAdaptor if installed)
      torch: alloc / reserved   (if torch is importable)
      host:  rss                (current Python process)
      alloc: rmm? true/false    (is CuPy's allocator rmm_cupy_allocator)
    """
    _ensure_rmm_stats_wrapped()
    _ensure_nvml_ready()
    parts: list[str] = []
    try:
        free_b, total_b = cp.cuda.Device().mem_info
        parts.append(f"nvml-dev={free_b/1e6:.0f}F/{total_b/1e6:.0f}T")
    except Exception:
        pass
    if _NVML_READY:
        try:
            import pynvml
            h = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
            mine = 0
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if proc.pid == _MY_PID and proc.usedGpuMemory is not None:
                    mine = proc.usedGpuMemory
            parts.append(f"nvml-proc={mine/1e6:.0f}")
        except Exception:
            pass
    try:
        pool = cp.get_default_memory_pool()
        parts.append(f"cupy={pool.used_bytes()/1e6:.0f}U/{pool.total_bytes()/1e6:.0f}T")
    except Exception:
        pass
    if _RMM_STATS_MR is not None:
        try:
            s = _RMM_STATS_MR.allocation_counts
            parts.append(f"rmm={s.current_bytes/1e6:.0f}C/{s.peak_bytes/1e6:.0f}P")
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            parts.append(
                f"torch={torch.cuda.memory_allocated()/1e6:.0f}A/"
                f"{torch.cuda.memory_reserved()/1e6:.0f}R"
            )
    except ImportError:
        pass
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
        parts.append(f"rss={rss/1e6:.0f}")
    except ImportError:
        pass
    try:
        import rmm.allocators.cupy as _rac
        parts.append(f"alloc-rmm={cp.cuda.get_allocator() is _rac.rmm_cupy_allocator}")
    except Exception:
        pass
    logger.info("[MEM %s] %s", tag, "  ".join(parts))


def log_memory(tag: str = "snap") -> None:
    """Public helper: emit a memory snapshot line without freeing anything."""
    _log_memory_snapshot(tag)


def dump_live_gpu_arrays(top_n: int = 30) -> None:
    """Log the biggest CuPy arrays currently alive, to identify leak owners.

    For each of the top N arrays, log (nbytes, shape, dtype).  Then try to
    attribute ownership via one level of ``gc.get_referrers`` (class name +
    attribute key if it is an attribute of an object).
    """
    import gc
    gc.collect()
    try:
        arrs = [o for o in gc.get_objects() if isinstance(o, cp.ndarray)]
    except Exception as e:
        logger.warning("[MEM dump] gc scan failed: %s", e)
        return
    arrs.sort(key=lambda a: a.nbytes, reverse=True)
    total = sum(a.nbytes for a in arrs)
    logger.info("[MEM dump] %d live cupy arrays, total %.0f MB", len(arrs), total / 1e6)
    for i, a in enumerate(arrs[:top_n]):
        owners = []
        try:
            for r in gc.get_referrers(a)[:3]:
                if isinstance(r, dict):
                    # Find which key maps to this array (common case: __dict__)
                    for k, v in r.items():
                        if v is a:
                            owners.append(f"dict[{k!r}]")
                            break
                else:
                    owners.append(type(r).__name__)
        except Exception:
            pass
        logger.info(
            "  #%d %.0f MB  shape=%s dtype=%s  via=%s",
            i, a.nbytes / 1e6, tuple(a.shape), a.dtype, ",".join(owners) or "?",
        )


def free_gpu_memory(dump_arrays: bool = False) -> None:
    """Return idle GPU memory to CUDA and log before/after allocator snapshots.

    Three memory subsystems are reset:

    1. CuPy pool — ``free_all_blocks()`` releases cached (freed) blocks to CUDA.
       Only meaningful when CuPy is NOT routed through RMM.
    2. RMM pool — ``rmm.reinitialize(pool_allocator=True, ...)`` replaces the
       pool with a fresh one.  This only shrinks true live usage if all CuPy
       arrays / cuDF / cuGraph objects allocated through the old pool are dead;
       any surviving array keeps the old pool alive.  The Statistics adaptor is
       re-installed on top of the new MR so subsequent snapshots keep reporting
       RMM live/peak bytes.
    3. PyTorch cache — ``empty_cache()`` releases reserved-but-idle blocks.

    We log snapshots ``before-free`` (with the run's live RMM stats still intact)
    and ``after-free`` (fresh pool), so the in-run RMM usage is visible.  Set
    ``dump_arrays=True`` to also dump the top CuPy arrays before reinitialize,
    which identifies objects pinning the old RMM pool.
    """
    import gc
    global _RMM_STATS_MR
    _log_memory_snapshot("before-free")
    if dump_arrays:
        dump_live_gpu_arrays(top_n=20)
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as e:
        logger.debug("CuPy pool free failed: %s", e)
    try:
        import rmm
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=2 ** 28,
        )
        gc.collect()
        # Re-wrap the fresh MR with a statistics adaptor so metrics persist.
        try:
            current = rmm.mr.get_current_device_resource()
            if not isinstance(current, rmm.mr.StatisticsResourceAdaptor):
                _RMM_STATS_MR = rmm.mr.StatisticsResourceAdaptor(current)
                rmm.mr.set_current_device_resource(_RMM_STATS_MR)
            else:
                _RMM_STATS_MR = current
        except Exception as e:
            logger.debug("RMM stats re-wrap failed: %s", e)
    except Exception as e:
        logger.debug("RMM reinitialize skipped: %s", e)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    _log_memory_snapshot("after-free")


_OOM_MARKERS = (
    "out_of_memory",
    "bad_alloc",
    "OutOfMemoryError",
    "Unable to allocate",
    "cudaErrorMemoryAllocation",
    "cudaErrorStreamCaptureInvalidated",  # cuOpt post-OOM stream corruption
)


def is_oom(exc: BaseException) -> bool:
    """True if the exception message looks like a CUDA OOM / RMM failure."""
    msg = str(exc)
    return any(m in msg for m in _OOM_MARKERS)


def handle_row_exception(
    exc: BaseException,
    row_desc: str,
    *,
    resume: bool,
) -> None:
    """Log a per-row experiment failure. If ``resume`` is True and the error
    looks like a CUDA OOM / RMM stream-capture failure, free GPU memory and
    raise ``SystemExit(1)`` so an outer restart loop can reclaim leaked
    device memory with a fresh process.

    All other exceptions are logged and swallowed, matching the long-standing
    per-row behaviour of the experiment scripts.
    """
    logger.error("  FAILED [%s]: %s", row_desc, exc, exc_info=True)
    if resume and is_oom(exc):
        free_gpu_memory()
        logger.error(
            "  Detected CUDA OOM / RMM error under --resume; exiting non-zero "
            "so the outer restart loop can reclaim GPU memory."
        )
        raise SystemExit(1)


class ExperimentRunner(ABC):
    """Base class for running parameter sweep experiments.

    Subclasses implement `run_single(params, seed)` which returns a dict
    of metrics.  The runner handles seed management, result persistence,
    progress logging, and skip-if-exists logic.
    """

    def __init__(self, name: str, output_dir: str | None = None):
        self.name = name
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", name,
        )

    @abstractmethod
    def run_single(self, params: dict, seed: int) -> dict[str, Any]:
        """Run one experiment configuration. Returns dict of metrics."""
        ...

    def _result_path(self, params: dict, seed: int) -> str:
        """Generate unique result path from params + seed."""
        parts = [f"{k}={v}" for k, v in sorted(params.items())]
        parts.append(f"seed={seed}")
        return os.path.join(self.output_dir, "raw", "_".join(parts))

    def run(self, param_grid: list[dict], seeds: list[int],
            skip_existing: bool = True) -> list[dict]:
        """Run all param combos x seeds, saving results incrementally."""
        os.makedirs(os.path.join(self.output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True)

        total = len(param_grid) * len(seeds)
        all_results = []
        completed = 0
        t_start = time.perf_counter()

        for params in param_grid:
            for seed in seeds:
                completed += 1
                result_path = self._result_path(params, seed)

                if skip_existing and os.path.exists(result_path + ".json"):
                    logger.info("[%d/%d] SKIP (exists): %s seed=%d",
                                completed, total, params, seed)
                    continue

                logger.info("[%d/%d] Running: %s seed=%d",
                            completed, total, params, seed)

                set_seed(seed)
                try:
                    with timed() as t:
                        result = self.run_single(params, seed)
                    result["_seed"] = seed
                    result["_params"] = params
                    result["_wall_time"] = t.elapsed
                    all_results.append(result)
                    save_run_result(result, result_path)
                except Exception as e:
                    logger.error("  FAILED: %s", e, exc_info=True)
                    result = {"_seed": seed, "_params": params, "_error": str(e)}
                    all_results.append(result)

                elapsed = time.perf_counter() - t_start
                eta = elapsed / completed * (total - completed)
                logger.info("  Done in %.1fs.  ETA: %.0fs remaining.",
                            t.elapsed if 't' in dir() else 0, eta)

        logger.info("Experiment '%s' complete: %d runs.", self.name, len(all_results))
        return all_results
