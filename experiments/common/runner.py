"""ExperimentRunner base class with seed management, timing, skip logic."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import cupy as cp
import numpy as np

from .persistence import save_run_result

logger = logging.getLogger(__name__)

# Memory instrumentation: we wrap RMM's MR with StatisticsResourceAdaptor
# and report all allocators side-by-side.

_RMM_STATS_MR = None  # StatisticsResourceAdaptor
_NVML_READY = False
_MY_PID = os.getpid()


def _ensure_rmm_stats_wrapped() -> None:
    """(Re)wrap RMM's current device MR with a statistics adaptor."""
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
    t = Timer()
    t.__enter__()
    try:
        yield t
    finally:
        t.__exit__(None, None, None)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    cp.random.seed(seed)


def _log_memory_snapshot(tag: str) -> None:
    """One multi-allocator memory line (MB):
      nvml: free / used (proc) / total
      cupy: used / total       (~0 when RMM owns CuPy)
      rmm:  current / peak     (from StatisticsResourceAdaptor)
      torch: alloc / reserved  (if importable)
      host: rss
      alloc-rmm: true/false    (is CuPy's allocator rmm_cupy_allocator)
    """
    _ensure_rmm_stats_wrapped()
    _ensure_nvml_ready()
    parts: list[str] = []
    try:
        free_b, total_b = cp.cuda.Device().mem_info
        parts.append(f"nvml-dev={free_b / 1e6:.0f}F/{total_b / 1e6:.0f}T")
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
            parts.append(f"nvml-proc={mine / 1e6:.0f}")
        except Exception:
            pass
    try:
        pool = cp.get_default_memory_pool()
        parts.append(f"cupy={pool.used_bytes() / 1e6:.0f}U/{pool.total_bytes() / 1e6:.0f}T")
    except Exception:
        pass
    if _RMM_STATS_MR is not None:
        try:
            s = _RMM_STATS_MR.allocation_counts
            parts.append(f"rmm={s.current_bytes / 1e6:.0f}C/{s.peak_bytes / 1e6:.0f}P")
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(
                f"torch={torch.cuda.memory_allocated() / 1e6:.0f}A/"
                f"{torch.cuda.memory_reserved() / 1e6:.0f}R"
            )
    except ImportError:
        pass
    try:
        import psutil

        rss = psutil.Process().memory_info().rss
        parts.append(f"rss={rss / 1e6:.0f}")
    except ImportError:
        pass
    try:
        import rmm.allocators.cupy as _rac

        parts.append(f"alloc-rmm={cp.cuda.get_allocator() is _rac.rmm_cupy_allocator}")
    except Exception:
        pass
    logger.info("[MEM %s] %s", tag, "  ".join(parts))


def log_memory(tag: str = "snap") -> None:
    _log_memory_snapshot(tag)


def dump_live_gpu_arrays(top_n: int = 30) -> None:
    """Log the biggest live CuPy arrays + one level of referrer attribution."""
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
    """Return idle GPU memory to CUDA. Resets CuPy pool, RMM pool, torch cache."""
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

        rmm.reinitialize(pool_allocator=True, initial_pool_size=2**28)
        gc.collect()
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
    msg = str(exc)
    return any(m in msg for m in _OOM_MARKERS)


def handle_row_exception(exc: BaseException, row_desc: str, *, resume: bool) -> None:
    """Log a per-row failure. Under --resume on OOM/RMM errors, free GPU
    memory and exit non-zero so an outer loop can restart with a clean process.
    """
    logger.error("  FAILED [%s]: %s", row_desc, exc, exc_info=True)
    if resume and is_oom(exc):
        free_gpu_memory()
        logger.error(
            "  Detected CUDA OOM / RMM error under --resume; exiting non-zero."
        )
        raise SystemExit(1)


class ExperimentRunner(ABC):
    """Base for parameter-sweep experiments.

    Subclasses implement run_single(params, seed) -> dict of metrics.
    Handles seed management, persistence, progress logging, skip-if-exists.
    """

    def __init__(self, name: str, output_dir: str | None = None):
        self.name = name
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", name,
        )

    @abstractmethod
    def run_single(self, params: dict, seed: int) -> dict[str, Any]:
        ...

    def _result_path(self, params: dict, seed: int) -> str:
        parts = [f"{k}={v}" for k, v in sorted(params.items())]
        parts.append(f"seed={seed}")
        return os.path.join(self.output_dir, "raw", "_".join(parts))

    def run(
        self, param_grid: list[dict], seeds: list[int], skip_existing: bool = True
    ) -> list[dict]:
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
                    logger.info("[%d/%d] SKIP (exists): %s seed=%d", completed, total, params, seed)
                    continue

                logger.info("[%d/%d] Running: %s seed=%d", completed, total, params, seed)

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
                logger.info(
                    "  Done in %.1fs.  ETA: %.0fs remaining.",
                    t.elapsed if "t" in dir() else 0, eta,
                )

        logger.info("Experiment '%s' complete: %d runs.", self.name, len(all_results))
        return all_results
