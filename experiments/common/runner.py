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


def free_gpu_memory() -> None:
    """Return idle GPU memory to CUDA and log available GPU memory.

    Call this after each experiment run. CuPy caches freed arrays in its own
    pool and does not return them to CUDA automatically. PyTorch similarly caches
    freed CUDA tensors; empty_cache() returns those to CUDA without touching any
    live tensors (e.g. visibility query BVH structures remain unaffected).
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    free_bytes, total_bytes = cp.cuda.Device().mem_info
    logger.info(
        "[GPU] Free: %.0f MB / %.0f MB total  (CuPy pool used: %.0f MB)",
        free_bytes / 1e6,
        total_bytes / 1e6,
        cp.get_default_memory_pool().used_bytes() / 1e6,
    )


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
