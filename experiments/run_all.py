#!/usr/bin/env python3
"""Master runner for all experiments.

    conda run -n isaaclab python -m experiments.run_all
    conda run -n isaaclab python -m experiments.run_all --experiments e01 e06 e07
    conda run -n isaaclab python -m experiments.run_all --skip e10
    conda run -n isaaclab python -m experiments.run_all --list

"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _free_gpu_memory() -> None:
    try:
        import cupy as cp

        from experiments.common.runner import free_gpu_memory

        free_gpu_memory()
    except Exception:
        pass


ALL_EXPERIMENTS = [
    "e01_sampling_strategy",
    "e02_candidate_scaling",
    "e03_iterative_sampler_params",
    "e04_curvature_sensitivity",
    "e05_visibility_comparison",
    "e06_set_cover_optimizers",
    "e07_vrp_fleet_scaling",
    "e08_vrp_alpha_blending",
    "e09_sampler_routing_impact",
    "e10_cross_model",
]


def run_experiment(name: str, resume: bool = False) -> bool:
    """Import and run one experiment's main(). Under ``resume=True`` exceptions propagate so an
    outer ``while ! ...; do ...; done`` loop can recover from CUDA OOM."""
    module_name = f"experiments.{name}"
    logger.info("=" * 70)
    logger.info("RUNNING: %s%s", name, " (--resume)" if resume else "")
    logger.info("=" * 70)
    t0 = time.perf_counter()
    old_argv = sys.argv
    try:
        child_argv = [module_name]
        if resume:
            child_argv.append("--resume")
        sys.argv = child_argv
        mod = importlib.import_module(module_name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            logger.warning("  %s has no main() function, skipping", name)
        elapsed = time.perf_counter() - t0
        logger.info("DONE: %s in %.1fs", name, elapsed)
        return True
    except BaseException as e:
        logger.error("FAILED: %s: %s", name, e, exc_info=not isinstance(e, SystemExit))
        if resume:
            raise
        return False
    finally:
        sys.argv = old_argv
        _free_gpu_memory()


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment_subprocess(name: str, log_dir: Path, resume: bool = False) -> bool:
    """Run an experiment in its own subprocess. stdout/stderr -> log_dir/{name}.log so parallel
    runs don't interleave. ``sys.executable`` is the isaaclab Python when launched via conda run."""
    log_path = log_dir / f"{name}.log"
    logger.info(
        "STARTING (subprocess): %s%s  →  %s", name, " (--resume)" if resume else "", log_path
    )
    t0 = time.perf_counter()
    argv = [sys.executable, "-m", f"experiments.{name}"]
    if resume:
        argv.append("--resume")
    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            argv,
            cwd=_PROJECT_ROOT,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - t0
    if result.returncode == 0:
        logger.info("DONE: %s in %.1fs", name, elapsed)
        return True
    else:
        logger.error(
            "FAILED: %s (exit %d) in %.1fs — see %s", name, result.returncode, elapsed, log_path
        )
        return False


def build_waves(to_run: list[str], max_jobs: int, gpu_budget: int) -> list[list[str]]:
    """Greedy bin-pack experiments into parallel waves (heaviest-first to reduce fragmentation).
    A new wave opens when the next experiment would exceed gpu_budget or max_jobs. An experiment
    whose own weight exceeds gpu_budget runs solo with a warning."""
    sorted_exps = sorted(
        to_run,
        key=lambda n: EXPERIMENT_WEIGHTS.get(n, 1),
        reverse=True,
    )

    waves: list[list[str]] = []
    current_wave: list[str] = []
    current_weight = 0

    for name in sorted_exps:
        w = EXPERIMENT_WEIGHTS.get(name, 1)
        if w > gpu_budget:
            logger.warning(
                "Experiment %s has weight %d > gpu_budget %d; running solo",
                name,
                w,
                gpu_budget,
            )
            if current_wave:
                waves.append(current_wave)
                current_wave, current_weight = [], 0
            waves.append([name])
            continue

        if current_weight + w <= gpu_budget and len(current_wave) < max_jobs:
            current_wave.append(name)
            current_weight += w
        else:
            if current_wave:
                waves.append(current_wave)
            current_wave, current_weight = [name], w

    if current_wave:
        waves.append(current_wave)

    logger.info(
        "Parallel plan: %d wave(s) for %d experiments (--jobs %d, --gpu-budget %d):",
        len(waves),
        len(to_run),
        max_jobs,
        gpu_budget,
    )
    for i, wave in enumerate(waves):
        total_w = sum(EXPERIMENT_WEIGHTS.get(n, 1) for n in wave)
        logger.info("  Wave %d: %s  [%d/%d units]", i + 1, wave, total_w, gpu_budget)

    return waves


def run_parallel(to_run: list[str], max_jobs: int, gpu_budget: int, resume: bool = False) -> bool:
    """Run waves respecting GPU budget. Under ``resume=True`` a failed wave aborts further waves
    so an outer restart loop can relaunch with a fresh GPU context."""
    log_dir = Path(__file__).resolve().parent / "results" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    waves = build_waves(to_run, max_jobs, gpu_budget)
    t_total = time.perf_counter()
    all_ok = True

    for i, wave in enumerate(waves):
        logger.info("=" * 70)
        logger.info("WAVE %d/%d  (%d experiment(s)): %s", i + 1, len(waves), len(wave), wave)
        t_wave = time.perf_counter()
        wave_ok = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {
                pool.submit(run_experiment_subprocess, name, log_dir, resume): name for name in wave
            }
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    ok = fut.result()
                except Exception as exc:
                    logger.error("Unexpected exception in %s: %s", name, exc, exc_info=True)
                    ok = False
                if not ok:
                    wave_ok = False

        logger.info("WAVE %d done in %.1fs", i + 1, time.perf_counter() - t_wave)

        if not wave_ok:
            all_ok = False
            if resume:
                logger.error(
                    "Wave %d had failures; aborting remaining waves under --resume so an outer "
                    "restart loop can recover.",
                    i + 1,
                )
                break

    logger.info("=" * 70)
    logger.info("ALL DONE: %d experiments in %.1fs", len(to_run), time.perf_counter() - t_total)
    return all_ok


def main():
    p = argparse.ArgumentParser(description="Run all or selected experiments")
    p.add_argument(
        "--experiments", nargs="+", default=None, help="Run only these experiments (e.g. e01 e04)"
    )
    p.add_argument("--skip", nargs="+", default=[], help="Skip these experiments")
    p.add_argument("--list", action="store_true", help="List all experiments")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Max experiments to run concurrently (default 1 = sequential).",
    )
    p.add_argument(
        "--gpu-budget",
        type=int,
        default=6,
        help="Max concurrent GPU-memory units (1 unit ≈ 2 GB). Default 6 ≈ 12 GB. "
        "Increase to 9 for ~4× speedup with ~5 GB buffer.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to each child (skip rows whose result JSON exists) and exit non-zero "
        "on first failure so an outer `while ! run_all.py --resume; do sleep 5; done` loop "
        "can restart after CUDA OOM.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.list:
        for name in ALL_EXPERIMENTS:
            print(f"  {name}")
        return

    if args.experiments:
        to_run = []
        for exp in args.experiments:
            matches = [n for n in ALL_EXPERIMENTS if n.startswith(exp)]
            if matches:
                to_run.extend(matches)
            else:
                logger.warning("Unknown experiment: %s", exp)
    else:
        to_run = ALL_EXPERIMENTS

    skip_full = []
    for s in args.skip:
        skip_full.extend(n for n in ALL_EXPERIMENTS if n.startswith(s))
    to_run = [n for n in to_run if n not in skip_full]

    logger.info("Running %d experiments: %s", len(to_run), to_run)

    if args.jobs > 1:
        ok = run_parallel(to_run, args.jobs, args.gpu_budget, resume=args.resume)
        if not ok:
            sys.exit(1)
    else:
        t_total = time.perf_counter()
        all_ok = True
        try:
            for name in to_run:
                if not run_experiment(name, resume=args.resume):
                    all_ok = False
                    # Without --resume, continue past failures (legacy). With --resume,
                    # run_experiment re-raises, so we never reach here.
        except BaseException as e:
            logger.error("Aborting run_all under --resume: %s", e)
            sys.exit(1)
        logger.info("=" * 70)
        logger.info("ALL DONE: %d experiments in %.1fs", len(to_run), time.perf_counter() - t_total)
        if not all_ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
