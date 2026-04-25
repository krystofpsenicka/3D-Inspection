#!/usr/bin/env python3
"""Master runner for all experiments.

Usage:
    conda run -n isaaclab python -m experiments.run_all
    conda run -n isaaclab python -m experiments.run_all --experiments e01 e04 e06
    conda run -n isaaclab python -m experiments.run_all --skip e05 e12
    conda run -n isaaclab python -m experiments.run_all --list

Parallel execution (subprocess-based, full CUDA-context isolation):
    conda run -n isaaclab python -m experiments.run_all --jobs 2 --gpu-budget 6
    conda run -n isaaclab python -m experiments.run_all --jobs 4 --gpu-budget 9

--gpu-budget N means at most N memory-units may run concurrently (1 unit ≈ 2 GB GPU).
Default budget 6 ≈ 12 GB, conservative for an RTX 3090.
Logs for parallel runs go to experiments/results/_logs/{name}.log.
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
    """Free CuPy memory pool between experiments."""
    try:
        import cupy as cp
        from experiments.common.runner import free_gpu_memory
        free_gpu_memory()
    except Exception:
        pass

ALL_EXPERIMENTS = [
    
    "e19_vrp_time_limit_sweep",
    "e01_sampling_strategy",
    "e02_candidate_scaling",
    "e03_visibility_comparison",
    "e15_cross_model",
    "e04_set_cover_optimizers",
    "e05_set_cover_scaling",
    "e06_vrp_fleet_scaling",
    "e07_vrp_alpha_blending",
    "e10_sta_resolution",
    "e17_sampler_routing_impact",
    "e14_curvature_sensitivity",

]

# GPU memory weight per experiment (1 unit ≈ 2 GB peak VRAM).
# Used by the parallel scheduler to ensure concurrent weights ≤ --gpu-budget.
# Tune these if you observe higher/lower actual usage via `nvidia-smi`.
EXPERIMENT_WEIGHTS: dict[str, int] = {
    "e01_sampling_strategy":     2,  # visibility matrix 1500×200K pts (~1.2 GB) + overhead
    "e02_candidate_scaling":     3,  # scales to 5000 candidates → ~4 GB for V alone
    "e03_visibility_comparison": 1,  # multiple smaller matrices, comparison runs
    "e04_set_cover_optimizers":  2,  # GPU set cover + visibility, ~3–4 GB
    "e05_set_cover_scaling":     1,  # moderate problem sizes
    "e06_vrp_fleet_scaling":     4,  # OG + distance matrix, ~6–8 GB (had OOM at 0.10 m)
    "e07_vrp_alpha_blending":    2,  # distance matrix + VRP solve, ~3–4 GB
    "e08_vrp_solver_comparison": 1,  # small problems, ~2 GB
    "e09_mapf_priority":         2,  # MAPF trajectories + distance matrix
    "e10_sta_resolution":        2,  # Space-Time A*, moderate memory
    "e11_end_to_end":            4,  # all pipeline stages combined, ~6–8 GB
    "e12_gpu_vs_cpu_scaling":    2,  # visibility scaling up to 2000×200K
    "e13_viewpoint_redundancy":  3,  # k=5 → 7500 candidates × 200K pts = ~6 GB for V
    "e14_curvature_sensitivity": 1,  # parameter sweep, light
    "e15_cross_model":           4,  # 10 models with full pipeline, ~6–8 GB
    "e16_frustum_sensitivity":   1,  # frustum parameter sweep, moderate
    "e19_vrp_time_limit_sweep":  2,  # VRP time limit sweep, ~3–4 GB
}


def run_experiment(name: str, resume: bool = False) -> bool:
    """Import and run a single experiment's main(). Returns True on success.

    Under ``resume=True`` the child sees ``--resume`` in its argv, and any
    exception (including ``SystemExit``) raised by ``main()`` is propagated up
    so the caller can abort the whole sequence — this is what lets an outer
    ``while ! ...; do ...; done`` loop recover from CUDA OOM.
    """
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
        logger.error("FAILED: %s: %s", name, e,
                     exc_info=not isinstance(e, SystemExit))
        if resume:
            raise
        return False
    finally:
        sys.argv = old_argv
        _free_gpu_memory()


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment_subprocess(name: str, log_dir: Path,
                              resume: bool = False) -> bool:
    """Run a single experiment in its own subprocess. Returns True on success.

    stdout/stderr are captured to log_dir/{name}.log so parallel runs don't
    interleave in the terminal.  sys.executable is already the isaaclab conda
    Python when the parent process was launched via `conda run -n isaaclab`.
    """
    log_path = log_dir / f"{name}.log"
    logger.info("STARTING (subprocess): %s%s  →  %s",
                name, " (--resume)" if resume else "", log_path)
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
        logger.error("FAILED: %s (exit %d) in %.1fs — see %s",
                     name, result.returncode, elapsed, log_path)
        return False


def build_waves(to_run: list[str], max_jobs: int, gpu_budget: int) -> list[list[str]]:
    """Greedy bin-packing of experiments into parallel waves.

    Experiments are sorted heaviest-first to reduce fragmentation.  A new wave
    is opened whenever the next experiment would exceed either the gpu_budget or
    the max_jobs cap.  An experiment whose own weight exceeds gpu_budget runs
    solo with a warning.
    """
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
                name, w, gpu_budget,
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

    logger.info("Parallel plan: %d wave(s) for %d experiments "
                "(--jobs %d, --gpu-budget %d):",
                len(waves), len(to_run), max_jobs, gpu_budget)
    for i, wave in enumerate(waves):
        total_w = sum(EXPERIMENT_WEIGHTS.get(n, 1) for n in wave)
        logger.info("  Wave %d: %s  [%d/%d units]", i + 1, wave, total_w, gpu_budget)

    return waves


def run_parallel(to_run: list[str], max_jobs: int, gpu_budget: int,
                 resume: bool = False) -> bool:
    """Run experiments in parallel waves, respecting the GPU memory budget.

    Returns True if every experiment succeeded, False otherwise. Under
    ``resume=True`` a failed wave aborts further waves so an outer restart
    loop can relaunch with a fresh GPU context.
    """
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
                pool.submit(run_experiment_subprocess, name, log_dir, resume): name
                for name in wave
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
                logger.error("Wave %d had failures; aborting remaining waves under "
                             "--resume so an outer restart loop can recover.", i + 1)
                break

    logger.info("=" * 70)
    logger.info("ALL DONE: %d experiments in %.1fs",
                len(to_run), time.perf_counter() - t_total)
    return all_ok


def main():
    p = argparse.ArgumentParser(description="Run all or selected experiments")
    p.add_argument("--experiments", nargs="+", default=None,
                   help="Run only these experiments (e.g. e01 e04)")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Skip these experiments")
    p.add_argument("--list", action="store_true", help="List all experiments")
    p.add_argument("--jobs", type=int, default=1,
                   help="Max experiments to run concurrently (default: 1 = sequential)")
    p.add_argument("--gpu-budget", type=int, default=6,
                   help="Max concurrent GPU-memory units (1 unit ≈ 2 GB). "
                        "Default 6 ≈ 12 GB — conservative for an RTX 3090. "
                        "Increase to 9 for ~4× speedup with ~5 GB buffer.")
    p.add_argument("--resume", action="store_true",
                   help="Pass --resume to each child experiment (skip rows "
                        "whose result JSON already exists) and exit non-zero "
                        "on the first failure, so an outer "
                        "`while ! run_all.py --resume; do sleep 5; done` loop "
                        "can restart after CUDA OOM with a fresh process.")
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
        # Match short names like "e01" to full names
        to_run = []
        for exp in args.experiments:
            matches = [n for n in ALL_EXPERIMENTS if n.startswith(exp)]
            if matches:
                to_run.extend(matches)
            else:
                logger.warning("Unknown experiment: %s", exp)
    else:
        to_run = ALL_EXPERIMENTS

    # Apply skip
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
                    # Without --resume, continue past failures (legacy behaviour).
                    # With --resume, run_experiment re-raises, so we never reach here.
        except BaseException as e:
            # --resume path: first failing experiment propagated an exception.
            logger.error("Aborting run_all under --resume: %s", e)
            sys.exit(1)
        logger.info("=" * 70)
        logger.info("ALL DONE: %d experiments in %.1fs",
                    len(to_run), time.perf_counter() - t_total)
        if not all_ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
