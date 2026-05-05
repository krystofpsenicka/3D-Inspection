#!/usr/bin/env python3
"""Master runner for all experiments.

    python -m experiments.run_all
    python -m experiments.run_all --experiments e01 e06 e07
    python -m experiments.run_all --skip e10
    python -m experiments.run_all --list

"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time

logger = logging.getLogger(__name__)


def _free_gpu_memory() -> None:
    try:
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


def main():
    p = argparse.ArgumentParser(description="Run all or selected experiments")
    p.add_argument(
        "--experiments", nargs="+", default=None, help="Run only these experiments (e.g. e01 e04)"
    )
    p.add_argument("--skip", nargs="+", default=[], help="Skip these experiments")
    p.add_argument("--list", action="store_true", help="List all experiments")
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
