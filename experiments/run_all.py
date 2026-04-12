#!/usr/bin/env python3
"""Master runner for all experiments.

Usage:
    conda run -n isaaclab python -m experiments.run_all
    conda run -n isaaclab python -m experiments.run_all --experiments e01 e04 e06
    conda run -n isaaclab python -m experiments.run_all --skip e05 e12
    conda run -n isaaclab python -m experiments.run_all --list
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time

logger = logging.getLogger(__name__)

ALL_EXPERIMENTS = [
    "e01_sampling_strategy",
    "e02_candidate_scaling",
    "e03_visibility_comparison",
    "e04_set_cover_optimizers",
    "e05_set_cover_scaling",
    "e06_vrp_fleet_scaling",
    "e07_vrp_alpha_blending",
    "e08_vrp_solver_comparison",
    "e09_mapf_priority",
    "e10_sta_resolution",
    "e11_end_to_end",
    "e12_gpu_vs_cpu_scaling",
    "e13_viewpoint_redundancy",
    "e14_curvature_sensitivity",
    "e15_cross_model",
]


def run_experiment(name: str):
    """Import and run a single experiment's main()."""
    module_name = f"experiments.{name}"
    logger.info("=" * 70)
    logger.info("RUNNING: %s", name)
    logger.info("=" * 70)
    t0 = time.perf_counter()
    try:
        # Override sys.argv so the experiment sees no args (uses defaults)
        old_argv = sys.argv
        sys.argv = [module_name]
        mod = importlib.import_module(module_name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            logger.warning("  %s has no main() function, skipping", name)
        sys.argv = old_argv
        elapsed = time.perf_counter() - t0
        logger.info("DONE: %s in %.1fs", name, elapsed)
    except Exception as e:
        sys.argv = old_argv
        logger.error("FAILED: %s: %s", name, e, exc_info=True)


def main():
    p = argparse.ArgumentParser(description="Run all or selected experiments")
    p.add_argument("--experiments", nargs="+", default=None,
                   help="Run only these experiments (e.g. e01 e04)")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Skip these experiments")
    p.add_argument("--list", action="store_true", help="List all experiments")
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
    t_total = time.perf_counter()

    for name in to_run:
        run_experiment(name)

    logger.info("=" * 70)
    logger.info("ALL DONE: %d experiments in %.1fs",
                len(to_run), time.perf_counter() - t_total)


if __name__ == "__main__":
    main()
