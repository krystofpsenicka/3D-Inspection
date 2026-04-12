#!/usr/bin/env python3
"""
VRP Planner – CLI Entry Point

Usage examples
--------------

Random 5 waypoints, 2 AUVs, save solution:
    python run_vrp.py --num_robots 2 --random_waypoints 5 --save_solution solution.pkl

Load waypoints from JSON, 3 AUVs, force HiGHS:
    python run_vrp.py --num_robots 3 --waypoints_file waypoints.json --solver highs --save_solution sol.pkl

Import waypoints from 3D-Inspection output:
    python run_vrp.py --num_robots 2 --waypoints_from_inspection methods_analysis/models --save_solution sol.pkl

Inside inspection (routes traverse mesh interior):
    python run_vrp.py --num_robots 2 --random_waypoints 5 --side inside

Visualize a previously planned solution in Isaac Sim (activate Isaac Sim env first):
    python VRP/visualize_solution.py --solution_file solution.pkl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure the repository root is on PYTHONPATH so `VRP` package resolves
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cupy as cp
import numpy as np

from VRP.core.types import PipelineConfig, VRPBackend
from shared.types import Side
from VRP.core.vrp_orchestrator import VRPFeedbackOrchestrator
from VRP.core.waypoint_loader import load_waypoints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collision-free multi-AUV VRP planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Robot / simulation ─────────────────────────────────────────────
    p.add_argument("--num_robots", "-n", type=int, default=2,
                   help="Number of AUV robots.")
    p.add_argument("--headless", action="store_true",
                   help="Isaac Sim headless flag.")
    p.add_argument("--side", choices=["outside", "inside"],
                   default="outside",
                   help="Inspection side: 'outside' (routes around mesh) "
                        "or 'inside' (routes inside mesh).")

    # ── Waypoint source ────────────────────────────────────────────────
    wp_group = p.add_mutually_exclusive_group()
    wp_group.add_argument("--random_waypoints", type=int, default=5,
                          metavar="N",
                          help="Sample N random collision-free waypoints.")
    wp_group.add_argument("--waypoints_file", type=str,
                          help="Path to waypoints .json / .npz / .csv file.")
    wp_group.add_argument("--waypoints_from_inspection", type=str,
                          metavar="MODELS_DIR",
                          help="Load waypoints from 3D-Inspection result directory.")

    p.add_argument("--random_seed", type=int, default=42,
                   help="Random seed for waypoint sampling.")

    # ── VRP solver ────────────────────────────────────────────────────
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Objective blending: 1.0=pure makespan, 0.0=pure "
                        "total distance, 0.5=balanced trade-off.")
    p.add_argument("--solver", choices=["cuopt", "highs"],
                   default="highs",
                   help="MIP backend: 'cuopt' (GPU) or 'highs' (CPU).")
    p.add_argument("--mip_time_limit", type=int, default=120,
                   help="MIP solver time budget (seconds).")
    p.add_argument("--mip_gap", type=float, default=0.05,
                   help="MIP solver relative optimality gap (0.05 = 5%%).")
    p.add_argument("--feedback_iterations", type=int, default=3,
                   help="Max VRP ↔ path-planning feedback loop iterations.")
    # ── Solution persistence ──────────────────────────────────
    p.add_argument("--save_solution", type=str, default="vrp_solution.pkl",
                   metavar="PATH",
                   help="Save the planned ExecutionResult to PATH (.pkl). "
                        "Load later with VRP/visualize_solution.py.")
    # ── Logging ───────────────────────────────────────────────────────
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging.")

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    # ── Load waypoints (caller responsibility) ───────────────────────
    if args.waypoints_file:
        waypoint_source = args.waypoints_file
        n_random = 0
    elif args.waypoints_from_inspection:
        waypoint_source = args.waypoints_from_inspection
        n_random = 0
    else:
        waypoint_source = "random"
        n_random = args.random_waypoints

    positions_np, rotmats_np = load_waypoints(
        source=waypoint_source,
        n_random=n_random,
        random_seed=args.random_seed,
    )
    insp_positions = cp.asarray(positions_np, dtype=cp.float64)
    insp_rotmats = cp.asarray(rotmats_np, dtype=cp.float64)

    # ── Build config ──────────────────────────────────────────────────
    cfg = PipelineConfig(
        num_robots=args.num_robots,
        side=Side(args.side),
        solver_backend=VRPBackend(args.solver),
        alpha=args.alpha,
        mip_time_limit=args.mip_time_limit,
        mip_gap=args.mip_gap,
        feedback_iterations=args.feedback_iterations,
        save_solution_path=args.save_solution,
    )

    # ── Run ───────────────────────────────────────────────────────────
    orchestrator = VRPFeedbackOrchestrator(cfg)
    result = orchestrator.run(insp_positions, insp_rotmats)

    # ── Summary ───────────────────────────────────────────────────────
    total_wps = sum(len(r) for r in result.all_waypoints)
    total_fail = sum(result.fail_counts)
    print("\n" + "=" * 60)
    print(f"VRP planning complete")
    print(f"  Robots        : {args.num_robots}")
    print(f"  Waypoints     : {total_wps} total  ({total_fail} failed)")
    print(f"  Replay steps  : {len(result.all_traj_positions[0]) if result.all_traj_positions else 0}")
    if args.save_solution:
        print(f"  Solution saved: {args.save_solution}")
        print(f"  Visualize with: python VRP/visualize_solution.py --solution_file {args.save_solution}")
    print("=" * 60)


if __name__ == "__main__":
    main()
