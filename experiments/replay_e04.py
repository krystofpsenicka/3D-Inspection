#!/usr/bin/env python3
"""Replay E04: Set Cover Optimizer Comparison in Isaac Sim.

Loads saved viz data for one optimizer run and builds an interactive Isaac Sim
stage showing the mesh, selected viewpoints (frustums + coverage dots), and
uncovered surface points.

Usage:
    conda run -n isaaclab python experiments/replay_e04.py --optimizer GreedySetCover
    conda run -n isaaclab python experiments/replay_e04.py --list
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import ModelConfig, RESULTS_DIR
from experiments.common.persistence import load_run_result

# visualization_isaac uses lazy pxr/omni imports — safe to import before SimulationApp
from visualization_isaac.visibility.set_cover import SetCoverVisualizer
from visualization_isaac.visibility.model import ModelVisualizer

logger = logging.getLogger(__name__)

VIZ_DIR = os.path.join(RESULTS_DIR, "e04_set_cover_optimizers", "viz")


def _list_available():
    if not os.path.isdir(VIZ_DIR):
        print(f"No viz data found at {VIZ_DIR}")
        print("Run e04 first: conda run -n isaaclab python -m experiments.e04_set_cover_optimizers"
              " --section A --seeds 42 --targets 0.95")
        return
    files = sorted(f.replace(".json", "") for f in os.listdir(VIZ_DIR) if f.endswith(".json"))
    print(f"Available viz runs in {VIZ_DIR}:")
    for f in files:
        print(f"  {f}")


def _load_viz(optimizer: str, seed: int = 42, target: float = 0.95):
    name = f"opt={optimizer}_target={target}_seed={seed}"
    path = os.path.join(VIZ_DIR, name)
    if not os.path.exists(path + ".json"):
        logger.error("Viz file not found: %s", path)
        logger.error("Available files:")
        _list_available()
        sys.exit(1)
    return load_run_result(path)


def main():
    p = argparse.ArgumentParser(description="Replay E04: Set Cover in Isaac Sim")
    p.add_argument("--optimizer", default="GreedySetCover",
                   help="Optimizer name to replay (e.g. GreedySetCover, LazyGreedySetCoverCuda)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--list", action="store_true", help="List available viz runs and exit")
    p.add_argument("--headless", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    if args.list:
        _list_available()
        return

    # ── Load viz data ──────────────────────────────────────────────────────
    logger.info("Loading viz data for optimizer=%s seed=%d target=%.2f",
                args.optimizer, args.seed, args.target)
    data = _load_viz(args.optimizer, args.seed, args.target)

    target_points = data["target_points"]
    logger.info("  %d selected viewpoints, coverage=%.2f%%",
                int(data["num_viewpoints"]), float(data["coverage"]) * 100)

    # ── Load mesh (trimesh, not Open3D) ───────────────────────────────────
    from shared.mesh_loader import load_and_transform_mesh
    from VRP.core.constants import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH
    mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)

    # ── Frustum params from ModelConfig ───────────────────────────────────
    cfg = ModelConfig.duke_of_lancaster()
    frustum_params = cfg.frustum

    # ── Reconstruct OptimizationResult-like object from saved numpy arrays ─
    import cupy as cp
    from types import SimpleNamespace
    result = SimpleNamespace(
        positions=cp.asarray(data["positions"]),
        rotations=cp.asarray(data["rotations"]),
        visibility_map=cp.asarray(data["visibility_map"]),
        num_viewpoints=int(data["num_viewpoints"]),
        total_coverage=float(data["coverage"]),
    )

    # ── Isaac Sim bootstrap ────────────────────────────────────────────────
    logger.info("Starting Isaac Sim ...")
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError:
        from isaacsim import SimulationApp  # type: ignore

    simulation_app = SimulationApp(
        {"headless": args.headless, "width": "1920", "height": "1080"}
    )

    from omni.isaac.core import World  # type: ignore

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    # ── Build scene ────────────────────────────────────────────────────────
    logger.info("Building scene ...")
    viz = SetCoverVisualizer(mesh, target_points, frustum_params)
    viz.add_solution(stage, "/World/E04", result)

    logger.info("Scene ready — %d viewpoints for optimizer=%s (coverage=%.2f%%)",
                result.num_viewpoints, args.optimizer, result.total_coverage * 100)
    logger.info("Close the Isaac Sim window to exit.")

    while simulation_app.is_running():
        my_world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
