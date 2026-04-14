#!/usr/bin/env python3
"""Replay E08: VRP Solver Comparison in Isaac Sim.

Loads saved viz data for one VRP solver run and builds an interactive Isaac Sim
stage showing the mesh, per-robot waypoint markers (color-coded), and route
paths as line sets.

Usage:
    conda run -n isaaclab python experiments/replay_e08.py --solver highs
    conda run -n isaaclab python experiments/replay_e08.py --list
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

from experiments.common.config import RESULTS_DIR
from experiments.common.persistence import load_run_result

# visualization_isaac uses lazy pxr/omni imports — safe to import before SimulationApp
from visualization_isaac.vrp.replay import ReplayVisualizer, ROBOT_COLORS
from visualization_isaac._usd_primitives import create_sphere_prim, create_lineset_prim

logger = logging.getLogger(__name__)

VIZ_DIR = os.path.join(RESULTS_DIR, "e08_vrp_solver_comparison", "viz")
_N_WAYPOINTS_DEFAULT = 50


def _list_available():
    if not os.path.isdir(VIZ_DIR):
        print(f"No viz data found at {VIZ_DIR}")
        print("Run e08 first: conda run -n isaaclab python -m experiments.e08_vrp_solver_comparison"
              " --seeds 42")
        return
    files = sorted(f.replace(".json", "") for f in os.listdir(VIZ_DIR) if f.endswith(".json"))
    print(f"Available viz runs in {VIZ_DIR}:")
    for f in files:
        print(f"  {f}")


def _load_viz(solver: str, n_waypoints: int = _N_WAYPOINTS_DEFAULT, seed: int = 42):
    # n_robots is saved in the file; infer from the problem size used (n_wp=50 → 3 robots)
    _n_robots_map = {20: 2, 50: 3, 75: 5}
    n_robots = _n_robots_map.get(n_waypoints, 3)
    name = f"wp={n_waypoints}_robots={n_robots}_solver={solver}_seed={seed}"
    path = os.path.join(VIZ_DIR, name)
    if not os.path.exists(path + ".json"):
        logger.error("Viz file not found: %s", path)
        _list_available()
        sys.exit(1)
    return load_run_result(path)


def _add_route_geometry(stage, base_path: str, all_pos: np.ndarray,
                        routes: list[np.ndarray]) -> list[str]:
    """Add waypoint spheres and route lineset prims for each robot."""
    paths: list[str] = []
    for r_idx, route_indices in enumerate(routes):
        color = tuple(ROBOT_COLORS[r_idx % len(ROBOT_COLORS)])
        # Waypoint spheres along the route (skip first/last which are the depot)
        waypoint_indices = route_indices[1:-1]  # exclude depot start/end
        for wi, wp_idx in enumerate(waypoint_indices):
            pos = all_pos[wp_idx].astype(np.float64)
            p = f"{base_path}/r{r_idx}_wp{wi}"
            paths.append(create_sphere_prim(stage, p, position=pos,
                                            radius=0.12, color=color))
        # Route line through all stops (depot → waypoints → depot)
        route_points = all_pos[route_indices].astype(np.float64)
        line_indices = np.column_stack(
            [np.arange(len(route_points) - 1),
             np.arange(1, len(route_points))]
        )
        p = f"{base_path}/r{r_idx}_route"
        if len(route_points) >= 2:
            paths.append(create_lineset_prim(stage, p,
                                             points=route_points,
                                             lines=line_indices,
                                             color=color, width=2.0))
    return paths


def main():
    p = argparse.ArgumentParser(description="Replay E08: VRP Solver in Isaac Sim")
    p.add_argument("--solver", default="highs",
                   help="Solver to replay: highs or cuopt")
    p.add_argument("--n_waypoints", type=int, default=_N_WAYPOINTS_DEFAULT,
                   help="Problem size (20, 50, or 75 waypoints)")
    p.add_argument("--seed", type=int, default=42)
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
    logger.info("Loading viz data for solver=%s n_wp=%d seed=%d",
                args.solver, args.n_waypoints, args.seed)
    data = _load_viz(args.solver, args.n_waypoints, args.seed)

    all_pos = data["all_pos"]          # (K_depots + N_wp, 3)
    home_pos = data["home_pos"]        # (K, 3)
    n_robots = int(data["n_robots"])
    routes = []
    for r_idx in range(n_robots):
        key = f"routes_r{r_idx}"
        if key in data:
            routes.append(data[key])

    logger.info("  %d robots, %d waypoints, solver=%s, makespan=%.1f, status=%s",
                n_robots, int(data["n_waypoints"]), args.solver,
                float(data["makespan"]), data["status"])

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
    from VRP.core.constants import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    # ── Build scene ────────────────────────────────────────────────────────
    logger.info("Building scene ...")

    # Mesh via ReplayVisualizer (reuses the same obstacle loader as the full pipeline)
    replay_viz = ReplayVisualizer([], [])  # no trajectories for static VRP view
    replay_viz.add_obstacles(stage, "/World",
                             mesh_path=MESH_PATH,
                             mesh_pose=MESH_POSE,
                             mesh_target_length=MESH_TARGET_LENGTH)
    replay_viz.add_dome_light(stage)

    # Depot markers (larger, grey)
    for k, pos in enumerate(home_pos):
        create_sphere_prim(stage, f"/World/Depots/depot_{k}",
                           position=pos.astype(np.float64),
                           radius=0.20, color=(0.7, 0.7, 0.7))

    # Per-robot route geometry
    _add_route_geometry(stage, "/World/Routes", all_pos, routes)

    logger.info("Scene ready — %d robot routes for solver=%s. Close window to exit.",
                n_robots, args.solver)

    while simulation_app.is_running():
        my_world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
