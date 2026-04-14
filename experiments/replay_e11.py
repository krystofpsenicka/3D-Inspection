#!/usr/bin/env python3
"""Replay E11: End-to-End Pipeline in Isaac Sim.

Loads saved viz data for one coverage target and builds an Isaac Sim stage
showing either the set-cover solution (static) or the full MAPF trajectory
replay (animated).

Modes (--stage):
  setcover  — static view: mesh + selected viewpoints + coverage dots
  vrp       — animated replay: mesh + robots moving along MAPF trajectories
  both      — setcover first, then vrp after closing the first window

Usage:
    conda run -n isaaclab python experiments/replay_e11.py --coverage 0.95 --stage setcover
    conda run -n isaaclab python experiments/replay_e11.py --coverage 0.95 --stage vrp
    conda run -n isaaclab python experiments/replay_e11.py --list
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
from visualization_isaac.vrp.replay import ReplayVisualizer, convert_trajectories, ROBOT_COLORS
from visualization_isaac._usd_primitives import set_prim_pose, create_sphere_prim

logger = logging.getLogger(__name__)

VIZ_DIR = os.path.join(RESULTS_DIR, "e11_end_to_end", "viz")


def _list_available():
    if not os.path.isdir(VIZ_DIR):
        print(f"No viz data found at {VIZ_DIR}")
        print("Run e11 first: conda run -n isaaclab python -m experiments.e11_end_to_end"
              " --seeds 42")
        return
    files = sorted(f.replace(".json", "") for f in os.listdir(VIZ_DIR) if f.endswith(".json"))
    print(f"Available viz runs in {VIZ_DIR}:")
    for f in files:
        print(f"  {f}")


def _load_viz(coverage: float, seed: int = 42):
    name = f"target={coverage}_seed={seed}"
    path = os.path.join(VIZ_DIR, name)
    if not os.path.exists(path + ".json"):
        logger.error("Viz file not found: %s", path)
        _list_available()
        sys.exit(1)
    return load_run_result(path)


def _reconstruct_result(data):
    """Rebuild OptimizationResult-like SimpleNamespace from saved numpy arrays."""
    import cupy as cp
    from types import SimpleNamespace
    return SimpleNamespace(
        positions=cp.asarray(data["positions"]),
        rotations=cp.asarray(data["rotations"]),
        visibility_map=cp.asarray(data["visibility_map"]),
        num_viewpoints=int(data["num_viewpoints"]),
        total_coverage=float(data["coverage"]),
    )


def _reconstruct_trajectories(data, n_robots: int):
    """Reconstruct traj_poses and all_waypoints from saved arrays."""
    # Trajectories: list of (T_r, 8) per robot
    all_traj_positions = []
    for r_idx in range(n_robots):
        key = f"traj_r{r_idx}"
        if key in data:
            traj = data[key]  # (T_r, 8)
            # convert_trajectories expects list[list[ndarray]] where each item is one step
            all_traj_positions.append([traj[i] for i in range(len(traj))])
        else:
            all_traj_positions.append([])

    # Waypoints: list of list of waypoint arrays per robot
    all_waypoints = []
    for r_idx in range(n_robots):
        key = f"waypoints_r{r_idx}"
        if key in data:
            wps = data[key]  # (N_wp, 8)
            all_waypoints.append([wps[i].tolist() for i in range(len(wps))])
        else:
            all_waypoints.append([])

    traj_poses = convert_trajectories(all_traj_positions)
    return traj_poses, all_waypoints


def _run_setcover_scene(simulation_app, my_world, stage, mesh,
                        target_points, frustum_params, result, coverage):
    """Build set-cover static scene and run render loop."""
    from VRP.core.constants import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    viz = SetCoverVisualizer(mesh, target_points, frustum_params)
    viz.add_solution(stage, "/World/E11_SetCover", result)

    logger.info("Set-cover scene ready — %d viewpoints, coverage=%.2f%%. "
                "Close window to exit.", result.num_viewpoints, result.total_coverage * 100)

    while simulation_app.is_running():
        my_world.step(render=True)


def _run_vrp_scene(simulation_app, my_world, stage,
                   traj_poses, all_waypoints, n_robots, data):
    """Build VRP replay scene and animate robots."""
    from VRP.core.constants import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH
    from VRP.core.robot_config import load_local_robot_config
    from VRP.core.constants import ASSETS_PATH

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    replay_viz = ReplayVisualizer(traj_poses, all_waypoints)

    try:
        replay_viz.add_zero_gravity(stage)
    except Exception:
        pass

    try:
        replay_viz.add_dome_light(stage)
    except Exception as e:
        logger.warning("Could not add dome light: %s", e)

    replay_viz.add_obstacles(stage, "/World",
                             mesh_path=MESH_PATH,
                             mesh_pose=MESH_POSE,
                             mesh_target_length=MESH_TARGET_LENGTH)

    replay_viz.add_waypoint_markers(stage, "/World")

    # Depot markers
    home_pos = data.get("home_pos")
    if home_pos is not None:
        for k, pos in enumerate(home_pos):
            create_sphere_prim(stage, f"/World/Depots/depot_{k}",
                               position=pos.astype(np.float64),
                               radius=0.20,
                               color=tuple(ROBOT_COLORS[k % len(ROBOT_COLORS)]))

    # Spawn robots
    robot_cfg = load_local_robot_config("brov.yml")
    asset_path = robot_cfg["kinematics"].get("external_asset_path", ASSETS_PATH)
    urdf_path = os.path.join(asset_path, robot_cfg["kinematics"]["urdf_path"])

    robot_pairs = replay_viz.add_robots(stage, "/World", urdf_path, n_robots)
    robots = [r for r, _ in robot_pairs]
    rob_prims = [p for _, p in robot_pairs]

    for robot in robots:
        my_world.scene.add(robot)

    ISAAC_SIM_45 = False
    try:
        from omni.importer.urdf import _urdf  # noqa: F401
    except ImportError:
        ISAAC_SIM_45 = True

    if ISAAC_SIM_45:
        simulation_app.update()
        simulation_app.update()
        my_world.initialize_physics()

    logger.info("VRP replay ready — %d robots, %d total steps. "
                "Press PLAY to start.", n_robots, replay_viz.total_steps)

    replay_idx = 0
    playing = False

    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            playing = False
            continue

        if not playing:
            replay_idx = 0
            playing = True
            logger.info("Replay started.")

        if replay_idx >= replay_viz.total_steps:
            logger.info("Replay finished — looping.")
            replay_idx = 0

        for i, prim in enumerate(rob_prims):
            safe_idx = min(replay_idx, len(traj_poses[i]) - 1)
            set_prim_pose(prim, traj_poses[i][safe_idx, :3], traj_poses[i][safe_idx, 3:])

        replay_idx += 1
        if replay_idx % 500 == 0:
            logger.info("Replay step %d / %d", replay_idx, replay_viz.total_steps)


def main():
    p = argparse.ArgumentParser(description="Replay E11: End-to-End Pipeline in Isaac Sim")
    p.add_argument("--coverage", type=float, default=0.95,
                   help="Coverage target to replay (e.g. 0.85, 0.90, 0.95, 0.97, 0.99)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", choices=["setcover", "vrp", "both"], default="setcover",
                   help="Which stage to visualize")
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
    logger.info("Loading viz data for coverage=%.2f seed=%d", args.coverage, args.seed)
    data = _load_viz(args.coverage, args.seed)

    n_robots = int(data["n_robots"])
    target_points = data["target_points"]
    logger.info("  %d viewpoints, coverage=%.2f%%, %d robots, vrp_status=%s",
                int(data["num_viewpoints"]), float(data["coverage"]) * 100,
                n_robots, data.get("vrp_status", "?"))

    # ── Load mesh ──────────────────────────────────────────────────────────
    from shared.mesh_loader import load_and_transform_mesh
    from VRP.core.constants import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH
    mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)

    # ── Shared objects ─────────────────────────────────────────────────────
    cfg = ModelConfig.duke_of_lancaster()
    frustum_params = cfg.frustum
    result = _reconstruct_result(data)
    traj_poses, all_waypoints = _reconstruct_trajectories(data, n_robots)

    stages_to_run = (["setcover", "vrp"] if args.stage == "both"
                     else [args.stage])

    for stage_name in stages_to_run:
        # ── Isaac Sim bootstrap (one app instance per stage) ───────────────
        logger.info("Starting Isaac Sim for stage=%s ...", stage_name)
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

        if stage_name == "setcover":
            _run_setcover_scene(simulation_app, my_world, stage,
                                mesh, target_points, frustum_params,
                                result, args.coverage)
        else:  # vrp
            _run_vrp_scene(simulation_app, my_world, stage,
                           traj_poses, all_waypoints, n_robots, data)

        simulation_app.close()
        logger.info("Stage=%s done.", stage_name)


if __name__ == "__main__":
    main()
