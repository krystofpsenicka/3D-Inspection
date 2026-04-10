"""Isaac Sim lifecycle orchestrator for VRP trajectory replay.

This module owns the SimulationApp bootstrap, World creation, and
frame-by-frame replay loop.  All scene-building (robot spawning, mesh
obstacles, waypoint markers) is delegated to :class:`ReplayVisualizer`.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from VRP.core.constants import (
    ASSETS_PATH,
    CONFIGS_PATH,
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
)
from VRP.core.robot_config import load_local_robot_config
from VRP.core.types import ExecutionResult

from visualization_isaac.vrp.replay import (
    ReplayVisualizer,
    convert_trajectories,
)
from visualization_isaac._usd_primitives import set_prim_pose

logger = logging.getLogger(__name__)


def replay_in_isaac_sim(
    exec_result: ExecutionResult,
    headless: bool = False,
) -> None:
    """Launch Isaac Sim and replay multi-robot trajectories.

    Parameters
    ----------
    exec_result:
        Result from :meth:`~MultiAgentPathPlanner.execute`.
    headless:
        If ``True``, run without a display (useful for CI / server use).
    """
    all_traj_positions = exec_result.all_traj_positions
    all_waypoints = exec_result.all_waypoints
    num_robots = len(all_traj_positions)
    traj_poses = convert_trajectories(all_traj_positions)

    viz = ReplayVisualizer(traj_poses, all_waypoints)

    # ── Bootstrap Isaac Sim ───────────────────────────────────────────
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError:
            logger.error(
                "[viz] Isaac Sim not found.  "
                "Skipping visual replay.  "
                "Trajectories saved to exec_result."
            )
            return

    simulation_app = SimulationApp(
        {"headless": headless, "width": "1920", "height": "1080"}
    )

    # ── Late imports (require Isaac Sim running) ──────────────────────
    from omni.isaac.core import World

    # ── World ─────────────────────────────────────────────────────────
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    try:
        viz.add_zero_gravity(stage)
    except Exception:
        pass

    try:
        viz.add_dome_light(stage)
    except Exception as _e:
        logger.warning("[viz] Could not add dome light: %s", _e)

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    # ── Spawn robots via ReplayVisualizer ─────────────────────────────
    robot_cfg = load_local_robot_config("brov.yml")
    asset_path = robot_cfg["kinematics"].get("external_asset_path", ASSETS_PATH)
    _kin = robot_cfg["kinematics"]
    urdf_path = os.path.join(asset_path, _kin["urdf_path"])

    robot_pairs = viz.add_robots(stage, "/World", urdf_path, num_robots)
    robots = [r for r, _ in robot_pairs]
    rob_prims = [p for _, p in robot_pairs]

    xform = stage.GetPrimAtPath("/World") or stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    for robot in robots:
        my_world.scene.add(robot)

    # ── Static obstacles (mesh) ───────────────────────────────────────
    viz.add_obstacles(
        stage, "/World",
        mesh_path=MESH_PATH,
        mesh_pose=MESH_POSE,
        mesh_target_length=MESH_TARGET_LENGTH,
    )
    my_world.scene.add_default_ground_plane()

    # ── Waypoint marker cubes ─────────────────────────────────────────
    viz.add_waypoint_markers(stage, "/World")

    # ── Initialise physics ────────────────────────────────────────────
    ISAAC_SIM_45 = False
    try:
        from omni.importer.urdf import _urdf  # noqa: F401
    except ImportError:
        ISAAC_SIM_45 = True

    if ISAAC_SIM_45:
        simulation_app.update()
        simulation_app.update()
        my_world.initialize_physics()

    logger.info(
        "[viz] Ready to replay %d steps for %d robots. Click PLAY.",
        viz.total_steps, num_robots,
    )

    # ── Replay loop ───────────────────────────────────────────────────
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
            logger.info("[viz] Replay started.")

        if replay_idx >= viz.total_steps:
            logger.info("[viz] Replay finished – looping.")
            replay_idx = 0

        for i, prim in enumerate(rob_prims):
            safe_idx = min(replay_idx, len(traj_poses[i]) - 1)
            set_prim_pose(
                prim,
                traj_poses[i][safe_idx, :3],
                traj_poses[i][safe_idx, 3:],
            )

        replay_idx += 1
        if replay_idx % 500 == 0:
            logger.info("[viz] Replay step %d / %d", replay_idx, viz.total_steps)

    simulation_app.close()
