"""Isaac Sim lifecycle orchestrator for VRP trajectory replay.

This module owns the SimulationApp bootstrap, World creation, URDF import,
robot spawning, and frame-by-frame replay loop.  Scene-building is delegated
to :class:`~visualization_isaac.vrp.replay.ReplayVisualizer`.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from VRP.core.constants import (
    ASSETS_PATH,
    CONFIGS_PATH,
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    STATIC_OBSTACLES,
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
        Result from :meth:`~route_executor.RouteExecutor.execute`.
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
    from omni.isaac.core.robots import Robot
    from curobo.util.usd_helper import set_prim_transform  # type: ignore

    ISAAC_SIM_45 = False
    try:
        from omni.importer.urdf import _urdf
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf  # type: ignore
        ISAAC_SIM_45 = True

    # ── Add cuRobo Isaac Sim extensions helper if present ─────────────
    try:
        import curobo as _cb  # type: ignore
        _cp = os.path.dirname(os.path.dirname(_cb.__file__))
        _ex = os.path.join(_cp, "examples", "isaac_sim")
        if os.path.exists(_ex):
            sys.path.insert(0, _ex)
            from helper import add_extensions  # type: ignore
            add_extensions(simulation_app, headless)
    except Exception:
        pass

    # ── World ─────────────────────────────────────────────────────────
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    # Scene setup via ReplayVisualizer
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
    stage.DefinePrim("/curobo", "Xform")

    # ── URDF import → temp USD ────────────────────────────────────────
    robot_cfg = load_local_robot_config("brov.yml")
    urdf_iface = _urdf.acquire_urdf_interface()

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 100_000.0
    import_config.default_position_drive_damping = 10_000.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = robot_cfg["kinematics"].get("external_asset_path", ASSETS_PATH)
    _kin = robot_cfg["kinematics"]
    urdf_path = os.path.join(asset_path, _kin["urdf_path"])
    robot_dir = os.path.dirname(urdf_path)
    urdf_file = os.path.basename(urdf_path)

    if ISAAC_SIM_45:
        import omni.kit.commands  # type: ignore
        dest_usd = os.path.join(
            robot_dir,
            os.path.splitext(urdf_file)[0] + "_vrp_temp.usd",
        )
        _, inner_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=os.path.join(robot_dir, urdf_file),
            import_config=import_config,
            dest_path=dest_usd,
        )
    else:
        imported = urdf_iface.parse_urdf(robot_dir, urdf_file, import_config)
        inner_prim_path = urdf_iface.import_robot(
            robot_dir, urdf_file, imported, import_config, ""
        )
        dest_usd = None

    xform = stage.GetPrimAtPath("/World") or stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    # ── Spawn robots ──────────────────────────────────────────────────
    import omni.usd  # type: ignore

    robots = []
    rob_prims = []
    for i in range(num_robots):
        dp = str(stage.GetDefaultPrim().GetPath())
        pp = omni.usd.get_stage_next_free_path(stage, dp + inner_prim_path, False)
        stage.OverridePrim(pp).GetReferences().AddReference(dest_usd)
        robot = Robot(prim_path=pp, name=f"brov_{i}")
        set_prim_transform(stage.GetPrimAtPath(pp), [0, 0, 0, 1, 0, 0, 0])
        robots.append(my_world.scene.add(robot))
        rob_prims.append(stage.GetPrimAtPath(pp))
        logger.info("[viz] Spawned robot %d  prim=%s", i, pp)

    # ── Static obstacles (mesh + cuboids) ─────────────────────────────
    viz.add_obstacles(
        stage, "/World",
        mesh_path=MESH_PATH,
        mesh_pose=MESH_POSE,
        mesh_target_length=MESH_TARGET_LENGTH,
        static_obstacles=STATIC_OBSTACLES,
    )
    my_world.scene.add_default_ground_plane()

    # ── Waypoint marker cubes ─────────────────────────────────────────
    viz.add_waypoint_markers(stage, "/World")

    # ── Initialise physics ────────────────────────────────────────────
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
            set_prim_pose(prim, traj_poses[i][safe_idx, :3], traj_poses[i][safe_idx, 3:])

        replay_idx += 1
        if replay_idx % 500 == 0:
            logger.info("[viz] Replay step %d / %d", replay_idx, viz.total_steps)

    simulation_app.close()
