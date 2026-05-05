"""Stage-builder for multi-robot trajectory replay scenes (Isaac Sim).

Lifecycle orchestration (SimulationApp, World, render loop) lives in
``visualization_isaac.app`` / ``visualization_isaac.phases``.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .._usd_primitives import create_cuboid_prim, create_mesh_prim, set_prim_pose

logger = logging.getLogger(__name__)


ROBOT_COLORS = [
    np.array([1.0, 0.2, 0.2]),
    np.array([0.2, 0.6, 1.0]),
    np.array([0.2, 1.0, 0.2]),
    np.array([1.0, 0.8, 0.0]),
    np.array([1.0, 0.2, 1.0]),
    np.array([0.0, 1.0, 1.0]),
    np.array([1.0, 0.5, 0.0]),
    np.array([0.5, 0.0, 1.0]),
]


def traj8_to_pose(traj_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """8 DOFs are ``[x, y, z, yaw, pitch, roll, cam_yaw, cam_pitch]``."""
    from scipy.spatial.transform import Rotation as R

    xyz = traj_pos[:3].copy().astype(np.float64)
    yaw, pitch, roll = float(traj_pos[3]), float(traj_pos[4]), float(traj_pos[5])
    rot = R.from_euler("ZYX", [yaw, pitch, roll])
    return xyz, rot.as_quat(scalar_first=True).astype(np.float64)


def convert_trajectories(all_traj_positions: list) -> list[np.ndarray]:
    """Pre-convert all robot trajectories to ``(N, 7)`` arrays of ``[x,y,z,qw,qx,qy,qz]``."""
    result = []
    for robot_traj in all_traj_positions:
        converted = np.zeros((len(robot_traj), 7), dtype=np.float64)
        for step, tp in enumerate(robot_traj):
            xyz, qwxyz = traj8_to_pose(tp)
            converted[step, :3] = xyz
            converted[step, 3:] = qwxyz
        result.append(converted)
    return result


class ReplayVisualizer:
    """``traj_poses``: per-robot ``(N, 7)`` arrays from :func:`convert_trajectories`.
    ``all_waypoints``: per-robot waypoint lists from ``ExecutionResult.all_waypoints``."""

    def __init__(self, traj_poses: list[np.ndarray], all_waypoints: list[list]):
        self.traj_poses = traj_poses
        self.all_waypoints = all_waypoints
        self.num_robots = len(traj_poses)
        self.total_steps = max(len(t) for t in traj_poses) if traj_poses else 0

    def add_waypoint_markers(self, stage, base_path: str, marker_size: float = 0.08) -> list[str]:
        paths: list[str] = []
        for i in range(self.num_robots):
            color = tuple(ROBOT_COLORS[i % len(ROBOT_COLORS)])
            for wi, wp in enumerate(self.all_waypoints[i]):
                p = f"{base_path}/wp_r{i}_{wi}"
                paths.append(
                    create_cuboid_prim(
                        stage,
                        p,
                        position=np.array(wp[:3], dtype=np.float64),
                        orientation=np.array(wp[3:7], dtype=np.float64),
                        color=color,
                        size=marker_size,
                    )
                )
        logger.info("Added waypoint markers for %d robots.", self.num_robots)
        return paths

    def add_obstacles(
        self,
        stage,
        base_path: str,
        mesh_path: str,
        mesh_pose: list,
        mesh_target_length: float,
    ) -> list[str]:
        from shared.mesh_loader import load_and_transform_mesh

        paths: list[str] = []
        if not os.path.isfile(mesh_path):
            logger.warning("Mesh not found at %s — skipping.", mesh_path)
            return paths

        mesh = load_and_transform_mesh(mesh_path, mesh_target_length, mesh_pose)
        prim_path = f"{base_path}/inspection_mesh"
        create_mesh_prim(stage, prim_path, mesh, color=(0.55, 0.55, 0.55), opacity=1.0)
        paths.append(prim_path)
        logger.info("Added mesh obstacle: %s", prim_path)
        return paths

    def add_robots(
        self,
        stage,
        base_path: str,
        urdf_path: str,
        num_robots: int,
    ) -> list:
        """Import a URDF robot and spawn ``num_robots`` instances. Returns (Robot, prim) tuples."""
        ISAAC_SIM_45 = False
        try:
            from omni.importer.urdf import _urdf
        except ImportError:
            from isaacsim.asset.importer.urdf import _urdf  # type: ignore

            ISAAC_SIM_45 = True

        import omni.usd  # type: ignore
        from omni.isaac.core.robots import Robot

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

        robots = []
        rob_prims = []
        for i in range(num_robots):
            dp = str(stage.GetDefaultPrim().GetPath())
            pp = omni.usd.get_stage_next_free_path(
                stage,
                dp + inner_prim_path,
                False,
            )
            stage.OverridePrim(pp).GetReferences().AddReference(dest_usd)
            robot = Robot(prim_path=pp, name=f"brov_{i}")
            set_prim_pose(
                stage.GetPrimAtPath(pp),
                np.zeros(3, dtype=np.float64),
                np.array([1, 0, 0, 0], dtype=np.float64),
            )
            robots.append(robot)
            rob_prims.append(stage.GetPrimAtPath(pp))
            logger.info("Spawned robot %d  prim=%s", i, pp)

        return list(zip(robots, rob_prims, strict=False))

    def add_dome_light(
        self,
        stage,
        path: str = "/World/DomeLight",
        intensity: float = 800.0,
        color: tuple = (0.75, 0.85, 1.0),
    ) -> str:
        from pxr import Gf, UsdLux

        dome_light = UsdLux.DomeLight.Define(stage, path)
        dome_light.GetIntensityAttr().Set(intensity)
        dome_light.GetColorAttr().Set(Gf.Vec3f(*color))
        logger.info("Dome light added.")
        return path

    def add_zero_gravity(self, stage, scene_path: str = "/physicsScene") -> str:
        from pxr import Gf, UsdPhysics

        ps = UsdPhysics.Scene.Get(stage, scene_path)
        if not ps:
            ps = UsdPhysics.Scene.Define(stage, scene_path)
        ps.GetGravityDirectionAttr().Set(Gf.Vec3f(0, 0, 0))
        ps.GetGravityMagnitudeAttr().Set(0.0)
        return scene_path

    def add_brov_robots(
        self,
        stage,
        base_path: str,
        num_robots: int,
        brov_usd_path: str,
    ) -> list:
        """One ``Xform`` prim per robot referencing ``brov_usd_path``. Pose is seeded so subsequent
        ``set_prim_pose`` calls reuse the existing ``xformOp:translate`` / ``xformOp:orient`` attributes."""
        from pxr import UsdGeom

        if not os.path.isfile(brov_usd_path):
            logger.warning("BROV USD not found at %s — skipping robot spawn.", brov_usd_path)
            return []

        prims = []
        for i in range(num_robots):
            prim_path = f"{base_path}/robot_{i}"
            UsdGeom.Xform.Define(stage, prim_path)
            prim = stage.GetPrimAtPath(prim_path)
            prim.GetReferences().AddReference(brov_usd_path)
            set_prim_pose(
                prim,
                np.zeros(3, dtype=np.float64),
                np.array([1, 0, 0, 0], dtype=np.float64),
            )
            prims.append(prim)
            logger.info("Spawned BROV robot %d at %s (Xform reference, no joints)", i, prim_path)
        return prims

    def step_replay(self, robot_prims: list, frame_idx: int) -> None:
        """``frame_idx`` is clamped per-robot to the trajectory length."""
        for i, prim in enumerate(robot_prims):
            if i >= len(self.traj_poses):
                continue
            traj = self.traj_poses[i]
            if len(traj) == 0:
                continue
            k = int(min(frame_idx, len(traj) - 1))
            xyz = traj[k, :3]
            qwxyz = traj[k, 3:]
            set_prim_pose(prim, xyz, qwxyz)
