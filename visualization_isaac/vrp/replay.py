"""Stage-builder for multi-robot trajectory replay scenes (Isaac Sim).

Extracts reusable scene-building operations from the VRP replay pipeline.
The lifecycle orchestration (SimulationApp, World, replay loop) lives in
``VRP.scripts._isaac_replay``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np

from .._usd_primitives import create_cuboid_prim

logger = logging.getLogger(__name__)


# ─── Waypoint-marker colours ─────────────────────────────────────────────────

ROBOT_COLORS = [
    np.array([1.0, 0.2, 0.2]),   # red
    np.array([0.2, 0.6, 1.0]),   # blue
    np.array([0.2, 1.0, 0.2]),   # green
    np.array([1.0, 0.8, 0.0]),   # yellow
    np.array([1.0, 0.2, 1.0]),   # magenta
    np.array([0.0, 1.0, 1.0]),   # cyan
    np.array([1.0, 0.5, 0.0]),   # orange
    np.array([0.5, 0.0, 1.0]),   # purple
]


# ─── Pose conversion utilities (pure NumPy / SciPy, no USD) ─────────────────

def traj8_to_pose(traj_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one 8-DOF joint position to ``(xyz, quat_wxyz)``.

    The 8 DOFs are ``[x, y, z, yaw, pitch, roll, cam_yaw, cam_pitch]``.
    """
    from scipy.spatial.transform import Rotation as R

    xyz = traj_pos[:3].copy().astype(np.float64)
    yaw, pitch, roll = float(traj_pos[3]), float(traj_pos[4]), float(traj_pos[5])
    rot = R.from_euler("ZYX", [yaw, pitch, roll])
    return xyz, rot.as_quat(scalar_first=True).astype(np.float64)


def convert_trajectories(all_traj_positions: list) -> List[np.ndarray]:
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


# ─── Stage-builder class ────────────────────────────────────────────────────

class ReplayVisualizer:
    """Stage-builder for VRP multi-robot trajectory replay.

    Follows the ``visualization_isaac`` convention: every ``add_*`` method
    takes ``(stage, ...)`` and returns created prim path(s).  No
    SimulationApp management or render loops.

    Parameters
    ----------
    traj_poses : list[np.ndarray]
        Per-robot ``(N, 7)`` arrays of ``[x, y, z, qw, qx, qy, qz]``
        (output of :func:`convert_trajectories`).
    all_waypoints : list[list]
        Per-robot waypoint lists from ``ExecutionResult.all_waypoints``.
    """

    def __init__(self, traj_poses: list[np.ndarray],
                 all_waypoints: list[list]):
        self.traj_poses = traj_poses
        self.all_waypoints = all_waypoints
        self.num_robots = len(traj_poses)
        self.total_steps = (
            max(len(t) for t in traj_poses) if traj_poses else 0
        )

    # ── Waypoint markers ─────────────────────────────────────────────

    def add_waypoint_markers(self, stage, base_path: str,
                             marker_size: float = 0.08) -> list[str]:
        """Add colour-coded waypoint cubes for each robot.

        Returns
        -------
        List of created prim paths.
        """
        paths: list[str] = []
        for i in range(self.num_robots):
            color = tuple(ROBOT_COLORS[i % len(ROBOT_COLORS)])
            for wi, wp in enumerate(self.all_waypoints[i]):
                p = f"{base_path}/wp_r{i}_{wi}"
                paths.append(create_cuboid_prim(
                    stage, p,
                    position=np.array(wp[:3], dtype=np.float64),
                    orientation=np.array(wp[3:7], dtype=np.float64),
                    color=color,
                    size=marker_size,
                ))
        logger.info("Added waypoint markers for %d robots.", self.num_robots)
        return paths

    # ── Environment obstacles ────────────────────────────────────────

    def add_obstacles(
        self, stage, base_path: str,
        mesh_path: str,
        mesh_pose: list,
        mesh_target_length: float,
    ) -> list[str]:
        """Add static obstacles and mesh via cuRobo ``WorldConfig``.

        Returns
        -------
        List of created prim paths (opaque — cuRobo manages internals).
        """
        from curobo.util.usd_helper import UsdHelper
        from curobo.wrap.reacher.motion_gen import WorldConfig

        usd_help = UsdHelper()
        usd_help.load_stage(stage)

        world_dict: dict = {}
        if os.path.isfile(mesh_path):
            import trimesh as _tm

            _raw = _tm.load(mesh_path, force="mesh")
            if isinstance(_raw, _tm.Scene):
                _raw = _tm.util.concatenate(list(_raw.geometry.values()))
            _longest = float(_raw.extents.max())
            _mesh_scale = mesh_target_length / _longest if _longest > 0 else 1.0

            world_dict["mesh"] = {
                "duke_of_lancaster": {
                    "file_path": mesh_path,
                    "pose": mesh_pose,
                    "scale": [_mesh_scale] * 3,
                }
            }
            logger.info("Adding mesh: %s  scale=%.4f", mesh_path, _mesh_scale)
        else:
            logger.warning("Mesh not found at %s — skipping.", mesh_path)

        usd_help.add_world_to_stage(
            WorldConfig.from_dict(world_dict),
            base_frame=base_path,
        )
        return []

    # ── Scene setup helpers ──────────────────────────────────────────

    def add_dome_light(
        self, stage, path: str = "/World/DomeLight",
        intensity: float = 800.0,
        color: tuple = (0.75, 0.85, 1.0),
    ) -> str:
        """Add a dome light prim.

        Returns
        -------
        The prim path string.
        """
        from pxr import UsdLux, Gf

        dome_light = UsdLux.DomeLight.Define(stage, path)
        dome_light.GetIntensityAttr().Set(intensity)
        dome_light.GetColorAttr().Set(Gf.Vec3f(*color))
        logger.info("Dome light added.")
        return path

    def add_zero_gravity(self, stage,
                         scene_path: str = "/physicsScene") -> str:
        """Configure a zero-gravity physics scene (for underwater AUVs).

        Returns
        -------
        The physics scene prim path.
        """
        from pxr import UsdPhysics, Gf

        ps = UsdPhysics.Scene.Get(stage, scene_path)
        if not ps:
            ps = UsdPhysics.Scene.Define(stage, scene_path)
        ps.GetGravityDirectionAttr().Set(Gf.Vec3f(0, 0, 0))
        ps.GetGravityMagnitudeAttr().Set(0.0)
        return scene_path
