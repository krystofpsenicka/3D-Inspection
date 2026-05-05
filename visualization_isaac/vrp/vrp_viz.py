"""Stage-builder for the VRP visualization phases (depots, assignment, trajectories, replay).

Composes :class:`ReplayVisualizer` for replay; camera-to-body conversion delegates to
``VRP.core.geometry.viewpoints_to_robot_waypoints`` so the offset stays in one place.
"""

from __future__ import annotations

import logging

import numpy as np

from .._helpers import rotmat_to_quat_wxyz
from .._usd_primitives import (
    create_cuboid_prim,
    create_lineset_prim,
    create_sphere_prim,
    set_prim_pose,
)
from .replay import ROBOT_COLORS, ReplayVisualizer

logger = logging.getLogger(__name__)


def _camera_to_body_positions(
    positions: np.ndarray,
    rotmats: np.ndarray,
    home_indices: set | list,
) -> np.ndarray:
    """CPU/numpy port of ``viewpoints_to_robot_waypoints``."""
    home_set = set(home_indices)
    out = positions.astype(np.float64).copy()

    forwards = rotmats[:, :, 0]
    xy_norm = np.linalg.norm(forwards[:, :2], axis=1)
    xy_norm = np.maximum(xy_norm, 1e-9)

    from VRP.core.constants import CAMERA_OFFSET_FORWARD, CAMERA_OFFSET_UP

    out[:, 0] -= CAMERA_OFFSET_FORWARD * (forwards[:, 0] / xy_norm)
    out[:, 1] -= CAMERA_OFFSET_FORWARD * (forwards[:, 1] / xy_norm)
    out[:, 2] -= CAMERA_OFFSET_UP

    for idx in home_set:
        if 0 <= idx < len(out):
            out[idx] = positions[idx]
    return out


class VRPVisualizer:
    """``traj_poses``: per-robot ``(N_i, 7)`` arrays of ``[x,y,z,qw,qx,qy,qz]`` (from
    :func:`convert_trajectories`); may be ``None`` for non-replay phases."""

    def __init__(
        self,
        num_robots: int,
        traj_poses: list[np.ndarray] | None = None,
        robot_colors: list | None = None,
    ):
        self.num_robots = num_robots
        self.traj_poses = traj_poses or []
        self.colors = [
            tuple(np.asarray(c, dtype=np.float64))
            for c in (
                robot_colors
                if robot_colors is not None
                else [ROBOT_COLORS[i % len(ROBOT_COLORS)] for i in range(num_robots)]
            )
        ]
        self._replay = ReplayVisualizer(self.traj_poses, [[] for _ in range(num_robots)])

    def add_inspection_mesh(
        self,
        stage,
        base_path: str,
        mesh_path: str,
        mesh_pose: list,
        mesh_target_length: float,
    ) -> list[str]:
        return self._replay.add_obstacles(
            stage, base_path, mesh_path, mesh_pose, mesh_target_length
        )

    def add_depots(
        self,
        stage,
        base_path: str,
        depot_xyzs: np.ndarray | list,
        radius: float = 0.25,
    ) -> list[str]:
        paths: list[str] = []
        for i, xyz in enumerate(depot_xyzs):
            color = self.colors[i % len(self.colors)]
            paths.append(
                create_sphere_prim(
                    stage,
                    f"{base_path}/depot_{i}",
                    position=np.asarray(xyz, dtype=np.float64),
                    radius=radius,
                    color=color,
                )
            )
        return paths

    def add_waypoints(
        self,
        stage,
        base_path: str,
        vp_positions: np.ndarray,
        vp_rotmats: np.ndarray,
        color: tuple = (0.95, 0.95, 0.95),
        size: float = 0.18,
    ) -> list[str]:
        """``vp_positions`` are camera positions; offset back to body frame before placing markers."""
        if len(vp_positions) == 0:
            return []
        body_xyzs = _camera_to_body_positions(
            np.asarray(vp_positions, dtype=np.float64),
            np.asarray(vp_rotmats, dtype=np.float64),
            home_indices=set(),
        )
        paths: list[str] = []
        for i, (pos, R) in enumerate(zip(body_xyzs, vp_rotmats, strict=False)):
            quat = rotmat_to_quat_wxyz(R)
            paths.append(
                create_cuboid_prim(
                    stage,
                    f"{base_path}/wp_{i}",
                    position=pos,
                    orientation=quat,
                    color=color,
                    size=size,
                )
            )
        return paths

    def add_assignment(
        self,
        stage,
        base_path: str,
        vp_positions: np.ndarray,
        vp_rotmats: np.ndarray,
        routes: list[list[int]],
        home_indices: list[int] | None = None,
        size: float = 0.18,
    ) -> list[str]:
        """``routes[i]`` is the inspection-only sequence for robot *i* (no homes), as returned by
        ``solve_vrp``. ``home_indices`` are global node indices reserved for depots."""
        K = self.num_robots
        if home_indices is None:
            home_indices = list(range(K))
        homes = set(home_indices)

        # Routes are inspection-only; node index >= K is an inspection node with insp_idx = node - K.
        wp_robot: dict[int, int] = {}
        for r, route in enumerate(routes):
            for node in route:
                if node in homes:
                    continue
                insp_idx = int(node) - K
                if 0 <= insp_idx < len(vp_positions):
                    wp_robot[insp_idx] = r

        if len(vp_positions) == 0:
            return []

        body_xyzs = _camera_to_body_positions(
            np.asarray(vp_positions, dtype=np.float64),
            np.asarray(vp_rotmats, dtype=np.float64),
            home_indices=set(),
        )

        paths: list[str] = []
        for i, (pos, R) in enumerate(zip(body_xyzs, vp_rotmats, strict=False)):
            color = self.colors[wp_robot.get(i, 0) % len(self.colors)] if i in wp_robot else (
                0.5,
                0.5,
                0.5,
            )
            quat = rotmat_to_quat_wxyz(R)
            paths.append(
                create_cuboid_prim(
                    stage,
                    f"{base_path}/wp_{i}",
                    position=pos,
                    orientation=quat,
                    color=color,
                    size=size,
                )
            )
        return paths

    def add_trajectories(
        self,
        stage,
        base_path: str,
        line_width: float = 0.05,
    ) -> list[str]:
        paths: list[str] = []
        for i, traj in enumerate(self.traj_poses):
            if len(traj) < 2:
                continue
            pts = traj[:, :3].astype(np.float64)
            seg_idx = np.column_stack(
                (np.arange(len(pts) - 1), np.arange(1, len(pts)))
            )
            color = self.colors[i % len(self.colors)]
            paths.append(
                create_lineset_prim(
                    stage,
                    f"{base_path}/traj_{i}",
                    points=pts,
                    lines=seg_idx,
                    color=color,
                    width=line_width,
                )
            )
        return paths

    def add_brov_robots(
        self,
        stage,
        base_path: str,
        brov_usd_path: str,
    ) -> list:
        return self._replay.add_brov_robots(stage, base_path, self.num_robots, brov_usd_path)

    def step(self, robot_prims: list, frame_idx: int) -> None:
        self._replay.step_replay(robot_prims, frame_idx)
