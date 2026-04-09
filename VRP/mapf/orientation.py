"""Heading orientation interpolation for dense trajectories.

Sets body yaw and camera pitch on a densely sampled trajectory.
During dwell windows the robot holds the waypoint's viewing direction;
between dwells yaw and camera pitch are cosine-eased.

All computation runs on GPU (CuPy). Callers are responsible for
converting inputs to CuPy and outputs back to NumPy if needed.

Trajectory layout: [x, y, z, yaw, cam_pitch, cam_roll]
"""

from __future__ import annotations

import math
from typing import Optional

import cupy as cp


def apply_heading_orientation(
    traj: cp.ndarray,
    t_dense: cp.ndarray,
    dt: float,
    wp_schedule_s: Optional[list] = None,
    waypoint_rotmats: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """Set yaw and camera pitch on a dense trajectory.

    Trajectory layout: [x, y, z, yaw, cam_pitch, cam_roll]
        yaw       = column 3
        cam_pitch = column 4

    Args:
        traj: (N, 6) CuPy — dense trajectory samples.
        t_dense: (N,) CuPy — dense time grid.
        dt: replay time step (seconds).
        wp_schedule_s: list of (t_dwell_start, t_dwell_end, node_idx)
            tuples, in seconds.
        waypoint_rotmats: (M, 3, 3) CuPy — rotation matrices for all
            VRP nodes (column 0 = forward direction).

    Returns:
        Modified *traj* with columns 3 (yaw) and 4 (cam_pitch) filled.
    """
    N = len(traj)
    if N < 2:
        return traj

    yaw = cp.empty(N)
    camera_pitch = cp.zeros(N)

    if not wp_schedule_s or waypoint_rotmats is None:
        yaw[:] = traj[0, 3]
        traj[:, 3] = (yaw + math.pi) % (2 * math.pi) - math.pi
        return traj

    wp_yaws: list = []
    wp_cpitch: list = []
    wp_t_ds: list = []
    wp_t_de: list = []

    for (t_ds, t_de, node_idx) in wp_schedule_s:
        forward = waypoint_rotmats[node_idx, :, 0]
        fx, fy, fz = float(forward[0]), float(forward[1]), float(forward[2])
        xy_norm = math.sqrt(fx * fx + fy * fy)
        wp_yaws.append(math.atan2(fy, fx))
        wp_cpitch.append(-math.atan2(fz, xy_norm) if xy_norm > 1e-9 else 0.0)
        wp_t_ds.append(t_ds)
        wp_t_de.append(t_de)

    wp_yaws = list(cp.asnumpy(cp.unwrap(cp.asarray(wp_yaws))))
    n_wp = len(wp_schedule_s)

    def _dense_idx(t: float, side: str = "left") -> int:
        return max(0, min(int(cp.searchsorted(t_dense, t, side=side)), N))

    d_start_0 = _dense_idx(wp_t_ds[0])
    yaw[:d_start_0] = wp_yaws[0]
    camera_pitch[:d_start_0] = 0.0

    for i in range(n_wp):
        d_start = _dense_idx(wp_t_ds[i])
        d_end = _dense_idx(wp_t_de[i], side="right")

        yaw[d_start:d_end] = wp_yaws[i]
        camera_pitch[d_start:d_end] = wp_cpitch[i]

        if i < n_wp - 1:
            next_start = _dense_idx(wp_t_ds[i + 1])
            if next_start > d_end:
                ts = t_dense[d_end:next_start]
                t0_t = t_dense[d_end]
                t1_t = t_dense[min(next_start, N - 1)]
                dur = t1_t - t0_t
                if float(dur) > 0:
                    blend = (ts - t0_t) / dur
                    ease = 0.5 * (1.0 - cp.cos(math.pi * blend))
                    yaw[d_end:next_start] = (
                        wp_yaws[i] + ease * (wp_yaws[i + 1] - wp_yaws[i])
                    )
                    camera_pitch[d_end:next_start] = (
                        wp_cpitch[i] + ease * (wp_cpitch[i + 1] - wp_cpitch[i])
                    )
                else:
                    yaw[d_end:next_start] = wp_yaws[i]
                    camera_pitch[d_end:next_start] = wp_cpitch[i]
        else:
            yaw[d_end:] = wp_yaws[i]
            camera_pitch[d_end:] = 0.0

    traj[:, 3] = yaw
    if traj.shape[1] > 4:
        traj[:, 4] = camera_pitch
    return traj
