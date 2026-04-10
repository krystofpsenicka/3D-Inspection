"""Heading orientation interpolation for dense trajectories.

Sets body yaw and camera pitch on a densely sampled trajectory.
During dwell windows the robot holds the waypoint's viewing direction;
between dwells yaw and camera pitch are cosine-eased.

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
    wp_schedule_s: Optional[cp.ndarray] = None,
    waypoint_rotmats: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """Set yaw and camera pitch on a dense trajectory.

    Args:
        traj: (N, 6) CuPy — dense trajectory samples.
        t_dense: (N,) CuPy — dense time grid.
        dt: replay time step (seconds).
        wp_schedule_s: (n_wp, 3) CuPy — columns [t_dwell_start, t_dwell_end,
            node_idx].  ``None`` disables waypoint-based orientation.
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

    if wp_schedule_s is None or waypoint_rotmats is None or len(wp_schedule_s) == 0:
        yaw[:] = traj[0, 3]
        traj[:, 3] = (yaw + math.pi) % (2 * math.pi) - math.pi
        return traj

    n_wp = len(wp_schedule_s)
    wp_t_ds = wp_schedule_s[:, 0]
    wp_t_de = wp_schedule_s[:, 1]
    node_indices = wp_schedule_s[:, 2].astype(cp.intp)

    # ── Vectorized yaw/pitch extraction from rotation matrices ───────
    forwards = waypoint_rotmats[node_indices, :, 0]  # (n_wp, 3)
    xy_norm = cp.linalg.norm(forwards[:, :2], axis=1)

    wp_yaws = cp.unwrap(cp.arctan2(forwards[:, 1], forwards[:, 0]))
    wp_cpitch = cp.where(
        xy_norm > 1e-9,
        -cp.arctan2(forwards[:, 2], xy_norm),
        cp.zeros(n_wp, dtype=forwards.dtype),
    )

    # ── Classify each dense sample into a segment ────────────────────
    # Build interleaved boundary array:
    #   [t_ds[0], t_de[0], t_ds[1], t_de[1], ...]
    # searchsorted gives bin b for each t:
    #   b=0: before first dwell
    #   b=2i+1: inside dwell i  (odd)
    #   b=2i+2: transition i→i+1  (even, >0), or post-last for i=n_wp-1
    boundaries = cp.empty(2 * n_wp, dtype=cp.float64)
    boundaries[0::2] = wp_t_ds
    boundaries[1::2] = wp_t_de

    bins = cp.searchsorted(boundaries, t_dense, side="left")  # (N,)

    # ── Pre-first segment (b == 0) ───────────────────────────────────
    pre_mask = bins == 0
    yaw[pre_mask] = wp_yaws[0]
    camera_pitch[pre_mask] = 0.0

    # ── Dwell segments (odd bins: b = 2i+1) ──────────────────────────
    for i in range(n_wp):
        dwell_mask = bins == (2 * i + 1)
        yaw[dwell_mask] = wp_yaws[i]
        camera_pitch[dwell_mask] = wp_cpitch[i]

    # ── Transition segments (even bins > 0: b = 2i+2 for i=0..n_wp-2) ──
    for i in range(n_wp - 1):
        trans_mask = bins == (2 * i + 2)
        if not trans_mask.any():
            continue
        t0 = wp_t_de[i]
        t1 = wp_t_ds[i + 1]
        dur = t1 - t0
        if float(dur) > 0:
            blend = (t_dense[trans_mask] - t0) / dur
            ease = 0.5 * (1.0 - cp.cos(math.pi * blend))
            yaw[trans_mask] = wp_yaws[i] + ease * (wp_yaws[i + 1] - wp_yaws[i])
            camera_pitch[trans_mask] = wp_cpitch[i] + ease * (wp_cpitch[i + 1] - wp_cpitch[i])
        else:
            yaw[trans_mask] = wp_yaws[i]
            camera_pitch[trans_mask] = wp_cpitch[i]

    # ── Post-last segment (b >= 2*n_wp) ──────────────────────────────
    post_mask = bins >= (2 * n_wp)
    yaw[post_mask] = wp_yaws[-1]
    camera_pitch[post_mask] = 0.0

    traj[:, 3] = yaw
    if traj.shape[1] > 4:
        traj[:, 4] = camera_pitch
    return traj
