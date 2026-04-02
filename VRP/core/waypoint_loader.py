"""Viewpoint/waypoint loading for the VRP pipeline.

In the integrated pipeline, viewpoints come directly from the visibility
module's ``OptimizationResult`` as GPU-resident CuPy arrays. This module
provides the GPU-native entry point plus a backward-compatible dispatcher
for standalone testing via file-based loaders in ``scripts/waypoint_utils.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


def load_viewpoints_gpu(
    positions: cp.ndarray,
    orientations: cp.ndarray,
) -> cp.ndarray:
    """Build 7-DOF waypoint array from GPU-resident viewpoint data.

    Accepts CuPy arrays directly from the visibility module's
    ``OptimizationResult.positions`` (K, 3) and
    ``OptimizationResult.orientations`` (K, 3, 3), keeping data on GPU.

    Orientations (rotation matrices) are converted to quaternions
    [qw, qx, qy, qz] for compatibility with the VRP pipeline.

    Returns:
        (K, 7) CuPy array of [x, y, z, qw, qx, qy, qz].
    """
    K = len(positions)
    if K == 0:
        return cp.empty((0, 7), dtype=cp.float32)

    positions = cp.asarray(positions, dtype=cp.float32)
    orientations = cp.asarray(orientations, dtype=cp.float32)

    # Convert rotation matrices to quaternions on GPU
    quats = _rotmat_to_quat_gpu(orientations)  # (K, 4) [qw, qx, qy, qz]
    return cp.concatenate([positions, quats], axis=1)


def _rotmat_to_quat_gpu(R: cp.ndarray) -> cp.ndarray:
    """Convert (K, 3, 3) rotation matrices to (K, 4) quaternions [qw, qx, qy, qz].

    Uses Shepperd's method for numerical stability.
    """
    K = len(R)
    quats = cp.empty((K, 4), dtype=cp.float32)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    m1 = trace > 0
    if m1.any():
        s = cp.sqrt(trace[m1] + 1.0) * 2.0
        quats[m1, 0] = 0.25 * s
        quats[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        quats[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
        quats[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    m2 = ~m1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m2.any():
        s = cp.sqrt(1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]) * 2.0
        quats[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        quats[m2, 1] = 0.25 * s
        quats[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        quats[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    m3 = ~m1 & ~m2 & (R[:, 1, 1] > R[:, 2, 2])
    if m3.any():
        s = cp.sqrt(1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]) * 2.0
        quats[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        quats[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        quats[m3, 2] = 0.25 * s
        quats[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    m4 = ~m1 & ~m2 & ~m3
    if m4.any():
        s = cp.sqrt(1.0 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]) * 2.0
        quats[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
        quats[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
        quats[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
        quats[m4, 3] = 0.25 * s

    # Normalize
    norms = cp.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / cp.maximum(norms, 1e-12)
    return quats


def load_waypoints(
    source: str,
    n_random: int | None = None,
    og=None,
    random_seed: int = 42,
):
    """Backward-compatible dispatcher for file-based waypoint loading.

    For standalone testing/benchmarking only. The integrated pipeline
    should use ``load_viewpoints_gpu()`` instead.
    """
    from VRP.scripts.waypoint_utils import load_waypoints as _load_file_waypoints
    return _load_file_waypoints(source, n_random=n_random, og=og, random_seed=random_seed)
