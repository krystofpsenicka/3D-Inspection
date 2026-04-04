"""Viewpoint/waypoint loading for the VRP pipeline.

In the integrated pipeline, viewpoints come directly from the visibility
module's ``OptimizationResult`` as GPU-resident CuPy arrays (positions
+ rotation matrices). This module provides the GPU-native entry point
plus a backward-compatible dispatcher for standalone testing via
file-based loaders in ``scripts/waypoint_utils.py``.

All loaders return ``(positions, rotmats)`` — no quaternions.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


def load_viewpoints_gpu(
    positions: cp.ndarray,
    orientations: cp.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Pass through GPU-resident viewpoint data.

    Accepts CuPy arrays directly from the visibility module's
    ``OptimizationResult.positions`` (K, 3) and
    ``OptimizationResult.orientations`` (K, 3, 3).

    Returns:
        (positions, orientations) as CuPy arrays.
    """
    return positions, orientations


def load_waypoints(
    source: str,
    n_random: int | None = None,
    og=None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Universal dispatcher for file-based waypoint loading.

    For standalone testing/benchmarking only. The integrated pipeline
    should use ``load_viewpoints_gpu()`` instead.

    Returns:
        (positions (N, 3), rotmats (N, 3, 3)) as numpy arrays.
    """
    from VRP.scripts.waypoint_utils import load_waypoints as _load_file_waypoints
    return _load_file_waypoints(source, n_random=n_random, og=og, random_seed=random_seed)
