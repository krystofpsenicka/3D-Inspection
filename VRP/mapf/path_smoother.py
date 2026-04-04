"""Path smoothing and resampling utilities.

OMPL ``PathSimplifier`` removes grid-aligned detours (shortcutting)
and rounds corners (B-spline smoothing) while guaranteeing the result
is collision-free with respect to the fine-resolution occupancy grid.

OMPL is a hard requirement — if not installed, an ImportError is raised.

References:
    Pan, J. & Manocha, D. (2012). GPU-Based Parallel Collision Detection
        for Fast Motion Planning. IJRR. — Future direction for GPU-native
        smoothing via batch collision checks and parallel shortcutting.
"""

from __future__ import annotations

import logging

import cupy as cp
import numpy as np

from ..core.constants import OMPL_SIMPLIFY_MAX_TIME

logger = logging.getLogger(__name__)


def simplify_path_ompl(
    path_xyz: np.ndarray | cp.ndarray,
    occupancy_grid,
    robot_radius: float = 0.35,
    max_time: float = OMPL_SIMPLIFY_MAX_TIME,
) -> cp.ndarray:
    """Simplify a 3D path using OMPL PathSimplifier.

    Applies shortcutPath() to remove grid-aligned detours followed by
    smoothBSpline() to round corners. The result is guaranteed
    obstacle-free w.r.t. the fine-resolution occupancy grid.

    Args:
        path_xyz: (M, 3) world-frame path waypoints.
        occupancy_grid: fine-resolution OccupancyGrid for collision checks.
        robot_radius: bounding-sphere radius for validity check resolution.
        max_time: time budget (seconds) for the shortcut simplification step.

    Returns:
        (K, 3) CuPy array of simplified path waypoints (K <= M).

    Raises:
        ImportError: if OMPL is not installed.
    """
    # Transfer to CPU if needed — OMPL requires NumPy
    if isinstance(path_xyz, cp.ndarray):
        path_xyz = cp.asnumpy(path_xyz)

    if len(path_xyz) < 3:
        return cp.asarray(path_xyz.copy())

    try:
        import ompl.base as ob
        import ompl.geometric as og_ompl
    except ImportError:
        raise ImportError(
            "[path_smoother] OMPL is required for path smoothing but is not "
            "installed. Install with: conda install -c conda-forge ompl"
        )

    class _Checker(ob.StateValidityChecker):
        def __init__(self, si):
            super().__init__(si)

        def isValid(self, state):  # noqa: N802
            xyz = cp.array([state[0], state[1], state[2]])
            return bool(occupancy_grid.is_free_world(xyz))

    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    # OMPL requires numpy
    lo = cp.asnumpy(occupancy_grid.origin).astype(np.float64)
    hi = lo + np.array(occupancy_grid.grid.shape, dtype=np.float64) * occupancy_grid.resolution
    for dim in range(3):
        bounds.setLow(dim, float(lo[dim]))
        bounds.setHigh(dim, float(hi[dim]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(_Checker(si))
    si.setStateValidityCheckingResolution(
        float(robot_radius / max(hi - lo))
    )
    si.setup()

    path = og_ompl.PathGeometric(si)
    for pt in path_xyz:
        s = space.allocState()
        s[0], s[1], s[2] = float(pt[0]), float(pt[1]), float(pt[2])
        path.append(s)

    simplifier = og_ompl.PathSimplifier(si)
    simplifier.simplifyMax(path)

    states = path.getStates()
    result = np.array([[s[0], s[1], s[2]] for s in states], dtype=np.float64)
    logger.info("[path_smoother] OMPL: %d -> %d waypoints", len(path_xyz), len(result))
    return cp.asarray(result)


def arc_length_resample(
    path_xyz: np.ndarray | cp.ndarray,
    n_samples: int,
) -> cp.ndarray:
    """Resample a 3D path to n_samples points at uniform arc-length.

    Returns (n_samples, 3) CuPy array.
    """
    path_g = cp.asarray(path_xyz, dtype=cp.float64)
    if len(path_g) < 2 or n_samples < 1:
        return cp.tile(path_g[0], (max(n_samples, 1), 1))

    diffs = cp.diff(path_g, axis=0)
    seg_lens = cp.linalg.norm(diffs, axis=1)
    cum_len = cp.concatenate([cp.array([0.0]), cp.cumsum(seg_lens)])
    total_len = float(cum_len[-1])
    if total_len < 1e-9:
        return cp.tile(path_g[0], (n_samples, 1))

    target_s = cp.linspace(0.0, total_len, n_samples)

    # CuPy interp is 1D — do per-axis
    cum_cpu = cp.asnumpy(cum_len)
    target_cpu = cp.asnumpy(target_s)
    path_cpu = cp.asnumpy(path_g)
    result = np.column_stack([
        np.interp(target_cpu, cum_cpu, path_cpu[:, d]) for d in range(3)
    ])
    return cp.asarray(result)
