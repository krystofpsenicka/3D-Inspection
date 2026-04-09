"""Path smoothing and resampling utilities.

OMPL ``PathSimplifier`` removes grid-aligned detours (shortcutting)
and rounds corners (B-spline smoothing) while guaranteeing the result
is collision-free with respect to the fine-resolution occupancy grid.
When a reservation table is provided, the validity checker also
rejects states that fall inside reserved space-time cells (time is
approximated via nearest-neighbour lookup on the original path).

Future direction
----------------
Pan, J. & Manocha, D. (2012). GPU-Based Parallel Collision Detection
    for Fast Motion Planning. IJRR.

A GPU-native shortcutter could replace the OMPL loop entirely: batch
collision-check many candidate line segments in one kernel, accept
free shortcuts, and repeat.  This would eliminate the expensive
per-validity-check kernel launch that the current OMPL approach
incurs (each ``isValid`` call round-trips through CuPy / the OG).
The same batch kernel could incorporate reservation-table lookups
with interpolated time, giving exact space-time awareness instead
of the nearest-neighbour approximation used here.
"""

from __future__ import annotations

import logging
from typing import Optional

import cupy as cp
import numpy as np

from ..core.constants import OMPL_SIMPLIFY_MAX_TIME
from shared.occupancy_grid import OccupancyGrid

logger = logging.getLogger(__name__)


def simplify_path_ompl(
    path_xyz: cp.ndarray,
    occupancy_grid: OccupancyGrid,
    robot_radius: float = 0.35,
    max_time: float = OMPL_SIMPLIFY_MAX_TIME,
    reservation=None,
    time_steps: Optional[cp.ndarray] = None,
    coarse_og: Optional[OccupancyGrid] = None,
) -> cp.ndarray:
    """Simplify a 3D path using OMPL PathSimplifier.

    Applies shortcutPath() to remove grid-aligned detours followed by
    smoothBSpline() to round corners. The result is guaranteed
    obstacle-free w.r.t. the fine-resolution occupancy grid.

    When *reservation*, *time_steps* and *coarse_og* are all provided,
    the validity checker additionally rejects states that overlap with
    previously committed robot trajectories.  Time is estimated by
    nearest-neighbour lookup on the original path — this is approximate
    but steers OMPL away from reserved corridors during shortcutting.

    Args:
        path_xyz: (M, 3) world-frame path waypoints.
        occupancy_grid: fine-resolution OccupancyGrid for collision checks.
        robot_radius: bounding-sphere radius for validity check resolution.
        max_time: time budget (seconds) for the simplification step.
        reservation: optional ReservationTable for space-time checks.
        time_steps: (M,) CuPy coarse time indices matching *path_xyz*.
        coarse_og: coarse OccupancyGrid for world-to-voxel conversion
            when checking the reservation table.

    Returns:
        (K, 3) CuPy array of simplified path waypoints (K <= M).

    Raises:
        ImportError: if OMPL is not installed.
    """

    path_np = cp.asnumpy(path_xyz).astype(np.float64)

    if len(path_np) < 3:
        return cp.asarray(path_np.copy())

    import ompl.base as ob
    import ompl.geometric as og_ompl

    # Precompute nearest-neighbour data for reservation time estimation.
    check_reservation = (
        reservation is not None
        and time_steps is not None
        and coarse_og is not None
    )
    if check_reservation:
        orig_path_gpu = path_xyz.copy()
        orig_time_gpu = time_steps.copy()

    class _Checker(ob.StateValidityChecker):
        def __init__(self, si):
            super().__init__(si)

        def isValid(self, state):  # noqa: N802
            xyz = cp.array([state[0], state[1], state[2]])
            if not bool(occupancy_grid.is_free_world(xyz)):
                return False
            if check_reservation:
                dists = cp.linalg.norm(orig_path_gpu - xyz, axis=1)
                nearest_idx = int(cp.argmin(dists))
                t = int(orig_time_gpu[nearest_idx])
                ijk = coarse_og.world_to_voxel(xyz.reshape(1, 3))[0]
                if reservation.is_reserved(
                    int(ijk[0]), int(ijk[1]), int(ijk[2]), t,
                ):
                    return False
            return True

    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
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
    for pt in path_np:
        s = space.allocState()
        s[0], s[1], s[2] = float(pt[0]), float(pt[1]), float(pt[2])
        path.append(s)

    simplifier = og_ompl.PathSimplifier(si)
    simplifier.simplify(path, max_time)

    states = path.getStates()
    result = np.array([[s[0], s[1], s[2]] for s in states], dtype=np.float64)
    logger.info("[path_smoother] OMPL: %d -> %d waypoints", len(path_np), len(result))
    return cp.asarray(result)


def arc_length_resample(
    path_xyz: cp.ndarray,
    n_samples: int,
) -> cp.ndarray:
    """Resample a 3D path to n_samples points at uniform arc-length.

    Returns (n_samples, 3) CuPy array.
    """
    if len(path_xyz) < 2 or n_samples < 1:
        return cp.tile(path_xyz[0], (max(n_samples, 1), 1))

    diffs = cp.diff(path_xyz, axis=0)
    seg_lens = cp.linalg.norm(diffs, axis=1)
    cum_len = cp.concatenate([cp.array([0.0]), cp.cumsum(seg_lens)])
    total_len = float(cum_len[-1])
    if total_len < 1e-9:
        return cp.tile(path_xyz[0], (n_samples, 1))

    target_s = cp.linspace(0.0, total_len, n_samples)

    result = cp.column_stack([
        cp.interp(target_s, cum_len, path_xyz[:, d]) for d in range(3)
    ])
    return result
