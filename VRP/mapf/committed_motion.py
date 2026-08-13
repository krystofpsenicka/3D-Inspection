"""Continuous-time inter-robot conflict model for space-time A* (CCBS-style).

The replayed trajectory of each robot is the *linear-in-time interpolation* of
its committed coarse world polyline (vertices at the shared coarse timesteps).
Two robots therefore collide iff, on some shared coarse interval [t, t+1], the
moving-point distance between their two segments drops below ``2*robot_radius``.
That distance has a closed form (a quadratic in the interpolation parameter), so
we can test a candidate A* move against every already-committed robot *exactly* -
no cell inflation, no margin tuning, and it captures swaps and diagonal
crossings that a discrete vertex/edge reservation misses.

This replaces the discrete cell :class:`ReservationTable` for *inter-robot*
avoidance; static-obstacle avoidance stays on the coarse occupancy grid.
"""

from __future__ import annotations

import cupy as cp

_EPS = 1e-9


def segment_min_gap_sq(A0, A1, B0, B1):
    """Squared minimum distance over t in [0,1] between two points moving
    linearly A0->A1 and B0->B1. Broadcasts over leading dims. Any NaN/inf
    endpoint (absent robot) yields +inf (no conflict)."""
    r0 = A0 - B0                       # (..., 3)  relative position at t=0
    dr = (A1 - A0) - (B1 - B0)         # (..., 3)  relative velocity * dt
    dr2 = (dr * dr).sum(-1)            # (...)
    r0dr = (r0 * dr).sum(-1)           # (...)
    tau = cp.where(dr2 > _EPS, cp.clip(-r0dr / cp.maximum(dr2, _EPS), 0.0, 1.0), 0.0)
    rel = r0 + tau[..., None] * dr
    return (rel * rel).sum(-1)         # (...)


def continuous_collision_report(robot_world_paths, robot_coarse_times, T, robot_radius):
    """Exact continuous-time inter-robot collision check on the committed coarse
    polylines (the replay is their linear-in-time interpolation, so this is the
    true collision status of the output). Returns (num_colliding_pairs,
    min_separation_m). Parked robots hold their final position.
    """
    two_r_sq = float(2.0 * robot_radius) ** 2
    rows = []
    for wp, ct in zip(robot_world_paths, robot_coarse_times):
        row = cp.full((T, 3), cp.inf, dtype=cp.float32)
        if wp is not None and len(wp) > 0:
            ct = cp.asarray(ct).astype(cp.intp)
            wp = cp.asarray(wp).astype(cp.float32)
            m = (ct >= 0) & (ct < T)
            if bool(m.any()):
                row[ct[m]] = wp[m]
                lt = int(ct[m].max())
                if 0 <= lt < T - 1:
                    row[lt + 1:] = row[lt]
                ft = int(ct[m].min())
                if ft > 0:
                    row[:ft] = row[ft]
        rows.append(row)
    if len(rows) < 2:
        return 0, float("inf")
    W = cp.stack(rows, axis=0)                 # (R, T, 3)
    R = W.shape[0]
    A0 = W[:, :-1, :]                           # (R, T-1, 3)  segment starts
    A1 = W[:, 1:, :]                            # (R, T-1, 3)  segment ends
    pairs = 0
    min_sep = float("inf")
    for a in range(R):
        for b in range(a + 1, R):
            gap2 = segment_min_gap_sq(A0[a], A1[a], A0[b], A1[b])   # (T-1,)
            gap2f = cp.where(cp.isfinite(gap2), gap2, cp.inf)
            mg = float(cp.sqrt(gap2f.min()))
            if mg < min_sep:
                min_sep = mg
            if bool((gap2 < two_r_sq).any()):
                pairs += 1
    return pairs, min_sep


class CommittedMotion:
    """Holds already-committed robots' coarse world polylines and answers, for a
    batch of candidate A* moves, whether each stays >= 2r from all of them."""

    def __init__(self, max_time_steps: int, robot_radius: float):
        self.T = int(max_time_steps)
        self.two_r_sq = float(2.0 * robot_radius) ** 2
        self._rows: list[cp.ndarray] = []
        self.W: cp.ndarray | None = None   # (C, T, 3) world positions per coarse step

    @property
    def num_committed(self) -> int:
        return 0 if self.W is None else int(self.W.shape[0])

    def commit(self, world_positions: cp.ndarray, coarse_times: cp.ndarray) -> None:
        """Add one robot's committed motion: its world position at each coarse
        timestep, holding the final position for all later steps (parked)."""
        row = cp.full((self.T, 3), cp.inf, dtype=cp.float32)
        ct = cp.asarray(coarse_times).astype(cp.intp)
        wp = cp.asarray(world_positions).astype(cp.float32)
        m = (ct >= 0) & (ct < self.T)
        if bool(m.any()):
            row[ct[m]] = wp[m]
            last_t = int(ct[m].max())
            if 0 <= last_t < self.T - 1:
                row[last_t + 1:] = row[last_t]          # park at final position
            first_t = int(ct[m].min())
            if first_t > 0:
                row[:first_t] = row[first_t]            # hold start before first step
        self._rows.append(row)
        self.W = cp.stack(self._rows, axis=0)

    def moves_valid(
        self,
        from_world: cp.ndarray,
        to_world: cp.ndarray,
        parent_t: cp.ndarray,
        pad: float = 0.0,
    ) -> cp.ndarray:
        """(K,) bool: True where the move from_world[k]->to_world[k] over
        [parent_t[k], parent_t[k]+1] keeps >= 2r + pad from every committed robot.

        ``pad`` is a positive clearance added to 2r. Used for holdability checks,
        where the check point is a cell centre but the robot's true dwell point
        can be up to a cell half-diagonal off-centre: padding by that half-diagonal
        guarantees the actual dwell/interp positions (anywhere in the cell) stay
        >= 2r from committed robots."""
        K = len(from_world)
        if self.W is None or self.W.shape[0] == 0:
            return cp.ones(K, dtype=cp.bool_)
        thr = (self.two_r_sq ** 0.5 + pad) ** 2 if pad else self.two_r_sq  # (2r + pad)^2
        pt = cp.clip(cp.asarray(parent_t).astype(cp.intp), 0, self.T - 1)
        nt = cp.clip(pt + 1, 0, self.T - 1)
        # committed positions at the move's endpoints -> (K, C, 3)
        B0 = self.W[:, pt, :].transpose(1, 0, 2)
        B1 = self.W[:, nt, :].transpose(1, 0, 2)
        A0 = from_world[:, None, :].astype(cp.float32)
        A1 = to_world[:, None, :].astype(cp.float32)
        gap2 = segment_min_gap_sq(A0, A1, B0, B1)         # (K, C)
        conflict = (gap2 < thr).any(axis=1)               # (K,)
        return ~conflict
