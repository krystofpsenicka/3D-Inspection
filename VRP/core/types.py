"""VRP core data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class VRPBackend(Enum):
    """MIP solver backend for VRP."""

    HIGHS = "highs"
    CUOPT = "cuopt"


@dataclass
class VRPResult:
    """Solution returned by any VRP solver backend.

    Routes exclude depot indices.

    ``best_bound`` is the solver's dual bound on the alpha-blended
    objective, in meters. For MIP solvers running branch-and-bound it is
    the best LP relaxation across open nodes at termination and is a
    provably-valid lower bound on the integer optimum. ``0.0`` indicates
    the backend did not expose a bound (CPU/HiGHS without PuLP support,
    solver failure, or older cuOpt builds). Readers should treat
    ``best_bound == 0.0`` as "unavailable".
    """

    routes: list[list[int]]
    total_cost: float
    makespan: float = 0.0
    per_vehicle_costs: list[float] = field(default_factory=list)
    alpha: float = 1.0
    objective_value: float = 0.0
    best_bound: float = 0.0
    solver: str = "unknown"
    status: str = "success"


@dataclass
class PlanningStats:
    """Per-robot metrics from Space-Time A* planning.

    Used by the priority ordering heuristic to identify robots that
    have the most conflicts and should be planned earlier.
    """

    wait_steps: int = 0
    detour_ratio: float = 0.0
    astar_failures: int = 0
    astar_retries: int = 0


@dataclass
class ExecutionResult:
    """Full trajectory output for every robot after route execution."""

    all_traj_positions: list[np.ndarray]
    all_traj_velocities: list[np.ndarray]
    all_waypoints: list[list[list[float]]]
    initial_positions: list[np.ndarray]
    fail_counts: list[int]
    actual_makespan: float = 0.0
    actual_per_vehicle_times: list[float] = None
    actual_per_leg_times: list[list[float]] = None

    def __post_init__(self):
        if self.actual_per_vehicle_times is None:
            self.actual_per_vehicle_times = []
        if self.actual_per_leg_times is None:
            self.actual_per_leg_times = []
