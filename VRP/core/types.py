"""VRP core data types."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from shared.types import Side


class VRPBackend(Enum):
    """MIP solver backend for VRP."""
    HIGHS = "highs"
    CUOPT = "cuopt"


@dataclass
class VRPResult:
    """Solution returned by any VRP solver backend.

    Routes exclude depot indices.
    """
    routes: list[list[int]]
    total_cost: float
    makespan: float = 0.0
    per_vehicle_costs: list[float] = field(default_factory=list)
    alpha: float = 1.0
    objective_value: float = 0.0
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
    all_traj_positions: list[list[np.ndarray]]
    all_traj_velocities: list[list[np.ndarray]]
    all_waypoints: list[list[list[float]]]
    initial_positions: list[np.ndarray]
    fail_counts: list[int]
    actual_makespan: float = 0.0
    actual_per_vehicle_times: list[float] = None

    def __post_init__(self):
        if self.actual_per_vehicle_times is None:
            self.actual_per_vehicle_times = []


@dataclass
class PipelineConfig:
    """All user-facing settings for one VRP planning run."""
    num_robots: int = 2
    side: Side = Side.OUTSIDE
    solver_backend: VRPBackend = VRPBackend.HIGHS
    alpha: float = 1.0
    rapids_python: str = ""
    gpu_timeout: int = 300
    mip_time_limit: int = 120
    mip_gap: float = 0.05
    feedback_iterations: int = 3
    feedback_threshold: float = 0.20
    headless: bool = True
    replay_in_isaac: bool = False
    save_solution_path: str | None = None
