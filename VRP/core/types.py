"""VRP core data types.

Centralizes all dataclasses used across the VRP module, following the
pattern established in ``visibility/core/types.py``.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

import cupy as cp


@dataclass
class VRPResult:
    """Solution returned by any VRP solver backend.

    Routes exclude depot indices. All non-depot nodes appear exactly once
    across all routes.
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
    suffer most from conflicts and should be planned earlier.
    """
    wait_steps: int = 0
    detour_ratio: float = 0.0
    astar_failures: int = 0
    astar_retries: int = 0


@dataclass
class ExecutionResult:
    """Full trajectory output for every robot after route execution.

    Array fields may be CuPy (GPU-resident) or NumPy. Call ``.get()``
    on CuPy arrays when CPU data is needed (e.g. for visualization or
    serialization).
    """
    all_traj_positions: list[list[np.ndarray | cp.ndarray]]
    all_traj_velocities: list[list[np.ndarray | cp.ndarray]]
    all_waypoints: list[list[list[float]]]
    initial_positions: list[np.ndarray]
    joint_names: list[str]
    fail_counts: list[int]
    actual_makespan: float = 0.0
    actual_per_vehicle_times: list[float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.actual_per_vehicle_times is None:
            self.actual_per_vehicle_times = []


@dataclass
class PipelineConfig:
    """All user-facing settings for one VRP planning run."""
    num_robots: int = 2
    solver_backend: str = "ortools"
    alpha: float = 1.0
    rapids_python: str = ""
    gpu_timeout: int = 300
    mip_time_limit: int = 120
    mip_gap: float = 0.05
    feedback_iterations: int = 3
    feedback_threshold: float = 0.20
    waypoint_source: str = "random"
    n_random_waypoints: int = 5
    random_seed: int = 42
    headless: bool = True
    replay_in_isaac: bool = False
    save_solution_path: str | None = None
