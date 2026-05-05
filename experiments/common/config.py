"""Model definitions, seeds, parameter grids."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TOSCA_DIR = os.path.join(MODELS_DIR, "TOSCA-dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

# Duke pose: trimesh Y-up GLB + Isaac Y->Z = R_x(-90 deg) composed with 180 deg X.
_SQRT2_2 = math.sqrt(2.0) / 2.0
_DUKE_MESH_POSE = [0, 0, 1.5, _SQRT2_2, -_SQRT2_2, 0.0, 0.0]
_IDENTITY_POSE = [0, 0, 0, 1.0, 0.0, 0.0, 0.0]


# ── Model configuration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class FrustumConfig:
    fov_deg: float = 40.0
    aspect: float = 1.0
    near: float = 0.1
    far: float = 6.0

    @property
    def fov_y_rad(self) -> float:
        return np.deg2rad(self.fov_deg)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    mesh_path: str
    target_length: float
    mesh_pose: list
    num_surface_points: int
    num_candidates: int
    frustum: FrustumConfig
    collision_radius: float = 0.35
    voxel_resolution: float = 0.20

    @staticmethod
    def duke_of_lancaster() -> ModelConfig:
        return ModelConfig(
            name="duke_of_lancaster",
            mesh_path=os.path.join(MODELS_DIR, "duke_of_lancaster_uk_clipped.glb"),
            target_length=50.0,
            mesh_pose=_DUKE_MESH_POSE,
            num_surface_points=200_000,
            num_candidates=1500,
            frustum=FrustumConfig(fov_deg=40.0, near=0.1, far=6.0),
        )

    @staticmethod
    def tosca(name: str) -> ModelConfig:
        """TOSCA dataset model, e.g. 'wolf0', 'cat0'."""
        mesh_path = os.path.join(TOSCA_DIR, f"{name}.off")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"TOSCA model not found: {mesh_path}")
        return ModelConfig(
            name=name,
            mesh_path=mesh_path,
            target_length=10.0,
            mesh_pose=_IDENTITY_POSE,
            num_surface_points=50_000,
            num_candidates=500,
            frustum=FrustumConfig(fov_deg=60.0, near=0.05, far=3.0),
        )


@dataclass
class ExperimentConfig:
    model: ModelConfig
    target_coverage: float = 0.95
    max_viewpoints: int = 1000


# ── Seeds ───────────────────────────────────────────────────────────────────

SEEDS_3 = [42, 123, 7]
SEEDS_5 = [42, 123, 7, 2024, 314]
SEEDS_10 = SEEDS_5 + [999, 55, 8888, 1337, 2025]

# ── TOSCA model lists ───────────────────────────────────────────────────────

TOSCA_REPRESENTATIVE = ["wolf0", "cat0", "david0"]  # small/medium/large

TOSCA_ALL = [
    "wolf0", "cat0", "centaur0", "david0", "dog0",
    "gorilla1", "horse0", "michael0", "victoria0",
]

TOSCA_VALID: list | None = None  # set by validate_models.py


# ── Parameter grids ─────────────────────────────────────────────────────────

# E01: Section A (k=1, Duke+TOSCA), Section B (k>1, TOSCA only)
E01_STRATEGIES_A = ["weighted", "weighted_curvature", "targeted_25", "cmaes_100"]
E01_STRATEGIES_B = ["weighted", "weighted_curvature", "targeted_25", "cmaes_100"]
E01_K_VALUES = [1, 2, 3, 4]
E01_STRATEGIES = E01_STRATEGIES_A

E02_CANDIDATE_COUNTS = [250, 500, 750, 1000, 1500, 2000, 3000, 5000]
E02_STRATEGIES = ["weighted", "weighted_curvature", "targeted_25", "cmaes_100"]

E03_COVERAGE_TARGETS = [0.85, 0.90, 0.925, 0.95, 0.97]

E04_COVERAGE_TARGETS = [0.85, 0.90, 0.925, 0.95, 0.97]
# Section A: all optimizers including expansion variants
E04_OPTIMIZERS_A = [
    "GreedySetCover", "GreedySetCoverCuda", "LazyGreedySetCover",
    "ExpansionIterative_weighted", "ExpansionIterative_weighted_curvature",
    "ExpansionIterative_cmaes",
]
# Section B: input-strategy robustness sweep
E04_OPTIMIZERS_B = ["GreedySetCover", "GreedySetCoverCuda", "LazyGreedySetCover"]
E04_INPUT_STRATEGIES = ["weighted", "weighted_curvature", "targeted_25", "cmaes_100"]
E04_OPTIMIZERS = E04_OPTIMIZERS_A  # legacy alias

E05_CANDIDATE_COUNTS = [500, 1000, 2000, 5000, 10000]
E05_POINT_COUNTS = [50_000, 100_000, 200_000]

E06_FLEET_SIZES = [1, 2, 3, 4, 5, 6, 8, 10]
E06_WAYPOINT_COUNTS = [10, 25, 50, 75, 100]

E07_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

E10_RESOLUTIONS = [0.25, 0.50, 0.75, 1.0, 1.5]

E13_K_VALUES = [1, 2, 3, 4]
E13_COVERAGE_TARGETS = [0.90, 0.95]
E13_STRATEGIES = ["targeted_25", "cmaes_100"]

# E16: Frustum sensitivity
E16_FOV_VALUES = [30.0, 45.0, 60.0, 90.0]
E16_NEAR_FAR_PAIRS = [(0.1, 5.0), (0.2, 10.0), (0.5, 15.0)]

# E03: Iterative sampler sweeps
_E03_BASE_N = 500  # TOSCA budget; scales as _E03_BASE_N * k_coverage

# Targeted
E03_T_K_VALUES = [1, 2, 3, 4, 6, 8]
E03_T_FRACTIONS = [12, 25, 50, 75, 100]  # % from targeted phase
E03_T_SPI_VALUES = [1, 5, 25, None]  # samples_per_iteration; None = all-at-once

# CMA-ES
E03_C_K_VALUES = [1, 2, 3, 4, 6, 8]
E03_C_FRACTIONS = [25, 50, 75, 100]
E03_C_TRAVEL_WEIGHTS_TOSCA = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
E03_C_TRAVEL_WEIGHTS_DUKE = [0.01, 0.02, 0.03, 0.06, 0.1]
E03_C_POPSIZE_VALUES = [5, 10, 15, 25, 40]  # default 15
E03_C_MAXITER_VALUES = [5, 10, 20, 40]  # default 20

# E17: Sampler routing impact
E17_STRATEGIES = ["weighted", "weighted_curvature", "targeted_25", "targeted_50", "cmaes_100"]
E17_CMAES_TRAVEL_WEIGHTS = [0.0, 0.1, 0.3]
E17_N_ROBOTS = 5
E17_N_CANDIDATES = 1500

# E19: VRP time-limit sweep
E19_TIME_LIMITS = [120, 240, 360, 600]
E19_FLEET_SIZE = 5
E19_N_WAYPOINTS = 75
