"""Central experiment configuration: model definitions, seeds, parameter grids."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TOSCA_DIR = os.path.join(MODELS_DIR, "TOSCA-dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

# ── Corrected mesh pose for Duke of Lancaster ────────────────────────────────
# trimesh loads Y-up GLB; Isaac Sim applies Y-up→Z-up.  We compose the
# mesh_pose (180° X) with the Y→Z correction (+90° X) = R_x(-90°).
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
    """Configuration for a 3D model used in experiments."""
    name: str
    mesh_path: str
    target_length: float
    mesh_pose: list
    num_surface_points: int
    num_candidates: int
    frustum: FrustumConfig
    collision_radius: float = 0.35
    voxel_resolution: float = 0.10

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
        """Create config for a TOSCA dataset model.

        Args:
            name: e.g. "wolf0", "cat0", "david0" (category + pose index).
        """
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


# ── Experiment configuration ─────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Top-level config wrapping a model and experiment parameters."""
    model: ModelConfig
    target_coverage: float = 0.95
    max_viewpoints: int = 1000


# ── Seed lists ───────────────────────────────────────────────────────────────

SEEDS_3 = [42, 123, 7]
SEEDS_5 = [42, 123, 7, 2024, 314]
SEEDS_10 = SEEDS_5 + [999, 55, 8888, 1337, 2025]

# ── TOSCA model lists (updated after validation in Step 0.4) ─────────────────

# Representative: small / medium / large vertex count
TOSCA_REPRESENTATIVE = ["wolf0", "cat0", "david0"]

# One per category (for E15 cross-model sweep)
TOSCA_ALL = [
    "wolf0", "cat0", "centaur0", "david0", "dog0",
    "gorilla1", "horse0", "michael0", "victoria0",
]

# After running validate_models.py, invalid models are removed from these lists.
# The validated list is stored here:
TOSCA_VALID: Optional[list] = None  # Set by validate_models.py


# ── Parameter grids ──────────────────────────────────────────────────────────

E01_STRATEGIES = ["weighted", "weighted_curvature", "targeted_25", "targeted_50", "cmaes"]

E02_CANDIDATE_COUNTS = [250, 500, 750, 1000, 1500, 2000, 3000, 5000]

E03_COVERAGE_TARGETS = [0.85, 0.90, 0.925, 0.95, 0.97]

E04_COVERAGE_TARGETS = [0.85, 0.90, 0.925, 0.95, 0.97]
E04_OPTIMIZERS = [
    "GreedySetCoverCuda",
    "LazyGreedySetCoverCuda",
    "ExpansionIterativeSetCover",
    "GreedySetCover",
    "LazyGreedySetCover",
]

E05_CANDIDATE_COUNTS = [500, 1000, 2000, 5000, 10000]
E05_POINT_COUNTS = [50_000, 100_000, 200_000]

E06_FLEET_SIZES = [1, 2, 3, 4, 5, 6, 8, 10]
E06_WAYPOINT_COUNTS = [10, 25, 50, 75, 100]

E07_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

E10_RESOLUTIONS = [0.25, 0.50, 0.75, 1.0, 1.5]
