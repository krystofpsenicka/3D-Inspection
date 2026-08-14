"""Model definitions, seeds, parameter grids."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TOSCA_DIR = os.path.join(MODELS_DIR, "TOSCA-dataset")
EXTRA_MESHES_DIR = os.path.join(MODELS_DIR, "extra_real")
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
    def extra_real(name: str) -> ModelConfig:
        """Additional real high-resolution laser-scanned meshes (Stanford 3D
        Scanning Repository), used for stage F to test that the pipeline
        generalizes beyond the single Duke wreck at comparable/greater geometric
        complexity (Armadillo 0.35M, Dragon 0.87M, Happy Buddha 1.09M faces vs
        Duke 0.58M). Same sensor/robot settings as Duke; each mesh is rescaled to
        ``target_length`` metres. See EXTRA_REAL_MODELS."""
        fname, tlen = _EXTRA_REAL_MESHES[name]
        mesh_path = os.path.join(EXTRA_MESHES_DIR, fname)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Extra real mesh not found: {mesh_path}")
        return ModelConfig(
            name=name,
            mesh_path=mesh_path,
            target_length=tlen,
            mesh_pose=_IDENTITY_POSE,
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

# ── Stage F: additional real high-complexity meshes ─────────────────────────
# name -> (filename under EXTRA_MESHES_DIR, target_length metres). Rescaled to a
# ~20 m structure so the OG/inspection scale is comparable to Duke.
_EXTRA_REAL_MESHES = {
    "armadillo": ("Armadillo.ply", 20.0),
    "dragon": ("dragon_vrip.ply", 20.0),
    "happy_buddha": ("happy_vrip.ply", 20.0),
    # High-complexity laser scans (Stanford XYZ RGB + Lucy), much denser than
    # Duke (0.58M faces): xyz_dragon 7.2M, statuette ~5M, lucy ~28M.
    "xyz_dragon": ("xyzrgb_dragon.ply", 20.0),
    "statuette": ("xyzrgb_statuette.ply", 20.0),
    "lucy": ("lucy.ply", 20.0),
}
EXTRA_REAL_MODELS = list(_EXTRA_REAL_MESHES)


# ── Parameter grids ─────────────────────────────────────────────────────────

E05_COVERAGE_TARGETS = [0.85, 0.90, 0.925, 0.95, 0.97]

E06_COVERAGE_TARGETS = [0.85, 0.90, 0.925, 0.95, 0.97]
E06_OPTIMIZERS = [
    "GreedySetCover", "GreedySetCoverCuda", "LazyGreedySetCover",
    "ExpansionIterative_weighted", "ExpansionIterative_weighted_curvature",
    "ExpansionIterative_cmaes",
]

E07_FLEET_SIZES = [1, 2, 3, 4, 5, 6, 8, 10]
E07_WAYPOINT_COUNTS = [10, 25, 50, 75, 100]

E08_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# E03: Iterative sampler sweeps
E03_T_SPI_VALUES = [1, 5, 25, None]  # samples_per_iteration; None = all-at-once
E03_C_K_VALUES = [1, 2, 3, 4, 6, 8]
E03_C_TRAVEL_WEIGHTS_TOSCA = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
E03_C_TRAVEL_WEIGHTS_DUKE = [0.01, 0.02, 0.03, 0.06, 0.1]
E03_C_POPSIZE_VALUES = [5, 10, 15, 25, 40]
E03_C_MAXITER_VALUES = [5, 10, 20, 40]

# E09: Sampler routing impact
E09_N_ROBOTS = 5
E09_N_CANDIDATES = 1500
