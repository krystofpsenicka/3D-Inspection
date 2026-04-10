import cupy as cp
import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass

from .constants import NORM_EPS


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes a 3D vector, handling zero-norm case."""
    n = norm(v)
    if n < NORM_EPS:
        return np.zeros(3)
    return v / n


@dataclass
class FrustumParams:
    """Parameters defining the camera view frustum."""
    fov_y: float  # Field of View in Y-direction (radians)
    aspect: float  # Aspect ratio (width/height)
    near: float  # Near plane distance
    far: float  # Far plane distance


@dataclass
class EpsilonHyperparams:
    """Tunable hyperparameters for epsilon-visibility."""
    gamma_method: str = "p30"
    epsilon_scale: float = 1.2
    delta_k: int = 8
    delta_agg: str = "max"
    back_face_threshold: float = -1e-6


@dataclass
class OptimizationResult:
    """Result from set-cover optimization. All arrays are on GPU (CuPy)."""
    positions: cp.ndarray       # (K, 3) selected viewpoint positions
    rotations: cp.ndarray       # (K, 3, 3) selected rotation matrices
    visibility_map: cp.ndarray  # (K, M) uint8 — selected viewpoints' visibility
    total_coverage: float
    num_viewpoints: int
    redundancy: float
    optimization_time: float
