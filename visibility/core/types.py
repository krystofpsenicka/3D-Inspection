import numpy as np
from numpy.linalg import norm
from time import time as get_time
from dataclasses import dataclass
from typing import List


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes a 3D vector, handling zero-norm case."""
    n = norm(v)
    if n < 1e-12:
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
    gamma_method: str = "median"         # how to aggregate front-facing distances for γ
                                         # options: median, mean, p10, p25, p75, p90
    epsilon_scale: float = 1.0           # multiplier on computed ε (0.5 = finer, 2.0 = coarser)
    delta_k: int = 5                     # k for k-NN in δ estimation
    delta_agg: str = "max"               # aggregation for δ: max, p99, p95, p90
    back_face_threshold: float = -1e-6   # dot-product threshold for back-face check


@dataclass
class ViewpointResult:
    """Result from a single viewpoint query/optimization step."""
    position: np.ndarray
    direction: np.ndarray
    visible_indices: np.ndarray
    coverage_score: float
    computation_time: float


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    method_name: str
    viewpoints: List[ViewpointResult]
    total_coverage: float
    num_viewpoints: int
    total_time: float
    coverage_per_viewpoint: List[float]
    redundancy: float
    visibility_computation_time: float
    optimization_time: float
