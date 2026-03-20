import numpy as np
from numpy.linalg import norm
from time import time as get_time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
class ViewpointResult:
    """Result from a single viewpoint query/optimization step."""
    position: np.ndarray
    orientation: np.ndarray  # quaternion [qw,qx,qy,qz]
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
    visibility_map: Optional[Dict[Tuple, np.ndarray]] = field(default=None, repr=False)
