import cupy as cp
import numpy as np


def compute_redundancy(visibility_map: cp.ndarray) -> float:
    """Compute mean coverage redundancy from a (K, M) visibility matrix.

    Returns the average number of viewpoints that see each covered point.
    """
    if visibility_map.size == 0:
        return 0.0
    coverage_count = visibility_map.astype(cp.float32).sum(axis=0)
    covered = coverage_count[coverage_count > 0]
    return float(cp.mean(covered)) if len(covered) > 0 else 0.0


def orient_normals_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flip normals if they predominantly point inward (toward centroid)."""
    centroid = np.mean(points, axis=0)
    radial = points - centroid   # outward radial vectors
    mean_dot = np.mean(np.sum(normals * radial, axis=1))
    if mean_dot < 0:             # majority pointing inward -> flip
        return -normals
    return normals
