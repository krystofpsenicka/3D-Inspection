import numpy as np


def orient_normals_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flip normals if they predominantly point inward (toward centroid)."""
    centroid = np.mean(points, axis=0)
    radial = points - centroid   # outward radial vectors
    mean_dot = np.mean(np.sum(normals * radial, axis=1))
    if mean_dot < 0:             # majority pointing inward → flip
        return -normals
    return normals
