import cupy as cp

from shared.geometry import orient_normals_outward as orient_normals_outward  # re-export


def compute_redundancy(visibility_map: cp.ndarray) -> float:
    """Compute mean coverage redundancy from a (K, M) visibility matrix.

    Returns the average number of viewpoints that see each covered point.
    """
    if visibility_map.size == 0:
        return 0.0
    coverage_count = visibility_map.astype(cp.float32).sum(axis=0)
    covered = coverage_count[coverage_count > 0]
    return float(cp.mean(covered)) if len(covered) > 0 else 0.0
