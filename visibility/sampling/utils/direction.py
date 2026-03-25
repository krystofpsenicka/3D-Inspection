"""GPU-accelerated viewing direction computation for viewpoint sampling."""

import cupy as cp

from ...core.constants import NORM_EPS, KNN_DIRECTION_K, GPU_NN_CHUNK_SIZE


def knn_centroid_direction(query_gpu, targets_gpu,
                           k: int = KNN_DIRECTION_K,
                           normals_gpu=None):
    """Compute viewing direction as centroid of K nearest surface points (GPU).

    When *normals_gpu* is provided, the centroid is weighted by the angular
    deviation of each neighbour's normal from the local mean normal, biasing
    the look-direction toward geometrically complex patches.

    Args:
        query_gpu:   (N, 3) CuPy array — query positions.
        targets_gpu: (M, 3) CuPy array — surface points.
        k:           number of nearest neighbours.
        normals_gpu: (M, 3) CuPy array — surface normals for curvature
                     weighting.  Pass None to disable.

    Returns:
        (N, 3) CuPy float32 — normalised direction vectors.
    """
    n_query = len(query_gpu)
    n_target = len(targets_gpu)
    k = min(k, n_target)

    targets_sq = cp.sum(targets_gpu ** 2, axis=1)  # (M,)
    all_dirs = cp.empty((n_query, 3), dtype=cp.float32)

    for start in range(0, n_query, GPU_NN_CHUNK_SIZE):
        end = min(start + GPU_NN_CHUNK_SIZE, n_query)
        q = query_gpu[start:end]
        q_sq = cp.sum(q ** 2, axis=1, keepdims=True)
        dist_sq = q_sq + targets_sq[cp.newaxis, :] - 2.0 * q @ targets_gpu.T
        cp.maximum(dist_sq, 0.0, out=dist_sq)
        # Get K nearest indices
        knn_idx = cp.argpartition(dist_sq, k, axis=1)[:, :k]
        # Compute centroid of K nearest points for each query
        # Shape: (chunk, k, 3)
        knn_pts = targets_gpu[knn_idx]

        if normals_gpu is not None:
            knn_n = normals_gpu[knn_idx]  # (chunk, k, 3)
            mean_n = knn_n.mean(axis=1, keepdims=True)  # (chunk, 1, 3)
            mean_n /= (cp.linalg.norm(mean_n, axis=2, keepdims=True) + NORM_EPS)
            cos_sim = cp.clip(cp.sum(knn_n * mean_n, axis=2), -1.0, 1.0)
            knn_w = cp.arccos(cos_sim) + NORM_EPS  # (chunk, k)
            knn_w /= knn_w.sum(axis=1, keepdims=True)
            centroid = (knn_pts * knn_w[..., cp.newaxis]).sum(axis=1)
        else:
            centroid = knn_pts.mean(axis=1)  # (chunk, 3)

        dirs = centroid - q
        all_dirs[start:end] = dirs

    norms = cp.linalg.norm(all_dirs, axis=1, keepdims=True)
    norms = cp.maximum(norms, NORM_EPS)
    return all_dirs / norms


def apply_angular_noise(directions_gpu, max_angle_rad):
    """GPU Rodrigues rotation for angular perturbation of direction vectors.

    Args:
        directions_gpu: (N, 3) CuPy array — unit direction vectors.
        max_angle_rad:  maximum rotation angle in radians.

    Returns:
        (N, 3) CuPy float32 — rotated unit direction vectors.
    """
    n = len(directions_gpu)
    if max_angle_rad < 1e-8 or n == 0:
        return directions_gpu.copy()

    angles = cp.random.uniform(0, max_angle_rad, size=n)

    rand_vec = cp.random.randn(n, 3, dtype=cp.float32)
    dot = cp.sum(rand_vec * directions_gpu, axis=1, keepdims=True)
    perp = rand_vec - dot * directions_gpu
    perp_norm = cp.linalg.norm(perp, axis=1, keepdims=True)

    degenerate = (perp_norm < 1e-8).ravel()
    if cp.any(degenerate):
        alt = cp.zeros_like(directions_gpu[degenerate])
        alt[:, 0] = -directions_gpu[degenerate, 1]
        alt[:, 1] = directions_gpu[degenerate, 0]
        alt_norm = cp.linalg.norm(alt, axis=1, keepdims=True)
        alt_norm = cp.maximum(alt_norm, NORM_EPS)
        perp[degenerate] = alt / alt_norm
        perp_norm[degenerate] = 1.0
    perp = perp / cp.maximum(perp_norm, NORM_EPS)

    cos_a = cp.cos(angles)[:, cp.newaxis]
    sin_a = cp.sin(angles)[:, cp.newaxis]
    cross = cp.cross(perp, directions_gpu)
    dot_kv = cp.sum(perp * directions_gpu, axis=1, keepdims=True)
    rotated = directions_gpu * cos_a + cross * sin_a + perp * dot_kv * (1 - cos_a)

    rotated_norm = cp.linalg.norm(rotated, axis=1, keepdims=True)
    rotated = rotated / cp.maximum(rotated_norm, NORM_EPS)
    return rotated
