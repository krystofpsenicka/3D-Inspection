"""GPU-accelerated local curvature estimation via KNN normal deviation."""

import cupy as cp

from ..core.constants import NORM_EPS, GPU_NN_CHUNK_SIZE, CURVATURE_KNN_K


def compute_local_curvature(query_gpu, targets_gpu, normals_gpu,
                            k=CURVATURE_KNN_K):
    """Curvature proxy: mean angular deviation of KNN normals from local mean.

    For each query point, finds K nearest points in targets, then measures
    how much the normals at those K points deviate from their local mean
    normal.  High deviation = high curvature / geometric complexity.

    Args:
        query_gpu:   (N, 3) CuPy array — positions to evaluate curvature at.
        targets_gpu: (M, 3) CuPy array — surface points.
        normals_gpu: (M, 3) CuPy array — surface normals.
        k:           number of nearest neighbours.

    Returns:
        (N,) CuPy float32 array — curvature proxy values (radians).
    """
    n_query = len(query_gpu)
    n_target = len(targets_gpu)
    k = min(k, n_target)

    targets_sq = cp.sum(targets_gpu ** 2, axis=1)  # (M,)
    curvature = cp.empty(n_query, dtype=cp.float32)

    for start in range(0, n_query, GPU_NN_CHUNK_SIZE):
        end = min(start + GPU_NN_CHUNK_SIZE, n_query)
        q = query_gpu[start:end]  # (chunk, 3)
        q_sq = cp.sum(q ** 2, axis=1, keepdims=True)  # (chunk, 1)
        dist_sq = q_sq + targets_sq[cp.newaxis, :] - 2.0 * q @ targets_gpu.T
        cp.maximum(dist_sq, 0.0, out=dist_sq)

        # K nearest indices: (chunk, k)
        knn_idx = cp.argpartition(dist_sq, k, axis=1)[:, :k]

        # Normals at KNN points: (chunk, k, 3)
        knn_normals = normals_gpu[knn_idx]

        # Local mean normal: (chunk, 1, 3)
        mean_n = knn_normals.mean(axis=1, keepdims=True)
        mean_n_norm = cp.linalg.norm(mean_n, axis=2, keepdims=True)
        mean_n = mean_n / (mean_n_norm + NORM_EPS)

        # Angular deviation of each KNN normal from the mean: (chunk, k)
        cos_sim = cp.clip(cp.sum(knn_normals * mean_n, axis=2), -1.0, 1.0)
        angles = cp.arccos(cos_sim)

        # Mean angular deviation per query point: (chunk,)
        curvature[start:end] = angles.mean(axis=1)

    return curvature
