"""Viewing direction computation for viewpoint sampling."""

import cupy as cp
import numpy as np

from ...core.constants import CUDA_BLOCK_SIZE, KNN_DIRECTION_K, NORM_EPS

_KNN_DIRECTION_KERNEL = cp.RawKernel(
    r"""
#define MAX_K 512
#define NORM_EPS 1e-12f

__device__ void heap_sift_down(float* dist, int* indices, int heap_size, int node) {
    while (true) {
        int largest = node;
        int left = 2 * node + 1;
        int right = 2 * node + 2;
        if (left < heap_size && dist[left] > dist[largest]) largest = left;
        if (right < heap_size && dist[right] > dist[largest]) largest = right;
        if (largest == node) break;
        float tmp_d = dist[node]; dist[node] = dist[largest]; dist[largest] = tmp_d;
        int tmp_i = indices[node]; indices[node] = indices[largest]; indices[largest] = tmp_i;
        node = largest;
    }
}

extern "C" __global__
void knn_direction(
    const float* __restrict__ queries,      // (num_queries, 3)
    const float* __restrict__ targets,      // (num_targets, 3)
    const float* __restrict__ normals,      // (num_targets, 3)  --  unused when use_normals==0
    float* __restrict__ directions,         // (num_queries, 3)
    int num_queries, int num_targets, int num_neighbors, int use_normals
) {
    int query_idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Load query point into registers
    float qx = 0.0f, qy = 0.0f, qz = 0.0f;
    if (query_idx < num_queries) {
        qx = queries[query_idx * 3];
        qy = queries[query_idx * 3 + 1];
        qz = queries[query_idx * 3 + 2];
    }

    // K-best max-heap in local memory (root = largest distance)
    float heap_dist[MAX_K];
    int   heap_idx[MAX_K];
    for (int i = 0; i < num_neighbors; i++) {
        heap_dist[i] = 1e30f;
        heap_idx[i] = 0;
    }

    // Shared memory for tiled target loading
    extern __shared__ float shared_targets[];  // blockDim.x * 3

    // Tiled KNN scan over all targets
    for (int tile = 0; tile < num_targets; tile += blockDim.x) {
        int target_idx = tile + threadIdx.x;
        if (target_idx < num_targets) {
            shared_targets[threadIdx.x * 3]     = targets[target_idx * 3];
            shared_targets[threadIdx.x * 3 + 1] = targets[target_idx * 3 + 1];
            shared_targets[threadIdx.x * 3 + 2] = targets[target_idx * 3 + 2];
        }
        __syncthreads();

        if (query_idx < num_queries) {
            int tile_end = min((int)blockDim.x, num_targets - tile);
            for (int j = 0; j < tile_end; j++) {
                float dx = qx - shared_targets[j * 3];
                float dy = qy - shared_targets[j * 3 + 1];
                float dz = qz - shared_targets[j * 3 + 2];
                float dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq < heap_dist[0]) {
                    heap_dist[0] = dist_sq;
                    heap_idx[0] = tile + j;
                    heap_sift_down(heap_dist, heap_idx, num_neighbors, 0);
                }
            }
        }
        __syncthreads();
    }

    if (query_idx >= num_queries) return;

    // ---------- Compute centroid direction ----------

    float cx = 0.0f, cy = 0.0f, cz = 0.0f;

    if (use_normals) {
        // Weight by angular deviation from mean normal

        // Step 1: mean normal of K nearest neighbours
        float mean_nx = 0.0f, mean_ny = 0.0f, mean_nz = 0.0f;
        for (int i = 0; i < num_neighbors; i++) {
            int neighbor_idx = heap_idx[i];
            mean_nx += normals[neighbor_idx * 3];
            mean_ny += normals[neighbor_idx * 3 + 1];
            mean_nz += normals[neighbor_idx * 3 + 2];
        }
        mean_nx /= (float)num_neighbors;
        mean_ny /= (float)num_neighbors;
        mean_nz /= (float)num_neighbors;

        float inv_norm = 1.0f / (sqrtf(mean_nx * mean_nx + mean_ny * mean_ny + mean_nz * mean_nz) + NORM_EPS);
        mean_nx *= inv_norm;
        mean_ny *= inv_norm;
        mean_nz *= inv_norm;

        // Step 2: weighted centroid  --  compute weight and accumulate in one pass
        float total_weight = 0.0f;
        for (int i = 0; i < num_neighbors; i++) {
            int neighbor_idx = heap_idx[i];
            // Dot product = cosine similarity between neighbor normal and mean normal
            float cos_sim = normals[neighbor_idx * 3]     * mean_nx
                          + normals[neighbor_idx * 3 + 1] * mean_ny
                          + normals[neighbor_idx * 3 + 2] * mean_nz;
            cos_sim = fminf(fmaxf(cos_sim, -1.0f), 1.0f);
            float w = acosf(cos_sim) + NORM_EPS;
            total_weight += w;
            cx += w * targets[neighbor_idx * 3];
            cy += w * targets[neighbor_idx * 3 + 1];
            cz += w * targets[neighbor_idx * 3 + 2];
        }
        float inv_w = 1.0f / total_weight;
        cx *= inv_w;
        cy *= inv_w;
        cz *= inv_w;
    } else {
        // Simple centroid: mean of K nearest target positions
        for (int i = 0; i < num_neighbors; i++) {
            int neighbor_idx = heap_idx[i];
            cx += targets[neighbor_idx * 3];
            cy += targets[neighbor_idx * 3 + 1];
            cz += targets[neighbor_idx * 3 + 2];
        }
        cx /= (float)num_neighbors;
        cy /= (float)num_neighbors;
        cz /= (float)num_neighbors;
    }

    // Direction = centroid - query, and normalize
    float dx = cx - qx;
    float dy = cy - qy;
    float dz = cz - qz;
    float inv_len = 1.0f / (sqrtf(dx * dx + dy * dy + dz * dz) + NORM_EPS);
    directions[query_idx * 3]     = dx * inv_len;
    directions[query_idx * 3 + 1] = dy * inv_len;
    directions[query_idx * 3 + 2] = dz * inv_len;
}
""",
    "knn_direction",
)


def knn_centroid_direction(query_gpu, targets_gpu, k: int = KNN_DIRECTION_K, normals_gpu=None):
    """Compute viewing direction as centroid of K nearest surface points.

    When *normals_gpu* is passed, the centroid computation is weighted by the angular
    deviation of each neighbour's normal from the local mean normal, biasing
    the look-direction toward geometrically complex patches.

    Args:
        query_gpu:   (N, 3) CuPy array  --  query positions.
        targets_gpu: (M, 3) CuPy array  --  surface points.
        k:           number of nearest neighbours.
        normals_gpu: (M, 3) CuPy array  --  surface normals for curvature
                     weighting.

    Returns:
        (N, 3) CuPy float32  --  normalized direction vectors.
    """
    num_queries = len(query_gpu)
    num_targets = len(targets_gpu)
    num_neighbors = min(k, num_targets)

    query_gpu = cp.ascontiguousarray(query_gpu, dtype=cp.float32)
    targets_gpu = cp.ascontiguousarray(targets_gpu, dtype=cp.float32)

    use_normals = 1 if normals_gpu is not None else 0
    if normals_gpu is not None:
        normals_gpu = cp.ascontiguousarray(normals_gpu, dtype=cp.float32)
    else:
        normals_gpu = targets_gpu  # dummy — never read when use_normals==0

    directions = cp.empty((num_queries, 3), dtype=cp.float32)
    grid = ((num_queries + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE,)
    block = (CUDA_BLOCK_SIZE,)
    shared_mem = CUDA_BLOCK_SIZE * 3 * 4  # bytes

    _KNN_DIRECTION_KERNEL(
        grid,
        block,
        (
            query_gpu,
            targets_gpu,
            normals_gpu,
            directions,
            np.int32(num_queries),
            np.int32(num_targets),
            np.int32(num_neighbors),
            np.int32(use_normals),
        ),
        shared_mem=shared_mem,
    )
    return directions


def apply_angular_noise(directions_gpu, max_angle_rad, rng=None):
    """GPU Rodrigues rotation for angular perturbation of direction vectors.

    Args:
        directions_gpu: (N, 3) CuPy array of unit direction vectors.
        max_angle_rad:  maximum rotation angle in radians.
        rng:            optional cp.random.Generator. Falls back to the
                        global RNG when ``None``.

    Returns:
        (N, 3) CuPy float32 - rotated unit direction vectors.
    """
    n = len(directions_gpu)
    if max_angle_rad < 1e-8 or n == 0:
        return directions_gpu.copy()

    if rng is None:
        rng = cp.random.default_rng()

    # random angles
    angles = cp.minimum(
        cp.abs(rng.standard_normal(size=n, dtype=cp.float32) * (max_angle_rad / 3)),
        max_angle_rad,
    )

    # random vectors to rotate around, perpendicular to the original direction
    rand_vec = rng.standard_normal(size=(n, 3), dtype=cp.float32)
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

    # Rodrigues rotation: v_rot = v*cos(a) + (k x v)*sin(a) + k*(k . v)*(1 - cos(a))
    cos_a = cp.cos(angles)[:, cp.newaxis]
    sin_a = cp.sin(angles)[:, cp.newaxis]
    cross = cp.cross(perp, directions_gpu)
    dot_kv = cp.sum(perp * directions_gpu, axis=1, keepdims=True)
    rotated = directions_gpu * cos_a + cross * sin_a + perp * dot_kv * (1 - cos_a)

    rotated_norm = cp.linalg.norm(rotated, axis=1, keepdims=True)
    rotated = rotated / cp.maximum(rotated_norm, NORM_EPS)
    return rotated
