"""Local curvature approximation using a fused CUDA kernel."""

import cupy as cp
import numpy as np

from ...core.constants import CUDA_BLOCK_SIZE, CURVATURE_KNN_K

_KNN_CURVATURE_KERNEL = cp.RawKernel(
    r"""
#define MAX_K 512

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
void knn_curvature(
    const float* __restrict__ queries,   // (num_queries, 3)
    const float* __restrict__ targets,   // (num_targets, 3)
    const float* __restrict__ normals,   // (num_targets, 3)
    float* __restrict__ curvature,       // (num_queries,)
    int num_queries, int num_targets, int num_neighbors
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

        if (query_idx < num_queries) {  // Skip padding threads
            int tile_end = min((int)blockDim.x, num_targets - tile);
            for (int j = 0; j < tile_end; j++) {
                float dx = qx - shared_targets[j * 3];
                float dy = qy - shared_targets[j * 3 + 1];
                float dz = qz - shared_targets[j * 3 + 2];
                // Squared distance (no sqrt for efficiency, since we only compare)
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

    // Early exit for padding threads (this is not done earlier to avoid deadlocks in __syncthreads())
    if (query_idx >= num_queries) return;

    // Compute mean normal of K nearest neighbours
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

    // Normalize mean normal
    float inv_norm = 1.0f / (sqrtf(mean_nx * mean_nx + mean_ny * mean_ny + mean_nz * mean_nz) + 1e-12f);
    mean_nx *= inv_norm;
    mean_ny *= inv_norm;
    mean_nz *= inv_norm;

    // Mean angular deviation from the mean normal
    float total_angle = 0.0f;
    for (int i = 0; i < num_neighbors; i++) {
        int neighbor_idx = heap_idx[i];
        // Dot product between neighbour normal and mean normal
        float cos_sim = normals[neighbor_idx * 3]     * mean_nx
                      + normals[neighbor_idx * 3 + 1] * mean_ny
                      + normals[neighbor_idx * 3 + 2] * mean_nz;
        cos_sim = fminf(fmaxf(cos_sim, -1.0f), 1.0f);
        total_angle += acosf(cos_sim);
    }

    curvature[query_idx] = total_angle / (float)num_neighbors;
}
""",
    "knn_curvature",
)


def compute_local_curvature(query_gpu, targets_gpu, normals_gpu, k=CURVATURE_KNN_K):
    """Curvature proxy: mean angular deviation of KNN normals from local mean.

    For each query point, find K nearest points in targets, then measure
    how much the normals at those K points deviate from their local mean
    normal.  High deviation = high curvature / geometric complexity.

    Args:
        query_gpu:   (N, 3) CuPy array  --  positions to evaluate curvature.
        targets_gpu: (M, 3) CuPy array  --  surface points.
        normals_gpu: (M, 3) CuPy array  --  surface normals.
        k:           number of nearest neighbours.

    Returns:
        (N,) CuPy float32 array  --  curvature proxy values.
    """
    num_queries = len(query_gpu)
    num_targets = len(targets_gpu)
    num_neighbors = min(k, num_targets)

    # Ensure contiguous and correct dtype for CUDA kernel (for pointer arithmetic)
    query_gpu = cp.ascontiguousarray(query_gpu, dtype=cp.float32)
    targets_gpu = cp.ascontiguousarray(targets_gpu, dtype=cp.float32)
    normals_gpu = cp.ascontiguousarray(normals_gpu, dtype=cp.float32)

    curvature = cp.empty(num_queries, dtype=cp.float32)
    grid = ((num_queries + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE,)
    block = (CUDA_BLOCK_SIZE,)
    shared_mem = CUDA_BLOCK_SIZE * 3 * 4  # bytes

    _KNN_CURVATURE_KERNEL(
        grid,
        block,
        (
            query_gpu,
            targets_gpu,
            normals_gpu,
            curvature,
            np.int32(num_queries),
            np.int32(num_targets),
            np.int32(num_neighbors),
        ),
        shared_mem=shared_mem,
    )
    return curvature
