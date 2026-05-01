"""KNN proximity weighting using a fused CUDA kernel."""

import logging

import cupy as cp
import numpy as np

from ...core.constants import CUDA_BLOCK_SIZE

logger = logging.getLogger(__name__)


_KNN_PROXIMITY_KERNEL = cp.RawKernel(
    r"""
#define MAX_K 512

__device__ void heap_sift_down(float* dist, int heap_size, int node) {
    while (true) {
        int largest = node;
        int left = 2 * node + 1;
        int right = 2 * node + 2;
        if (left < heap_size && dist[left] > dist[largest]) largest = left;
        if (right < heap_size && dist[right] > dist[largest]) largest = right;
        if (largest == node) break;
        float tmp = dist[node]; dist[node] = dist[largest]; dist[largest] = tmp;
        node = largest;
    }
}

extern "C" __global__
void knn_proximity(
    const float* __restrict__ queries,   // (num_queries, 3)
    const float* __restrict__ targets,   // (num_targets, 3)
    float* __restrict__ weights,         // (num_queries,)
    int num_queries, int num_targets, int num_neighbors, float inv_sigma
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
    for (int i = 0; i < num_neighbors; i++) {
        heap_dist[i] = 1e30f;
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
                    heap_sift_down(heap_dist, num_neighbors, 0);
                }
            }
        }
        __syncthreads();
    }

    // Early exit for padding threads
    if (query_idx >= num_queries) return;

    // Sum exponential proximity contributions from K nearest
    float total = 0.0f;
    for (int i = 0; i < num_neighbors; i++) {
        total += expf(-sqrtf(heap_dist[i]) * inv_sigma);
    }
    weights[query_idx] = total;
}
""",
    "knn_proximity",
)


def compute_knn_proximity_weights(queries_gpu, targets_gpu, k, sigma):
    """KNN proximity weighting: sum of exponential decay to K nearest targets.

    For each query point, finds the K nearest target points and computes
    sum(exp(-dist * inv_sigma)) as a proximity score.

    Args:
        queries_gpu: (N, 3) CuPy array  --  query positions.
        targets_gpu: (M, 3) CuPy array  --  target positions.
        k:           number of nearest neighbours.
        sigma:       length scale for exponential decay.

    Returns:
        (N,) CuPy float32 array  --  proximity weights (higher = closer to targets).
    """
    num_queries = len(queries_gpu)
    num_targets = len(targets_gpu)
    num_neighbors = min(k, num_targets, 512)
    if num_neighbors < k:
        logger.warning(
            "Clamped k from %d to %d (num_targets=%d, MAX_K=512)", k, num_neighbors, num_targets
        )

    # Ensure input arrays are contiguous for pointer arithmetic in kernel
    queries_gpu = cp.ascontiguousarray(queries_gpu, dtype=cp.float32)
    targets_gpu = cp.ascontiguousarray(targets_gpu, dtype=cp.float32)

    weights = cp.empty(num_queries, dtype=cp.float32)
    grid = ((num_queries + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE,)
    block = (CUDA_BLOCK_SIZE,)
    shared_mem = CUDA_BLOCK_SIZE * 3 * 4  # bytes

    inv_sigma = np.float32(1.0 / sigma)

    _KNN_PROXIMITY_KERNEL(
        grid,
        block,
        (
            queries_gpu,
            targets_gpu,
            weights,
            np.int32(num_queries),
            np.int32(num_targets),
            np.int32(num_neighbors),
            inv_sigma,
        ),
        shared_mem=shared_mem,
    )
    return weights
