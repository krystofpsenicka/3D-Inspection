import logging
import numpy as np
import cupy as cp
import open3d as o3d
from numpy.linalg import norm
from time import time as get_time
from typing import Optional, Tuple

from ..core.types import FrustumParams, EpsilonHyperparams
from ..core.base_cuda import VisibilityQueryCuda
from ..core.constants import (
    NORM_EPS, CUDA_BLOCK_SIZE, DELTA_AGG_FUNCS_CP, GAMMA_AGG_FUNCS_CP,
    DELTA_DEFAULT, DELTA_SAMPLE_SIZE, GAMMA_FALLBACK_DIVISOR,
)

logger = logging.getLogger(__name__)

# ── Single-viewpoint CUDA kernels ────────────────────────────────────────────

_SCATTER_MIN_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void scatter_min_occluders(
    const int* bin_keys,        // (M,) bin index for each occluder
    const float* distances,     // (M,) distance for each occluder
    int* bin_min_dist,          // (num_bins,) output: min distance as int bits
    int M
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= M) return;

    int bin = bin_keys[idx];
    float d = distances[idx];
    int d_int = __float_as_int(d);
    atomicMin(&bin_min_dist[bin], d_int);
}
''', 'scatter_min_occluders')


_VISIBILITY_CHECK_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void check_visibility(
    const int* bin_keys,        // (K,) bin index for each front-facing point
    const float* distances,     // (K,) distance for each front-facing point
    const int* bin_min_dist,    // (num_bins,) occluder min distance as int bits
    int* visible,               // (K,) output: 1 if visible, 0 if occluded
    int K
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= K) return;

    int bin = bin_keys[idx];
    int occluder_int = bin_min_dist[bin];
    float occluder_dist = __int_as_float(occluder_int);
    float my_dist = distances[idx];

    visible[idx] = (my_dist <= occluder_dist) ? 1 : 0;
}
''', 'check_visibility')

# ── Batch CUDA kernels ───────────────────────────────────────────────────────

_BATCH_PREPROCESS_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void batch_epsilon_preprocess(
    const float* points,            // (M, 3)
    const float* normals,           // (M, 3)
    const float* viewpoints,        // (N, 3)
    const int* vp_idx,              // (R,) compact pair viewpoint indices
    const int* pt_idx,              // (R,) compact pair point indices
    float* pair_distances,          // (R,) output
    float* pair_theta,              // (R,) output
    float* pair_phi,                // (R,) output
    unsigned char* pair_front_face, // (R,) output
    float back_face_threshold, int R
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= R) return;
    int vp = vp_idx[idx], pt = pt_idx[idx];

    float dx = points[pt*3]   - viewpoints[vp*3];
    float dy = points[pt*3+1] - viewpoints[vp*3+1];
    float dz = points[pt*3+2] - viewpoints[vp*3+2];

    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float inv_d = 1.0f / (dist + 1e-12f);
    pair_distances[idx] = dist;

    // Back-face: dot(view_dir, normal)
    float dot_val = (dx*inv_d)*normals[pt*3] + (dy*inv_d)*normals[pt*3+1] + (dz*inv_d)*normals[pt*3+2];
    pair_front_face[idx] = (dot_val < back_face_threshold) ? 1 : 0;

    // Spherical angles
    pair_theta[idx] = atan2f(dy, dx);
    pair_phi[idx] = asinf(fminf(fmaxf(dz * inv_d, -1.0f), 1.0f));
}
''', 'batch_epsilon_preprocess')


_BATCH_SCATTER_MIN_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void batch_scatter_min_occluders(
    const int* pair_vp_idx,         // (R_occ,)
    const int* pair_local_bin,      // (R_occ,)
    const float* pair_dist,         // (R_occ,)
    const int* bin_offsets,         // (N+1,) cumulative bin counts
    int* bin_min_dist,              // (total_bins,) output
    int R_occ
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= R_occ) return;
    int global_bin = bin_offsets[pair_vp_idx[idx]] + pair_local_bin[idx];
    float d = pair_dist[idx];
    atomicMin(&bin_min_dist[global_bin], __float_as_int(d));
}
''', 'batch_scatter_min_occluders')


_BATCH_VISIBILITY_CHECK_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void batch_check_visibility(
    const int* pair_vp_idx,         // (R_front,)
    const int* pair_pt_idx,         // (R_front,)
    const int* pair_local_bin,      // (R_front,)
    const float* pair_dist,         // (R_front,)
    const int* bin_offsets,         // (N+1,)
    const int* bin_min_dist,        // (total_bins,)
    unsigned char* V,               // (N * M) output
    int R_front, int M
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= R_front) return;
    int global_bin = bin_offsets[pair_vp_idx[idx]] + pair_local_bin[idx];
    float occ_dist = __int_as_float(bin_min_dist[global_bin]);
    if (pair_dist[idx] <= occ_dist)
        V[pair_vp_idx[idx] * M + pair_pt_idx[idx]] = 1;
}
''', 'batch_check_visibility')


_SEGMENTED_MINMAX_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void segmented_minmax(
    const float* pair_theta_shifted,  // (R,) shifted theta values per pair
    const float* pair_phi_shifted,    // (R,) shifted phi values per pair
    const int* segment_starts,        // (n_active,) first pair index for each active VP
    const int* segment_lengths,       // (n_active,) number of pairs in each segment
    const int* segment_vp_ids,        // (n_active,) which VP each segment belongs to
    float* theta_min_per_vp,          // (N,) output
    float* theta_max_per_vp,          // (N,) output
    float* phi_min_per_vp,            // (N,) output
    float* phi_max_per_vp,            // (N,) output
    int n_active
) {
    int seg = blockDim.x * blockIdx.x + threadIdx.x;
    if (seg >= n_active) return;

    int vp = segment_vp_ids[seg];
    int start = segment_starts[seg];
    int len = segment_lengths[seg];

    float t_min = 1e30f, t_max = -1e30f;
    float p_min = 1e30f, p_max = -1e30f;

    for (int i = start; i < start + len; i++) {
        float t = pair_theta_shifted[i];
        float p = pair_phi_shifted[i];
        if (t < t_min) t_min = t;
        if (t > t_max) t_max = t;
        if (p < p_min) p_min = p;
        if (p > p_max) p_max = p;
    }

    theta_min_per_vp[vp] = t_min;
    theta_max_per_vp[vp] = t_max;
    phi_min_per_vp[vp] = p_min;
    phi_max_per_vp[vp] = p_max;
}
''', 'segmented_minmax')


class EpsilonVisibilityQueryCuda(VisibilityQueryCuda):
    """GPU-accelerated epsilon-visibility using CuPy and custom CUDA kernels.

    The main speedup is in _check_occlusion_gpu which replaces the Python
    for-loops with scatter-min and parallel visibility kernels.
    """

    def __init__(self, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams,
                 epsilon_deg: Optional[float] = None,
                 hyperparams: Optional[EpsilonHyperparams] = None):
        super().__init__(target_points, normals, frustum_params)
        self.hp = hyperparams or EpsilonHyperparams()

        if epsilon_deg is not None:
            logger.info("Using provided epsilon: %s degrees", epsilon_deg)
            self.fixed_epsilon = np.deg2rad(epsilon_deg)
            self.delta = None
            logger.info("Using Epsilon (radians): %.6f (%.3f degrees)",
                        self.fixed_epsilon, epsilon_deg)
        else:
            logger.info("Epsilon not provided, estimating delta from point set...")
            self.fixed_epsilon = None
            self.delta = self._estimate_delta()
            logger.info("Estimated delta (sampling density): %.6f", self.delta)

    # ── Delta estimation ─────────────────────────────────────────────────

    def _estimate_delta(self):
        """Estimate sampling density delta (Lemma 6.1): GPU brute-force k-NN."""
        sample_size = min(DELTA_SAMPLE_SIZE, self.num_points)
        if sample_size == 0:
            return DELTA_DEFAULT

        k = self.hp.delta_k
        agg_func = DELTA_AGG_FUNCS_CP[self.hp.delta_agg]

        sample_indices = cp.random.choice(self.num_points, sample_size, replace=False)
        sample_points = self.gpu_points[sample_indices]  # (S, 3)

        # Squared-distance matrix: (S, N)
        all_sq = cp.sum(self.gpu_points ** 2, axis=1)  # (N,)
        sample_sq = cp.sum(sample_points ** 2, axis=1, keepdims=True)  # (S, 1)
        sq_dists = sample_sq + all_sq[None, :] - 2.0 * sample_points @ self.gpu_points.T
        cp.maximum(sq_dists, 0.0, out=sq_dists)

        sq_dists[cp.arange(sample_size), sample_indices] = cp.inf
        k = min(k, sq_dists.shape[1] - 1)
        topk_sq = cp.partition(sq_dists, k - 1, axis=1)[:, :k]
        knn_dists = cp.sqrt(topk_sq)  # (S, k)

        row_agg = agg_func(knn_dists, axis=1)  # (S,)
        return float(agg_func(row_agg))

    # ── Per-viewpoint epsilon ────────────────────────────────────────────

    def _compute_epsilon_gpu(self, distances_gpu, front_facing_gpu):
        """Compute per-viewpoint epsilon = 2*arctan(delta/(4*gamma))."""
        front_distances = distances_gpu[front_facing_gpu]
        gamma_func = GAMMA_AGG_FUNCS_CP[self.hp.gamma_method]
        if len(front_distances) == 0:
            gamma = self.frustum_params.far / GAMMA_FALLBACK_DIVISOR
        else:
            gamma = float(gamma_func(front_distances))
        gamma = max(gamma, 1e-6)
        return 2.0 * np.arctan(self.delta / (4.0 * gamma)) * self.hp.epsilon_scale

    # ── Single-viewpoint visibility ──────────────────────────────────────

    def compute_visibility(self, viewpoint_gpu: cp.ndarray,
                           rotmat_gpu: cp.ndarray) -> Tuple[cp.ndarray, float]:
        """Compute epsilon-visible region using GPU back-face and occlusion checks.

        Args:
            viewpoint_gpu: (3,) CuPy array -- viewpoint position on GPU.
            rotmat_gpu:    (3, 3) CuPy array -- rotation matrix on GPU.

        Returns:
            (visible_indices, comp_time) where visible_indices is CuPy int64.
        """
        start = get_time()

        frustum_indices = self.points_in_frustum_gpu(viewpoint_gpu, rotmat_gpu)

        if len(frustum_indices) == 0:
            return cp.array([], dtype=cp.int64), get_time() - start

        # GPU back-face check
        frustum_points = self.gpu_points[frustum_indices]
        frustum_normals = self.gpu_normals[frustum_indices]

        vp_gpu = viewpoint_gpu.astype(cp.float32)
        view_dirs = frustum_points - vp_gpu
        view_dirs_norm = cp.linalg.norm(view_dirs, axis=1)
        view_dirs = view_dirs / (view_dirs_norm[:, cp.newaxis] + NORM_EPS)

        dot_products = cp.sum(view_dirs * frustum_normals, axis=1)
        front_facing = dot_products < self.hp.back_face_threshold

        # Compute per-viewpoint epsilon
        if self.fixed_epsilon is not None:
            epsilon = self.fixed_epsilon
        else:
            epsilon = self._compute_epsilon_gpu(view_dirs_norm, front_facing)

        visible_mask = self._check_occlusion_gpu(
            vp_gpu, frustum_points, front_facing, epsilon
        )

        visible_indices = frustum_indices[cp.where(visible_mask)[0]]

        comp_time = get_time() - start
        return visible_indices, comp_time

    def _check_occlusion_gpu(self, viewpoint_gpu: cp.ndarray, points_gpu: cp.ndarray,
                              front_facing: cp.ndarray, epsilon: float) -> cp.ndarray:
        """GPU occlusion check using scatter-min CUDA kernels."""
        num_points = len(points_gpu)
        if num_points == 0:
            return cp.array([], dtype=cp.bool_)

        visible = front_facing.copy()

        if not cp.any(front_facing):
            return visible

        if epsilon < 1e-6:
            epsilon = 1e-6

        # Compute relative positions and distances
        relative = points_gpu - viewpoint_gpu
        distances = cp.linalg.norm(relative, axis=1)

        # Spherical binning -- scoped to frustum culled target points angular extents
        theta = cp.arctan2(relative[:, 1], relative[:, 0])
        phi = cp.arcsin(cp.clip(relative[:, 2] / (distances + NORM_EPS), -1, 1))

        theta_min, theta_max = float(theta.min()), float(theta.max())
        phi_min, phi_max = float(phi.min()), float(phi.max())

        num_bins_theta = max(1, int(np.ceil((theta_max - theta_min + epsilon) / epsilon)))
        num_bins_phi = max(1, int(np.ceil((phi_max - phi_min + epsilon) / epsilon)))

        theta_bins = cp.clip(((theta - theta_min) / epsilon).astype(cp.int32), 0, num_bins_theta - 1)
        phi_bins = cp.clip(((phi - phi_min) / epsilon).astype(cp.int32), 0, num_bins_phi - 1)

        num_bins = num_bins_theta * num_bins_phi

        bin_keys_all = (theta_bins * num_bins_phi + phi_bins).astype(cp.int32)

        # --- Scatter-min for occluders (non-front-facing points) ---
        occluder_mask = ~front_facing
        occluder_indices = cp.where(occluder_mask)[0]

        bin_min_dist = cp.full(num_bins, 0x7F7FFFFF, dtype=cp.int32)

        if len(occluder_indices) > 0:
            occluder_bin_keys = bin_keys_all[occluder_indices].astype(cp.int32)
            occluder_distances = distances[occluder_indices].astype(cp.float32)

            M = len(occluder_indices)
            grid_size = (M + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
            _SCATTER_MIN_KERNEL(
                (grid_size,), (CUDA_BLOCK_SIZE,),
                (occluder_bin_keys, occluder_distances, bin_min_dist, np.int32(M))
            )

        # --- Parallel visibility check for front-facing points ---
        front_indices = cp.where(front_facing)[0]

        if len(front_indices) > 0:
            front_bin_keys = bin_keys_all[front_indices].astype(cp.int32)
            front_distances = distances[front_indices].astype(cp.float32)

            K = len(front_indices)
            vis_result = cp.zeros(K, dtype=cp.int32)

            grid_size = (K + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
            _VISIBILITY_CHECK_KERNEL(
                (grid_size,), (CUDA_BLOCK_SIZE,),
                (front_bin_keys, front_distances, bin_min_dist, vis_result, np.int32(K))
            )

            visible[front_indices] = vis_result.astype(cp.bool_)

        return visible

    # ── Batch epsilon visibility ─────────────────────────────────────────

    def compute_visibility_batch(self, positions, rotmats) -> Tuple[cp.ndarray, float]:
        """Batch epsilon visibility with multi-VP CUDA kernels.

        Uses 4 CUDA kernels: batch frustum cull, fused pair preprocessing,
        multi-VP scatter-min for occluders, multi-VP visibility check.

        Args:
            positions: (N, 3) CuPy float32 -- viewpoint positions.
            rotmats:   (N, 3, 3) CuPy float32 -- rotation matrices.

        Returns:
            ``(V, total_time)`` where *V* is ``(N, M)`` uint8 CuPy array (visibility matrix).
        """
        start = get_time()
        N, M = len(positions), self.num_points
        V_flat = cp.zeros(N * M, dtype=cp.uint8)

        # 1. Batch frustum cull
        frustum_mask = self.batch_points_in_frustum_gpu(positions, rotmats)
        vp_idx, pt_idx = cp.where(frustum_mask)
        del frustum_mask
        R = len(vp_idx)
        if R == 0:
            return V_flat.reshape(N, M), get_time() - start

        vp_idx_i32 = vp_idx.astype(cp.int32)
        pt_idx_i32 = pt_idx.astype(cp.int32)

        # 2. Fused pair preprocessing
        points_f32 = cp.ascontiguousarray(self.gpu_points.astype(cp.float32))
        normals_f32 = cp.ascontiguousarray(self.gpu_normals.astype(cp.float32))
        positions_f32 = cp.ascontiguousarray(cp.asarray(positions, dtype=cp.float32))

        pair_dist = cp.empty(R, dtype=cp.float32)
        pair_theta = cp.empty(R, dtype=cp.float32)
        pair_phi = cp.empty(R, dtype=cp.float32)
        pair_front = cp.empty(R, dtype=cp.uint8)

        grid_size = (R + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
        _BATCH_PREPROCESS_KERNEL(
            (grid_size,), (CUDA_BLOCK_SIZE,),
            (points_f32, normals_f32, positions_f32,
             vp_idx_i32, pt_idx_i32,
             pair_dist, pair_theta, pair_phi, pair_front,
             np.float32(self.hp.back_face_threshold), np.int32(R))
        )

        # 3. Compute epsilon (fixed or per-VP)
        if self.fixed_epsilon is not None:
            epsilon_per_vp = cp.full(N, self.fixed_epsilon, dtype=cp.float32)
        else:
            epsilon_per_vp = self._compute_batch_epsilon(
                vp_idx_i32, pair_dist, pair_front, N)

        # 4. Angular extents and bin parameters
        # Shift theta/phi to non-negative for segmented min/max
        THETA_SHIFT = cp.float32(np.pi)
        PHI_SHIFT = cp.float32(np.pi / 2.0)
        theta_shifted = pair_theta + THETA_SHIFT
        phi_shifted = pair_phi + PHI_SHIFT

        # Segmented min/max via CUDA kernel
        unique_vps, first_idx, counts = cp.unique(
            vp_idx_i32, return_index=True, return_counts=True)
        n_active = len(unique_vps)

        theta_min_vp = cp.zeros(N, dtype=cp.float32)
        theta_max_vp = cp.zeros(N, dtype=cp.float32)
        phi_min_vp = cp.zeros(N, dtype=cp.float32)
        phi_max_vp = cp.zeros(N, dtype=cp.float32)

        grid_seg = (int(n_active) + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
        _SEGMENTED_MINMAX_KERNEL(
            (grid_seg,), (CUDA_BLOCK_SIZE,),
            (theta_shifted, phi_shifted,
             first_idx.astype(cp.int32), counts.astype(cp.int32),
             unique_vps.astype(cp.int32),
             theta_min_vp, theta_max_vp, phi_min_vp, phi_max_vp,
             np.int32(int(n_active)))
        )

        # Compute num_bins per VP
        eps_per_vp = cp.maximum(epsilon_per_vp, cp.float32(1e-6))
        num_bins_theta_vp = cp.maximum(
            cp.int32(1),
            cp.ceil((theta_max_vp - theta_min_vp + eps_per_vp) / eps_per_vp
                    ).astype(cp.int32))
        num_bins_phi_vp = cp.maximum(
            cp.int32(1),
            cp.ceil((phi_max_vp - phi_min_vp + eps_per_vp) / eps_per_vp
                    ).astype(cp.int32))

        # 5. Compute bin keys for each pair
        num_bins_per_vp = (num_bins_theta_vp * num_bins_phi_vp).astype(cp.int32)
        bin_offsets = cp.zeros(N + 1, dtype=cp.int32)
        bin_offsets[1:] = cp.cumsum(num_bins_per_vp)
        total_bins = int(bin_offsets[-1])

        # Per-pair bin assignment
        pair_eps = epsilon_per_vp[vp_idx]
        pair_eps = cp.maximum(pair_eps, cp.float32(1e-6))
        pair_theta_min = theta_min_vp[vp_idx]
        pair_phi_min = phi_min_vp[vp_idx]
        pair_nbt = num_bins_theta_vp[vp_idx]
        pair_nbp = num_bins_phi_vp[vp_idx]

        t_bins = cp.clip(
            ((theta_shifted - pair_theta_min) / pair_eps).astype(cp.int32),
            0, pair_nbt - 1)
        p_bins = cp.clip(
            ((phi_shifted - pair_phi_min) / pair_eps).astype(cp.int32),
            0, pair_nbp - 1)
        local_bin = (t_bins * pair_nbp + p_bins).astype(cp.int32)

        # 6. Scatter-min for occluders
        bin_min_dist = cp.full(total_bins, 0x7F7FFFFF, dtype=cp.int32)
        occ_mask = pair_front == 0
        occ_idx = cp.where(occ_mask)[0]

        if len(occ_idx) > 0:
            R_occ = len(occ_idx)
            grid_occ = (R_occ + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
            _BATCH_SCATTER_MIN_KERNEL(
                (grid_occ,), (CUDA_BLOCK_SIZE,),
                (vp_idx_i32[occ_idx], local_bin[occ_idx],
                 pair_dist[occ_idx], bin_offsets, bin_min_dist,
                 np.int32(R_occ))
            )

        # 7. Visibility check for front-facing pairs
        front_mask = pair_front.astype(cp.bool_)
        front_idx = cp.where(front_mask)[0]

        if len(front_idx) > 0:
            R_front = len(front_idx)
            grid_front = (R_front + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
            _BATCH_VISIBILITY_CHECK_KERNEL(
                (grid_front,), (CUDA_BLOCK_SIZE,),
                (vp_idx_i32[front_idx], pt_idx_i32[front_idx],
                 local_bin[front_idx], pair_dist[front_idx],
                 bin_offsets, bin_min_dist,
                 V_flat, np.int32(R_front), np.int32(M))
            )

        V = V_flat.reshape(N, M)
        total_time = get_time() - start
        logger.info("[EpsilonCuda] Batch visibility for %d VPs: %d pairs, "
                    "%d bins, %.2fs", N, R, total_bins, total_time)
        return V, total_time

    def _compute_batch_epsilon(self, vp_idx_i32, pair_dist, pair_front, N):
        """Compute per-VP epsilon from segmented front-facing distances.

        For ``mean`` gamma: fully vectorised via ``cp.scatter_add``.
        For percentile/median gamma: per-segment GPU slicing (no CPU transfer).
        Final epsilon is computed in one vectorised CuPy expression.
        """
        fallback_gamma = cp.float32(self.frustum_params.far / GAMMA_FALLBACK_DIVISOR)
        gamma_per_vp = cp.full(N, fallback_gamma, dtype=cp.float32)

        # Mask to front-facing pairs only
        front_mask = pair_front.astype(cp.bool_)
        front_vp = vp_idx_i32[front_mask]
        front_dist = pair_dist[front_mask]

        if len(front_vp) > 0:
            if self.hp.gamma_method == "mean":
                # Fully vectorised: scatter_add for sum + count
                sum_per_vp = cp.zeros(N, dtype=cp.float32)
                cp.scatter_add(sum_per_vp, front_vp, front_dist)
                count_per_vp = cp.zeros(N, dtype=cp.int32)
                cp.scatter_add(count_per_vp, front_vp,
                               cp.ones(len(front_vp), dtype=cp.int32))
                has_front = count_per_vp > 0
                gamma_per_vp[has_front] = (
                    sum_per_vp[has_front]
                    / count_per_vp[has_front].astype(cp.float32))
            else:
                # Per-segment GPU slicing for percentile/median
                gamma_func = GAMMA_AGG_FUNCS_CP[self.hp.gamma_method]
                unique_front_vps, front_first, front_counts = cp.unique(
                    front_vp, return_index=True, return_counts=True)
                for k in range(len(unique_front_vps)):
                    vp = unique_front_vps[k]
                    s = front_first[k]
                    e = s + front_counts[k]
                    gamma_per_vp[vp] = cp.float32(
                        max(float(gamma_func(front_dist[s:e])), 1e-6))

        # Vectorised epsilon: 2 * arctan(delta / (4 * gamma)) * scale
        gamma_per_vp = cp.maximum(gamma_per_vp, cp.float32(1e-6))
        epsilon_per_vp = (
            2.0 * cp.arctan(cp.float32(self.delta) / (4.0 * gamma_per_vp))
            * cp.float32(self.hp.epsilon_scale))

        return epsilon_per_vp
