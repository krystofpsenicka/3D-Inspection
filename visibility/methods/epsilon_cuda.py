import numpy as np
import cupy as cp
import open3d as o3d
from numpy.linalg import norm
from time import time as get_time
from typing import Optional

from ..core.types import FrustumParams, EpsilonHyperparams
from ..core.base_cuda import VisibilityQueryCuda

_DELTA_AGG_FUNCS = {
    "max": np.max,
    "p99": lambda x: np.percentile(x, 99),
    "p95": lambda x: np.percentile(x, 95),
    "p90": lambda x: np.percentile(x, 90),
}

_GAMMA_AGG_FUNCS_CP = {
    "median": cp.median,
    "mean": cp.mean,
    "p10": lambda x: float(cp.percentile(x, 10)),
    "p25": lambda x: float(cp.percentile(x, 25)),
    "p30": lambda x: float(cp.percentile(x, 30)),
    "p40": lambda x: float(cp.percentile(x, 40)),
    "p60": lambda x: float(cp.percentile(x, 60)),
    "p75": lambda x: float(cp.percentile(x, 75)),
    "p90": lambda x: float(cp.percentile(x, 90)),
}


# CUDA kernel for scatter-min of occluder distances into angular bins.
# Uses the int-reinterpret trick: positive floats preserve ordering when
# viewed as unsigned ints, so atomicMin on the int representation gives
# the correct minimum float value.
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
    // Reinterpret float bits as int for atomicMin (works for positive floats)
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

    // Visible if closer than or equal to occluder (occluder_dist == INF means no occluder)
    visible[idx] = (my_dist <= occluder_dist) ? 1 : 0;
}
''', 'check_visibility')


class EpsilonVisibilityQueryCuda(VisibilityQueryCuda):
    """GPU-accelerated epsilon-visibility using CuPy and custom CUDA kernels.

    The main speedup is in _check_occlusion_gpu which replaces the Python
    for-loops with scatter-min and parallel visibility kernels.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams,
                 epsilon_deg: Optional[float] = None,
                 frustum_method: str = "bruteforce",
                 hyperparams: Optional[EpsilonHyperparams] = None):
        super().__init__(mesh, target_points, normals, frustum_params,
                         frustum_method=frustum_method)
        self.hp = hyperparams or EpsilonHyperparams()

        if epsilon_deg is not None:
            print(f"Using provided epsilon: {epsilon_deg} degrees")
            self.fixed_epsilon = np.deg2rad(epsilon_deg)
            self.delta = None
            print(f"Using Epsilon (radians): {self.fixed_epsilon:.6f} "
                  f"({np.rad2deg(self.fixed_epsilon):.3f} degrees)")
        else:
            print("Epsilon not provided, estimating δ from point set...")
            self.fixed_epsilon = None
            self.delta = self._estimate_delta()
            print(f"Estimated δ (sampling density): {self.delta:.6f}")

    def _estimate_delta(self):
        """Estimate sampling density δ (Lemma 6.1): aggregation of k-neighbor distances."""
        sample_size = min(1000, self.num_points)
        if sample_size == 0:
            return 0.1

        sample_indices = np.random.choice(self.num_points, sample_size, replace=False)
        agg_func = _DELTA_AGG_FUNCS[self.hp.delta_agg]

        distances = []
        for idx in sample_indices:
            dists, _ = self.kdtree.query(self.target_points[idx], k=self.hp.delta_k)
            if len(dists) > 1:
                distances.append(agg_func(dists[1:]))

        return agg_func(distances) if distances else 0.1

    def _compute_epsilon_gpu(self, distances_gpu, front_facing_gpu):
        """Compute per-viewpoint ε = 2·arctan(δ/(4γ)) where γ = aggregated viewing distance."""
        front_distances = distances_gpu[front_facing_gpu]
        gamma_func = _GAMMA_AGG_FUNCS_CP[self.hp.gamma_method]
        if len(front_distances) == 0:
            gamma = self.frustum_params.far / 4.0
        else:
            gamma = float(gamma_func(front_distances))
        gamma = max(gamma, 1e-6)
        return 2.0 * np.arctan(self.delta / (4.0 * gamma)) * self.hp.epsilon_scale

    def compute_visibility(self, viewpoint, direction):
        """Compute epsilon-visible region using GPU back-face and occlusion checks."""
        start = get_time()

        frustum_indices = self.points_in_frustum_gpu(viewpoint, direction)

        if len(frustum_indices) == 0:
            return np.array([]), get_time() - start

        # GPU back-face check
        frustum_indices_gpu = cp.asarray(frustum_indices)
        frustum_points = self.gpu_points[frustum_indices_gpu]
        frustum_normals = self.gpu_normals[frustum_indices_gpu]

        vp_gpu = cp.asarray(viewpoint, dtype=cp.float32)
        view_dirs = frustum_points - vp_gpu
        view_dirs_norm = cp.linalg.norm(view_dirs, axis=1)
        view_dirs = view_dirs / (view_dirs_norm[:, cp.newaxis] + 1e-12)

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

        visible_local = cp.where(visible_mask)[0].get()
        visible_indices = frustum_indices[visible_local]

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

        # Spherical binning
        num_bins_theta = max(1, int(np.ceil(2 * np.pi / epsilon)))
        num_bins_phi = max(1, int(np.ceil(np.pi / epsilon)))

        theta = cp.arctan2(relative[:, 1], relative[:, 0])
        phi = cp.arcsin(cp.clip(relative[:, 2] / (distances + 1e-12), -1, 1))

        theta_bins = ((theta + np.pi) / (2 * np.pi) * num_bins_theta).astype(cp.int32) % num_bins_theta
        phi_bins = ((phi + np.pi / 2) / np.pi * num_bins_phi).astype(cp.int32) % num_bins_phi

        num_bins = num_bins_theta * num_bins_phi

        bin_keys_all = (theta_bins * num_bins_phi + phi_bins).astype(cp.int32)

        # --- Scatter-min for occluders (non-front-facing points) ---
        occluder_mask = ~front_facing
        occluder_indices = cp.where(occluder_mask)[0]

        # Initialize bin distances to INT_MAX (= +INF as float)
        bin_min_dist = cp.full(num_bins, 0x7F7FFFFF, dtype=cp.int32)  # max positive float as int

        if len(occluder_indices) > 0:
            occluder_bin_keys = bin_keys_all[occluder_indices].astype(cp.int32)
            occluder_distances = distances[occluder_indices].astype(cp.float32)

            M = len(occluder_indices)
            block_size = 256
            grid_size = (M + block_size - 1) // block_size
            _SCATTER_MIN_KERNEL(
                (grid_size,), (block_size,),
                (occluder_bin_keys, occluder_distances, bin_min_dist, np.int32(M))
            )

        # --- Parallel visibility check for front-facing points ---
        front_indices = cp.where(front_facing)[0]

        if len(front_indices) > 0:
            front_bin_keys = bin_keys_all[front_indices].astype(cp.int32)
            front_distances = distances[front_indices].astype(cp.float32)

            K = len(front_indices)
            vis_result = cp.zeros(K, dtype=cp.int32)

            block_size = 256
            grid_size = (K + block_size - 1) // block_size
            _VISIBILITY_CHECK_KERNEL(
                (grid_size,), (block_size,),
                (front_bin_keys, front_distances, bin_min_dist, vis_result, np.int32(K))
            )

            # Update visible mask: only front-facing points that pass the check
            visible[front_indices] = vis_result.astype(cp.bool_)

        return visible
