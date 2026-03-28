import logging
import numpy as np
import cupy as cp
from time import time as get_time
from typing import List

from ..core.types import ViewpointResult, OptimizationResult
from ..core.base import VisibilityQueryBase
from ..core.constants import NORM_EPS, KERNEL_N_SAMPLES, KERNEL_RADIUS
from shared.geometry import direction_roll_to_rotmat

logger = logging.getLogger(__name__)


class KernelGreedyOptimizerCuda:
    """GPU-accelerated greedy set-cover optimizer with random-sphere expansion.

    Pre-computes initial visibility for all candidates, applies _expand_gpu()
    to each, then runs a GPU greedy loop on the expanded visibility matrix.
    """

    def __init__(self, visibility_query: VisibilityQueryBase):
        self.query = visibility_query
        self.num_points = visibility_query.num_points
        logger.info("[KernelGreedyOptimizerCuda] Initialized with %s.", type(visibility_query).__name__)

    def _expand_gpu(self, vp, vis_indices, n_samples=KERNEL_N_SAMPLES, radius=KERNEL_RADIUS):
        """GPU-accelerated expansion: sample n_samples positions around vp, pick best.

        Args:
            vp:          numpy (3,) position.
            vis_indices: CuPy int64 array of visible point indices.
        """
        if len(vis_indices) == 0:
            return vp, vis_indices

        # Generate random offsets on GPU
        offsets = cp.random.randn(n_samples, 3).astype(cp.float32)
        norms = cp.linalg.norm(offsets, axis=1, keepdims=True)
        offsets /= (norms + NORM_EPS)
        scales = cp.random.uniform(0, radius, (n_samples, 1)).astype(cp.float32)
        offsets *= scales
        vp_gpu = cp.asarray(vp, dtype=cp.float32)
        samples_gpu = vp_gpu + offsets          # (n_samples, 3) on GPU

        # Compute centroid of currently visible points on GPU
        visible_centroid_gpu = cp.mean(
            self.query.gpu_points[vis_indices], axis=0).astype(cp.float32)

        best_vp = vp
        best_vis = vis_indices

        for i in range(n_samples):
            sample = samples_gpu[i]
            direction = visible_centroid_gpu - sample
            direction /= (cp.linalg.norm(direction) + NORM_EPS)
            rotmat = direction_roll_to_rotmat(direction.get())
            rotmat_gpu = cp.asarray(rotmat, dtype=cp.float32)
            vis, _ = self.query.compute_visibility(sample, rotmat_gpu)
            if len(vis) > len(best_vis):
                best_vp = sample.get()
                best_vis = vis

        return best_vp, best_vis

    def optimize(self, positions, rotmats,
                 target_coverage: float = 0.95,
                 max_viewpoints: int = 50) -> OptimizationResult:
        """GPU greedy set-cover with random-sphere expansion.

        Parameters
        ----------
        positions : array (N, 3)
            Candidate positions (numpy or CuPy).
        rotmats : array (N, 3, 3)
            Candidate rotation matrices (numpy or CuPy).
        """
        n_cand = len(positions)
        logger.info("[KernelGreedyOptimizerCuda] Starting optimization with %d candidates.", n_cand)
        start_time = get_time()

        if self.num_points == 0 or n_cand == 0:
            return OptimizationResult("KernelGreedyCuda_Empty", [], 0.0, 0, 0.0, [], 0.0, 0.0, 0.0)

        # 1. Pre-compute initial visibility — (N, M) uint8 GPU matrix
        V, vis_comp_time = self.query.compute_visibility_batch(positions, rotmats)

        # 2. Apply _expand_gpu to each candidate
        logger.info("[KernelGreedyOptimizerCuda] Expanding kernels for %d candidates...", n_cand)
        expanded_entries = []  # list of (expanded_vp, rotmat, expanded_vis_indices)
        for i in range(n_cand):
            vp = np.asarray(positions[i]) if not hasattr(positions[i], 'get') else positions[i].get()
            rot_i = rotmats[i]
            if hasattr(rot_i, 'get'):
                rot_i = rot_i.get()
            initial_vis = cp.where(V[i])[0]  # CuPy indices from matrix row
            exp_vp, exp_vis = self._expand_gpu(vp, initial_vis)
            expanded_entries.append((exp_vp, rot_i, exp_vis))

        # 3. Rebuild dense GPU visibility matrix from expanded sets
        n = len(expanded_entries)
        V = cp.zeros((n, self.num_points), dtype=cp.uint8)
        for i, (_, _, vis_idx) in enumerate(expanded_entries):
            if len(vis_idx) > 0:
                V[i, vis_idx] = 1

        logger.info("[KernelGreedyOptimizerCuda] Visibility matrix on GPU: %s (%.1f MB)",
                    V.shape, V.nbytes / 1e6)

        # 4. GPU greedy loop
        uncovered = cp.ones(self.num_points, dtype=cp.uint8)
        candidate_mask = cp.ones(n, dtype=cp.bool_)
        total_covered = 0
        target_covered = int(target_coverage * self.num_points)

        selected_viewpoints: List[ViewpointResult] = []
        optimization_start_time = get_time()

        while total_covered < target_covered and len(selected_viewpoints) < max_viewpoints:
            if not cp.any(candidate_mask):
                logger.info("  [KernelGreedyOptimizerCuda] No more candidates. Breaking.")
                break

            scores = V.astype(cp.float32) @ uncovered.astype(cp.float32)
            scores[~candidate_mask] = 0

            best = int(cp.argmax(scores))
            best_score = int(scores[best])

            if best_score == 0:
                logger.info("  [KernelGreedyOptimizerCuda] No candidate provides new coverage. Stopping.")
                break

            newly_covered_mask = V[best] & uncovered
            uncovered &= ~V[best]
            candidate_mask[best] = False

            total_covered = self.num_points - int(cp.sum(uncovered))
            coverage = total_covered / self.num_points

            exp_vp, best_rot, _ = expanded_entries[best]

            # Store the *full* visibility set, not just the incremental contribution
            full_visible_indices = cp.where(V[best])[0].get()
            selected_viewpoints.append(ViewpointResult(
                position=np.asarray(exp_vp),
                orientation=np.asarray(best_rot),
                visible_indices=full_visible_indices,
                coverage_score=len(full_visible_indices) / self.num_points,
                computation_time=0.0,
            ))

            logger.info("  [KernelGreedyOptimizerCuda] Selected VP %d: +%d points, Total coverage=%.1f%%",
                        len(selected_viewpoints), best_score, coverage * 100)

        optimization_time = get_time() - optimization_start_time
        total_time = get_time() - start_time
        coverage = total_covered / self.num_points
        redundancy = self.query.compute_redundancy(selected_viewpoints)

        return OptimizationResult(
            method_name="KernelGreedyCuda",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=redundancy,
            visibility_computation_time=vis_comp_time,
            optimization_time=optimization_time,
            visibility_map=None,
        )
