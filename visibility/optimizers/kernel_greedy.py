import numpy as np
from time import time as get_time
from typing import List, Tuple

from ..core.types import ViewpointResult, OptimizationResult
from ..core.base import VisibilityQuery


class KernelGreedyOptimizer:
    """
    Greedy set-cover optimizer with random-sphere expansion.

    Same greedy loop as GreedyOptimizer, but after getting precomputed
    visibility for each candidate, calls _expand() to sample nearby positions
    and pick the one with the most visible points before scoring.
    """

    def __init__(self, visibility_query: VisibilityQuery):
        self.query = visibility_query
        self.num_points = visibility_query.num_points
        print(f"[KernelGreedyOptimizer] Initialized with {type(visibility_query).__name__}.")

    def _expand(self, vp, vis_indices, n_samples=20, radius=0.5):
        """Sample n_samples random positions in a sphere around vp, return the best."""
        if len(vis_indices) == 0:
            return vp, vis_indices

        visible_centroid = np.mean(self.query.target_points[vis_indices], axis=0)

        best_vp = vp
        best_vis = vis_indices

        offsets = np.random.randn(n_samples, 3)
        offsets /= (np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-12)
        offsets *= np.random.uniform(0, radius, (n_samples, 1))
        samples = vp + offsets

        for sample in samples:
            direction = visible_centroid - sample
            direction /= (np.linalg.norm(direction) + 1e-12)
            vis, _ = self.query.compute_visibility(sample, direction)
            if len(vis) > len(best_vis):
                best_vp = sample
                best_vis = vis

        return best_vp, best_vis

    def optimize(self, candidates: List[Tuple[np.ndarray, np.ndarray]],
                 target_coverage: float = 0.95,
                 max_viewpoints: int = 50) -> OptimizationResult:
        """Greedy set-cover with kernel expansion."""
        print(f"\n[KernelGreedyOptimizer] Starting optimization with {len(candidates)} candidates.")
        start_time = get_time()

        if self.num_points == 0 or not candidates:
            return OptimizationResult("KernelGreedy_Empty", [], 0.0, 0, 0.0, [], 0.0, 0.0, 0.0)

        initial_visibility_map, vis_comp_time = self.query.compute_visibility_for_all_candidates(candidates)

        uncovered = set(range(self.num_points))
        selected_viewpoints: List[ViewpointResult] = []
        candidate_list = list(initial_visibility_map.keys())
        remaining_candidate_indices = set(range(len(candidate_list)))
        target_uncovered_count = int((1.0 - target_coverage) * self.num_points)

        optimization_start_time = get_time()

        while len(uncovered) > target_uncovered_count and len(selected_viewpoints) < max_viewpoints:

            if not remaining_candidate_indices:
                print("  [KernelGreedyOptimizer] No more candidates to check.")
                break

            best_candidate_idx = -1
            best_expanded_set = set()
            best_score = 0

            for candidate_idx in remaining_candidate_indices:
                vp_tuple, dir_tuple = candidate_list[candidate_idx]
                vp = np.array(vp_tuple)

                visible_indices = initial_visibility_map[(vp_tuple, dir_tuple)]

                # Kernel expansion
                expanded_vp, expanded_visible = self._expand(vp, visible_indices)

                newly_covered = set(expanded_visible) & uncovered
                score = len(newly_covered)

                if score > best_score:
                    best_score = score
                    best_candidate_idx = candidate_idx
                    best_expanded_set = newly_covered

            if best_candidate_idx == -1 or best_score == 0:
                print("  [KernelGreedyOptimizer] No candidate provides new coverage. Stopping.")
                break

            best_vp_tuple, best_dir_tuple = candidate_list[best_candidate_idx]

            uncovered -= best_expanded_set
            coverage = 1.0 - len(uncovered) / self.num_points

            selected_viewpoints.append(ViewpointResult(
                position=np.array(best_vp_tuple),
                direction=np.array(best_dir_tuple),
                visible_indices=np.array(list(best_expanded_set)),
                coverage_score=best_score / self.num_points,
                computation_time=0.0,
            ))

            remaining_candidate_indices.remove(best_candidate_idx)

            print(f"  [KernelGreedyOptimizer] VP {len(selected_viewpoints)} "
                  f"(Candidate {best_candidate_idx}): +{best_score} points, "
                  f"coverage={coverage * 100:.1f}%")

        optimization_time = get_time() - optimization_start_time
        total_time = get_time() - start_time
        coverage = 1.0 - len(uncovered) / self.num_points
        redundancy = self.query.compute_redundancy(selected_viewpoints)

        return OptimizationResult(
            method_name="KernelGreedy",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=redundancy,
            visibility_computation_time=vis_comp_time,
            optimization_time=optimization_time,
        )
