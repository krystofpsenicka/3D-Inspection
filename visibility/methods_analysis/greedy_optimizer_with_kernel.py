from utils import * # type: ignore # Assuming utils is available
from epsilon_visibility import EpsilonVisibilityQuery # type: ignore # Assuming EpsilonQuery is available

class GreedyOptimizer:
    """
    Implements a standard greedy set-cover optimization.
    
    This optimizer can work with any VisibilityQuery. It will be used for the
    ground-truth Raycasting method, and the Epsilon method will use its own
    `optimize_with_candidates` which integrates kernel expansion.
    """
    def __init__(self, visibility_query: VisibilityQuery):
        self.query = visibility_query
        self.num_points = visibility_query.num_points
        print(f"[GreedyOptimizer] Initialized with {type(visibility_query).__name__}.")
        
    # Removed expand_kernel and _stub_compute_kernel as they are now fully handled 
    # by EpsilonVisibilityQuery.optimize_with_candidates

    def optimize(self, candidates: List[Tuple[np.ndarray, np.ndarray]], target_coverage: float = 0.95, max_viewpoints: int = 50) -> OptimizationResult:
        """
        Standard greedy set-cover optimization.
        """
        print(f"\n[GreedyOptimizer] Starting standard greedy optimization with {len(candidates)} candidates.")
        start_time = get_time()

        if self.num_points == 0 or not candidates:
            return OptimizationResult("Greedy_Empty", [], 0.0, 0, 0.0, [], 0.0, 0.0, 0.0) # Added time fields

        # 1. Pre-compute the *initial* visibility for all candidates
        # This will return a map and the total time spent in visibility computation
        initial_visibility_map, vis_comp_time = self.query.compute_visibility_for_all_candidates(candidates)
            
        uncovered = set(range(self.num_points))
        selected_viewpoints: List[ViewpointResult] = []
        candidate_list = list(initial_visibility_map.keys())
        remaining_candidate_indices = set(range(len(candidate_list)))
        target_uncovered_count = int((1.0 - target_coverage) * self.num_points)
        
        optimization_start_time = get_time()
        
        while len(uncovered) > target_uncovered_count and len(selected_viewpoints) < max_viewpoints:
            
            if not remaining_candidate_indices:
                print("  [GreedyOptimizer] No more candidates to check. Breaking.")
                break

            best_candidate_idx = -1
            best_newly_covered = set()
            best_score = 0
            
            # Find the best remaining candidate
            for candidate_idx in remaining_candidate_indices:
                vp_tuple, dir_tuple = candidate_list[candidate_idx]
                
                # Get the pre-computed initial visibility
                visible_indices = initial_visibility_map[(vp_tuple, dir_tuple)]
                
                # Score is the number of NEWLY covered points
                newly_covered = set(visible_indices) & uncovered
                score = len(newly_covered)
                
                if score > best_score:
                    best_score = score
                    best_candidate_idx = candidate_idx
                    best_newly_covered = newly_covered
            
            if best_candidate_idx == -1 or best_score == 0:
                print("  [GreedyOptimizer] No candidate provides new coverage. Stopping.")
                break
                
            # Select and update
            best_vp_tuple, best_dir_tuple = candidate_list[best_candidate_idx]
            
            uncovered -= best_newly_covered
            coverage = 1.0 - len(uncovered) / self.num_points
            
            selected_viewpoints.append(ViewpointResult(
                position=np.array(best_vp_tuple),
                direction=np.array(best_dir_tuple),
                visible_indices=np.array(list(best_newly_covered)), 
                coverage_score=best_score / self.num_points,
                computation_time=0.0 # Time per-VP is hard to track accurately here
            ))
            
            remaining_candidate_indices.remove(best_candidate_idx)
            
            print(f"  [GreedyOptimizer] Selected VP {len(selected_viewpoints)}: +{best_score} points, Total coverage={coverage*100:.1f}%")

        optimization_time = get_time() - optimization_start_time
        total_time = get_time() - start_time
        coverage = 1.0 - len(uncovered) / self.num_points
        redundancy = self.query.compute_redundancy(selected_viewpoints)
        
        return OptimizationResult(
            method_name="Greedy_Standard",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=redundancy,
            visibility_computation_time=vis_comp_time,
            optimization_time=optimization_time
        )

print("[greedy_optimizer.py] Greedy Optimizer module defined.")