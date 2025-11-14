from utils import * # type: ignore # Assuming utils is available
from epsilon_visibility import EpsilonVisibilityQuery # type: ignore # Assuming EpsilonQuery is available

class GreedyOptimizer:
    """
    Implements a greedy set-cover optimization that leverages the 
    kernel expansion technique (from Lien, 2007) for better viewpoint selection.
    
    This optimizer requires the EpsilonVisibilityQuery to utilize its 
    kernel-related methods, but can work with any VisibilityQuery for scoring.
    """
    def __init__(self, visibility_query: VisibilityQuery):
        self.query = visibility_query
        self.num_points = visibility_query.num_points
        self.candidate_kdtree = None # For fast neighborhood checks (optional)
        print(f"[GreedyOptimizer] Initialized with {type(visibility_query).__name__}.")
        
        if not isinstance(visibility_query, EpsilonVisibilityQuery):
            print("[GreedyOptimizer] Warning: Kernel expansion requires EpsilonVisibilityQuery for _compute_kernel.")
            self._compute_kernel = self._stub_compute_kernel
            
    def _stub_compute_kernel(self, viewpoint, visible_indices) -> np.ndarray:
        """Fallback for _compute_kernel if not using EpsilonVisibilityQuery."""
        print("  [GreedyOptimizer] Using stub kernel computation: returning only the current viewpoint.")
        return np.array([viewpoint])

    def expand_kernel(self, viewpoint: np.ndarray, visible_indices: np.ndarray, 
                      max_iterations: int = 3, recursion_depth: int = 0, c_factor: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        A* expansion to find a better viewpoint (guard) in the kernel (Algorithm 5.3).
        """
        if recursion_depth >= max_iterations:
            return viewpoint, visible_indices
            
        if len(visible_indices) < 4:
            return viewpoint, visible_indices
        
        visible_points = self.query.target_points[visible_indices]
        visible_normals = self.query.normals[visible_indices]
        
        # Compute kernel: This calls the method from EpsilonVisibilityQuery if available
        # If not, it uses the stub from __init__
        # kernel_points = self.query._compute_kernel(viewpoint, visible_points, visible_normals) # type: ignore
        kernel_points = list()

        if len(kernel_points) == 0:
            return viewpoint, visible_indices
        
        num_test = min(max(1, int(np.log(len(kernel_points)))), len(kernel_points), 10)
        test_indices = np.random.choice(len(kernel_points), num_test, replace=False)
        
        best_vp = viewpoint
        best_visible = visible_indices
        
        for idx in test_indices:
            test_vp = kernel_points[idx]
            
            # Simple direction heuristic: average vector to visible points
            direction = np.mean(visible_points - test_vp, axis=0)
            direction = normalize_vector(direction)
            
            test_visible, _ = self.query.compute_visibility(test_vp, direction)
            
            if len(test_visible) > len(best_visible):
                best_vp = test_vp
                best_visible = test_visible
        
        # Recursion Check
        improvement_threshold = self.num_points / c_factor
        
        if len(best_visible) > len(visible_indices) + improvement_threshold:
            return self.expand_kernel(best_vp, best_visible, max_iterations, 
                                      recursion_depth + 1, c_factor)
        else:
            return best_vp, best_visible

    def optimize(self, candidates: List[Tuple[np.ndarray, np.ndarray]], target_coverage: float = 0.95, max_viewpoints: int = 50) -> OptimizationResult:
        """
        Greedy set-cover optimization using pre-computed visibility and kernel expansion.
        """
        print(f"\n[GreedyOptimizer] Starting optimization with {len(candidates)} candidates.")
        start_time = get_time()

        if self.num_points == 0 or not candidates:
            return OptimizationResult("Greedy_Empty", [], 0.0, 0, 0.0, [], 0.0)

        # 1. Pre-compute the *initial* visibility for all candidates
        initial_visibility_map = self.query.compute_visibility_for_all_candidates(candidates)
            
        uncovered = set(range(self.num_points))
        selected_viewpoints: List[ViewpointResult] = []
        candidate_list = list(initial_visibility_map.keys())
        remaining_candidate_indices = set(range(len(candidate_list)))
        target_uncovered_count = int((1.0 - target_coverage) * self.num_points)
        
        while len(uncovered) > target_uncovered_count and len(selected_viewpoints) < max_viewpoints:
            
            if not remaining_candidate_indices:
                print("  [GreedyOptimizer] No more candidates to check. Breaking.")
                break

            best_candidate_idx = -1
            best_expanded_set = set()
            best_score = 0
            
            current_loop_start = get_time()
            
            # Find the best remaining candidate
            for candidate_idx in remaining_candidate_indices:
                vp_tuple, dir_tuple = candidate_list[candidate_idx]
                vp = np.array(vp_tuple)
                
                # Get the pre-computed initial visibility
                visible_indices = initial_visibility_map[(vp_tuple, dir_tuple)]
                
                # TODO: needs to be fixed (should expand the best found point, not expanding all the points)
                # Run kernel expansion
                # expanded_vp, expanded_visible_indices = self.expand_kernel(vp, visible_indices)

                # print(f"Expanded to: pos={expanded_vp}")
                
                newly_covered = set(visible_indices) & uncovered
                score = len(newly_covered)
                
                if score > best_score:
                    best_score = score
                    best_candidate_idx = candidate_idx
                    best_expanded_set = newly_covered
            
            loop_time = get_time() - current_loop_start
            print(f"  [GreedyOptimizer] Loop {len(selected_viewpoints) + 1} took {loop_time:.2f}s.")
            
            if best_candidate_idx == -1 or best_score == 0:
                print("  [GreedyOptimizer] No candidate provides new coverage. Stopping.")
                break
                
            # Final Viewpoint Selection (use the expanded VP position but the original direction)
            best_vp_tuple, best_dir_tuple = candidate_list[best_candidate_idx]
            
            uncovered -= best_expanded_set
            coverage = 1.0 - len(uncovered) / self.num_points
            
            selected_viewpoints.append(ViewpointResult(
                position=candidate_list[best_candidate_idx][0], # Use the optimized kernel position
                direction=np.array(best_dir_tuple),
                visible_indices=np.array(list(best_expanded_set)), 
                coverage_score=best_score / self.num_points,
                computation_time=loop_time 
            ))
            
            remaining_candidate_indices.remove(best_candidate_idx)
            
            print(f"  [GreedyOptimizer] Selected VP {len(selected_viewpoints)}: +{best_score} points, Total coverage={coverage*100:.1f}%")

        total_time = get_time() - start_time
        coverage = 1.0 - len(uncovered) / self.num_points
        redundancy = self.query.compute_redundancy(selected_viewpoints)
        
        return OptimizationResult(
            method_name="Greedy_Kernel_Expansion",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=redundancy
        )

print("[greedy_optimizer.py] Greedy Optimizer module defined.")