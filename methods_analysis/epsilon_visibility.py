from utils import * # type: ignore # Assuming utils is available
import math

class EpsilonVisibilityQuery(VisibilityQuery):
    """
    ε-visibility based on Lien's paper.
    
    This is the FIXED version, correctly implementing:
    1. Epsilon estimation and usage.
    2. Epsilon-based occlusion (Algorithm 5.1).
    3. Correct kernel computation (Lemma 5.2).
    4. Recursive kernel expansion (Algorithm 5.3).
    """
    
    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray, normals: np.ndarray, frustum_params: FrustumParams, epsilon_deg: Optional[float] = None):
        super().__init__(mesh, target_points, normals, frustum_params)

        # Compute normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        # Optional: Orient normals for consistency
        pcd.orient_normals_consistent_tangent_plane(k=10)

        self.normals = np.asarray(pcd.normals)
        self.pcd = pcd # Store the PointCloud
        
        # self._visualize_normals(normal_scale=0.05)

        if epsilon_deg is None:
            print("Epsilon not provided, estimating from point set...")
            self.epsilon = self._estimate_epsilon()
        else:
            print(f"Using provided epsilon: {epsilon_deg} degrees")
            self.epsilon = np.deg2rad(epsilon_deg)
            
        print(f"Using Epsilon (radians): {self.epsilon:.6f} "
              f"({np.rad2deg(self.epsilon):.3f} degrees)")

    def _visualize_normals(self, normal_scale=0.05):
        """Visualizes the mesh, the target points, and their computed normals."""
        # This function seems correct and is untouched.
        print("\nVisualizing Normals")
        
        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
        mesh_vis.compute_vertex_normals()
        
        pcd_vis = o3d.geometry.PointCloud(self.pcd)
        pcd_vis.paint_uniform_color([1.0, 0.0, 0.0])
        
        points = np.asarray(pcd_vis.points)
        normals = np.asarray(pcd_vis.normals)
        
        normal_endpoints = points + (normals * normal_scale)
        normal_vertices = np.concatenate((points, normal_endpoints), axis=0)
        
        indices = np.arange(len(points))
        normal_lines_indices = np.vstack((indices, indices + len(points))).T
        
        normal_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(normal_vertices),
            lines=o3d.utility.Vector2iVector(normal_lines_indices)
        )
        normal_lines.colors = o3d.utility.Vector3dVector(
            [[0, 0, 0] for _ in range(len(normal_lines_indices))]
        )
        
        geometries = [mesh_vis, pcd_vis, normal_lines]
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Approximate Star-Shaped Initialization", 
                          width=1600, height=900)
        
        render_option = vis.get_render_option()
        render_option.point_size = 4.0 
        render_option.mesh_show_back_face = True
        
        for geom in geometries:
            vis.add_geometry(geom)

        vis.run()
        vis.destroy_window()

    def _estimate_epsilon(self):
        """Estimate epsilon based on sampling density (Lemma 6.1 from paper)."""
        # Sample distances to nearest neighbors
        sample_size = min(1000, self.num_points)
        if sample_size == 0:
            return np.deg2rad(1.0) # Default to 1 degree if no points

        sample_indices = np.random.choice(self.num_points, sample_size, replace=False)
        
        distances = []
        for idx in sample_indices:
            # Query for 5 neighbors, take max distance (excluding self)
            dists, _ = self.kdtree.query(self.target_points[idx], k=5)
            if len(dists) > 1:
                distances.append(np.max(dists[1:]))
        
        if not distances:
             delta = 0.1 # Fallback delta
        else:
            delta = np.max(distances)  # Sampling density
        
        # Estimate gamma (distance to medial axis) as half max visibility distance
        gamma = self.frustum_params.far / 4.0
        if gamma < 1e-6:
            gamma = 10.0 # Fallback
        
        # Compute epsilon per Lemma 6.1: ε = 2 * arctan(δ / (4γ))
        epsilon = 2 * np.arctan(delta / (4 * gamma))
        
        # *** FIXED: Return radians, not degrees ***
        return epsilon
    
    def compute_visibility(self, viewpoint, direction):
        """
        Compute ε-visible region using back-face and occlusion checks.
        """
        start = get_time()
        
        # Frustrum culling
        frustum_indices = self.points_in_frustum_with_kdtree(viewpoint, direction)
        
        if len(frustum_indices) == 0:
            print("[EpsilonQuery] Frustrum empty.")
            return np.array([]), get_time() - start
        
        frustum_points = self.target_points[frustum_indices]
        frustum_normals = self.normals[frustum_indices]
        
        # Check back-face visibility
        view_dirs = frustum_points - viewpoint
        view_dirs_norm = norm(view_dirs, axis=1)
        # Add epsilon to avoid division by zero
        view_dirs = view_dirs / (view_dirs_norm[:, np.newaxis] + 1e-12)
        
        # Point is front-facing if normal points towards viewpoint
        # (p - x) . n_p > 0
        dot_products = np.sum(view_dirs * frustum_normals, axis=1)
        
        # *** FIXED: Paper Def 3.4 is n_p . xp < 0 for back-face. ***
        # xp = p - x. So (p - x) . n_p > 0 is front-facing.
        front_facing = dot_products > 1e-6 # Use small epsilon for numerical stability
        
        # *** FIXED: Pass self.epsilon to occlusion check ***
        visible_mask = self._check_occlusion(
            viewpoint, frustum_points, frustum_normals, front_facing, self.epsilon
        )
        
        visible_indices = frustum_indices[visible_mask]
        
        comp_time = get_time() - start
        print(f"  [EpsilonQuery] Visibility query complete. {len(visible_indices)} visible points in {comp_time:.4f}s.")
        return visible_indices, comp_time
    
    def _check_occlusion(self, viewpoint: np.ndarray, points: np.ndarray, normals: np.ndarray, front_facing: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Check occlusion using radial partitioning (Algorithm 5.1).
        """
        
        num_points = len(points)
        if num_points == 0:
            return np.array([], dtype=bool)

        # Initialize all points as visible (respecting front-facing)
        visible = front_facing.copy()
        
        if not np.any(front_facing):
            return visible # No front-facing points, nothing is visible

        # --- Algorithm 5.1: Radial Partitioning ---
        
        # 1. Calculate number of bins based on epsilon
        if epsilon < 1e-6:
            epsilon = 1e-6 # Avoid division by zero
            
        # Discretize spherical coordinates based on epsilon
        num_bins_theta = max(1, int(np.ceil(2 * np.pi / epsilon)))
        num_bins_phi = max(1, int(np.ceil(np.pi / epsilon)))
        
        # 2. Convert to spherical coordinates relative to viewpoint
        relative = points - viewpoint
        distances = norm(relative, axis=1)
        
        # (theta, phi)
        theta = np.arctan2(relative[:, 1], relative[:, 0]) # Azimuthal, [-pi, pi]
        phi = np.arcsin(np.clip(relative[:, 2] / (distances + 1e-12), -1, 1)) # Polar, [-pi/2, pi/2]
        
        # 3. Assign points to bins
        theta_bins = ((theta + np.pi) / (2 * np.pi) * num_bins_theta).astype(int) % num_bins_theta
        phi_bins = ((phi + np.pi/2) / np.pi * num_bins_phi).astype(int) % num_bins_phi
        
        # 4. Find closest back point (occluder) in each bin
        occluder_bins = {} # Stores {bin_key: min_occluder_distance}
        
        for i in range(num_points):
            # If this point is back-facing, it's a potential occluder
            if not front_facing[i]:
                bin_key = (theta_bins[i], phi_bins[i])
                dist = distances[i]
                
                if bin_key not in occluder_bins or dist < occluder_bins[bin_key]:
                    occluder_bins[bin_key] = dist
                    
        # 5. Check occlusion for each front-facing point
        for i in range(num_points):
            # Only front-facing points can be occluded
            if front_facing[i]:
                bin_key = (theta_bins[i], phi_bins[i])
                
                # If there is an occluder in this bin...
                if bin_key in occluder_bins:
                    occluder_dist = occluder_bins[bin_key]
                    
                    # ...and this point is farther away than the occluder...
                    if distances[i] > occluder_dist:
                        visible[i] = False # ...then it is occluded.
        
        return visible
    
    def expand_kernel(self, viewpoint, visible_indices, 
                      max_iterations=3, recursion_depth=0, c_factor=1000):
        """
        A* expansion to find better guard in kernel (Algorithm 5.3: A*-EXPAND2).
        
        This is the FIXED recursive implementation.
        """
        
        # --- Base Cases for Recursion ---
        if recursion_depth >= max_iterations:
            print(f"  expand_kernel: Max recursion depth ({max_iterations}) reached.")
            return viewpoint, visible_indices
            
        if len(visible_indices) < 4:
            print("  expand_kernel: Too few points to compute kernel.")
            return viewpoint, visible_indices
        
        visible_points = self.target_points[visible_indices]
        visible_normals = self.normals[visible_indices]
        
        # Compute kernel
        kernel_points = self._compute_kernel(viewpoint, visible_points, visible_normals)
        
        if len(kernel_points) == 0:
            # print("  Expand: Kernel is empty.")
            return viewpoint, visible_indices
        
        # --- Algorithm 5.3: Test random points from kernel ---
        
        # "Let K' contain log|Kx| random vertices from K"
        num_test = min(max(1, int(np.log(len(kernel_points)))), 
                       len(kernel_points), 
                       10) # Also cap at 10 (as in original code)
                       
        test_indices = np.random.choice(len(kernel_points), num_test, replace=False)
        
        best_vp = viewpoint
        best_visible = visible_indices
        
        for idx in test_indices:
            test_vp = kernel_points[idx]
            
            # Heuristic for new direction: average vector to visible points
            # This direction is only used for the *visibility check*
            # from the new test_vp, which *needs* a direction.
            direction = np.mean(visible_points - test_vp, axis=0)
            direction = direction / (norm(direction) + 1e-12)
            
            test_visible, _ = self.compute_epsilon_visibility(test_vp, direction)
            
            if len(test_visible) > len(best_visible):
                best_vp = test_vp
                best_visible = test_visible
        
        # --- Algorithm 5.3: Recursion Check ---
        
        # "stops when improvement is less than |P|/c"
        improvement_threshold = self.num_points / c_factor
        
        if len(best_visible) > len(visible_indices) + improvement_threshold:
            print(f"  Expand (Recurse): Found {len(best_visible)} (from {len(visible_indices)})")
            # Recurse from the new best viewpoint
            return self.expand_kernel(best_vp, best_visible, max_iterations, 
                                      recursion_depth + 1, c_factor)
        else:
            print(f"  Expand (Done): Best is {len(best_visible)} (from {len(visible_indices)})")
            # Improvement not significant, stop recursion
            return best_vp, best_visible
    
    def _compute_kernel(self, viewpoint: np.ndarray, visible_points: np.ndarray, visible_normals: np.ndarray) -> np.ndarray:
        """
        Compute approximate kernel (region that can see all visible points).
        
        This is the FIXED implementation based on Lemma 5.2.
        It samples points and checks if they are in the *true* kernel.
        """
        # Sample points around viewpoint (as in original code)
        num_samples = 50
        radius = 0.5

        samples = []
        for _ in range(num_samples):
            offset = np.random.randn(3)
            offset = offset / (norm(offset) + 1e-12) * np.random.uniform(0, radius)
            samples.append(viewpoint + offset)

        samples = np.array(samples)

        # Keep only points that are in the kernel
        kernel = []
        for sample in samples:
            
            # *** FIXED: Implement Lemma 5.2 ***
            # A point x (sample) is in the kernel if for all points p (visible_points),
            # the point p is front-facing from x.
            # Def 3.4 (front-face): (p - x) . n_p > 0
            
            dirs = visible_points - sample  # (p - x)
            
            # (p - x) . n_p
            dots = np.sum(dirs * visible_normals, axis=1)
            
            # Check if (p - x) . n_p > 0 for ALL visible points
            if np.all(dots > 1e-6):
                kernel.append(sample)
        
        return np.array(kernel) if kernel else np.array([])

    def optimize_with_candidates(self, target_coverage=0.95, max_viewpoints=50):
        """
        Solves the set-cover problem using the provided candidates.
        
        This implements a greedy set-cover algorithm that:
        1. Calculates visibility for a candidate `v` -> `V_v`
        2. Expands the kernel `expand_kernel(v, V_v)` -> `(g, V_g)`
        3. Credits the *original candidate* `v` with the *expanded*
           coverage `V_g`
        4. Greedily picks the candidate that covers the most new points.
        """
        
        print("\nRunning 'optimize_with_candidates' (Greedy Set-Cover with Kernel Expansion)...")
        start_time = get_time()

        if self.num_points == 0:
            print("Error: No target points to cover.")
            return None
        if len(self.candidate_viewpoints) == 0:
            print("Error: No candidate viewpoints provided.")
            return None

        # 1. Pre-compute the *initial* visibility for all candidates
        #    This is the most expensive part.
        try:
            initial_visibility_map = self.compute_visibility_for_all_candidates()
        except Exception as e:
            print(f"Failed to compute initial visibility: {e}")
            return None
            
        uncovered = set(range(self.num_points))
        selected_viewpoints = []
        
        # Store candidates as a list of tuples for stable indexing
        candidate_list = [(tuple(vp), tuple(direction)) for vp, direction in self.candidate_viewpoints]
        remaining_candidate_indices = set(range(len(candidate_list)))
        
        target_uncovered_count = int((1.0 - target_coverage) * self.num_points)
        
        print(f"Finding viewpoints for {target_coverage*100}% coverage...")
        
        while len(uncovered) > target_uncovered_count and \
              len(selected_viewpoints) < max_viewpoints:
            
            if not remaining_candidate_indices:
                print("  No more candidates to check.")
                break

            best_candidate_idx = -1
            best_expanded_set = set()
            best_score = 0
            
            # Find the best remaining candidate
            # This loop is the greedy part
            for candidate_idx in remaining_candidate_indices:
                vp_tuple, dir_tuple = candidate_list[candidate_idx]
                vp = np.array(vp_tuple)
                
                # Get the pre-computed initial visibility
                visible_indices = initial_visibility_map[(vp_tuple, dir_tuple)]
                
                # "with the kernel computation"
                # Run kernel expansion
                expanded_vp, expanded_visible = self.expand_kernel(vp, visible_indices)
                
                # Check how many *new* points this expanded view covers
                newly_covered = set(expanded_visible) & uncovered
                score = len(newly_covered)
                
                if score > best_score:
                    best_score = score
                    best_candidate_idx = candidate_idx
                    best_expanded_set = newly_covered
            
            if best_candidate_idx == -1 or best_score == 0:
                print("  No candidate provides new coverage. Stopping.")
                break
                
            # Add the *original candidate* (not the expanded point)
            best_vp_tuple, best_dir_tuple = candidate_list[best_candidate_idx]
            
            uncovered -= best_expanded_set
            coverage = 1.0 - len(uncovered) / self.num_points
            
            selected_viewpoints.append(ViewpointResult(
                position=np.array(best_vp_tuple),
                direction=np.array(best_dir_tuple), # Store the direction
                visible_indices=np.array(list(best_expanded_set)), # Store the *expanded* set
                coverage_score=best_score / self.num_points,
                computation_time=0 # Timing not tracked per-step here
            ))
            
            # Remove this candidate from future consideration
            remaining_candidate_indices.remove(best_candidate_idx)
            
            print(f"  Viewpoint {len(selected_viewpoints)} (Candidate {best_candidate_idx}): "
                  f"+{best_score} points, coverage={coverage*100:.1f}%")

        total_time = get_time() - start_time
        coverage = 1.0 - len(uncovered) / self.num_points
        
        return OptimizationResult(
            method_name="Approximate_StarShaped_Candidates",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=self._compute_redundancy(selected_viewpoints)
        )
        
    def _compute_redundancy(self, viewpoints):
        """Compute average coverage redundancy."""
        if self.num_points == 0 or not viewpoints:
            return 0
        
        coverage_count = np.zeros(self.num_points)
        for vp in viewpoints:
            if len(vp.visible_indices) > 0:
                coverage_count[vp.visible_indices] += 1
        
        covered_points = coverage_count[coverage_count > 0]
        if len(covered_points) == 0:
            return 0
            
        return np.mean(covered_points)
    
    def visualize_candidate_visibility(self, 
                                       visibility_map: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], np.ndarray], 
                                       candidate_index: int = 0):
        """
        Visualizes the target points, all candidate viewpoints, and highlights 
        the visible points for a specific candidate using Open3D.
        
        Args:
            visibility_map: The output dictionary from compute_visibility_for_all_candidates.
            candidate_index: The index of the candidate viewpoint to highlight.
        """
        if not self.target_points.any():
            print("No target points to visualize.")
            return

        candidate_list = list(visibility_map.keys())
        if not candidate_list:
            print("Visibility map is empty.")
            return

        # Handle index wrapping
        index_to_visualize = candidate_index % len(candidate_list)
        selected_key = candidate_list[index_to_visualize]
        selected_vp_pos = np.array(selected_key[0])
        selected_vp_dir = np.array(selected_key[1])
        visible_indices = visibility_map[selected_key]

        # --- 1. Geometry List ---
        geometries = []
        
        # Mesh (Grey/Transparent)
        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
        mesh_vis.compute_vertex_normals()
        geometries.append(mesh_vis)
        
        # Target Point Cloud (Base color)
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(self.target_points)
        
        # Set all points to a neutral color (light grey/dark grey)
        colors = np.full((self.num_points, 3), [0.3, 0.3, 0.3], dtype=np.float64)
        
        # Highlight visible points (Green)
        colors[visible_indices] = [0.0, 1.0, 0.0] # Green for visible
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)
        
        # --- 2. Candidate Viewpoints ---
        # Create spheres for all candidates
        for i, key in enumerate(candidate_list):
            pos = np.array(key[0])
            color = [0.0, 0.0, 1.0] # Blue for normal candidates
            radius = 0.01 
            
            if i == index_to_visualize:
                color = [1.0, 0.0, 0.0] # Red for the selected candidate
                radius = 0.015 # Slightly larger
            
            vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            vp_sphere.translate(pos)
            vp_sphere.paint_uniform_color(color)
            geometries.append(vp_sphere)

        # --- 3. Direction Vector ---
        # Draw the direction vector for the selected viewpoint (Yellow line)
        line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pcd1=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector([selected_vp_pos])),
            pcd2=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector([selected_vp_pos + selected_vp_dir * self.frustum_params.far * 0.2])), # Extend direction line
            correspondences=[[0, 0]]
        )
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]]) # Yellow
        geometries.append(line_set)
        
        geometries.append(pcd_vis) # Add the colored point cloud last

        # --- 4. Visualization ---
        print(f"\nVisualizing Candidate Viewpoint {index_to_visualize} / {len(candidate_list) - 1}:")
        print(f"  Position: {selected_vp_pos}")
        print(f"  Direction: {selected_vp_dir}")
        print(f"  Visible Points: {len(visible_indices)} / {self.num_points}")
        
        o3d.visualization.draw_geometries(geometries, 
                                          window_name=f"Visibility Visualization (Candidate {index_to_visualize})")