from utils import * # type: ignore # Assuming utils is available

class RaycastingVisibilityQuery(VisibilityQuery):
    """
    Implements ground-truth visibility using Open3D's RaycastingScene (BVH).
    This method checks for line-of-sight occlusion between a viewpoint and target points.
    """
    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray, normals: np.ndarray, frustum_params: FrustumParams):
        super().__init__(mesh, target_points, normals, frustum_params)


        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        print("[RaycastingQuery] Initialized O3D RaycastingScene (BVH) for occlusion checks.")

    def compute_visibility(self, viewpoint: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Checks visibility using the RaycastingScene: a ray is cast from the 
        viewpoint to each target point. If the ray hits the mesh *before* it hits the target point, the target is occluded.
        """
        start = get_time()
        
        # 1. Frustum Culling (Saves raycasting time)
        candidate_indices = self.points_in_frustum_with_kdtree(viewpoint, direction)
        
        if len(candidate_indices) == 0:
            print("  [RaycastingQuery] Frustum empty.")
            return np.array([]), get_time() - start
        
        candidate_points = self.target_points[candidate_indices]
        num_candidates = len(candidate_indices)
        
        # 2. Prepare Rays (Viewpoint -> Target Point)
        origins = np.tile(viewpoint, (num_candidates, 1))
        vectors = candidate_points - origins
        
        # Normalize vectors and compute ray length
        distances = norm(vectors, axis=1)
        directions = vectors / (distances[:, np.newaxis] + 1e-12)
        
        # Ray data structure: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
        rays = np.hstack([origins, directions])
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        
        # 3. Cast Rays
        # 't_hit' is the distance along the ray to the first intersection with the mesh.
        ans = self.scene.cast_rays(rays_tensor)
        t_hit = ans['t_hit'].numpy()
        
        # 4. Occlusion Check
        # Occlusion occurs if:
        # a) t_hit is < infinity (i.e., the ray hit something)
        # b) t_hit is LESS THAN the distance to the target point (i.e., occluder is closer)
        
        # We need a small tolerance (epsilon) for floating point comparison.
        TOLERANCE = 1e-4 
        
        # If t_hit < distances - TOLERANCE, it's occluded
        is_visible_mask = (t_hit >= distances - TOLERANCE)
        
        visible_indices = candidate_indices[is_visible_mask]
        
        comp_time = get_time() - start
        print(f"  [RaycastingQuery] Raycasting query complete. {len(visible_indices)} visible points in {comp_time:.4f}s.")
        return visible_indices, comp_time

print("[raycasting_visibility.py] Raycasting Visibility Query module defined.")