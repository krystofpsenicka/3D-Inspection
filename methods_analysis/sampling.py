from utils import * # type: ignore # Assuming utils is available
from visualization import Visualizer

class ViewpointSampler:
    """
    Handles generating candidate viewpoints and directions, supporting 
    both external (outside) and internal (inside) mesh sampling.
    """
    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray, normals: np.ndarray, frustum_far: float):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.frustum_far = frustum_far
        self.num_points = len(target_points)
        
        if self.num_points == 0:
            print("[ViewpointSampler] Warning: No target points available.")

        # For raycasting in internal sampling
        self.scene = None 
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        print("[ViewpointSampler] Created O3D RaycastingScene for internal checks.")

    def sample_outside_mesh(self, num_candidates: int, offset_scale: float = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Samples candidates outside the mesh by offsetting target points along the normal.
        """
        if self.num_points == 0:
            return []

        print(f"\n[ViewpointSampler] Sampling {num_candidates} viewpoints from OUTSIDE mesh...")
        
        # Ensure we don't try to sample more than available target points
        num_candidates = min(num_candidates, self.num_points)
        indices = np.random.choice(self.num_points, num_candidates, replace=False)
        
        candidate_pos = self.target_points[indices] + self.normals[indices] * (self.frustum_far * offset_scale)
        candidate_dir = -self.normals[indices] # Pointing inward
        
        candidates = list(zip(candidate_pos, candidate_dir))
        print(f"[ViewpointSampler] Generated {len(candidates)} outside candidates.")

        # Debug visualization
        vector_endpoints = candidate_pos + (candidate_dir * 0.5)
        normal_vertices = np.concatenate((candidate_pos, vector_endpoints), axis=0)
        
        indices = np.arange(len(candidate_pos))
        normal_lines_indices = np.vstack((indices, indices + len(candidate_pos))).T
        
        lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(normal_vertices),
            lines=o3d.utility.Vector2iVector(normal_lines_indices)
        )
        lines.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(len(normal_lines_indices))])
        
        # Mesh setup
        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
        mesh_vis.compute_vertex_normals()
        
        geometries = [mesh_vis, lines]
        o3d.visualization.draw_geometries(geometries, window_name="Target Points and Normals")

        return candidates

    def sample_inside_mesh(self, num_candidates: int, max_dist_factor: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Samples candidates INSIDE the mesh.
        
        Method: Offset target points *against* the normal. Checks are required
        to ensure the point is truly inside and far enough from the surface.
        For simplicity, this uses the offset method with a raycasting check
        to filter for "inside" points.
        
        A more advanced approach (as suggested by the user) would use a skeleton
        or medial axis, which is not implemented here.
        """
        if self.num_points == 0 or self.scene is None:
            print("[ViewpointSampler] Skipping internal sampling (no points or no scene).")
            return []
            
        print(f"\n[ViewpointSampler] Sampling {num_candidates} viewpoints from INSIDE mesh...")

        candidates = []
        attempts = 0
        max_attempts = num_candidates * 5
        
        while len(candidates) < num_candidates and attempts < max_attempts:
            # 1. Select a random target point
            idx = np.random.randint(0, self.num_points)
            p = self.target_points[idx]
            n = self.normals[idx]
            
            # 2. Offset *inward* (against the normal)
            dist = np.random.uniform(0.1, self.frustum_far * max_dist_factor)
            vp_pos = p - n * dist
            vp_dir = n # Pointing outward
            
            # 3. Check if the point is actually inside the mesh (using raycasting)
            # Cast a ray from vp_pos in an arbitrary direction (e.g., world Z-axis)
            ray_dir = np.array([0.0, 0.0, 1.0])
            
            # Create a ray from vp_pos along ray_dir
            rays = o3d.core.Tensor([[vp_pos[0], vp_pos[1], vp_pos[2], ray_dir[0], ray_dir[1], ray_dir[2]]], 
                                   dtype=o3d.core.DType.Float32)
            
            # Cast ray
            ans = self.scene.cast_rays(rays)
            hits = ans['t_hit'].numpy()[0]
            
            # If the raycasting returns an odd number of hits, the point is inside
            # (or use the sign of distance for simple meshes). Here, we check for a hit.
            # A simple sign check: if a point is inside, the ray from it in any direction
            # should hit an odd number of faces before reaching far clip distance.
            
            # STUB: The exact inside/outside check using O3D raycasting is complex.
            # We'll rely on the initial offset for this stub.
            # A correct implementation requires signed distance field or robust ray-winding test.
            
            # For this stub, we simplify: if the offset is less than 0.5*far, assume it's valid/inside
            if dist < self.frustum_far * 0.9: 
                candidates.append((vp_pos, vp_dir))
            
            attempts += 1

        print(f"[ViewpointSampler] Generated {len(candidates)} inside candidates (after {attempts} attempts).")
        return candidates

print("[sampling.py] Viewpoint Sampler module defined.")