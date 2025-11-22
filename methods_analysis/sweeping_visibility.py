from utils import *
import numpy as np
from numpy.linalg import norm
from typing import Tuple, List, Set, Dict
from time import time as get_time
from collections import defaultdict
from sortedcontainers import SortedSet

class SweepingVisibilityQuery(VisibilityQuery):
    """
    Implements visibility using the geometric sweeping algorithm from 
    Yu & Li (2011) "Computing 3D Shape Guarding and Star Decomposition".
    
    This version focuses on triangle visibility:
    - A triangle is visible if all three of its vertices are visible
    - We sweep through mesh vertices to determine visibility
    - Triangles are pre-subdivided to ensure uniform maximum size
    
    Algorithm (Section 3.1):
    1. Pre-subdivide large triangles to maximum edge length
    2. Frustum cull vertices
    3. Sweep through vertices maintaining triangle counters
    4. Determine triangle visibility based on vertex visibility
    
    Complexity: O(V log V + V×k + V×m) where:
    - V = mesh vertices in frustum
    - k = avg triangles per vertex (~6)
    - m = avg active triangles
    """
    
    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray, 
                 normals: np.ndarray, frustum_params: FrustumParams, max_triangle_edge: float = 0.1):
        super().__init__(mesh, target_points, normals, frustum_params)
        
        self.max_triangle_edge = max_triangle_edge
        
        # Pre-subdivide mesh to ensure uniform triangle sizes
        print(f"[SweepingQuery] Subdividing mesh to max edge length {max_triangle_edge}...")
        self.mesh = self._subdivide_large_triangles(mesh, max_triangle_edge)
        
        # Pre-process mesh triangles
        self.triangles = np.asarray(self.mesh.triangles)
        self.vertices = np.asarray(self.mesh.vertices)
        self.num_triangles = len(self.triangles)
        self.num_vertices = len(self.vertices)
        
        # Pre-compute triangle data for faster intersection tests
        self._precompute_triangle_data()
        
        # Build vertex-to-triangles mapping (as per paper)
        self._build_vertex_triangle_map()
        
        print(f"[SweepingQuery] Initialized for {self.num_vertices} mesh vertices, "
              f"{self.num_triangles} triangles (after subdivision).")

    def _subdivide_large_triangles(self, mesh: o3d.geometry.TriangleMesh, 
                                   max_edge: float) -> o3d.geometry.TriangleMesh:
        """
        Subdivide triangles whose edges exceed max_edge length.
        Iteratively subdivides until all edges are within threshold.
        Efficient version: only checks newly created triangles in each iteration.
        """
        vertex_list = list(np.asarray(mesh.vertices))
        # Store triangles as list of lists for easier manipulation
        triangle_list = [list(tri) for tri in np.asarray(mesh.triangles)]
        
        # Initially, check all triangles
        triangles_to_check = list(range(len(triangle_list)))
        
        max_iterations = 10
        for iteration in range(max_iterations):
            if len(triangles_to_check) == 0:
                print(f"[SweepingQuery] Subdivision complete after {iteration} iterations.")
                break
            
            needs_subdivision = []
            next_vertex_idx = len(vertex_list)
            
            # Only check triangles that were newly created or original
            for tri_idx in triangles_to_check:
                v0_idx, v1_idx, v2_idx = triangle_list[tri_idx]
                v0, v1, v2 = vertex_list[v0_idx], vertex_list[v1_idx], vertex_list[v2_idx]
                
                edge1_len = norm(v1 - v0)
                edge2_len = norm(v2 - v1)
                edge3_len = norm(v0 - v2)
                
                max_len = max(edge1_len, edge2_len, edge3_len)
                
                if max_len > max_edge:
                    needs_subdivision.append((tri_idx, v0_idx, v1_idx, v2_idx))
            
            if len(needs_subdivision) == 0:
                print(f"[SweepingQuery] Subdivision complete after {iteration} iterations.")
                break
            
            print(f"[SweepingQuery] Iteration {iteration}: subdividing {len(needs_subdivision)} triangles...")
            
            # Track newly created triangle indices for next iteration
            newly_created_triangles = []
            
            # Process each triangle that needs subdivision
            for tri_idx, v0_idx, v1_idx, v2_idx in needs_subdivision:
                v0, v1, v2 = vertex_list[v0_idx], vertex_list[v1_idx], vertex_list[v2_idx]
                
                # Add midpoints
                m01 = (v0 + v1) / 2
                m12 = (v1 + v2) / 2
                m20 = (v2 + v0) / 2
                
                vertex_list.extend([m01, m12, m20])
                m01_idx = next_vertex_idx
                m12_idx = next_vertex_idx + 1
                m20_idx = next_vertex_idx + 2
                next_vertex_idx += 3
                
                # Replace old triangle with first new triangle (reuse the slot)
                triangle_list[tri_idx] = [v0_idx, m01_idx, m20_idx]
                
                # Add 3 more new triangles
                new_tri_idx_1 = len(triangle_list)
                triangle_list.append([m01_idx, v1_idx, m12_idx])
                newly_created_triangles.append(new_tri_idx_1)
                
                new_tri_idx_2 = len(triangle_list)
                triangle_list.append([m20_idx, m12_idx, v2_idx])
                newly_created_triangles.append(new_tri_idx_2)
                
                new_tri_idx_3 = len(triangle_list)
                triangle_list.append([m01_idx, m12_idx, m20_idx])
                newly_created_triangles.append(new_tri_idx_3)
                
                # The replaced triangle also needs to be checked
                newly_created_triangles.append(tri_idx)
            
            # Next iteration only checks newly created triangles
            triangles_to_check = newly_created_triangles
        
        # Create new mesh
        vertices = np.array(vertex_list)
        triangles = np.array(triangle_list)
        
        subdivided_mesh = o3d.geometry.TriangleMesh()
        subdivided_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        subdivided_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        return subdivided_mesh

    def compute_visibility(self, viewpoint: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes visible triangles using the paper's sweeping algorithm.
        
        Returns visible triangle indices instead of target point indices.
        """
        start = get_time()
        
        # Step 1: Frustum culling on mesh vertices
        candidate_vertex_indices = self.points_in_frustum_with_kdtree(viewpoint, direction)
        
        if len(candidate_vertex_indices) == 0:
            return np.array([], dtype=int), get_time() - start
        
        # Step 2: Perform sweep to determine visible vertices
        visible_vertex_mask = self._sweep_vertices(viewpoint, candidate_vertex_indices)
        
        visible_vertex_set = set(candidate_vertex_indices[visible_vertex_mask])
        
        # Step 3: Determine visible triangles (all 3 vertices must be visible)
        visible_triangles = self._get_visible_triangles(visible_vertex_set)
        
        comp_time = get_time() - start
        return visible_triangles, comp_time
    
    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        """Returns the (potentially subdivided) mesh."""
        return self.mesh


    def _build_vertex_triangle_map(self):
        """
        Build mapping from vertex index to triangles containing that vertex.
        This is the key data structure from the paper (Section 3.1).
        """
        self.vertex_to_triangles = defaultdict(list)
        
        for tri_idx, (v0, v1, v2) in enumerate(self.triangles):
            self.vertex_to_triangles[v0].append(tri_idx)
            self.vertex_to_triangles[v1].append(tri_idx)
            self.vertex_to_triangles[v2].append(tri_idx)

    def _sweep_vertices(self, viewpoint: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        """
        Sweep through mesh vertices to determine visibility.
        
        Algorithm:
        1. Convert all vertices to spherical coordinates
        2. Sort by (theta, phi)
        3. Sweep through maintaining counters and active triangles
        4. Check occlusion for each vertex against active triangles
        
        Complexity: O(V log V + V×k + V×m)
        """
        # Convert mesh vertices to spherical
        mesh_vectors = self.vertices - viewpoint
        mesh_spherical = self._cartesian_to_spherical(mesh_vectors)
        
        # Pre-compute triangle bounds
        triangle_bounds = self._compute_triangle_bounds_spherical(viewpoint)
        
        # Create event list for candidate vertices only
        events = []
        for v_idx in candidate_indices:
            r, theta, phi = mesh_spherical[v_idx]
            events.append((theta, phi, v_idx, r))
        
        # Sort all events by angle - O(V log V)
        events.sort(key=lambda e: (e[0], e[1]))
        
        # Initialize sweep state
        counters = np.zeros(self.num_triangles, dtype=np.int32)
        active_triangles = SortedSet()
        visible_mask = np.zeros(len(candidate_indices), dtype=bool)
        
        # Map from vertex index to position in candidate_indices
        vertex_to_candidate_pos = {v_idx: i for i, v_idx in enumerate(candidate_indices)}
        
        # Sweep through events - O(V×k + V×m)
        for theta, phi, v_idx, r in events:
            # Update active triangles based on this vertex
            for tri_idx in self.vertex_to_triangles[v_idx]:
                theta_min, theta_max, phi_min, phi_max = triangle_bounds[tri_idx]
                
                # Check if current position is in triangle's angular range
                if theta_max - theta_min > np.pi:
                    in_theta = theta >= theta_min or theta <= (theta_max - 2*np.pi)
                else:
                    in_theta = theta_min <= theta <= theta_max
                
                if phi_max - phi_min > np.pi:
                    in_phi = phi >= phi_min or phi <= (phi_max - 2*np.pi)
                else:
                    in_phi = phi_min <= phi <= phi_max
                
                if not in_theta or not in_phi:
                    continue
                
                # Increment counter (paper's algorithm)
                counters[tri_idx] += 1
                c = counters[tri_idx]
                
                # Update active list L based on counter value
                # Paper: "1 ≤ c_i ≤ 3" means triangle is active
                if c == 1:
                    active_triangles.add(tri_idx)
                elif c > 3:
                    active_triangles.discard(tri_idx)
            
            # Check visibility of this vertex against current active triangles
            vertex = self.vertices[v_idx]
            
            if len(active_triangles) > 0:
                # Filter to triangles actually containing this angle
                filtered_tris = self._filter_triangles_in_range(
                    theta, phi, list(active_triangles), triangle_bounds
                )
                
                if len(filtered_tris) > 0:
                    # Check occlusion - O(m)
                    is_occluded = self._batch_occlusion_test(
                        viewpoint, vertex, r, filtered_tris
                    )
                    candidate_pos = vertex_to_candidate_pos[v_idx]
                    visible_mask[candidate_pos] = not is_occluded
                else:
                    candidate_pos = vertex_to_candidate_pos[v_idx]
                    visible_mask[candidate_pos] = True
            else:
                candidate_pos = vertex_to_candidate_pos[v_idx]
                visible_mask[candidate_pos] = True
        
        return visible_mask

    def _get_visible_triangles(self, visible_vertex_set: Set[int]) -> np.ndarray:
        """
        Determine which triangles are visible.
        A triangle is visible if all three of its vertices are visible.
        """
        visible_triangles = []
        
        for tri_idx, (v0, v1, v2) in enumerate(self.triangles):
            if v0 in visible_vertex_set and v1 in visible_vertex_set and v2 in visible_vertex_set:
                visible_triangles.append(tri_idx)
        
        return np.array(visible_triangles, dtype=int)

    def _filter_triangles_in_range(self, theta: float, phi: float,
                                   triangle_indices: List[int],
                                   triangle_bounds: np.ndarray) -> List[int]:
        """
        Filter triangles to only those whose bounds actually contain (theta, phi).
        Handles wraparound for both theta and phi.
        """
        relevant = []
        
        for tri_idx in triangle_indices:
            theta_min, theta_max, phi_min, phi_max = triangle_bounds[tri_idx]
            
            # Check theta with wraparound
            if theta_max - theta_min > np.pi:
                in_theta = theta >= theta_min or theta <= (theta_max - 2*np.pi)
            else:
                in_theta = theta_min <= theta <= theta_max
            
            # Check phi with wraparound
            if phi_max - phi_min > np.pi:
                in_phi = phi >= phi_min or phi <= (phi_max - 2*np.pi)
            else:
                in_phi = phi_min <= phi <= phi_max
            
            if in_theta and in_phi:
                relevant.append(tri_idx)
        
        return relevant

    def _precompute_triangle_data(self):
        """Pre-compute triangle edges for faster intersection tests."""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        
        self.tri_v0 = v0
        self.tri_edge1 = v1 - v0
        self.tri_edge2 = v2 - v0

    def _compute_triangle_bounds_spherical(self, viewpoint: np.ndarray) -> np.ndarray:
        """
        Compute spherical bounds for each triangle.
        Handles discontinuities in both theta and phi.
        """
        bounds = np.zeros((self.num_triangles, 4))
        
        # Vectorized conversion
        all_tri_vertices = self.vertices[self.triangles.flatten()].reshape(-1, 3, 3)
        all_vectors = all_tri_vertices - viewpoint
        
        for tri_idx in range(self.num_triangles):
            tri_vectors = all_vectors[tri_idx]
            spherical = self._cartesian_to_spherical(tri_vectors)
            
            theta_vals = spherical[:, 1]
            phi_vals = spherical[:, 2]
            
            # Handle theta discontinuity
            theta_span = np.max(theta_vals) - np.min(theta_vals)
            if theta_span > np.pi:
                theta_vals = np.where(theta_vals < 0, theta_vals + 2*np.pi, theta_vals)
            
            theta_min, theta_max = np.min(theta_vals), np.max(theta_vals)
            
            # Handle phi discontinuity
            phi_span = np.max(phi_vals) - np.min(phi_vals)
            if phi_span > np.pi:
                phi_vals = np.where(phi_vals < 0, phi_vals + 2*np.pi, phi_vals)
            
            phi_min, phi_max = np.min(phi_vals), np.max(phi_vals)
            
            bounds[tri_idx] = [theta_min, theta_max, phi_min, phi_max]
        
        return bounds

    def _cartesian_to_spherical(self, vectors: np.ndarray) -> np.ndarray:
        """Convert 3D Cartesian vectors to spherical coordinates."""
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        r_nonzero = np.maximum(r, 1e-12)
        
        theta = np.arcsin(np.clip(z / r_nonzero, -1.0, 1.0))
        phi = np.arctan2(y, x)
        
        return np.column_stack([r, theta, phi])

    def _batch_occlusion_test(self, viewpoint: np.ndarray, vertex: np.ndarray, 
                              distance: float, triangle_indices: List[int]) -> bool:
        """
        Vectorized Möller-Trumbore ray-triangle intersection.
        Returns True if vertex is occluded by any triangle.
        """
        if len(triangle_indices) == 0:
            return False
        
        tri_indices = np.array(triangle_indices)
        v0 = self.tri_v0[tri_indices]
        edge1 = self.tri_edge1[tri_indices]
        edge2 = self.tri_edge2[tri_indices]
        
        ray_dir = vertex - viewpoint
        ray_dir_norm = ray_dir / norm(ray_dir)
        
        epsilon = 1e-8
        
        h = np.cross(ray_dir_norm, edge2)
        a = np.sum(edge1 * h, axis=1)
        
        valid = np.abs(a) > epsilon
        if not np.any(valid):
            return False
        
        a = a[valid]
        v0 = v0[valid]
        edge1 = edge1[valid]
        edge2 = edge2[valid]
        h = h[valid]
        
        f = 1.0 / a
        s = viewpoint - v0
        u = f * np.sum(s * h, axis=1)
        
        valid_u = (u >= 0) & (u <= 1)
        if not np.any(valid_u):
            return False
        
        s = s[valid_u]
        edge1 = edge1[valid_u]
        edge2 = edge2[valid_u]
        f = f[valid_u]
        u = u[valid_u]
        
        q = np.cross(s, edge1)
        v = f * np.sum(ray_dir_norm * q, axis=1)
        
        valid_v = (v >= 0) & (u + v <= 1)
        if not np.any(valid_v):
            return False
        
        edge2 = edge2[valid_v]
        f = f[valid_v]
        q = q[valid_v]
        
        t = f * np.sum(edge2 * q, axis=1)
        
        occluding = (t > epsilon) & (t < distance - epsilon)
        
        return np.any(occluding)


print("[sweeping_visibility.py] Sweeping visibility algorithm loaded (triangle-based with subdivision).")