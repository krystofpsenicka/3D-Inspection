
import open3d as o3d
import open3d.core as o3c
import numpy as np
from numpy.linalg import norm
from scipy.spatial import KDTree, ConvexHull
from scipy.optimize import linprog
from time import time as get_time
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt

# Try to import MILP solver
try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Warning: PuLP not available. Install with: pip install pulp")

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Warning: OR-Tools not available. Install with: pip install ortools")

# ============================================================================
# SHARED UTILITIES
# ============================================================================

@dataclass
class ViewpointResult:
    """Result from a single viewpoint."""
    position: np.ndarray
    direction: np.ndarray
    visible_indices: np.ndarray
    coverage_score: float
    computation_time: float

@dataclass
class OptimizationResult:
    """Complete optimization result."""
    method_name: str
    viewpoints: List[ViewpointResult]
    total_coverage: float
    num_viewpoints: int
    total_time: float
    coverage_per_viewpoint: List[float]
    redundancy: float  # Average times each point is covered

class FrustumParams:
    """Frustum parameters."""
    def __init__(self, fov_deg=30.0, near=1e-6, far=4.5):
        self.fov_deg = fov_deg
        self.near = near
        self.far = far
        self.tan_half_fov = np.tan(np.deg2rad(fov_deg / 2.0))

def get_frustum_basis(direction):
    """Calculate orthonormal basis for frustum."""
    direction = direction / norm(direction)
    
    if 1.0 - np.abs(direction[2]) < 1e-6:
        temp_up = np.array([1.0, 0.0, 0.0])
    else:
        temp_up = np.array([0.0, 0.0, 1.0])
    
    right = np.cross(direction, temp_up)
    right = right / norm(right)
    up = np.cross(right, direction)
    
    return right, up

def get_frustum_bounding_sphere(viewpoint, direction, params):
    """Calculate bounding sphere for frustum (for spatial query)."""
    # Sphere center is midpoint between near and far planes
    center = viewpoint + direction * (params.near + params.far) / 2.0
    
    # Radius is distance from center to far corner of frustum
    half_depth = (params.far - params.near) / 2.0
    far_half_size = params.far * params.tan_half_fov
    radius = np.sqrt(half_depth**2 + 2 * far_half_size**2)
    
    return center, radius

def points_in_frustum_with_kdtree(kdtree, points, viewpoint, direction, params):
    """Frustum culling with KD-tree spatial query."""
    # Step 1: Query bounding sphere to get candidates
    center, radius = get_frustum_bounding_sphere(viewpoint, direction, params)
    candidate_indices = kdtree.query_ball_point(center, radius)
    
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)
    
    # Step 2: Exact frustum test on candidates
    candidate_points = points[candidate_indices]
    
    vp_vectors = candidate_points - viewpoint
    proj_distance = np.dot(vp_vectors, direction)
    
    mask = (proj_distance >= params.near) & (proj_distance <= params.far)
    
    right, up = get_frustum_basis(direction)
    max_size = proj_distance * params.tan_half_fov
    
    lateral_right = np.abs(np.dot(vp_vectors, right))
    lateral_up = np.abs(np.dot(vp_vectors, up))
    
    mask &= (lateral_right < max_size) & (lateral_up < max_size)
    
    return np.array(candidate_indices)[mask]

def points_in_frustum(points, viewpoint, direction, params):
    """Vectorized frustum culling (without KD-tree)."""
    vp_vectors = points - viewpoint
    proj_distance = np.dot(vp_vectors, direction)
    
    mask = (proj_distance >= params.near) & (proj_distance <= params.far)
    
    right, up = get_frustum_basis(direction)
    max_size = proj_distance * params.tan_half_fov
    
    lateral_right = np.abs(np.dot(vp_vectors, right))
    lateral_up = np.abs(np.dot(vp_vectors, up))
    
    mask &= (lateral_right < max_size) & (lateral_up < max_size)
    
    return mask

def sample_viewpoint_candidates(mesh, num_candidates, distance=3.0, seed=42):
    """Generate candidate viewpoints around mesh."""
    np.random.seed(seed)
    
    pcd = mesh.sample_points_uniformly(number_of_points=num_candidates)
    surface_points = np.asarray(pcd.points)
    surface_normals = np.asarray(pcd.normals)
    
    viewpoints = surface_points + surface_normals * distance
    directions = -surface_normals
    
    return viewpoints, directions


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_solution(result: OptimizationResult, target_points: np.ndarray, 
                       mesh: o3d.geometry.TriangleMesh, frustum_params: FrustumParams,
                       title: str = "Viewpoint Solution"):
    """
    Visualize the complete solution with viewpoints and their coverage.
    
    Shows:
    - Base mesh in gray
    - Each viewpoint as a colored sphere
    - Frustum wireframes for each viewpoint
    - Points visible from each viewpoint (color-coded)
    - Coverage statistics
    """
    print(f"\nVisualizing solution: {title}")
    print(f"Total viewpoints: {result.num_viewpoints}")
    print(f"Coverage: {result.total_coverage*100:.2f}%")
    
    geometries = []
    
    # 1. Add base mesh
    base_mesh = o3d.geometry.TriangleMesh(mesh)
    base_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    base_mesh.compute_vertex_normals()
    geometries.append(base_mesh)
    
    # 2. Collect all covered points
    all_covered = set()
    for vp in result.viewpoints:
        all_covered.update(vp.visible_indices)
    
    uncovered_indices = set(range(len(target_points))) - all_covered
    
    # 3. Show uncovered points in red
    if uncovered_indices:
        uncovered_pcd = o3d.geometry.PointCloud()
        uncovered_pcd.points = o3d.utility.Vector3dVector(target_points[list(uncovered_indices)])
        uncovered_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(uncovered_pcd)
        print(f"Uncovered points: {len(uncovered_indices)} ({len(uncovered_indices)/len(target_points)*100:.2f}%)")
    
    # 4. Generate colors for each viewpoint
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, result.num_viewpoints)))
    
    # 5. Add each viewpoint with its coverage
    for i, vp in enumerate(result.viewpoints):
        color = colors[i % len(colors)][:3]
        
        # Viewpoint marker (sphere)
        viewpoint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        viewpoint_sphere.translate(vp.position)
        viewpoint_sphere.paint_uniform_color(color)
        viewpoint_sphere.compute_vertex_normals()
        geometries.append(viewpoint_sphere)
        
        # Frustum wireframe
        frustum = create_frustum_lineset(vp.position, vp.direction, frustum_params)
        frustum.paint_uniform_color(color)
        geometries.append(frustum)
        
        # Visible points from this viewpoint
        if len(vp.visible_indices) > 0:
            visible_pcd = o3d.geometry.PointCloud()
            visible_pcd.points = o3d.utility.Vector3dVector(target_points[vp.visible_indices])
            visible_pcd.paint_uniform_color(color)
            geometries.append(visible_pcd)
        
        # Direction arrow
        arrow_length = 0.5
        arrow_end = vp.position + vp.direction * arrow_length
        arrow_points = np.array([vp.position, arrow_end])
        arrow_lines = np.array([[0, 1]])
        arrow = o3d.geometry.LineSet()
        arrow.points = o3d.utility.Vector3dVector(arrow_points)
        arrow.lines = o3d.utility.Vector2iVector(arrow_lines)
        arrow.paint_uniform_color(color)
        geometries.append(arrow)
    
    # 6. Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{title} - {result.method_name}", width=1920, height=1080)
    
    render_option = vis.get_render_option()
    render_option.point_size = 4.0
    render_option.line_width = 2.0
    render_option.mesh_show_back_face = True
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Add text info (simulate with console output)
    print(f"\nVisualization Legend:")
    print(f"  - Gray mesh: Base geometry")
    print(f"  - Red points: Uncovered")
    print(f"  - Colored spheres: Viewpoint positions")
    print(f"  - Colored points: Points visible from corresponding viewpoint")
    print(f"  - Wireframe pyramids: Frustum volumes")
    print(f"  - Lines: View directions")
    print(f"\nPress Q to close visualization")
    
    vis.run()
    vis.destroy_window()

def create_frustum_lineset(viewpoint, direction, params):
    """Create a LineSet representing the frustum volume."""
    # Get frustum corners
    half_angle_rad = np.deg2rad(params.fov_deg / 2.0)
    far_half_size = params.far * np.tan(half_angle_rad)
    
    right, up = get_frustum_basis(direction)
    
    # Far plane center
    far_center = viewpoint + direction * params.far
    
    # Far plane corners
    r = right * far_half_size
    u = up * far_half_size
    
    corners = [
        viewpoint,  # 0: apex
        far_center + r + u,  # 1: far top-right
        far_center - r + u,  # 2: far top-left
        far_center - r - u,  # 3: far bottom-left
        far_center + r - u,  # 4: far bottom-right
    ]
    
    # Define lines
    lines = [
        # Far plane rectangle
        [1, 2], [2, 3], [3, 4], [4, 1],
        # Lines from apex to corners
        [0, 1], [0, 2], [0, 3], [0, 4]
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(corners))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    
    return line_set

def visualize_coverage_heatmap(result: OptimizationResult, target_points: np.ndarray,
                               mesh: o3d.geometry.TriangleMesh, title: str = "Coverage Heatmap"):
    """
    Visualize coverage as a heatmap showing how many times each point is seen.
    """
    print(f"\nCreating coverage heatmap for: {title}")
    
    # Count how many viewpoints see each point
    coverage_count = np.zeros(len(target_points))
    for vp in result.viewpoints:
        coverage_count[vp.visible_indices] += 1
    
    # Normalize to 0-1 range
    max_count = np.max(coverage_count) if np.max(coverage_count) > 0 else 1
    coverage_normalized = coverage_count / max_count
    
    # Create point cloud with heatmap colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_points)
    
    # Map counts to colors (blue = not covered, red = heavily covered)
    colors = plt.cm.jet(coverage_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add mesh
    base_mesh = o3d.geometry.TriangleMesh(mesh)
    base_mesh.paint_uniform_color([0.9, 0.9, 0.9])
    base_mesh.compute_vertex_normals()
    
    # Add viewpoint markers
    viewpoint_spheres = []
    for vp in result.viewpoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(vp.position)
        sphere.paint_uniform_color([0, 0, 0])
        viewpoint_spheres.append(sphere)
    
    # Visualize
    geometries = [base_mesh, pcd] + viewpoint_spheres
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{title} - Heatmap", width=1920, height=1080)
    
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    print(f"\nHeatmap Legend:")
    print(f"  - Blue: Not covered or seen once")
    print(f"  - Green/Yellow: Seen 2-3 times")
    print(f"  - Red: Seen {int(max_count)}+ times (maximum redundancy)")
    print(f"  - Black spheres: Viewpoint positions")
    print(f"\nPress Q to close")
    
    vis.run()
    vis.destroy_window()

# ============================================================================
# CONTINUE WITH REST OF CODE...
# ============================================================================
"""
Complete implementation of three viewpoint optimization approaches plus hybrids:
1. Ray Casting + Set Cover (Your approach)
2. Approximate Star-Shaped Decomposition (Lien's approach)
3. Progressive ILP with Skeleton (Yu & Li's approach)
4-7. Hybrid methods combining the above

Includes benchmarking and analysis framework.
"""

# ============================================================================
# METHOD 1: RAY CASTING + SET COVER (Your Original Approach)
# ============================================================================

class RayCastingSetCover:
    """Ray casting with exact occlusion + greedy set cover."""
    
    def __init__(self, mesh, target_points, frustum_params):
        self.mesh = mesh
        self.target_points = target_points
        self.frustum_params = frustum_params
        self.num_points = len(target_points)
        
        # Setup raycasting
        self.device = o3c.Device("CPU:0")
        self.dtype = o3c.float32
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh, device=self.device)
        self.scene = o3d.t.geometry.RaycastingScene(device=self.device)
        self.object_id = self.scene.add_triangles(mesh_t)
        
        # Spatial index
        self.kdtree = KDTree(target_points)
    
    def compute_visibility(self, viewpoint, direction):
        """Compute visible points using ray casting."""
        start = get_time()
        
        # Frustum culling with KD-tree
        frustum_indices = points_in_frustum_with_kdtree(
            self.kdtree, self.target_points, viewpoint, direction, self.frustum_params
        )
        
        if len(frustum_indices) == 0:
            return np.array([]), get_time() - start
        
        frustum_points = self.target_points[frustum_indices]
        
        # Ray casting
        ray_origins = np.tile(viewpoint, (len(frustum_points), 1))
        ray_dirs = frustum_points - ray_origins
        ray_lengths = norm(ray_dirs, axis=1)
        ray_dirs = ray_dirs / ray_lengths[:, np.newaxis]
        
        rays = o3c.Tensor(
            np.hstack([ray_origins, ray_dirs]),
            dtype=self.dtype,
            device=self.device
        )
        
        ans = self.scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy()
        geometry_ids = ans['geometry_ids'].numpy()
        
        # Check visibility
        visible_mask = (
            (geometry_ids == self.object_id) &
            (np.abs(t_hit - ray_lengths) < 1e-5)
        )
        
        visible_indices = frustum_indices[visible_mask]
        
        return visible_indices, get_time() - start
    
    def solve_set_cover(self, viewpoint_candidates, direction_candidates,
                       target_coverage=0.95, max_viewpoints=50):
        """
        Solve set cover using fastest available solver (OR-Tools, Gurobi, or PuLP).
        This replaces the greedy approach with optimal/near-optimal solution.
        """
        start_time = get_time()
        
        print(f"Ray Casting Set Cover: Computing visibility for all candidates...")
        
        # Step 1: Compute visibility for ALL candidates
        visibility_matrix = []
        valid_candidates = []
        valid_directions = []
        
        for i, (vp, direction) in enumerate(zip(viewpoint_candidates, direction_candidates)):
            visible, _ = self.compute_visibility(vp, direction)
            if len(visible) > 0:
                visibility_matrix.append(set(visible))
                valid_candidates.append(vp)
                valid_directions.append(direction)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(viewpoint_candidates)} candidates")
        
        print(f"  Valid candidates: {len(valid_candidates)}")
        
        if len(valid_candidates) == 0:
            print("ERROR: No valid candidates found!")
            return OptimizationResult(
                method_name="RayCasting_SetCover",
                viewpoints=[],
                total_coverage=0.0,
                num_viewpoints=0,
                total_time=get_time() - start_time,
                coverage_per_viewpoint=[],
                redundancy=0.0
            )
        
        # Step 2: Solve set cover with best available solver
        print(f"  Solving set cover problem...")
        selected_indices = self._solve_set_cover_optimal(
            visibility_matrix, target_coverage, max_viewpoints
        )
        
        # Step 3: Build result
        selected_viewpoints = []
        covered = set()
        
        for idx in selected_indices:
            vp = valid_candidates[idx]
            direction = valid_directions[idx]
            visible = list(visibility_matrix[idx])
            
            selected_viewpoints.append(ViewpointResult(
                position=vp,
                direction=direction,
                visible_indices=np.array(visible),
                coverage_score=len(visible) / self.num_points,
                computation_time=0.0
            ))
            covered.update(visible)
        
        total_time = get_time() - start_time
        coverage = len(covered) / self.num_points
        
        print(f"  Selected {len(selected_indices)} viewpoints")
        print(f"  Coverage: {coverage*100:.2f}%")
        
        return OptimizationResult(
            method_name="RayCasting_SetCover",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=self._compute_redundancy(selected_viewpoints)
        )
    
    def _solve_set_cover_optimal(self, visibility_sets, target_coverage, max_viewpoints):
        """Solve set cover with best available solver."""
        num_candidates = len(visibility_sets)
        target_points_to_cover = int(target_coverage * self.num_points)
        
        # Try OR-Tools first (fastest)
        if ORTOOLS_AVAILABLE:
            return self._solve_with_ortools(visibility_sets, target_points_to_cover, max_viewpoints)
        
        # Try Gurobi second (very fast, but commercial)
        if GUROBI_AVAILABLE:
            return self._solve_with_gurobi(visibility_sets, target_points_to_cover, max_viewpoints)
        
        # Fall back to PuLP
        if PULP_AVAILABLE:
            return self._solve_with_pulp(visibility_sets, target_points_to_cover, max_viewpoints)
        
        # Last resort: greedy
        print("  Warning: No optimization solver available, using greedy approach")
        return self._solve_greedy(visibility_sets, target_points_to_cover, max_viewpoints)
    
    def _solve_with_ortools(self, visibility_sets, target_coverage, max_viewpoints):
        """Solve using Google OR-Tools (recommended - fast and free)."""
        print("    Using OR-Tools solver...")
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return self._solve_greedy(visibility_sets, target_coverage, max_viewpoints)
        
        num_candidates = len(visibility_sets)
        
        # Variables: x[i] = 1 if candidate i is selected
        x = [solver.BoolVar(f'x_{i}') for i in range(num_candidates)]
        
        # Objective: minimize number of viewpoints
        solver.Minimize(sum(x))
        
        # Constraint: each point must be covered
        point_coverage = defaultdict(list)
        for i, vis_set in enumerate(visibility_sets):
            for point_idx in vis_set:
                point_coverage[point_idx].append(i)
        
        # Add constraints for most frequent points (to keep problem manageable)
        points_to_constrain = sorted(point_coverage.keys(), 
                                     key=lambda p: len(point_coverage[p]), 
                                     reverse=True)[:target_coverage]
        
        for point_idx in points_to_constrain:
            solver.Add(sum(x[i] for i in point_coverage[point_idx]) >= 1)
        
        # Limit max viewpoints
        solver.Add(sum(x) <= max_viewpoints)
        
        # Solve
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            selected = [i for i in range(num_candidates) if x[i].solution_value() > 0.5]
            print(f"    OR-Tools found solution with {len(selected)} viewpoints")
            return selected
        
        return self._solve_greedy(visibility_sets, target_coverage, max_viewpoints)
    
    def _solve_with_gurobi(self, visibility_sets, target_coverage, max_viewpoints):
        """Solve using Gurobi (commercial solver)."""
        print("    Using Gurobi solver...")
        
        try:
            model = gp.Model("set_cover")
            model.setParam('OutputFlag', 0)
            
            num_candidates = len(visibility_sets)
            
            # Variables
            x = model.addVars(num_candidates, vtype=GRB.BINARY, name="x")
            
            # Objective
            model.setObjective(gp.quicksum(x[i] for i in range(num_candidates)), GRB.MINIMIZE)
            
            # Constraints
            point_coverage = defaultdict(list)
            for i, vis_set in enumerate(visibility_sets):
                for point_idx in vis_set:
                    point_coverage[point_idx].append(i)
            
            for point_idx, covering_candidates in point_coverage.items():
                model.addConstr(gp.quicksum(x[i] for i in covering_candidates) >= 1)
            
            model.addConstr(gp.quicksum(x[i] for i in range(num_candidates)) <= max_viewpoints)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                selected = [i for i in range(num_candidates) if x[i].X > 0.5]
                print(f"    Gurobi found optimal solution with {len(selected)} viewpoints")
                return selected
        except Exception as e:
            print(f"    Gurobi error: {e}")
        
        return self._solve_greedy(visibility_sets, target_coverage, max_viewpoints)
    
    def _solve_with_pulp(self, visibility_sets, target_coverage, max_viewpoints):
        """Solve using PuLP."""
        print("    Using PuLP solver...")
        
        prob = LpProblem("SetCover", LpMinimize)
        
        num_candidates = len(visibility_sets)
        
        # Variables
        x = [LpVariable(f"x_{i}", cat='Binary') for i in range(num_candidates)]
        
        # Objective
        prob += lpSum(x)
        
        # Constraints
        point_coverage = defaultdict(list)
        for i, vis_set in enumerate(visibility_sets):
            for point_idx in vis_set:
                point_coverage[point_idx].append(i)
        
        for point_idx, covering_candidates in point_coverage.items():
            prob += lpSum(x[i] for i in covering_candidates) >= 1
        
        prob += lpSum(x) <= max_viewpoints
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:  # Optimal
            selected = [i for i in range(num_candidates) if x[i].varValue > 0.5]
            print(f"    PuLP found solution with {len(selected)} viewpoints")
            return selected
        
        return self._solve_greedy(visibility_sets, target_coverage, max_viewpoints)
    
    def _solve_greedy(self, visibility_sets, target_coverage, max_viewpoints):
        """Greedy fallback algorithm."""
        print("    Using greedy algorithm...")
        
        uncovered = set(range(self.num_points))
        selected = []
        
        while len(selected) < max_viewpoints and len(uncovered) > (self.num_points - target_coverage):
            best_idx = None
            best_new_coverage = 0
            
            for i, vis_set in enumerate(visibility_sets):
                if i in selected:
                    continue
                new_coverage = len(vis_set & uncovered)
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_idx = i
            
            if best_idx is None:
                break
            
            selected.append(best_idx)
            uncovered -= visibility_sets[best_idx]
        
        print(f"    Greedy found solution with {len(selected)} viewpoints")
        return selected
    
    def _compute_redundancy(self, viewpoints):
        """Compute average coverage redundancy."""
        coverage_count = np.zeros(self.num_points)
        for vp in viewpoints:
            coverage_count[vp.visible_indices] += 1
        return np.mean(coverage_count[coverage_count > 0])

# ============================================================================
# METHOD 2: APPROXIMATE STAR-SHAPED (Lien's Approach)
# ============================================================================

class ApproximateStarShaped:
    """ε-visibility based on Lien's paper."""
    
    def __init__(self, mesh, target_points, frustum_params, epsilon=None):
        self.mesh = mesh
        self.target_points = target_points
        self.frustum_params = frustum_params
        self.num_points = len(target_points)
        
        # Build KD-tree FIRST
        self.kdtree = KDTree(target_points)
        
        # Compute normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        # Optional: Orient normals for consistency
        pcd.orient_normals_consistent_tangent_plane(k=10)

        self.normals = np.asarray(pcd.normals)
        self.pcd = pcd # Store the PointCloud for visualization
        
        # 3. --- Visualization within __init__ ---
        self._visualize_normals(normal_scale=0.05)

    def _visualize_normals(self, normal_scale=0.05):
        """Visualizes the mesh, the target points, and their computed normals."""
        print("\nVisualizing Normals")
        
        # A. Prepare Geometries
        
        # 1. Mesh
        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8]) # Light gray
        mesh_vis.compute_vertex_normals()
        
        # 2. Point Cloud (Target Points)
        pcd_vis = o3d.geometry.PointCloud(self.pcd)
        pcd_vis.paint_uniform_color([1.0, 0.0, 0.0]) # Red points
        
        # 3. Normals as LineSet (Manual Construction) 🚀
        
        points = np.asarray(pcd_vis.points)
        normals = np.asarray(pcd_vis.normals)
        
        # Calculate the end point of the normal vector: Start_Point + (Normal * Scale)
        normal_endpoints = points + (normals * normal_scale)
        
        # Combine the starting points and the endpoints into a single list of vertices
        # [P1, P2, P3, ..., Pn, E1, E2, E3, ..., En]
        normal_vertices = np.concatenate((points, normal_endpoints), axis=0)
        
        # Create the lineset indices: 
        # Line 1 connects vertex 0 (P1) to vertex N (E1)
        # Line 2 connects vertex 1 (P2) to vertex N+1 (E2)
        # ... and so on
        indices = np.arange(len(points))
        normal_lines_indices = np.vstack((indices, indices + len(points))).T
        
        # Create the LineSet object
        normal_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(normal_vertices),
            lines=o3d.utility.Vector2iVector(normal_lines_indices)
        )
        # Give the normal lines a distinct color (e.g., black)
        normal_lines.colors = o3d.utility.Vector3dVector(
            [[0, 0, 0] for _ in range(len(normal_lines_indices))]
        )
        
        geometries = [mesh_vis, pcd_vis, normal_lines]
        
        # B. Setup and Run Visualizer
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
        sample_indices = np.random.choice(self.num_points, sample_size, replace=False)
        
        distances = []
        for idx in sample_indices:
            _, dists = self.kdtree.query(self.target_points[idx], k=5)
            distances.append(np.max(dists[1:]))  # Skip self
        
        delta = np.max(distances)  # Sampling density
        
        # Estimate gamma (distance to medial axis) as half max visibility distance
        gamma = self.frustum_params.far / 4.0
        
        # Compute epsilon
        epsilon = 2 * np.arctan(delta / (4 * gamma))
        return np.rad2deg(epsilon)
    
    def compute_epsilon_visibility(self, viewpoint, direction):
        """Compute ε-visible region using back-face and occlusion checks."""
        start = get_time()
        
        # Frustum culling with KD-tree
        frustum_indices = points_in_frustum_with_kdtree(
            self.kdtree, self.target_points, viewpoint, direction, self.frustum_params
        )
        
        if len(frustum_indices) == 0:
            return np.array([]), get_time() - start
        
        frustum_points = self.target_points[frustum_indices]
        frustum_normals = self.normals[frustum_indices]
        
        # Check back-face visibility
        view_dirs = frustum_points - viewpoint
        view_dirs = view_dirs / norm(view_dirs, axis=1)[:, np.newaxis]
        
        # Point is back-facing if normal points away from viewpoint
        dot_products = np.sum(frustum_normals * view_dirs, axis=1)
        front_facing = dot_products > 0
        
        # Radial partitioning for occlusion check
        visible_mask = self._check_occlusion(
            viewpoint, frustum_points, frustum_normals, front_facing
        )
        
        visible_indices = frustum_indices[visible_mask]
        
        return visible_indices, get_time() - start
    
    def _check_occlusion(self, viewpoint, points, normals, front_facing):
        """Check occlusion using radial partitioning."""
        num_points = len(points)
        visible = front_facing.copy()
        
        if not np.any(front_facing):
            return visible
        
        # Convert to spherical coordinates
        relative = points - viewpoint
        distances = norm(relative, axis=1)
        
        # Simple angular binning (simplified version of paper's approach)
        theta = np.arctan2(relative[:, 1], relative[:, 0])
        phi = np.arcsin(np.clip(relative[:, 2] / np.maximum(distances, 1e-6), -1, 1))
        
        # Discretize angles
        num_bins = 36
        theta_bins = ((theta + np.pi) / (2 * np.pi) * num_bins).astype(int) % num_bins
        phi_bins = ((phi + np.pi/2) / np.pi * num_bins).astype(int) % num_bins
        
        # For each bin, find closest back-facing point
        bins = {}
        for i in range(num_points):
            if not front_facing[i]:
                bin_key = (theta_bins[i], phi_bins[i])
                if bin_key not in bins or distances[i] < bins[bin_key][0]:
                    bins[bin_key] = (distances[i], i)
        
        # Check occlusion for each front-facing point
        for i in range(num_points):
            if front_facing[i]:
                bin_key = (theta_bins[i], phi_bins[i])
                if bin_key in bins:
                    occluder_dist, _ = bins[bin_key]
                    if distances[i] > occluder_dist:
                        visible[i] = False
        
        return visible
    
    def expand_kernel(self, viewpoint, visible_indices, max_iterations=3):
        """A* expansion to find better guard in kernel (Algorithm 5.3)."""
        if len(visible_indices) < 4:
            return viewpoint, visible_indices
        
        visible_points = self.target_points[visible_indices]
        visible_normals = self.normals[visible_indices]
        
        # Compute kernel using half-space intersections
        kernel_points = self._compute_kernel(viewpoint, visible_points, visible_normals)
        
        if len(kernel_points) == 0:
            return viewpoint, visible_indices
        
        # Test random points in kernel
        best_vp = viewpoint
        best_visible = visible_indices
        
        num_test = min(10, len(kernel_points))
        test_indices = np.random.choice(len(kernel_points), num_test, replace=False)
        
        for idx in test_indices:
            test_vp = kernel_points[idx]
            direction = np.mean(visible_points - test_vp, axis=0)
            direction = direction / norm(direction)
            
            test_visible, _ = self.compute_epsilon_visibility(test_vp, direction)
            
            if len(test_visible) > len(best_visible):
                best_vp = test_vp
                best_visible = test_visible
        
        return best_vp, best_visible
    
    def _compute_kernel(self, viewpoint, visible_points, visible_normals):
        """Compute approximate kernel (region that can see all visible points)."""
        # Sample points around viewpoint
        num_samples = 50
        radius = 0.5

        samples = []
        for _ in range(num_samples):
            offset = np.random.randn(3)
            offset = offset / norm(offset) * np.random.uniform(0, radius)
            samples.append(viewpoint + offset)

        samples = np.array(samples)

        # Keep only points that can "see" all visible points (simplified)
        kernel = []
        for sample in samples:
            dirs = visible_points - sample
            dirs = dirs / norm(dirs, axis=1)[:, np.newaxis]
            dots = np.sum(dirs * visible_normals, axis=1)
            if np.all(dots > -0.5):  # Relaxed visibility check
                kernel.append(sample)
        
        return np.array(kernel) if kernel else np.array([])
    
    def optimize(self, viewpoint_candidates, direction_candidates,
                 target_coverage=0.95, max_viewpoints=50):
        """Main optimization with A* expansion."""
        start_time = get_time()
        
        uncovered = set(range(self.num_points))
        selected_viewpoints = []
        
        print(f"Approximate Star-Shaped: Finding viewpoints for {target_coverage*100}% coverage...")
        
        while len(uncovered) > (1 - target_coverage) * self.num_points and len(selected_viewpoints) < max_viewpoints:
            best_vp = None
            best_visible = None
            best_score = 0
            best_time = 0
            best_direction = None
            
            # Sample candidates
            num_to_check = min(len(viewpoint_candidates), 50)
            check_indices = np.random.choice(
                len(viewpoint_candidates), num_to_check, replace=False
            )
            
            for idx in check_indices:
                vp = viewpoint_candidates[idx]
                direction = direction_candidates[idx]
                
                # Compute ε-visibility
                visible, comp_time = self.compute_epsilon_visibility(vp, direction)
                
                # Expand kernel
                expanded_vp, visible = self.expand_kernel(vp, visible)
                
                new_coverage = len(set(visible) & uncovered)
                
                if new_coverage > best_score:
                    best_score = new_coverage
                    best_vp = expanded_vp
                    best_visible = visible
                    best_time = comp_time
                    best_direction = direction
            
            if best_vp is None or best_score == 0:
                break
            
            uncovered -= set(best_visible)
            coverage = 1.0 - len(uncovered) / self.num_points
            
            selected_viewpoints.append(ViewpointResult(
                position=best_vp,
                direction=best_direction,
                visible_indices=best_visible,
                coverage_score=best_score / self.num_points,
                computation_time=best_time
            ))
            
            print(f"  Viewpoint {len(selected_viewpoints)}: +{best_score} points, "
                  f"coverage={coverage*100:.1f}%")
        
        total_time = get_time() - start_time
        coverage = 1.0 - len(uncovered) / self.num_points
        
        return OptimizationResult(
            method_name="Approximate_StarShaped",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=self._compute_redundancy(selected_viewpoints)
        )
    
    def _compute_redundancy(self, viewpoints):
        """Compute average coverage redundancy."""
        coverage_count = np.zeros(self.num_points)
        for vp in viewpoints:
            coverage_count[vp.visible_indices] += 1
        return np.mean(coverage_count[coverage_count > 0])

# ============================================================================
# METHOD 3: PROGRESSIVE ILP (Yu & Li's Approach)
# ============================================================================

class ProgressiveILP:
    """Progressive Integer Linear Programming with skeleton."""
    
    def __init__(self, mesh, target_points, frustum_params):
        self.mesh = mesh
        self.target_points = target_points
        self.frustum_params = frustum_params
        self.num_points = len(target_points)
        
        # Setup raycasting FIRST (before skeleton extraction)
        self.device = o3c.Device("CPU:0")
        self.dtype = o3c.float32
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh, device=self.device)
        self.scene = o3d.t.geometry.RaycastingScene(device=self.device)
        self.object_id = self.scene.add_triangles(mesh_t)
        
        # Extract skeleton (uses self.scene)
        print("Extracting skeleton...")
        self.skeleton_points = self._extract_skeleton()
        print(f"  Skeleton has {len(self.skeleton_points)} points")
    
    def _extract_skeleton(self):
        """Extract skeleton using surface sampling + thinning."""
        # Sample interior points
        bbox = self.mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        # Generate grid of interior points
        resolution = 10
        x = np.linspace(center[0] - extent[0]/2, center[0] + extent[0]/2, resolution)
        y = np.linspace(center[1] - extent[1]/2, center[1] + extent[1]/2, resolution)
        z = np.linspace(center[2] - extent[2]/2, center[2] + extent[2]/2, resolution)
        
        grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        
        # Keep only interior points (simplified - use raycasting)
        interior_mask = self._filter_interior_points(grid_points)
        interior_points = grid_points[interior_mask]
        
        # Thin to skeleton (simplified - just subsample)
        if len(interior_points) > 100:
            indices = np.random.choice(len(interior_points), 100, replace=False)
            interior_points = interior_points[indices]
        
        return interior_points
    
    def _filter_interior_points(self, points):
        """Check which points are inside the mesh."""
        # Use raycasting in multiple directions to determine inside/outside
        interior = np.ones(len(points), dtype=bool)
        
        # Cast rays in +X direction
        ray_dirs = np.tile([1, 0, 0], (len(points), 1))
        rays = o3c.Tensor(
            np.hstack([points, ray_dirs]),
            dtype=self.dtype,
            device=self.device
        )
        
        ans = self.scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy()
        
        # Point is inside if ray hits something
        interior = np.isfinite(t_hit)
        
        return interior
    
    def compute_visibility_from_skeleton(self, skeleton_point):
        """Compute visibility of surface points from skeleton point."""
        start = get_time()
        
        # Direction: from skeleton toward surface (average)
        direction = np.mean(self.target_points - skeleton_point, axis=0)
        direction = direction / norm(direction)
        
        # Cast rays to all target points
        ray_origins = np.tile(skeleton_point, (self.num_points, 1))
        ray_dirs = self.target_points - ray_origins
        ray_lengths = norm(ray_dirs, axis=1)
        ray_dirs = ray_dirs / ray_lengths[:, np.newaxis]
        
        rays = o3c.Tensor(
            np.hstack([ray_origins, ray_dirs]),
            dtype=self.dtype,
            device=self.device
        )
        
        ans = self.scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy()
        geometry_ids = ans['geometry_ids'].numpy()
        
        # Check visibility
        visible_mask = (
            (geometry_ids == self.object_id) &
            (np.abs(t_hit - ray_lengths) < 1e-5)
        )
        
        visible_indices = np.where(visible_mask)[0]
        
        return visible_indices, get_time() - start
    
    def solve_ilp_level(self, skeleton_subset, uncovered_indices, max_guards=10):
        """Solve ILP for one level."""
        if not PULP_AVAILABLE:
            # Fallback to greedy
            return self._greedy_subset(skeleton_subset, uncovered_indices, max_guards)
        
        # Compute visibility matrix
        visibility = {}
        for i, skel_pt in enumerate(skeleton_subset):
            visible, _ = self.compute_visibility_from_skeleton(skel_pt)
            visible_in_uncovered = set(visible) & set(uncovered_indices)
            if visible_in_uncovered:
                visibility[i] = visible_in_uncovered
        
        if not visibility:
            return []
        
        # Setup ILP
        prob = LpProblem("Guarding", LpMinimize)
        
        # Variables: x_i = 1 if skeleton point i is selected
        x = {i: LpVariable(f"x_{i}", cat='Binary') for i in visibility.keys()}
        
        # Objective: minimize number of guards
        prob += lpSum([x[i] for i in x])
        
        # Constraints: each uncovered point must be seen by at least one guard
        for point_idx in uncovered_indices:
            covering_guards = [i for i, visible_set in visibility.items() 
                             if point_idx in visible_set]
            if covering_guards:
                prob += lpSum([x[i] for i in covering_guards]) >= 1
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract solution
        selected = [i for i in x if x[i].varValue > 0.5]
        return [skeleton_subset[i] for i in selected]
    
    def _greedy_subset(self, skeleton_subset, uncovered_indices, max_guards):
        """Greedy fallback when ILP not available."""
        uncovered = set(uncovered_indices)
        selected = []
        
        for _ in range(max_guards):
            if not uncovered:
                break
            
            best_skel = None
            best_coverage = 0
            
            for skel_pt in skeleton_subset:
                visible, _ = self.compute_visibility_from_skeleton(skel_pt)
                coverage = len(set(visible) & uncovered)
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_skel = skel_pt
            
            if best_skel is None:
                break
            
            selected.append(best_skel)
            visible, _ = self.compute_visibility_from_skeleton(best_skel)
            uncovered -= set(visible)
        
        return selected
    
    def progressive_optimization(self, target_coverage=0.95, max_viewpoints=50):
        """Progressive ILP with multi-resolution."""
        start_time = get_time()
        
        selected_guards = []
        uncovered = set(range(self.num_points))
        
        # Level 0: Coarse skeleton (25% of points)
        level_sizes = [0.25, 0.5, 1.0]
        
        print(f"Progressive ILP: Finding viewpoints for {target_coverage*100}% coverage...")
        
        for level, size_fraction in enumerate(level_sizes):
            print(f"  Level {level}: Using {size_fraction*100:.0f}% of skeleton...")
            
            num_skel = max(5, int(len(self.skeleton_points) * size_fraction))
            skeleton_subset = self.skeleton_points[
                np.random.choice(len(self.skeleton_points), num_skel, replace=False)
            ]
            
            # Solve ILP for this level
            uncovered_list = list(uncovered)
            new_guards = self.solve_ilp_level(skeleton_subset, uncovered_list, 
                                              max_guards=10)
            
            # Add new guards and update coverage
            for guard_pos in new_guards:
                visible, comp_time = self.compute_visibility_from_skeleton(guard_pos)
                
                uncovered -= set(visible)
                coverage = 1.0 - len(uncovered) / self.num_points
                
                # Direction: toward visible points
                visible_points = self.target_points[visible]
                direction = np.mean(visible_points - guard_pos, axis=0)
                direction = direction / norm(direction)
                
                selected_guards.append(ViewpointResult(
                    position=guard_pos,
                    direction=direction,
                    visible_indices=np.array(visible),
                    coverage_score=len(visible) / self.num_points,
                    computation_time=comp_time
                ))
                
                if len(selected_guards) >= max_viewpoints:
                    break
            
            if len(selected_guards) >= max_viewpoints:
                break
            
            # Check if we've reached target coverage
            coverage = 1.0 - len(uncovered) / self.num_points
            if coverage >= target_coverage:
                break
        
        total_time = get_time() - start_time
        final_coverage = 1.0 - len(uncovered) / self.num_points
        
        return OptimizationResult(
            method_name="Progressive_ILP",
            viewpoints=selected_guards,
            total_coverage=final_coverage,
            num_viewpoints=len(selected_guards),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_guards],
            redundancy=self._compute_redundancy(selected_guards)
        )
    
    def _compute_redundancy(self, viewpoints):
        """Compute average coverage redundancy."""
        coverage_count = np.zeros(self.num_points)
        for vp in viewpoints:
            coverage_count[vp.visible_indices] += 1
        return np.mean(coverage_count[coverage_count > 0])

# ============================================================================
# HYBRID METHODS
# ============================================================================

class HybridMethod1:
    """Hybrid: Ray Casting visibility + A* kernel expansion."""
    
    def __init__(self, mesh, target_points, frustum_params):
        self.ray_caster = RayCastingSetCover(mesh, target_points, frustum_params)
        self.star_shaped = ApproximateStarShaped(mesh, target_points, frustum_params)
        self.num_points = len(target_points)
    
    def optimize(self, viewpoint_candidates, direction_candidates,
                 target_coverage=0.95, max_viewpoints=50):
        """Use exact ray casting but expand with kernel optimization."""
        start_time = get_time()
        
        uncovered = set(range(self.num_points))
        selected_viewpoints = []
        
        print(f"Hybrid RayCast+Kernel: Finding viewpoints for {target_coverage*100}% coverage...")
        
        while len(uncovered) > (1 - target_coverage) * self.num_points and len(selected_viewpoints) < max_viewpoints:
            best_vp = None
            best_visible = None
            best_score = 0
            best_time = 0
            best_direction = None
            
            num_to_check = min(len(viewpoint_candidates), 50)
            check_indices = np.random.choice(
                len(viewpoint_candidates), num_to_check, replace=False
            )
            
            for idx in check_indices:
                vp = viewpoint_candidates[idx]
                direction = direction_candidates[idx]
                
                # Step 1: Exact visibility with ray casting
                visible, comp_time = self.ray_caster.compute_visibility(vp, direction)
                
                # Step 2: Expand kernel to find better viewpoint
                if len(visible) > 0:
                    expanded_vp, visible = self.star_shaped.expand_kernel(vp, visible)
                    # Re-compute exact visibility from expanded point
                    avg_dir = np.mean(self.ray_caster.target_points[visible] - expanded_vp, axis=0)
                    avg_dir = avg_dir / norm(avg_dir)
                    visible, _ = self.ray_caster.compute_visibility(expanded_vp, avg_dir)
                
                new_coverage = len(set(visible) & uncovered)
                
                if new_coverage > best_score:
                    best_score = new_coverage
                    best_vp = expanded_vp if len(visible) > 0 else vp
                    best_visible = visible
                    best_time = comp_time
                    best_direction = direction
            
            if best_vp is None or best_score == 0:
                break
            
            uncovered -= set(best_visible)
            coverage = 1.0 - len(uncovered) / self.num_points
            
            selected_viewpoints.append(ViewpointResult(
                position=best_vp,
                direction=best_direction,
                visible_indices=best_visible,
                coverage_score=best_score / self.num_points,
                computation_time=best_time
            ))
            
            print(f"  Viewpoint {len(selected_viewpoints)}: +{best_score} points, "
                  f"coverage={coverage*100:.1f}%")
        
        total_time = get_time() - start_time
        coverage = 1.0 - len(uncovered) / self.num_points
        
        return OptimizationResult(
            method_name="Hybrid_RayCast_Kernel",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=self._compute_redundancy(selected_viewpoints)
        )
    
    def _compute_redundancy(self, viewpoints):
        coverage_count = np.zeros(self.num_points)
        for vp in viewpoints:
            coverage_count[vp.visible_indices] += 1
        return np.mean(coverage_count[coverage_count > 0])


class HybridMethod2:
    """Hybrid: Skeleton-guided candidates + ε-visibility (fast approximate)."""
    
    def __init__(self, mesh, target_points, frustum_params):
        self.pilp = ProgressiveILP(mesh, target_points, frustum_params)
        self.star_shaped = ApproximateStarShaped(mesh, target_points, frustum_params)
        self.num_points = len(target_points)
    
    def optimize(self, viewpoint_candidates, direction_candidates,
                 target_coverage=0.95, max_viewpoints=50):
        """Use skeleton as candidates but fast ε-visibility for checks."""
        start_time = get_time()
        
        # Use skeleton points as primary candidates
        skeleton_candidates = self.pilp.skeleton_points
        
        # Generate directions from skeleton to surface
        skeleton_directions = []
        for skel_pt in skeleton_candidates:
            direction = np.mean(self.star_shaped.target_points - skel_pt, axis=0)
            direction = direction / norm(direction)
            skeleton_directions.append(direction)
        skeleton_directions = np.array(skeleton_directions)
        
        # Combine with surface candidates
        combined_candidates = np.vstack([skeleton_candidates, viewpoint_candidates])
        combined_directions = np.vstack([skeleton_directions, direction_candidates])
        
        # Use approximate visibility for speed
        result = self.star_shaped.optimize(
            combined_candidates, combined_directions, target_coverage, max_viewpoints
        )
        result.method_name = "Hybrid_Skeleton_Epsilon"
        result.total_time = get_time() - start_time
        
        print(f"Hybrid Skeleton+Epsilon completed in {result.total_time:.2f}s")
        
        return result


class HybridMethod3:
    """Hybrid: Progressive coarse-to-fine with different visibility methods."""
    
    def __init__(self, mesh, target_points, frustum_params):
        self.mesh = mesh
        self.target_points = target_points
        self.frustum_params = frustum_params
        self.num_points = len(target_points)
        
        # Initialize all methods
        self.star_shaped = ApproximateStarShaped(mesh, target_points, frustum_params)
        self.ray_caster = RayCastingSetCover(mesh, target_points, frustum_params)
    
    def optimize(self, viewpoint_candidates, direction_candidates,
                 target_coverage=0.95, max_viewpoints=50):
        """Coarse selection with ε-visibility, fine-tune with ray casting."""
        start_time = get_time()
        
        print(f"Hybrid Progressive: Finding viewpoints for {target_coverage*100}% coverage...")
        
        # Phase 1: Coarse selection with ε-visibility (fast, 70% coverage)
        print("  Phase 1: Coarse selection with ε-visibility...")
        coarse_result = self.star_shaped.optimize(
            viewpoint_candidates, direction_candidates,
            target_coverage=0.7, max_viewpoints=max_viewpoints // 2
        )
        
        selected_viewpoints = coarse_result.viewpoints
        covered = set()
        for vp in selected_viewpoints:
            covered.update(vp.visible_indices)
        
        # Phase 2: Fine-tune remaining with exact ray casting
        print(f"  Phase 2: Fine-tuning with ray casting (current coverage: {len(covered)/self.num_points*100:.1f}%)...")
        uncovered = set(range(self.num_points)) - covered
        
        while len(uncovered) > (1 - target_coverage) * self.num_points and len(selected_viewpoints) < max_viewpoints:
            best_vp = None
            best_visible = None
            best_score = 0
            best_time = 0
            best_direction = None
            
            num_to_check = min(len(viewpoint_candidates), 30)
            check_indices = np.random.choice(
                len(viewpoint_candidates), num_to_check, replace=False
            )
            
            for idx in check_indices:
                vp = viewpoint_candidates[idx]
                direction = direction_candidates[idx]
                
                visible, comp_time = self.ray_caster.compute_visibility(vp, direction)
                new_coverage = len(set(visible) & uncovered)
                
                if new_coverage > best_score:
                    best_score = new_coverage
                    best_vp = vp
                    best_visible = visible
                    best_time = comp_time
                    best_direction = direction
            
            if best_vp is None or best_score == 0:
                break
            
            uncovered -= set(best_visible)
            coverage = 1.0 - len(uncovered) / self.num_points
            
            selected_viewpoints.append(ViewpointResult(
                position=best_vp,
                direction=best_direction,
                visible_indices=best_visible,
                coverage_score=best_score / self.num_points,
                computation_time=best_time
            ))
            
            print(f"    Viewpoint {len(selected_viewpoints)}: +{best_score} points, "
                  f"coverage={coverage*100:.1f}%")
        
        total_time = get_time() - start_time
        final_coverage = 1.0 - len(uncovered) / self.num_points
        
        return OptimizationResult(
            method_name="Hybrid_Progressive",
            viewpoints=selected_viewpoints,
            total_coverage=final_coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=self._compute_redundancy(selected_viewpoints)
        )
    
    def _compute_redundancy(self, viewpoints):
        coverage_count = np.zeros(self.num_points)
        for vp in viewpoints:
            coverage_count[vp.visible_indices] += 1
        return np.mean(coverage_count[coverage_count > 0])


class HybridMethod4:
    """Hybrid: Two-phase with quality-weighted selection."""
    
    def __init__(self, mesh, target_points, frustum_params):
        self.star_shaped = ApproximateStarShaped(mesh, target_points, frustum_params)
        self.ray_caster = RayCastingSetCover(mesh, target_points, frustum_params)
        self.num_points = len(target_points)
    
    def compute_quality_score(self, viewpoint, visible_indices):
        """Compute quality score based on coverage + compactness."""
        if len(visible_indices) == 0:
            return 0.0
        
        visible_points = self.star_shaped.target_points[visible_indices]
        
        # Coverage component
        coverage = len(visible_indices) / self.num_points
        
        # Compactness component (lower variance = more compact)
        centroid = np.mean(visible_points, axis=0)
        distances = norm(visible_points - centroid, axis=1)
        compactness = 1.0 / (1.0 + np.std(distances))
        
        # Combined score
        return 0.7 * coverage + 0.3 * compactness
    
    def optimize(self, viewpoint_candidates, direction_candidates,
                 target_coverage=0.95, max_viewpoints=50):
        """Select viewpoints based on quality-weighted scoring."""
        start_time = get_time()
        
        uncovered = set(range(self.num_points))
        selected_viewpoints = []
        
        print(f"Hybrid Quality-Weighted: Finding viewpoints for {target_coverage*100}% coverage...")
        
        while len(uncovered) > (1 - target_coverage) * self.num_points and len(selected_viewpoints) < max_viewpoints:
            best_vp = None
            best_visible = None
            best_score = 0
            best_time = 0
            best_direction = None
            
            num_to_check = min(len(viewpoint_candidates), 40)
            check_indices = np.random.choice(
                len(viewpoint_candidates), num_to_check, replace=False
            )
            
            for idx in check_indices:
                vp = viewpoint_candidates[idx]
                direction = direction_candidates[idx]
                
                # Use ε-visibility for speed
                visible, comp_time = self.star_shaped.compute_epsilon_visibility(vp, direction)
                
                if len(visible) == 0:
                    continue
                
                # Compute quality-weighted score
                new_coverage_count = len(set(visible) & uncovered)
                quality = self.compute_quality_score(vp, visible)
                combined_score = new_coverage_count * quality
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_vp = vp
                    best_visible = visible
                    best_time = comp_time
                    best_direction = direction
            
            if best_vp is None:
                break
            
            uncovered -= set(best_visible)
            coverage = 1.0 - len(uncovered) / self.num_points
            
            selected_viewpoints.append(ViewpointResult(
                position=best_vp,
                direction=best_direction,
                visible_indices=best_visible,
                coverage_score=len(best_visible) / self.num_points,
                computation_time=best_time
            ))
            
            print(f"  Viewpoint {len(selected_viewpoints)}: +{len(set(best_visible) & (set(range(self.num_points)) - (set(range(self.num_points)) - uncovered - set(best_visible))))} points, "
                  f"coverage={coverage*100:.1f}%")
        
        total_time = get_time() - start_time
        final_coverage = 1.0 - len(uncovered) / self.num_points
        
        return OptimizationResult(
            method_name="Hybrid_Quality_Weighted",
            viewpoints=selected_viewpoints,
            total_coverage=final_coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=self._compute_redundancy(selected_viewpoints)
        )
    
    def _compute_redundancy(self, viewpoints):
        coverage_count = np.zeros(self.num_points)
        for vp in viewpoints:
            coverage_count[vp.visible_indices] += 1
        return np.mean(coverage_count[coverage_count > 0])

# ============================================================================
# BENCHMARKING AND ANALYSIS
# ============================================================================

class BenchmarkSuite:
    """Comprehensive benchmarking of all methods."""
    
    def __init__(self, mesh, num_target_points=10000, num_candidates=200):
        self.mesh = mesh
        self.num_target_points = num_target_points
        self.num_candidates = num_candidates
        
        # Sample target points
        print(f"Sampling {num_target_points} target points...")
        target_pcd = mesh.sample_points_uniformly(number_of_points=num_target_points)
        self.target_points = np.asarray(target_pcd.points)
        
        # Generate viewpoint candidates
        print(f"Generating {num_candidates} viewpoint candidates...")
        self.viewpoint_candidates, self.direction_candidates = sample_viewpoint_candidates(
            mesh, num_candidates, distance=3.0
        )
        
        self.frustum_params = FrustumParams()
        self.results = {}
    
    def run_all_methods(self, target_coverage=0.95, max_viewpoints=30, 
                       visualize=False, visualize_heatmap=False):
        """Run all methods and collect results."""
        methods = {
            "RayCasting_SetCover": lambda: RayCastingSetCover(
                self.mesh, self.target_points, self.frustum_params
            ).solve_set_cover(  # Changed from greedy_set_cover
                self.viewpoint_candidates, self.direction_candidates,
                target_coverage, max_viewpoints
            ),
            "Approximate_StarShaped": lambda: ApproximateStarShaped(
                self.mesh, self.target_points, self.frustum_params
            ).optimize(
                self.viewpoint_candidates, self.direction_candidates,
                target_coverage, max_viewpoints
            ),
            "Progressive_ILP": lambda: ProgressiveILP(
                self.mesh, self.target_points, self.frustum_params
            ).progressive_optimization(target_coverage, max_viewpoints),
            "Hybrid_RayCast_Kernel": lambda: HybridMethod1(
                self.mesh, self.target_points, self.frustum_params
            ).optimize(
                self.viewpoint_candidates, self.direction_candidates,
                target_coverage, max_viewpoints
            ),
            "Hybrid_Skeleton_Epsilon": lambda: HybridMethod2(
                self.mesh, self.target_points, self.frustum_params
            ).optimize(
                self.viewpoint_candidates, self.direction_candidates,
                target_coverage, max_viewpoints
            ),
            "Hybrid_Progressive": lambda: HybridMethod3(
                self.mesh, self.target_points, self.frustum_params
            ).optimize(
                self.viewpoint_candidates, self.direction_candidates,
                target_coverage, max_viewpoints
            ),
            "Hybrid_Quality_Weighted": lambda: HybridMethod4(
                self.mesh, self.target_points, self.frustum_params
            ).optimize(
                self.viewpoint_candidates, self.direction_candidates,
                target_coverage, max_viewpoints
            ),
        }
        
        for method_name, method_func in methods.items():
            print(f"\n{'='*60}")
            print(f"Running: {method_name}")
            print(f"{'='*60}")
            
            try:
                result = method_func()
                self.results[method_name] = result
                
                print(f"\nResult Summary:")
                print(f"  Coverage: {result.total_coverage*100:.2f}%")
                print(f"  Viewpoints: {result.num_viewpoints}")
                print(f"  Time: {result.total_time:.2f}s")
                if result.num_viewpoints > 0:
                    print(f"  Efficiency: {result.total_coverage*100/result.num_viewpoints:.2f}% per viewpoint")
                print(f"  Redundancy: {result.redundancy:.2f}x")
                
                # Visualize if requested
                if visualize and result.num_viewpoints > 0:
                    print(f"\n  Opening visualization window...")
                    visualize_solution(result, self.target_points, self.mesh, 
                                     self.frustum_params, f"{method_name} Solution")
                    
                    if visualize_heatmap:
                        print(f"  Opening heatmap window...")
                        visualize_coverage_heatmap(result, self.target_points, self.mesh,
                                                 f"{method_name} Coverage")
                
            except Exception as e:
                print(f"ERROR in {method_name}: {e}")
                import traceback
                traceback.print_exc()
    
    def generate_comparison_plots(self, output_dir="benchmark_results"):
        """Generate comprehensive comparison plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            print("No results to plot!")
            return
        
        # Extract data
        methods = list(self.results.keys())
        coverages = [self.results[m].total_coverage * 100 for m in methods]
        num_viewpoints = [self.results[m].num_viewpoints for m in methods]
        times = [self.results[m].total_time for m in methods]
        redundancies = [self.results[m].redundancy for m in methods]
        
        # Efficiency metrics
        coverage_per_vp = [c / n if n > 0 else 0 for c, n in zip(coverages, num_viewpoints)]
        time_per_vp = [t / n if n > 0 else 0 for t, n in zip(times, num_viewpoints)]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Coverage comparison
        ax1 = plt.subplot(2, 3, 1)
        bars = ax1.bar(range(len(methods)), coverages, color='steelblue')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Coverage (%)', fontsize=12)
        ax1.set_title('Total Coverage Achieved', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=95, color='r', linestyle='--', label='Target (95%)')
        ax1.legend()
        
        # 2. Number of viewpoints
        ax2 = plt.subplot(2, 3, 2)
        bars = ax2.bar(range(len(methods)), num_viewpoints, color='coral')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Number of Viewpoints', fontsize=12)
        ax2.set_title('Viewpoints Required', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Computation time
        ax3 = plt.subplot(2, 3, 3)
        bars = ax3.bar(range(len(methods)), times, color='lightgreen')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Time (seconds)', fontsize=12)
        ax3.set_title('Computation Time', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Coverage per viewpoint (efficiency)
        ax4 = plt.subplot(2, 3, 4)
        bars = ax4.bar(range(len(methods)), coverage_per_vp, color='mediumpurple')
        ax4.set_xticks(range(len(methods)))
        ax4.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Coverage % per Viewpoint', fontsize=12)
        ax4.set_title('Efficiency (Coverage/Viewpoint)', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Redundancy
        ax5 = plt.subplot(2, 3, 5)
        bars = ax5.bar(range(len(methods)), redundancies, color='gold')
        ax5.set_xticks(range(len(methods)))
        ax5.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('Average Redundancy', fontsize=12)
        ax5.set_title('Coverage Redundancy', fontsize=14, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        ax5.axhline(y=1.0, color='g', linestyle='--', label='No redundancy')
        ax5.legend()
        
        # 6. Efficiency vs Speed tradeoff
        ax6 = plt.subplot(2, 3, 6)
        scatter = ax6.scatter(times, coverages, s=200, c=num_viewpoints, cmap='viridis', alpha=0.7)
        for i, method in enumerate(methods):
            ax6.annotate(method.replace('_', '\n'), (times[i], coverages[i]),
                        fontsize=7, ha='center', va='bottom')
        ax6.set_xlabel('Computation Time (s)', fontsize=12)
        ax6.set_ylabel('Coverage (%)', fontsize=12)
        ax6.set_title('Efficiency vs Speed Tradeoff', fontsize=14, fontweight='bold')
        ax6.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('# Viewpoints', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_plots.png', dpi=150, bbox_inches='tight')
        print(f"\nComparison plots saved to {output_dir}/comparison_plots.png")
        plt.show()
    
    def print_summary_table(self):
        """Print detailed comparison table."""
        if not self.results:
            print("No results to display!")
            return
        
        print("\n" + "="*120)
        print("BENCHMARK SUMMARY TABLE")
        print("="*120)
        print(f"{'Method':<30} {'Coverage':<12} {'Viewpoints':<12} {'Time(s)':<10} {'Cov/VP':<10} {'Redundancy':<12}")
        print("-"*120)
        
        for method_name, result in self.results.items():
            cov_per_vp = result.total_coverage * 100 / result.num_viewpoints if result.num_viewpoints > 0 else 0
            print(f"{method_name:<30} {result.total_coverage*100:>10.2f}% "
                  f"{result.num_viewpoints:>11} {result.total_time:>9.2f} "
                  f"{cov_per_vp:>9.2f}% {result.redundancy:>11.2f}x")
        
        print("="*120)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark viewpoint optimization methods")
    parser.add_argument('--mesh_path', type=str, default=None,
                       help="Path to mesh file")
    parser.add_argument('--num_points', type=int, default=5000,
                       help="Number of target points to sample")
    parser.add_argument('--num_candidates', type=int, default=150,
                       help="Number of viewpoint candidates")
    parser.add_argument('--target_coverage', type=float, default=0.95,
                       help="Target coverage percentage")
    parser.add_argument('--max_viewpoints', type=int, default=25,
                       help="Maximum number of viewpoints")
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help="Specific methods to run (default: all)")
    parser.add_argument('--visualize', action='store_true',
                       help="Visualize solutions (shows each method's result)")
    parser.add_argument('--visualize_heatmap', action='store_true',
                       help="Also show coverage heatmap (requires --visualize)")
    parser.add_argument('--no_plots', action='store_true',
                       help="Skip generating matplotlib comparison plots")
    
    args = parser.parse_args()
    
    # Load or create mesh
    if args.mesh_path:
        print(f"Loading mesh from: {args.mesh_path}")
        mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    else:
        print("Using default mesh (Sphere)")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    
    mesh.compute_vertex_normals()
    
    # Run benchmark
    print("\n" + "="*60)
    print("STARTING BENCHMARK SUITE")
    print("="*60)
    print(f"Target points: {args.num_points}")
    print(f"Viewpoint candidates: {args.num_candidates}")
    print(f"Target coverage: {args.target_coverage*100}%")
    print(f"Max viewpoints: {args.max_viewpoints}")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    if args.visualize and args.visualize_heatmap:
        print(f"Heatmap: Enabled")
    
    benchmark = BenchmarkSuite(mesh, args.num_points, args.num_candidates)
    benchmark.run_all_methods(args.target_coverage, args.max_viewpoints, 
                             visualize=args.visualize,
                             visualize_heatmap=args.visualize_heatmap)
    
    # Generate analysis
    benchmark.print_summary_table()
    
    if not args.no_plots:
        benchmark.generate_comparison_plots()
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()