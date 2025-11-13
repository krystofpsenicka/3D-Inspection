import open3d as o3d
import open3d.core as o3c
import numpy as np
from numpy.linalg import norm
import argparse
import os
from scipy.spatial import KDTree
import time

# --- 1. Frustum Definition Parameters (Static) ---
FRUSTUM_FOV_DEGREE = 30.0
FRUSTUM_NEAR_D = 1e-6 
FRUSTUM_FAR_D = 4.5
MAX_DIRECTION_OFFSET = 0.1

def get_frustum_basis(direction):
    """Calculates the Up and Right vectors for the frustum."""
    direction = direction / norm(direction)
    
    if 1.0 - np.abs(direction[2]) < 1e-6: 
        temp_up = np.array([1.0, 0.0, 0.0])
    else:
        temp_up = np.array([0.0, 0.0, 1.0])
        
    right = np.cross(direction, temp_up)
    r_norm = norm(right)
    if r_norm < 1e-6:
        right = np.array([direction[1], -direction[0], 0])
        r_norm = norm(right)
    right = right / r_norm
    
    up = np.cross(right, direction)
    up = up / norm(up)
    
    return right, up

def get_frustum_corners(viewpoint, direction, fov_deg, near_d, far_d):
    """Calculates the 8 corners of the frustum volume."""
    half_angle_rad = np.deg2rad(fov_deg / 2.0)
    near_half_size = near_d * np.tan(half_angle_rad)
    far_half_size = far_d * np.tan(half_angle_rad)
    
    right, up = get_frustum_basis(direction)
    
    corners = []
    for distance, half_size in [(near_d, near_half_size), (far_d, far_half_size)]:
        center = viewpoint + distance * direction
        r = right * half_size
        u = up * half_size
        
        corners.append(center + r + u)
        corners.append(center - r + u)
        corners.append(center - r - u)
        corners.append(center + r - u)
        
    return np.array(corners)

def get_frustum_mask(points, viewpoint, direction, fov_deg, near_d, far_d):
    """Vectorized check to find which points are inside the frustum volume."""
    TOLERANCE = 1e-4

    vp_vectors = points - viewpoint
    vp_lengths = np.clip(norm(vp_vectors, axis=1), 1e-6, None) 
    vp_norm = vp_vectors / vp_lengths[:, np.newaxis]
    
    proj_distance = np.dot(vp_vectors, direction)
    distance_mask = (proj_distance >= near_d - TOLERANCE) & (proj_distance <= far_d + TOLERANCE)
    
    right, up = get_frustum_basis(direction)
    half_angle_rad = np.deg2rad(fov_deg / 2.0)
    tan_half_angle = np.tan(half_angle_rad)
    max_size = proj_distance * tan_half_angle

    lateral_right = np.dot(vp_vectors, right)
    lateral_up = np.dot(vp_vectors, up)

    right_mask = np.abs(lateral_right) < max_size + TOLERANCE
    up_mask = np.abs(lateral_up) < max_size + TOLERANCE
    angular_mask = right_mask & up_mask
    
    return distance_mask & angular_mask

def get_frustum_bounding_sphere(viewpoint, direction, fov_deg, near_d, far_d):
    """
    Calculate a bounding sphere that contains the entire frustum.
    This is used for initial spatial queries.
    """
    # Get all 8 corners of the frustum
    corners = get_frustum_corners(viewpoint, direction, fov_deg, near_d, far_d)
    
    # Include the viewpoint (apex) as well
    all_points = np.vstack([viewpoint.reshape(1, 3), corners])
    
    # Calculate center as the mean of all points
    center = np.mean(all_points, axis=0)
    
    # Calculate radius as the maximum distance from center to any point
    distances = norm(all_points - center, axis=1)
    radius = np.max(distances)
    
    return center, radius

class SpatialIndex:
    """Wrapper for spatial indexing with KD-tree."""
    
    def __init__(self, points):
        """
        Initialize the spatial index with points.
        
        Args:
            points: Nx3 numpy array of 3D points
        """
        print(f"Building KD-tree for {len(points)} points...")
        start = time.time()
        self.points = points
        self.tree = KDTree(points)
        print(f"KD-tree built in {time.time() - start:.3f} seconds")
    
    def query_frustum(self, viewpoint, direction, fov_deg, near_d, far_d):
        """
        Query points that might be in the frustum using spatial indexing.
        
        Returns:
            candidate_indices: Indices of points that might be in frustum
            candidate_points: The actual point coordinates
        """
        # Step 1: Get bounding sphere for the frustum
        center, radius = get_frustum_bounding_sphere(viewpoint, direction, fov_deg, near_d, far_d)
        
        # Step 2: Query all points within the bounding sphere
        candidate_indices = self.tree.query_ball_point(center, radius)
        
        if len(candidate_indices) == 0:
            return np.array([], dtype=int), np.array([]).reshape(0, 3)
        
        candidate_points = self.points[candidate_indices]
        
        # Step 3: Refine with exact frustum test
        in_frustum_mask = get_frustum_mask(
            candidate_points, viewpoint, direction, fov_deg, near_d, far_d
        )
        
        # Filter to only points actually in frustum
        final_indices = np.array(candidate_indices)[in_frustum_mask]
        final_points = candidate_points[in_frustum_mask]
        
        return final_indices, final_points

def generate_viewpoints(num_viewpoints, mesh, distance_from_surface=1.0, seed=None):
    """
    Generate viewpoints distributed evenly around the mesh surface.
    Places viewpoints at a fixed distance from sampled surface points, pointing inward.
    
    Args:
        num_viewpoints: Number of viewpoints to generate
        mesh: Open3D triangle mesh
        distance_from_surface: Distance to place viewpoints from the surface
        seed: Random seed for reproducibility
    
    Returns:
        viewpoints: Nx3 array of viewpoint positions
        directions: Nx3 array of view directions
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample points uniformly on the mesh surface
    surface_pcd = mesh.sample_points_uniformly(number_of_points=num_viewpoints)
    surface_points = np.asarray(surface_pcd.points)
    surface_normals = np.asarray(surface_pcd.normals)
    
    viewpoints = []
    directions = []
    
    for i in range(num_viewpoints):
        # Place viewpoint along the outward normal at distance_from_surface
        viewpoint = surface_points[i] + surface_normals[i] * distance_from_surface
        
        # Base direction points back toward the surface point
        base_direction = -surface_normals[i]
        
        # Add small random offset to avoid all viewpoints being perfectly perpendicular
        right, up = get_frustum_basis(base_direction)
        r_offset = np.random.uniform(-MAX_DIRECTION_OFFSET, MAX_DIRECTION_OFFSET)
        u_offset = np.random.uniform(-MAX_DIRECTION_OFFSET, MAX_DIRECTION_OFFSET)
        
        direction = base_direction + r_offset * right + u_offset * up
        direction /= norm(direction)
        
        viewpoints.append(viewpoint)
        directions.append(direction)
    
    return np.array(viewpoints), np.array(directions)

def batch_visibility_check(spatial_index, scene, object_id, viewpoints, directions, 
                           fov_deg, near_d, far_d, device, dtype):
    """
    Perform visibility checks for multiple viewpoints efficiently.
    
    Args:
        spatial_index: SpatialIndex object with target points
        scene: Open3D raycasting scene
        object_id: ID of the object in the scene
        viewpoints: Nx3 array of viewpoint positions
        directions: Nx3 array of view directions
        fov_deg, near_d, far_d: Frustum parameters
        device, dtype: Open3D device and data type
    
    Returns:
        results: List of dicts, each containing visibility info for one viewpoint
    """
    results = []
    
    print(f"\nProcessing {len(viewpoints)} viewpoints...")
    start_time = time.time()
    
    for i, (viewpoint, direction) in enumerate(zip(viewpoints, directions)):
        # Step 1: Query spatial index for candidate points
        candidate_indices, frustum_points = spatial_index.query_frustum(
            viewpoint, direction, fov_deg, near_d, far_d
        )
        
        if len(frustum_points) == 0:
            results.append({
                'viewpoint': viewpoint,
                'direction': direction,
                'num_in_frustum': 0,
                'num_visible': 0,
                'num_occluded': 0,
                'visible_indices': np.array([], dtype=int),
                'occluded_indices': np.array([], dtype=int)
            })
            continue
        
        # Step 2: Raycast to check occlusion
        num_frustum_points = len(frustum_points)
        ray_origins = np.tile(viewpoint, (num_frustum_points, 1))
        ray_directions = frustum_points - ray_origins
        ray_lengths = norm(ray_directions, axis=1)
        ray_directions /= ray_lengths[:, np.newaxis]
        
        rays = o3c.Tensor(
            np.hstack([ray_origins, ray_directions]), 
            dtype=dtype, 
            device=device
        )
        
        ans = scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy()
        geometry_ids = ans['geometry_ids'].numpy()
        
        # Step 3: Determine visibility
        epsilon = 1e-5
        hit_object_mask = (geometry_ids == object_id)
        distance_match_mask = np.abs(t_hit - ray_lengths) < epsilon
        visible_mask = hit_object_mask & distance_match_mask
        
        visible_indices = candidate_indices[visible_mask]
        occluded_indices = candidate_indices[~visible_mask]
        
        results.append({
            'viewpoint': viewpoint,
            'direction': direction,
            'num_in_frustum': num_frustum_points,
            'num_visible': len(visible_indices),
            'num_occluded': len(occluded_indices),
            'visible_indices': visible_indices,
            'occluded_indices': occluded_indices
        })
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"Processed {i + 1}/{len(viewpoints)} viewpoints ({rate:.1f} viewpoints/sec)")
    
    total_time = time.time() - start_time
    print(f"\nCompleted all viewpoints in {total_time:.2f} seconds ({len(viewpoints)/total_time:.1f} viewpoints/sec)")
    
    return results

def print_summary_statistics(results):
    """Print summary statistics from batch visibility results."""
    total_viewpoints = len(results)
    
    viewpoints_with_visible = sum(1 for r in results if r['num_visible'] > 0)
    total_visible = sum(r['num_visible'] for r in results)
    total_in_frustum = sum(r['num_in_frustum'] for r in results)
    total_occluded = sum(r['num_occluded'] for r in results)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total viewpoints processed: {total_viewpoints}")
    print(f"Viewpoints with visible points: {viewpoints_with_visible} ({100*viewpoints_with_visible/total_viewpoints:.1f}%)")
    print(f"\nTotal points in frustums: {total_in_frustum}")
    print(f"Total visible points: {total_visible} ({100*total_visible/max(1,total_in_frustum):.1f}%)")
    print(f"Total occluded points: {total_occluded} ({100*total_occluded/max(1,total_in_frustum):.1f}%)")
    
    if total_viewpoints > 0:
        avg_in_frustum = total_in_frustum / total_viewpoints
        avg_visible = total_visible / total_viewpoints
        print(f"\nAverage points in frustum per viewpoint: {avg_in_frustum:.1f}")
        print(f"Average visible points per viewpoint: {avg_visible:.1f}")

def run_batch_visibility():
    """Main function for batch visibility analysis."""
    parser = argparse.ArgumentParser(description="Run batch 3D visibility check with spatial indexing.")
    parser.add_argument('--mesh_path', type=str, default=None, help="Path to mesh file (.glb, .ply, .obj)")
    parser.add_argument('--num_points', type=int, default=1000000, help="Number of points to sample from mesh")
    parser.add_argument('--num_viewpoints', type=int, default=200, help="Number of viewpoints to test")
    parser.add_argument('--viewpoint_distance', type=float, default=3.0, help="Distance from mesh surface to place viewpoints")
    parser.add_argument('--visualize_first', action='store_true', help="Visualize the first viewpoint only")
    parser.add_argument('--visualize_all', action='store_true', help="Visualize all viewpoints and results")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    dtype = o3c.float32
    device = o3c.Device("CPU:0")

    # --- 1. Load or Create Mesh ---
    object_mesh = None
    if args.mesh_path and os.path.exists(args.mesh_path):
        print(f"Loading mesh from: {args.mesh_path}")
        object_mesh = o3d.io.read_triangle_mesh(args.mesh_path)
        if not object_mesh.has_vertices() or not object_mesh.has_triangles():
            print(f"Warning: Failed to load mesh. Using Icosahedron.")
            object_mesh = None
    
    if object_mesh is None:
        print("Using default Icosahedron geometry.")
        object_mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.2)
    
    object_mesh.compute_vertex_normals()
    object_t = o3d.t.geometry.TriangleMesh.from_legacy(object_mesh, device=device)

    # --- 2. Setup Raycasting Scene ---
    scene = o3d.t.geometry.RaycastingScene(device=device)
    object_id = scene.add_triangles(object_t)

    # --- 3. Sample Points and Build Spatial Index ---
    print(f"\nSampling {args.num_points} points from mesh...")
    target_pcd = object_mesh.sample_points_uniformly(number_of_points=args.num_points)
    target_points = np.asarray(target_pcd.points)
    
    spatial_index = SpatialIndex(target_points)

    # --- 4. Generate Viewpoints ---
    print(f"\nGenerating {args.num_viewpoints} viewpoints around mesh...")
    viewpoints, directions = generate_viewpoints(
        args.num_viewpoints, 
        object_mesh, 
        distance_from_surface=args.viewpoint_distance,
        seed=args.seed
    )

    # --- 5. Batch Visibility Check ---
    results = batch_visibility_check(
        spatial_index, scene, object_id, viewpoints, directions,
        FRUSTUM_FOV_DEGREE, FRUSTUM_NEAR_D, FRUSTUM_FAR_D, device, dtype
    )

    # --- 6. Print Statistics ---
    print_summary_statistics(results)

    # --- 7. Visualization ---
    if args.visualize_all:
        print("\nVisualizing all viewpoints and results...")
        visualize_all_results(results, target_points, object_mesh)
    elif args.visualize_first and len(results) > 0:
        print("\nVisualizing first viewpoint...")
        visualize_single_result(results[0], target_points, object_mesh)

def get_frustum_lineset(viewpoint, direction, fov_deg, near_d, far_d):
    """Helper function to create a LineSet for visualizing the frustum."""
    corners_8 = get_frustum_corners(viewpoint, direction, fov_deg, near_d, far_d)
    
    points_viz = [viewpoint] 
    points_viz.extend(corners_8[4:])
    points_viz = np.array(points_viz)
    
    lines = [
        [1, 2], [2, 3], [3, 4], [4, 1],
        [0, 1], [0, 2], [0, 3], [0, 4]
    ]
    
    colors = [[1, 0.5, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_viz),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def visualize_single_result(result, target_points, object_mesh):
    """Visualize a single viewpoint result."""
    # Get points for visualization
    visible_indices = result['visible_indices']
    occluded_indices = result['occluded_indices']
    
    visible_points = target_points[visible_indices]
    occluded_points = target_points[occluded_indices]
    
    # Create geometries
    context_mesh = o3d.geometry.TriangleMesh(object_mesh)
    context_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    
    frustum_lines = get_frustum_lineset(
        result['viewpoint'], result['direction'],
        FRUSTUM_FOV_DEGREE, FRUSTUM_NEAR_D, FRUSTUM_FAR_D
    )
    
    viewpoint_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    viewpoint_marker.translate(result['viewpoint'])
    viewpoint_marker.paint_uniform_color([0, 1, 0])
    
    visible_pcd = o3d.geometry.PointCloud()
    if len(visible_points) > 0:
        visible_pcd.points = o3d.utility.Vector3dVector(visible_points)
        visible_pcd.paint_uniform_color([0.1, 1.0, 0.1])
    
    occluded_pcd = o3d.geometry.PointCloud()
    if len(occluded_points) > 0:
        occluded_pcd.points = o3d.utility.Vector3dVector(occluded_points)
        occluded_pcd.paint_uniform_color([1.0, 0.1, 0.1])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    
    for geom in [context_mesh, frustum_lines, viewpoint_marker, visible_pcd, occluded_pcd]:
        vis.add_geometry(geom)
    
    vis.run()
    vis.destroy_window()

def visualize_all_results(results, target_points, object_mesh):
    """Visualize all viewpoints and their visibility results."""
    print(f"Creating visualization for {len(results)} viewpoints...")
    
    # Create base mesh
    context_mesh = o3d.geometry.TriangleMesh(object_mesh)
    context_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    
    geometries = [context_mesh]
    
    # Collect all visible and occluded points across all viewpoints
    all_visible_indices = set()
    all_occluded_indices = set()
    
    for result in results:
        all_visible_indices.update(result['visible_indices'])
        all_occluded_indices.update(result['occluded_indices'])
    
    # Remove overlap (if a point is visible from any viewpoint, mark it as visible)
    all_occluded_indices -= all_visible_indices
    
    # Create point clouds
    if len(all_visible_indices) > 0:
        visible_points = target_points[list(all_visible_indices)]
        visible_pcd = o3d.geometry.PointCloud()
        visible_pcd.points = o3d.utility.Vector3dVector(visible_points)
        visible_pcd.paint_uniform_color([0.1, 1.0, 0.1])  # Bright green
        geometries.append(visible_pcd)
        print(f"Points visible from at least one viewpoint: {len(all_visible_indices)}")
    
    if len(all_occluded_indices) > 0:
        occluded_points = target_points[list(all_occluded_indices)]
        occluded_pcd = o3d.geometry.PointCloud()
        occluded_pcd.points = o3d.utility.Vector3dVector(occluded_points)
        occluded_pcd.paint_uniform_color([1.0, 0.1, 0.1])  # Bright red
        geometries.append(occluded_pcd)
        print(f"Points never visible from any viewpoint: {len(all_occluded_indices)}")
    
    # Points never in any frustum
    all_tested_indices = all_visible_indices | all_occluded_indices
    never_tested_indices = set(range(len(target_points))) - all_tested_indices
    if len(never_tested_indices) > 0:
        never_tested_points = target_points[list(never_tested_indices)]
        never_tested_pcd = o3d.geometry.PointCloud()
        never_tested_pcd.points = o3d.utility.Vector3dVector(never_tested_points)
        never_tested_pcd.paint_uniform_color([0.3, 0.3, 0.3])  # Gray
        geometries.append(never_tested_pcd)
        print(f"Points never in any frustum: {len(never_tested_indices)}")
    
    # Add all frustum wireframes (simplified to avoid clutter)
    print("Adding frustum wireframes...")
    for i, result in enumerate(results):
        frustum_lines = get_frustum_lineset(
            result['viewpoint'], result['direction'],
            FRUSTUM_FOV_DEGREE, FRUSTUM_NEAR_D, FRUSTUM_FAR_D
        )
        # Make frustums semi-transparent by using a lighter color
        frustum_lines.paint_uniform_color([1.0, 0.7, 0.3])  # Light orange
        geometries.append(frustum_lines)
        
        if (i + 1) % 100 == 0:
            print(f"  Added {i + 1}/{len(results)} frustums")
    
    # Add all viewpoint markers
    print("Adding viewpoint markers...")
    for i, result in enumerate(results):
        viewpoint_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        viewpoint_marker.translate(result['viewpoint'])
        
        # Color viewpoints based on how many points they can see
        if result['num_visible'] == 0:
            color = [0.5, 0.5, 0.5]  # Gray for no visibility
        elif result['num_visible'] < 100:
            color = [1.0, 1.0, 0.0]  # Yellow for low visibility
        else:
            color = [0.0, 1.0, 0.0]  # Green for good visibility
        
        viewpoint_marker.paint_uniform_color(color)
        geometries.append(viewpoint_marker)
        
        if (i + 1) % 100 == 0:
            print(f"  Added {i + 1}/{len(results)} viewpoints")
    
    print("\nLaunching visualizer...")
    print("Legend:")
    print("  - Bright Green points: Visible from at least one viewpoint")
    print("  - Bright Red points: In frustum but occluded from all viewpoints")
    print("  - Gray points: Never in any frustum")
    print("  - Green spheres: Viewpoints with good visibility (>100 points)")
    print("  - Yellow spheres: Viewpoints with low visibility (<100 points)")
    print("  - Gray spheres: Viewpoints with no visibility")
    print("  - Light Orange wireframes: Frustums")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="All Viewpoints Visibility Analysis", width=1920, height=1080)
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.line_width = 1.0
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    run_batch_visibility()