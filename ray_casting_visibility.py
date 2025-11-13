import open3d as o3d
import open3d.core as o3c
import numpy as np
from numpy.linalg import norm
import math
import argparse # New import for command line arguments
import os # New import for file path checking

# --- 1. Frustum Definition Parameters (Static) ---
# The target object (Icosahedron, radius 1.2) is centered around the origin [0,0,0].
FRUSTUM_FOV_DEGREE = 30.0    # Total horizontal/vertical angle of the pyramidal frustum
# Near clipping distance set to a small epsilon (1e-6) to make the pyramid peak
FRUSTUM_NEAR_D = 1e-6 
FRUSTUM_FAR_D = 4.5          # Far clipping distance from the viewpoint (must be > NEAR_D)

# --- CONSTANT FOR RANDOMNESS CONTROL ---
# Maximum magnitude for the proportional offset applied to the base direction.
# This ensures the frustum is still likely pointing at the object for reasonable results.
MAX_DIRECTION_OFFSET = 0.1 

# Global variables for dynamic setup
VIEWPOINT_POS = None
FRUSTUM_DIRECTION = None

def get_frustum_basis(direction):
    """
    Calculates the Up and Right vectors for the frustum, orthogonal to the direction.
    Uses Gram-Schmidt-like process for robustness.
    """
    direction = direction / norm(direction)
    
    # 1. Define a temporary up vector (temp_up) that is NOT parallel to the direction.
    # If the direction is close to the Z-axis, use the X-axis as temp_up to avoid issues.
    if 1.0 - np.abs(direction[2]) < 1e-6: 
        temp_up = np.array([1.0, 0.0, 0.0])
    else:
        temp_up = np.array([0.0, 0.0, 1.0])
        
    # 2. Calculate Right vector (orthogonal to Direction and Temp Up)
    right = np.cross(direction, temp_up)
    
    # Robust normalization for right vector
    r_norm = norm(right)
    if r_norm < 1e-6:
        # Fallback for an extremely rare edge case (should be unreachable)
        right = np.array([direction[1], -direction[0], 0])
        r_norm = norm(right)
        
    right = right / r_norm
    
    # 3. Calculate true Up vector (orthogonal to Direction and Right)
    up = np.cross(right, direction)
    up = up / norm(up)
    
    return right, up

def get_frustum_corners(viewpoint, direction, fov_deg, near_d, far_d):
    """Calculates the 8 corners of the frustum volume."""
    
    # Half angle and distances
    half_angle_rad = np.deg2rad(fov_deg / 2.0)
    
    # Calculate half-height/width at near and far planes
    near_half_size = near_d * np.tan(half_angle_rad)
    far_half_size = far_d * np.tan(half_angle_rad)
    
    # Get the frustum basis vectors
    right, up = get_frustum_basis(direction)
    
    corners = []
    # Corners are defined in the order: [0-3 Near Plane, 4-7 Far Plane]
    for distance, half_size in [(near_d, near_half_size), (far_d, far_half_size)]:
        # Center of the plane
        center = viewpoint + distance * direction
        
        # Corners of the plane
        r = right * half_size
        u = up * half_size
        
        # Corners definition (+r/+u, -r/+u, -r/-u, +r/-u)
        corners.append(center + r + u)
        corners.append(center - r + u)
        corners.append(center - r - u)
        corners.append(center + r - u)
        
    return np.array(corners)

def get_frustum_mask(points, viewpoint, direction, fov_deg, near_d, far_d):
    """
    Vectorized check to find which points are inside the frustum volume.
    Applies tolerance to boundary checks to handle floating point error.
    
    Crucially, this performs a PYRAMIDAL (rectangular) check, not a cone check.
    """
    # Tolerance for geometric checks (1 part in 10,000)
    TOLERANCE = 1e-4

    # 1. Vector from Viewpoint (V) to Point (P)
    vp_vectors = points - viewpoint
    # Clip length to prevent division by zero for points exactly at viewpoint
    vp_lengths = np.clip(norm(vp_vectors, axis=1), 1e-6, None) 
    
    # Normalize the vectors
    vp_norm = vp_vectors / vp_lengths[:, np.newaxis]
    
    # 2. Distance Check (Projection onto the view direction)
    proj_distance = np.dot(vp_vectors, direction)
    
    # Apply tolerance: Near boundary check (>= NEAR - TOL) and Far boundary check (<= FAR + TOL)
    distance_mask = (proj_distance >= near_d - TOLERANCE) & (proj_distance <= far_d + TOLERANCE)
    
    # --- 3. PYRAMIDAL (Rectangular) Angular Check ---
    
    right, up = get_frustum_basis(direction)

    # Calculate half angle and tangent
    half_angle_rad = np.deg2rad(fov_deg / 2.0)
    tan_half_angle = np.tan(half_angle_rad)

    # 3a. Calculate max acceptable size at the point's depth (proj_distance)
    # The half-width/height of the frustum aperture at this depth.
    max_size = proj_distance * tan_half_angle

    # 3b. Calculate point's lateral displacement along Right and Up axes
    lateral_right = np.dot(vp_vectors, right)
    lateral_up = np.dot(vp_vectors, up)

    # 3c. Pyramidal Check: Displacement along BOTH axes must be less than max_size
    # Right/Left bounds check
    right_mask = np.abs(lateral_right) < max_size + TOLERANCE
    
    # Up/Down bounds check
    up_mask = np.abs(lateral_up) < max_size + TOLERANCE
    
    angular_mask = right_mask & up_mask
    
    # Final mask
    return distance_mask & angular_mask

def get_frustum_lineset(viewpoint, direction, fov_deg, near_d, far_d):
    """
    Helper function to create a LineSet for visualizing the pure pyramid.
    It uses the Viewpoint as the Apex and connects it to the 4 Far Plane corners.
    """
    
    # Corners: [0-3 Near Plane, 4-7 Far Plane]
    corners_8 = get_frustum_corners(viewpoint, direction, fov_deg, near_d, far_d)

    # 1. Combine visualization points: Apex (Viewpoint) + Far Plane Corners
    # Points_viz indices: 0 (Apex), 1-4 (Far Corners)
    points_viz = [viewpoint] 
    points_viz.extend(corners_8[4:])
    points_viz = np.array(points_viz)
    
    # Edges definition
    lines = [
        # Base Edges (Far Plane)
        [1, 2], [2, 3], [3, 4], [4, 1], 
        # Apex Edges (from Apex/0 to Far Corners/1-4)
        [0, 1], [0, 2], [0, 3], [0, 4]
    ]
    
    colors = [[1, 0.5, 0] for _ in range(len(lines))] # Orange frustum
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_viz),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def run_visibility_check():
    """Main function to run the visibility analysis."""
    
    # --- Argument Parsing for Mesh Loading ---
    parser = argparse.ArgumentParser(description="Run 3D visibility check with ray casting.")
    parser.add_argument('--mesh_path', type=str, default=None, help="Path to a .glb, .ply, or .obj file for the target geometry.")
    args = parser.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    dtype = o3c.float32
    device = o3c.Device("CPU:0") 

    # --- 1. Setup Complex Geometry (Mesh Loading or Icosahedron Fallback) ---
    object_mesh = None
    if args.mesh_path and os.path.exists(args.mesh_path):
        print(f"Attempting to load mesh from: {args.mesh_path}")
        object_mesh = o3d.io.read_triangle_mesh(args.mesh_path)
        if not object_mesh.has_vertices() or not object_mesh.has_triangles():
            print(f"Warning: Failed to load mesh from {args.mesh_path}. Falling back to Icosahedron.")
            object_mesh = None
    
    if object_mesh is None:
        print("Using default Icosahedron geometry.")
        object_mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.2)
    
    object_mesh.compute_vertex_normals()
    object_t = o3d.t.geometry.TriangleMesh.from_legacy(object_mesh, device=device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    object_id = scene.add_triangles(object_t)

    # --- 2. Define Dynamic Viewpoint and Frustum Apex ---
    # Generate a random position on a sphere of radius 4.0
    theta = np.random.uniform(0, 2 * np.pi) 
    phi = np.random.uniform(0, np.pi)       
    radius = 4.0
    
    cam_x = radius * np.sin(phi) * np.cos(theta)
    cam_y = radius * np.sin(phi) * np.sin(theta)
    cam_z = radius * np.cos(phi)
    
    global VIEWPOINT_POS, FRUSTUM_DIRECTION
    VIEWPOINT_POS = np.array([cam_x, cam_y, cam_z])
    
    # --- CALCULATE RANDOM ORIENTATION WITH INTERSECTION PROBABILITY ---
    
    # 1. Base direction (from viewpoint towards the center of the object [0,0,0])
    base_direction = -VIEWPOINT_POS / norm(VIEWPOINT_POS)
    
    # 2. Find orthogonal basis vectors for offset calculation
    right, up = get_frustum_basis(base_direction)
    
    # 3. Generate small random offsets (r_offset for yaw, u_offset for pitch)
    r_offset = np.random.uniform(-MAX_DIRECTION_OFFSET, MAX_DIRECTION_OFFSET)
    u_offset = np.random.uniform(-MAX_DIRECTION_OFFSET, MAX_DIRECTION_OFFSET)
    
    # 4. Calculate the randomly offset direction vector
    FRUSTUM_DIRECTION = base_direction + r_offset * right + u_offset * up
    
    # 5. Normalize the final direction
    FRUSTUM_DIRECTION /= norm(FRUSTUM_DIRECTION)
    
    # --- 3. Generate Target Points ---
    # Sample points from the mesh for visibility testing
    target_pcd = object_mesh.sample_points_uniformly(number_of_points=1000000)
    target_points = np.asarray(target_pcd.points)

    # --- 4. Filter Points in Frustum ---
    print(f"Testing {target_points.shape[0]} points on object surface...")
    
    in_frustum_mask = get_frustum_mask(
        target_points, 
        VIEWPOINT_POS, 
        FRUSTUM_DIRECTION, 
        FRUSTUM_FOV_DEGREE, 
        FRUSTUM_NEAR_D, 
        FRUSTUM_FAR_D
    )
    frustum_points = target_points[in_frustum_mask]
    
    if frustum_points.shape[0] == 0:
        print("No points from the object surface are inside the frustum.")
        print("Try adjusting frustum parameters (FOV, Near/Far distance) or MAX_DIRECTION_OFFSET.")
        return
    
    print(f"Found {frustum_points.shape[0]} points inside the frustum.")
    
    # --- 5. Prepare and Cast Rays ---
    num_frustum_points = frustum_points.shape[0]
    
    # Ray origins are all the viewpoint
    ray_origins = np.tile(VIEWPOINT_POS, (num_frustum_points, 1))
    
    # Ray directions point from the viewpoint to each target point
    ray_directions = frustum_points - ray_origins
    
    # Calculate the true distance from viewpoint to each point
    ray_lengths = norm(ray_directions, axis=1)
    
    # Normalize the directions
    ray_directions /= ray_lengths[:, np.newaxis]

    # Create the ray tensor for Open3D
    rays = o3c.Tensor(
        np.hstack([ray_origins, ray_directions]), 
        dtype=dtype, 
        device=device
    )

    # Cast all rays at once
    ans = scene.cast_rays(rays)
    t_hit = ans['t_hit'].numpy()
    geometry_ids = ans['geometry_ids'].numpy()

    # --- 6. Check Occlusion ---
    # A point is visible if:
    # 1. The ray hit the object (geometry_ids == object_id)
    # 2. The hit distance (t_hit) is almost equal to the point's distance (ray_lengths)
    
    epsilon = 1e-5
    
    # Check if ray hit the correct geometry
    hit_object_mask = (geometry_ids == object_id)
    
    # Check if hit distance matches point distance
    distance_match_mask = np.abs(t_hit - ray_lengths) < epsilon
    
    # Final mask for visible points
    visible_mask = hit_object_mask & distance_match_mask
    
    visible_points = frustum_points[visible_mask]
    occluded_points = frustum_points[~visible_mask]
    
    print(f"\n--- Results ---")
    print(f"Out of {num_frustum_points} points in frustum:")
    print(f"  {visible_points.shape[0]} are visible (Green).")
    print(f"  {occluded_points.shape[0]} are occluded (Red).")

    # --- 7. Visualize (Improved) ---
    
    # A. Original Mesh (for visual context)
    context_mesh = object_mesh
    context_mesh.paint_uniform_color([0.7, 0.7, 0.7]) # Light gray for context

    # B. Gray wireframe for the object (still useful for edges)
    object_lines = o3d.geometry.LineSet.create_from_triangle_mesh(object_mesh)
    object_lines.paint_uniform_color([0.4, 0.4, 0.4])

    # C. Frustum lines
    frustum_lines = get_frustum_lineset(
        VIEWPOINT_POS, 
        FRUSTUM_DIRECTION, 
        FRUSTUM_FOV_DEGREE, 
        FRUSTUM_NEAR_D, 
        FRUSTUM_FAR_D
    )
    
    # D. Viewpoint marker
    viewpoint_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    viewpoint_marker.translate(VIEWPOINT_POS)
    viewpoint_marker.paint_uniform_color([0, 1, 0])

    # E. Colored Points (Enhanced Visibility)

    # Green points: Visible and in frustum (Brighter Green)
    visible_pcd = o3d.geometry.PointCloud()
    if visible_points.shape[0] > 0:
        visible_pcd.points = o3d.utility.Vector3dVector(visible_points)
        visible_pcd.paint_uniform_color([0.1, 1.0, 0.1]) # Bright Green

    # Red points: Occluded but in frustum (Brighter Red)
    occluded_pcd = o3d.geometry.PointCloud()
    if occluded_points.shape[0] > 0:
        occluded_pcd.points = o3d.utility.Vector3dVector(occluded_points)
        occluded_pcd.paint_uniform_color([1.0, 0.1, 0.1]) # Bright Red
    
    # Blue points: On object but *outside* frustum (for context)
    out_of_frustum_points = target_points[~in_frustum_mask]
    out_pcd = o3d.geometry.PointCloud()
    if out_of_frustum_points.shape[0] > 0:
        out_pcd.points = o3d.utility.Vector3dVector(out_of_frustum_points)
        out_pcd.paint_uniform_color([0.2, 0.5, 1.0]) # Same Blue

    print("\nVisualizing scene...")
    
    geometries = [
        context_mesh, # The full mesh for context
        object_lines, 
        frustum_lines, 
        viewpoint_marker,
        visible_pcd,
        occluded_pcd,
        out_pcd
    ]

    # Use a Visualizer class to set render options (e.g., point size)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Set the point size to be larger (e.g., 5.0)
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    
    for geom in geometries:
        vis.add_geometry(geom)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    run_visibility_check()
