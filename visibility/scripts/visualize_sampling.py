"""
Visualize free-space sampling probability heatmap.

Shows feasible viewpoint regions as 3D point clouds colored by sampling
probability (sdf² weights), overlaid on a wireframe mesh.

Usage:
    python -m visibility.scripts.visualize_sampling [mesh_path]
           [--num-points 2000] [--collision-radius 0.5]
"""
import argparse
import numpy as np
import open3d as o3d

from visibility.core import FrustumParams, ViewpointSampler, orient_normals_outward
from visibility.visualization import Visualizer


def load_mesh(mesh_path: str | None) -> o3d.geometry.TriangleMesh:
    if mesh_path is None:
        print("No mesh path provided — using default sphere (r=5).")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
    else:
        print(f"Loading mesh from: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Visualize free-space sampling probability heatmap.")
    parser.add_argument("mesh_path", nargs="?", default=None,
                        help="Path to mesh file (default: sphere r=5)")
    parser.add_argument("--num-points", type=int, default=100000,
                        help="Number of surface points to sample (default: 100000)")
    parser.add_argument("--collision-radius", type=float, default=0.5,
                        help="Collision radius for viewpoint sampling (default: 0.5)")
    parser.add_argument("--point-size", type=float, default=3.0,
                        help="Point size for visualization (default: 3.0)")
    args = parser.parse_args()

    # --- Setup (same pattern as other scripts) ---
    mesh = load_mesh(args.mesh_path)

    pcd = mesh.sample_points_poisson_disk(number_of_points=args.num_points)
    target_points = np.asarray(pcd.points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals)
    normals = orient_normals_outward(target_points, normals)

    frustum_params = FrustumParams(fov_y=np.deg2rad(45), aspect=1.0, near=0.01, far=7.0)

    # --- Build sampler and get feasible data ---
    sampler = ViewpointSampler(
        mesh, target_points, normals, frustum_params.far,
        collision_radius=args.collision_radius,
    )

    print("\nBuilding feasible regions...")
    outside_pos, outside_w = sampler.get_feasible_data(side="outside")
    inside_pos, inside_w = sampler.get_feasible_data(side="inside")

    print(f"  Outside: {len(outside_pos)} feasible positions")
    print(f"  Inside:  {len(inside_pos)} feasible positions")

    # --- Visualize ---
    visualizer = Visualizer(mesh, target_points, normals, frustum_params)
    visualizer.visualize_free_space(
        outside_pos, outside_w,
        inside_pos, inside_w,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
