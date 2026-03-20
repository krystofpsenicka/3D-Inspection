"""
Visual inspection of per-viewpoint visibility results.

Usage:
    python -m visibility.scripts.check_visibility [mesh_path] [--method raycast|epsilon]
                                                   [--num-viewpoints N] [--num-points M]
"""
import argparse
import numpy as np
import open3d as o3d
from time import time as get_time

from visibility.core import FrustumParams, orient_normals_outward
from visibility.sampling import ViewpointSampler
from visibility.methods.raycast import RaycastingVisibilityQuery
from visibility.methods.epsilon import EpsilonVisibilityQuery
from visibility.methods.epsilon_cuda import EpsilonVisibilityQueryCuda
from visibility.methods.raycast_cuda import RaycastingVisibilityQueryCuda
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
    parser = argparse.ArgumentParser(description="Visual inspection of visibility results.")
    parser.add_argument("mesh_path", nargs="?", default=None,
                        help="Path to mesh file (default: sphere r=5)")
    parser.add_argument("--method", choices=["raycast", "epsilon", "epsilon_cuda", "raycast_cuda"], default="raycast",
                        help="Visibility method to visualize (default: raycast)")
    parser.add_argument("--num-viewpoints", type=int, default=5,
                        help="Number of viewpoints to generate (default: 5)")
    parser.add_argument("--num-points", type=int, default=2000,
                        help="Number of surface points to sample (default: 2000)")
    parser.add_argument("--sampler", choices=["free_space"], default="free_space",
                        help="Sampling method for viewpoints (default: free_space)")
    parser.add_argument("--side", choices=["outside", "inside"], default="outside",
                        help="Whether to sample viewpoints outside or inside the mesh (default: outside)")
    parser.add_argument("--separate", action="store_true", default=False,
                        help="Show each viewpoint in a separate window (default: all together)")
    args = parser.parse_args()

    # --- Setup ---
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

    # --- Viewpoints ---
    sampler = ViewpointSampler(mesh, target_points, normals, frustum_params.far, collision_radius=0.5)
    if args.side == "outside":
        viewpoints = sampler.sample_outside_mesh(num_candidates=args.num_viewpoints)
    else:
        viewpoints = sampler.sample_inside_mesh(num_candidates=args.num_viewpoints)
        
    # --- Query ---
    match args.method:
        case "raycast":
            query = RaycastingVisibilityQuery(mesh, target_points, normals, frustum_params)
        case "epsilon":
            query = EpsilonVisibilityQuery(mesh, target_points, normals, frustum_params)
        case "epsilon_cuda":
            query = EpsilonVisibilityQueryCuda(mesh, target_points, normals, frustum_params)
        case "raycast_cuda":
            query = RaycastingVisibilityQueryCuda(mesh, target_points, normals, frustum_params)
        case _:
            raise ValueError(f"Unknown method: {args.method}")

    print(f"\nRunning '{args.method}' visibility on {len(viewpoints)} viewpoints "
          f"({len(target_points)} surface points)...\n")

    visibility_map = {}
    for i, (pos, orientation) in enumerate(viewpoints):
        t0 = get_time()
        visible_indices, comp_time = query.compute_visibility(pos, orientation)
        print(f"  VP {i}: visible {len(visible_indices)} / {len(target_points)}  "
              f"time={get_time() - t0:.3f} s")
        visibility_map[i] = visible_indices.astype(int)

    # --- Visualize ---
    visualizer = Visualizer(mesh, target_points, normals, frustum_params)
    if args.separate:
        print("\nOpening visualization windows (close each to advance)...")
        for i in range(len(viewpoints)):
            visualizer.visualize_visibility_results(visibility_map, i, candidates=viewpoints)
    else:
        visualizer.visualize_all_visibility_results(visibility_map, candidates=viewpoints)


if __name__ == "__main__":
    main()
