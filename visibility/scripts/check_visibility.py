"""
Visual inspection of per-viewpoint visibility results.

Usage:
    python -m visibility.scripts.check_visibility [mesh_path] [--method raycast|epsilon]
                                                   [--num-viewpoints N] [--num-points M]
"""
import argparse
import numpy as np
import cupy as cp
import open3d as o3d
from time import time as get_time

from visibility.core import FrustumParams
from shared.surface_sampler import SurfacePointSampler
from visibility.sampling import WeightedViewpointSampler
from visibility.methods.raycast import RaycastingVisibilityQuery
from visibility.methods.epsilon import EpsilonVisibilityQuery
from visibility.methods.epsilon_cuda import EpsilonVisibilityQueryCuda
from visibility.methods.raycast_cuda import RaycastingVisibilityQueryCuda
from visualization import VisibilityVisualizer


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

    surface_sampler = SurfacePointSampler()
    target_points, normals = surface_sampler.sample(mesh, args.num_points)

    frustum_params = FrustumParams(fov_y=np.deg2rad(45), aspect=1.0, near=0.01, far=7.0)

    # --- Viewpoints ---
    sampler = WeightedViewpointSampler(mesh, target_points, normals, frustum_params.far, collision_radius=0.5)
    pos_gpu, rot_gpu = sampler.sample(num_candidates=args.num_viewpoints, side=args.side)

    # --- Query ---
    use_gpu = args.method in ("epsilon_cuda", "raycast_cuda")
    match args.method:
        case "raycast":
            query = RaycastingVisibilityQuery(mesh, target_points, normals, frustum_params)
        case "epsilon":
            query = EpsilonVisibilityQuery(target_points, normals, frustum_params)
        case "epsilon_cuda":
            query = EpsilonVisibilityQueryCuda(target_points, normals, frustum_params)
        case "raycast_cuda":
            query = RaycastingVisibilityQueryCuda(mesh, target_points, normals, frustum_params)
        case _:
            raise ValueError(f"Unknown method: {args.method}")

    n_vp = len(pos_gpu)
    print(f"\nRunning '{args.method}' visibility on {n_vp} viewpoints "
          f"({len(target_points)} surface points)...\n")

    # Transfer to CPU for CPU queries
    if use_gpu:
        positions, rotmats = pos_gpu, rot_gpu
    else:
        positions, rotmats = cp.asnumpy(pos_gpu), cp.asnumpy(rot_gpu)

    visibility_map = {}
    for i in range(n_vp):
        t0 = get_time()
        visible_indices, comp_time = query.compute_visibility(positions[i], rotmats[i])
        print(f"  VP {i}: visible {len(visible_indices)} / {len(target_points)}  "
              f"time={get_time() - t0:.3f} s")
        visibility_map[i] = visible_indices.astype(int)

    # --- Visualize (CPU arrays + list of tuples) ---
    pos_cpu = cp.asnumpy(pos_gpu) if use_gpu else positions
    rot_cpu = cp.asnumpy(rot_gpu) if use_gpu else rotmats
    candidates = list(zip(pos_cpu, rot_cpu))

    viz = VisibilityVisualizer(mesh, target_points, frustum_params)
    if args.separate:
        print("\nOpening visualization windows (close each to advance)...")
        for i in range(n_vp):
            viz.visualize_single(visibility_map[i], pos_cpu[i], rot_cpu[i])
    else:
        viz.visualize_all(visibility_map, candidates=candidates)


if __name__ == "__main__":
    main()
