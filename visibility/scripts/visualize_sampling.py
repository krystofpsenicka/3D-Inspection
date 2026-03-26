"""
Visualize free-space sampling probability heatmap.

Shows feasible viewpoint regions as 3D point clouds colored by sampling
probability, overlaid on a wireframe mesh. Supports normal, curvature-weighted,
and targeted resampling modes.

Usage:
    python -m visibility.scripts.visualize_sampling [mesh_path]
           [--num-points 2000] [--collision-radius 0.5]
           [--mode normal|curvature|resampling]
"""
import argparse
import logging
import numpy as np
import cupy as cp
import open3d as o3d

from visibility.core import FrustumParams, orient_normals_outward
from visibility.sampling import TargetedViewpointSampler, DEViewpointSampler
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


def print_coverage_summary(visibility_map, num_target_points, vis_time, label="Visibility"):
    all_covered = set()
    for visible_indices in visibility_map.values():
        all_covered.update(visible_indices.tolist())
    total_coverage = len(all_covered) / num_target_points * 100

    print(f"\n--- {label} Summary ---")
    for i, visible_indices in visibility_map.items():
        print(f"  Viewpoint {i}: {len(visible_indices)} visible points")
    print(f"  Total coverage: {len(all_covered)} / {num_target_points} "
          f"({total_coverage:.1f}%)")
    print(f"  Computation time: {vis_time:.2f}s")
    return all_covered, total_coverage


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
    parser.add_argument("--num-viewpoints", type=int, default=10,
                        help="Number of viewpoints to sample (default: 10)")
    parser.add_argument("--side", choices=["outside", "inside"], default="outside",
                        help="Which side of the mesh to sample from (default: outside)")
    parser.add_argument("--mode", choices=["normal", "curvature", "resampling"],
                        default="normal", help="Sampling mode to visualize")
    parser.add_argument("--resample-fraction", type=float, default=0.25,
                        help="Fraction of candidates for targeted resampling (default: 0.25)")
    parser.add_argument("--resampling-strategy", choices=["random", "optimal"],
                        default="random", help="Targeted resampling strategy")
    args = parser.parse_args()

    curvature_weighting = (args.mode == "curvature")

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
    sampler = TargetedViewpointSampler(
        mesh, target_points, normals, frustum_params.far,
        collision_radius=args.collision_radius,
    )

    print("\nBuilding feasible regions...")
    outside_pos, outside_w = sampler.get_feasible_data(
        side="outside", curvature_weighting=curvature_weighting)
    inside_pos, inside_w = sampler.get_feasible_data(
        side="inside", curvature_weighting=curvature_weighting)

    print(f"  Outside: {len(outside_pos)} feasible positions")
    print(f"  Inside:  {len(inside_pos)} feasible positions")

    # --- Visualize free-space heatmap (Window 1) ---
    mode_labels = {"normal": "SDF²", "curvature": "Curvature-Weighted", "resampling": "SDF²"}
    window_name = f"Free-Space Sampling Heatmap ({mode_labels[args.mode]})"

    visualizer = Visualizer(mesh, target_points, normals, frustum_params)
    visualizer.visualize_free_space(
        outside_pos, outside_w,
        inside_pos, inside_w,
        point_size=args.point_size,
        window_name=window_name,
    )

    if args.mode in ("normal", "curvature"):
        # --- Normal / Curvature: sample and show all VPs (Window 2) ---
        print(f"\nSampling {args.num_viewpoints} viewpoints from {args.side} "
              f"(mode={args.mode})...")
        candidates = sampler.sample(
            num_candidates=args.num_viewpoints,
            side=args.side,
            curvature_weighting=curvature_weighting)

        if not candidates:
            print("No valid viewpoints sampled — skipping visibility visualization.")
            return

        print(f"  Sampled {len(candidates)} valid viewpoints.")

        vis_query = RaycastingVisibilityQueryCuda(mesh, target_points, normals, frustum_params)
        visibility_map, vis_time = vis_query.compute_visibility_batch(candidates)

        print_coverage_summary(visibility_map, len(target_points), vis_time)
        visualizer.visualize_all_visibility_results(visibility_map, candidates)

    else:
        # --- Resampling mode: two-phase sampling ---
        n_targeted = int(args.num_viewpoints * args.resample_fraction)
        n_normal = args.num_viewpoints - n_targeted

        print(f"\nResampling mode: {n_normal} normal + {n_targeted} targeted VPs "
              f"(strategy={args.resampling_strategy})")

        # Phase 1: sample normal candidates
        print(f"  Sampling {n_normal} normal viewpoints from {args.side}...")
        normal_candidates = sampler.sample(num_candidates=n_normal, side=args.side)

        if not normal_candidates:
            print("No valid normal viewpoints sampled — aborting.")
            return

        print(f"  Sampled {len(normal_candidates)} normal viewpoints.")

        vis_query = RaycastingVisibilityQueryCuda(mesh, target_points, normals, frustum_params)
        normal_vis_map, normal_time = vis_query.compute_visibility_batch(
            normal_candidates)

        normal_covered, normal_coverage = print_coverage_summary(
            normal_vis_map, len(target_points), normal_time, label="Normal VPs")

        # Phase 2: identify uncovered and sample targeted candidates
        all_indices = set(range(len(target_points)))
        uncovered_indices = np.array(sorted(all_indices - normal_covered), dtype=int)
        print(f"\n  {len(uncovered_indices)} uncovered points remaining.")

        if len(uncovered_indices) == 0 or n_targeted == 0:
            print("  No targeted resampling needed.")
            visualizer.visualize_all_visibility_results(normal_vis_map, normal_candidates)
            return

        if args.resampling_strategy == "optimal":
            # Build per-point coverage counts for DE
            coverage_count_gpu = cp.zeros(len(target_points), dtype=cp.int32)
            for v in normal_vis_map.values():
                if len(v) > 0:
                    coverage_count_gpu[cp.asarray(v)] += 1

            de_sampler = DEViewpointSampler(
                mesh, target_points, normals, frustum_params.far,
                collision_radius=args.collision_radius,
            )
            targeted_candidates = de_sampler.sample_de(
                n_targeted, coverage_count_gpu, vis_query,
                existing_candidates=normal_candidates,
                side=args.side, verbose=True)

            if not targeted_candidates:
                print("  No valid targeted viewpoints found via DE.")
                visualizer.visualize_all_visibility_results(
                    normal_vis_map, normal_candidates)
                return

            targeted_vis_map = {}
            targeted_vis_all, _ = vis_query.compute_visibility_batch(targeted_candidates)
            for k, v in targeted_vis_all.items():
                targeted_vis_map[k] = v

        else:
            # Random proximity-weighted targeted sampling
            print(f"  Sampling {n_targeted} targeted viewpoints...")
            targeted_candidates = sampler.sample_targeted(
                uncovered_indices, n_targeted, side=args.side)

            if not targeted_candidates:
                print("  No valid targeted viewpoints sampled.")
                visualizer.visualize_all_visibility_results(
                    normal_vis_map, normal_candidates)
                return

            print(f"  Sampled {len(targeted_candidates)} targeted viewpoints.")

            targeted_vis_map, _ = vis_query.compute_visibility_batch(
                targeted_candidates)
            targeted_vis_map = dict(targeted_vis_map)

        # Coverage summary for targeted VPs
        targeted_covered = set()
        for vis_indices in targeted_vis_map.values():
            targeted_covered.update(vis_indices.tolist()
                                    if hasattr(vis_indices, 'tolist')
                                    else list(vis_indices))
        print(f"\n--- Targeted VPs Summary ---")
        for i, vis_indices in targeted_vis_map.items():
            vis_arr = vis_indices if hasattr(vis_indices, '__len__') else []
            print(f"  Viewpoint {i}: {len(vis_arr)} visible points")
        print(f"  Total targeted: {len(targeted_covered)} unique points")

        # Combined summary
        combined_covered = set(normal_covered)
        combined_covered.update(targeted_covered)
        combined_coverage = len(combined_covered) / len(target_points) * 100
        print(f"\n  Combined coverage: {len(combined_covered)} / {len(target_points)} "
              f"({combined_coverage:.1f}%)")

        # Visualize resampling progression
        visualizer.visualize_resampling_progression(
            normal_vis_map, normal_candidates,
            targeted_vis_map, targeted_candidates)


if __name__ == "__main__":
    main()
