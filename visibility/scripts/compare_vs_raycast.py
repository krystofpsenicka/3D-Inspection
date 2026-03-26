"""
Quantitative + visual comparison of any visibility method against raycast ground-truth.

Usage:
    python -m visibility.scripts.compare_vs_raycast [mesh_path] [--method epsilon]
                                                     [--num-viewpoints N] [--num-points M]
"""
import argparse
import numpy as np
import open3d as o3d
from time import time as get_time

from visibility.core import FrustumParams, orient_normals_outward, get_frustum_basis_from_rotation
from visibility.sampling import UniformViewpointSampler
from visibility.methods.raycast import RaycastingVisibilityQuery
from visibility.methods.epsilon import EpsilonVisibilityQuery
from visibility.methods.epsilon_cuda import EpsilonVisibilityQueryCuda
from visibility.methods.raycast_cuda import RaycastingVisibilityQueryCuda


def load_mesh(mesh_path: str | None) -> o3d.geometry.TriangleMesh:
    if mesh_path is None:
        print("No mesh path provided — using default sphere (r=5).")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
    else:
        print(f"Loading mesh from: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    return mesh


def visualize_diff(mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                   gt_set: set, pred_set: set, viewpoints, vp_index: int,
                   frustum_params: FrustumParams):
    """
    Render a per-viewpoint diff window:
      Green  = true positive  (visible in both GT and pred)
      Red    = false positive (visible in pred only)
      Orange = false negative (visible in GT only)
      Gray   = true negative  (hidden in both)
    """
    tp = gt_set & pred_set
    fp = pred_set - gt_set
    fn = gt_set - pred_set

    num_points = len(target_points)
    colors = np.full((num_points, 3), [0.4, 0.4, 0.4], dtype=np.float64)  # gray = TN
    for idx in tp:
        colors[idx] = [0.0, 1.0, 0.0]   # green
    for idx in fp:
        colors[idx] = [1.0, 0.0, 0.0]   # red
    for idx in fn:
        colors[idx] = [1.0, 0.5, 0.0]   # orange

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(target_points)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)

    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
    mesh_vis.compute_vertex_normals()

    geometries = [mesh_vis, pcd_vis]

    for i, (pos, orientation) in enumerate(viewpoints):
        radius = 0.015 if i == vp_index else 0.01
        color = [1.0, 0.0, 0.0] if i == vp_index else [0.0, 0.0, 1.0]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    pos, orientation = viewpoints[vp_index]
    forward = orientation.as_matrix()[:, 0]
    arrow_length = frustum_params.far * 0.2
    arrow_end = pos + forward * arrow_length
    arrow = o3d.geometry.LineSet()
    arrow.points = o3d.utility.Vector3dVector(np.array([pos, arrow_end]))
    arrow.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    arrow.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])
    geometries.append(arrow)

    # Build frustum lineset without constructing a full Visualizer
    half_angle = frustum_params.fov_y / 2.0
    far_half = frustum_params.far * np.tan(half_angle)
    _, right, up = get_frustum_basis_from_rotation(orientation)
    far_center = pos + forward * frustum_params.far
    r, u = right * far_half, up * far_half
    f_corners = [pos, far_center + r + u, far_center - r + u,
                 far_center - r - u, far_center + r - u]
    f_lines = [[1, 2], [2, 3], [3, 4], [4, 1], [0, 1], [0, 2], [0, 3], [0, 4]]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(np.array(f_corners))
    frustum.lines = o3d.utility.Vector2iVector(np.array(f_lines))
    frustum.paint_uniform_color([0.8, 0.8, 0.0])
    geometries.append(frustum)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Diff VP {vp_index}  TP=green  FP=red  FN=orange  TN=gray"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative comparison of a visibility method against raycast GT."
    )
    parser.add_argument("mesh_path", nargs="?", default=None,
                        help="Path to mesh file (default: sphere r=5)")
    parser.add_argument("--method", choices=["epsilon", "epsilon_cuda"], default="epsilon",
                        help="Method to compare against raycast (default: epsilon)")
    parser.add_argument("--num-viewpoints", type=int, default=5,
                        help="Number of viewpoints to evaluate (default: 5)")
    parser.add_argument("--num-points", type=int, default=2000,
                        help="Number of surface points to sample (default: 2000)")
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
    sampler = UniformViewpointSampler(mesh, target_points, normals, frustum_params.far, collision_radius=0.5)
    viewpoints = sampler.sample(num_candidates=args.num_viewpoints, side="outside")

    # --- Instantiate both queries ---
    gt_query = RaycastingVisibilityQueryCuda(mesh, target_points, normals, frustum_params)

    match args.method:
        case "epsilon":
            pred_query = EpsilonVisibilityQuery(target_points, normals, frustum_params)
        case "epsilon_cuda":
            pred_query = EpsilonVisibilityQueryCuda(target_points, normals, frustum_params)
        case _:
            raise ValueError(f"Unknown method: {args.method}")

    print(f"\nComparing '{args.method}' vs raycast on {len(viewpoints)} viewpoints "
          f"({len(target_points)} surface points)...\n")

    header = f"{'VP':>4}  {'GT':>6}  {'Pred':>6}  {'TP':>6}  "
    header += f"{'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Time':>8}"
    print(header)
    print("-" * len(header))

    precisions, recalls, f1s = [], [], []
    vp_results = []  # store (gt_set, pred_set) for visualization

    for i, (pos, orientation) in enumerate(viewpoints):
        t0 = get_time()

        gt_indices, _ = gt_query.compute_visibility(pos, orientation)
        pred_indices, _ = pred_query.compute_visibility(pos, orientation)

        elapsed = get_time() - t0

        gt_set = set(gt_indices.tolist())
        pred_set = set(pred_indices.tolist())
        vp_results.append((gt_set, pred_set))

        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"VP {i:>2}: GT={len(gt_set):>6}  Pred={len(pred_set):>6}  TP={tp:>6}  "
              f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  "
              f"({elapsed:.3f} s)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std across viewpoints)")
    print(f"  Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    print(f"  Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    print(f"  F1:        {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print("=" * 60)

    # --- Visual diff per viewpoint (reuse stored results) ---
    print("\nOpening visual diff windows (close each to advance)...")
    for i, (gt_set, pred_set) in enumerate(vp_results):
        visualize_diff(mesh, target_points, gt_set, pred_set, viewpoints, i, frustum_params)


if __name__ == "__main__":
    main()
