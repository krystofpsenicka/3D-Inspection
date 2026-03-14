"""
Hyperparameter tuning for epsilon-visibility against raycast ground truth.

Usage:
    python -m visibility.scripts.tune_epsilon [mesh_path]
           [--num-viewpoints 200] [--num-points 100000]
           [--seeds 42 123 7]
"""
import argparse
import csv
import itertools
import numpy as np
import open3d as o3d
from time import time as get_time

from visibility.core import FrustumParams, EpsilonHyperparams, ViewpointSampler, orient_normals_outward
from visibility.methods.raycast import RaycastingVisibilityQuery
from visibility.methods.epsilon import EpsilonVisibilityQuery


PARAM_GRID = {
    "gamma_method": ["p25", "p30", "p40", "median", "p60", "mean", "p75"],
    "epsilon_scale": [0.85, 0.9, 0.95, 1.0, 1.05, 1.1],
    "delta_k": [4, 5, 6, 7],
    "delta_agg": ["max", "p99"],
    "back_face_threshold": [-1e-6],  # fixed — 0.0 is identical, -0.01 is worse
}
# 7 × 6 × 4 × 2 × 1 = 336 configs


def load_mesh(mesh_path):
    if mesh_path is None:
        print("No mesh path provided — using default sphere (r=5).")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
    else:
        print(f"Loading mesh from: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    return mesh


def setup_seed_data(mesh, num_points, num_viewpoints, frustum_params, seed):
    """Sample points, normals, and viewpoints for a given seed."""
    np.random.seed(seed)
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    target_points = np.asarray(pcd.points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals)
    normals = orient_normals_outward(target_points, normals)

    sampler = ViewpointSampler(mesh, target_points, normals, frustum_params.far)
    viewpoints = sampler.sample_outside_mesh(num_candidates=num_viewpoints)

    return target_points, normals, viewpoints


def compute_gt_visibility(mesh, target_points, normals, frustum_params, viewpoints):
    """Pre-compute ground-truth visibility sets for all viewpoints."""
    gt_query = RaycastingVisibilityQuery(mesh, target_points, normals, frustum_params)
    gt_sets = []
    for pos, direction in viewpoints:
        gt_indices, _ = gt_query.compute_visibility(pos, direction)
        gt_sets.append(set(gt_indices.tolist()))
    return gt_sets


def compute_f_scores(gt_set, pred_set):
    """Compute precision, recall, F1, F2 for a single viewpoint."""
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0
    return precision, recall, f1, f2


def evaluate_config(eq, viewpoints, gt_sets, gamma_method, epsilon_scale, back_face_threshold):
    """Evaluate one (gamma_method, epsilon_scale, back_face_threshold) combo by swapping hp fields."""
    eq.hp.gamma_method = gamma_method
    eq.hp.epsilon_scale = epsilon_scale
    eq.hp.back_face_threshold = back_face_threshold

    f1s, precisions, recalls, f2s = [], [], [], []
    for (pos, direction), gt_set in zip(viewpoints, gt_sets):
        pred_indices, _ = eq.compute_visibility(pos, direction)
        pred_set = set(pred_indices.tolist())
        p, r, f1, f2 = compute_f_scores(gt_set, pred_set)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        f2s.append(f2)

    return precisions, recalls, f1s, f2s


def main():
    parser = argparse.ArgumentParser(
        description="Tune epsilon-visibility hyperparameters against raycast GT."
    )
    parser.add_argument("mesh_path", nargs="?", default=None,
                        help="Path to mesh file (default: sphere r=5)")
    parser.add_argument("--num-viewpoints", type=int, default=200,
                        help="Number of viewpoints per seed (default: 200)")
    parser.add_argument("--num-points", type=int, default=100000,
                        help="Number of surface points to sample (default: 100000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7],
                        help="Random seeds for multi-seed evaluation (default: 42 123 7)")
    args = parser.parse_args()

    mesh = load_mesh(args.mesh_path)
    frustum_params = FrustumParams(fov_y=np.deg2rad(45), aspect=1.0, near=0.01, far=7.0)

    # Pre-compute per-seed data and ground truth
    seed_data = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Setting up seed {seed}...")
        target_points, normals, viewpoints = setup_seed_data(
            mesh, args.num_points, args.num_viewpoints, frustum_params, seed
        )
        print(f"Computing ground-truth raycast visibility for {len(viewpoints)} viewpoints...")
        gt_sets = compute_gt_visibility(mesh, target_points, normals, frustum_params, viewpoints)
        seed_data.append((target_points, normals, viewpoints, gt_sets))

    # Enumerate all configs
    delta_groups = list(itertools.product(PARAM_GRID["delta_k"], PARAM_GRID["delta_agg"]))
    inner_combos = list(itertools.product(
        PARAM_GRID["gamma_method"],
        PARAM_GRID["epsilon_scale"],
        PARAM_GRID["back_face_threshold"],
    ))

    total_configs = len(delta_groups) * len(inner_combos)
    print(f"\n{'='*60}")
    print(f"Total configs: {total_configs} ({len(delta_groups)} delta groups × {len(inner_combos)} inner combos)")
    print(f"Seeds: {args.seeds} ({len(args.seeds)} seeds × {args.num_viewpoints} VP = "
          f"{len(args.seeds) * args.num_viewpoints} total evaluations per config)")
    print(f"{'='*60}\n")

    # Results: config_key -> pooled metrics
    results = {}
    start_time = get_time()
    config_count = 0

    for delta_k, delta_agg in delta_groups:
        print(f"\n--- Delta group: k={delta_k}, agg={delta_agg} ---")

        # Build one EpsilonVisibilityQuery per seed for this delta group
        seed_queries = []
        for si, (target_points, normals, viewpoints, gt_sets) in enumerate(seed_data):
            np.random.seed(args.seeds[si])  # reproducible delta estimation
            hp = EpsilonHyperparams(delta_k=delta_k, delta_agg=delta_agg)
            eq = EpsilonVisibilityQuery(mesh, target_points, normals, frustum_params, hyperparams=hp)
            seed_queries.append((eq, viewpoints, gt_sets))

        for gamma_method, epsilon_scale, bf_thr in inner_combos:
            config_count += 1
            config_key = (gamma_method, epsilon_scale, delta_k, delta_agg, bf_thr)

            pooled_f1, pooled_prec, pooled_rec, pooled_f2 = [], [], [], []

            for eq, viewpoints, gt_sets in seed_queries:
                precs, recs, f1s, f2s = evaluate_config(
                    eq, viewpoints, gt_sets, gamma_method, epsilon_scale, bf_thr
                )
                pooled_f1.extend(f1s)
                pooled_prec.extend(precs)
                pooled_rec.extend(recs)
                pooled_f2.extend(f2s)

            results[config_key] = {
                "mean_f1": np.mean(pooled_f1),
                "mean_prec": np.mean(pooled_prec),
                "mean_rec": np.mean(pooled_rec),
                "mean_f2": np.mean(pooled_f2),
                "std_f1": np.std(pooled_f1),
                "min_f1": np.min(pooled_f1),
            }

            if config_count % 20 == 0:
                elapsed = get_time() - start_time
                eta = elapsed / config_count * (total_configs - config_count)
                print(f"  [{config_count}/{total_configs}] "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                      f"current: gamma={gamma_method} scale={epsilon_scale} bf={bf_thr} "
                      f"F1={results[config_key]['mean_f1']:.5f}")

    total_elapsed = get_time() - start_time
    print(f"\n{'='*60}")
    print(f"Tuning complete in {total_elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Rank by mean F1
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_f1"], reverse=True)

    # Print top-20
    header = (f"{'Rank':>4}  {'gamma':>7}  {'scale':>5}  {'dk':>2}  {'d_agg':>5}  "
              f"{'bf_thr':>8}  {'F1':>8}  {'Prec':>8}  {'Recall':>8}  "
              f"{'F2':>8}  {'std':>8}  {'min':>8}")
    print(header)
    print("-" * len(header))

    for rank, (key, m) in enumerate(ranked[:20], 1):
        gamma_method, epsilon_scale, delta_k, delta_agg, bf_thr = key
        print(f"{rank:>4}  {gamma_method:>7}  {epsilon_scale:>5.2f}  {delta_k:>2}  {delta_agg:>5}  "
              f"{bf_thr:>8.1e}  {m['mean_f1']:>8.5f}  {m['mean_prec']:>8.5f}  {m['mean_rec']:>8.5f}  "
              f"{m['mean_f2']:>8.5f}  {m['std_f1']:>8.5f}  {m['min_f1']:>8.5f}")

    # Save full CSV
    csv_path = "tuning_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "gamma_method", "epsilon_scale", "delta_k", "delta_agg",
                          "back_face_threshold", "mean_f1", "mean_precision", "mean_recall",
                          "mean_f2", "std_f1", "min_f1"])
        for rank, (key, m) in enumerate(ranked, 1):
            gamma_method, epsilon_scale, delta_k, delta_agg, bf_thr = key
            writer.writerow([rank, gamma_method, epsilon_scale, delta_k, delta_agg, bf_thr,
                             f"{m['mean_f1']:.10f}", f"{m['mean_prec']:.10f}",
                             f"{m['mean_rec']:.10f}", f"{m['mean_f2']:.10f}",
                             f"{m['std_f1']:.10f}", f"{m['min_f1']:.10f}"])

    print(f"\nFull results saved to {csv_path}")

    # Print best config for copy-paste
    best_key = ranked[0][0]
    gamma_method, epsilon_scale, delta_k, delta_agg, bf_thr = best_key
    print(f"\nBest config (copy-paste):")
    print(f"EpsilonHyperparams(")
    print(f"    gamma_method=\"{gamma_method}\",")
    print(f"    epsilon_scale={epsilon_scale},")
    print(f"    delta_k={delta_k},")
    print(f"    delta_agg=\"{delta_agg}\",")
    print(f"    back_face_threshold={bf_thr},")
    print(f")")


if __name__ == "__main__":
    main()
