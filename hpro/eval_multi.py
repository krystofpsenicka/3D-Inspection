"""Stage 2 evaluation harness — multi-viewpoint HPRO vs. greedy baseline.

Evaluates ``optimize_viewpoints`` (joint differentiable optimisation) against
a discrete greedy set-cover baseline across multiple meshes and V values.
Mirrors the structure of ``eval_frustum.py`` so results can be directly
compared.

Outputs (written under ``--out``, default ``results/``):
  * ``eval_multi.csv``         — one row per (mesh, V, method).
  * ``coverage_vs_V.png``      — GT coverage vs V, optimizer vs baseline.
  * ``coverage_curve_V{v}.png`` — soft C training curves per V value.

Examples
--------
Quick local sanity (CPU, tiny)::

    python hpro/eval_multi.py --mesh_dir hpro --num_points 1000 \\
        --num_viewpoints "1,3,5" --n_steps 50 --no_show

Full run::

    python hpro/eval_multi.py --mesh_dir hpro --num_points 3000 \\
        --num_viewpoints "1,2,5,10,20" --n_steps 500 --no_show
"""

import argparse
import csv
import glob
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402
import trimesh                    # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from HPRO_limited import HPRO_limited                              # noqa: E402
from frustum_gt import (                                           # noqa: E402
    build_camera_frame,
    compute_ground_truth_batched,
    points_inside_frustum,
)
from multi_viewpoint import (                                      # noqa: E402
    optimize_viewpoints,
    fibonacci_sphere_viewpoints,
    MultiViewOptResult,
)


CSV_FIELDS = [
    "mesh", "seed", "V", "n_points",
    "gt_coverage", "baseline_gt_coverage",
    "soft_coverage_final", "wall_time_s", "baseline_wall_time_s",
    "lambda_standoff", "lambda_diversity", "gamma", "k", "n_steps",
]


# ---------------------------------------------------------------------------
# Shared helpers (copied from eval_frustum.py to avoid side-effect imports)
# ---------------------------------------------------------------------------

def load_meshes(mesh_dir, max_meshes=0, seed=0):
    """Load and unit-normalise meshes under ``mesh_dir`` (or a single file)."""
    if os.path.isfile(mesh_dir):
        paths = [mesh_dir]
    else:
        paths = []
        for ext in ("*.off", "*.obj", "*.ply", "*.stl", "*.glb"):
            paths.extend(sorted(glob.glob(os.path.join(mesh_dir, ext))))
    if not paths:
        raise FileNotFoundError(f"No meshes found under {mesh_dir!r}")

    if max_meshes and len(paths) > max_meshes:
        rng = np.random.default_rng(seed)
        paths = sorted(rng.choice(paths, size=max_meshes, replace=False).tolist())

    meshes = []
    for p in paths:
        m = trimesh.load_mesh(p)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        m.vertices -= np.mean(m.vertices, axis=0)
        m.vertices /= np.linalg.norm(m.vertices, axis=1).max()
        meshes.append((os.path.basename(p), m))
    return meshes


def fibonacci_sphere(n):
    """``n`` roughly-uniform unit directions on the sphere (deterministic)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


# ---------------------------------------------------------------------------
# Greedy baseline
# ---------------------------------------------------------------------------

def greedy_baseline(
    mesh, pts_np, V, fov_h, fov_v, near, far,
    n_candidates=100, seed=17,
):
    """
    Discrete greedy set-cover baseline.

    Samples ``n_candidates`` viewpoints on a Fibonacci sphere (same radius
    convention as the optimiser's Fibonacci init), computes exact GT coverage
    for each candidate via ray-casting, then greedily selects V candidates
    maximising the union of covered points.

    Returns:
        covered_union  — set[int] of union-covered point indices.
        wall_time_s    — float.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    centroid = pts_np.mean(axis=0)
    radius = (near + far) / 2.0

    dirs = fibonacci_sphere(n_candidates)
    dirs = dirs + 0.05 * rng.standard_normal(dirs.shape)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    world_up = np.array([0.0, 0.0, 1.0])
    intersector = mesh.ray

    # Pre-compute GT visibility for all candidates
    candidate_gt = []
    for d in dirs:
        vp = centroid + radius * d
        look = -d
        up = world_up if abs(np.dot(look / np.linalg.norm(look), world_up)) < 0.95 \
            else np.array([0.0, 1.0, 0.0])
        look_n, up_n, right_n = build_camera_frame(look, up)
        gt_idx, _ = compute_ground_truth_batched(
            mesh, vp, pts_np, look_n, up_n, right_n,
            fov_h, fov_v, near, far, intersector=intersector,
        )
        candidate_gt.append(set(int(i) for i in gt_idx))

    # Greedy selection
    covered = set()
    for _ in range(V):
        best_gain = -1
        best_j = 0
        for j, gt_j in enumerate(candidate_gt):
            gain = len(gt_j - covered)
            if gain > best_gain:
                best_gain = gain
                best_j = j
        covered |= candidate_gt[best_j]

    return covered, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Optimizer evaluation
# ---------------------------------------------------------------------------

def eval_optimizer(
    mesh_name, mesh, pts_np, V, model, cfg, seed,
):
    """Run optimize_viewpoints and evaluate the result with GT ray-casting."""
    result = optimize_viewpoints(
        pts_np, V, model,
        gamma=cfg["gamma"],
        k=cfg["k"],
        n_steps=cfg["n_steps"],
        lr=cfg["lr"],
        lambda_standoff=cfg["lambda_standoff"],
        lambda_diversity=cfg["lambda_diversity"],
        near_dist=cfg["near"],
        far_dist=cfg["far"],
        min_inter_dist=cfg["min_inter_dist"],
        log_every=max(1, cfg["n_steps"] // 10),
        dtype=torch.float32,
        seed=seed,
    )

    # GT evaluation of final viewpoints (hard ray-cast)
    intersector = mesh.ray
    covered_union = set()
    for v in range(V):
        look, up, right = build_camera_frame(
            result.rot_6d_np[v, :3], result.rot_6d_np[v, 3:]
        )
        gt_idx, _ = compute_ground_truth_batched(
            mesh, result.viewpoints_np[v], pts_np,
            look, up, right,
            cfg["fov_h"], cfg["fov_v"], cfg["near"], cfg["far"],
            intersector=intersector,
        )
        covered_union.update(int(i) for i in gt_idx)

    N = len(pts_np)
    gt_coverage = len(covered_union) / N if N > 0 else 0.0
    soft_final = result.coverage_history[-1] if result.coverage_history else 0.0

    return {
        "gt_coverage": gt_coverage,
        "soft_coverage_final": soft_final,
        "wall_time_s": result.wall_time_s,
        "opt_result": result,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_coverage_vs_V(rows, out_dir, show):
    """GT coverage vs V, optimizer vs baseline."""
    V_values = sorted(set(r["V"] for r in rows))

    opt_by_V = {v: [] for v in V_values}
    bas_by_V = {v: [] for v in V_values}
    for r in rows:
        opt_by_V[r["V"]].append(float(r["gt_coverage"]))
        if r["baseline_gt_coverage"] != "":
            bas_by_V[r["V"]].append(float(r["baseline_gt_coverage"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    opt_means = [np.mean(opt_by_V[v]) for v in V_values]
    opt_stds  = [np.std(opt_by_V[v])  for v in V_values]
    ax.errorbar(V_values, opt_means, yerr=opt_stds,
                fmt="o-", color="royalblue", linewidth=1.8, capsize=4,
                label="HPRO optimizer")

    if any(bas_by_V[v] for v in V_values):
        bas_means = [np.mean(bas_by_V[v]) if bas_by_V[v] else np.nan for v in V_values]
        bas_stds  = [np.std(bas_by_V[v])  if bas_by_V[v] else 0.0  for v in V_values]
        ax.errorbar(V_values, bas_means, yerr=bas_stds,
                    fmt="s--", color="tomato", linewidth=1.5, capsize=4,
                    label="Greedy (Fibonacci)")

    ax.set_xlabel("Number of viewpoints (V)")
    ax.set_ylabel("GT coverage (fraction)")
    ax.set_title("Multi-viewpoint HPRO: GT coverage vs V")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    path = os.path.join(out_dir, "coverage_vs_V.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close()


def plot_coverage_curves(rows_by_V, out_dir, show):
    """Soft coverage training curve per V value (averaged over meshes/seeds)."""
    for V, v_rows in rows_by_V.items():
        opt_results = [r["opt_result"] for r in v_rows if "opt_result" in r]
        if not opt_results:
            continue

        steps = opt_results[0].steps_log
        curves = np.array([r.coverage_history for r in opt_results
                           if len(r.coverage_history) == len(steps)])
        if len(curves) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        mean_c = curves.mean(axis=0)
        std_c  = curves.std(axis=0)
        ax.plot(steps, mean_c, color="royalblue", linewidth=1.8, label="mean C")
        ax.fill_between(steps, mean_c - std_c, mean_c + std_c,
                        color="royalblue", alpha=0.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Soft coverage C")
        ax.set_title(f"Soft coverage during optimisation (V={V})")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.35)
        ax.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, f"coverage_curve_V{V}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        if show:
            plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh_dir",       default="hpro",
                   help="Directory with mesh files (or path to single mesh).")
    p.add_argument("--num_points",     type=int, default=3000,
                   help="Surface sample count per mesh.")
    p.add_argument("--seed",           type=int, default=17)
    p.add_argument("--num_viewpoints", default="1,2,5,10",
                   help="Comma-separated V values to sweep.")
    p.add_argument("--n_steps",        type=int, default=500)
    p.add_argument("--lr",             type=float, default=1e-2)
    p.add_argument("--lambda_standoff", type=float, default=1.0)
    p.add_argument("--lambda_diversity", type=float, default=0.1)
    p.add_argument("--near",           type=float, default=0.1)
    p.add_argument("--far",            type=float, default=6.0)
    p.add_argument("--fov_h",          type=float, default=30.0,
                   help="Horizontal FOV (degrees).")
    p.add_argument("--fov_v",          type=float, default=35.0,
                   help="Vertical FOV (degrees).")
    p.add_argument("--gamma",          type=float, default=None,
                   help="HPRO gamma (default: -exp(-7)).")
    p.add_argument("--k",              type=int, default=10)
    p.add_argument("--min_inter_dist", type=float, default=0.3)
    p.add_argument("--n_candidates",   type=int, default=100,
                   help="Fibonacci candidates for greedy baseline.")
    p.add_argument("--max_meshes",     type=int, default=0,
                   help="Limit number of meshes (0 = all).")
    p.add_argument("--out",            default=os.path.join(_DIR, "results"),
                   help="Output directory.")
    p.add_argument("--device",         default="auto",
                   help="'auto', 'cuda', or 'cpu'.")
    p.add_argument("--no_show",        action="store_true",
                   help="Do not call plt.show() (headless).")
    p.add_argument("--no_baseline",    action="store_true",
                   help="Skip the greedy baseline.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    gamma = args.gamma if args.gamma is not None else -math.exp(-7.0)
    fov_h = math.radians(args.fov_h)
    fov_v = math.radians(args.fov_v)
    V_values = [int(v.strip()) for v in args.num_viewpoints.split(",")]

    cfg = dict(
        gamma=gamma, k=args.k,
        n_steps=args.n_steps, lr=args.lr,
        lambda_standoff=args.lambda_standoff,
        lambda_diversity=args.lambda_diversity,
        near=args.near, far=args.far,
        fov_h=fov_h, fov_v=fov_v,
        min_inter_dist=args.min_inter_dist,
    )

    print(f"Loading meshes from {args.mesh_dir!r} ...")
    meshes = load_meshes(args.mesh_dir, max_meshes=args.max_meshes, seed=args.seed)
    print(f"  Found {len(meshes)} mesh(es).")

    # Build model once (reused across meshes / V values)
    model = HPRO_limited(
        fits_in_memory=True,
        visibility_score_thresh=0.5,
        fov_h=fov_h,
        fov_v=fov_v,
        near=args.near,
        far=args.far,
        frustum_sharpness=50.0,
        device=device,
    )

    csv_path = os.path.join(args.out, "eval_multi.csv")
    rows_all = []          # for plots
    rows_by_V: dict = {v: [] for v in V_values}

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for mesh_name, mesh in meshes:
            print(f"\n{'='*55}")
            print(f"Mesh: {mesh_name}")

            np.random.seed(args.seed)
            pts_np, _ = trimesh.sample.sample_surface(
                mesh, count=args.num_points, face_weight=None, sample_color=False
            )   # (N, 3) float64
            N = len(pts_np)

            for V in V_values:
                print(f"\n  V={V} ...")

                # --- Optimizer -------------------------------------------
                print(f"    Running optimizer ...")
                opt = eval_optimizer(mesh_name, mesh, pts_np, V, model, cfg, args.seed)

                # --- Greedy baseline -------------------------------------
                bas_coverage = ""
                bas_time = ""
                if not args.no_baseline:
                    print(f"    Running greedy baseline ...")
                    covered_bas, t_bas = greedy_baseline(
                        mesh, pts_np, V,
                        fov_h, fov_v, args.near, args.far,
                        n_candidates=args.n_candidates,
                        seed=args.seed,
                    )
                    bas_coverage = len(covered_bas) / N if N > 0 else 0.0
                    bas_time = t_bas

                row = {
                    "mesh": mesh_name,
                    "seed": args.seed,
                    "V": V,
                    "n_points": N,
                    "gt_coverage": f"{opt['gt_coverage']:.4f}",
                    "baseline_gt_coverage": f"{bas_coverage:.4f}" if bas_coverage != "" else "",
                    "soft_coverage_final": f"{opt['soft_coverage_final']:.4f}",
                    "wall_time_s": f"{opt['wall_time_s']:.2f}",
                    "baseline_wall_time_s": f"{bas_time:.2f}" if bas_time != "" else "",
                    "lambda_standoff": cfg["lambda_standoff"],
                    "lambda_diversity": cfg["lambda_diversity"],
                    "gamma": f"{cfg['gamma']:.6f}",
                    "k": cfg["k"],
                    "n_steps": cfg["n_steps"],
                }
                writer.writerow(row)
                f.flush()

                # Stash opt_result for curve plots
                plot_row = dict(row)
                plot_row["opt_result"] = opt["opt_result"]
                rows_all.append(plot_row)
                rows_by_V[V].append(plot_row)

                print(
                    f"    GT coverage (optimizer): {opt['gt_coverage']:.3f}  "
                    + (f"(baseline: {bas_coverage:.3f})" if bas_coverage != "" else "")
                )

    print(f"\nCSV saved to {csv_path}")

    # ---- Plots -------------------------------------------------------------
    show = not args.no_show
    plot_coverage_vs_V(rows_all, args.out, show)
    plot_coverage_curves(rows_by_V, args.out, show)
    print("Done.")


if __name__ == "__main__":
    main()
