"""Quantitative evaluation harness for the frustum-restricted HPRO operator.

Generalises the one-off ``demo_limited.py`` into a reproducible, multi-mesh /
multi-pose evaluator that scores ``HPRO_limited`` against ray-cast ground truth
(inside-frustum AND unoccluded), sweeps the operator's hyper-parameters, and
compares against a non-differentiable baseline (plain HPRO + hard frustum cull).

Outputs (all written under ``--out``, default ``results/``):
  * ``hpro_eval.csv``     — one row per (mesh, pose, method, gamma, sharpness, k, thresh).
  * ``f1_vs_gamma.png``   — F1 vs gamma (mean ± std over poses/meshes).
  * ``f1_vs_sharpness.png``
  * ``f1_vs_thresh.png``
  * ``f1_vs_k.png``
  * ``summary.txt``       — best config, baseline comparison, headline numbers.

Headless-safe (``matplotlib`` Agg backend, no blocking ``show``). Runs on CPU or
GPU; pass ``--device cpu`` to force CPU.

Examples
--------
Quick local sanity (CPU, tiny)::

    python hpro/eval_frustum.py --mesh_dir hpro --num_poses 4 --num_points 2000 --no_show

Full cluster run (GPU)::

    python hpro/eval_frustum.py --no_show --out results/
"""

import argparse
import csv
import glob
import math
import os
import time

import matplotlib
matplotlib.use("Agg")          # headless: must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import trimesh                   # noqa: E402

from HPRO import HPRO                       # noqa: E402
from HPRO_limited import HPRO_limited       # noqa: E402
from frustum_gt import build_camera_frame, points_inside_frustum, compute_ground_truth  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults (centred on the values that work in demo_limited.py)
# ---------------------------------------------------------------------------

# gamma must be negative and — for frustum-limited dense views — close to 0
# (the HPRO paper notes γ should sit "slightly closer to 0"; empirically larger
# |γ| collapses recall). The sweep spans e^-3 .. e^-11; the default sits near the
# observed optimum.
GAMMA_EXPS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
GAMMA_SWEEP = [-math.exp(-e) for e in GAMMA_EXPS]
DEF_GAMMA = -math.exp(-7.0)
# Higher sharpness -> sharper frustum boundary -> higher F1 (approaches the hard
# cull), but steeper sigmoids = smaller gradient basins for downstream viewpoint
# optimisation (Stage 2). 50 is a deliberate accuracy/optimizability compromise.
DEF_SHARPNESS = 50.0
DEF_K = 10
DEF_THRESH = 0.5

# One-at-a-time sweep grids (each varies one param, others held at default).
SHARPNESS_SWEEP = [10.0, 20.0, 50.0, 100.0, 200.0]
K_SWEEP = [5, 10, 20, 40]
THRESH_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

CSV_FIELDS = [
    "mesh", "pose_idx", "method", "gamma", "sharpness", "k", "thresh",
    "n_points", "n_gt", "n_pred", "tp", "precision", "recall", "f1", "iou",
    "op_time_s",
]


# ---------------------------------------------------------------------------
# Mesh / pose helpers
# ---------------------------------------------------------------------------

def load_meshes(mesh_dir):
    """Load and unit-normalise every mesh under ``mesh_dir`` (or a single file)."""
    if os.path.isfile(mesh_dir):
        paths = [mesh_dir]
    else:
        paths = []
        for ext in ("*.off", "*.obj", "*.ply", "*.stl"):
            paths.extend(sorted(glob.glob(os.path.join(mesh_dir, ext))))
    if not paths:
        raise FileNotFoundError(f"No meshes found under {mesh_dir!r}")

    meshes = []
    for p in paths:
        m = trimesh.load_mesh(p)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        m.vertices = m.vertices - np.mean(m.vertices, axis=0)
        m.vertices = m.vertices / np.linalg.norm(m.vertices, axis=1).max()
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


def make_poses(num_poses, radius, seed):
    """Viewpoints on a sphere of given radius, each looking at the origin.

    Returns a list of ``(viewpoint(3,), look_approx(3,), up_approx(3,))``.
    The mesh is centred at the origin and unit-scaled, so a fixed radius gives
    a consistent stand-off across meshes.
    """
    rng = np.random.default_rng(seed)
    dirs = fibonacci_sphere(num_poses)
    # Small deterministic jitter so poses are not perfectly symmetric.
    dirs = dirs + 0.05 * rng.standard_normal(dirs.shape)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    poses = []
    world_up = np.array([0.0, 0.0, 1.0])
    for d in dirs:
        vp = radius * d
        look = -d                       # point back toward the centred object
        up = world_up
        if abs(np.dot(look, up)) > 0.95:   # avoid degenerate up ∥ look
            up = np.array([0.0, 1.0, 0.0])
        poses.append((vp, look, up))
    return poses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score(pred_idx, gt_idx):
    """precision, recall, f1, iou, tp from two index iterables."""
    pred, gt = set(int(i) for i in pred_idx), set(int(i) for i in gt_idx)
    tp = len(pred & gt)
    union = len(pred | gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / union if union else 0.0
    return precision, recall, f1, iou, tp


# ---------------------------------------------------------------------------
# Operator evaluation
# ---------------------------------------------------------------------------

def hpro_limited_scores(model, pts_t, vp_t, rot_t, gamma, sharpness, k):
    """Run HPRO_limited once and return (w_combined numpy (N,), op_time_s)."""
    if torch.cuda.is_available() and pts_t.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _, _, w = model(pts_t, vp_t, rot_t, gamma=gamma,
                        frustum_sharpness=sharpness, k=k)
    if torch.cuda.is_available() and pts_t.is_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return w[0].detach().cpu().numpy(), dt


def baseline_scores(hpro, pts_t, vp_t, pts_np, vp_np, look, up, right, cfg, gamma, k):
    """Plain HPRO (full 360 self-occlusion) intersected with a hard frustum cull."""
    if torch.cuda.is_available() and pts_t.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _, idx, _ = hpro(pts_t, vp_t, gamma=gamma, k=k)
    if torch.cuda.is_available() and pts_t.is_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    idx = np.atleast_1d(idx.detach().cpu().numpy().astype(np.int64).ravel())
    in_frustum = points_inside_frustum(pts_np, vp_np, look, up, right,
                                       cfg["fov_h"], cfg["fov_v"], cfg["near"], cfg["far"])
    pred = [i for i in idx if in_frustum[i]]
    return pred, dt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _agg_f1_by(rows, key, fixed, method="hpro_limited"):
    """Mean/std F1 grouped by ``rows[key]`` among rows matching ``fixed`` dict."""
    buckets = {}
    for r in rows:
        if r["method"] != method:
            continue
        if any(abs(float(r[k]) - v) > 1e-12 for k, v in fixed.items()):
            continue
        buckets.setdefault(float(r[key]), []).append(r["f1"])
    xs = sorted(buckets)
    means = [float(np.mean(buckets[x])) for x in xs]
    stds = [float(np.std(buckets[x])) for x in xs]
    return xs, means, stds


def plot_gamma(rows, path):
    """F1 vs |gamma| (log x) for HPRO_limited and the HPRO+hard-cull baseline."""
    fixed = dict(sharpness=DEF_SHARPNESS, k=DEF_K, thresh=DEF_THRESH)
    xs_l, m_l, s_l = _agg_f1_by(rows, "gamma", fixed, method="hpro_limited")
    xs_b, m_b, s_b = _agg_f1_by(rows, "gamma", {"k": DEF_K}, method="baseline")
    if not xs_l:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    # x = |gamma|; both methods share the same gamma grid.
    ax.errorbar([-x for x in xs_l], m_l, yerr=s_l, marker="o", capsize=3,
                color="royalblue", label="HPRO_limited (soft frustum)")
    if xs_b:
        ax.errorbar([-x for x in xs_b], m_b, yerr=s_b, marker="s", capsize=3,
                    color="crimson", label="baseline (HPRO + hard cull)")
    ax.axvline(-DEF_GAMMA, color="grey", linestyle=":", label=f"default |γ|={-DEF_GAMMA:.2g}")
    ax.set_xscale("log")
    ax.set_xlabel("|gamma|  (closer to 0 → right)")
    ax.set_ylabel("F1 (mean ± std over poses)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweep(rows, key, fixed, xlabel, path, logx=False, baseline_f1=None):
    xs, means, stds = _agg_f1_by(rows, key, fixed)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, color="royalblue",
                label="HPRO_limited")
    if baseline_f1 is not None:
        ax.axhline(baseline_f1, color="crimson", linestyle="--",
                   label=f"baseline (HPRO+hard cull) {baseline_f1:.3f}")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1 (mean ± std over poses)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_combos():
    """Set of (gamma, sharpness, k) at which to compute w (one-at-a-time sweeps)."""
    combos = {(DEF_GAMMA, DEF_SHARPNESS, DEF_K)}
    combos |= {(g, DEF_SHARPNESS, DEF_K) for g in GAMMA_SWEEP}
    combos |= {(DEF_GAMMA, s, DEF_K) for s in SHARPNESS_SWEEP}
    combos |= {(DEF_GAMMA, DEF_SHARPNESS, k) for k in K_SWEEP}
    return sorted(combos)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh_dir", default="hpro",
                    help="Directory of meshes or a single mesh file "
                         "(default: hpro/ -> bundled lamp_0001.off).")
    ap.add_argument("--num_points", type=int, default=10000)
    ap.add_argument("--num_poses", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--radius", type=float, default=2.0,
                    help="Stand-off distance of viewpoints from the centred object.")
    ap.add_argument("--fov_h", type=float, default=30.0, help="degrees")
    ap.add_argument("--fov_v", type=float, default=35.0, help="degrees")
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--far", type=float, default=2.5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--no_show", action="store_true",
                    help="Accepted for symmetry; harness never blocks on show().")
    ap.add_argument("--no_baseline", action="store_true",
                    help="Skip the plain-HPRO + hard-cull baseline.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = dict(fov_h=math.radians(args.fov_h), fov_v=math.radians(args.fov_v),
               near=args.near, far=args.far)

    meshes = load_meshes(args.mesh_dir)
    poses = make_poses(args.num_poses, args.radius, args.seed)
    combos = build_combos()

    print(f"device={device}  meshes={len(meshes)}  poses={len(poses)}  "
          f"combos={len(combos)}  points={args.num_points}")

    model = HPRO_limited(fits_in_memory=True, visibility_score_thresh=DEF_THRESH,
                         device=device, **cfg)
    hpro_plain = HPRO(fits_in_memory=True, device=device)

    rows = []
    for mesh_name, mesh in meshes:
        pts_np, _ = trimesh.sample.sample_surface(mesh, count=args.num_points)
        pts_np = np.asarray(pts_np, dtype=np.float64)
        pts_t = torch.tensor(pts_np.T[np.newaxis], dtype=torch.float64, device=device)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        for pose_idx, (vp_np, look_approx, up_approx) in enumerate(poses):
            look, up, right = build_camera_frame(look_approx, up_approx)

            # Ground truth: inside frustum AND unoccluded (ray-cast vs mesh).
            gt_idx, frustum_idx = compute_ground_truth(
                mesh, vp_np, pts_np, look, up, right,
                cfg["fov_h"], cfg["fov_v"], cfg["near"], cfg["far"],
                intersector=intersector,
            )
            if len(gt_idx) == 0:
                continue   # pose sees nothing — skip to avoid degenerate metrics

            vp_t = torch.tensor(vp_np[np.newaxis], dtype=torch.float64, device=device)
            rot_t = torch.tensor(np.concatenate([look_approx, up_approx])[np.newaxis],
                                 dtype=torch.float64, device=device)

            # HPRO_limited: compute w once per (gamma, sharpness, k); threshold post-hoc.
            for (gamma, sharpness, k) in combos:
                w, dt = hpro_limited_scores(model, pts_t, vp_t, rot_t, gamma, sharpness, k)
                is_default_combo = (abs(gamma - DEF_GAMMA) < 1e-12
                                    and sharpness == DEF_SHARPNESS and k == DEF_K)
                thr_list = THRESH_SWEEP if is_default_combo else [DEF_THRESH]
                for thr in thr_list:
                    pred = np.where(w > thr)[0]
                    p, r, f1, iou, tp = score(pred, gt_idx)
                    rows.append(dict(
                        mesh=mesh_name, pose_idx=pose_idx, method="hpro_limited",
                        gamma=gamma, sharpness=sharpness, k=k, thresh=thr,
                        n_points=args.num_points, n_gt=len(gt_idx), n_pred=len(pred),
                        tp=tp, precision=p, recall=r, f1=f1, iou=iou, op_time_s=dt,
                    ))

            # Baseline: plain HPRO (360) + hard frustum cull, at every swept gamma
            # (so the soft vs hard frustum comparison is fair across gamma).
            if not args.no_baseline:
                for gamma in GAMMA_SWEEP:
                    pred_b, dt_b = baseline_scores(
                        hpro_plain, pts_t, vp_t, pts_np, vp_np, look, up, right,
                        cfg, gamma, DEF_K)
                    p, r, f1, iou, tp = score(pred_b, gt_idx)
                    rows.append(dict(
                        mesh=mesh_name, pose_idx=pose_idx, method="baseline",
                        gamma=gamma, sharpness=float("nan"), k=DEF_K, thresh=float("nan"),
                        n_points=args.num_points, n_gt=len(gt_idx), n_pred=len(pred_b),
                        tp=tp, precision=p, recall=r, f1=f1, iou=iou, op_time_s=dt_b,
                    ))

        print(f"  [{mesh_name}] done ({len([r for r in rows if r['mesh']==mesh_name])} rows)")

    # ---- Write CSV ----------------------------------------------------------
    csv_path = os.path.join(args.out, "hpro_eval.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # ---- Headline numbers ---------------------------------------------------
    def mean_f1(method, **fixed):
        vals = [r["f1"] for r in rows if r["method"] == method
                and all(abs(float(r[k]) - v) < 1e-12 for k, v in fixed.items())]
        return float(np.mean(vals)) if vals else float("nan")

    default_f1 = mean_f1("hpro_limited", gamma=DEF_GAMMA, sharpness=DEF_SHARPNESS,
                         k=DEF_K, thresh=DEF_THRESH)
    # Fair baseline reference: same (tuned) gamma as the HPRO_limited default.
    baseline_f1 = mean_f1("baseline", gamma=DEF_GAMMA) if not args.no_baseline else None

    # ---- Plots --------------------------------------------------------------
    plot_gamma(rows, os.path.join(args.out, "f1_vs_gamma.png"))
    plot_sweep(rows, "sharpness",
               dict(gamma=DEF_GAMMA, k=DEF_K, thresh=DEF_THRESH),
               "frustum_sharpness", os.path.join(args.out, "f1_vs_sharpness.png"),
               baseline_f1=baseline_f1)
    plot_sweep(rows, "thresh",
               dict(gamma=DEF_GAMMA, sharpness=DEF_SHARPNESS, k=DEF_K),
               "visibility_score_thresh", os.path.join(args.out, "f1_vs_thresh.png"),
               baseline_f1=baseline_f1)
    plot_sweep(rows, "k",
               dict(gamma=DEF_GAMMA, sharpness=DEF_SHARPNESS, thresh=DEF_THRESH),
               "k (top-k)", os.path.join(args.out, "f1_vs_k.png"),
               baseline_f1=baseline_f1)

    # ---- Summary ------------------------------------------------------------
    def headline(method, **fixed):
        sel = [r for r in rows if r["method"] == method
               and all(abs(float(r[k]) - v) < 1e-12 for k, v in fixed.items())]
        if not sel:
            return None
        agg = {m: float(np.mean([r[m] for r in sel]))
               for m in ("precision", "recall", "f1", "iou", "op_time_s")}
        agg["n"] = len(sel)
        return agg

    summary_lines = [
        "HPRO_limited frustum evaluation summary",
        "=" * 42,
        f"device={device}  meshes={len(meshes)}  poses={len(poses)}  "
        f"points={args.num_points}",
        f"default config: gamma={DEF_GAMMA:.5g}  sharpness={DEF_SHARPNESS}  "
        f"k={DEF_K}  thresh={DEF_THRESH}",
        "",
    ]
    d = headline("hpro_limited", gamma=DEF_GAMMA, sharpness=DEF_SHARPNESS,
                 k=DEF_K, thresh=DEF_THRESH)
    if d:
        summary_lines.append(
            f"HPRO_limited (default): P={d['precision']:.3f} R={d['recall']:.3f} "
            f"F1={d['f1']:.3f} IoU={d['iou']:.3f}  t={d['op_time_s']*1e3:.1f}ms  (n={d['n']})")
    if baseline_f1 is not None:
        b = headline("baseline", gamma=DEF_GAMMA)
        summary_lines.append(
            f"baseline HPRO+cull   : P={b['precision']:.3f} R={b['recall']:.3f} "
            f"F1={b['f1']:.3f} IoU={b['iou']:.3f}  t={b['op_time_s']*1e3:.1f}ms  (n={b['n']})")
        summary_lines.append(
            f"  (both at tuned default gamma={DEF_GAMMA:.5g})")
        # Best baseline across the gamma sweep, for reference.
        base_by_g = {}
        for r in rows:
            if r["method"] == "baseline":
                base_by_g.setdefault(r["gamma"], []).append(r["f1"])
        if base_by_g:
            bg = max(base_by_g.items(), key=lambda kv: np.mean(kv[1]))
            summary_lines.append(
                f"  best baseline over gamma: F1={np.mean(bg[1]):.3f} at gamma={bg[0]:.5g}")
        summary_lines.append(
            f"\nF1 gap (HPRO_limited - baseline) at default gamma: {default_f1 - baseline_f1:+.3f}")

    # Best HPRO_limited config by mean F1 across the swept points.
    by_cfg = {}
    for r in rows:
        if r["method"] != "hpro_limited":
            continue
        key = (r["gamma"], r["sharpness"], r["k"], r["thresh"])
        by_cfg.setdefault(key, []).append(r["f1"])
    best = max(by_cfg.items(), key=lambda kv: np.mean(kv[1]))
    summary_lines.append(
        f"\nbest swept config: gamma={best[0][0]:.5g} sharpness={best[0][1]} "
        f"k={best[0][2]} thresh={best[0][3]} -> mean F1={np.mean(best[1]):.3f}")

    summary = "\n".join(summary_lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
