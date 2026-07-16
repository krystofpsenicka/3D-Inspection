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
from frustum_gt import build_camera_frame, points_inside_frustum, compute_ground_truth_batched  # noqa: E402


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

# delta (Eq. 7) and alpha (Eq. 8) default to *off*, matching the values every
# result so far was produced with; the sweeps below measure what enabling them
# buys. See ablate_delta_alpha() for the semantics of alpha = 0.
DEF_DELTA = 0.0
DEF_ALPHA = 0.0

# One-at-a-time sweep grids (each varies one param, others held at default).
SHARPNESS_SWEEP = [10.0, 20.0, 50.0, 100.0, 200.0]
K_SWEEP = [5, 10, 20, 40]
THRESH_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# delta shifts the tested point toward the camera by delta before scoring, so a
# point that noise pushed just inside the surface can still register as visible.
# Katz & Tal report the optimum at delta = (noise radius) + 1 % and show it helps
# even on clean data (their Fig. 10a peaks near 0.01-0.015 and decays past 0.03).
# Meshes here are normalised into a unit sphere, so delta is in those units and
# the paper's "1 %" reads directly as 0.01.
DELTA_SWEEP = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
# alpha places a second projection centre at C* = alpha*M + (1-alpha)*C (M = the
# transformed cloud's centre of mass), giving silhouette points a second chance
# to be extremal. Both ends of [0, 1] are no-ops: alpha = 0 puts C* on the camera
# (the paper: "using one direction is equivalent to setting alpha = 0"), and
# alpha = 1 empirically changes no point's score. The paper's Fig. 11 peaks at
# small alpha (~0.05 at 5k points) and wants smaller alpha for denser clouds.
ALPHA_SWEEP = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]

CSV_FIELDS = [
    "mesh", "pose_idx", "method", "gamma", "sharpness", "k", "thresh",
    "delta", "alpha",
    "n_points", "n_gt", "n_pred", "tp", "precision", "recall", "f1", "iou",
    "op_time_s",
]


# ---------------------------------------------------------------------------
# Mesh / pose helpers
# ---------------------------------------------------------------------------

def load_meshes(mesh_dir, max_meshes=0, seed=0):
    """Load and unit-normalise meshes under ``mesh_dir`` (or a single file).

    If ``max_meshes > 0`` and more are found, a deterministic random subset of
    that size is selected (for tuning on a manageable slice of a large set).
    """
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

def hpro_limited_scores(model, pts_t, vp_t, rot_t, gamma, sharpness, k,
                        delta=DEF_DELTA, alpha=DEF_ALPHA):
    """Run HPRO_limited once and return (w_combined numpy (N,), op_time_s).

    ``alpha = 0`` is passed as ``alphas=[]`` rather than ``alphas=[0.0]``: the two
    agree to float64 round-off (C* lands on the camera, so the second direction
    collapses onto the first), and skipping the pass avoids paying for a second
    O(N^2) projection that provably cannot change the score.
    """
    alphas = [] if alpha == 0.0 else [alpha]
    if torch.cuda.is_available() and pts_t.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _, _, w = model(pts_t, vp_t, rot_t, gamma=gamma,
                        frustum_sharpness=sharpness, k=k,
                        delta=delta, alphas=alphas)
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

#: The combo every one-at-a-time sweep passes through. Used to pin the
#: non-swept parameters when aggregating.
DEFAULT_COMBO = dict(gamma=DEF_GAMMA, sharpness=DEF_SHARPNESS, k=DEF_K,
                     thresh=DEF_THRESH, delta=DEF_DELTA, alpha=DEF_ALPHA)


def _fixed_except(key):
    """Pin every swept parameter to its default except ``key`` (the x-axis).

    Each sweep holds the other parameters at their defaults, so every sweep's
    rows collide at the default combo -- e.g. a delta-sweep row also has
    sharpness = DEF_SHARPNESS. Pinning only *some* parameters would therefore mix
    rows from other sweeps into the shared bucket and silently bias it, so the
    filter is built from the full default combo rather than listed per call.
    """
    return {k: v for k, v in DEFAULT_COMBO.items() if k != key}


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
    xs_l, m_l, s_l = _agg_f1_by(rows, "gamma", _fixed_except("gamma"),
                                method="hpro_limited")
    # The baseline has no soft frustum or threshold (both NaN in its rows), so it
    # cannot be filtered by the full default combo -- pin only what it defines.
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


def ablate_delta_alpha(rows, default_f1):
    """Print the delta/alpha ablation: does enabling either beat delta=alpha=0?

    Both are implemented but have been disabled in every result so far
    (RESEARCH_PLAN.md §2 flagged them as "free accuracy on the shelf"). Katz & Tal
    report delta > 0 helping even on clean data and alpha adding ~1 % on
    silhouettes -- this is where that claim is tested on *our* frustum-gated
    operator and mesh, rather than assumed.

    Paired over (mesh, pose): each swept value is compared against the default
    combo on the same poses, so the delta is not confounded by pose difficulty.
    """
    print("\n=== delta / alpha ablation (Stage-1 harness) ===")
    print(f"default (delta=0, alpha=0): F1 = {default_f1:.4f}")

    for key, sweep in (("delta", DELTA_SWEEP), ("alpha", ALPHA_SWEEP)):
        fixed = _fixed_except(key)
        # Pair by (mesh, pose) so each value is scored on the same poses.
        by_val: dict = {}
        for r in rows:
            if r["method"] != "hpro_limited":
                continue
            if any(abs(float(r[k]) - v) > 1e-12 for k, v in fixed.items()):
                continue
            by_val.setdefault(float(r[key]), {})[(r["mesh"], r["pose_idx"])] = r["f1"]
        if not by_val:
            continue

        base_key = DEFAULT_COMBO[key]
        base = by_val.get(base_key, {})
        if not base:
            continue

        stats = {}
        for val, d in by_val.items():
            common = sorted(set(d) & set(base))
            if not common:
                continue
            diff = np.array([d[c] - base[c] for c in common])
            stats[val] = (np.array([d[c] for c in common]), diff)

        gains = {v: s[1].mean() for v, s in stats.items() if v != base_key}
        # Only flag a winner if it actually beats the default; otherwise the
        # "best" of a set of regressions reads as a recommendation.
        best = max(gains, key=gains.get) if gains else None
        if best is not None and gains[best] <= 0:
            best = None

        print(f"\n  {key:>5}   F1 (mean±std)        vs default   poses improved")
        for val in sweep:
            if val not in stats:
                continue
            vals, diff = stats[val]
            better = int((diff > 1e-9).sum())
            mark = "  <-- best" if val == best else ""
            print(f"  {val:>5.3f}   {vals.mean():.4f} ± {vals.std():.4f}   "
                  f"{diff.mean():+.4f}      {better:>3d}/{len(diff)}{mark}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_combos():
    """(gamma, sharpness, k, delta, alpha) at which to compute w.

    One-at-a-time sweeps: each grid varies a single parameter with the rest held
    at their defaults, so every row is comparable to the default combo.
    """
    base = (DEF_GAMMA, DEF_SHARPNESS, DEF_K, DEF_DELTA, DEF_ALPHA)
    combos = {base}
    combos |= {(g, DEF_SHARPNESS, DEF_K, DEF_DELTA, DEF_ALPHA) for g in GAMMA_SWEEP}
    combos |= {(DEF_GAMMA, s, DEF_K, DEF_DELTA, DEF_ALPHA) for s in SHARPNESS_SWEEP}
    combos |= {(DEF_GAMMA, DEF_SHARPNESS, k, DEF_DELTA, DEF_ALPHA) for k in K_SWEEP}
    combos |= {(DEF_GAMMA, DEF_SHARPNESS, DEF_K, d, DEF_ALPHA) for d in DELTA_SWEEP}
    combos |= {(DEF_GAMMA, DEF_SHARPNESS, DEF_K, DEF_DELTA, a) for a in ALPHA_SWEEP}
    return sorted(combos)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh_dir", default="hpro",
                    help="Directory of meshes or a single mesh file "
                         "(default: hpro/ -> bundled lamp_0001.off).")
    ap.add_argument("--num_points", type=int, default=10000)
    ap.add_argument("--num_poses", type=int, default=20)
    ap.add_argument("--max_meshes", type=int, default=0,
                    help="If >0, randomly subsample this many meshes from --mesh_dir.")
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

    meshes = load_meshes(args.mesh_dir, max_meshes=args.max_meshes, seed=args.seed)
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
        intersector = mesh.ray   # Embree BVH (ray_pyembree) if embreex present

        for pose_idx, (vp_np, look_approx, up_approx) in enumerate(poses):
            look, up, right = build_camera_frame(look_approx, up_approx)

            # Ground truth: inside frustum AND unoccluded (batched ray-cast vs mesh).
            gt_idx, frustum_idx = compute_ground_truth_batched(
                mesh, vp_np, pts_np, look, up, right,
                cfg["fov_h"], cfg["fov_v"], cfg["near"], cfg["far"],
                intersector=intersector,
            )
            if len(gt_idx) == 0:
                continue   # pose sees nothing — skip to avoid degenerate metrics

            vp_t = torch.tensor(vp_np[np.newaxis], dtype=torch.float64, device=device)
            rot_t = torch.tensor(np.concatenate([look_approx, up_approx])[np.newaxis],
                                 dtype=torch.float64, device=device)

            # HPRO_limited: compute w once per combo; threshold post-hoc.
            for (gamma, sharpness, k, delta, alpha) in combos:
                w, dt = hpro_limited_scores(model, pts_t, vp_t, rot_t, gamma,
                                            sharpness, k, delta, alpha)
                is_default_combo = (abs(gamma - DEF_GAMMA) < 1e-12
                                    and sharpness == DEF_SHARPNESS and k == DEF_K
                                    and delta == DEF_DELTA and alpha == DEF_ALPHA)
                thr_list = THRESH_SWEEP if is_default_combo else [DEF_THRESH]
                for thr in thr_list:
                    pred = np.where(w > thr)[0]
                    p, r, f1, iou, tp = score(pred, gt_idx)
                    rows.append(dict(
                        mesh=mesh_name, pose_idx=pose_idx, method="hpro_limited",
                        gamma=gamma, sharpness=sharpness, k=k, thresh=thr,
                        delta=delta, alpha=alpha,
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
                        delta=DEF_DELTA, alpha=DEF_ALPHA,
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

    default_f1 = mean_f1("hpro_limited", **DEFAULT_COMBO)
    # Fair baseline reference: same (tuned) gamma as the HPRO_limited default.
    baseline_f1 = mean_f1("baseline", gamma=DEF_GAMMA) if not args.no_baseline else None

    # ---- Plots --------------------------------------------------------------
    plot_gamma(rows, os.path.join(args.out, "f1_vs_gamma.png"))
    plot_sweep(rows, "sharpness", _fixed_except("sharpness"),
               "frustum_sharpness", os.path.join(args.out, "f1_vs_sharpness.png"),
               baseline_f1=baseline_f1)
    plot_sweep(rows, "thresh", _fixed_except("thresh"),
               "visibility_score_thresh", os.path.join(args.out, "f1_vs_thresh.png"),
               baseline_f1=baseline_f1)
    plot_sweep(rows, "k", _fixed_except("k"),
               "k (top-k)", os.path.join(args.out, "f1_vs_k.png"),
               baseline_f1=baseline_f1)
    plot_sweep(rows, "delta", _fixed_except("delta"),
               "delta (noise offset, Eq. 7)",
               os.path.join(args.out, "f1_vs_delta.png"),
               baseline_f1=baseline_f1)
    plot_sweep(rows, "alpha", _fixed_except("alpha"),
               "alpha (second direction, Eq. 8; 0 = off)",
               os.path.join(args.out, "f1_vs_alpha.png"),
               baseline_f1=baseline_f1)

    # ---- delta/alpha ablation (§7 step 1) -----------------------------------
    ablate_delta_alpha(rows, default_f1)

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
        f"k={DEF_K}  thresh={DEF_THRESH}  delta={DEF_DELTA}  alpha={DEF_ALPHA}",
        "",
    ]
    # Pin the full default combo: without delta/alpha pinned this silently
    # averaged every delta/alpha sweep row into the "default" headline.
    d = headline("hpro_limited", **DEFAULT_COMBO)
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

    # Best HPRO_limited config by mean F1 across the swept points. delta/alpha
    # belong in the key: without them, configs differing only in delta/alpha
    # collapse into one bucket and get averaged together.
    by_cfg = {}
    for r in rows:
        if r["method"] != "hpro_limited":
            continue
        key = (r["gamma"], r["sharpness"], r["k"], r["thresh"], r["delta"], r["alpha"])
        by_cfg.setdefault(key, []).append(r["f1"])
    best = max(by_cfg.items(), key=lambda kv: np.mean(kv[1]))
    summary_lines.append(
        f"\nbest swept config: gamma={best[0][0]:.5g} sharpness={best[0][1]} "
        f"k={best[0][2]} thresh={best[0][3]} delta={best[0][4]} alpha={best[0][5]} "
        f"-> mean F1={np.mean(best[1]):.3f}")

    summary = "\n".join(summary_lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
