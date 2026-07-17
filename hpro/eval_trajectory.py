"""Stage B evaluation — receding-horizon trajectories vs. sample-and-select.

The go/no-go experiment for the paper's headline (RESEARCH_PLAN.md §6.1/§7 Stage
B): does continuous gradient planning of an *ordered path* beat the discrete
decomposition (sample candidates -> set cover -> route) on **coverage per metre
travelled**, and is warm-started replanning cheaper than re-solving?

The baseline is deliberately strong, because a weak one proves nothing:

* candidates are generated the way an inspection pipeline actually does it --
  offset along surface normals at mid-standoff, not on a sphere (a sphere is
  hopeless for a concave wreck under a tight camera);
* selection is **oracle** greedy set-cover over hard ray-cast visibility, i.e.
  it sees the true answer our planner only estimates, and inherits greedy's
  (1-1/e) guarantee;
* routing is nearest-neighbour + 2-opt, not selection order.

So the baseline is an upper reference on *selection* quality. If the continuous
planner wins, it wins because it is not restricted to the candidate set and
because it pays for motion while planning -- not because the baseline was
crippled.

Usage::

    ipy hpro/eval_trajectory.py --mesh models/duke_of_lancaster_uk_clipped.glb \\
        --backbones nvps,hpro --seeds 3 --no_show
"""

import argparse
import csv
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import trimesh                    # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from backbones import make_backbone                              # noqa: E402
from frustum_gt import build_camera_frame, compute_ground_truth_batched  # noqa: E402
from trajectory import TrajectoryConfig, receding_horizon_plan   # noqa: E402
from visibility_layer import GatedVisibilityLayer                # noqa: E402

CSV_FIELDS = [
    "mesh", "seed", "method", "backbone", "n_points",
    "gt_coverage", "path_length", "coverage_per_m", "n_poses",
    "wall_time_s", "replan_cold_ms", "replan_warm_ms", "audit_gap_mean",
]


# ---------------------------------------------------------------------------
# Target / camera
# ---------------------------------------------------------------------------

def load_mesh(path):
    m = trimesh.load_mesh(path)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    m.vertices -= np.mean(m.vertices, axis=0)
    m.vertices /= np.linalg.norm(m.vertices, axis=1).max()
    return m


def make_audit_fn(mesh, pts_np, cam):
    """Hard ray-cast audit: which points does this pose *really* see?

    Offline this is the mesh; online it would be the sensor. Same signature --
    that symmetry is the point of §6.4 item 6.
    """
    intersector = mesh.ray

    def audit(pos, r6):
        look, up, right = build_camera_frame(r6[:3], r6[3:])
        idx, _ = compute_ground_truth_batched(
            mesh, pos, pts_np, look, up, right,
            cam["fov_h"], cam["fov_v"], cam["near"], cam["far"],
            intersector=intersector)
        return set(int(i) for i in idx)

    return audit


# ---------------------------------------------------------------------------
# Discrete baseline: candidates -> oracle greedy set cover -> route
# ---------------------------------------------------------------------------

def normal_offset_candidates(pts_np, normals_np, n_cand, standoff, seed):
    """Candidate poses offset along surface normals, looking back at the surface.

    This is how sample-and-select inspection planners generate candidates, and
    the only sensible choice under a tight camera: a Fibonacci sphere at a fixed
    radius cannot reach into a concave wreck's standoff band at all.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts_np), size=min(n_cand, len(pts_np)), replace=False)
    n = normals_np[idx]
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    pos = pts_np[idx] + n * standoff
    look = -n                                   # look back at the surface
    world_up = np.array([0.0, 0.0, 1.0])
    up = np.tile(world_up, (len(pos), 1))
    # Fall back for viewpoints whose gaze is nearly collinear with world up.
    bad = np.abs(look @ world_up) > 0.95
    up[bad] = np.array([0.0, 1.0, 0.0])
    return pos, np.concatenate([look, up], axis=1)


def greedy_select(cand_gt, budget):
    """Oracle greedy set cover. Returns the selected candidate indices, in order."""
    covered, sel = set(), []
    for _ in range(budget):
        best_gain, best_j = 0, None
        for j, g in enumerate(cand_gt):
            if j in sel:
                continue
            gain = len(g - covered)
            if gain > best_gain:
                best_gain, best_j = gain, j
        if best_j is None:          # no candidate adds anything
            break
        covered |= cand_gt[best_j]
        sel.append(best_j)
    return sel


def route_nn_2opt(start, positions):
    """Open-path tour: nearest-neighbour construction + 2-opt improvement.

    Routing matters: greedy's *selection* order is not a sensible flight order,
    and scoring the baseline on selection order would inflate its path length and
    hand us an unearned win.
    """
    n = len(positions)
    if n == 0:
        return [], 0.0
    pts = np.vstack([start[None], positions])            # index 0 = start
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)

    unvisited = set(range(1, n + 1))
    tour, cur = [], 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    def tour_len(t):
        seq = [0] + t
        return float(D[seq[:-1], seq[1:]].sum())

    best = tour_len(tour)
    improved = True
    while improved:                       # 2-opt on the open path
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                L = tour_len(cand)
                if L < best - 1e-9:
                    tour, best, improved = cand, L, True
    return [t - 1 for t in tour], best


def run_discrete(mesh, pts_np, normals_np, cam, cfg, audit, start, args, seed):
    """Candidates -> oracle greedy -> route, swept over the viewpoint budget."""
    standoff = 0.5 * (cfg.near_dist + cfg.far_dist)
    cand_pos, cand_r6 = normal_offset_candidates(
        pts_np, normals_np, args.n_candidates, standoff, seed)

    t0 = time.perf_counter()
    cand_gt = [audit(cand_pos[j], cand_r6[j]) for j in range(len(cand_pos))]
    t_raycast = time.perf_counter() - t0

    out = []
    for budget in args.budgets:
        t1 = time.perf_counter()
        sel = greedy_select(cand_gt, budget)
        order, length = route_nn_2opt(start, cand_pos[sel])
        t_solve = time.perf_counter() - t1

        covered = set()
        for j in sel:
            covered |= cand_gt[j]
        out.append(dict(
            budget=budget, gt_coverage=len(covered) / len(pts_np),
            path_length=length, n_poses=len(sel),
            # The ray-cast is amortised across budgets: one pass builds the
            # candidate visibility sets that every budget reuses.
            wall_time_s=t_solve + t_raycast / len(args.budgets),
        ))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=os.path.join(
        os.path.dirname(_DIR), "models", "duke_of_lancaster_uk_clipped.glb"))
    p.add_argument("--num_points", type=int, default=3000)
    p.add_argument("--backbones", default="nvps,hpro",
                   help="Comma-separated: hpro, nvps, ensemble.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max_cycles", type=int, default=40)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--replan_steps", type=int, default=30)
    p.add_argument("--n_candidates", type=int, default=300)
    p.add_argument("--budgets", default="5,10,20,30,40",
                   help="Viewpoint budgets for the discrete baseline.")
    p.add_argument("--fov_h", type=float, default=30.0)
    p.add_argument("--fov_v", type=float, default=35.0)
    p.add_argument("--near", type=float, default=0.1)
    p.add_argument("--far", type=float, default=1.5,
                   help="Tight camera (§6.3): one view should see a few percent.")
    p.add_argument("--near_dist", type=float, default=0.5)
    p.add_argument("--far_dist", type=float, default=1.5)
    p.add_argument("--out", default=os.path.join(_DIR, "results", "trajectory"))
    p.add_argument("--device", default="auto")
    p.add_argument("--no_guide", action="store_true",
                   help="Disable the two-timescale global guide (ablation).")
    p.add_argument("--no_show", action="store_true")
    a = p.parse_args()
    a.budgets = [int(x) for x in a.budgets.split(",")]
    a.backbones = [x.strip() for x in a.backbones.split(",")]
    return a


def main():
    import torch
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    cam = dict(fov_h=math.radians(args.fov_h), fov_v=math.radians(args.fov_v),
               near=args.near, far=args.far)
    mesh = load_mesh(args.mesh)
    mesh_name = os.path.basename(args.mesh)

    rows, curves = [], {}
    csv_path = os.path.join(args.out, "eval_trajectory.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for seed in range(args.seeds):
            np.random.seed(seed)
            pts_np, fidx = trimesh.sample.sample_surface(mesh, count=args.num_points)
            pts_np = np.asarray(pts_np, dtype=np.float64)
            normals_np = np.asarray(mesh.face_normals[fidx], dtype=np.float64)
            audit = make_audit_fn(mesh, pts_np, cam)

            cfg = TrajectoryConfig(
                horizon=args.horizon, replan_steps=args.replan_steps,
                max_cycles=args.max_cycles, near_dist=args.near_dist,
                far_dist=args.far_dist, seed=seed,
                use_global_guide=not args.no_guide)

            # Common start pose, so path lengths are comparable.
            rng = np.random.default_rng(seed)
            d0 = rng.normal(size=3)
            d0 /= np.linalg.norm(d0)
            start = pts_np.mean(axis=0) + d0 * (1.0 + args.far_dist)

            # ---- discrete baseline ---------------------------------------
            for r in run_discrete(mesh, pts_np, normals_np, cam, cfg, audit,
                                  start, args, seed):
                row = dict(
                    mesh=mesh_name, seed=seed, method=f"greedy+route(V={r['budget']})",
                    backbone="oracle", n_points=len(pts_np),
                    gt_coverage=f"{r['gt_coverage']:.4f}",
                    path_length=f"{r['path_length']:.3f}",
                    coverage_per_m=f"{r['gt_coverage']/max(r['path_length'],1e-9):.4f}",
                    n_poses=r["n_poses"], wall_time_s=f"{r['wall_time_s']:.2f}",
                    replan_cold_ms="", replan_warm_ms="", audit_gap_mean="",
                )
                writer.writerow(row); fh.flush(); rows.append(row)
                curves.setdefault(("greedy+route", seed), []).append(
                    (r["path_length"], r["gt_coverage"]))
                print(f"  [seed {seed}] greedy+route V={r['budget']:<3d} "
                      f"cov={r['gt_coverage']:.3f} len={r['path_length']:.2f}m")

            # ---- continuous planner --------------------------------------
            for bname in args.backbones:
                b = make_backbone(bname, device=device, gamma=-math.exp(-7.0), k=10)
                if b.requires_normals:
                    b.prepare(pts_np, normals_np)
                layer = GatedVisibilityLayer(
                    b, fov_h=cam["fov_h"], fov_v=cam["fov_v"],
                    near=cam["near"], far=cam["far"],
                    frustum_sharpness=50.0, device=device)

                res = receding_horizon_plan(layer, pts_np, audit, cfg,
                                            start_pos=start, device=device)
                warm = (float(np.mean(res.replan_times[1:])) * 1e3
                        if len(res.replan_times) > 1 else float("nan"))
                row = dict(
                    mesh=mesh_name, seed=seed, method="receding-horizon",
                    backbone=bname, n_points=len(pts_np),
                    gt_coverage=f"{res.gt_coverage:.4f}",
                    path_length=f"{res.path_length:.3f}",
                    coverage_per_m=f"{res.gt_coverage/max(res.path_length,1e-9):.4f}",
                    n_poses=res.n_cycles, wall_time_s=f"{res.wall_time_s:.2f}",
                    replan_cold_ms=f"{res.replan_times[0]*1e3:.1f}",
                    replan_warm_ms=f"{warm:.1f}",
                    audit_gap_mean=f"{float(np.mean(res.audit_gap)):.1f}",
                )
                writer.writerow(row); fh.flush(); rows.append(row)
                curves[(f"rh-{bname}", seed)] = list(
                    zip(res.length_curve, res.coverage_curve))
                print(f"  [seed {seed}] rh-{bname:<8s} cov={res.gt_coverage:.3f} "
                      f"len={res.path_length:.2f}m "
                      f"cov/m={res.gt_coverage/max(res.path_length,1e-9):.3f} "
                      f"replan {res.replan_times[0]*1e3:.0f}->{warm:.0f} ms "
                      f"audit_gap {float(np.mean(res.audit_gap)):+.0f}")

    print(f"\nCSV: {csv_path}")
    plot_curves(curves, args.out, not args.no_show)
    summarise(rows)


def plot_curves(curves, out_dir, show):
    """GT coverage vs metres travelled — the headline figure."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"greedy+route": "tomato", "rh-hpro": "royalblue", "rh-nvps": "seagreen",
              "rh-ensemble": "purple"}
    seen = set()
    for (method, seed), pts in sorted(curves.items()):
        if not pts:
            continue
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = colors.get(method, "grey")
        ax.plot(xs, ys, marker="o", ms=3, color=c, alpha=0.75,
                label=method if method not in seen else None)
        seen.add(method)
    ax.set_xlabel("path length (m)")
    ax.set_ylabel("GT coverage")
    ax.set_title("Coverage per metre travelled: continuous vs sample-and-select")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "coverage_vs_length.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def summarise(rows):
    """Report mean ± std with n, and the per-seed spread.

    Means alone are dangerous here: the rollout is bimodal (§3.4 / trajectory.py),
    so a mean over 3 seeds can hide a 0.33-vs-0.75 split, and a single run can be
    a lucky draw. Print the spread next to the mean so it cannot be quoted away.
    Note ``cov/m`` is diagnostic only -- a planner that stalls early scores well
    on it precisely because it gave up; judge on coverage at comparable length.
    """
    print("\n=== summary (mean ± std over seeds; cov/m is diagnostic only) ===")
    by = {}
    for r in rows:
        key = (r["method"], r["backbone"])
        by.setdefault(key, []).append(
            (float(r["gt_coverage"]), float(r["path_length"]),
             float(r["coverage_per_m"])))
    print(f"{'method':<26} {'backbone':<9} {'coverage':>16} {'length':>15} "
          f"{'cov/m':>8}  {'n':>2}  per-seed coverage")
    for (m, b), v in sorted(by.items()):
        a = np.array(v)
        spread = " ".join(f"{x:.3f}" for x in a[:, 0])
        print(f"{m:<26} {b:<9} {a[:,0].mean():>7.3f} ± {a[:,0].std():<6.3f} "
              f"{a[:,1].mean():>7.2f} ± {a[:,1].std():<5.2f} {a[:,2].mean():>8.3f}  "
              f"{len(a):>2}  [{spread}]")

    print("\nNote: identical config+seed is reproducible within a process but NOT\n"
          "across processes (GPU float non-determinism amplified by the audit\n"
          "threshold). Observed: 0.563 in 7/8 processes, 0.933 in the 8th.\n"
          "Treat any single rollout as one sample, never as the result.")


if __name__ == "__main__":
    main()
