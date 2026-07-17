"""No-prior exploration — §6.4's second online setting (§6.3 pillar 3).

The robot starts with **nothing**: no mesh, no prior cloud. A depth sensor
(the same OutdatedWorld sensor as §3.6) seeds and grows its belief — seen
points are covered, and unknown surface within the discovery halo of what was
seen becomes known-but-uncovered demand. Planning happens *on the accumulated
cloud only*. This is the regime where a point-cloud visibility surrogate is
not merely faster than ray-casting the mesh but the **only possible
evaluator** — there is no mesh to ray-cast (the CMA-ES/greedy-on-raycast
objection of §6.3 evaporates structurally).

Arms (same sensor, pose cap, and stop rule):

* ``rh-auto`` — the belief-aware receding-horizon planner of ``eval_outdated``
  with the §6.4-item-7 backbone schedule: HPRO while the accumulated cloud is
  small (sparse partial clouds are out-of-distribution for NVPS, natural for
  HPRO), NVPS once it exceeds ~1500 points.
* ``nbv-oracle`` — greedy next-best-view: each step, offset candidate poses
  from current demand points and pick the one that **truly** sees the most new
  surface (audited against the hidden mesh — knowledge no real system has, so
  this is an upper reference on view selection, paying nothing for planning
  wall-clock honesty).
* ``nbv-frontier`` — go to a view of the nearest demand point. The cheap
  heuristic anchor.

Coverage is fraction of the full (hidden) surface sample. Note the planner
cannot know that denominator — the 0.95 stop is an experiment-level rule
applied identically to every arm.

Usage::

    ipy hpro/eval_noprior.py --seeds 3 --no_show
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
import torch                      # noqa: E402
import trimesh                    # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from eval_outdated import OutdatedWorld, run_rh_arm          # noqa: E402
from eval_trajectory import load_mesh                        # noqa: E402
from trajectory import TrajectoryConfig                      # noqa: E402

CSV_FIELDS = ["mesh", "seed", "method", "true_coverage", "path_length",
              "n_poses", "plan_time_s"]


def look_r6(pos, target):
    look = np.asarray(target, dtype=np.float64) - pos
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(look / (np.linalg.norm(look) + 1e-12), up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    return np.concatenate([look, up])


def run_nbv_arm(world, args, start, seed, mode):
    """Greedy NBV over candidates offset from current demand points.

    ``mode="oracle"``: score candidates by TRUE new coverage (audit against
    the hidden mesh — an upper reference). ``mode="frontier"``: fly to a view
    of the nearest demand point (no scoring at all).
    """
    rng = np.random.default_rng(seed)
    cur = start.copy()
    total_len = 0.0
    curve = []
    t_plan = 0.0

    for _ in range(args.max_poses):
        demand_idx = np.where(world.demand > 0.5)[0]
        if len(demand_idx) == 0:
            break
        t0 = time.perf_counter()
        n = world.normals[demand_idx]
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        cand_pos = world.pts[demand_idx] + n * args.standoff
        cand_r6 = np.concatenate([-n, np.tile([0.0, 0.0, 1.0],
                                              (len(n), 1))], axis=1)
        bad = np.abs(cand_r6[:, 2]) > 0.95 * np.linalg.norm(
            cand_r6[:, :3], axis=1)
        cand_r6[bad, 3:] = [0.0, 1.0, 0.0]

        if mode == "frontier":
            j = int(np.argmin(np.linalg.norm(cand_pos - cur[None], axis=1)))
        else:                                     # oracle
            pick = rng.choice(len(cand_pos),
                              size=min(args.nbv_candidates, len(cand_pos)),
                              replace=False)
            best_gain, j = -1, int(pick[0])
            for c in pick:
                seen = world.peek_true_seen(cand_pos[c], cand_r6[c])
                gain = len(seen - world.covered_true)
                if gain > best_gain:
                    best_gain, j = gain, int(c)
        t_plan += time.perf_counter() - t0

        total_len += float(np.linalg.norm(cand_pos[j] - cur))
        cur = cand_pos[j]
        world.observe(cur, cand_r6[j])
        curve.append((total_len, world.true_coverage()))
        if world.true_coverage() >= args.target_coverage:
            break

    return dict(curve=curve, path_length=total_len, n_poses=len(curve),
                plan_time_s=t_plan)


def build_world(mesh, args, seed):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    pts_true, fidx = trimesh.sample.sample_surface(mesh, count=args.num_points)
    pts_true = np.asarray(pts_true, dtype=np.float64)
    normals_true = np.asarray(mesh.face_normals[fidx], dtype=np.float64)

    cam = dict(fov_h=math.radians(args.fov_h), fov_v=math.radians(args.fov_v),
               near=args.near, far=args.far)
    empty = np.zeros((0, 3))
    world = OutdatedWorld(mesh, empty, empty, pts_true, normals_true,
                          np.zeros(len(pts_true), dtype=bool), cam,
                          discover_radius=args.discover_radius, no_prior=True)
    # Deploy the way a real mission starts: at standoff from a random point of
    # the (unknown) surface, looking at it — the far-sphere start of the other
    # harnesses would put the whole structure outside the 1.5 m sensor range
    # and seed an empty belief.
    a = rng.integers(len(pts_true))
    n = normals_true[a] / (np.linalg.norm(normals_true[a]) + 1e-12)
    start = pts_true[a] + n * args.standoff
    # The first observation seeds the belief — every arm gets it for free
    # from the same pose, before any planning happens.
    world.observe(start, look_r6(start, pts_true[a]))
    return world, start


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=os.path.join(
        os.path.dirname(_DIR), "models", "duke_of_lancaster_uk_clipped.glb"))
    p.add_argument("--num_points", type=int, default=3000)
    p.add_argument("--backbone", default="auto",
                   help="auto = HPRO below 1500 believed points, NVPS above.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max_poses", type=int, default=40)
    p.add_argument("--nbv_candidates", type=int, default=30)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--discover_radius", type=float, default=0.15)
    p.add_argument("--standoff", type=float, default=1.0)
    p.add_argument("--fov_h", type=float, default=30.0)
    p.add_argument("--fov_v", type=float, default=35.0)
    p.add_argument("--near", type=float, default=0.1)
    p.add_argument("--far", type=float, default=1.5)
    p.add_argument("--near_dist", type=float, default=0.5)
    p.add_argument("--far_dist", type=float, default=1.5)
    p.add_argument("--out", default=os.path.join(_DIR, "results", "noprior"))
    p.add_argument("--device", default="auto")
    p.add_argument("--no_show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    mesh = load_mesh(args.mesh)
    mesh_name = os.path.basename(args.mesh)

    rows, curves = [], {}
    csv_path = os.path.join(args.out, "eval_noprior.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for seed in range(args.seeds):
            cfg = TrajectoryConfig(
                max_cycles=args.max_poses, near_dist=args.near_dist,
                far_dist=args.far_dist, seed=seed,
                target_coverage=args.target_coverage)

            arms = [
                ("nbv-frontier",
                 lambda w, s: run_nbv_arm(w, args, s, seed, "frontier")),
                ("nbv-oracle",
                 lambda w, s: run_nbv_arm(w, args, s, seed, "oracle")),
                (f"rh-{args.backbone}",
                 lambda w, s: run_rh_arm(w, args.backbone, w.cam, cfg, args,
                                         s, device)),
            ]
            for method, fn in arms:
                world, start = build_world(mesh, args, seed)
                r = fn(world, start)
                row = dict(
                    mesh=mesh_name, seed=seed, method=method,
                    true_coverage=f"{world.true_coverage():.4f}",
                    path_length=f"{r['path_length']:.3f}",
                    n_poses=r["n_poses"],
                    plan_time_s=f"{r['plan_time_s']:.2f}",
                )
                writer.writerow(row); fh.flush(); rows.append(row)
                curves[(method, seed)] = r["curve"]
                print(f"  [seed {seed}] {method:<13s} "
                      f"cov={world.true_coverage():.3f} "
                      f"len={r['path_length']:.2f}m poses={r['n_poses']} "
                      f"plan={r['plan_time_s']:.1f}s", flush=True)

    print(f"\nCSV: {csv_path}")
    plot_curves(curves, args.out, not args.no_show)
    summarise(rows)


def plot_curves(curves, out_dir, show):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"nbv-frontier": "grey", "nbv-oracle": "tomato",
              "rh-auto": "seagreen", "rh-nvps": "seagreen",
              "rh-hpro": "royalblue"}
    seen = set()
    for (method, seed), pts in sorted(curves.items()):
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", ms=2.5,
                color=colors.get(method, "purple"), alpha=0.7,
                label=method if method not in seen else None)
        seen.add(method)
    ax.set_xlabel("path length (m)")
    ax.set_ylabel("coverage of the (hidden) surface")
    ax.set_title("No-prior exploration: coverage vs metres, belief from sensor only")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "noprior_coverage.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def summarise(rows):
    print("\n=== summary (mean ± std over seeds) ===")
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(
            (float(r["true_coverage"]), float(r["path_length"]),
             float(r["plan_time_s"])))
    print(f"{'method':<14} {'coverage':>16} {'length':>14} {'plan s':>8}")
    for m, v in sorted(by.items()):
        a = np.array(v)
        print(f"{m:<14} {a[:,0].mean():>7.3f} ± {a[:,0].std():<6.3f} "
              f"{a[:,1].mean():>6.2f} ± {a[:,1].std():<5.2f} "
              f"{a[:,2].mean():>8.1f}")
    print("\nnbv-oracle scores candidates against the hidden mesh (an upper\n"
          "reference no real system has); rh plans on the accumulated cloud\n"
          "only — the setting where the surrogate is the only evaluator.")


if __name__ == "__main__":
    main()
