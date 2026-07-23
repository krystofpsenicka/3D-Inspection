"""Visualize the no-prior exploration rollout (companion to eval_noprior.py).

Replays the receding-horizon arm for one seed with state recording and renders
the mission as belief-growth snapshots: the hidden true surface in light grey
(the robot cannot see this), the accumulated believed-but-uncovered demand in
orange, audited covered surface in green, and the executed path in blue. The
point of the figure: the belief is *empty* at the start — everything the
planner ever optimizes over arrived through the sensor.

Usage::

    ipy hpro/viz_noprior.py --no_show   (shares eval_noprior's arguments)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from eval_outdated import run_rh_arm                  # noqa: E402
from eval_trajectory import load_mesh                 # noqa: E402
from eval_noprior import build_world, parse_args      # noqa: E402
from trajectory import TrajectoryConfig               # noqa: E402


def snapshot_panel(ax, world, rec, path_xyz, title):
    pts = world.pts
    unknown = ~rec["believed"]
    cov_idx = np.fromiter(rec["covered"], dtype=int) if rec["covered"] \
        else np.zeros(0, dtype=int)
    covered = np.zeros(len(pts), dtype=bool)
    covered[cov_idx] = True
    demand = rec["believed"] & ~covered

    ax.scatter(*pts[unknown].T, s=1, c="0.85", alpha=0.25, linewidths=0)
    ax.scatter(*pts[demand].T, s=2, c="darkorange", alpha=0.8, linewidths=0)
    ax.scatter(*pts[covered].T, s=2, c="seagreen", alpha=0.8, linewidths=0)
    if len(path_xyz) > 1:
        ax.plot(*np.asarray(path_xyz).T, c="royalblue", lw=1.5)
    ax.scatter(*rec["pos"], c="royalblue", s=40, marker="^")
    ax.set_title(title, fontsize=10, pad=0)
    ax.set_axis_off()
    ax.view_init(elev=55, azim=-60)
    # equal aspect, zoomed to the cloud (the path can leave the box)
    lim = np.array([pts.min(axis=0), pts.max(axis=0)])
    c, r = lim.mean(axis=0), 0.42 * (lim[1] - lim[0]).max()
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    mesh = load_mesh(args.mesh)
    seed = args.viz_seed

    cfg = TrajectoryConfig(
        max_cycles=args.max_poses, near_dist=args.near_dist,
        far_dist=args.far_dist, seed=seed,
        target_coverage=args.target_coverage)
    world, start = build_world(mesh, args, seed)
    record = []
    r = run_rh_arm(world, args.backbone, world.cam, cfg, args, start, device,
                   record=record)
    print(f"replayed seed {seed}: cov={world.true_coverage():.3f} "
          f"len={r['path_length']:.2f}m poses={r['n_poses']} "
          f"discovered={world.believed.mean():.3f} "
          f"covered-of-discovered="
          f"{world.true_coverage() / max(world.believed.mean(), 1e-9):.3f}")

    n = len(record)
    picks = sorted(set([0, max(1, n // 3), max(2, 2 * n // 3), n - 1]))
    fig = plt.figure(figsize=(4.5 * len(picks), 3.8))
    path_xyz = [start]
    k = 0
    for i, rec in enumerate(record):
        path_xyz.append(rec["pos"])
        if i in picks:
            k += 1
            ax = fig.add_subplot(1, len(picks), k, projection="3d")
            snapshot_panel(ax, world, rec, path_xyz,
                           f"pose {i + 1}/{n} — coverage {rec['coverage']:.2f}")
    fig.suptitle("No-prior exploration: belief growth (grey = hidden truth, "
                 "orange = discovered demand, green = covered, blue = path)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.out, "noprior_rollout.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
