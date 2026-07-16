"""Visual inspection of receding-horizon rollouts.

Scalars hide the failures that matter. "GT coverage 0.53" does not tell you
*why*: whether the camera flew inside the hull, jittered in place, stared at
already-covered geometry, or covered the object honestly and simply ran out of
cycles. §3.4's stall was found by staring at numbers and guessing; it should have
been obvious in one picture. This module draws the rollout so correctness can be
checked by eye:

* **3D rollout** — the cloud coloured by covered/uncovered, the executed path,
  camera frustums, and the start marker. Answers: is the camera at a sane
  standoff, is it looking at the object, is the path sensible?
* **Surrogate vs truth** — per pose, which points the surrogate *claimed* were
  visible and which were *actually* seen. This is the C2 exploitation story
  rendered rather than summarised: HPRO's over-prediction is a visible red halo.
* **Diagnostics** — coverage vs metres travelled (versus baselines), metres per
  cycle (a flat line = a stalled robot), and the audit gap per cycle.
* **Animation** — the rollout replayed pose by pose as a GIF.

Everything is matplotlib, so it runs in the `isaaclab` env with no extra deps and
writes PNG/GIF for viewing anywhere.

Usage::

    ipy hpro/viz_trajectory.py --backbones nvps,hpro --max_cycles 40 --animate
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from frustum_gt import build_camera_frame   # noqa: E402

# Consistent semantics across every figure in this module. Uncovered points must
# stay clearly visible: they are the *result* being judged, and a washed-out grey
# makes a 56 %-covered object look finished.
C_COVERED = "#2e9e5b"     # green  -- audited as seen
C_UNCOVERED = "#6b7280"   # slate  -- still owed
C_PATH = "#2b6cb0"        # blue   -- executed path
C_FRUSTUM = "#ed8936"     # orange -- camera frustum
C_SEEN = "#2e9e5b"        # green  -- truly seen at this pose
C_FALSE = "#e53e3e"       # red    -- surrogate claimed visible, ray-cast says no


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def frustum_corners(pos, r6, fov_h, fov_v, near, far):
    """The 8 corners of a camera frustum, in world coordinates.

    Args:
        pos: (3,) camera position.
        r6: (6,) 6D rotation [look | up] (unnormalised is fine).
        fov_h, fov_v: field of view in radians.
        near, far: clip-plane depths.

    Returns:
        (8, 3) corners: near plane first (bl, br, tr, tl), then far plane.
    """
    look, up, right = build_camera_frame(r6[:3], r6[3:])
    out = []
    for depth in (near, far):
        h = depth * math.tan(fov_v / 2.0)
        w = depth * math.tan(fov_h / 2.0)
        for dw, dh in ((-w, -h), (w, -h), (w, h), (-w, h)):
            out.append(pos + look * depth + right * dw + up * dh)
    return np.array(out)


def draw_frustum(ax, pos, r6, cam, color=C_FRUSTUM, alpha=0.5, lw=0.8):
    """Draw a camera frustum as a wireframe pyramid."""
    c = frustum_corners(pos, r6, cam["fov_h"], cam["fov_v"], cam["near"], cam["far"])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),        # near plane
             (4, 5), (5, 6), (6, 7), (7, 4),        # far plane
             (0, 4), (1, 5), (2, 6), (3, 7)]        # connecting
    for i, j in edges:
        ax.plot(*zip(c[i], c[j]), color=color, alpha=alpha, lw=lw)


def _equal_aspect(ax, pts):
    """Equal aspect for 3D axes — without it a path looks distorted and you
    cannot judge whether the standoff is sane."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid, span = (lo + hi) / 2.0, (hi - lo).max() / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


# ---------------------------------------------------------------------------
# 3D rollout
# ---------------------------------------------------------------------------

def plot_rollout(ax, pts_np, result, cam, covered=None, upto=None,
                 every_frustum=3, title=None):
    """Draw one rollout: cloud by coverage, executed path, frustums.

    Args:
        ax: a 3D axes.
        pts_np: (N, 3) target cloud.
        result: TrajectoryResult.
        cam: dict(fov_h, fov_v, near, far).
        covered: set of covered indices; if None, taken from ``result`` up to
            ``upto`` so the same function serves both stills and animation.
        upto: draw only the first ``upto`` executed poses (None = all).
        every_frustum: draw a frustum every k-th pose (all of them is a mess).
        title: axes title.
    """
    T = len(result.positions) if upto is None else min(upto, len(result.positions))
    if covered is None:
        covered = set()
        for s in result.seen_per_pose[:T]:
            covered |= s

    mask = np.zeros(len(pts_np), dtype=bool)
    if covered:
        mask[np.fromiter(covered, dtype=int, count=len(covered))] = True

    ax.scatter(*pts_np[~mask].T, s=1.5, c=C_UNCOVERED, alpha=0.35, linewidths=0)
    if mask.any():
        ax.scatter(*pts_np[mask].T, s=2.0, c=C_COVERED, alpha=0.9, linewidths=0)

    if T > 0:
        p = result.positions[:T]
        ax.plot(*p.T, color=C_PATH, lw=1.8, alpha=0.9, zorder=5)
        ax.scatter(*p.T, s=14, c=C_PATH, depthshade=False, zorder=6)
        # Start and current pose: the two the eye looks for.
        ax.scatter(*p[0], s=90, marker="^", c="black", depthshade=False, zorder=7)
        ax.scatter(*p[-1], s=90, marker="*", c=C_FRUSTUM, depthshade=False, zorder=7)
        for i in range(0, T, max(1, every_frustum)):
            draw_frustum(ax, p[i], result.rot_6d[i], cam, alpha=0.35)
        draw_frustum(ax, p[-1], result.rot_6d[-1], cam, alpha=0.95, lw=1.4)

    _equal_aspect(ax, np.vstack([pts_np, result.positions[:max(T, 1)]]))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)


def plot_surrogate_vs_truth(ax, pts_np, result, cam, pose_idx, title=None):
    """Render the audit at one pose: what was claimed vs what was seen.

    Green = the surrogate said visible and the ray-cast agrees. Red = the
    surrogate said visible and the ray-cast says no. A red halo is exactly the
    over-prediction that makes HPRO park the camera and stall (§3.4).
    """
    seen = result.seen_per_pose[pose_idx]
    pred = result.pred_per_pose[pose_idx]
    tp = np.fromiter(pred & seen, dtype=int, count=len(pred & seen))
    fp = np.fromiter(pred - seen, dtype=int, count=len(pred - seen))

    ax.scatter(*pts_np.T, s=1.0, c=C_UNCOVERED, alpha=0.15, linewidths=0)
    if len(tp):
        ax.scatter(*pts_np[tp].T, s=3.0, c=C_SEEN, alpha=0.9, linewidths=0)
    if len(fp):
        ax.scatter(*pts_np[fp].T, s=3.0, c=C_FALSE, alpha=0.9, linewidths=0)

    p = result.positions[pose_idx]
    ax.scatter(*p, s=90, marker="*", c=C_FRUSTUM, depthshade=False, zorder=7)
    draw_frustum(ax, p, result.rot_6d[pose_idx], cam, alpha=0.9, lw=1.2)
    _equal_aspect(ax, np.vstack([pts_np, p[None]]))
    ax.set_axis_off()
    ax.set_title(title or f"pose {pose_idx}: claimed {len(pred)}, "
                          f"seen {len(seen)}, false {len(fp)}", fontsize=9)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def plot_diagnostics(fig, results, baseline=None):
    """Coverage vs metres, metres per cycle, and the audit gap.

    Args:
        fig: figure to draw into (3 axes are created).
        results: dict name -> TrajectoryResult.
        baseline: optional list of (path_length, coverage, label) points.
    """
    ax1, ax2, ax3 = (fig.add_subplot(1, 3, i) for i in (1, 2, 3))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (name, r) in enumerate(results.items()):
        c = colors[i % 10]
        ax1.plot(r.length_curve, r.coverage_curve, marker="o", ms=2.5,
                 color=c, label=name)
        # Metres per cycle: a flat line means the robot has stopped moving,
        # which is what a stalled rollout looks like.
        steps = np.diff(np.concatenate([[0.0], r.length_curve]))
        ax2.plot(steps, color=c, label=name, lw=1.2)
        ax3.plot(r.audit_gap, color=c, label=name, lw=1.2)

    if baseline:
        bl = sorted(baseline)
        ax1.plot([b[0] for b in bl], [b[1] for b in bl], marker="s", ms=4,
                 color="tomato", ls="--", label="greedy+route (oracle)")

    ax1.set_xlabel("path length (m)"); ax1.set_ylabel("GT coverage")
    ax1.set_title("Coverage per metre travelled", fontsize=10)
    ax1.set_ylim(0, 1); ax1.grid(alpha=0.3); ax1.legend(fontsize=7)

    ax2.set_xlabel("cycle"); ax2.set_ylabel("metres moved")
    ax2.set_title("Motion per cycle (flat = stalled)", fontsize=10)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=7)

    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xlabel("cycle"); ax3.set_ylabel("predicted − actually seen (pts)")
    ax3.set_title("Audit gap (>0 = surrogate over-predicts)", fontsize=10)
    ax3.grid(alpha=0.3); ax3.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_rollout(pts_np, result, cam, out_path, fps=3, elev=22, azim0=-60):
    """Replay the rollout pose by pose as a GIF, slowly orbiting."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    T = len(result.positions)

    def frame(t):
        ax.clear()
        plot_rollout(ax, pts_np, result, cam, upto=t + 1, every_frustum=4,
                     title=f"pose {t+1}/{T}   coverage "
                           f"{result.coverage_curve[t]:.3f}   "
                           f"{result.length_curve[t]:.2f} m")
        ax.view_init(elev=elev, azim=azim0 + 1.5 * t)

    anim = FuncAnimation(fig, frame, frames=T, interval=1000 // fps)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def rollout_report(pts_np, results, cam, out_dir, baseline=None, animate=False):
    """Write the full visual report for one or more rollouts."""
    os.makedirs(out_dir, exist_ok=True)

    # --- 3D rollouts, side by side -----------------------------------------
    n = len(results)
    fig = plt.figure(figsize=(6.5 * n, 5.8))
    for i, (name, r) in enumerate(results.items()):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.view_init(elev=22, azim=-60)
        plot_rollout(ax, pts_np, r, cam,
                     title=f"{name}\ncoverage {r.gt_coverage:.3f}   "
                           f"{r.path_length:.2f} m   {r.n_cycles} poses")
    handles = [
        Line2D([], [], marker="o", ls="", color=C_COVERED, label="covered (audited)"),
        Line2D([], [], marker="o", ls="", color=C_UNCOVERED, label="not covered"),
        Line2D([], [], color=C_PATH, label="executed path"),
        Line2D([], [], color=C_FRUSTUM, label="camera frustum"),
        Line2D([], [], marker="^", ls="", color="black", label="start"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(out_dir, "rollout_3d.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"Saved: {p}")

    # --- Surrogate vs truth -------------------------------------------------
    fig = plt.figure(figsize=(5.5 * n, 5.2))
    for i, (name, r) in enumerate(results.items()):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.view_init(elev=22, azim=-60)
        # The worst-over-prediction pose is the informative one.
        k = int(np.argmax(r.audit_gap))
        plot_surrogate_vs_truth(
            ax, pts_np, r, cam, k,
            title=f"{name} — worst pose ({k})\n"
                  f"claimed {len(r.pred_per_pose[k])}, seen {len(r.seen_per_pose[k])}, "
                  f"false {len(r.pred_per_pose[k] - r.seen_per_pose[k])}")
    handles = [
        Line2D([], [], marker="o", ls="", color=C_SEEN,
               label="claimed visible AND truly seen"),
        Line2D([], [], marker="o", ls="", color=C_FALSE,
               label="claimed visible BUT occluded (surrogate error)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = os.path.join(out_dir, "surrogate_vs_truth.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"Saved: {p}")

    # --- Diagnostics --------------------------------------------------------
    fig = plt.figure(figsize=(15, 4))
    plot_diagnostics(fig, results, baseline=baseline)
    fig.tight_layout()
    p = os.path.join(out_dir, "diagnostics.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"Saved: {p}")

    if animate:
        for name, r in results.items():
            animate_rollout(pts_np, r, cam,
                            os.path.join(out_dir, f"rollout_{name}.gif"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    import torch
    import trimesh

    from backbones import make_backbone
    from eval_trajectory import (load_mesh, make_audit_fn,
                                 normal_offset_candidates, greedy_select,
                                 route_nn_2opt)
    from trajectory import TrajectoryConfig, receding_horizon_plan
    from visibility_layer import GatedVisibilityLayer

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default=os.path.join(
        os.path.dirname(_DIR), "models", "duke_of_lancaster_uk_clipped.glb"))
    ap.add_argument("--num_points", type=int, default=3000)
    ap.add_argument("--backbones", default="nvps,hpro")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_cycles", type=int, default=40)
    ap.add_argument("--budgets", default="5,10,20,30")
    ap.add_argument("--n_candidates", type=int, default=300)
    ap.add_argument("--fov_h", type=float, default=30.0)
    ap.add_argument("--fov_v", type=float, default=35.0)
    ap.add_argument("--near", type=float, default=0.1)
    ap.add_argument("--far", type=float, default=1.5)
    ap.add_argument("--animate", action="store_true", help="Also write GIFs.")
    ap.add_argument("--out", default=os.path.join(_DIR, "results", "viz"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cam = dict(fov_h=math.radians(args.fov_h), fov_v=math.radians(args.fov_v),
               near=args.near, far=args.far)
    mesh = load_mesh(args.mesh)

    np.random.seed(args.seed)
    pts_np, fidx = trimesh.sample.sample_surface(mesh, count=args.num_points)
    pts_np = np.asarray(pts_np, dtype=np.float64)
    normals_np = np.asarray(mesh.face_normals[fidx], dtype=np.float64)
    audit = make_audit_fn(mesh, pts_np, cam)

    rng = np.random.default_rng(args.seed)
    d0 = rng.normal(size=3); d0 /= np.linalg.norm(d0)
    start = pts_np.mean(axis=0) + d0 * (1.0 + args.far)

    results = {}
    for name in [b.strip() for b in args.backbones.split(",")]:
        b = make_backbone(name, device=device, gamma=-math.exp(-7.0), k=10)
        if b.requires_normals:
            b.prepare(pts_np, normals_np)
        layer = GatedVisibilityLayer(b, fov_h=cam["fov_h"], fov_v=cam["fov_v"],
                                     near=cam["near"], far=cam["far"],
                                     frustum_sharpness=50.0, device=device)
        cfg = TrajectoryConfig(max_cycles=args.max_cycles, seed=args.seed)
        r = receding_horizon_plan(layer, pts_np, audit, cfg, start_pos=start,
                                  device=device)
        results[name] = r
        print(f"[{name}] coverage {r.gt_coverage:.3f}  {r.path_length:.2f} m  "
              f"audit gap {np.mean(r.audit_gap):+.0f}")

    # Baseline reference points for the coverage-vs-metres panel.
    standoff = 0.5 * (0.5 + 1.5)
    cpos, cr6 = normal_offset_candidates(pts_np, normals_np, args.n_candidates,
                                         standoff, args.seed)
    cgt = [audit(cpos[j], cr6[j]) for j in range(len(cpos))]
    baseline = []
    for budget in [int(x) for x in args.budgets.split(",")]:
        sel = greedy_select(cgt, budget)
        _, length = route_nn_2opt(start, cpos[sel])
        cov = len(set().union(*[cgt[j] for j in sel])) / len(pts_np) if sel else 0.0
        baseline.append((length, cov, f"V={budget}"))
        print(f"[greedy V={budget}] coverage {cov:.3f}  {length:.2f} m")

    rollout_report(pts_np, results, cam, args.out, baseline=baseline,
                   animate=args.animate)


if __name__ == "__main__":
    main()
