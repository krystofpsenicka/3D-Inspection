"""Stage 2 demo — joint multi-viewpoint optimisation on lamp_0001.off.

Runs optimize_viewpoints for V=5 cameras, compares the initial Fibonacci-
sphere poses against the optimised solution using exact ray-cast ground truth,
and produces two figures:
  1. Two-panel training curves (loss + soft coverage).
  2. 3-D scatter: covered points (green), viewpoints (red), frustum wireframes.

Meant to be run from the hpro/ directory or from the project root:
    cd hpro && python demo_multi.py
    python hpro/demo_multi.py
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

# Ensure hpro/ is importable regardless of working directory.
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from frustum_gt import (          # noqa: E402
    build_camera_frame,
    compute_ground_truth_batched,
    draw_frustum,
    set_axes_equal_3d,
)
from HPRO_limited import HPRO_limited             # noqa: E402
from multi_viewpoint import (                      # noqa: E402
    fibonacci_sphere_viewpoints,
    optimize_viewpoints,
    MultiViewOptResult,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED       = 17
N_POINTS   = 3000
V          = 5           # number of viewpoints to optimise jointly

N_STEPS    = 300
LR         = 1e-2

GAMMA      = -math.exp(-7.0)   # Stage-1 optimal value
K          = 10

FOV_H      = math.radians(30.0)
FOV_V      = math.radians(35.0)
NEAR       = 0.5
FAR        = 2.5

LAMBDA_STANDOFF  = 1.0
LAMBDA_DIVERSITY = 0.1
MIN_INTER_DIST   = 0.3
FRUSTUM_SHARPNESS = 50.0
VISIBILITY_THRESH = 0.5

MESH_PATH  = os.path.join(_DIR, "lamp_0001.off")
RESULTS_DIR = os.path.join(_DIR, "results")


# ---------------------------------------------------------------------------
# Ground-truth evaluation helper
# ---------------------------------------------------------------------------

def eval_gt_coverage(mesh, pts_np, viewpoints_np, rot_6d_np, fov_h, fov_v, near, far):
    """
    Evaluate hard GT (frustum + ray-cast) coverage for a set of viewpoints.

    Returns:
        covered_union  — set[int] of covered point indices (union over all VPs)
        per_vp_counts  — list[int] of GT visible counts per viewpoint
    """
    intersector = mesh.ray
    covered_union = set()
    per_vp_counts = []
    for v in range(len(viewpoints_np)):
        look, up, right = build_camera_frame(rot_6d_np[v, :3], rot_6d_np[v, 3:])
        gt_idx, _ = compute_ground_truth_batched(
            mesh, viewpoints_np[v], pts_np,
            look, up, right,
            fov_h, fov_v, near, far,
            intersector=intersector,
        )
        covered_union.update(int(i) for i in gt_idx)
        per_vp_counts.append(len(gt_idx))
    return covered_union, per_vp_counts


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_curves(result: MultiViewOptResult, save_path=None):
    """Two-panel figure: loss curve and soft coverage curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Stage 2 optimisation (V={result.n_viewpoints}, N={result.n_points})",
                 fontsize=13)

    ax1.plot(result.steps_log, result.loss_history, color="crimson", linewidth=1.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss  (−coverage + penalties)")
    ax1.set_title("Total loss")
    ax1.grid(True, alpha=0.35)

    ax2.plot(result.steps_log, result.coverage_history, color="royalblue", linewidth=1.8,
             label="soft C")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Soft coverage C")
    ax2.set_title("Soft set-cover coverage")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.35)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_coverage_3d(pts_np, result: MultiViewOptResult,
                     covered_init: set, covered_opt: set,
                     fov_h, fov_v, near, far,
                     init_vp_np, init_r6d_np,
                     save_path=None):
    """
    Two-panel 3-D figure: initial poses (left) vs optimised poses (right).
    Grey = all points; blue/green = covered; red triangles = viewpoints;
    orange frustum wireframes.
    """
    p = pts_np
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f"Multi-viewpoint HPRO (V={result.n_viewpoints}): "
                 f"initial vs optimised", fontsize=13)

    def _make_ax(pos):
        return fig.add_subplot(pos, projection="3d")

    def _scatter(ax, idx, color, size=1, alpha=1.0):
        idx = list(idx)
        if idx:
            ax.scatter(p[idx, 0], p[idx, 1], p[idx, 2],
                       c=color, s=size, alpha=alpha, depthshade=False)

    def _finish(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.tick_params(labelsize=7)
        set_axes_equal_3d(ax)

    colors = ["tab:orange", "tab:green", "tab:purple", "tab:cyan", "tab:pink",
              "tab:brown", "tab:olive", "tab:red"]

    # --- Initial poses -------------------------------------------------------
    ax_init = _make_ax(121)
    _scatter(ax_init, range(len(p)), "lightgrey", size=0.3, alpha=0.3)
    _scatter(ax_init, covered_init, "steelblue", size=2, alpha=0.9)
    for v in range(len(init_vp_np)):
        look, up, right = build_camera_frame(init_r6d_np[v, :3], init_r6d_np[v, 3:])
        draw_frustum(ax_init, init_vp_np[v], look, up, right,
                     fov_h, fov_v, near, far,
                     color=colors[v % len(colors)], linewidth=1.0)
    ax_init.scatter(init_vp_np[:, 0], init_vp_np[:, 1], init_vp_np[:, 2],
                    c="red", s=60, marker="^", zorder=5, depthshade=False)
    _finish(ax_init, f"Initial (Fibonacci)  —  GT covered: {len(covered_init)}/{len(p)}")

    # --- Optimised poses -----------------------------------------------------
    ax_opt = _make_ax(122)
    _scatter(ax_opt, range(len(p)), "lightgrey", size=0.3, alpha=0.3)
    _scatter(ax_opt, covered_opt, "limegreen", size=2, alpha=0.9)
    vp_opt = result.viewpoints_np
    r6d_opt = result.rot_6d_np
    for v in range(len(vp_opt)):
        look, up, right = build_camera_frame(r6d_opt[v, :3], r6d_opt[v, 3:])
        draw_frustum(ax_opt, vp_opt[v], look, up, right,
                     fov_h, fov_v, near, far,
                     color=colors[v % len(colors)], linewidth=1.0)
    ax_opt.scatter(vp_opt[:, 0], vp_opt[:, 1], vp_opt[:, 2],
                   c="red", s=60, marker="^", zorder=5, depthshade=False)
    _finish(ax_opt, f"Optimised  —  GT covered: {len(covered_opt)}/{len(p)}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load and normalise mesh -------------------------------------------
    print(f"Loading {MESH_PATH} ...")
    np.random.seed(SEED)
    mesh = trimesh.load_mesh(MESH_PATH)
    mesh.vertices -= np.mean(mesh.vertices, axis=0)
    mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

    # ---- Sample surface points ---------------------------------------------
    print(f"Sampling {N_POINTS} surface points ...")
    pts_surface, _ = trimesh.sample.sample_surface(
        mesh, count=N_POINTS, face_weight=None, sample_color=False
    )   # (N, 3) float64

    # ---- Build HPRO_limited model ------------------------------------------
    model = HPRO_limited(
        fits_in_memory=True,
        visibility_score_thresh=VISIBILITY_THRESH,
        fov_h=FOV_H,
        fov_v=FOV_V,
        near=NEAR,
        far=FAR,
        frustum_sharpness=FRUSTUM_SHARPNESS,
        device=device,
    )

    # ---- Fibonacci initialisation preview ----------------------------------
    print(f"\nInitialising V={V} viewpoints on Fibonacci sphere ...")
    centroid = pts_surface.mean(axis=0)
    init_radius = (NEAR + FAR) / 2.0
    init_vp_np, init_r6d_np = fibonacci_sphere_viewpoints(
        V, init_radius, centroid
    )

    print("Evaluating initial poses (GT ray-cast) ...")
    covered_init, counts_init = eval_gt_coverage(
        mesh, pts_surface, init_vp_np, init_r6d_np,
        FOV_H, FOV_V, NEAR, FAR,
    )
    print(f"  Initial GT coverage : {len(covered_init)}/{N_POINTS} "
          f"({100*len(covered_init)/N_POINTS:.1f}%)")
    print(f"  Per-viewpoint counts: {counts_init}")

    # ---- Run joint optimisation --------------------------------------------
    print(f"\nRunning optimize_viewpoints (V={V}, N={N_POINTS}, steps={N_STEPS}) ...")
    result = optimize_viewpoints(
        pts_surface, V, model,
        gamma=GAMMA,
        k=K,
        n_steps=N_STEPS,
        lr=LR,
        lambda_standoff=LAMBDA_STANDOFF,
        lambda_diversity=LAMBDA_DIVERSITY,
        near_dist=NEAR,
        far_dist=FAR,
        min_inter_dist=MIN_INTER_DIST,
        init_viewpoints_np=init_vp_np,
        init_rot_6d_np=init_r6d_np,
        log_every=25,
        dtype=torch.float32,
        seed=SEED,
    )

    # ---- Evaluate optimised poses (GT) -------------------------------------
    print("\nEvaluating optimised poses (GT ray-cast) ...")
    covered_opt, counts_opt = eval_gt_coverage(
        mesh, pts_surface, result.viewpoints_np, result.rot_6d_np,
        FOV_H, FOV_V, NEAR, FAR,
    )
    print(f"  Optimised GT coverage: {len(covered_opt)}/{N_POINTS} "
          f"({100*len(covered_opt)/N_POINTS:.1f}%)")
    print(f"  Per-viewpoint counts : {counts_opt}")

    # ---- Summary -----------------------------------------------------------
    delta = len(covered_opt) - len(covered_init)
    print("\n" + "=" * 55)
    print(f"  V={V}  N={N_POINTS}  steps={N_STEPS}  time={result.wall_time_s:.1f}s")
    print(f"  Init GT coverage  : {100*len(covered_init)/N_POINTS:.1f}%  "
          f"({len(covered_init)}/{N_POINTS} pts)")
    print(f"  Final GT coverage : {100*len(covered_opt)/N_POINTS:.1f}%  "
          f"({len(covered_opt)}/{N_POINTS} pts)")
    print(f"  Improvement       : {delta:+d} pts  "
          f"({100*delta/max(len(covered_init),1):+.1f}%)")
    print(f"  Soft coverage (C) : {result.coverage_history[-1]:.4f}")
    print("=" * 55)

    # ---- Plots -------------------------------------------------------------
    print("\nPlotting training curves ...")
    plot_training_curves(
        result,
        save_path=os.path.join(RESULTS_DIR, "demo_multi_curves.png"),
    )

    print("Plotting 3-D coverage comparison ...")
    plot_coverage_3d(
        pts_surface, result,
        covered_init, covered_opt,
        FOV_H, FOV_V, NEAR, FAR,
        init_vp_np, init_r6d_np,
        save_path=os.path.join(RESULTS_DIR, "demo_multi_3d.png"),
    )
