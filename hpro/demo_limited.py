# Compares HPRO_limited (differentiable frustum-constrained visibility) against
# a ground-truth computed by combining:
#   1. Exact geometric frustum test (is the point inside the pyramid frustum?)
#   2. Ray-casting occlusion test (is the point visible from the viewpoint?)
#
# The script mirrors the structure of demo.py and prints error statistics, then
# visualises the results including a wireframe of the frustum in every panel.

import math
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt

from GroundTruthGenerator import GroundTruthGenerator
from HPRO_limited import HPRO_limited


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def build_camera_frame(look_approx_np, up_approx_np):
    """
    Return an orthonormal (look, up, right) frame via Gram-Schmidt,
    mirroring the logic inside HPRO_limited._6d_to_rotation_matrix.
    All inputs/outputs are 1-D numpy arrays of shape (3,).
    """
    look  = look_approx_np / np.linalg.norm(look_approx_np)
    up    = up_approx_np - np.dot(up_approx_np, look) * look
    up    = up / np.linalg.norm(up)
    right = np.cross(look, up)
    return look, up, right


def points_inside_frustum(pts_np, viewpoint_np, look, up, right,
                           fov_h, fov_v, near, far):
    """
    Exact (hard) geometric frustum test — no sigmoid approximation.

    Args:
        pts_np      : (N, 3) point positions.
        viewpoint_np: (3,)   camera position.
        look/up/right: orthonormal camera-frame axes, each (3,).
        fov_h, fov_v: horizontal / vertical FOV in radians.
        near, far   : depth clip distances.

    Returns:
        Bool array (N,) — True if the point is inside the frustum.
    """
    v = pts_np - viewpoint_np[np.newaxis, :]

    depth   = v @ look
    h_coord = v @ right
    v_coord = v @ up

    tan_h_lim = math.tan(fov_h / 2.0)
    tan_v_lim = math.tan(fov_v / 2.0)

    eps      = 1e-8
    in_depth = (depth > near) & (depth < far)
    in_h     = np.abs(h_coord) < tan_h_lim * (depth + eps)
    in_v     = np.abs(v_coord) < tan_v_lim * (depth + eps)

    return in_depth & in_h & in_v


def compute_ground_truth(mesh, viewpoint_np, pts_np,
                          look, up, right,
                          fov_h, fov_v, near, far):
    """
    Ground-truth: points inside the frustum AND not occluded (ray-cast).
    Ray-cast runs only on the frustum subset to save time.

    Returns:
        gt_indices      — frustum-visible AND unoccluded indices.
        frustum_indices — all indices inside the frustum (pre-occlusion).
    """
    in_frustum      = points_inside_frustum(
        pts_np, viewpoint_np, look, up, right, fov_h, fov_v, near, far
    )
    frustum_indices = list(np.where(in_frustum)[0])

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    gt_indices  = [
        idx for idx in frustum_indices
        if GroundTruthGenerator.singleRayIntersection(
            idx, intersector, viewpoint_np, pts_np
        )
    ]
    return gt_indices, frustum_indices


# ---------------------------------------------------------------------------
# Frustum wireframe helpers
# ---------------------------------------------------------------------------

def frustum_corners(viewpoint_np, look, up, right, fov_h, fov_v, near, far):
    """
    Compute the 8 corners of the rectangular frustum.

    Near/far plane corner order:
        0: (+h, +v)  top-right
        1: (-h, +v)  top-left
        2: (-h, -v)  bottom-left
        3: (+h, -v)  bottom-right

    Returns:
        near_corners : (4, 3) numpy array
        far_corners  : (4, 3) numpy array
    """
    th = math.tan(fov_h / 2.0)
    tv = math.tan(fov_v / 2.0)

    offsets = np.array([
        [ th,  tv],
        [-th,  tv],
        [-th, -tv],
        [ th, -tv],
    ])  # (4, 2)  — (h_scale, v_scale)

    def plane_corners(dist):
        centre = viewpoint_np + dist * look
        return np.array([
            centre + dist * (h * right + v * up)
            for h, v in offsets
        ])  # (4, 3)

    return plane_corners(near), plane_corners(far)


def draw_frustum(ax, viewpoint_np, look, up, right,
                 fov_h, fov_v, near, far,
                 color='orange', linewidth=1.2, alpha=0.85):
    """
    Draw a frustum wireframe on a Matplotlib 3D axis using individual plot3D
    calls (one per edge).  Using plot3D — rather than Line3DCollection —
    ensures that the frustum corner world coordinates are registered in the
    axis data limits, which is required for set_axes_equal_3d to produce
    correct equal-scale rendering of tilted frustums.

    Draws:
      - 4 edges of the near rectangle
      - 4 edges of the far  rectangle
      - 4 lateral edges: viewpoint → near corner → far corner
    """
    nc, fc = frustum_corners(viewpoint_np, look, up, right, fov_h, fov_v, near, far)

    def seg(p0, p1):
        ax.plot3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                  color=color, linewidth=linewidth, alpha=alpha)

    # Near-plane rectangle
    for i in range(4):
        seg(nc[i], nc[(i + 1) % 4])

    # Far-plane rectangle
    for i in range(4):
        seg(fc[i], fc[(i + 1) % 4])

    # Lateral edges: viewpoint → near corner → far corner
    for i in range(4):
        seg(viewpoint_np, nc[i])
        seg(nc[i], fc[i])


def set_axes_equal_3d(ax):
    """Set equal data-range on all three axes so tilted frustums are not distorted."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(pts_np, viewpoint_np,
                 look, up, right,
                 fov_h, fov_v, near, far,
                 gt_indices, frustum_only_indices,
                 hpro_limited_indices,
                 only_in_gt, only_in_hpro):
    """
    Seven-panel figure (2 rows × 4 columns, last slot empty):
      1. All sampled points + frustum wireframe
      2. Ground-truth visible (frustum ∩ not occluded)
      3. Frustum interior (orange) vs unoccluded subset (green)
      4. HPRO_limited result
      5. GT (green) vs HPRO_limited (blue) overlaid
      6. False negatives — in GT, missed by HPRO_limited
      7. False positives — in HPRO_limited, not in GT
    """
    p  = pts_np
    vp = viewpoint_np

    fig = plt.figure(figsize=(22, 12))
    fig.suptitle("HPRO_limited vs Ground Truth (frustum + ray-cast)", fontsize=14)

    def scatter3(ax, idx, color, size=1, alpha=1.0, label=None):
        idx = list(idx)
        if not idx:
            return
        ax.scatter(p[idx, 0], p[idx, 1], p[idx, 2],
                   c=color, s=size, alpha=alpha, label=label, depthshade=False)

    def add_viewpoint(ax):
        ax.scatter(*vp, c='red', s=60, marker='^', zorder=5, label='viewpoint')

    def add_frustum(ax):
        draw_frustum(ax, vp, look, up, right, fov_h, fov_v, near, far)

    def finish(ax, title):
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.tick_params(labelsize=7)

    # 1 — all points + frustum
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    scatter3(ax1, range(len(p)), 'steelblue', size=0.4, alpha=0.3)
    add_viewpoint(ax1); add_frustum(ax1)
    ax1.legend(fontsize=7, markerscale=2)
    finish(ax1, f'All sampled points ({len(p)})')
    set_axes_equal_3d(ax1)

    # 2 — ground truth
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    scatter3(ax2, gt_indices, 'green', size=2, label='GT visible')
    add_viewpoint(ax2); add_frustum(ax2)
    ax2.legend(fontsize=7, markerscale=2)
    finish(ax2, f'Ground truth visible ({len(gt_indices)})')
    set_axes_equal_3d(ax2)

    # 3 — frustum interior vs unoccluded
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    scatter3(ax3, frustum_only_indices, 'orange', size=1,  alpha=0.45, label='in frustum')
    scatter3(ax3, gt_indices,           'green',  size=2,  alpha=0.9,  label='unoccluded')
    add_viewpoint(ax3); add_frustum(ax3)
    ax3.legend(fontsize=7, markerscale=2)
    finish(ax3, f'Frustum ({len(frustum_only_indices)}) / visible ({len(gt_indices)})')
    set_axes_equal_3d(ax3)

    # 4 — HPRO_limited
    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    scatter3(ax4, hpro_limited_indices, 'royalblue', size=2, label='HPRO_limited')
    add_viewpoint(ax4); add_frustum(ax4)
    ax4.legend(fontsize=7, markerscale=2)
    finish(ax4, f'HPRO_limited ({len(hpro_limited_indices)})')
    set_axes_equal_3d(ax4)

    # 5 — GT vs HPRO_limited overlay
    ax5 = fig.add_subplot(2, 4, 5, projection='3d')
    scatter3(ax5, gt_indices,           'green',     size=2, alpha=0.55, label='GT')
    scatter3(ax5, hpro_limited_indices, 'royalblue', size=2, alpha=0.55, label='HPRO_limited')
    add_viewpoint(ax5); add_frustum(ax5)
    ax5.legend(fontsize=7, markerscale=2)
    finish(ax5, 'GT (green) vs HPRO_limited (blue)')
    set_axes_equal_3d(ax5)

    # 6 — false negatives
    ax6 = fig.add_subplot(2, 4, 6, projection='3d')
    scatter3(ax6, range(len(p)), 'lightgrey', size=0.2, alpha=0.25)
    scatter3(ax6, only_in_gt,   'red',       size=4,   label=f'missed ({len(only_in_gt)})')
    add_viewpoint(ax6); add_frustum(ax6)
    ax6.legend(fontsize=7, markerscale=2)
    finish(ax6, f'False negatives — GT only ({len(only_in_gt)})')
    set_axes_equal_3d(ax6)

    # 7 — false positives
    ax7 = fig.add_subplot(2, 4, 7, projection='3d')
    scatter3(ax7, range(len(p)), 'lightgrey', size=0.2, alpha=0.25)
    scatter3(ax7, only_in_hpro, 'purple',    size=4,   label=f'extra ({len(only_in_hpro)})')
    add_viewpoint(ax7); add_frustum(ax7)
    ax7.legend(fontsize=7, markerscale=2)
    finish(ax7, f'False positives — HPRO only ({len(only_in_hpro)})')
    set_axes_equal_3d(ax7)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def run_optimization(model, pts_torch, init_viewpoint_np, init_rot_6d_np,
                     gamma, alphas, k, delta, use_linear_kernel,
                     n_steps=300, lr=5e-3, log_every=10):
    """
    Optimize viewpoint position and frustum orientation to maximise the
    total differentiable visibility score (sum of w_combined).

    Args:
        model           : HPRO_limited instance (fixed FOV / near / far).
        pts_torch       : (1, 3, N) point cloud tensor.
        init_viewpoint_np: (3,) numpy array — starting viewpoint.
        init_rot_6d_np  : (6,) numpy array — starting 6D orientation.
        gamma, alphas, k, delta, use_linear_kernel: HPRO parameters.
        n_steps (int)   : Number of gradient steps.
        lr (float)      : Adam learning rate.
        log_every (int) : Print / record every this many steps.

    Returns:
        dict with keys:
            'losses'          — list of loss values at logged steps
            'soft_counts'     — list of sum(w_combined) at logged steps
            'steps'           — list of step indices
            'viewpoint_traj'  — (T, 3) numpy array of viewpoint positions
            'final_viewpoint' — (3,) numpy array
            'final_rot_6d'    — (6,) numpy array
    """
    device = model.device

    viewpoint = torch.nn.Parameter(
        torch.from_numpy(init_viewpoint_np[np.newaxis].copy()).to(device)
    )  # (1, 3)
    rot_6d = torch.nn.Parameter(
        torch.from_numpy(init_rot_6d_np[np.newaxis].copy()).to(device)
    )  # (1, 6)

    optimizer = torch.optim.Adam([viewpoint, rot_6d], lr=lr)

    pts_dev = pts_torch.to(device)

    losses         = []
    soft_counts    = []
    steps_log      = []
    viewpoint_traj = []

    print(f"\nOptimizing for {n_steps} steps (lr={lr})...")
    for step in range(n_steps):
        optimizer.zero_grad()

        _, _, w_combined = model(
            pts_dev, viewpoint, rot_6d,
            gamma=gamma, alphas=alphas, k=k, delta=delta,
            use_linear_kernel=use_linear_kernel,
        )

        # Maximise total soft visibility
        loss = -w_combined.sum()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == n_steps - 1:
            l_val  = loss.item()
            sc_val = w_combined.sum().item()
            vp_val = viewpoint.detach().cpu().numpy().flatten().copy()
            losses.append(l_val)
            soft_counts.append(sc_val)
            steps_log.append(step)
            viewpoint_traj.append(vp_val)
            print(f"  step {step:4d}  loss={l_val:10.2f}  "
                  f"soft_visible={sc_val:8.2f}  "
                  f"vp={np.round(vp_val, 3)}")

    return {
        'losses':          losses,
        'soft_counts':     soft_counts,
        'steps':           steps_log,
        'viewpoint_traj':  np.array(viewpoint_traj),   # (T, 3)
        'final_viewpoint': viewpoint.detach().cpu().numpy().flatten(),
        'final_rot_6d':    rot_6d.detach().cpu().numpy().flatten(),
    }


def plot_optimization(pts_np,
                      init_viewpoint_np, init_rot_6d_np,
                      opt_result,
                      fov_h, fov_v, near, far,
                      init_visible_idx, opt_visible_idx):
    """
    Four-panel figure for the optimization result:
      1. Loss curve
      2. Soft visible-count curve
      3. Initial frustum + highlighted visible points
      4. Optimised frustum + highlighted visible points + viewpoint trajectory
    """
    p    = pts_np
    traj = opt_result['viewpoint_traj']   # (T, 3)
    vp_init = init_viewpoint_np
    vp_opt  = opt_result['final_viewpoint']

    # Recover camera frames
    init_look, init_up, init_right = build_camera_frame(
        init_rot_6d_np[:3], init_rot_6d_np[3:]
    )
    opt_6d = opt_result['final_rot_6d']
    opt_look, opt_up, opt_right = build_camera_frame(
        opt_6d[:3], opt_6d[3:]
    )

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("HPRO_limited Optimization", fontsize=14)

    # ---- 1: Loss curve -------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(opt_result['steps'], opt_result['losses'],
             color='crimson', linewidth=1.8)
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss  (−Σw)')
    ax1.set_title('Loss (−total soft visibility)')
    ax1.grid(True, alpha=0.4)

    # ---- 2: Soft visible count -----------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(opt_result['steps'], opt_result['soft_counts'],
             color='royalblue', linewidth=1.8)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Σ w_combined')
    ax2.set_title('Total soft visibility score')
    ax2.grid(True, alpha=0.4)

    # ---- 3: Initial pose -----------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')

    # All points (background)
    ax3.scatter(p[:, 0], p[:, 1], p[:, 2],
                c='lightgrey', s=0.3, alpha=0.25, depthshade=False)
    # HPRO_limited–visible at initial pose
    if len(init_visible_idx) > 0:
        ax3.scatter(p[init_visible_idx, 0],
                    p[init_visible_idx, 1],
                    p[init_visible_idx, 2],
                    c='steelblue', s=3, alpha=0.8,
                    label=f'visible ({len(init_visible_idx)})', depthshade=False)
    ax3.scatter(*vp_init, c='red', s=60, marker='^', zorder=5, label='viewpoint')
    draw_frustum(ax3, vp_init, init_look, init_up, init_right,
                 fov_h, fov_v, near, far, color='orange')
    ax3.set_title(f'Initial pose  ({len(init_visible_idx)} visible)')
    ax3.legend(fontsize=7, markerscale=2)
    ax3.set_xlabel('X', fontsize=8); ax3.set_ylabel('Y', fontsize=8)
    ax3.tick_params(labelsize=7)
    set_axes_equal_3d(ax3)

    # ---- 4: Optimised pose + trajectory --------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    # All points (background)
    ax4.scatter(p[:, 0], p[:, 1], p[:, 2],
                c='lightgrey', s=0.3, alpha=0.25, depthshade=False)
    # HPRO_limited–visible at optimised pose
    if len(opt_visible_idx) > 0:
        ax4.scatter(p[opt_visible_idx, 0],
                    p[opt_visible_idx, 1],
                    p[opt_visible_idx, 2],
                    c='green', s=3, alpha=0.8,
                    label=f'visible ({len(opt_visible_idx)})', depthshade=False)
    # Viewpoint trajectory
    ax4.plot(traj[:, 0], traj[:, 1], traj[:, 2],
             c='tomato', linewidth=1.5, label='vp trajectory', zorder=4)
    ax4.scatter(*vp_init, c='orange',   s=60, marker='o', zorder=5, label='vp initial')
    ax4.scatter(*vp_opt,  c='red',      s=80, marker='^', zorder=5, label='vp final')
    draw_frustum(ax4, vp_opt, opt_look, opt_up, opt_right,
                 fov_h, fov_v, near, far, color='limegreen')
    ax4.set_title(f'Optimised pose  ({len(opt_visible_idx)} visible)')
    ax4.legend(fontsize=7, markerscale=2)
    ax4.set_xlabel('X', fontsize=8); ax4.set_ylabel('Y', fontsize=8)
    ax4.tick_params(labelsize=7)
    set_axes_equal_3d(ax4)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # ---- Configuration -------------------------------------------------------
    seed       = 17
    num_points = 10000
    device     = 'cuda'

    viewpoint_list = [0.0, 0.0, -2.0]

    # Camera orientation: look along +Z, up along +Y
    look_approx = np.array([0.0, 0.0,  1.0], dtype=np.float64)
    up_approx   = np.array([0.0, 1.0,  0.0], dtype=np.float64)

    # Frustum parameters
    fov_h             = math.radians(30)
    fov_v             = math.radians(35)
    near              = 0.5
    far               = 2.5
    frustum_sharpness = 50.0    # sigmoid steepness — higher = sharper boundary

    # HPRO parameters (match demo.py)
    gamma             = -math.exp(-3.0)
    k                 = 10
    alphas            = []
    delta             = 0.0
    use_linear_kernel = False

    # Combined-score threshold (lower than plain HPRO's 0.99 because the
    # frustum sigmoid attenuates the raw w score)
    visibility_thresh = 0.6

    # ---- Load and normalise mesh ---------------------------------------------
    print("Loading mesh...")
    np.random.seed(seed)
    mesh = trimesh.load_mesh("lamp_0001.off")
    mesh.vertices -= np.mean(mesh.vertices, axis=0)
    mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

    # ---- Sample surface points -----------------------------------------------
    print(f"Sampling {num_points} points...")
    pts_surface, _ = trimesh.sample.sample_surface(
        mesh, count=num_points, face_weight=None, sample_color=False
    )  # (N, 3) numpy float64

    # ---- Build orthonormal camera frame (Gram-Schmidt) -----------------------
    look, up, right = build_camera_frame(look_approx, up_approx)
    viewpoint_np    = np.array(viewpoint_list, dtype=np.float64)

    print(f"\nCamera frame:")
    print(f"  viewpoint : {viewpoint_np}")
    print(f"  look      : {np.round(look,  4)}")
    print(f"  up        : {np.round(up,    4)}")
    print(f"  right     : {np.round(right, 4)}")
    print(f"  fov_h={math.degrees(fov_h):.1f}°  fov_v={math.degrees(fov_v):.1f}°  "
          f"near={near}  far={far}")

    # ---- Ground truth (frustum test + ray-cast) ------------------------------
    print("\nComputing ground truth (frustum test + ray-cast)...")
    gt_indices, frustum_indices = compute_ground_truth(
        mesh, viewpoint_np, pts_surface,
        look, up, right,
        fov_h, fov_v, near, far,
    )
    print(f"  Points inside frustum           : {len(frustum_indices)} / {num_points}")
    print(f"  GT visible (unoccluded)         : {len(gt_indices)}      / {num_points}")

    # ---- Prepare tensors for HPRO_limited ------------------------------------
    pts_torch = torch.tensor(
        pts_surface.T[np.newaxis], dtype=torch.float64,   # (1, 3, N)
    )
    viewpoint_torch = torch.tensor(
        [viewpoint_list], dtype=torch.float64,            # (1, 3)
    )
    # rot_6d = [a1 (≈ look) | a2 (≈ up)] — Gram-Schmidt applied inside the model
    rot_6d_torch = torch.tensor(
        [np.concatenate([look_approx, up_approx])],       # (1, 6)
        dtype=torch.float64,
    )

    # ---- HPRO_limited --------------------------------------------------------
    print("\nRunning HPRO_limited...")
    model = HPRO_limited(
        fits_in_memory=True,
        visibility_score_thresh=visibility_thresh,
        fov_h=fov_h, fov_v=fov_v,
        near=near, far=far,
        frustum_sharpness=frustum_sharpness,
        device=device,
    )
    with torch.no_grad():
        vis_pts, vis_idx, w_combined = model(
            pts_torch, viewpoint_torch, rot_6d_torch,
            gamma=gamma, alphas=alphas, k=k, delta=delta,
            use_linear_kernel=use_linear_kernel,
        )
    hpro_limited_indices = vis_idx.cpu().numpy().flatten().astype(np.int32)

    # ---- Error metrics -------------------------------------------------------
    gt_set   = set(int(i) for i in gt_indices)
    hpro_set = set(int(i) for i in hpro_limited_indices)

    only_in_gt   = sorted(gt_set   - hpro_set)   # false negatives
    only_in_hpro = sorted(hpro_set - gt_set)      # false positives
    tp           = gt_set & hpro_set              # true positives

    precision = len(tp) / max(len(hpro_set), 1)
    recall    = len(tp) / max(len(gt_set),   1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    print("\n--- Results ---")
    print(f"  Total points sampled              : {num_points}")
    print(f"  GT visible (frustum + ray-cast)   : {len(gt_set)}")
    print(f"  HPRO_limited visible              : {len(hpro_set)}")
    print(f"  True  positives (intersection)    : {len(tp)}")
    print(f"  False negatives (in GT only)      : {len(only_in_gt)}")
    print(f"  False positives (in HPRO only)    : {len(only_in_hpro)}")
    print(f"  Precision                         : {precision:.4f}")
    print(f"  Recall                            : {recall:.4f}")
    print(f"  F1 score                          : {f1:.4f}")

    # ---- Visualise evaluation ------------------------------------------------
    print("\nPlotting evaluation results...")
    plot_results(
        pts_surface, viewpoint_np,
        look, up, right,
        fov_h, fov_v, near, far,
        gt_indices, frustum_indices,
        list(hpro_set),
        only_in_gt, only_in_hpro,
    )

    # =========================================================================
    # Optimization test
    # =========================================================================
    # Start from a perturbed viewpoint and orientation and let Adam maximise
    # the total differentiable visibility score.  We then evaluate the
    # optimised pose with the hard frustum + ray-cast ground truth and compare
    # against the initial pose.
    # =========================================================================

    print("\n" + "="*60)
    print("OPTIMIZATION TEST")
    print("="*60)

    # ---- Initial (perturbed) pose --------------------------------------------
    # Offset the viewpoint and tilt the look direction so the frustum starts
    # pointing away from the object centre, giving the optimizer room to work.
    opt_init_viewpoint = np.array([0.0, 0.0, -2.0], dtype=np.float64)
    opt_init_look_approx = np.array([0.3, -0.4, 1.0], dtype=np.float64)   # tilted away
    opt_init_up_approx   = np.array([0.0,  1.0, 0.0], dtype=np.float64)
    opt_init_rot_6d = np.concatenate([opt_init_look_approx, opt_init_up_approx])

    # Evaluate initial pose with HPRO_limited (no grad)
    with torch.no_grad():
        _, init_vis_idx, _ = model(
            pts_torch.to(device),
            torch.tensor([opt_init_viewpoint], dtype=torch.float64).to(device),
            torch.tensor([opt_init_rot_6d],    dtype=torch.float64).to(device),
            gamma=gamma, alphas=alphas, k=k, delta=delta,
            use_linear_kernel=use_linear_kernel,
        )
    init_visible_idx = init_vis_idx.cpu().numpy().flatten().astype(np.int32)

    init_look_opt, init_up_opt, _ = build_camera_frame(
        opt_init_look_approx, opt_init_up_approx
    )
    print(f"\nInitial (perturbed) pose:")
    print(f"  viewpoint : {opt_init_viewpoint}")
    print(f"  look      : {np.round(init_look_opt, 3)}")
    print(f"  HPRO_limited visible : {len(init_visible_idx)}")

    # ---- Run optimizer -------------------------------------------------------
    opt_result = run_optimization(
        model, pts_torch,
        init_viewpoint_np=opt_init_viewpoint,
        init_rot_6d_np=opt_init_rot_6d,
        gamma=gamma, alphas=alphas, k=k, delta=delta,
        use_linear_kernel=use_linear_kernel,
        n_steps=300,
        lr=5e-3,
        log_every=20,
    )

    # ---- Evaluate optimised pose with HPRO_limited --------------------------
    vp_opt_t  = torch.tensor([opt_result['final_viewpoint']], dtype=torch.float64)
    r6d_opt_t = torch.tensor([opt_result['final_rot_6d']],    dtype=torch.float64)
    with torch.no_grad():
        _, opt_vis_idx, _ = model(
            pts_torch.to(device),
            vp_opt_t.to(device),
            r6d_opt_t.to(device),
            gamma=gamma, alphas=alphas, k=k, delta=delta,
            use_linear_kernel=use_linear_kernel,
        )
    opt_visible_idx = opt_vis_idx.cpu().numpy().flatten().astype(np.int32)

    # ---- Evaluate optimised pose with hard GT --------------------------------
    opl, opu, opr = build_camera_frame(
        opt_result['final_rot_6d'][:3],
        opt_result['final_rot_6d'][3:],
    )
    print("\nComputing GT for optimised pose (frustum test + ray-cast)...")
    opt_gt_indices, opt_frustum_indices = compute_ground_truth(
        mesh, opt_result['final_viewpoint'], pts_surface,
        opl, opu, opr, fov_h, fov_v, near, far,
    )

    opt_gt_set   = set(int(i) for i in opt_gt_indices)
    opt_hpro_set = set(int(i) for i in opt_visible_idx)
    opt_tp       = opt_gt_set & opt_hpro_set
    opt_prec     = len(opt_tp) / max(len(opt_hpro_set), 1)
    opt_rec      = len(opt_tp) / max(len(opt_gt_set),   1)
    opt_f1       = 2 * opt_prec * opt_rec / max(opt_prec + opt_rec, 1e-9)

    print(f"\n--- Optimised Pose Results ---")
    print(f"  Final viewpoint        : {np.round(opt_result['final_viewpoint'], 4)}")
    print(f"  Final look direction   : {np.round(opl, 4)}")
    print(f"  GT visible (hard)      : {len(opt_gt_set)}")
    print(f"  HPRO_limited visible   : {len(opt_hpro_set)}")
    print(f"  True  positives        : {len(opt_tp)}")
    print(f"  Precision              : {opt_prec:.4f}")
    print(f"  Recall                 : {opt_rec:.4f}")
    print(f"  F1 score               : {opt_f1:.4f}")

    # ---- Improvement summary -------------------------------------------------
    init_gt_indices, _ = compute_ground_truth(
        mesh, opt_init_viewpoint, pts_surface,
        init_look_opt, init_up_opt, np.cross(init_look_opt, init_up_opt),
        fov_h, fov_v, near, far,
    )
    print(f"\n--- Improvement (hard GT visible count) ---")
    print(f"  Initial pose : {len(init_gt_indices)}  points")
    print(f"  Optimised    : {len(opt_gt_indices)}  points")
    delta_count = len(opt_gt_indices) - len(init_gt_indices)
    print(f"  Delta        : {delta_count:+d}  ({'+' if delta_count>=0 else ''}"
          f"{100*delta_count/max(len(init_gt_indices),1):.1f}%)")

    # ---- Visualise optimization ----------------------------------------------
    print("\nPlotting optimization results...")
    plot_optimization(
        pts_surface,
        opt_init_viewpoint, opt_init_rot_6d,
        opt_result,
        fov_h, fov_v, near, far,
        init_visible_idx,
        opt_visible_idx,
    )
