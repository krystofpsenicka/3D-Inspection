"""Diagnostics for Stage-2 multi-viewpoint HPRO optimisation.

Tests three hypotheses on lamp_0001.off:
  H1: soft coverage C is miscalibrated (overestimates GT coverage).
  H2: pure gradient from Fibonacci init hits local minima (esp. V=1).
  H3: greedy-oracle init + gradient refinement beats both pure greedy
      and pure gradient.

Also reports final viewpoint standoff distances and per-view precision /
recall of thresholded w_combined vs. ray-cast GT at the final poses.
"""
import math, os, sys, time
import numpy as np
import torch
import trimesh

HPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HPRO_DIR)

from HPRO_limited import HPRO_limited
from frustum_gt import build_camera_frame, compute_ground_truth_batched
from multi_viewpoint import optimize_viewpoints, fibonacci_sphere_viewpoints
from eval_multi import compute_baseline_gt, fibonacci_sphere

SEED = 17
N_POINTS = 3000
NEAR, FAR = 0.1, 6.0
FOV_H, FOV_V = math.radians(30.0), math.radians(35.0)
GAMMA = -math.exp(-7.0)
K = 10
N_STEPS = 300
LR = 1e-2

device = "cuda" if torch.cuda.is_available() else "cpu"

MESH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HPRO_DIR, "lamp_0001.off")
mesh = trimesh.load_mesh(MESH)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
mesh.vertices -= np.mean(mesh.vertices, axis=0)
mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()
np.random.seed(SEED)
pts_np, _ = trimesh.sample.sample_surface(mesh, count=N_POINTS)
N = len(pts_np)
centroid = pts_np.mean(axis=0)

model = HPRO_limited(fits_in_memory=True, visibility_score_thresh=0.5,
                     fov_h=FOV_H, fov_v=FOV_V, near=NEAR, far=FAR,
                     frustum_sharpness=50.0, device=device)

intersector = mesh.ray


def gt_union(vps, r6ds):
    covered = set()
    per_vp = []
    for v in range(len(vps)):
        look, up, right = build_camera_frame(r6ds[v, :3], r6ds[v, 3:])
        gt_idx, _ = compute_ground_truth_batched(
            mesh, vps[v], pts_np, look, up, right,
            FOV_H, FOV_V, NEAR, FAR, intersector=intersector)
        per_vp.append(set(int(i) for i in gt_idx))
        covered |= per_vp[-1]
    return covered, per_vp


def soft_scores(vps, r6ds):
    pts_t = torch.tensor(pts_np.T, dtype=torch.float32, device=device
                         ).unsqueeze(0).expand(len(vps), -1, -1)
    vp_t = torch.tensor(vps, dtype=torch.float32, device=device)
    r6_t = torch.tensor(r6ds, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, w = model(pts_t, vp_t, r6_t, gamma=GAMMA, k=K)
    return w.cpu().numpy()   # (V, N)


def calib_report(vps, r6ds, tag):
    w = soft_scores(vps, r6ds)
    _, per_vp_gt = gt_union(vps, r6ds)
    soft_cover = float(np.mean(1.0 - np.prod(1.0 - np.clip(w, 0, 1 - 1e-7), axis=0)))
    hard_union = set()
    precs, recs = [], []
    for v in range(len(vps)):
        pred = set(np.where(w[v] > 0.5)[0].tolist())
        gt = per_vp_gt[v]
        hard_union |= pred
        tp = len(pred & gt)
        precs.append(tp / max(len(pred), 1))
        recs.append(tp / max(len(gt), 1))
    gt_u = set().union(*per_vp_gt) if per_vp_gt else set()
    print(f"  [{tag}] soft C={soft_cover:.3f}  GT union={len(gt_u)/N:.3f}  "
          f"pred>0.5 union={len(hard_union)/N:.3f}  "
          f"mean prec={np.mean(precs):.3f}  mean rec={np.mean(recs):.3f}")
    d = np.linalg.norm(vps[:, None, :] - pts_np[None], axis=2).min(axis=1)
    print(f"           standoff min-dist per vp: {np.round(d, 2).tolist()}")
    return len(gt_u) / N


def greedy_poses(candidate_gt, dirs, V):
    """Greedy set-cover, returning selected candidate indices."""
    covered = set()
    sel = []
    for _ in range(V):
        best_gain, best_j = -1, 0
        for j, gt_j in enumerate(candidate_gt):
            gain = len(gt_j - covered)
            if gain > best_gain:
                best_gain, best_j = gain, j
        covered |= candidate_gt[best_j]
        sel.append(best_j)
    return sel, covered


print(f"device={device}  N={N}")
print("Precomputing greedy candidates (50, oracle GT)...")
candidate_gt, t_rc = compute_baseline_gt(mesh, pts_np, FOV_H, FOV_V, NEAR, FAR,
                                         n_candidates=50, seed=SEED)
rng = np.random.default_rng(SEED)
dirs = fibonacci_sphere(50)
dirs = dirs + 0.05 * rng.standard_normal(dirs.shape)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
radius = (NEAR + FAR) / 2.0

for V in (1, 5):
    print(f"\n================ V={V} ================")
    sel, covered_greedy = greedy_poses(candidate_gt, dirs, V)
    print(f"greedy oracle GT coverage: {len(covered_greedy)/N:.3f}")

    # Greedy candidate poses -> vp + rot6d
    world_up = np.array([0., 0., 1.])
    g_vps = np.array([centroid + radius * dirs[j] for j in sel])
    g_r6d = np.zeros((V, 6))
    for i, j in enumerate(sel):
        look = -dirs[j]
        ln = look / np.linalg.norm(look)
        up = world_up if abs(np.dot(ln, world_up)) < 0.95 else np.array([0., 1., 0.])
        g_r6d[i, :3] = look
        g_r6d[i, 3:] = up

    # A) pure gradient from Fibonacci init
    res_f = optimize_viewpoints(pts_np, V, model, gamma=GAMMA, k=K,
                                n_steps=N_STEPS, lr=LR, near_dist=NEAR,
                                far_dist=FAR, log_every=10**9, seed=SEED)
    print(f"\nA) Fibonacci-init gradient:  final soft C={res_f.coverage_history[-1]:.3f}")
    calib_report(res_f.viewpoints_np, res_f.rot_6d_np, "A final")

    # B) greedy-init + gradient refinement
    res_g = optimize_viewpoints(pts_np, V, model, gamma=GAMMA, k=K,
                                n_steps=N_STEPS, lr=LR, near_dist=NEAR,
                                far_dist=FAR,
                                init_viewpoints_np=g_vps, init_rot_6d_np=g_r6d,
                                log_every=10**9, seed=SEED)
    print(f"\nB) Greedy-init + refine:  final soft C={res_g.coverage_history[-1]:.3f}")
    calib_report(g_vps, g_r6d, "B init (greedy)")
    calib_report(res_g.viewpoints_np, res_g.rot_6d_np, "B final")
