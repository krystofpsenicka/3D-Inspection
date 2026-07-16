"""Go/no-go experiment: NVPS backbone inside the Stage-2 frustum-gated
multi-viewpoint coverage optimizer, on the wreck (and lamp).

Combined score per viewpoint v, point j:
    w[v,j] = p_visible_NVPS(dir(v,j)) * f_frustum(v,j)
NVPS UNet features are viewpoint-independent -> computed once, frozen.
Optimize (viewpoints, rot6d) with Adam on soft set-cover coverage +
standoff/diversity penalties (identical to multi_viewpoint.py), from
(a) Fibonacci init and (b) oracle-greedy init.  Evaluate with hard
ray-cast GT.  Compare against the HPRO-backbone numbers from earlier.
"""
import math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

HPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NV_DIR = os.environ.get("NEUVIS_DIR",
                        os.path.join(HPRO_DIR, "external", "neural-visibility"))
CKPT = os.environ.get("NEUVIS_CKPT",
                      os.path.join(HPRO_DIR, "external", "neuvis_00040.pth"))
sys.path.insert(0, NV_DIR)
sys.path.insert(0, HPRO_DIR)

import ocnn_compat  # noqa: F401  -- must precede ocnn; see hpro/ocnn_compat.py
import ocnn
from ocnn.octree import Points, Octree
from models import MyNet, get_embedder
from HPRO_limited import HPRO_limited
from frustum_gt import build_camera_frame, compute_ground_truth_batched
from multi_viewpoint import (fibonacci_sphere_viewpoints, compute_coverage_soft,
                             _standoff_penalty, _diversity_penalty)
from eval_multi import compute_baseline_gt, fibonacci_sphere

SEED = 17
N_POINTS = 3000
NEAR, FAR = 0.1, 6.0
FOV_H, FOV_V = math.radians(30.0), math.radians(35.0)
SHARP = 50.0
N_STEPS = 300
LR = 1e-2
LAM_STAND, LAM_DIV, MIN_INTER = 1.0, 0.1, 0.3
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- NVPS model ----------------
model = MyNet(6, 63, 2).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=True)
ckpt = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
model.load_state_dict(ckpt)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
embedder, _ = get_embedder(10)


def get_input_feature(octree):
    depth = octree.depth
    local_points = octree.points[depth].frac() - 0.5
    scale = 2 ** (1 - depth)
    global_points = octree.points[depth] * scale - 1.0
    return torch.cat([local_points, global_points], dim=1)


def nvps_features(pts_np, normals_np):
    xyz = torch.tensor(pts_np, dtype=torch.float32, device=device)
    nrm = torch.tensor(normals_np, dtype=torch.float32, device=device)
    points = Points(xyz.clone(), nrm.clone())
    bbmin, bbmax = points.bbox()
    points.normalize(bbmin, bbmax, scale=0.8)
    octree = Octree(8, 2, device=device)
    octree.build_octree(points)
    octree.construct_all_neigh()
    feat = get_input_feature(octree)
    bid = torch.zeros(points.points.shape[0], 1, device=device)
    qp = torch.cat([points.points, bid], dim=1)
    with torch.no_grad():
        return model.UNet(feat, octree, octree.depth, qp)   # (N, 63)


# frustum helpers reused from HPRO_limited (static)
hl = HPRO_limited(device=device)   # only for _6d_to_rotation_matrix / mask


def nvps_w_combined(feature, pts_t, viewpoints, rot_6d):
    """(V, N) differentiable combined scores. pts_t: (3, N) float32."""
    V = viewpoints.shape[0]
    N = pts_t.shape[1]
    # NVPS visibility per (v, j)
    vd = pts_t.T.unsqueeze(0) - viewpoints.unsqueeze(1)     # (V, N, 3)
    vd = vd / vd.norm(dim=2, keepdim=True)
    vd_e = embedder(vd.reshape(-1, 3))                       # (V*N, 63)
    feat = feature.unsqueeze(0).expand(V, -1, -1).reshape(-1, 63)
    logits = model.VisNet(feat * vd_e).view(-1, 2)
    p_vis = torch.softmax(logits, dim=1)[:, 1].view(V, N)    # class 1 = visible
    # frustum gate
    look, up, right = HPRO_limited._6d_to_rotation_matrix(rot_6d)
    f = hl.compute_frustum_mask(
        pts_t.unsqueeze(0).expand(V, -1, -1), viewpoints.unsqueeze(2),
        look, up, right, FOV_H, FOV_V, NEAR, FAR, SHARP)
    return p_vis * f


def optimize_nvps(pts_np, feature, init_vp, init_r6d):
    pts_t = torch.tensor(pts_np.T, dtype=torch.float32, device=device)
    pts_surf = torch.tensor(pts_np, dtype=torch.float32, device=device)
    vp = torch.nn.Parameter(torch.tensor(init_vp, dtype=torch.float32, device=device))
    r6 = torch.nn.Parameter(torch.tensor(init_r6d, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([vp, r6], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS,
                                                       eta_min=LR * 0.01)
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        opt.zero_grad()
        w = nvps_w_combined(feature, pts_t, vp, r6)
        C = compute_coverage_soft(w)
        loss = (-C + LAM_STAND * _standoff_penalty(vp, pts_surf, NEAR, FAR)
                + LAM_DIV * _diversity_penalty(vp, MIN_INTER))
        loss.backward()
        opt.step()
        sched.step()
    return (vp.detach().cpu().numpy(), r6.detach().cpu().numpy(),
            C.item(), time.perf_counter() - t0)


def gt_union(mesh, pts_np, vps, r6ds):
    intersector = mesh.ray
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


def report(mesh, pts_np, feature, vps, r6ds, tag):
    N = len(pts_np)
    pts_t = torch.tensor(pts_np.T, dtype=torch.float32, device=device)
    with torch.no_grad():
        w = nvps_w_combined(feature, pts_t,
                            torch.tensor(vps, dtype=torch.float32, device=device),
                            torch.tensor(r6ds, dtype=torch.float32, device=device))
    w = w.cpu().numpy()
    soft = float(np.mean(1 - np.prod(1 - np.clip(w, 0, 1 - 1e-7), axis=0)))
    gt_u, per_vp = gt_union(mesh, pts_np, vps, r6ds)
    precs, recs = [], []
    for v in range(len(vps)):
        pred = set(np.where(w[v] > 0.5)[0].tolist())
        tp = len(pred & per_vp[v])
        precs.append(tp / max(len(pred), 1))
        recs.append(tp / max(len(per_vp[v]), 1))
    d = np.linalg.norm(vps[:, None] - pts_np[None], axis=2).min(axis=1)
    print(f"  [{tag}] soft C={soft:.3f}  GT union={len(gt_u)/N:.3f}  "
          f"prec={np.mean(precs):.3f}  rec={np.mean(recs):.3f}  "
          f"standoff={np.round(d, 2).tolist()}")
    return len(gt_u) / N


def greedy_poses(candidate_gt, dirs, V, centroid, radius):
    covered, sel = set(), []
    for _ in range(V):
        best_gain, best_j = -1, 0
        for j, gt_j in enumerate(candidate_gt):
            gain = len(gt_j - covered)
            if gain > best_gain:
                best_gain, best_j = gain, j
        covered |= candidate_gt[best_j]
        sel.append(best_j)
    world_up = np.array([0., 0., 1.])
    vps = np.array([centroid + radius * dirs[j] for j in sel])
    r6d = np.zeros((V, 6))
    for i, j in enumerate(sel):
        look = -dirs[j]
        ln = look / np.linalg.norm(look)
        up = world_up if abs(np.dot(ln, world_up)) < 0.95 else np.array([0., 1., 0.])
        r6d[i, :3], r6d[i, 3:] = look, up
    return vps, r6d, covered


def run(mesh_path, name, V_list=(1, 5)):
    print(f"\n{'='*62}\nMesh: {name}")
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.vertices -= np.mean(mesh.vertices, axis=0)
    mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()
    np.random.seed(SEED)
    pts_np, fidx = trimesh.sample.sample_surface(mesh, count=N_POINTS)
    pts_np = np.asarray(pts_np, dtype=np.float64)
    normals_np = np.asarray(mesh.face_normals[fidx], dtype=np.float64)
    N = len(pts_np)
    centroid = pts_np.mean(axis=0)
    radius = (NEAR + FAR) / 2.0

    feature = nvps_features(pts_np, normals_np)

    candidate_gt, _ = compute_baseline_gt(mesh, pts_np, FOV_H, FOV_V, NEAR, FAR,
                                          n_candidates=50, seed=SEED)
    rng = np.random.default_rng(SEED)
    dirs = fibonacci_sphere(50)
    dirs = dirs + 0.05 * rng.standard_normal(dirs.shape)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    for V in V_list:
        print(f"\n---- V={V} ----")
        g_vps, g_r6d, covered_g = greedy_poses(candidate_gt, dirs, V, centroid, radius)
        print(f"  greedy oracle GT coverage: {len(covered_g)/N:.3f}")

        f_vp, f_r6d = fibonacci_sphere_viewpoints(V, radius, centroid)
        vps, r6, C, t = optimize_nvps(pts_np, feature, f_vp, f_r6d)
        print(f"  A) NVPS fib-init ({t:.1f}s):")
        report(mesh, pts_np, feature, vps, r6, "A final")

        vps, r6, C, t = optimize_nvps(pts_np, feature, g_vps, g_r6d)
        print(f"  B) NVPS greedy-init + refine ({t:.1f}s):")
        report(mesh, pts_np, feature, g_vps, g_r6d, "B init")
        report(mesh, pts_np, feature, vps, r6, "B final")


run(os.path.join(os.path.dirname(HPRO_DIR), "models",
                 "duke_of_lancaster_uk_clipped.glb"),
    "duke_of_lancaster wreck")
run(os.path.join(HPRO_DIR, "lamp_0001.off"), "lamp_0001.off")
