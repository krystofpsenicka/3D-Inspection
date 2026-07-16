"""Probe the pretrained NVPS (neural-visibility) model on our meshes.

Measures occlusion-only (full-sphere, no frustum) visibility accuracy vs
Embree ray-cast GT, at three viewpoint radii:
  - R_train convention: bbox-diagonal norm (prepare_data.py line 140)
  - R=1.5 and R=6.0 (inspection standoffs, OOD)
Compares with plain HPRO (gamma=-e^-7, thresh 0.99) on the same viewpoints.
Also verifies gradients flow from the visibility output to the viewpoint.
"""
import os, sys, time
import numpy as np
import torch
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
from models import MyNet, get_embedder          # neural-visibility
from HPRO import HPRO                            # ours

device = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH, FULL_DEPTH = 8, 2
N_POINTS = 3000
N_VP = 12
SEED = 17

# ---------------- model ----------------
model = MyNet(in_channels=6, vp_channels=63, out_channels=2).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=True)
if isinstance(ckpt, dict) and "model_dict" in ckpt:
    ckpt = ckpt["model_dict"]
elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
    ckpt = ckpt["model"]
ckpt = { (k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items() }
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f"checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
if missing[:5]: print("  missing e.g.:", missing[:5])
if unexpected[:5]: print("  unexpected e.g.:", unexpected[:5])
model.eval()
embedder, _ = get_embedder(10)


def get_input_feature(octree):
    depth = octree.depth
    local_points = octree.points[depth].frac() - 0.5
    scale = 2 ** (1 - depth)
    global_points = octree.points[depth] * scale - 1.0
    return torch.cat([local_points, global_points], dim=1)


def nvps_features(pts_np, normals_np):
    """Build octree + run UNet once. Returns (feature (N,63), order info)."""
    xyz = torch.tensor(pts_np, dtype=torch.float32, device=device)
    nrm = torch.tensor(normals_np, dtype=torch.float32, device=device)
    points = Points(xyz.clone(), nrm.clone())
    bbmin, bbmax = points.bbox()
    points.normalize(bbmin, bbmax, scale=0.8)
    octree = Octree(DEPTH, FULL_DEPTH, device=device)
    octree.build_octree(points)
    octree.construct_all_neigh()
    feat = get_input_feature(octree)
    batch_id = torch.zeros(points.points.shape[0], 1, device=device)
    query_pts = torch.cat([points.points, batch_id], dim=1)
    with torch.no_grad():
        t0 = time.perf_counter()
        feature = model.UNet(feat, octree, octree.depth, query_pts)
        torch.cuda.synchronize() if device == "cuda" else None
        t_unet = time.perf_counter() - t0
    return feature, t_unet


def nvps_predict(feature, pts_np, vp_np, grad_check=False):
    """Per-point 2-class logits for one viewpoint. Returns (probs, t)."""
    xyz = torch.tensor(pts_np, dtype=torch.float32, device=device)
    vp = torch.tensor(vp_np, dtype=torch.float32, device=device,
                      requires_grad=grad_check)
    t0 = time.perf_counter()
    vd = xyz - vp                       # (N, 3): viewpoint -> point
    vd = vd / vd.norm(dim=1, keepdim=True)
    vd_e = embedder(vd)
    logits = model.VisNet(feature * vd_e).view(-1, 2)
    probs = torch.softmax(logits, dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    if grad_check:
        probs[:, 0].mean().backward()
        g = vp.grad
        print(f"  [grad check] dC/dvp = {g.detach().cpu().numpy().round(5)}  "
              f"|g|={g.norm().item():.3e}")
    return probs.detach(), t


def occlusion_gt(mesh, vp_np, pts_np, tol=1e-6):
    """Occlusion-only GT: visible iff nearest ray hit is the point itself."""
    intersector = mesh.ray
    origins = np.repeat(vp_np[None], len(pts_np), axis=0)
    dirs = pts_np - vp_np[None]
    locs, idx_ray, _ = intersector.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True)
    visible = np.zeros(len(pts_np), dtype=bool)
    if len(idx_ray) > 0:
        d_hit = np.linalg.norm(locs - vp_np[None], axis=1)
        order = np.lexsort((d_hit, idx_ray))
        rays_s, locs_s = idx_ray[order], locs[order]
        first = np.ones(len(rays_s), dtype=bool)
        first[1:] = rays_s[1:] != rays_s[:-1]
        nearest_ray, nearest_loc = rays_s[first], locs_s[first]
        dd = np.linalg.norm(nearest_loc - pts_np[nearest_ray], axis=1)
        visible[nearest_ray[dd < tol]] = True
    return visible


def prf(pred, gt):
    tp = np.sum(pred & gt); fp = np.sum(pred & ~gt); fn = np.sum(~pred & gt)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    acc = np.mean(pred == gt)
    return acc, p, r, f1


hpro_op = HPRO(fits_in_memory=True, visiblity_score_thresh=0.99, device=device)
GAMMA = -np.exp(-7.0)


def hpro_predict(pts_np, vp_np):
    pts_t = torch.tensor(pts_np.T, dtype=torch.float64, device=device).unsqueeze(0)
    vp_t = torch.tensor(vp_np, dtype=torch.float64, device=device).view(1, 3)
    t0 = time.perf_counter()
    with torch.no_grad():
        _, _, w = hpro_op(pts_t, vp_t, gamma=GAMMA, k=10)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return (w.view(-1) > 0.99).cpu().numpy(), t


def run_mesh(mesh_path, name):
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

    feature, t_unet = nvps_features(pts_np, normals_np)
    print(f"UNet feature extraction: {t_unet*1e3:.0f} ms (once per cloud)")

    r_train = np.linalg.norm(pts_np.max(axis=0) - pts_np.min(axis=0))
    rng = np.random.default_rng(SEED)

    for R, tag in [(r_train, f"R_train={r_train:.2f}"), (1.5, "R=1.50"), (6.0, "R=6.00")]:
        dirs = rng.standard_normal((N_VP, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        stats = {m: [] for m in ("nv0", "nv1", "hpro")}
        times = {"nv": [], "hpro": []}
        for i in range(N_VP):
            vp = dirs[i] * R
            gt = occlusion_gt(mesh, vp, pts_np)
            probs, t_nv = nvps_predict(feature, pts_np, vp)
            pred0 = (probs[:, 0] > 0.5).cpu().numpy()   # class 0 = visible?
            pred1 = (probs[:, 1] > 0.5).cpu().numpy()   # class 1 = visible?
            ph, t_h = hpro_predict(pts_np, vp)
            stats["nv0"].append(prf(pred0, gt))
            stats["nv1"].append(prf(pred1, gt))
            stats["hpro"].append(prf(ph, gt))
            times["nv"].append(t_nv); times["hpro"].append(t_h)
        for m, label in (("nv0", "NVPS(cls0=vis)"), ("nv1", "NVPS(cls1=vis)"),
                         ("hpro", "HPRO w>0.99  ")):
            a = np.mean(stats[m], axis=0)
            print(f"  [{tag}] {label}: acc={a[0]:.3f} prec={a[1]:.3f} "
                  f"rec={a[2]:.3f} F1={a[3]:.3f}")
        print(f"  [{tag}] per-view time: NVPS-MLP {np.mean(times['nv'])*1e3:.1f} ms | "
              f"HPRO {np.mean(times['hpro'])*1e3:.1f} ms")

    # gradient check at R=2.0
    print("  gradient flow check (R=2):")
    nvps_predict(feature, pts_np, dirs[0] * 2.0, grad_check=True)


run_mesh(os.path.join(HPRO_DIR, "lamp_0001.off"), "lamp_0001.off (ModelNet40)")
run_mesh(os.path.join(os.path.dirname(HPRO_DIR), "models",
                      "duke_of_lancaster_uk_clipped.glb"),
         "duke_of_lancaster wreck")
