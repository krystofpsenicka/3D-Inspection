"""NeOF head-to-head — the attack-0 numbers (RESEARCH_PLAN.md §6.3, §0.3 action 4).

Runs [NeOF (Cao et al., RA-L 2024)](https://arxiv.org/abs/2412.08266) from its
public code on our wreck benchmark and compares static camera *placement*
against our discrete baseline and our gradient refinement, all scored by the
same oracle: frustum ∩ Embree ray-cast against the mesh.

What is faithfully theirs: `initRandomCameras` (poses seeded on surface normals
at `height`), the full hybrid loop in `CameraLayerOpt.opt` (per-epoch neural
observation field refit, gradient stage through the field, elite resampling of
the worst camera every 5 epochs), their HPR-based visibility, their defaults
(epochs 20 × iterations 20). What is adapted, and why:

* **Intrinsics are patched to our tight camera** (fov 30°×35° instead of their
  ~90°×74°) and `height=1.0` so their depth band [0.5h, 1.5h] coincides with
  our standoff band [0.5, 1.5] — otherwise the comparison is between two
  different sensors, not two methods.
* **k-coverage = 1** (their default asks every point to be seen 3×; ours is
  set-cover).
* **Their `main.py` is bypassed**: it crashes on its own args (`args.scene` vs
  `--isscene`), and its dataset loader re-normalizes the cloud; we feed the
  same normalized cloud all our tables use.
* Their per-epoch prints are parsed from captured stdout to split wall-clock
  into field-refit vs the rest — the refit cost is reply (ii) to attack 0 and
  must be measured, not asserted.

Their rotation convention: ``p_cam = R (p - pos)``, camera looks along +z, so
world look = R[2], world up = R[1] (frustum test is symmetric, sign-free).

Usage::

    ipy hpro/eval_neof.py --seeds 3 --cameras 10,20 --no_show
"""

import argparse
import contextlib
import csv
import io
import math
import os
import re
import sys
import time
import types

import numpy as np
import torch
import trimesh

_DIR = os.path.dirname(os.path.abspath(__file__))
_NEOF = os.path.join(_DIR, "external", "NeOF-HybridCamOpt")
for p in (_DIR, _NEOF):
    if p not in sys.path:
        sys.path.insert(0, p)

from backbones import make_backbone                                    # noqa: E402
from eval_trajectory import (load_mesh, make_audit_fn,                 # noqa: E402
                             normal_offset_candidates, greedy_select)
from trajectory import standoff_penalty                                # noqa: E402
from visibility_layer import GatedVisibilityLayer                      # noqa: E402

CSV_FIELDS = ["mesh", "seed", "method", "V", "gt_coverage", "wall_time_s",
              "refit_time_s", "neof_rate_p"]


# ---------------------------------------------------------------------------
# NeOF plumbing
# ---------------------------------------------------------------------------

def patch_neof_intrinsics(fov_h, fov_v):
    """Point every loaded NeOF module at our tight camera.

    Their pinhole intrinsic is a module-level global star-imported into several
    modules, so each module holds its own binding — patch them all.
    """
    K = np.array([[319.5 / math.tan(fov_h / 2.0), 0, 319.5],
                  [0, 239.5 / math.tan(fov_v / 2.0), 239.5],
                  [0, 0, 1]])
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name.startswith("dataset") or
                               name.startswith("field")):
            continue
        if hasattr(mod, "intrinsic"):
            mod.intrinsic = K
    return K


def run_neof(pts_np, normals_np, audit, V, seed, epochs, iterations,
             scratch_dir):
    """Run NeOF's full hybrid optimization; score final poses with OUR audit."""
    from dataset.init_camera import initRandomCameras, getCameraPose
    from dataset.utils import npToTensor
    import optimization as neof_optimization
    from optimization import CameraLayerOpt
    # Their released optimization.py uses `copy` without importing it (it
    # crashes at epoch 0 as shipped, like main.py's args.scene bug). Inject it.
    import copy as _copy
    neof_optimization.copy = _copy

    pointnormals = np.concatenate([pts_np, normals_np], axis=1)
    # Their "voxels" are a coarser optimization target; a 0.05 voxel grid on
    # the unit wreck gives them the same ~1-2k working set their demos use.
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_np)
    pc.normals = o3d.utility.Vector3dVector(normals_np)
    down = pc.voxel_down_sample(voxel_size=0.05)
    voxelnormals = np.concatenate([np.asarray(down.points),
                                   np.asarray(down.normals)], axis=1)

    args = types.SimpleNamespace(
        cameranum=V, height=1.0, kcoverage=1, epoches=epochs,
        iterations=iterations, lr1=1e-3, lr2=1e-3, decay=1e-4,
        optimizer="Adam", scene=False, isscene=False)

    np.random.seed(seed)
    Rs, Cs = initRandomCameras(V, pointnormals, args.height)
    camerapose = npToTensor(getCameraPose(Rs, Cs), dtype=torch.float)

    class _NullWriter:
        def add_scalar(self, *a, **k):
            pass

    posepath = os.path.join(scratch_dir, f"neof_seed{seed}_V{V}")
    os.makedirs(posepath, exist_ok=True)
    opt = CameraLayerOpt(args, pointnormals, voxelnormals,
                         np.min(pts_np, axis=0), [1.0, np.zeros(3)],
                         posepath + os.sep, _NullWriter())

    t0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        position, rotation = opt.opt(camerapose)
    wall = time.perf_counter() - t0
    log = buf.getvalue()
    refit = sum(float(x) for x in re.findall(r"voxelmodel time: ([0-9.]+)", log))
    rate_p = [float(x) for x in re.findall(r"RateP: ([0-9.]+)", log)]

    covered = set()
    for i in range(len(position)):
        look = rotation[i][2]          # camera +z in world
        up = rotation[i][1]
        covered |= audit(np.asarray(position[i], dtype=np.float64),
                         np.concatenate([look, up]).astype(np.float64))
    return dict(gt_coverage=len(covered) / len(pts_np), wall_time_s=wall,
                refit_time_s=refit,
                neof_rate_p=rate_p[-1] if rate_p else float("nan"))


# ---------------------------------------------------------------------------
# Our arms
# ---------------------------------------------------------------------------

def run_greedy(pts_np, normals_np, audit, V, seed, n_cand=300):
    """Oracle greedy set-cover over normal-offset candidates (no routing —
    static placement has no path metric)."""
    t0 = time.perf_counter()
    cand_pos, cand_r6 = normal_offset_candidates(
        pts_np, normals_np, n_cand, 1.0, seed)
    cand_gt = [audit(cand_pos[j], cand_r6[j]) for j in range(len(cand_pos))]
    sel = greedy_select(cand_gt, V)
    covered = set()
    for j in sel:
        covered |= cand_gt[j]
    return (dict(gt_coverage=len(covered) / len(pts_np),
                 wall_time_s=time.perf_counter() - t0),
            cand_pos[sel], cand_r6[sel], cand_gt)


def run_refine(pts_np, normals_np, audit, init_pos, init_r6, V, seed,
               device, steps=300, lr=3e-2):
    """Our gradient stage: joint 6-DoF refinement of all V poses through the
    NVPS-backed differentiable visibility layer (soft set-cover objective)."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    b = make_backbone("nvps", device=device, gamma=-math.exp(-7.0), k=10)
    b.prepare(pts_np, normals_np)
    layer = GatedVisibilityLayer(
        b, fov_h=math.radians(30), fov_v=math.radians(35),
        near=0.1, far=1.5, frustum_sharpness=50.0, device=device)

    pts_t = torch.tensor(pts_np.T, dtype=torch.float32, device=device)
    pts_surface = torch.tensor(pts_np, dtype=torch.float32, device=device)
    pos = torch.nn.Parameter(torch.tensor(init_pos, dtype=torch.float32,
                                          device=device))
    rot = torch.nn.Parameter(torch.tensor(init_r6, dtype=torch.float32,
                                          device=device))
    optim = torch.optim.Adam([pos, rot], lr=lr)
    for _ in range(steps):
        optim.zero_grad()
        w = layer(pts_t, pos, rot).clamp(0.0, 1.0 - 1e-7)
        C = (1.0 - torch.prod(1.0 - w, dim=0)).mean()
        loss = -C + 1.0 * standoff_penalty(pos, pts_surface, 0.5, 1.5)
        loss.backward()
        optim.step()

    covered = set()
    p = pos.detach().cpu().numpy().astype(np.float64)
    r = rot.detach().cpu().numpy().astype(np.float64)
    for i in range(V):
        covered |= audit(p[i], r[i])
    return dict(gt_coverage=len(covered) / len(pts_np),
                wall_time_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=os.path.join(
        os.path.dirname(_DIR), "models", "duke_of_lancaster_uk_clipped.glb"))
    p.add_argument("--num_points", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--cameras", default="10,20")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--fov_h", type=float, default=30.0)
    p.add_argument("--fov_v", type=float, default=35.0)
    p.add_argument("--out", default=os.path.join(_DIR, "results", "neof"))
    p.add_argument("--device", default="auto")
    p.add_argument("--no_show", action="store_true")
    a = p.parse_args()
    a.cameras = [int(x) for x in a.cameras.split(",")]
    return a


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    patch_neof_intrinsics(math.radians(args.fov_h), math.radians(args.fov_v))

    cam = dict(fov_h=math.radians(args.fov_h), fov_v=math.radians(args.fov_v),
               near=0.1, far=1.5)
    mesh = load_mesh(args.mesh)
    mesh_name = os.path.basename(args.mesh)

    rows = []
    csv_path = os.path.join(args.out, "eval_neof.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for seed in range(args.seeds):
            np.random.seed(seed)
            pts_np, fidx = trimesh.sample.sample_surface(
                mesh, count=args.num_points)
            pts_np = np.asarray(pts_np, dtype=np.float64)
            normals_np = np.asarray(mesh.face_normals[fidx], dtype=np.float64)
            audit = make_audit_fn(mesh, pts_np, cam)

            for V in args.cameras:
                res = {}
                g, g_pos, g_r6, _ = run_greedy(pts_np, normals_np, audit,
                                               V, seed)
                res["greedy(oracle)"] = g
                res["greedy+refine(nvps)"] = run_refine(
                    pts_np, normals_np, audit, g_pos, g_r6, V, seed, device)
                res["neof"] = run_neof(pts_np, normals_np, audit, V, seed,
                                       args.epochs, args.iterations, args.out)

                for method, r in res.items():
                    row = dict(
                        mesh=mesh_name, seed=seed, method=method, V=V,
                        gt_coverage=f"{r['gt_coverage']:.4f}",
                        wall_time_s=f"{r['wall_time_s']:.1f}",
                        refit_time_s=f"{r.get('refit_time_s', float('nan')):.1f}",
                        neof_rate_p=f"{r.get('neof_rate_p', float('nan')):.4f}",
                    )
                    writer.writerow(row); fh.flush(); rows.append(row)
                    print(f"  [seed {seed}] V={V:<3d} {method:<20s} "
                          f"cov={r['gt_coverage']:.3f} "
                          f"wall={r['wall_time_s']:.1f}s "
                          f"refit={r.get('refit_time_s', float('nan')):.1f}s",
                          flush=True)

    print(f"\nCSV: {csv_path}")
    summarise(rows)


def summarise(rows):
    print("\n=== summary (mean ± std over seeds; coverage = frustum ∩ Embree "
          "ray-cast, identical for every arm) ===")
    by = {}
    for r in rows:
        by.setdefault((r["method"], int(r["V"])), []).append(
            (float(r["gt_coverage"]), float(r["wall_time_s"]),
             float(r["refit_time_s"])))
    print(f"{'method':<22} {'V':>3} {'coverage':>16} {'wall s':>14} {'refit s':>8}")
    for (m, V), v in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        a = np.array(v)
        refit = a[:, 2][~np.isnan(a[:, 2])]
        refit_s = f"{refit.mean():>8.1f}" if len(refit) else f"{'—':>8}"
        print(f"{m:<22} {V:>3} {a[:,0].mean():>7.3f} ± {a[:,0].std():<6.3f} "
              f"{a[:,1].mean():>7.1f} ± {a[:,1].std():<4.1f} {refit_s}")
    print("\nNotes: NeOF runs its released hybrid (field refit + gradient + elite\n"
          "resets) at its README defaults, patched to our tight camera and k=1.\n"
          "'refit' = time inside its per-epoch neural-field training alone.\n"
          "greedy+refine starts from greedy(oracle)'s poses, so the delta between\n"
          "those two rows is the value of the gradient stage.")


if __name__ == "__main__":
    main()
