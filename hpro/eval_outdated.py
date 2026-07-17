"""The outdated-mesh experiment — Stage B's go/no-go (RESEARCH_PLAN.md §7 step 5).

The one setting where the paper's warm-starting headline can be won (§3.4 item
3): on a static mesh the discrete baseline re-solves over *cached* candidate
ray-casts, so warm-started gradient replanning is only ~1.3x faster. Here the
world contradicts the prior model mid-mission, so every adaptation forces the
discrete pipeline to pay its ray-casting again, while the gradient planner keeps
stepping from its current solution.

Setup
-----
The robot is given a **prior mesh** M0 (the survey) but flies over the **true
mesh** M1: M0 with a region collapsed inward (the wreck degraded since the
survey). Its belief starts as a cloud sampled from M0. A simulated depth sensor
(hard ray-cast against M1) classifies every believed/true point in the frustum
each cycle:

* ``seen``      — nearest hit coincides with the point: surface is there;
  covers it (score counts true points only) and reveals unknown true points.
* ``refuted``   — clear line of sight *beyond* the point: the believed surface
  does not exist; the ghost is deleted from the belief.
* occluded      — hit nearer than the point: no information.

Seeing also reveals a **discovery halo**: unknown true points within
``discover_radius`` of a seen point become known-but-uncovered demand (a depth
image reveals a patch, not isolated samples). The same sensor drives every arm.

Arms
----
* ``open-loop``  — candidates + oracle greedy set-cover + 2-opt route planned
  once on the *prior* belief, executed blind on M1. Shows the miss.
* ``adaptive``   — same initial plan, but when the world contradicts the belief
  (refutations/discoveries since the last solve exceed a threshold) it re-plans:
  regenerates candidates from the current believed demand, **re-ray-casts them
  against M1** (an oracle-strength grant — the real pipeline would have to
  reconstruct a mesh first), re-runs greedy over remaining demand and re-routes.
  Pays full ray-cast price per adaptation; that is the point.
* ``rh-<backbone>`` — the receding-horizon planner of ``trajectory.py`` with
  belief updates: demand shifts smoothly, the horizon warm-starts across world
  changes, the backbone re-prepares on belief changes (~24 ms for NVPS).

Report per arm: true-surface coverage vs path length, planning wall-clock, and
adaptation latency (time from a world contradiction to a usable updated plan).

Usage::

    ipy hpro/eval_outdated.py --seeds 5 --backbones nvps --no_show
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
from scipy.spatial import cKDTree  # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from backbones import make_backbone                              # noqa: E402
from eval_trajectory import (load_mesh, normal_offset_candidates,  # noqa: E402
                             greedy_select, route_nn_2opt)
from frustum_gt import (build_camera_frame, compute_ground_truth_batched,  # noqa: E402
                        points_inside_frustum)
from trajectory import (TrajectoryConfig, GlobalGuide,            # noqa: E402
                        optimize_horizon, horizon_toward)
from visibility_layer import GatedVisibilityLayer                 # noqa: E402

CSV_FIELDS = [
    "mesh", "seed", "method", "backbone",
    "true_coverage", "changed_coverage", "path_length", "n_poses",
    "plan_time_s", "n_resolves", "adapt_latency_ms",
]


# ---------------------------------------------------------------------------
# World: prior belief vs true mesh
# ---------------------------------------------------------------------------

def collapse_region(mesh, seed_point, radius, depth):
    """Return a copy of ``mesh`` with the region around ``seed_point`` caved in.

    Vertices within ``radius`` move along -vertex_normal by up to ``depth``,
    with a smooth cosine falloff. Inward collapse is deliberate: a ghost point
    then floats in *empty space* in front of the new surface, so a clear line
    of sight refutes it. (An outward bulge hides ghosts *inside* the new
    surface — refuting those needs real depth-image differencing, out of scope.)
    """
    m = mesh.copy()
    d = np.linalg.norm(m.vertices - seed_point[None], axis=1)
    w = np.zeros_like(d)
    inside = d < radius
    w[inside] = 0.5 * (1.0 + np.cos(np.pi * d[inside] / radius))
    # Displace along the REGION's mean normal, not per-vertex normals: scanned
    # meshes have noisy/degenerate vertex normals (norms down to 0 here), and
    # per-vertex displacement shreds triangles — stretched faces then dominate
    # area-weighted sampling and 98 % of the "true" cloud lands on artefacts.
    # A coherent dent keeps the surface a surface.
    n_dir = m.vertex_normals[inside].mean(axis=0)
    n_dir = n_dir / (np.linalg.norm(n_dir) + 1e-12)
    m.vertices = m.vertices - n_dir[None] * (depth * w)[:, None]
    # Faces that actually moved appreciably (any vertex displaced > 0.03):
    # cheap and exact, unlike a proximity query against the prior mesh.
    moved = (depth * w) > 0.03
    changed_faces = moved[m.faces].any(axis=1)
    return m, changed_faces


class OutdatedWorld:
    """Union point cloud over (prior sample, true sample) + the sensor.

    Indexing is fixed over the union for the whole rollout; belief changes flip
    per-point flags instead of resizing arrays (resizing would break demand
    smoothness and every warm start).
    """

    def __init__(self, mesh_true, pts_prior, normals_prior, pts_true,
                 normals_true, changed_true_mask, cam,
                 discover_radius=0.15, tol=0.02):
        self.mesh_true = mesh_true
        self.pts = np.vstack([pts_prior, pts_true])
        self.normals = np.vstack([normals_prior, normals_true])
        self.N_prior = len(pts_prior)
        self.N_true = len(pts_true)
        self.is_true = np.zeros(len(self.pts), dtype=bool)
        self.is_true[self.N_prior:] = True
        # True points on the collapsed surface — the region the change created.
        self.changed_true = np.zeros(len(self.pts), dtype=bool)
        self.changed_true[self.N_prior:] = changed_true_mask
        self.cam = cam
        self.tol = tol
        self.discover_radius = discover_radius
        self._true_tree = cKDTree(pts_true)
        self._intersector = mesh_true.ray

        # Belief state: the robot starts believing the survey, knowing nothing
        # of the true sample.
        self.believed = ~self.is_true
        self.demand = np.where(self.believed, 1.0, 0.0)
        self.covered_true: set = set()
        self.refuted_total = 0
        self.contradictions_since_solve = 0

    # -- sensor ------------------------------------------------------------

    def observe(self, pos, r6):
        """Execute a pose: classify union points, update belief, return stats."""
        look, up, right = build_camera_frame(r6[:3], r6[3:])
        in_f = points_inside_frustum(
            self.pts, pos, look, up, right,
            self.cam["fov_h"], self.cam["fov_v"],
            self.cam["near"], self.cam["far"])
        cand = np.where(in_f & (self.believed | self.is_true))[0]
        seen = np.zeros(0, dtype=int)
        refuted = np.zeros(0, dtype=int)
        if len(cand):
            targets = self.pts[cand]
            dirs = targets - pos[None]
            d_p = np.linalg.norm(dirs, axis=1)
            locs, idx_ray, _ = self._intersector.intersects_location(
                ray_origins=np.repeat(pos[None], len(cand), axis=0),
                ray_directions=dirs, multiple_hits=True)
            d_hit_first = np.full(len(cand), np.inf)
            if len(idx_ray):
                d_hit = np.linalg.norm(locs - pos[None], axis=1)
                order = np.lexsort((d_hit, idx_ray))
                rays_s, d_s = idx_ray[order], d_hit[order]
                first = np.ones(len(rays_s), dtype=bool)
                first[1:] = rays_s[1:] != rays_s[:-1]
                d_hit_first[rays_s[first]] = d_s[first]
            seen_m = np.abs(d_hit_first - d_p) <= self.tol
            refuted_m = d_hit_first > d_p + self.tol      # incl. no-hit (inf)
            seen = cand[seen_m]
            refuted = cand[refuted_m & ~self.is_true[cand]]

        # -- belief update --------------------------------------------------
        self.believed[refuted] = False
        self.demand[refuted] = 0.0
        self.refuted_total += len(refuted)

        seen_true = seen[self.is_true[seen]]
        newly_covered = set(int(i) for i in seen_true) - self.covered_true
        self.covered_true |= set(int(i) for i in seen_true)
        self.believed[seen] = True
        self.demand[seen] = 0.0

        discovered = np.zeros(0, dtype=int)
        if len(seen_true):
            halo = self._true_tree.query_ball_point(
                self.pts[seen_true], r=self.discover_radius)
            halo_idx = np.unique(np.concatenate(
                [np.asarray(h, dtype=int) for h in halo])) + self.N_prior
            discovered = halo_idx[~self.believed[halo_idx]]
            self.believed[discovered] = True
            self.demand[discovered] = 1.0

        self.contradictions_since_solve += len(refuted) + len(discovered)
        return dict(seen=seen, seen_true=seen_true, refuted=refuted,
                    discovered=discovered, n_new_covered=len(newly_covered))

    # -- scoring -----------------------------------------------------------

    def true_coverage(self):
        return len(self.covered_true) / self.N_true

    def changed_coverage(self):
        idx = np.where(self.changed_true)[0]
        if len(idx) == 0:
            return float("nan")
        return len(self.covered_true & set(int(i) for i in idx)) / len(idx)


# ---------------------------------------------------------------------------
# Discrete arms
# ---------------------------------------------------------------------------

def plan_discrete(mesh, pts, normals, demand_mask, cam, cfg, n_cand, budget,
                  start, seed, intersector=None):
    """Candidates from ``pts[demand_mask]`` -> ray-cast vs ``mesh`` -> greedy
    over demanded points -> route. Returns (poses, r6s, wall_time)."""
    t0 = time.perf_counter()
    src = np.where(demand_mask)[0]
    if len(src) == 0:
        return np.zeros((0, 3)), np.zeros((0, 6)), time.perf_counter() - t0
    cand_pos, cand_r6 = normal_offset_candidates(
        pts[src], normals[src], n_cand, 0.5 * (cfg.near_dist + cfg.far_dist),
        seed)
    intersector = intersector or mesh.ray
    cand_gt = []
    demanded = set(int(i) for i in src)
    for j in range(len(cand_pos)):
        look, up, right = build_camera_frame(cand_r6[j, :3], cand_r6[j, 3:])
        idx, _ = compute_ground_truth_batched(
            mesh, cand_pos[j], pts, look, up, right,
            cam["fov_h"], cam["fov_v"], cam["near"], cam["far"],
            intersector=intersector)
        cand_gt.append(set(int(i) for i in idx) & demanded)
    sel = greedy_select(cand_gt, budget)
    order, _ = route_nn_2opt(start, cand_pos[sel])
    sel_ordered = [sel[i] for i in order]
    return (cand_pos[sel_ordered], cand_r6[sel_ordered],
            time.perf_counter() - t0)


def run_discrete_arm(world, mesh_prior, cam, cfg, args, seed, start,
                     adaptive):
    """Open-loop or adaptive discrete pipeline under the outdated world."""
    # Initial plan on the PRIOR belief: candidates from the prior cloud,
    # visibility ray-cast against the prior mesh (all the robot has).
    prior_mask = ~world.is_true
    poses, r6s, t_plan = plan_discrete(
        mesh_prior, world.pts, world.normals, prior_mask, cam, cfg,
        args.n_candidates, args.budget, start, seed)
    plan_time = t_plan
    latencies = []
    n_resolves = 0

    exec_pos = []
    cur = start.copy()
    total_len = 0.0
    curve = []
    i = 0
    while i < len(poses) and len(exec_pos) < args.max_poses:
        p, r = poses[i], r6s[i]
        total_len += float(np.linalg.norm(p - cur))
        cur = p
        exec_pos.append(p)
        world.observe(p, r)
        curve.append((total_len, world.true_coverage()))
        i += 1

        if (adaptive and world.contradictions_since_solve >= args.resolve_after
                and len(exec_pos) < args.max_poses):
            # The world contradicted the belief: the discrete pipeline must
            # re-plan, and that means re-ray-casting candidates. Grant it the
            # TRUE mesh for the re-cast (oracle-strength; a real pipeline
            # would first have to reconstruct a surface from the sensor data).
            t0 = time.perf_counter()
            remaining = args.max_poses - len(exec_pos)
            poses, r6s, t_solve = plan_discrete(
                world.mesh_true, world.pts, world.normals,
                world.demand > 0.5, cam, cfg, args.n_candidates,
                min(args.budget, remaining), cur, seed,
                intersector=world._intersector)
            latencies.append((time.perf_counter() - t0) * 1e3)
            plan_time += t_solve
            world.contradictions_since_solve = 0
            n_resolves += 1
            i = 0
        if world.true_coverage() >= cfg.target_coverage:
            break

    return dict(curve=curve, path_length=total_len, n_poses=len(exec_pos),
                plan_time_s=plan_time, n_resolves=n_resolves,
                adapt_latency_ms=float(np.mean(latencies)) if latencies
                else float("nan"))


# ---------------------------------------------------------------------------
# Continuous arm: receding horizon over the shifting belief
# ---------------------------------------------------------------------------

def run_rh_arm(world, backbone_name, cam, cfg, args, start, device):
    """Receding-horizon planner over the believed cloud, belief-aware.

    Mirrors ``trajectory.receding_horizon_plan`` but re-slices the believed
    subset (and re-prepares the backbone) whenever the sensor changes the
    belief. Demand stays smooth in union indexing, so the warm start survives
    every world change — which is the property being measured.
    """
    b = make_backbone(backbone_name, device=device,
                      gamma=-math.exp(-7.0), k=10)

    def prepare():
        bidx = np.where(world.believed)[0]
        if b.requires_normals:
            b.prepare(world.pts[bidx], world.normals[bidx])
        else:
            b.prepare(world.pts[bidx])
        layer = GatedVisibilityLayer(
            b, fov_h=cam["fov_h"], fov_v=cam["fov_v"],
            near=cam["near"], far=cam["far"],
            frustum_sharpness=50.0, device=device)
        pts_t = torch.tensor(world.pts[bidx].T, dtype=torch.float32,
                             device=device)
        pts_surface = torch.tensor(world.pts[bidx], dtype=torch.float32,
                                   device=device)
        return bidx, layer, pts_t, pts_surface

    t_all = time.perf_counter()
    bidx, layer, pts_t, pts_surface = prepare()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    guide = GlobalGuide(world.pts, cfg, rng)

    start_t = torch.tensor(start, dtype=torch.float32, device=device)
    centroid = torch.tensor(world.pts[world.believed].mean(axis=0),
                            dtype=torch.float32, device=device)
    H = cfg.horizon
    init_pos = torch.stack([
        start_t + torch.tensor(rng.normal(scale=0.3, size=3),
                               dtype=torch.float32, device=device)
        for _ in range(H)])
    look = centroid.unsqueeze(0) - init_pos
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32,
                      device=device).expand(H, 3)
    init_rot = torch.cat([look, up], dim=1)

    total_len = 0.0
    curve = []
    replan_times = []
    adapt_latencies = []
    prev_retargets = 0
    world_changed = False
    pending_prepare_ms = 0.0

    for cycle in range(args.max_poses):
        demand_t = torch.tensor(world.demand[bidx], dtype=torch.float32,
                                device=device)
        # Slow timescale over the union demand (masking handles belief).
        demand_terminal = None
        target = guide.select(world.demand, start_t.cpu().numpy(), cycle)
        retargeted = guide.retargets != prev_retargets
        prev_retargets = guide.retargets
        if target is not None:
            mask = np.zeros(len(world.pts), dtype=np.float32)
            mask[target] = 1.0
            demand_terminal = demand_t * torch.tensor(
                mask[bidx], device=device)
        if retargeted and cycle > 0 and target is not None:
            init_pos, init_rot = horizon_toward(
                start_t, world.pts[target], cfg, device)

        if cycle == 0:
            n_steps = cfg.init_steps
        elif retargeted:
            n_steps = cfg.retarget_steps
        else:
            n_steps = cfg.replan_steps
        t0 = time.perf_counter()
        pos, rot, _ = optimize_horizon(
            layer, pts_t, pts_surface, demand_t, start_t, init_pos, init_rot,
            cfg, n_steps, demand_terminal=demand_terminal)
        dt = time.perf_counter() - t0
        replan_times.append(dt)
        if world_changed:
            # Latency from "world contradicted the belief" to "usable updated
            # plan": backbone re-prepare + one warm replan.
            adapt_latencies.append(dt * 1e3 + pending_prepare_ms)
            world_changed = False
            pending_prepare_ms = 0.0

        p0 = pos[0].cpu().numpy()
        r0 = rot[0].cpu().numpy()
        total_len += float(np.linalg.norm(p0 - start_t.cpu().numpy()))
        obs = world.observe(p0.astype(np.float64), r0.astype(np.float64))
        curve.append((total_len, world.true_coverage()))
        guide.report_harvest(obs["n_new_covered"])

        # Belief changed -> re-prepare the backbone and re-slice, in-place of
        # the union indexing so demand and the warm start survive untouched.
        if len(obs["refuted"]) or len(obs["discovered"]):
            t0 = time.perf_counter()
            bidx, layer, pts_t, pts_surface = prepare()
            pending_prepare_ms = (time.perf_counter() - t0) * 1e3
            world_changed = True
        start_t = pos[0]
        init_pos = torch.cat([pos[1:], pos[-1:].clone()], dim=0)
        init_rot = torch.cat([rot[1:], rot[-1:].clone()], dim=0)

        if world.true_coverage() >= cfg.target_coverage:
            break

    return dict(curve=curve, path_length=total_len, n_poses=len(curve),
                plan_time_s=time.perf_counter() - t_all,
                n_resolves=len(adapt_latencies),
                adapt_latency_ms=float(np.mean(adapt_latencies))
                if adapt_latencies else float("nan"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=os.path.join(
        os.path.dirname(_DIR), "models", "duke_of_lancaster_uk_clipped.glb"))
    p.add_argument("--num_points", type=int, default=3000)
    p.add_argument("--backbones", default="nvps")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--budget", type=int, default=20,
                   help="Viewpoint budget per discrete solve.")
    p.add_argument("--max_poses", type=int, default=40)
    p.add_argument("--n_candidates", type=int, default=300)
    p.add_argument("--resolve_after", type=int, default=15,
                   help="Adaptive arm re-solves after this many contradictions.")
    p.add_argument("--collapse_radius", type=float, default=0.25,
                   help="Sized to the wreck: hull cross-section is ~0.3.")
    p.add_argument("--collapse_depth", type=float, default=0.15)
    p.add_argument("--discover_radius", type=float, default=0.15)
    p.add_argument("--fov_h", type=float, default=30.0)
    p.add_argument("--fov_v", type=float, default=35.0)
    p.add_argument("--near", type=float, default=0.1)
    p.add_argument("--far", type=float, default=1.5)
    p.add_argument("--near_dist", type=float, default=0.5)
    p.add_argument("--far_dist", type=float, default=1.5)
    p.add_argument("--out", default=os.path.join(_DIR, "results", "outdated"))
    p.add_argument("--device", default="auto")
    p.add_argument("--no_show", action="store_true")
    a = p.parse_args()
    a.backbones = [x.strip() for x in a.backbones.split(",")]
    return a


def build_world(mesh_prior, args, seed):
    """Sample prior + true clouds and the collapsed true mesh, seeded."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    pts_prior, fidx_p = trimesh.sample.sample_surface(
        mesh_prior, count=args.num_points)
    pts_prior = np.asarray(pts_prior, dtype=np.float64)
    normals_prior = np.asarray(mesh_prior.face_normals[fidx_p],
                               dtype=np.float64)

    anchor = pts_prior[rng.integers(len(pts_prior))]
    mesh_true, changed_faces = collapse_region(
        mesh_prior, anchor, args.collapse_radius, args.collapse_depth)
    pts_true, fidx_t = trimesh.sample.sample_surface(
        mesh_true, count=args.num_points)
    pts_true = np.asarray(pts_true, dtype=np.float64)
    normals_true = np.asarray(mesh_true.face_normals[fidx_t],
                              dtype=np.float64)
    changed_true = changed_faces[np.asarray(fidx_t)]

    cam = dict(fov_h=math.radians(args.fov_h), fov_v=math.radians(args.fov_v),
               near=args.near, far=args.far)
    world = OutdatedWorld(mesh_true, pts_prior, normals_prior, pts_true,
                          normals_true, changed_true, cam,
                          discover_radius=args.discover_radius)
    d0 = rng.normal(size=3)
    d0 /= np.linalg.norm(d0)
    start = pts_prior.mean(axis=0) + d0 * (1.0 + args.far_dist)
    return world, cam, start, float(changed_true.mean())


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    mesh_prior = load_mesh(args.mesh)
    mesh_name = os.path.basename(args.mesh)

    rows, curves = [], {}
    csv_path = os.path.join(args.out, "eval_outdated.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for seed in range(args.seeds):
            cfg = TrajectoryConfig(
                max_cycles=args.max_poses, near_dist=args.near_dist,
                far_dist=args.far_dist, seed=seed)

            arms = [("open-loop", "oracle",
                     lambda w: run_discrete_arm(
                         w, mesh_prior, w.cam, cfg, args, seed,
                         w.pts[:w.N_prior].mean(axis=0), adaptive=False)),
                    ("adaptive", "oracle",
                     lambda w: run_discrete_arm(
                         w, mesh_prior, w.cam, cfg, args, seed,
                         w.pts[:w.N_prior].mean(axis=0), adaptive=True))]
            for bname in args.backbones:
                arms.append((f"rh", bname,
                             lambda w, bn=bname: run_rh_arm(
                                 w, bn, w.cam, cfg, args,
                                 w.pts[:w.N_prior].mean(axis=0), device)))

            for method, backbone, fn in arms:
                world, cam, start, changed_frac = build_world(
                    mesh_prior, args, seed)
                r = fn(world)
                row = dict(
                    mesh=mesh_name, seed=seed, method=method,
                    backbone=backbone,
                    true_coverage=f"{world.true_coverage():.4f}",
                    changed_coverage=f"{world.changed_coverage():.4f}",
                    path_length=f"{r['path_length']:.3f}",
                    n_poses=r["n_poses"],
                    plan_time_s=f"{r['plan_time_s']:.2f}",
                    n_resolves=r["n_resolves"],
                    adapt_latency_ms=f"{r['adapt_latency_ms']:.0f}",
                )
                writer.writerow(row); fh.flush(); rows.append(row)
                curves[(f"{method}-{backbone}", seed)] = r["curve"]
                print(f"  [seed {seed}] {method:<10s} {backbone:<7s} "
                      f"cov={world.true_coverage():.3f} "
                      f"changed={world.changed_coverage():.3f} "
                      f"len={r['path_length']:.2f}m "
                      f"plan={r['plan_time_s']:.1f}s "
                      f"resolves={r['n_resolves']} "
                      f"lat={r['adapt_latency_ms']:.0f}ms", flush=True)

    print(f"\nCSV: {csv_path}")
    plot_curves(curves, args.out, not args.no_show)
    summarise(rows)


def plot_curves(curves, out_dir, show):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"open-loop-oracle": "grey", "adaptive-oracle": "tomato",
              "rh-nvps": "seagreen", "rh-hpro": "royalblue"}
    seen = set()
    for (method, seed), pts in sorted(curves.items()):
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = colors.get(method, "purple")
        ax.plot(xs, ys, marker="o", ms=2.5, color=c, alpha=0.7,
                label=method if method not in seen else None)
        seen.add(method)
    ax.set_xlabel("path length (m)")
    ax.set_ylabel("true-surface coverage")
    ax.set_title("Outdated mesh: coverage of the TRUE surface vs metres")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "outdated_coverage.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def summarise(rows):
    print("\n=== summary (mean ± std over seeds) ===")
    by = {}
    for r in rows:
        key = (r["method"], r["backbone"])
        by.setdefault(key, []).append((
            float(r["true_coverage"]), float(r["changed_coverage"]),
            float(r["path_length"]), float(r["plan_time_s"]),
            float(r["n_resolves"]),
            float(r["adapt_latency_ms"])))
    print(f"{'method':<11} {'backbone':<8} {'coverage':>15} {'changed-region':>15} "
          f"{'length':>13} {'plan s':>12} {'resolves':>8} {'latency ms':>10}")
    for (m, b), v in sorted(by.items()):
        a = np.array(v)
        lat = a[:, 5][~np.isnan(a[:, 5])]
        lat_s = f"{lat.mean():>10.0f}" if len(lat) else f"{'—':>10}"
        print(f"{m:<11} {b:<8} {a[:,0].mean():>6.3f} ± {a[:,0].std():<6.3f} "
              f"{a[:,1].mean():>6.3f} ± {a[:,1].std():<6.3f} "
              f"{a[:,2].mean():>5.2f} ± {a[:,2].std():<5.2f} "
              f"{a[:,3].mean():>5.1f} ± {a[:,3].std():<4.1f} "
              f"{a[:,4].mean():>8.1f} {lat_s}")
    print("\nEvery arm shares the same sensor, discovery halo and pose budget.\n"
          "'latency' = wall-clock from a world contradiction to a usable plan:\n"
          "full candidate re-ray-cast + greedy + route for the discrete arm,\n"
          "one warm replan cycle for the gradient arm. ≥5 seeds or it's noise.")


if __name__ == "__main__":
    main()
