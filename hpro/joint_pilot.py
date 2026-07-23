"""Headroom pilot for the §9 joint-OCP direction (RESEARCH_PLAN.md §9.3).

Loads a saved run of the thesis pipeline (`scripts/run_full_pipeline.py`,
persisted via `save_pipeline`) and asks the one question this direction hangs
on: **can joint gradient optimization of the multi-robot trajectories move the
pipeline's own numbers** (coverage, makespan, total length) — warm-started from
the pipeline's solution AND cold-started from nothing but geometry?

The optimization is one differentiable objective over all robots' waypoint
chains (positions + 6D orientations):

    L = − soft-coverage(NVPS ∘ frustum, all poses)
        + λ_route · (α · softmax-makespan + (1−α) · total length)
        + AL_collision(ESDF along segment samples;  margin = ROBOT_RADIUS + ε)
        + AL_separation(pairwise robot distance on a common time grid)

Collision and separation are augmented-Lagrangian constraints (multiplier
updates every `al_every` steps), NOT plain penalties — violations are driven to
zero rather than traded against coverage, per the §9.2 requirement. There is no
post-hoc repair anywhere: the pipeline's machinery is used only to SCORE
results (its own OptiX ray-cast oracle) and the ESDF only to VERIFY clearance.

Fairness rules baked in: pose budget equals the pipeline's selected viewpoint
count; final coverage is judged by the pipeline's own
`RaycastingVisibilityQueryCuda` (hard ray-cast, their frustum convention);
baseline path lengths are measured on the pipeline's *executed* trajectories.
Sanity gate (§9.4): before optimizing anything, the harness re-scores the
pipeline's own poses through the same scorer and must reproduce the coverage
recorded in the manifest — if conventions are wrong, it stops there.

Run (needs the `inspection` env: cupy, open3d, triro, torch, ocnn)::

    ipy_inspection hpro/joint_pilot.py --pipeline_dir outputs/pilot_baseline
"""

import argparse
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402
import torch.nn.functional as F   # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
for p in (_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import ocnn_compat  # noqa: F401,E402  (must precede ocnn import in backbones)
from backbones import make_backbone                    # noqa: E402
from frustum import six_d_to_rotation_matrix           # noqa: E402
from trajectory import _kmeans, _tour_nn_2opt          # noqa: E402
from visibility_layer import GatedVisibilityLayer      # noqa: E402


# ---------------------------------------------------------------------------
# ESDF: differentiable clearance from the pipeline's own occupancy grid
# ---------------------------------------------------------------------------

class Esdf:
    """Euclidean signed-ish distance field: metres to the nearest occupied
    voxel, trilinearly interpolated (differentiable w.r.t. query position).

    Built from an UNinflated, interior-filled occupancy grid, so "distance"
    means distance to the structure itself and the collision constraint is
    `esdf(x) >= ROBOT_RADIUS + margin` — semantics a reader can check, instead
    of an inflated grid with a zero margin. Outside the grid the border value
    is used (the grid has `padding` metres of free space around the mesh, so
    the border is far from the structure and this is conservative-enough for a
    pilot; the AL constraint keeps trajectories interior to the grid anyway).
    """

    def __init__(self, og, device):
        from scipy.ndimage import distance_transform_edt
        import cupy as cp
        occ = cp.asnumpy(og.grid)                       # True = occupied
        # SIGNED field: positive clearance outside, NEGATIVE depth inside.
        # An unsigned field is 0 throughout occupied space, so a segment that
        # ends up inside the hull gets zero gradient and the AL can never push
        # it out (observed: colViol pinned at exactly the margin for 2000
        # steps). The signed form restores the escape gradient everywhere.
        dist_vox = distance_transform_edt(~occ) - distance_transform_edt(occ)
        self.d = torch.tensor(dist_vox * og.resolution, dtype=torch.float32,
                              device=device)
        self.origin = torch.tensor(cp.asnumpy(og.origin), dtype=torch.float32,
                                   device=device)
        self.res = float(og.resolution)
        self.shape = torch.tensor(self.d.shape, device=device)

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        """pos (..., 3) world metres -> (...) clearance in metres."""
        # Voxel-center coordinates: value d[i,j,k] lives at origin+(ijk+0.5)res.
        q = (pos - self.origin) / self.res - 0.5
        q0 = q.detach().floor().long()
        q0 = torch.stack([q0[..., i].clamp(0, int(self.shape[i]) - 2)
                          for i in range(3)], dim=-1)
        f = (q - q0.to(q.dtype)).clamp(0.0, 1.0)        # (..., 3) fractional
        d = self.d
        i, j, k = q0[..., 0], q0[..., 1], q0[..., 2]
        c000 = d[i, j, k];         c100 = d[i + 1, j, k]
        c010 = d[i, j + 1, k];     c110 = d[i + 1, j + 1, k]
        c001 = d[i, j, k + 1];     c101 = d[i + 1, j, k + 1]
        c011 = d[i, j + 1, k + 1]; c111 = d[i + 1, j + 1, k + 1]
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz


def project_clear(esdf, pos_list, margin, iters=2):
    """Projected-gradient step for the structure-collision constraint.

    After each optimizer step, waypoints violating ``esdf(x) >= margin`` are
    moved along the ESDF gradient until clear. This is PGD — constraint
    enforcement inside the optimization, not post-hoc repair of a finished
    plan — and it exists because Adam normalizes per-parameter gradient
    magnitudes, so no penalty weight can reliably out-shove a competing term
    (measured: poses driven 1.4 m into the hull while the collision AL's mu
    grew 100-fold). Segment interiors stay AL-penalized; the projection
    handles the waypoints the segments hang from.
    """
    with torch.enable_grad():
        for p in pos_list:
            for _ in range(iters):
                x = p.detach().clone().requires_grad_(True)
                d = esdf(x)
                need = (margin - d).clamp_min(0.0)
                if float(need.max()) <= 0.0:
                    break
                g = torch.autograd.grad(d.sum(), x)[0]
                g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)
                with torch.no_grad():
                    p.add_(g * need.unsqueeze(1) * 1.1)
    return pos_list


class AugmentedLagrangian:
    """Inequality AL for g(x) <= 0:  psi = (relu(lam + mu g)^2 - lam^2) / (2 mu).

    `step()` is called with the current constraint values every `al_every`
    optimizer iterations: lam <- relu(lam + mu g), mu grows geometrically.
    """

    def __init__(self, n, device, mu0=10.0, mu_growth=1.3, mu_max=500.0):
        self.lam = torch.zeros(n, device=device)
        self.mu = mu0
        self.mu_growth = mu_growth
        self.mu_max = mu_max

    def penalty(self, g: torch.Tensor) -> torch.Tensor:
        return ((F.relu(self.lam + self.mu * g) ** 2 - self.lam ** 2)
                / (2.0 * self.mu)).sum()

    def step(self, g: torch.Tensor) -> None:
        with torch.no_grad():
            self.lam = F.relu(self.lam + self.mu * g)
            self.mu = min(self.mu * self.mu_growth, self.mu_max)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotmats_to_6d(rotmats: np.ndarray) -> np.ndarray:
    """Pipeline rotmats (V,3,3), columns (forward,right,up) -> our 6D (look,up)."""
    return np.concatenate([rotmats[:, :, 0], rotmats[:, :, 2]], axis=1)


def sixd_to_rotmats(rot6d: torch.Tensor) -> np.ndarray:
    """Our 6D -> pipeline rotmat columns (forward, right, up)."""
    look, up, right = six_d_to_rotation_matrix(rot6d)
    return torch.stack([look, right, up], dim=2).detach().cpu().numpy()


def chain(home: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """home -> waypoints -> home, as (T+2, 3)."""
    return torch.cat([home.unsqueeze(0), pos, home.unsqueeze(0)], dim=0)


def chain_lengths(ch: torch.Tensor) -> torch.Tensor:
    return (ch[1:] - ch[:-1]).norm(dim=1)


def segment_samples(ch: torch.Tensor, per_seg: int) -> torch.Tensor:
    """(S,3) points along every segment (excluding each segment's start)."""
    t = torch.linspace(0.0, 1.0, per_seg + 1, device=ch.device)[1:]
    a, b = ch[:-1], ch[1:]                               # (M, 3)
    return (a.unsqueeze(1) + (b - a).unsqueeze(1) * t.view(1, -1, 1)).reshape(-1, 3)


def positions_at_times(ch: torch.Tensor, times: torch.Tensor,
                       speed: float) -> torch.Tensor:
    """Robot position at global times, flying `ch` at constant `speed`.

    Differentiable w.r.t. the chain positions (segment index selection is
    detached, as in any piecewise-linear interpolation). Past the end of the
    path the robot parks at the final position.
    """
    seg = chain_lengths(ch)                              # (M,)
    cum = torch.cat([seg.new_zeros(1), torch.cumsum(seg, 0)])   # (M+1,)
    s = (times * speed).clamp(max=float(cum[-1].detach()))
    idx = torch.searchsorted(cum.detach(), s.detach(), right=True) - 1
    idx = idx.clamp(0, len(seg) - 1)
    frac = ((s - cum[idx]) / seg[idx].clamp_min(1e-9)).clamp(0.0, 1.0)
    return ch[idx] + (ch[idx + 1] - ch[idx]) * frac.unsqueeze(1)


# ---------------------------------------------------------------------------
# The joint solve
# ---------------------------------------------------------------------------

def optimize_joint(pts_t, layer, homes, init_pos, init_rot, esdf, args,
                   label="", keep_pairs=None):
    """Jointly optimize all robots' chains. Returns (pos list, rot list, log).

    init_pos/init_rot: lists of (T_r, 3)/(T_r, 6) tensors per robot.

    Inner solver (4th design, and the first sound one — see §9.5 of the plan
    for the post-mortem of the previous three):

    * **SGD + momentum with a hard per-waypoint displacement cap**, NOT Adam.
      Adam's per-parameter normalization erases force ratios, so no penalty
      weight can dominate another — measured: a weak route pull tore a
      feasible warm start apart against constraints whose multipliers had
      grown 100-fold. With SGD the AL force hierarchy is real, and the cap
      bounds every waypoint's motion per step (metres), so nothing explodes.
    * **Per-pose keep-constraints** (warm arm): ``keep_pairs = (pose, point)``
      index tensors — pose i must keep w[i, j] >= keep_target for the points
      j it was verified to see (pipeline ray-cast ∩ surrogate agreement at
      init). Anchoring responsibility per pose is much better conditioned
      than a union constraint (which lets poses negotiate handoffs they then
      fail to execute) and makes the measured headroom a LOWER bound.
    * Structure collision: projected (waypoints, ``project_clear``) + AL on
      segment samples. Separation: AL on a common time grid.
    """
    device = pts_t.device
    R = len(init_pos)
    pos = [torch.nn.Parameter(p.clone()) for p in init_pos]
    rot = [torch.nn.Parameter(r.clone()) for r in init_rot]
    opt = torch.optim.SGD([{"params": pos}, {"params": rot}],
                          lr=1.0, momentum=0.9)
    margin = args.robot_radius + args.clearance_margin

    n_col = sum((len(p) + 1) * args.col_samples for p in init_pos)
    al_col = AugmentedLagrangian(n_col, device, mu0=20.0, mu_max=4000.0)
    with torch.no_grad():
        t_max = 1.3 * max(float(chain_lengths(chain(homes[r], init_pos[r])).sum())
                          for r in range(R)) / args.speed
    times = torch.linspace(0.0, t_max, args.sep_samples, device=device)
    n_sep = (R * (R - 1)) // 2 * args.sep_samples
    al_sep = AugmentedLagrangian(n_sep, device)
    al_keep = (AugmentedLagrangian(len(keep_pairs), device, mu0=10.0,
                                   mu_max=400.0)
               if keep_pairs is not None else None)

    log = []
    t0 = time.perf_counter()
    for it in range(args.steps):
        opt.zero_grad()
        all_pos = torch.cat(pos, dim=0)
        all_rot = torch.cat(rot, dim=0)
        w = layer(pts_t, all_pos, all_rot).clamp(0.0, 1.0 - 1e-7)
        c_pt = 1.0 - torch.prod(1.0 - w, dim=0)          # (N,) per-point
        cover = c_pt.mean()

        chains = [chain(homes[r], pos[r]) for r in range(R)]
        lengths = torch.stack([chain_lengths(c).sum() for c in chains])
        makespan = torch.logsumexp(lengths * args.beta, dim=0) / args.beta
        route = args.alpha * makespan + (1 - args.alpha) * lengths.sum()
        # Cold start must FIND coverage before it economizes on route: ramp
        # the route weight in over the run (warm keeps it flat).
        lam_route = args.lambda_route * (
            1.0 if keep_pairs is not None
            else (0.1 + 0.9 * min(1.0, 2.0 * it / max(args.steps, 1))))

        g_col = margin - esdf(torch.cat(
            [segment_samples(c, args.col_samples) for c in chains]))
        g_sep_list = []
        for a in range(R):
            for b in range(a + 1, R):
                pa = positions_at_times(chains[a], times, args.speed)
                pb = positions_at_times(chains[b], times, args.speed)
                g_sep_list.append(args.d_separation - (pa - pb).norm(dim=1))
        g_sep = torch.cat(g_sep_list) if g_sep_list else torch.zeros(0, device=device)

        if keep_pairs is None:
            loss = (-args.w_cover * cover + lam_route * route
                    + al_col.penalty(g_col) + al_sep.penalty(g_sep))
            g_keep = None
        else:
            # Keep-constraints: per-POINT, in METRIC space, via the max over
            # poses of the min over margins (§9.5 for the two failed
            # predecessors: sigmoid constraints ratchet — the restoring
            # gradient saturates away once violated; per-POSE anchors make
            # each pose's hundreds of keep-gradients cancel while forbidding
            # the coverage handoffs shortening needs). Point j stays covered
            # iff its BEST pose has slack: handoffs stay legal (the argmax
            # switches pose), gradients are unit-scale everywhere, and only
            # one pose per point feels the constraint at a time.
            m_occ = layer.backbone.raw_margin(pts_t, all_pos)     # (V, N)
            look, up, right = six_d_to_rotation_matrix(all_rot)
            dvec = pts_t.T.unsqueeze(0) - all_pos.unsqueeze(1)    # (V, N, 3)
            depth = (dvec * look.unsqueeze(1)).sum(-1)
            m_h = (math.tan(layer.fov_h / 2.0) * depth
                   - (dvec * right.unsqueeze(1)).sum(-1).abs())
            m_v = (math.tan(layer.fov_v / 2.0) * depth
                   - (dvec * up.unsqueeze(1)).sum(-1).abs())
            M = torch.stack([m_occ, depth - layer.near,
                             layer.far - depth, m_h, m_v]).min(dim=0).values
            best = M[:, keep_pairs].max(dim=0).values             # (K,)
            g_keep = args.keep_margin - best
            loss = (lam_route * route - 0.01 * cover
                    + al_keep.penalty(g_keep)
                    + al_col.penalty(g_col) + al_sep.penalty(g_sep))
        loss.backward()

        pos_before = [p.detach().clone() for p in pos]
        rot_before = [r.detach().clone() for r in rot]
        opt.step()
        with torch.no_grad():
            # Displacement caps: whatever the forces, no waypoint moves more
            # than cap_pos metres (cap_rot in 6D units) per iteration.
            for p, p0 in zip(pos, pos_before):
                d = p - p0
                n = d.norm(dim=1, keepdim=True)
                p.copy_(p0 + d * (args.cap_pos / n.clamp_min(args.cap_pos)))
            for r6, r0 in zip(rot, rot_before):
                d = r6 - r0
                n = d.norm(dim=1, keepdim=True)
                r6.copy_(r0 + d * (args.cap_rot / n.clamp_min(args.cap_rot)))
        project_clear(esdf, pos, margin)

        if (it + 1) % args.al_every == 0:
            al_col.step(g_col.detach())
            al_sep.step(g_sep.detach())
            if al_keep is not None:
                al_keep.step(g_keep.detach())
        if (it + 1) % 50 == 0 or it == 0:
            log.append(dict(
                it=it + 1, soft_cover=float(cover),
                makespan=float(lengths.max()), total=float(lengths.sum()),
                max_col_violation=float(F.relu(g_col).max()),
                max_sep_violation=float(F.relu(g_sep).max()) if len(g_sep) else 0.0,
            ))
            keep_s = ("" if g_keep is None else
                      f" keepViol={int((g_keep > 0).sum())}/{len(g_keep)}")
            print(f"    [{label} it {it+1:4d}] softC={float(cover):.3f} "
                  f"makespan={float(lengths.max()):.1f} m "
                  f"colViol={float(F.relu(g_col).max()):.3f} m "
                  f"sepViol={(float(F.relu(g_sep).max()) if len(g_sep) else 0):.3f} m"
                  f"{keep_s}", flush=True)
    wall = time.perf_counter() - t0
    return ([p.detach() for p in pos], [r.detach() for r in rot],
            dict(wall_s=wall, trace=log))


def cold_start(pts_np, normals_np, homes_np, n_poses, R, standoff, rng):
    """Geometric init: kmeans view targets -> normal-offset poses -> R chains.

    Uses no ray-casts and no set cover — only the cloud and its normals. One
    global NN+2-opt tour over all poses is cut into R contiguous chunks of
    near-equal path length; each chunk goes to the nearest free home. Simple,
    deterministic, and dumb on purpose: everything smarter is the optimizer's
    job.
    """
    labels = _kmeans(pts_np, n_poses, rng)
    cents, looks = [], []
    for u in np.unique(labels):
        m = labels == u
        c = pts_np[m].mean(axis=0)
        n = normals_np[m].mean(axis=0)
        n /= (np.linalg.norm(n) + 1e-12)
        cents.append(c + n * standoff)
        looks.append(-n)
    cents, looks = np.array(cents), np.array(looks)

    tour = _tour_nn_2opt(pts_np.mean(axis=0), cents)
    cents, looks = cents[tour], looks[tour]
    seg = np.linalg.norm(np.diff(cents, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    cuts = [int(np.searchsorted(cum, total * r / R)) for r in range(1, R)]
    chunks = np.split(np.arange(len(cents)), cuts)

    init_pos, init_rot = [None] * R, [None] * R
    up_w = np.array([0.0, 0.0, 1.0])
    free = set(range(R))
    for ch_idx in chunks:
        if len(ch_idx) == 0:
            continue
        c0 = cents[ch_idx].mean(axis=0)
        r = min(free, key=lambda r_: np.linalg.norm(c0 - homes_np[r_])) \
            if free else int(rng.integers(R))
        free.discard(r)
        p, lk = cents[ch_idx], looks[ch_idx]
        # Orient the chunk so it starts at the end nearer the home.
        if np.linalg.norm(p[-1] - homes_np[r]) < np.linalg.norm(p[0] - homes_np[r]):
            p, lk = p[::-1].copy(), lk[::-1].copy()
        up = np.tile(up_w, (len(p), 1))
        bad = np.abs(lk @ up_w) > 0.95
        up[bad] = np.array([0.0, 1.0, 0.0])
        init_pos[r], init_rot[r] = p, np.concatenate([lk, up], axis=1)
    for r in range(R):        # a robot left without a chunk idles at home
        if init_pos[r] is None:
            init_pos[r] = homes_np[r][None] + np.array([[0.0, 0.0, 1.0]])
            init_rot[r] = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return init_pos, init_rot


# ---------------------------------------------------------------------------
# Scoring (the pipeline's own machinery — never ours)
# ---------------------------------------------------------------------------

def hard_coverage(query, positions_np, rotmats_np):
    import cupy as cp
    V, _ = query.compute_visibility_batch(
        cp.asarray(positions_np, dtype=cp.float32),
        cp.asarray(rotmats_np, dtype=cp.float32))
    Vn = cp.asnumpy(V).astype(bool)
    return float(Vn.any(axis=0).mean()), Vn


def evaluate(tag, query, esdf_np_fn, homes_t, pos_list, rot_list, args):
    positions = torch.cat(pos_list).cpu().numpy()
    rotmats = sixd_to_rotmats(torch.cat(rot_list))
    cov, _ = hard_coverage(query, positions, rotmats)
    chains = [chain(homes_t[r], pos_list[r]) for r in range(len(pos_list))]
    lengths = [float(chain_lengths(c).sum()) for c in chains]
    dense = torch.cat([segment_samples(c, 40) for c in chains])
    min_clear = float(esdf_np_fn(dense).min())
    return dict(tag=tag, coverage=cov, makespan=max(lengths),
                total_length=sum(lengths), per_robot=lengths,
                min_clearance=min_clear, n_poses=len(positions))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline_dir",
                   default=os.path.join(_ROOT, "outputs", "pilot_baseline"))
    p.add_argument("--out", default=os.path.join(_DIR, "results", "joint_pilot"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--cold_steps", type=int, default=1200)
    p.add_argument("--lambda_route", type=float, default=0.02,
                   help="Route weight at SGD scale: grad per interior "
                        "waypoint is <= 2*lambda_route m/step.")
    p.add_argument("--alpha", type=float, default=None,
                   help="Makespan-vs-total blend; default = the pipeline run's.")
    p.add_argument("--lr_pos", type=float, default=0.05)
    p.add_argument("--lr_rot", type=float, default=0.02)
    p.add_argument("--beta", type=float, default=1.0,
                   help="Soft-makespan logsumexp temperature (1/m).")
    p.add_argument("--backbone", default="zbuf",
                   help="zbuf (geometric, validated F1 0.74 here) | nvps "
                        "(measured uninformative at this scale, F1 0.2).")
    p.add_argument("--keep_target", type=float, default=0.55)
    p.add_argument("--keep_margin", type=float, default=0.05,
                   help="Required metric slack (m) on every keep-pair margin.")
    p.add_argument("--lambda_route_cold", type=float, default=2e-3)
    p.add_argument("--w_cover", type=float, default=20.0,
                   help="Coverage weight for the cold (tradeoff) arm; "
                        "SGD-scale balancing is provisional (see plan).")
    p.add_argument("--cap_pos", type=float, default=0.04)
    p.add_argument("--cap_rot", type=float, default=0.02)
    p.add_argument("--objective", choices=["constrained", "tradeoff"],
                   default="constrained",
                   help="constrained = min route s.t. soft-coverage >= the "
                        "baseline poses' soft-coverage (the crisp headroom "
                        "question); tradeoff = -C + lambda_route*route.")
    p.add_argument("--al_every", type=int, default=30)
    p.add_argument("--col_samples", type=int, default=10)
    p.add_argument("--sep_samples", type=int, default=48)
    p.add_argument("--d_separation", type=float, default=0.8)
    p.add_argument("--robot_radius", type=float, default=0.35)
    p.add_argument("--clearance_margin", type=float, default=0.15)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--frustum_sharpness", type=float, default=12.0,
                   help="Soft-frustum steepness. Lower than hpro's 50 on "
                        "purpose: at metric scale (far=6 m) a 50-sharp gate "
                        "has a ~2 cm transition and starves the near/far "
                        "gradients that must pull poses toward the surface.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modes", default="warm,cold")
    p.add_argument("--no_show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    import cupy as cp
    import open3d as o3d
    from VRP.utils.serialization import load_pipeline
    from shared.mesh_loader import load_and_transform_mesh
    from shared.grid_builder_utils import build_occupancy_grid
    from visibility.core.types import FrustumParams
    from visibility.visibility.raycast_cuda import RaycastingVisibilityQueryCuda

    print(f"[1] Loading pipeline run: {args.pipeline_dir}")
    data = load_pipeline(args.pipeline_dir)
    pts_np = np.asarray(data["target_points"], dtype=np.float64)
    normals_np = np.asarray(data["normals"], dtype=np.float64)
    fr = data["frustum_params"]
    if args.alpha is None:
        args.alpha = float(data.get("alpha") or 0.5)
    K = int(data["num_robots"])
    homes_np = np.stack([np.asarray(x, dtype=np.float64)
                         for x in data["robot_start_xyzs"]])
    sel_pos = np.asarray(data["selected_positions"], dtype=np.float64)
    sel_rot = np.asarray(data["selected_rotmats"], dtype=np.float64)
    print(f"    N={len(pts_np)} pts, {len(sel_pos)} selected viewpoints, "
          f"{K} robots, alpha={args.alpha}")

    print("[2] Rebuilding mesh, occupancy grid, ESDF …")
    raw_tm = load_and_transform_mesh(data["mesh_path"],
                                     data["mesh_target_length"],
                                     data["mesh_pose"])
    og = build_occupancy_grid(mesh=raw_tm, padding=2.0, inflation_voxels=0,
                              resolution=0.10, fill_interior=True)
    esdf = Esdf(og, device)

    print("[3] Pipeline's own scorer (OptiX ray-cast) + §9.4 sanity gate …")
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(raw_tm.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(raw_tm.faces))
    query = RaycastingVisibilityQueryCuda(
        o3d_mesh, cp.asarray(pts_np), cp.asarray(normals_np),
        FrustumParams(fov_y=float(fr["fov_y_rad"]), aspect=float(fr["aspect"]),
                      near=float(fr["near"]), far=float(fr["far"])))
    base_cov, _ = hard_coverage(query, sel_pos, sel_rot)
    recorded = float(data["optimization_result"].total_coverage)
    print(f"    re-scored baseline coverage {base_cov:.4f} vs recorded "
          f"{recorded:.4f}")
    if abs(base_cov - recorded) > 0.01:
        raise SystemExit("SANITY GATE FAILED: cannot reproduce the pipeline's "
                         "own coverage for its own poses — fix conventions "
                         "before optimizing anything (§9.4).")

    print("[4] NVPS + frustum layer at metric scale …")
    fov_v = float(fr["fov_y_rad"])
    fov_h = 2.0 * math.atan(math.tan(fov_v / 2.0) * float(fr["aspect"]))
    zb_kw = (dict(grid_el=96, tol_rel=0.08, tol_abs=0.15)
             if args.backbone == "zbuf" else {})
    backbone = make_backbone(args.backbone, device=device, **zb_kw)
    t0 = time.perf_counter()
    backbone.prepare(pts_np, normals_np)
    print(f"    prepare({len(pts_np)} pts): {(time.perf_counter()-t0)*1e3:.0f} ms")
    layer = GatedVisibilityLayer(backbone, fov_h=fov_h, fov_v=fov_v,
                                 near=float(fr["near"]), far=float(fr["far"]),
                                 frustum_sharpness=args.frustum_sharpness,
                                 device=device)
    pts_t = torch.tensor(pts_np.T, dtype=torch.float32, device=device)
    homes_t = torch.tensor(homes_np, dtype=torch.float32, device=device)

    # Baseline record: coverage from their scorer, lengths from the EXECUTED
    # trajectories (what actually gets flown), makespan = max robot length.
    exec_lengths = []
    for traj in data["exec_result"].all_traj_positions:
        t = np.asarray(traj)[:, :3]
        exec_lengths.append(float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())
                            if len(t) > 1 else 0.0)
    results = [dict(tag="pipeline", coverage=base_cov,
                    makespan=max(exec_lengths), total_length=sum(exec_lengths),
                    per_robot=exec_lengths, min_clearance=float("nan"),
                    n_poses=len(sel_pos), wall_s=float("nan"))]
    print(f"    baseline: cov={base_cov:.4f} makespan={max(exec_lengths):.1f} m "
          f"total={sum(exec_lengths):.1f} m ({len(sel_pos)} poses)")

    # Warm start: per-robot ordered waypoints from the VRP routes.
    route_pos, route_rot = [], []
    for r, wp_idx in enumerate(data["robot_inspection_wp_indices"]):
        idx = np.asarray(wp_idx, dtype=int)
        route_pos.append(torch.tensor(sel_pos[idx], dtype=torch.float32,
                                      device=device))
        route_rot.append(torch.tensor(rotmats_to_6d(sel_rot[idx]),
                                      dtype=torch.float32, device=device))

    # The warm arm's keep-PAIRS: pose i is anchored to the points its OWN
    # ray-cast row says it sees (pipeline visibility_map) AND that the
    # surrogate agrees it sees at init (w0[i, j] > keep_target). Per-pose
    # responsibility instead of a union constraint: a union lets poses
    # negotiate coverage handoffs the surrogate then fails to execute (the
    # 0.951 -> 0.756 hard-coverage collapse), while anchors keep each pose in
    # the basin where its assignment is visible — the measured headroom is
    # then a LOWER bound (handoff gains are forgone).
    with torch.no_grad():
        w0 = layer(pts_t,
                   torch.tensor(sel_pos, dtype=torch.float32, device=device),
                   torch.tensor(rotmats_to_6d(sel_rot), dtype=torch.float32,
                                device=device)).clamp(0.0, 1.0 - 1e-7)
        c0 = 1.0 - torch.prod(1.0 - w0, dim=0)
    vis_gt = np.asarray(data["visibility_map"]).astype(bool)     # (V, N) sel order
    gt_union = vis_gt.any(axis=0)
    # Constrain the points the baseline covers AND the surrogate can plausibly
    # hold (soft union > keep_target at init): starting an AL on a constraint
    # that is infeasible for the surrogate just blows its multiplier up.
    agree = gt_union & (c0.cpu().numpy() > args.keep_target)
    keep_pairs = torch.tensor(np.nonzero(agree)[0], dtype=torch.long,
                              device=device)
    print(f"    soft union {float(c0.mean()):.4f} vs hard {base_cov:.4f}; "
          f"keep-points {len(keep_pairs)} of {int(gt_union.sum())} GT-covered "
          f"({args.objective} objective, margin {args.keep_margin} m)")
    if args.objective != "constrained":
        keep_pairs = None

    modes = [m.strip() for m in args.modes.split(",")]
    arm_chains = {}
    if "warm" in modes:
        print(f"[5] WARM joint solve ({args.steps} steps) …")
        pos_w, rot_w, info_w = optimize_joint(
            pts_t, layer, homes_t, route_pos, route_rot, esdf, args,
            "warm", keep_pairs=keep_pairs)
        res = evaluate("joint-warm", query, esdf, homes_t, pos_w, rot_w, args)
        res["wall_s"] = info_w["wall_s"]
        results.append(res)
        arm_chains["joint-warm"] = [
            chain(homes_t[r], pos_w[r]).cpu().numpy() for r in range(K)]
        print(f"    -> cov={res['coverage']:.4f} makespan={res['makespan']:.1f} m "
              f"clear={res['min_clearance']:.2f} m wall={res['wall_s']:.1f} s")

    if "cold" in modes:
        print(f"[6] COLD joint solve ({args.cold_steps} steps) …")
        standoff = 0.5 * (float(fr["near"]) + float(fr["far"]))
        cp_pos, cp_rot = cold_start(pts_np, normals_np, homes_np,
                                    len(sel_pos), K, standoff, rng)
        init_pos = [torch.tensor(p, dtype=torch.float32, device=device)
                    for p in cp_pos]
        init_rot = [torch.tensor(r, dtype=torch.float32, device=device)
                    for r in cp_rot]
        steps_saved, lr_saved = args.steps, args.lambda_route
        args.steps, args.lambda_route = args.cold_steps, args.lambda_route_cold
        pos_c, rot_c, info_c = optimize_joint(
            pts_t, layer, homes_t, init_pos, init_rot, esdf, args,
            "cold", keep_pairs=None)
        args.steps, args.lambda_route = steps_saved, lr_saved
        res = evaluate("joint-cold", query, esdf, homes_t, pos_c, rot_c, args)
        res["wall_s"] = info_c["wall_s"]
        results.append(res)
        arm_chains["joint-cold"] = [
            chain(homes_t[r], pos_c[r]).cpu().numpy() for r in range(K)]
        print(f"    -> cov={res['coverage']:.4f} makespan={res['makespan']:.1f} m "
              f"clear={res['min_clearance']:.2f} m wall={res['wall_s']:.1f} s")

    out_json = os.path.join(args.out, "joint_pilot.json")
    with open(out_json, "w") as fh:
        json.dump(dict(args=vars(args), results=results), fh, indent=2)
    print(f"\nJSON: {out_json}")

    print(f"\n=== headroom pilot (lambda_route={args.lambda_route}, "
          f"alpha={args.alpha}) ===")
    print(f"{'arm':<12} {'coverage':>9} {'makespan':>9} {'total':>8} "
          f"{'minclear':>9} {'poses':>6} {'wall s':>7}")
    for r in results:
        print(f"{r['tag']:<12} {r['coverage']:>9.4f} {r['makespan']:>8.1f}m "
              f"{r['total_length']:>7.1f}m {r['min_clearance']:>8.2f}m "
              f"{r['n_poses']:>6d} {r['wall_s']:>7.1f}")

    plot(results, pts_np, homes_np, data, args, arm_chains)


def plot(results, pts_np, homes_np, data, args, arm_chains):
    """arm_chains: {tag: list of (T+2, 3) numpy chains per robot}."""
    n3d = 1 + len(arm_chains)
    fig = plt.figure(figsize=(4.6 * (1 + n3d) * 0.9, 4.6))
    ax = fig.add_subplot(1, 1 + n3d, 1)
    marks = {"pipeline": "s", "joint-warm": "o", "joint-cold": "^"}
    for r in results:
        ax.scatter(r["makespan"], r["coverage"], s=70,
                   marker=marks.get(r["tag"], "x"),
                   label=f"{r['tag']}  {r['coverage']:.3f} @ {r['makespan']:.0f} m")
    ax.set_xlabel("makespan (m)")
    ax.set_ylabel("hard coverage (pipeline's own ray-cast)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("coverage vs makespan")

    def panel(k, title, chains):
        a = fig.add_subplot(1, 1 + n3d, k, projection="3d")
        a.scatter(*pts_np[::10].T, s=0.5, c="0.82", alpha=0.35, linewidths=0)
        for r, c in enumerate(chains):
            a.plot(*np.asarray(c).T, lw=1.6,
                   color=["royalblue", "seagreen", "darkorange",
                          "purple", "crimson"][r % 5])
        a.scatter(*homes_np.T, c="k", marker="s", s=40)
        a.set_title(title, fontsize=9)
        a.set_axis_off()
        lim = np.array([pts_np.min(axis=0), pts_np.max(axis=0)])
        c0, rad = lim.mean(axis=0), 0.55 * (lim[1] - lim[0]).max()
        a.set_xlim(c0[0] - rad, c0[0] + rad)
        a.set_ylim(c0[1] - rad, c0[1] + rad)
        a.set_zlim(c0[2] - rad, c0[2] + rad)

    base_chains = [np.asarray(t)[:, :3]
                   for t in data["exec_result"].all_traj_positions]
    res_by = {r["tag"]: r for r in results}
    panel(2, f"pipeline ({res_by['pipeline']['makespan']:.0f} m makespan)",
          base_chains)
    for i, (tag, chains) in enumerate(arm_chains.items()):
        panel(3 + i, f"{tag} ({res_by[tag]['makespan']:.0f} m makespan)",
              chains)

    fig.tight_layout()
    path = os.path.join(args.out, "joint_pilot.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
