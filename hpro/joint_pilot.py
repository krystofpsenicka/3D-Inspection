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


def _metric_margins(layer, pts_t: torch.Tensor, pos: torch.Tensor,
                    rot: torch.Tensor) -> torch.Tensor:
    """(V, N) metric visibility margin: min over the five constraints that
    make a point visible from a pose — occlusion (backbone ``raw_margin``),
    near plane, far plane, horizontal and vertical frustum walls — all in
    metres, positive = visible with that much slack. Unit-scale gradients
    everywhere (no sigmoid ratchet, §9.5)."""
    m_occ = layer.backbone.raw_margin(pts_t, pos)
    look, up, right = six_d_to_rotation_matrix(rot)
    dvec = pts_t.T.unsqueeze(0) - pos.unsqueeze(1)        # (V, N, 3)
    depth = (dvec * look.unsqueeze(1)).sum(-1)
    m_h = (math.tan(layer.fov_h / 2.0) * depth
           - (dvec * right.unsqueeze(1)).sum(-1).abs())
    m_v = (math.tan(layer.fov_v / 2.0) * depth
           - (dvec * up.unsqueeze(1)).sum(-1).abs())
    return torch.stack([m_occ, depth - layer.near,
                        layer.far - depth, m_h, m_v]).min(dim=0).values


def _chains_of(homes, pos_list):
    return [chain(homes[r], pos_list[r]) for r in range(len(pos_list))]


def _max_sep_violation(chains, times, speed, d_sep) -> float:
    v = 0.0
    for a in range(len(chains)):
        for b in range(a + 1, len(chains)):
            pa = positions_at_times(chains[a], times, speed)
            pb = positions_at_times(chains[b], times, speed)
            v = max(v, float((d_sep - (pa - pb).norm(dim=1)).max()))
    return max(v, 0.0)


def _pose_seg_samples(x, prev, nxt, per_seg):
    """(Vm, 2*per_seg, 3): samples along prev->x and x->nxt per moving pose."""
    t = torch.linspace(0.0, 1.0, per_seg + 1, device=x.device)[1:].view(1, -1, 1)
    s1 = prev.unsqueeze(1) + (x - prev).unsqueeze(1) * t
    s2 = x.unsqueeze(1) + (nxt - x).unsqueeze(1) * t
    return torch.cat([s1, s2], dim=1)


def _pose_col_violation(esdf, x, prev, nxt, margin, per_seg):
    """(Vm,) worst clearance violation over each pose's two adjacent segments."""
    s = torch.cat([_pose_seg_samples(x, prev, nxt, per_seg),
                   x.unsqueeze(1)], dim=1)
    return F.relu(margin - esdf(s)).amax(dim=1)


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
            M = _metric_margins(layer, pts_t, all_pos, all_rot)   # (V, N)
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


def _prune_pass(pos, rot, view, homes, layer, pts_t, esdf, args, keep_idx,
                req, times, gt_vis=None):
    """Greedily remove poses whose held points are all redundantly held.

    Pose margins are pose-local (removing a pose changes no other pose's
    visibility), so one margin matrix per pass suffices: a pose is removable
    iff it is the SOLE holder of no keep-point, the straight shortcut between
    its neighbours is collision-clear, and separation does not worsen. Via
    waypoints hold nothing by construction, so they are removable whenever
    their shortcut is clear. Poses are removed best-route-saving-first;
    savings are recomputed as neighbours merge. This is the discrete gain
    sliding unlocks — after the sweeps, poses drift until some are redundant,
    which set cover could never undo once routing made them expensive.
    Returns the number removed (``pos``/``rot``/``view`` rebuilt in place).
    """
    device = pts_t.device
    R = len(pos)
    margin = args.robot_radius + args.clearance_margin
    tol = 1e-3
    with torch.no_grad():
        view_all = torch.cat(view)
        if gt_vis is not None:
            # Truth-based redundancy: a pose is removable only if it is the
            # sole RAY-CAST cover of nothing (via rows are already False).
            held = gt_vis
        else:
            M_keep = _metric_margins(layer, pts_t, torch.cat(pos),
                                     torch.cat(rot))[:, keep_idx]
            held = (M_keep >= req.unsqueeze(0)) & view_all.unsqueeze(1)
        cnt = held.sum(dim=0)
        T = [len(p) for p in pos]
        off = np.concatenate([[0], np.cumsum(T)]).astype(int)
        alive = [list(range(T[r])) for r in range(R)]
        sep0 = (_max_sep_violation(_chains_of(homes, pos), times,
                                   args.speed, args.d_separation)
                if R > 1 else 0.0)
        blocked = set()
        n_removed = 0
        while True:
            best = None
            for r in range(R):
                seq = alive[r]
                for li, i in enumerate(seq):
                    g = int(off[r] + i)
                    if g in blocked:
                        continue
                    if bool((held[g] & (cnt == 1)).any()):
                        continue                      # sole holder of something
                    p_prev = pos[r][seq[li - 1]] if li > 0 else homes[r]
                    p_next = pos[r][seq[li + 1]] if li < len(seq) - 1 else homes[r]
                    x = pos[r][i]
                    save = float((x - p_prev).norm() + (x - p_next).norm()
                                 - (p_next - p_prev).norm())
                    if best is None or save > best[0]:
                        best = (save, r, li, g, p_prev, p_next)
            if best is None or best[0] <= 1e-3:
                break
            save, r, li, g, p_prev, p_next = best
            t = torch.linspace(0.0, 1.0, 4 * args.col_samples + 1, device=device)
            s = p_prev.unsqueeze(0) + (p_next - p_prev).unsqueeze(0) * t.unsqueeze(1)
            if float(esdf(s).min()) < margin - tol:
                blocked.add(g)                        # shortcut cuts the hull
                continue
            trial_seq = alive[r][:li] + alive[r][li + 1:]
            if R > 1:
                idx = torch.tensor(trial_seq, dtype=torch.long, device=device)
                trial_pos = [pos[rr] if rr != r else pos[r][idx]
                             for rr in range(R)]
                sep_t = _max_sep_violation(_chains_of(homes, trial_pos), times,
                                           args.speed, args.d_separation)
                if sep_t > max(sep0, tol):
                    blocked.add(g)
                    continue
            alive[r] = trial_seq
            cnt = cnt - held[g].to(cnt.dtype)
            n_removed += 1
        for r in range(R):
            idx = torch.tensor(alive[r], dtype=torch.long, device=device)
            pos[r] = pos[r][idx]
            rot[r] = rot[r][idx]
            view[r] = view[r][idx]
        if gt_vis is not None:
            keep_rows = torch.tensor(
                [off[r] + i for r in range(R) for i in alive[r]],
                dtype=torch.long, device=device)
            gt_vis = gt_vis[keep_rows]
    return n_removed, gt_vis


def _insert_vias(pos, rot, view, homes, esdf, args, max_vias=8):
    """Insert route-only via waypoints into segments that violate clearance.

    The analogue of the pipeline's ST-A* detours: a via carries no visibility
    duty (view=False — excluded from keep bookkeeping, coverage scoring and
    the viewpoint budget) and exists only so the chain can bend around the
    structure. The via seed is the violating segment's worst sample pushed
    outward along the ESDF gradient until it clears margin + 0.1 m; the
    elastic sweeps then shorten it like any other waypoint. Returns the
    number inserted.
    """
    device = homes.device
    margin = args.robot_radius + args.clearance_margin
    n_ins = 0
    for r in range(len(pos)):
        guard = 0
        while guard < max_vias:
            ch = chain(homes[r], pos[r])
            s = segment_samples(ch, 40).reshape(len(ch) - 1, 40, 3)
            with torch.no_grad():
                d = esdf(s)
            seg_min, seg_arg = d.min(dim=1)
            viol = (seg_min < margin - 1e-3).nonzero(as_tuple=True)[0]
            if len(viol) == 0:
                break
            k = int(viol[0])
            x = s[k, int(seg_arg[k])].detach().clone()
            for _ in range(30):
                xx = x.unsqueeze(0).clone().requires_grad_(True)
                dd = esdf(xx)
                if float(dd) >= margin + 0.1:
                    break
                g = torch.autograd.grad(dd.sum(), xx)[0][0]
                x = (x + g / g.norm().clamp_min(1e-6)
                     * float(margin + 0.1 - dd) * 1.2).detach()
            if len(pos[r]):
                ri = min(k, len(pos[r]) - 1)
                rrow = rot[r][ri:ri + 1]
            else:
                rrow = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                                    device=device)
            pos[r] = torch.cat([pos[r][:k], x.unsqueeze(0), pos[r][k:]])
            rot[r] = torch.cat([rot[r][:k], rrow, rot[r][k:]])
            view[r] = torch.cat([view[r][:k],
                                 torch.zeros(1, dtype=torch.bool, device=device),
                                 view[r][k:]])
            n_ins += 1
            guard += 1
    return n_ins


def optimize_elastic(pts_t, layer, homes, init_pos, init_rot, esdf, args,
                     keep_idx, label="warm", uncovered_idx=None, budgets=None,
                     audit=None):
    """Elastic-band coordinate descent over poses (§9.6 solver design #6).

    ``audit`` (optional but used by default): callable
    ``(pos (n,3), rot6d (n,6)) -> bool (n, N)`` wrapping the pipeline's own
    ray-caster. When given, every half-sweep's accepted moves are AUDITED and
    offenders reverted until aggregate hard coverage is non-decreasing — hard
    coverage becomes monotone by construction (the Stage-B audit-loop
    pattern: the surrogate steers gradients, measurement gates acceptance;
    never repair, only accept/reject). Captures, prune redundancy and the
    expand pool then all use ray-cast truth instead of surrogate claims. The
    audited pose count is reported (fairness: the pipeline's own planning
    ray-casts all 600 candidates up front).

    Why this shape: all five §9.5 formulations were *simultaneous* first-order
    descent over ~50 coupled poses, and each either exploded (Adam erases
    force hierarchies), ratcheted (sigmoid constraints), deadlocked (per-pose
    anchors), or chattered (global argmax). Coordinate descent removes the
    coupling instead of fighting it:

    * **Alternating half-sweeps** over odd/even chain indices: every moving
      pose's two neighbours are frozen (adjacent indices have the other
      parity; chain ends are homes), so the batch of movers is a set of
      INDEPENDENT 9-parameter problems — no route-gradient cancellation, no
      shared segments — solved together in one batched penalty inner loop.
    * **Per-pose acceptance test** after each inner loop keeps the state
      monotone: a move is taken only if it strictly reduces that pose's own
      worst constraint violation, or is feasible and does not lengthen its
      two segments (shorten phase) / pays route for captured points within
      the budget (expand phase). Rejected movers revert.
    * **Coverage as a maintained invariant, not a traded term**: every
      ray-cast-covered point j carries a requirement
      ``req_j = min(keep_margin, best init metric margin)`` — the best pose
      margin for j may never drop below what the warm start provided (capped
      at ``keep_margin`` so well-seen points still leave slack to move
      through). Each half-sweep, points no frozen pose holds are assigned to
      their argmax *mover* — handoffs between poses stay legal across sweeps
      (the §9.5 per-pose-anchor lesson) while parallel moves stay safe (only
      one mover is ever responsible for a point).
    * **Via waypoints** (``_insert_vias``, at init): route-only points that
      bend chain segments around the structure, the analogue of the
      pipeline's ST-A* detours — excluded from coverage scoring and the
      viewpoint budget. With them the state is feasible from the start and
      the acceptance test keeps it feasible forever.
    * **Prune rounds** (``_prune_pass``) interleave with sweep rounds:
      sliding makes poses redundant, removal shortens the tour, which frees
      more sliding.
    * **Expand phase** (after shorten+prune): spend freed route budget and
      free orientation changes capturing baseline-UNCOVERED points — the
      surface sample-and-select left behind (greedy stopped, or no candidate
      sees it at all). Same feasibility invariants; per-robot chain length
      capped at ``budgets`` (the baseline's own executed lengths); a capture
      is a pool point brought to metric margin >= keep_margin by a view
      pose, and is immediately appended to the keep set so gains are
      monotone. Captures are surrogate CLAIMS — the pipeline's ray-cast has
      the last word at evaluation.

    Structure collision: penalty on each mover's two segments + waypoint
    projection (``project_clear``) + the acceptance test. Separation: penalty
    inside the inner loop on the assembled chains + a per-half-sweep global
    check that reverts the half-sweep if the worst violation grew.
    """
    device = pts_t.device
    R = len(init_pos)
    pos = [p.clone() for p in init_pos]
    rot = [r.clone() for r in init_rot]
    view = [torch.ones(len(p), dtype=torch.bool, device=device) for p in pos]
    margin = args.robot_radius + args.clearance_margin
    cap_req = args.keep_margin
    tol = 1e-3
    t0 = time.perf_counter()

    with torch.no_grad():
        best0 = _metric_margins(layer, pts_t, torch.cat(pos),
                                torch.cat(rot))[:, keep_idx].max(dim=0).values
        req = torch.minimum(torch.full_like(best0, args.keep_margin), best0)
        t_max = 1.3 * max(float(chain_lengths(chain(homes[r], pos[r])).sum())
                          for r in range(R)) / args.speed
    times = torch.linspace(0.0, t_max, args.sep_samples, device=device)
    pool = (uncovered_idx.clone() if uncovered_idx is not None
            else torch.zeros(0, dtype=torch.long, device=device))
    print(f"    [{label}] keep set {len(keep_idx)} pts, req<0 for "
          f"{int((req < 0).sum())} (surrogate-blind: only non-degradation is "
          f"asked of them), target margin {args.keep_margin} m")

    n_vias = _insert_vias(pos, rot, view, homes, esdf, args)
    if n_vias:
        print(f"    [{label}] inserted {n_vias} via waypoint(s): segments of "
              f"the straight warm start ran below the {margin:.2f} m "
              f"clearance constraint")

    audit_count = [0]
    gt_vis = None
    if audit is not None:
        def audited(p, r6):
            audit_count[0] += len(p)
            return audit(p, r6)
        with torch.no_grad():
            gt_vis = audited(torch.cat(pos), torch.cat(rot))
            gt_vis[~torch.cat(view)] = False
        floor_count = int(gt_vis.any(dim=0).sum())
        print(f"    [{label}] audit gate ON: hard coverage floor "
              f"{float(gt_vis.any(dim=0).float().mean()):.4f} (surplus above "
              f"the floor is tradeable — final coverage can only exceed it)")

    def scaffold(parity):
        T = [len(p) for p in pos]
        off = np.concatenate([[0], np.cumsum(T)]).astype(int)
        all_pos, all_rot = torch.cat(pos), torch.cat(rot)
        V = len(all_pos)
        mov_l, rob_l, prev_l, nxt_l = [], [], [], []
        for r in range(R):
            for i in range(parity, T[r], 2):
                mov_l.append(off[r] + i)
                rob_l.append(r)
                prev_l.append(pos[r][i - 1] if i > 0 else homes[r])
                nxt_l.append(pos[r][i + 1] if i < T[r] - 1 else homes[r])
        if not mov_l:
            return None
        mov = torch.tensor(mov_l, dtype=torch.long, device=device)
        fix_mask = torch.ones(V, dtype=torch.bool, device=device)
        fix_mask[mov] = False
        return dict(T=T, off=off, all_pos=all_pos, all_rot=all_rot,
                    view_all=torch.cat(view), mov=mov,
                    rob_of=np.asarray(rob_l), fix_mask=fix_mask,
                    prev=torch.stack(prev_l), nxt=torch.stack(nxt_l))

    def responsibilities(sc):
        """Full margin matrix + per-point assignment of unheld keep points to
        their argmax VIEW mover (vias never take responsibility)."""
        M_full = _metric_margins(layer, pts_t, sc["all_pos"], sc["all_rot"])
        M_keep = M_full[:, keep_idx]
        fix_view = sc["fix_mask"] & sc["view_all"]
        held_fixed = ((M_keep[fix_view] >= req.unsqueeze(0)).any(dim=0)
                      if int(fix_view.sum()) else
                      torch.zeros(len(keep_idx), dtype=torch.bool,
                                  device=device))
        un = (~held_fixed).nonzero(as_tuple=True)[0]
        out = dict(M_full=M_full, un=un)
        if len(un):
            sub = M_keep[sc["mov"]][:, un].clone()
            sub[~sc["view_all"][sc["mov"]]] = -torch.inf
            out.update(resp_v=sub.argmax(dim=0), kk=keep_idx[un],
                       req_un=req[un], m_old=sub.max(dim=0).values)
        return out

    def sweep_step(parity, expand=False):
        """One half-sweep. Returns (accepted moves, points newly captured)."""
        nonlocal keep_idx, req, pool, gt_vis
        if expand and gt_vis is not None:
            pool = (~gt_vis.any(dim=0)).nonzero(as_tuple=True)[0]
        sc = scaffold(parity)
        if sc is None:
            return 0, 0
        mov, prev, nxt = sc["mov"], sc["prev"], sc["nxt"]
        all_pos, all_rot = sc["all_pos"], sc["all_rot"]
        view_all, fix_mask, off = sc["view_all"], sc["fix_mask"], sc["off"]
        Vm = len(mov)
        view_mov = view_all[mov]

        with torch.no_grad():
            rsp = responsibilities(sc)
            un = rsp["un"]
            U = len(un)
            if U:
                resp_v, kk, req_un = rsp["resp_v"], rsp["kk"], rsp["req_un"]
                old_kv = torch.zeros(Vm, device=device).scatter_reduce(
                    0, resp_v, req_un - rsp["m_old"], reduce="amax",
                    include_self=True)
            else:
                old_kv = torch.zeros(Vm, device=device)
            old_route = ((all_pos[mov] - prev).norm(dim=1)
                         + (all_pos[mov] - nxt).norm(dim=1))
            old_cv = _pose_col_violation(esdf, all_pos[mov], prev, nxt,
                                         margin, args.col_samples)
            sep_before = (_max_sep_violation(_chains_of(homes, pos), times,
                                             args.speed, args.d_separation)
                          if R > 1 else 0.0)
            pool_act = pool
            old_cap = torch.zeros(Vm, device=device)
            if expand and len(pool):
                if gt_vis is None:
                    # No audit: exclude pool points the surrogate already
                    # (unverifiably) claims via a frozen pose.
                    fix_view = fix_mask & view_all
                    pf = ((rsp["M_full"][fix_view][:, pool]
                           >= cap_req).any(dim=0)
                          if int(fix_view.sum()) else
                          torch.zeros(len(pool), dtype=torch.bool,
                                      device=device))
                    pool_act = pool[~pf]
                if len(pool_act):
                    oc = (rsp["M_full"][mov][:, pool_act]
                          >= cap_req).sum(dim=1).float()
                    old_cap = torch.where(view_mov, oc, torch.zeros_like(oc))
            do_gain = expand and len(pool_act) > 0
            L_now = [float(chain_lengths(chain(homes[r], pos[r])).sum())
                     for r in range(R)]

        x = all_pos[mov].clone().requires_grad_(True)
        q = all_rot[mov].clone().requires_grad_(True)
        x_start = all_pos[mov].clone()
        q_start = all_rot[mov].clone()
        opt = torch.optim.SGD([x, q], lr=args.eb_lr, momentum=0.9)
        route_w = args.eb_route_reg if expand else 1.0

        def clamp_trust():
            # Expand trust region: net-positive coverage trades are SMALL
            # leans (measured: unclamped movers dive metres toward prey,
            # abandon their old points wholesale, and the GT gate reverts
            # every one). Bounding each half-sweep's total displacement
            # keeps candidates in the incremental regime; the ratchet
            # (capture -> re-anchor -> lean again) supplies the range.
            with torch.no_grad():
                dx = x - x_start
                n = dx.norm(dim=1, keepdim=True)
                x.copy_(x_start + dx * (args.eb_trust
                                        / n.clamp_min(args.eb_trust)))
                dq = q - q_start
                n = dq.norm(dim=1, keepdim=True)
                q.copy_(q_start + dq * (args.eb_trust_rot
                                        / n.clamp_min(args.eb_trust_rot)))
        for _ in range(args.eb_inner):
            opt.zero_grad()
            route = ((x - prev).norm(dim=1) + (x - nxt).norm(dim=1)).sum()
            pen = x.new_zeros(())
            gain = x.new_zeros(())
            if U or do_gain:
                Mm = _metric_margins(layer, pts_t, x, q)
            if U:
                m = Mm[resp_v, kk]
                # In expand the keep term is GUIDANCE, not a constraint —
                # the GT gate arbitrates coverage trades; a hard keep pull
                # would (measured) pin every mover and forbid all capture.
                kw = args.eb_keep_soft if expand else 1.0
                pen = pen + kw * F.relu(req_un + args.eb_slack
                                        - m).square().sum()
            if do_gain:
                gm = torch.sigmoid(args.eb_gain_sharp
                                   * (Mm[:, pool_act] - cap_req))
                gain = gm[view_mov].sum()
            s = _pose_seg_samples(x, prev, nxt, args.col_samples)
            pen = pen + F.relu(margin + args.eb_slack - esdf(s)).square().sum()
            if R > 1 or (expand and budgets is not None):
                full = all_pos.clone()
                full[mov] = x
                chs = [chain(homes[r], full[off[r]:off[r + 1]])
                       for r in range(R)]
                for a in range(R):
                    for b in range(a + 1, R):
                        pa = positions_at_times(chs[a], times, args.speed)
                        pb = positions_at_times(chs[b], times, args.speed)
                        pen = pen + F.relu(
                            args.d_separation - (pa - pb).norm(dim=1)
                        ).square().sum()
                if expand and budgets is not None:
                    for r in range(R):
                        pen = pen + F.relu(chain_lengths(chs[r]).sum()
                                           - budgets[r]).square()
            (route_w * route + args.eb_mu * pen
             - args.eb_w_gain * gain).backward()
            x0, q0 = x.detach().clone(), q.detach().clone()
            opt.step()
            with torch.no_grad():
                d = x - x0
                n = d.norm(dim=1, keepdim=True)
                x.copy_(x0 + d * (args.cap_pos / n.clamp_min(args.cap_pos)))
                d = q - q0
                n = d.norm(dim=1, keepdim=True)
                q.copy_(q0 + d * (args.cap_rot / n.clamp_min(args.cap_rot)))
            if expand:
                clamp_trust()
            project_clear(esdf, [x], margin)

        if expand:
            # Repair pass: the capture reward may pull segments through the
            # structure; project back to collision feasibility (the one hard
            # per-pose constraint in expand) before the acceptance test.
            opt2 = torch.optim.SGD([x, q], lr=args.eb_lr, momentum=0.5)
            for _ in range(args.eb_repair):
                opt2.zero_grad()
                s = _pose_seg_samples(x, prev, nxt, args.col_samples)
                pen = F.relu(margin + args.eb_slack - esdf(s)).square().sum()
                if float(pen) <= 1e-10:
                    break
                (args.eb_mu * pen).backward()
                x0, q0 = x.detach().clone(), q.detach().clone()
                opt2.step()
                with torch.no_grad():
                    d = x - x0
                    n = d.norm(dim=1, keepdim=True)
                    x.copy_(x0 + d * (args.cap_pos / n.clamp_min(args.cap_pos)))
                    d = q - q0
                    n = d.norm(dim=1, keepdim=True)
                    q.copy_(q0 + d * (args.cap_rot / n.clamp_min(args.cap_rot)))
                project_clear(esdf, [x], margin)

        with torch.no_grad():
            xd, qd = x.detach(), q.detach()
            Mn = (_metric_margins(layer, pts_t, xd, qd)
                  if (U or do_gain) else None)
            if U:
                new_kv = torch.zeros(Vm, device=device).scatter_reduce(
                    0, resp_v, req_un - Mn[resp_v, kk], reduce="amax",
                    include_self=True)
            else:
                new_kv = torch.zeros(Vm, device=device)
            new_route = (xd - prev).norm(dim=1) + (xd - nxt).norm(dim=1)
            new_cv = _pose_col_violation(esdf, xd, prev, nxt, margin,
                                         args.col_samples)
            old_v = torch.maximum(old_kv, old_cv)
            new_v = torch.maximum(new_kv, new_cv)
            if expand:
                # Coverage feasibility is the GT gate's job in expand (net
                # aggregate non-decrease); only collision stays hard.
                feas = torch.where(old_cv > tol, new_cv < old_cv - 1e-6,
                                   new_cv <= tol)
            else:
                feas = torch.where(old_v > tol, new_v < old_v - 1e-6,
                                   new_v <= tol)
            rows_cand = None
            if not expand:
                accept = feas & ((old_v > tol)
                                 | (new_route <= old_route + 1e-6))
            else:
                if gt_vis is not None:
                    # Audit every mover candidate: moves are judged on their
                    # RAY-CAST-verified net effect — captures minus the
                    # sole-covered points the move abandons — not surrogate
                    # claims (the surrogate's false positives are invisible
                    # to its own gradients and must not buy route). Requiring
                    # net >= 0 per mover stops route gains from silently
                    # spending coverage surplus (measured: floor-only gating
                    # dissipated +64 captured points to +12 in one sweep).
                    rows_cand = audited(xd, qd)
                    rows_cand[~view_mov] = False
                    prey = ~gt_vis.any(dim=0)
                    sole = gt_vis.sum(dim=0) == 1
                    gain_t = (rows_cand & prey.unsqueeze(0)).sum(dim=1)
                    loss_t = (gt_vis[mov] & sole.unsqueeze(0)
                              & ~rows_cand).sum(dim=1)
                    dc = (gain_t - loss_t).float().cpu().numpy()
                elif do_gain:
                    nc = (Mn[:, pool_act] >= cap_req).sum(dim=1).float()
                    new_cap = torch.where(view_mov, nc, torch.zeros_like(nc))
                    dc = (new_cap - old_cap).cpu().numpy()
                else:
                    dc = np.zeros(Vm)
                dr = (new_route - old_route).cpu().numpy()
                feas_np = feas.cpu().numpy()
                infeas_old = (old_v > tol).cpu().numpy()
                rob_of = sc["rob_of"]
                acc_np = np.zeros(Vm, dtype=bool)
                L_new = list(L_now)
                for v in range(Vm):          # free moves + feasibility repair
                    if feas_np[v] and (infeas_old[v] or
                                       (dr[v] <= 1e-6
                                        and (dc[v] > 0
                                             or (dr[v] < -1e-6
                                                 and dc[v] >= 0)))):
                        acc_np[v] = True
                        L_new[rob_of[v]] += dr[v]
                order = sorted([v for v in range(Vm)
                                if feas_np[v] and not acc_np[v]
                                and dr[v] > 1e-6 and dc[v] > 0],
                               key=lambda v: -(dc[v] / dr[v]))
                for v in order:              # paid moves, best value first
                    rr = int(rob_of[v])
                    if budgets is None or L_new[rr] + dr[v] <= budgets[rr] + 1e-6:
                        acc_np[v] = True
                        L_new[rr] += dr[v]
                accept = torch.tensor(acc_np, device=device)
                if args.eb_debug:
                    print(f"      [dbg expand p{parity}] feas="
                          f"{int(feas.sum())}/{Vm} dc>0={int((dc > 0).sum())} "
                          f"dc max={dc.max():.0f} dr=({dr.min():+.3f},"
                          f"{dr.max():+.3f}) newv max={float(new_v.max()):.4f} "
                          f"acc={int(accept.sum())}", flush=True)
            # GT audit gate: aggregate hard coverage may never drop. Audit
            # the accepted movers' rows, then revert the least valuable
            # offenders until coverage is non-decreasing (accept/reject only
            # — never repair).
            n_capt = 0
            gt_new = None
            base_any = None
            if gt_vis is not None and int(accept.sum()):
                acc_idx = accept.nonzero(as_tuple=True)[0]
                if rows_cand is not None:
                    rows_new = rows_cand[acc_idx]
                else:
                    rows_new = audited(xd[acc_idx], qd[acc_idx])
                    rows_new[~view_mov[acc_idx]] = False
                gt_new = gt_vis.clone()
                gt_new[mov[acc_idx]] = rows_new
                base_any = gt_vis.any(dim=0)
                base_cov = int(base_any.sum())
                # Gate against the BASELINE floor, not the current count:
                # surplus captures are tradeable (a pose front advancing
                # must be allowed to drop trailing points another sweep
                # will re-cover); the final guarantee is unchanged.
                while int(gt_new.any(dim=0).sum()) < floor_count:
                    best = None
                    for v in acc_idx.tolist():
                        if not bool(accept[v]):
                            continue
                        g = int(mov[v])
                        tmp = gt_new.clone()
                        tmp[g] = gt_vis[g]
                        c = int(tmp.any(dim=0).sum())
                        if best is None or c > best[0]:
                            best = (c, v, g)
                    if best is None:
                        break
                    _, v, g = best
                    accept[v] = False
                    gt_new[g] = gt_vis[g]
                n_capt = max(0, int(gt_new.any(dim=0).sum()) - base_cov)
            acc = mov[accept]
            if len(acc) == 0:
                return 0, 0
            all_pos[acc] = xd[accept]
            all_rot[acc] = qd[accept]
            new_pos = [all_pos[off[r]:off[r + 1]].clone() for r in range(R)]
            new_rot = [all_rot[off[r]:off[r + 1]].clone() for r in range(R)]
            if R > 1:
                sep_after = _max_sep_violation(_chains_of(homes, new_pos),
                                               times, args.speed,
                                               args.d_separation)
                if sep_after > max(sep_before, tol):
                    return 0, 0              # revert whole half-sweep
            for r in range(R):
                pos[r], rot[r] = new_pos[r], new_rot[r]
            if gt_new is not None:
                gt_vis = gt_new
                if expand:
                    # Re-anchor the differentiable keep set to the new
                    # VERIFIED coverage (captures enter, traded-away points
                    # leave; the GT gate already guaranteed the trade was
                    # net non-negative).
                    keep_idx = gt_vis.any(dim=0).nonzero(as_tuple=True)[0]
                    Mcur = _metric_margins(layer, pts_t, torch.cat(pos),
                                           torch.cat(rot))
                    best = Mcur[torch.cat(view)][:, keep_idx].max(dim=0).values
                    req = torch.minimum(
                        torch.full_like(best, cap_req), best)
                elif n_capt and Mn is not None:
                    # Shorten phase: lock incidental verified captures into
                    # the keep set so the surrogate defends them.
                    newly = (gt_new.any(dim=0) & ~base_any).nonzero(
                        as_tuple=True)[0]
                    if len(newly):
                        mbest = Mn[:, newly].max(dim=0).values
                        keep_idx = torch.cat([keep_idx, newly])
                        req = torch.cat([req, torch.minimum(
                            torch.full_like(mbest, cap_req), mbest)])
            elif expand and len(pool):
                # No audit: surrogate-claimed captures (unverified).
                M2 = _metric_margins(layer, pts_t, torch.cat(pos),
                                     torch.cat(rot))
                capt = (M2[torch.cat(view)][:, pool] >= cap_req).any(dim=0)
                n_capt = int(capt.sum())
                if n_capt:
                    keep_idx = torch.cat([keep_idx, pool[capt]])
                    req = torch.cat([req, torch.full((n_capt,), cap_req,
                                                     device=device)])
                    pool = pool[~capt]
            return int(accept.sum()), n_capt

    def lengths_now():
        return [float(chain_lengths(chain(homes[r], pos[r])).sum())
                for r in range(R)]

    log = []
    n_pruned_total = 0
    L = lengths_now()
    print(f"    [{label}] init: total={sum(L):.1f} m makespan={max(L):.1f} m "
          f"({sum(int(v.sum()) for v in view)} poses + {n_vias} vias)")
    for rnd in range(args.eb_rounds):
        prev_total = sum(lengths_now())
        stall = 0
        for sweep in range(args.eb_sweeps):
            sweep_step(0)
            sweep_step(1)
            L = lengths_now()
            imp = prev_total - sum(L)
            prev_total = sum(L)
            if (sweep + 1) % 5 == 0 or sweep == 0:
                print(f"    [{label} r{rnd} sweep {sweep+1:3d}] "
                      f"total={sum(L):.1f} m makespan={max(L):.1f} m "
                      f"(-{imp:.2f} m/sweep)", flush=True)
            log.append(dict(round=rnd, sweep=sweep + 1, total=sum(L),
                            makespan=max(L)))
            stall = stall + 1 if imp < args.eb_tol else 0
            if stall >= 2:
                break
        if args.no_prune:
            break
        npr, gt_vis = _prune_pass(pos, rot, view, homes, layer, pts_t, esdf,
                                  args, keep_idx, req, times, gt_vis=gt_vis)
        n_pruned_total += npr
        L = lengths_now()
        print(f"    [{label}] prune round {rnd}: removed {npr} poses -> "
              f"total={sum(L):.1f} m makespan={max(L):.1f} m")
        log.append(dict(round=rnd, pruned=npr, total=sum(L), makespan=max(L)))
        if npr == 0:
            break

    claimed_capt = 0
    if not args.no_expand and (len(pool) or gt_vis is not None):
        if gt_vis is not None:
            pool = (~gt_vis.any(dim=0)).nonzero(as_tuple=True)[0]
            print(f"    [{label}] EXPAND (audit-gated): {len(pool)} "
                  f"ray-cast-uncovered points in play, budgets "
                  + (", ".join(f'{b:.1f} m' for b in budgets)
                     if budgets else "none"))
        else:
            with torch.no_grad():
                Mf = _metric_margins(layer, pts_t, torch.cat(pos),
                                     torch.cat(rot))
                already = (Mf[torch.cat(view)][:, pool] >= cap_req).any(dim=0)
            pool = pool[~already]
            print(f"    [{label}] EXPAND: {len(pool)} baseline-uncovered "
                  f"points in play ({int(already.sum())} more are "
                  f"surrogate-claimed already — excluded, unverifiable), "
                  f"budgets "
                  + (", ".join(f'{b:.1f} m' for b in budgets)
                     if budgets else "none"))
        for sweep in range(args.eb_expand_sweeps):
            a0, c0 = sweep_step(0, expand=True)
            a1, c1 = sweep_step(1, expand=True)
            if gt_vis is not None:
                claimed_capt = int(gt_vis.any(dim=0).sum()) - floor_count
            else:
                claimed_capt += c0 + c1
            L = lengths_now()
            print(f"    [{label} expand {sweep+1:3d}] net +{claimed_capt} "
                  f"pts vs floor (pool {len(pool)}), total={sum(L):.1f} m "
                  f"makespan={max(L):.1f} m", flush=True)
            log.append(dict(expand=sweep + 1, captured=claimed_capt,
                            total=sum(L), makespan=max(L)))
            if a0 + a1 == 0:
                break

        # Post-expand shorten round: convert whatever route the captures did
        # not spend back into makespan (keep set is re-anchored to the
        # verified coverage, so this cannot undo the gains).
        prev_total = sum(lengths_now())
        stall = 0
        for sweep in range(args.eb_sweeps):
            sweep_step(0)
            sweep_step(1)
            L = lengths_now()
            imp = prev_total - sum(L)
            prev_total = sum(L)
            if (sweep + 1) % 5 == 0 or sweep == 0:
                print(f"    [{label} tighten {sweep+1:3d}] total={sum(L):.1f} m "
                      f"makespan={max(L):.1f} m (-{imp:.2f} m/sweep)",
                      flush=True)
            log.append(dict(tighten=sweep + 1, total=sum(L), makespan=max(L)))
            stall = stall + 1 if imp < args.eb_tol else 0
            if stall >= 2:
                break
        if not args.no_prune:
            npr, gt_vis = _prune_pass(pos, rot, view, homes, layer, pts_t,
                                      esdf, args, keep_idx, req, times,
                                      gt_vis=gt_vis)
            n_pruned_total += npr
            if npr:
                L = lengths_now()
                print(f"    [{label}] post-expand prune: removed {npr} poses "
                      f"-> total={sum(L):.1f} m makespan={max(L):.1f} m")

    wall = time.perf_counter() - t0
    if gt_vis is not None:
        print(f"    [{label}] audit gate: {audit_count[0]} poses ray-cast "
              f"during optimization (pipeline planning ray-casts 600)")
    return pos, rot, dict(wall_s=wall, trace=log, n_pruned=n_pruned_total,
                          n_vias=n_vias, view=view,
                          claimed_captures=claimed_capt,
                          audited_poses=audit_count[0])


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


def evaluate(tag, query, esdf_np_fn, homes_t, pos_list, rot_list, args,
             view_list=None):
    """Score by the pipeline's own machinery. Coverage uses only VIEW poses
    (via waypoints are path, not viewpoints — the analogue of ST-A* detour
    points); lengths and clearance use the full chains including vias."""
    R = len(pos_list)
    if view_list is None:
        view_list = [torch.ones(len(p), dtype=torch.bool, device=p.device)
                     for p in pos_list]
    vpos = torch.cat([pos_list[r][view_list[r]] for r in range(R)])
    vrot = torch.cat([rot_list[r][view_list[r]] for r in range(R)])
    cov, _ = hard_coverage(query, vpos.cpu().numpy(), sixd_to_rotmats(vrot))
    chains = [chain(homes_t[r], pos_list[r]) for r in range(R)]
    lengths = [float(chain_lengths(c).sum()) for c in chains]
    dense = torch.cat([segment_samples(c, 40) for c in chains])
    min_clear = float(esdf_np_fn(dense).min())
    return dict(tag=tag, coverage=cov, makespan=max(lengths),
                total_length=sum(lengths), per_robot=lengths,
                min_clearance=min_clear, n_poses=int(vpos.shape[0]),
                n_vias=sum(len(p) for p in pos_list) - int(vpos.shape[0]))


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
    p.add_argument("--solver", choices=["elastic", "sgd"], default="elastic",
                   help="Warm-arm inner solver: elastic = coordinate-descent "
                        "elastic band (§9.6 design); sgd = the §9.5 "
                        "simultaneous-descent formulation (kept for "
                        "comparison).")
    p.add_argument("--eb_sweeps", type=int, default=40,
                   help="Max full sweeps (both parities) per elastic round.")
    p.add_argument("--eb_inner", type=int, default=25,
                   help="Penalty inner-loop steps per half-sweep.")
    p.add_argument("--eb_lr", type=float, default=0.02)
    p.add_argument("--eb_mu", type=float, default=200.0,
                   help="Penalty weight in the elastic inner loop (the "
                        "acceptance test is the hard gate, the penalty only "
                        "shapes the local solve).")
    p.add_argument("--eb_slack", type=float, default=0.02,
                   help="Extra metric slack (m) targeted by the inner loop "
                        "beyond req, so accepted poses clear the acceptance "
                        "threshold with room.")
    p.add_argument("--eb_tol", type=float, default=0.02,
                   help="Stop sweeping after 2 consecutive sweeps improving "
                        "total length by less than this (m).")
    p.add_argument("--eb_rounds", type=int, default=3,
                   help="Max alternations of sweep-until-stall and prune.")
    p.add_argument("--no_prune", action="store_true",
                   help="Disable the redundant-pose prune pass.")
    p.add_argument("--no_expand", action="store_true",
                   help="Disable the coverage-expansion phase.")
    p.add_argument("--eb_expand_sweeps", type=int, default=30)
    p.add_argument("--eb_debug", action="store_true")
    p.add_argument("--eb_trust", type=float, default=0.4,
                   help="Expand trust region: max waypoint displacement "
                        "(m) per half-sweep.")
    p.add_argument("--eb_trust_rot", type=float, default=0.25,
                   help="Expand trust region for 6D rotation per "
                        "half-sweep.")
    p.add_argument("--eb_keep_soft", type=float, default=0.05,
                   help="Keep-penalty weight multiplier during expand (keep\n                        "
                        "is guidance there; the GT gate arbitrates trades).")
    p.add_argument("--eb_repair", type=int, default=25,
                   help="Constraint-only projection steps after each expand "
                        "inner loop (returns movers to the feasible pocket "
                        "before the acceptance test).")
    p.add_argument("--eb_gain_sharp", type=float, default=12.0,
                   help="Sigmoid sharpness (1/m) of the capture reward on "
                        "uncovered-point metric margins.")
    p.add_argument("--eb_w_gain", type=float, default=1.0)
    p.add_argument("--eb_route_reg", type=float, default=0.3,
                   help="Route weight during the expand phase (length is "
                        "governed by the per-robot budget constraint; this "
                        "only keeps chains taut).")
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
        if args.solver == "elastic":
            # Reference arm: the VRP waypoints joined by straight segments,
            # before any optimization — so the elastic solver's gain is
            # attributable to sliding/pruning, not to merely straightening
            # the pipeline's ST-A* detours.
            res0 = evaluate("warm-init", query, esdf, homes_t,
                            [p.clone() for p in route_pos],
                            [r.clone() for r in route_rot], args)
            res0["wall_s"] = 0.0
            results.append(res0)
            print(f"    warm-init (straight chains, unoptimized): "
                  f"cov={res0['coverage']:.4f} "
                  f"makespan={res0['makespan']:.1f} m "
                  f"clear={res0['min_clearance']:.2f} m")
            print(f"[5] WARM joint solve (elastic-band coordinate descent) …")
            # The elastic solver's coverage invariant protects EVERY ray-cast
            # covered point at min(keep_margin, its init margin) — feasible at
            # init by construction, so no agree-filter is needed (that filter
            # existed to keep the AL from blowing up on constraints the
            # surrogate could never satisfy).
            keep_idx_full = torch.tensor(np.nonzero(gt_union)[0],
                                         dtype=torch.long, device=device)
            uncov_idx = torch.tensor(np.nonzero(~gt_union)[0],
                                     dtype=torch.long, device=device)

            def audit_fn(p, r6):
                """Pipeline ray-cast rows for arbitrary poses (bool (n, N))."""
                Vb, _ = query.compute_visibility_batch(
                    cp.asarray(p.detach().cpu().numpy(), dtype=cp.float32),
                    cp.asarray(sixd_to_rotmats(r6), dtype=cp.float32))
                return torch.tensor(cp.asnumpy(Vb).astype(bool),
                                    device=device)

            pos_w, rot_w, info_w = optimize_elastic(
                pts_t, layer, homes_t, route_pos, route_rot, esdf, args,
                keep_idx_full, label="warm", uncovered_idx=uncov_idx,
                budgets=exec_lengths, audit=audit_fn)
            view_w = info_w["view"]
        else:
            print(f"[5] WARM joint solve ({args.steps} steps) …")
            pos_w, rot_w, info_w = optimize_joint(
                pts_t, layer, homes_t, route_pos, route_rot, esdf, args,
                "warm", keep_pairs=keep_pairs)
            view_w = None
        res = evaluate("joint-warm", query, esdf, homes_t, pos_w, rot_w, args,
                       view_list=view_w)
        res["wall_s"] = info_w["wall_s"]
        for k in ("n_pruned", "n_vias", "claimed_captures"):
            if k in info_w:
                res[k] = info_w[k]
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
    marks = {"pipeline": "s", "warm-init": "D", "joint-warm": "o",
             "joint-cold": "^"}
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
