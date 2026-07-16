"""Stage 3 / C4 — receding-horizon differentiable inspection trajectories.

Where ``multi_viewpoint.py`` optimises an unordered *set* of V poses, this
module optimises an ordered *path*: a horizon of H future poses whose motion cost
counts, executed one pose at a time. That is the difference between camera
placement (NeOF, RA-L 2024, and Moraza et al., VISAPP 2026 — see §5.2 of
RESEARCH_PLAN.md) and inspection *planning*, and it is where this work is not
already claimed.

The loop, per §6.4::

    while demand remains and budget allows:
        tail = optimize(tail, k gradient steps)      # warm-started from last cycle
        execute(tail[0])                             # commit the first pose
        harvest(tail[0])                             # hard ray-cast audit
        demand[covered] -> 0                         # smooth, not a deletion
        tail = shift(tail)                           # slide the horizon forward

Two design choices carry the argument and are worth stating plainly:

**Demand weighting, not point removal.** Deleting covered points changes the loss
discontinuously and destroys the warm start. Residual demand ``d_j`` decays to 0
instead, so covered points *smoothly* stop attracting the trajectory and the
previous solution stays a good initialisation for the next cycle.

**Warm-starting is the whole point.** A discrete set-cover + VRP pipeline must
re-solve from scratch when the world changes, and NeOF must refit its neural
field; a gradient optimiser on a fitting-free surrogate keeps stepping from where
it was. That is what makes the planner anytime and incremental, and it is the
property the paper's headline rests on — so ``replan_steps`` is deliberately
small (a per-cycle budget), not "optimise to convergence".

The audit that harvests coverage is a hard ray-cast against the mesh, which
doubles as C2's re-anchoring signal: ``audit_gap`` records surrogate-vs-truth
disagreement per cycle for free.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from visibility_layer import GatedVisibilityLayer  # noqa: E402


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryConfig:
    """Knobs for :func:`receding_horizon_plan`.

    Attributes:
        horizon: H, number of future poses optimised jointly each cycle.
        replan_steps: gradient steps per cycle. Small on purpose — the claim is
            that a warm start makes a handful of steps enough.
        init_steps: gradient steps for the very first cycle, which has no warm
            start to inherit and must find the trajectory from scratch.
        lr: Adam learning rate for pose parameters.
        max_cycles: hard cap on executed poses.
        target_coverage: stop once the audited GT coverage reaches this.
        lambda_length: weight on path length. The trade-off knob: 0 recovers
            pure coverage (a teleporting camera), large values freeze the robot.
        lambda_smooth: weight on the discrete second difference of positions
            (curvature), keeping the committed path flyable.
        lambda_standoff: weight on the near/far standoff band. Doubles as the
            collision proxy: near_dist keeps the camera off the surface.
        lambda_terminal: weight on the terminal cost pulling the horizon's last
            pose toward the remaining demand (the two-timescale myopia fix).
            Must be strong enough to drag the robot out of a locally-exhausted
            region against lambda_length, or the rollout stalls.
        near_dist, far_dist: standoff band (m), matched to the camera clip planes.
        step_size: max distance the robot may move between consecutive poses.
        seed: RNG seed.
    """
    horizon: int = 6
    replan_steps: int = 30
    init_steps: int = 150
    lr: float = 3e-2
    max_cycles: int = 40
    target_coverage: float = 0.95
    # Tuned on the wreck, seed 0, NVPS backbone (see RESEARCH_PLAN.md §3.4).
    # The original lambda_length=0.05 / lambda_terminal=0.02 made the robot
    # stall: once local demand is exhausted the coverage gradient vanishes, and
    # the length penalty (0.05 x ~3.6 m = 0.18) outvoted the terminal pull
    # (0.02 x ~4 = 0.08) -- the only term that can steer toward fresh geometry.
    # The rollout then crawled at ~0.05 m/cycle and finished at 0.53 coverage.
    # These weights are sensitive and non-monotone; treat them as a starting
    # point, not a converged choice.
    lambda_length: float = 0.01
    lambda_smooth: float = 0.01
    lambda_standoff: float = 1.0
    lambda_terminal: float = 5.0
    near_dist: float = 0.5
    far_dist: float = 1.5
    step_size: float = 0.6
    seed: int = 0


@dataclass
class TrajectoryResult:
    """Outputs of a receding-horizon rollout."""
    positions: np.ndarray = None          # (T, 3) executed poses
    rot_6d: np.ndarray = None             # (T, 6) executed orientations
    gt_coverage: float = 0.0              # audited union coverage in [0, 1]
    path_length: float = 0.0              # metres travelled
    coverage_curve: list = field(default_factory=list)   # GT coverage after each pose
    length_curve: list = field(default_factory=list)     # cumulative length
    audit_gap: list = field(default_factory=list)        # soft - GT per cycle (C2)
    replan_times: list = field(default_factory=list)     # seconds per cycle
    n_cycles: int = 0
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Loss terms
# ---------------------------------------------------------------------------

def demand_weighted_coverage(w: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """Soft set-cover over a horizon, weighted by residual demand.

    ``C = Σ_j d_j·[1 − Π_s (1 − w[s,j])] / Σ_j d_j``

    Args:
        w: (H, N) visibility scores of the horizon's poses.
        demand: (N,) residual demand in [0, 1]; 0 = already covered.

    Returns:
        Scalar in [0, 1]. Points with d_j = 0 contribute no gradient, which is
        how covered points stop attracting the trajectory without being deleted.
    """
    w_c = w.clamp(0.0, 1.0 - 1e-7)
    covered = 1.0 - torch.prod(1.0 - w_c, dim=0)      # (N,)
    total = demand.sum()
    if float(total) <= 0.0:
        return covered.new_zeros(())
    return (demand * covered).sum() / total


def path_length(positions: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """Total length of ``start -> positions[0] -> ... -> positions[H-1]``."""
    pts = torch.cat([start.unsqueeze(0), positions], dim=0)   # (H+1, 3)
    return (pts[1:] - pts[:-1]).norm(dim=1).sum()


def smoothness(positions: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """Sum of squared second differences (discrete curvature) along the path."""
    pts = torch.cat([start.unsqueeze(0), positions], dim=0)   # (H+1, 3)
    if pts.shape[0] < 3:
        return pts.new_zeros(())
    accel = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
    return accel.pow(2).sum()


def standoff_penalty(positions: torch.Tensor, pts_surface: torch.Tensor,
                     near_dist: float, far_dist: float) -> torch.Tensor:
    """Penalise poses outside the [near, far] band around the surface.

    The near side is also the collision proxy: it keeps the camera from entering
    the structure. A proper occupancy-grid SDF (§6.4) can replace this later
    without touching the rest of the loss.
    """
    d_min = torch.cdist(positions, pts_surface).min(dim=1).values   # (H,)
    return (F.relu(near_dist - d_min).pow(2) + F.relu(d_min - far_dist).pow(2)).sum()


def step_penalty(positions: torch.Tensor, start: torch.Tensor,
                 step_size: float) -> torch.Tensor:
    """Penalise consecutive poses further apart than ``step_size``.

    Keeps the committed segment executable at the robot's speed and stops the
    optimiser from producing a teleporting camera that trivially covers
    everything.
    """
    pts = torch.cat([start.unsqueeze(0), positions], dim=0)
    seg = (pts[1:] - pts[:-1]).norm(dim=1)
    return F.relu(seg - step_size).pow(2).sum()


def terminal_cost(positions: torch.Tensor, pts_surface: torch.Tensor,
                  demand: torch.Tensor, standoff_target: float,
                  tau: float = 0.1) -> torch.Tensor:
    """Pull the horizon's final pose to standoff range of the nearest *uncovered* point.

    Coverage inside a short horizon is myopic: it has no reason to leave a
    locally-exhausted region. This is the cheap global signal of §6.4 item 4 -- a
    single attractor rather than a full greedy/TSP pass over demand clusters,
    which is the natural upgrade if myopia persists.

    Note the attractor is the nearest *surviving demand*, not the demand
    centroid: for a closed surface the centroid lies **inside the object**, so
    pulling toward it drags the camera into the hull and deadlocks against the
    standoff penalty. That is not hypothetical -- it is what made the HPRO
    rollout freeze at 0.46 coverage for 18 consecutive cycles.

    Args:
        positions: (H, 3) horizon positions.
        pts_surface: (N, 3) surface points.
        demand: (N,) residual demand; satisfied points are excluded.
        standoff_target: distance to aim for from that point (m).
        tau: soft-min temperature (m). Smaller = a harder "nearest" choice.
    """
    total = demand.sum()
    if float(total) <= 0.0:
        return positions.new_zeros(())

    d = torch.cdist(positions[-1:], pts_surface)[0]          # (N,)
    # Push satisfied points out of the soft-min rather than masking, so the
    # expression stays differentiable everywhere.
    d_masked = d + (1.0 - demand) * 1e6
    weights = torch.softmax(-d_masked / tau, dim=0)          # (N,)
    d_near = (weights * d).sum()
    return (d_near - standoff_target).pow(2)


# ---------------------------------------------------------------------------
# Horizon optimisation
# ---------------------------------------------------------------------------

def optimize_horizon(
    layer: GatedVisibilityLayer,
    pts_t: torch.Tensor,
    pts_surface: torch.Tensor,
    demand: torch.Tensor,
    start: torch.Tensor,
    init_pos: torch.Tensor,
    init_rot: torch.Tensor,
    cfg: TrajectoryConfig,
    n_steps: int,
) -> tuple:
    """Optimise one horizon of poses by gradient descent.

    Args:
        layer: frustum-gated visibility over any backbone.
        pts_t: (3, N) cloud on device.
        pts_surface: (N, 3) same cloud, for distance terms.
        demand: (N,) residual demand.
        start: (3,) current robot position — the path is measured from here.
        init_pos: (H, 3) warm-started positions.
        init_rot: (H, 6) warm-started orientations.
        cfg: config.
        n_steps: gradient steps (the per-cycle budget).

    Returns:
        (positions (H,3), rot_6d (H,6), soft_coverage float) — detached.
    """
    pos = nn.Parameter(init_pos.clone())
    rot = nn.Parameter(init_rot.clone())
    opt = torch.optim.Adam([pos, rot], lr=cfg.lr)

    soft_c = 0.0
    for _ in range(n_steps):
        opt.zero_grad()
        w = layer(pts_t, pos, rot)                       # (H, N)
        C = demand_weighted_coverage(w, demand)
        loss = (
            -C
            + cfg.lambda_length * path_length(pos, start)
            + cfg.lambda_smooth * smoothness(pos, start)
            + cfg.lambda_standoff * standoff_penalty(
                pos, pts_surface, cfg.near_dist, cfg.far_dist)
            + cfg.lambda_standoff * step_penalty(pos, start, cfg.step_size)
            + cfg.lambda_terminal * terminal_cost(
                pos, pts_surface, demand, 0.5 * (cfg.near_dist + cfg.far_dist))
        )
        loss.backward()
        opt.step()
        soft_c = float(C)

    return pos.detach(), rot.detach(), soft_c


# ---------------------------------------------------------------------------
# Receding-horizon rollout
# ---------------------------------------------------------------------------

def receding_horizon_plan(
    layer: GatedVisibilityLayer,
    pts_np: np.ndarray,
    audit_fn: Callable[[np.ndarray, np.ndarray], set],
    cfg: Optional[TrajectoryConfig] = None,
    start_pos: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> TrajectoryResult:
    """Plan an inspection trajectory by receding-horizon gradient descent.

    Args:
        layer: frustum-gated visibility layer (backbone already ``prepare``d).
        pts_np: (N, 3) target surface points.
        audit_fn: ``(position (3,), rot_6d (6,)) -> set[int]`` of point indices
            genuinely seen from that pose. Offline this is a ray-cast against the
            mesh; online it would be the sensor itself — the same signature, which
            is the symmetry §6.4 item 6 sells.
        cfg: config (defaults used if None).
        start_pos: (3,) initial robot position; defaults to a point on the
            far-plane sphere around the cloud's centroid.
        device: compute device; defaults to the layer's.

    Returns:
        TrajectoryResult with the executed path, audited coverage and per-cycle
        replan times.
    """
    cfg = cfg or TrajectoryConfig()
    device = device or layer.device
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    N = pts_np.shape[0]
    centroid = pts_np.mean(axis=0)
    pts_t = torch.tensor(pts_np.T, dtype=torch.float32, device=device)      # (3, N)
    pts_surface = torch.tensor(pts_np, dtype=torch.float32, device=device)  # (N, 3)
    demand = torch.ones(N, dtype=torch.float32, device=device)

    if start_pos is None:
        d0 = rng.normal(size=3)
        d0 /= np.linalg.norm(d0)
        start_pos = centroid + d0 * cfg.far_dist
    start = torch.tensor(start_pos, dtype=torch.float32, device=device)

    # Initial horizon: march outward from the start along the standoff shell,
    # each pose looking at the centroid. Cheap and deterministic; the first
    # cycle's init_steps then does the real work.
    H = cfg.horizon
    init_pos = torch.stack([start + torch.tensor(
        rng.normal(scale=0.3, size=3), dtype=torch.float32, device=device)
        for _ in range(H)])
    look = torch.tensor(centroid, dtype=torch.float32, device=device) - init_pos
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32,
                      device=device).expand(H, 3)
    init_rot = torch.cat([look, up], dim=1)

    result = TrajectoryResult(coverage_curve=[], length_curve=[],
                              audit_gap=[], replan_times=[])
    covered: set = set()
    exec_pos, exec_rot = [], []
    total_len = 0.0
    t_start = time.perf_counter()

    for cycle in range(cfg.max_cycles):
        n_steps = cfg.init_steps if cycle == 0 else cfg.replan_steps
        t0 = time.perf_counter()
        pos, rot, soft_c = optimize_horizon(
            layer, pts_t, pts_surface, demand, start, init_pos, init_rot,
            cfg, n_steps)
        result.replan_times.append(time.perf_counter() - t0)

        # ---- Commit and execute the first pose of the horizon -------------
        p0 = pos[0].cpu().numpy()
        r0 = rot[0].cpu().numpy()
        total_len += float(np.linalg.norm(p0 - start.cpu().numpy()))
        exec_pos.append(p0)
        exec_rot.append(r0)

        # ---- Hard audit: what did we *actually* see? ----------------------
        seen = audit_fn(p0, r0)
        new = seen - covered
        covered |= seen

        # Surrogate-vs-truth gap at the executed pose (C2 re-anchoring signal).
        with torch.no_grad():
            w0 = layer(pts_t, pos[:1], rot[:1])[0]
        pred = float((w0 > 0.5).sum())
        result.audit_gap.append(pred - float(len(seen)))

        # ---- Demand update: decay, never delete --------------------------
        if new:
            idx = torch.tensor(sorted(new), dtype=torch.long, device=device)
            demand[idx] = 0.0

        result.coverage_curve.append(len(covered) / N)
        result.length_curve.append(total_len)

        # ---- Slide the horizon: this is the warm start --------------------
        start = pos[0]
        init_pos = torch.cat([pos[1:], pos[-1:].clone()], dim=0)
        init_rot = torch.cat([rot[1:], rot[-1:].clone()], dim=0)

        if len(covered) / N >= cfg.target_coverage:
            break
        if float(demand.sum()) <= 0.0:
            break

    result.positions = np.array(exec_pos)
    result.rot_6d = np.array(exec_rot)
    result.gt_coverage = len(covered) / N
    result.path_length = total_len
    result.n_cycles = len(exec_pos)
    result.wall_time_s = time.perf_counter() - t_start
    return result
