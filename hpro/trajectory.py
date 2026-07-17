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

.. warning::

   **The rollout is chaotic: never report a single run.** A rollout is
   reproducible within one process but *not across processes* — the same config
   and seed produced GT coverage 0.563 in 7 of 8 processes and 0.933 in the 8th
   (§3.4). The closed loop amplifies ~1e-7 GPU float differences (per-process
   kernel/TF32 selection): a marginally different pose flips points across the
   hard ray-cast audit threshold, which changes the harvested demand, which
   changes every subsequent cycle. The outcome is **bimodal** — the planner
   either escapes its first basin or stalls in it — so a single number is a
   coin flip, not a measurement. Always aggregate over repeats
   (``eval_trajectory.py --repeats``) and report mean ± std with the number of
   runs.

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
        use_global_guide: enable the discrete global pass (§6.4 item 4). When
            on, the terminal attractor is the current *target cluster* chosen by
            :class:`GlobalGuide` (greedy NN + 2-opt over demand clusters) rather
            than the nearest surviving demand — which is what un-sticks the
            planner from a locally-exhausted region.
        guide_pts_per_cluster: aim for roughly this many demand points per
            cluster when choosing k.
        guide_k_max: cap on the number of demand clusters.
        guide_patience: consecutive low-harvest cycles before the current
            target region is declared stalled and deferred.
        guide_min_new: a cycle harvesting fewer new points than this counts as
            "low-harvest" for the stall detector.
        guide_defer_cycles: how long a stalled target's points are excluded
            from re-targeting (they get another chance later, presumably from a
            different approach direction).
        guide_mass_min: a target is exhausted (and re-targeting triggered) when
            its remaining demand mass drops below this many points.
        retarget_steps: gradient steps on the cycle where the guide commits a
            new target. The horizon is re-initialised toward the new target on
            that cycle (see :func:`horizon_toward`), so it needs more than the
            warm ``replan_steps`` budget but less than a cold start.
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
    use_global_guide: bool = True
    guide_pts_per_cluster: int = 250
    guide_k_max: int = 8
    guide_patience: int = 3
    guide_min_new: int = 5
    guide_defer_cycles: int = 8
    guide_mass_min: float = 5.0
    retarget_steps: int = 90


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
    # Per-executed-pose audit records, kept so a rollout can be replayed and
    # inspected visually (viz_trajectory.py) rather than trusted from a scalar:
    seen_per_pose: list = field(default_factory=list)    # list[set[int]] actually seen
    pred_per_pose: list = field(default_factory=list)    # list[set[int]] surrogate w>0.5
    guide_retargets: int = 0      # times the global guide committed a new target
    guide_deferrals: int = 0      # times a stalled target region was deferred


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
    locally-exhausted region. This term is the continuous end of the fix; on its
    own it is a single attractor and demonstrably not enough (the rollout
    stalled at ~0.05 m/cycle, §3.4 item 2). :class:`GlobalGuide` supplies the
    missing discrete half of §6.4 item 4 by masking ``demand`` to a committed
    target region, so the same expression becomes "go *there* next" instead of
    "hover near the closest leftover".

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
# Two-timescale global guide (§6.4 item 4)
# ---------------------------------------------------------------------------

def _kmeans(pts: np.ndarray, k: int, rng: np.random.Generator,
            n_iters: int = 20) -> np.ndarray:
    """Deterministic Lloyd k-means with farthest-point init. Returns labels (n,).

    Farthest-point init instead of k-means++ so the result depends only on the
    rng state, not on sampling luck — the guide must not add a *new* source of
    run-to-run variance to a rollout that is already chaotic (§3.4 item 1).
    """
    n = len(pts)
    k = min(k, n)
    centers = np.empty((k, 3))
    centers[0] = pts[rng.integers(n)]
    d2 = ((pts - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        centers[i] = pts[int(np.argmax(d2))]
        d2 = np.minimum(d2, ((pts - centers[i]) ** 2).sum(axis=1))
    labels = np.full(n, -1, dtype=np.int64)
    for _it in range(n_iters):
        d = ((pts[:, None] - centers[None]) ** 2).sum(axis=2)   # (n, k)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            m = labels == i
            if m.any():
                centers[i] = pts[m].mean(axis=0)
    return labels


def _tour_nn_2opt(start: np.ndarray, centroids: np.ndarray) -> list:
    """Greedy nearest-neighbour + 2-opt open tour over cluster centroids.

    The same routing the discrete baseline gets (eval_trajectory.route_nn_2opt),
    scaled down to a handful of centroids — this *is* the "cheap discrete global
    pass" of §6.4 item 4, so it deliberately mirrors the C3 hybrid's anchor.
    """
    n = len(centroids)
    if n == 0:
        return []
    pts = np.vstack([start[None], centroids])
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    unvisited = set(range(1, n + 1))
    tour, cur = [], 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    def tour_len(t):
        seq = [0] + t
        return float(D[seq[:-1], seq[1:]].sum())

    best, improved = tour_len(tour), True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                L = tour_len(cand)
                if L < best - 1e-9:
                    tour, best, improved = cand, L, True
    return [t - 1 for t in tour]


class GlobalGuide:
    """The discrete half of the two-timescale hierarchy (§6.4 item 4).

    Clusters the remaining demand, orders the clusters by greedy NN + 2-opt
    from the robot, and commits to the tour's first cluster as the **target
    region**. The differentiable optimiser then refines the local horizon
    against a terminal cost that pulls toward *that region only* — a real visit
    order, not the single nearest-demand attractor that let the planner orbit a
    locally-exhausted (or genuinely uncoverable) patch forever.

    Two rules make it robust rather than merely global:

    * **Hysteresis** — the target is kept until it is exhausted
      (mass < ``guide_mass_min``) or stalled, so per-cycle re-clustering cannot
      make the attractor ping-pong between two clusters.
    * **Stall deferral** — a target that stops making progress is deferred for
      ``guide_defer_cycles`` cycles and the tour moves on. This is the explicit
      "leave the exhausted region" escape: some points (deep concavities,
      interior surfaces) may be uncoverable from the standoff band, and without
      deferral they are a tar pit. Crucially, "progress" is judged by *phase*:
      while **engaged** (within ``far_dist`` of the target) the signal is new
      points harvested; while **in transit** it is whether the robot is still
      closing distance — low harvest during travel is normal and must not count
      (the first implementation counted it, and churned through 8 deferrals in
      40 cycles without ever arriving anywhere). Transit stalls accumulate at
      half weight, since crossing the structure takes several cycles.
    """

    def __init__(self, pts_np: np.ndarray, cfg: TrajectoryConfig,
                 rng: np.random.Generator):
        self.pts = np.asarray(pts_np, dtype=np.float64)
        self.cfg = cfg
        self.rng = rng
        self.N = len(pts_np)
        self.deferred_until = np.zeros(self.N, dtype=np.int64)
        self.target_idx: Optional[np.ndarray] = None
        self.stall_score = 0.0
        self.engage_dist = cfg.far_dist
        self._last_new: Optional[int] = None
        self._prev_dist: Optional[float] = None
        self.retargets = 0          # diagnostics
        self.deferrals = 0          # diagnostics

    def _retarget(self, demand_np: np.ndarray, robot_pos: np.ndarray,
                  cycle: int) -> None:
        active = (demand_np > 0.5) & (self.deferred_until <= cycle)
        if not active.any():
            # Everything left is deferred: clear deferrals rather than idle.
            self.deferred_until[:] = 0
            active = demand_np > 0.5
            if not active.any():
                self.target_idx = None
                return
        idx = np.nonzero(active)[0]
        k = int(np.clip(round(len(idx) / self.cfg.guide_pts_per_cluster),
                        1, self.cfg.guide_k_max))
        labels = _kmeans(self.pts[idx], k, self.rng)
        uniq = np.unique(labels)          # Lloyd can leave clusters empty
        centroids = np.stack([self.pts[idx[labels == u]].mean(axis=0)
                              for u in uniq])
        tour = _tour_nn_2opt(robot_pos, centroids)
        self.target_idx = idx[labels == uniq[tour[0]]]
        self.stall_score = 0.0
        self._prev_dist = None
        self.retargets += 1

    def select(self, demand_np: np.ndarray, robot_pos: np.ndarray,
               cycle: int) -> Optional[np.ndarray]:
        """Return the indices of the current target region (None = no demand)."""
        # Assess the last executed cycle (see class docstring): harvesting new
        # points OR closing distance to the target is progress; anything else
        # accumulates stall — at half weight beyond engage range, because a
        # long transit leg legitimately alternates good and mediocre cycles.
        if self.target_idx is not None and self._last_new is not None:
            d = float(np.linalg.norm(self.pts[self.target_idx] - robot_pos,
                                     axis=1).min())
            closing = (self._prev_dist is not None
                       and d < self._prev_dist - 0.25 * self.cfg.step_size)
            if closing or self._last_new >= self.cfg.guide_min_new:
                self.stall_score = 0.0
            else:
                self.stall_score += 1.0 if d <= self.engage_dist else 0.5
            self._prev_dist = d
            self._last_new = None

        target_mass = (0.0 if self.target_idx is None
                       else float(demand_np[self.target_idx].sum()))
        if self.target_idx is not None and \
                self.stall_score >= self.cfg.guide_patience:
            # Stalled on this target: defer its surviving points and move on.
            survivors = self.target_idx[demand_np[self.target_idx] > 0.5]
            self.deferred_until[survivors] = cycle + self.cfg.guide_defer_cycles
            self.deferrals += 1
            self._retarget(demand_np, robot_pos, cycle)
        elif target_mass < self.cfg.guide_mass_min:
            self._retarget(demand_np, robot_pos, cycle)
        return self.target_idx

    def report_harvest(self, n_new: int) -> None:
        """Feed back how many new points the executed pose actually covered."""
        self._last_new = n_new


def horizon_toward(start: torch.Tensor, target_pts: np.ndarray,
                   cfg: TrajectoryConfig, device) -> tuple:
    """Re-initialise the horizon as a straight polyline toward a target region.

    Warm-starting is the right default *within* the pursuit of one target, but
    across a retarget the inherited horizon is precisely the local minimum the
    guide is trying to escape: a knot of poses in the exhausted region that
    ``replan_steps`` Adam steps cannot unfold against the length/step penalties
    (measured: the robot crawled at 0.02–0.1 m/cycle through nine retargets).
    So on retarget the *plan* — never the robot — teleports: poses are laid out
    from the current position toward a view point at standoff from the target
    centroid, spaced at most ``step_size`` apart, each looking at the target.
    The next optimisation bends this polyline around obstacles via the standoff
    penalty and repairs its spacing.

    Returns:
        (init_pos (H, 3), init_rot (H, 6)) on ``device``.
    """
    H = cfg.horizon
    standoff = 0.5 * (cfg.near_dist + cfg.far_dist)
    c_t = torch.tensor(target_pts.mean(axis=0), dtype=torch.float32,
                       device=device)
    d = c_t - start
    dist = float(d.norm())
    if dist < 1e-9:
        d = torch.tensor([1.0, 0.0, 0.0], device=device)
        dist = 1.0
    view = c_t - d / dist * standoff                     # approach-side view point
    seg = view - start
    L = float(seg.norm())
    # March toward the view point without exceeding the per-step motion cap.
    reach = min(L, H * cfg.step_size)
    steps = torch.arange(1, H + 1, dtype=torch.float32,
                         device=device).unsqueeze(1) / H
    init_pos = start.unsqueeze(0) + seg / max(L, 1e-9) * reach * steps
    look = c_t.unsqueeze(0) - init_pos
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32,
                      device=device).expand(H, 3).clone()
    # Fall back where gaze is nearly collinear with world up.
    ln = F.normalize(look, dim=1)
    bad = ln[:, 2].abs() > 0.95
    up[bad] = torch.tensor([0.0, 1.0, 0.0], device=device)
    return init_pos, torch.cat([look, up], dim=1)


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
    demand_terminal: Optional[torch.Tensor] = None,
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
        demand_terminal: (N,) demand the *terminal* attractor sees. The global
            guide passes demand masked to the current target region here, while
            coverage still sees the full demand (opportunistic harvesting along
            the way stays free). Defaults to ``demand``.

    Returns:
        (positions (H,3), rot_6d (H,6), soft_coverage float) — detached.
    """
    if demand_terminal is None:
        demand_terminal = demand
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
                pos, pts_surface, demand_terminal,
                0.5 * (cfg.near_dist + cfg.far_dist))
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
    guide = GlobalGuide(pts_np, cfg, rng) if cfg.use_global_guide else None
    prev_retargets = 0

    for cycle in range(cfg.max_cycles):
        # ---- Slow timescale: discrete global pass over demand clusters ----
        demand_terminal = None
        retargeted = False
        if guide is not None:
            target = guide.select(demand.cpu().numpy(),
                                  start.cpu().numpy(), cycle)
            if target is not None:
                mask = torch.zeros_like(demand)
                mask[torch.as_tensor(target, dtype=torch.long,
                                     device=device)] = 1.0
                demand_terminal = demand * mask
            retargeted = guide.retargets != prev_retargets
            prev_retargets = guide.retargets
            if retargeted and cycle > 0 and target is not None:
                # The plan teleports, the robot does not: re-seed the horizon
                # toward the new target instead of unfolding the old knot.
                init_pos, init_rot = horizon_toward(
                    start, pts_np[target], cfg, device)

        if cycle == 0:
            n_steps = cfg.init_steps
        elif retargeted:
            n_steps = cfg.retarget_steps
        else:
            n_steps = cfg.replan_steps
        t0 = time.perf_counter()
        pos, rot, soft_c = optimize_horizon(
            layer, pts_t, pts_surface, demand, start, init_pos, init_rot,
            cfg, n_steps, demand_terminal=demand_terminal)
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
        pred_idx = torch.nonzero(w0 > 0.5).squeeze(-1).cpu().numpy()
        result.audit_gap.append(float(len(pred_idx)) - float(len(seen)))
        result.pred_per_pose.append(set(int(i) for i in pred_idx))
        result.seen_per_pose.append(set(seen))

        # ---- Demand update: decay, never delete --------------------------
        if new:
            idx = torch.tensor(sorted(new), dtype=torch.long, device=device)
            demand[idx] = 0.0
        if guide is not None:
            guide.report_harvest(len(new))

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

    if guide is not None:
        result.guide_retargets = guide.retargets
        result.guide_deferrals = guide.deferrals
    result.positions = np.array(exec_pos)
    result.rot_6d = np.array(exec_rot)
    result.gt_coverage = len(covered) / N
    result.path_length = total_len
    result.n_cycles = len(exec_pos)
    result.wall_time_s = time.perf_counter() - t_start
    return result
