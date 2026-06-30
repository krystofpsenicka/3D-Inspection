"""Stage 2 — Joint multi-viewpoint differentiable visibility optimiser.

Jointly optimises V camera positions and orientations to maximise soft
set-cover coverage using HPRO_limited's differentiable visibility scores:

    C = (1/N) * Σ_j [ 1 − Π_{v=1}^{V} (1 − w_combined[v, j]) ]

Two regularising penalty terms keep the solution physically valid:

* **Standoff penalty** — each viewpoint must be between ``near_dist`` and
  ``far_dist`` metres from the nearest surface point (matches the camera's
  near/far clip planes so the object is always in frame).
* **Diversity penalty** — prevents all viewpoints collapsing to the same
  location by penalising pairs closer than ``min_inter_dist``.

dtype note: this module uses ``torch.float32`` by default (halving the
memory of the O(V×N²) HPRO inner-product matrix compared with the float64
used in Stage 1).  The HPRO_limited model has no learnable weight tensors,
so the output dtype matches the input tensors automatically.
"""

import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the hpro/ directory is on sys.path regardless of where this module
# is imported from (project root, hpro/ subdir, or installed package).
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from HPRO_limited import HPRO_limited  # noqa: E402


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiViewOptResult:
    """All outputs from a joint multi-viewpoint optimisation run."""
    viewpoints_np: np.ndarray          # (V, 3) final camera positions
    rot_6d_np: np.ndarray              # (V, 6) final 6D orientations (Zhou 2019)
    loss_history: list = field(default_factory=list)
    coverage_history: list = field(default_factory=list)    # soft C ∈ [0, 1]
    standoff_history: list = field(default_factory=list)
    diversity_history: list = field(default_factory=list)
    steps_log: list = field(default_factory=list)
    wall_time_s: float = 0.0
    n_points: int = 0
    n_viewpoints: int = 0


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def fibonacci_sphere_viewpoints(
    V: int,
    radius: float,
    centroid: np.ndarray,
    world_up: Optional[np.ndarray] = None,
) -> tuple:
    """
    Initialise V viewpoints on a Fibonacci sphere of given radius around
    ``centroid``, each oriented to look toward ``centroid``.

    Args:
        V: number of viewpoints.
        radius: sphere radius (metres); ``(near_dist + far_dist) / 2`` is a
            reasonable default so the object fills the frustum.
        centroid: (3,) object centroid; typically ``pts_np.mean(axis=0)``.
        world_up: (3,) preferred world-up vector (default ``[0, 0, 1]``).
            Automatically falls back to ``[0, 1, 0]`` for viewpoints whose
            look direction is nearly collinear with world_up.

    Returns:
        positions (V, 3) — viewpoint XYZ on the sphere.
        rot_6d   (V, 6) — 6D rotation representation [a1 | a2] where a1 is
            the unnormalised look direction and a2 is the unnormalised up
            approximation.  Gram-Schmidt is applied inside HPRO_limited.
    """
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    world_up = np.asarray(world_up, dtype=np.float64)

    # Deterministic Fibonacci sphere
    i = np.arange(V) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / V)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)   # (V, 3) unit vectors

    positions = centroid[np.newaxis] + radius * dirs  # (V, 3)

    rot_6d = np.zeros((V, 6), dtype=np.float64)
    fallback_up = np.array([0.0, 1.0, 0.0])
    for v in range(V):
        look_approx = centroid - positions[v]   # unnormalised, toward centroid
        look_norm = look_approx / (np.linalg.norm(look_approx) + 1e-12)
        up = world_up if abs(np.dot(look_norm, world_up)) < 0.95 else fallback_up
        rot_6d[v, :3] = look_approx
        rot_6d[v, 3:] = up

    return positions, rot_6d


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def compute_coverage_soft(w_combined: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft set-cover coverage averaged over N points.

    Args:
        w_combined: (V, N) scores in [0, 1], one row per viewpoint.

    Returns:
        C: scalar in [0, 1].  Gradient flows through every entry of
           ``w_combined`` that is strictly between 0 and 1.
    """
    # Clamp to open (0, 1) so the product gradient is never exactly zero at
    # perfect coverage (w=1 would give 0 multiplicative contribution and kill
    # the gradient for other viewpoints).
    w_c = w_combined.clamp(0.0, 1.0 - 1e-7)
    covered = 1.0 - torch.prod(1.0 - w_c, dim=0)   # (N,)
    return covered.mean()


def _standoff_penalty(
    viewpoints: torch.Tensor,
    pts_surface_t: torch.Tensor,
    near_dist: float,
    far_dist: float,
) -> torch.Tensor:
    """
    Penalise viewpoints outside the [near_dist, far_dist] standoff band.

    Args:
        viewpoints: (V, 3) current camera positions (learnable).
        pts_surface_t: (N, 3) surface points (constant, on the same device).
        near_dist: minimum distance to the surface (metres).
        far_dist: maximum distance to the surface (metres).

    Returns:
        Scalar penalty (≥ 0).
    """
    diff = viewpoints.unsqueeze(1) - pts_surface_t.unsqueeze(0)  # (V, N, 3)
    d_min = diff.norm(dim=2).min(dim=1).values                   # (V,)
    pen = F.relu(near_dist - d_min).pow(2) + F.relu(d_min - far_dist).pow(2)
    return pen.sum()


def _diversity_penalty(
    viewpoints: torch.Tensor,
    min_inter_dist: float,
) -> torch.Tensor:
    """
    Penalise pairs of viewpoints closer than ``min_inter_dist``.

    Args:
        viewpoints: (V, 3) current camera positions.
        min_inter_dist: minimum allowed pairwise separation (metres).

    Returns:
        Scalar penalty (≥ 0).
    """
    V = viewpoints.shape[0]
    if V < 2:
        return viewpoints.new_zeros(())
    diff = viewpoints.unsqueeze(0) - viewpoints.unsqueeze(1)   # (V, V, 3)
    dists = diff.norm(dim=2)                                    # (V, V)
    triu = torch.triu(
        torch.ones(V, V, dtype=torch.bool, device=viewpoints.device),
        diagonal=1,
    )
    pen = F.relu(min_inter_dist - dists[triu]).pow(2)
    return pen.sum()


# ---------------------------------------------------------------------------
# Main optimisation entry point
# ---------------------------------------------------------------------------

def optimize_viewpoints(
    pts_np: np.ndarray,
    V: int,
    model: HPRO_limited,
    *,
    gamma: float = -math.exp(-7.0),
    k: int = 10,
    n_steps: int = 500,
    lr: float = 1e-2,
    lambda_standoff: float = 1.0,
    lambda_diversity: float = 0.1,
    near_dist: float = 0.1,
    far_dist: float = 6.0,
    min_inter_dist: float = 0.3,
    init_radius: Optional[float] = None,
    init_viewpoints_np: Optional[np.ndarray] = None,
    init_rot_6d_np: Optional[np.ndarray] = None,
    log_every: int = 25,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> MultiViewOptResult:
    """
    Jointly optimise V viewpoints to maximise soft set-cover coverage.

    Uses Adam with cosine-annealed LR.  The HPRO_limited forward pass is
    called once per gradient step with B=V (all viewpoints in a single batch),
    requiring O(V × N²) memory on GPU.  A budget check is performed before
    the optimisation loop; pass ``fits_in_memory=False`` to the model
    constructor to trade memory for speed if the check warns.

    Args:
        pts_np: (N, 3) float64 surface points (numpy).
        V: number of viewpoints to optimise jointly.
        model: a pre-constructed ``HPRO_limited`` instance.
        gamma: HPRO radial-transform parameter.  Negative and close to 0;
            default ``-exp(-7) ≈ -9.1e-4`` is the Stage-1 optimal value.
        k: top-k parameter forwarded to HPRO.
        n_steps: number of Adam gradient steps.
        lr: initial Adam learning rate (cosine-annealed to lr×0.01).
        lambda_standoff: weight of the standoff penalty term.
        lambda_diversity: weight of the diversity penalty term.
        near_dist: minimum standoff distance to the surface (m).
        far_dist: maximum standoff distance to the surface (m).
        min_inter_dist: minimum inter-viewpoint separation (m).
        init_radius: Fibonacci-sphere radius for initialisation; defaults to
            ``(near_dist + far_dist) / 2``.
        init_viewpoints_np: (V, 3) override for initial positions.
        init_rot_6d_np: (V, 6) override for initial 6D orientations.
        log_every: print a progress line every this many steps.
        dtype: tensor dtype for all computations (default float32).
        seed: manual seed for reproducibility.

    Returns:
        MultiViewOptResult with final parameters and training histories.
    """
    torch.manual_seed(seed)
    device = model.device
    N = pts_np.shape[0]

    if init_radius is None:
        init_radius = (near_dist + far_dist) / 2.0

    # ---- Memory budget check -----------------------------------------------
    mem_gb = V * N * N * 4 / 1024 ** 3
    if torch.cuda.is_available() and device != "cpu":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if mem_gb > 0.75 * total_vram:
            raise ValueError(
                f"Estimated HPRO memory {mem_gb:.1f} GB exceeds 75 % of GPU VRAM "
                f"({total_vram:.1f} GB) at V={V}, N={N}.  Reduce V or N, or "
                f"construct the model with fits_in_memory=False (slower but O(V×N) memory)."
            )
        if mem_gb > 0.5 * total_vram:
            warnings.warn(
                f"HPRO memory estimate {mem_gb:.1f} GB is > 50 % of GPU VRAM "
                f"({total_vram:.1f} GB).  Consider fits_in_memory=False for large V×N.",
                stacklevel=2,
            )

    # ---- Precompute tensors ------------------------------------------------
    # (N, 3) for standoff penalty — note: (N, 3) not (3, N)
    pts_surface_t = torch.tensor(pts_np, dtype=dtype, device=device)

    # (V, 3, N) for HPRO_limited: same point cloud for all viewpoints.
    # .expand shares memory; the element-wise subtraction inside HPRO creates
    # a contiguous (V, 3, N) centered_points tensor, so this is safe.
    pts_t = (
        torch.tensor(pts_np.T, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(V, -1, -1)
    )

    # ---- Initialisation ----------------------------------------------------
    centroid = pts_np.mean(axis=0)
    if init_viewpoints_np is None or init_rot_6d_np is None:
        init_vp, init_r6d = fibonacci_sphere_viewpoints(V, init_radius, centroid)
    else:
        init_vp, init_r6d = init_viewpoints_np, init_rot_6d_np

    viewpoints = nn.Parameter(
        torch.tensor(init_vp, dtype=dtype, device=device)
    )   # (V, 3)
    rot_6d = nn.Parameter(
        torch.tensor(init_r6d, dtype=dtype, device=device)
    )   # (V, 6)

    # ---- Optimiser ---------------------------------------------------------
    optimizer = torch.optim.Adam([viewpoints, rot_6d], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01
    )

    loss_history: list = []
    coverage_history: list = []
    standoff_history: list = []
    diversity_history: list = []
    steps_log: list = []

    # ---- Optimisation loop -------------------------------------------------
    t0 = time.perf_counter()
    for step in range(n_steps):
        optimizer.zero_grad()

        # Batched HPRO_limited forward: B = V viewpoints simultaneously.
        # Only w_combined (V, N) is used in the loss; visible_pts and
        # visible_indices are Python lists (non-differentiable) and must not
        # enter the backward graph.
        _, _, w_combined = model(
            pts_t,       # (V, 3, N)
            viewpoints,  # (V, 3)
            rot_6d,      # (V, 6)
            gamma=gamma,
            k=k,
        )   # w_combined: (V, N)

        C = compute_coverage_soft(w_combined)
        stand = _standoff_penalty(viewpoints, pts_surface_t, near_dist, far_dist)
        div = _diversity_penalty(viewpoints, min_inter_dist)

        loss = -C + lambda_standoff * stand + lambda_diversity * div
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == n_steps - 1:
            l_v = loss.item()
            c_v = C.item()
            s_v = stand.item()
            d_v = div.item()
            loss_history.append(l_v)
            coverage_history.append(c_v)
            standoff_history.append(s_v)
            diversity_history.append(d_v)
            steps_log.append(step)
            print(
                f"  step {step:4d}  loss={l_v:9.4f}  coverage={c_v:.4f}  "
                f"standoff={s_v:.5f}  diversity={d_v:.5f}"
            )

    wall_time = time.perf_counter() - t0
    print(f"\nOptimisation finished in {wall_time:.1f} s")

    return MultiViewOptResult(
        viewpoints_np=viewpoints.detach().cpu().numpy(),
        rot_6d_np=rot_6d.detach().cpu().numpy(),
        loss_history=loss_history,
        coverage_history=coverage_history,
        standoff_history=standoff_history,
        diversity_history=diversity_history,
        steps_log=steps_log,
        wall_time_s=wall_time,
        n_points=N,
        n_viewpoints=V,
    )
