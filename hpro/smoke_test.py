"""Fast, deterministic CPU smoke test for the frustum-restricted HPRO operator.

This guards the correctness-critical behaviours of ``HPRO_limited`` *without*
needing a GPU, so it can be run on a laptop or a MetaCentrum login node before
submitting the (slow, queued) GPU evaluation job:

    python hpro/smoke_test.py

It checks deterministic, analytically-known properties only — it does **not**
measure HPRO's statistical occlusion accuracy (that is what ``eval_frustum.py``
does against ray-cast ground truth). Specifically:

  1. Output shapes: ``w_combined`` is ``(B, N)``; ``forward`` returns sane types.
  2. Soft frustum mask  ≈  hard geometric frustum test as sharpness → ∞.
  3. Points behind / in front-but-outside the frustum get combined score ≈ 0.
  4. Finite gradients flow to both ``viewpoint`` and ``rot_6d``.
  5. Batched (B>1) operator output matches looping single (B=1) calls
     (regression guard for the batch-broadcast fix in ``HPRO.detect_max_in_direction``).
  6. The memory-efficient path (``fits_in_memory=False``) matches the batched
     path across alpha/delta configurations, and preserves input dtype
     (regression guard for the alpha-frame and float32-accumulator fixes).
  7. ``GatedVisibilityLayer(HPROBackbone)`` reproduces ``HPRO_limited`` exactly,
     pinning the backbone interface to the Stage-1-validated operator.
  8. Analytic properties of the receding-horizon trajectory losses, including
     that the terminal attractor follows remaining demand rather than the
     object's (interior) centroid.

Exit code is non-zero if any check fails.
"""

import math
import sys

import numpy as np
import torch

from HPRO import HPRO
from HPRO_limited import HPRO_limited
from frustum_gt import build_camera_frame, points_inside_frustum

DTYPE = torch.float64
DEVICE = "cpu"          # smoke test is intentionally CPU-only
SEED = 0


def _make_cloud(n=400, seed=SEED):
    """Random points in a box in front of a camera at the origin looking +Z."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(low=[-1.5, -1.5, -1.0], high=[1.5, 1.5, 3.0], size=(n, 3))
    return pts


def _frustum_cfg():
    return dict(fov_h=math.radians(40.0), fov_v=math.radians(30.0),
                near=0.5, far=2.5)


def _to_tensor(a):
    return torch.tensor(a, dtype=DTYPE, device=DEVICE)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_shapes(model, pts_t, vp_t, rot_t, gamma):
    vis_pts, vis_idx, w = model(pts_t, vp_t, rot_t, gamma=gamma)
    n = pts_t.shape[2]
    assert w.shape == (1, n), f"w_combined shape {tuple(w.shape)} != (1, {n})"
    assert torch.isfinite(w).all(), "w_combined contains non-finite values"
    assert vis_pts.shape[0] == 1 and vis_pts.shape[1] == 3, \
        f"visible_pts shape unexpected: {tuple(vis_pts.shape)}"
    assert vis_idx.dim() == 1, f"visible_indices should be 1-D, got {vis_idx.dim()}-D"
    print(f"  [ok] shapes: w_combined={tuple(w.shape)}, "
          f"visible={vis_pts.shape[2]} pts")


def check_frustum_mask_matches_hard(model, pts, vp, look_approx, up_approx, cfg):
    """At high sharpness, the soft frustum mask must agree with the exact test."""
    look, up, right = build_camera_frame(look_approx, up_approx)
    hard = points_inside_frustum(pts, vp, look, up, right,
                                 cfg["fov_h"], cfg["fov_v"], cfg["near"], cfg["far"])

    pts_t = _to_tensor(pts.T[np.newaxis])            # (1, 3, N)
    vp_t = _to_tensor(vp[np.newaxis, :, np.newaxis])  # (1, 3, 1)
    look_t, up_t, right_t = (_to_tensor(x[np.newaxis]) for x in (look, up, right))

    f = model.compute_frustum_mask(
        pts_t, vp_t, look_t, up_t, right_t,
        cfg["fov_h"], cfg["fov_v"], cfg["near"], cfg["far"],
        sharpness=500.0,
    )[0].cpu().numpy()                                # (N,)
    soft = f > 0.5

    disagree = int(np.sum(soft != hard))
    frac = disagree / len(hard)
    # A few points may straddle the boundary; require near-perfect agreement.
    assert frac < 0.01, (f"soft frustum mask disagrees with hard test on "
                         f"{disagree}/{len(hard)} points ({frac:.3%})")
    # No point behind the camera (depth <= 0) may be marked inside.
    depth = (pts - vp) @ look
    assert not np.any(soft & (depth <= cfg["near"])), \
        "a point at/behind the near plane was marked inside the frustum"
    print(f"  [ok] soft≈hard frustum: disagree {disagree}/{len(hard)} "
          f"({frac:.3%}); {int(hard.sum())} pts in frustum")


def check_behind_camera_zero(model, cfg, gamma):
    """Points behind the camera must receive combined score ~0."""
    # 5 points strictly behind the camera (negative Z) + 5 inside the frustum.
    behind = np.array([[0.0, 0.0, -1.0], [0.2, 0.1, -0.5], [-0.3, 0.0, -2.0],
                       [0.1, -0.2, -0.8], [0.0, 0.3, -1.5]])
    inside = np.array([[0.0, 0.0, 1.5], [0.1, 0.1, 1.2], [-0.1, 0.0, 1.8],
                       [0.05, -0.05, 1.0], [0.0, 0.1, 2.0]])
    pts = np.vstack([behind, inside])

    pts_t = _to_tensor(pts.T[np.newaxis])
    vp_t = _to_tensor([[0.0, 0.0, 0.0]])
    rot_t = _to_tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])   # look +Z, up +Y

    _, _, w = model(pts_t, vp_t, rot_t, gamma=gamma)
    w = w[0].cpu().numpy()
    behind_max = float(w[:5].max())
    assert behind_max < 1e-3, \
        f"behind-camera points have non-zero score (max={behind_max:.4g})"
    print(f"  [ok] behind-camera suppressed: max score {behind_max:.2e}")


def check_finite_gradients(model, pts, cfg, gamma):
    pts_t = _to_tensor(pts.T[np.newaxis])
    vp = torch.nn.Parameter(_to_tensor([[0.0, 0.0, -1.0]]))
    rot = torch.nn.Parameter(_to_tensor([[0.1, -0.1, 1.0, 0.0, 1.0, 0.0]]))

    _, _, w = model(pts_t, vp, rot, gamma=gamma)
    loss = -w.sum()
    loss.backward()

    assert vp.grad is not None and torch.isfinite(vp.grad).all(), \
        f"viewpoint gradient missing/non-finite: {None if vp.grad is None else vp.grad}"
    assert rot.grad is not None and torch.isfinite(rot.grad).all(), \
        f"rot_6d gradient missing/non-finite: {None if rot.grad is None else rot.grad}"
    gn = float(vp.grad.norm() + rot.grad.norm())
    assert gn > 0, "gradients are exactly zero everywhere (no signal)"
    print(f"  [ok] finite gradients: |grad| = {gn:.4g}")


def check_trajectory_losses():
    """Analytic properties of the receding-horizon loss terms (trajectory.py).

    All are deterministic and closed-form, so they belong in the CPU smoke test
    rather than in a rollout. The terminal-cost check is a regression guard: the
    first implementation pulled toward the demand-weighted *centroid*, which for a
    closed surface lies inside the object, so it dragged the camera into the hull
    and deadlocked against the standoff penalty (the HPRO rollout froze at 0.46
    coverage for 18 cycles).
    """
    import trajectory as tj

    # --- demand_weighted_coverage -------------------------------------------
    N = 20
    w_all = torch.ones(2, N, dtype=DTYPE)
    d_all = torch.ones(N, dtype=DTYPE)
    c = float(tj.demand_weighted_coverage(w_all, d_all))
    assert abs(c - 1.0) < 1e-6, f"w=1, d=1 should give coverage 1, got {c}"

    c0 = float(tj.demand_weighted_coverage(torch.zeros(2, N, dtype=DTYPE), d_all))
    assert abs(c0) < 1e-9, f"w=0 should give coverage 0, got {c0}"

    # Satisfied demand must not contribute: covering only zero-demand points
    # scores 0, which is what stops covered points attracting the trajectory.
    d_half = torch.cat([torch.ones(N // 2, dtype=DTYPE),
                        torch.zeros(N // 2, dtype=DTYPE)])
    w_second = torch.zeros(1, N, dtype=DTYPE)
    w_second[0, N // 2:] = 1.0
    c_wasted = float(tj.demand_weighted_coverage(w_second, d_half))
    assert abs(c_wasted) < 1e-9, \
        f"covering only satisfied points should score 0, got {c_wasted}"

    # --- path_length ---------------------------------------------------------
    start = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE)
    pos = torch.tensor([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=DTYPE)
    L = float(tj.path_length(pos, start))
    assert abs(L - 3.0) < 1e-9, f"path length should be 1+2=3, got {L}"

    # A straight, evenly-spaced path has zero curvature.
    straight = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                            dtype=DTYPE)
    s = float(tj.smoothness(straight, start))
    assert s < 1e-9, f"straight path should have zero smoothness cost, got {s}"

    # --- terminal_cost -------------------------------------------------------
    # A hollow shell: every surface point is on a sphere of radius 1, so the
    # centroid (the origin) is empty space *inside* the object. Demand survives
    # on one side only.
    ang = torch.linspace(0, 2 * math.pi, 33, dtype=DTYPE)[:-1]
    shell = torch.stack([torch.cos(ang), torch.sin(ang),
                         torch.zeros_like(ang)], dim=1)          # (32, 3)
    demand = (shell[:, 0] > 0.7).to(DTYPE)                        # +x side only
    assert float(demand.sum()) > 0

    target = 0.5
    # Pose already at the correct standoff from the demanded (+x) region.
    good = torch.tensor([[1.0 + target, 0.0, 0.0]], dtype=DTYPE)
    # Pose at the centroid — the old implementation's attractor, and the worst
    # possible place to be (inside the object).
    centre = torch.tensor([[0.0, 0.0, 0.0]], dtype=DTYPE)

    c_good = float(tj.terminal_cost(good, shell, demand, target))
    c_centre = float(tj.terminal_cost(centre, shell, demand, target))
    # Not ~0: the soft-min averages over several nearby demanded points, so the
    # effective distance overshoots the true nearest by O(tau). Bound it loosely
    # -- the load-bearing assertions are the comparisons below.
    assert c_good < 0.02, f"pose at correct standoff should cost ~0, got {c_good}"
    assert c_centre > 10 * c_good,\
        (f"the object's centroid must NOT be the terminal attractor "
         f"(centre={c_centre:.4g} <= standoff pose={c_good:.4g})")

    # No demand left -> no terminal pull at all.
    c_none = float(tj.terminal_cost(good, shell, torch.zeros_like(demand), target))
    assert abs(c_none) < 1e-12, f"zero demand should give zero terminal cost, got {c_none}"

    # The attractor must follow demand: with demand on -x, the +x pose is bad.
    demand_neg = (shell[:, 0] < -0.7).to(DTYPE)
    c_far = float(tj.terminal_cost(good, shell, demand_neg, target))
    assert c_far > c_good, \
        f"terminal cost must track where demand is (got {c_far:.4g} vs {c_good:.4g})"

    print(f"  [ok] trajectory losses: coverage/length/smoothness exact; "
          f"terminal attractor tracks demand, not centroid "
          f"(centre {c_centre:.3f} > standoff {c_good:.3f})")


def check_global_guide():
    """Contract of the two-timescale global guide (trajectory.GlobalGuide).

    Deterministic, CPU-only. Guards the stall fix of §3.4 item 2: the guide must
    (1) target the nearest demand cluster, (2) hold that target under
    re-selection (hysteresis — no ping-pong), (3) advance when the target is
    exhausted, (4) defer and leave a target that stops yielding new points (the
    escape from uncoverable regions), and (5) clear deferrals rather than idle
    when everything left is deferred.
    """
    import numpy as np
    import trajectory as tj

    rng = np.random.default_rng(0)
    # Two well-separated blobs of 30 points each; robot starts next to blob A.
    blob_a = rng.normal(scale=0.1, size=(30, 3)) + np.array([0.0, 0.0, 0.0])
    blob_b = rng.normal(scale=0.1, size=(30, 3)) + np.array([10.0, 0.0, 0.0])
    pts = np.vstack([blob_a, blob_b])
    idx_a, idx_b = np.arange(30), np.arange(30, 60)
    robot = np.array([-1.0, 0.0, 0.0])

    cfg = tj.TrajectoryConfig(guide_pts_per_cluster=30, guide_k_max=4,
                              guide_patience=2, guide_min_new=5,
                              guide_defer_cycles=6, guide_mass_min=3.0)
    guide = tj.GlobalGuide(pts, cfg, np.random.default_rng(cfg.seed))
    demand = np.ones(60)

    # (1) nearest cluster first
    t0 = guide.select(demand, robot, cycle=0)
    assert set(t0) <= set(idx_a), "guide must target the nearest cluster first"

    # (2) hysteresis: good harvests -> same target under re-selection
    guide.report_harvest(20)
    t1 = guide.select(demand, robot, cycle=1)
    assert np.array_equal(t0, t1), "target must not change while it progresses"

    # (3) exhaustion: demand of A satisfied -> guide advances to B
    demand[idx_a] = 0.0
    t2 = guide.select(demand, robot, cycle=2)
    assert set(t2) <= set(idx_b), "exhausted target must advance to the next cluster"

    # (4) engaged stall: robot is within engage range of A (d ~ 0.9 < far_dist)
    # but harvests nothing for `patience` cycles -> A deferred, tour moves to B.
    demand = np.ones(60)
    guide2 = tj.GlobalGuide(pts, cfg, np.random.default_rng(cfg.seed))
    t = guide2.select(demand, robot, cycle=0)
    assert set(t) <= set(idx_a)
    cyc = 0
    for _ in range(cfg.guide_patience):
        cyc += 1
        guide2.report_harvest(0)                 # nothing new: A is a tar pit
        t = guide2.select(demand, robot, cycle=cyc)
    assert set(t) <= set(idx_b), "a stalled engaged target must be deferred and left"
    assert guide2.deferrals == 1
    assert (guide2.deferred_until[idx_a] > cyc).all(), \
        "the stalled cluster's surviving points must be deferred"

    # (4b) transit is NOT a stall while distance is closing: march toward B with
    # good speed and no harvest — the target must survive the whole trip.
    pos = robot.copy()
    for _ in range(6):
        cyc += 1
        pos = pos + np.array([1.0, 0.0, 0.0])   # closing > 0.25*step_size
        guide2.report_harvest(0)                 # transit harvests nothing
        t = guide2.select(demand, pos, cycle=cyc)
    assert set(t) <= set(idx_b), \
        "low harvest while closing distance must not count as a stall"

    # (5) all deferred -> deferrals cleared, not an idle planner. Stall out B
    # (engaged now), with A still deferred.
    for _ in range(20):
        cyc += 1
        guide2.report_harvest(0)
        t = guide2.select(demand, pos, cycle=cyc)
        if guide2.deferrals >= 2:
            break
    assert guide2.deferrals >= 2, "a parked, non-harvesting robot must defer"
    assert t is not None and len(t) > 0, \
        "with every cluster deferred the guide must reset, not go idle"

    print("  [ok] global guide: nearest-first, hysteresis, exhaustion advance, "
          "engaged-stall deferral, transit exemption, deferral reset")


def check_backbone_layer_matches_hpro_limited(pts, cfg, gamma):
    """``HPROBackbone`` + ``GatedVisibilityLayer`` must equal ``HPRO_limited``.

    ``HPRO_limited`` is the Stage-1-validated operator (F1 0.911 on the lamp), so
    the generalised backbone interface is pinned to it by equivalence rather than
    by re-deriving the maths: any drift here silently invalidates §3 of
    RESEARCH_PLAN.md. Both compute clamp(w_hpro, 0) x f_frustum, so on identical
    input they must agree exactly, not merely closely.

    NVPS/ensemble equivalence needs a GPU and the git-ignored external assets, so
    it is verified separately (see RESEARCH_PLAN.md §8.1); this check keeps the
    CPU-only smoke test dependency-free.
    """
    from backbones import HPROBackbone
    from visibility_layer import GatedVisibilityLayer

    V = 3
    rng = np.random.default_rng(11)
    vps_np = rng.normal(size=(V, 3))
    vps_np = 2.0 * vps_np / np.linalg.norm(vps_np, axis=1, keepdims=True)
    # look roughly at the origin, world-up +Z
    r6_np = np.concatenate([-vps_np, np.tile([0.0, 0.0, 1.0], (V, 1))], axis=1)

    pts_t = _to_tensor(pts.T)                       # (3, N)
    vps = _to_tensor(vps_np)                        # (V, 3)
    r6 = _to_tensor(r6_np)                          # (V, 6)

    old = HPRO_limited(fits_in_memory=True, visibility_score_thresh=0.5,
                       frustum_sharpness=50.0, device=DEVICE, **cfg)
    _, _, w_old = old(pts_t.unsqueeze(0).expand(V, -1, -1), vps, r6, gamma=gamma)

    layer = GatedVisibilityLayer(
        HPROBackbone(gamma=gamma, k=10, device=DEVICE),
        frustum_sharpness=50.0, device=DEVICE, **cfg,
    )
    w_new = layer(pts_t, vps, r6)

    assert w_new.shape == w_old.shape, \
        f"layer shape {tuple(w_new.shape)} != HPRO_limited {tuple(w_old.shape)}"
    diff = float((w_old - w_new).abs().max())
    assert diff == 0.0, \
        f"GatedVisibilityLayer(HPROBackbone) differs from HPRO_limited by {diff:.3g}"
    print(f"  [ok] backbone layer == HPRO_limited: max |Δw| = {diff:.2e}")


def check_memory_path_matches_batched(pts, gamma):
    """``fits_in_memory=False`` must reproduce the batched path exactly.

    Guards two fixed defects in ``HPRO.detect_max_in_direction``:

    * the alpha (second-direction) pass projected against the *unshifted*
      points and scored against ``||F(p_i)||`` instead of ``||F(p_i) - C*||``,
      so the score and its top-k reference lived in different frames
      (divergence ~1.5 in a score of order 1);
    * the per-point accumulators were allocated by ``torch.zeros`` without
      ``dtype=``, i.e. always float32, truncating float64 clouds (residual
      disagreement ~5e-8 = float32 epsilon, present even with ``alphas=[]``).

    Both paths implement the same maths, so on identical float64 input they
    must agree to float64 round-off -- in practice bit-exactly.
    """
    pts_t = _to_tensor(pts.T[np.newaxis])              # (1, 3, N)
    vp_t = _to_tensor([[0.0, 0.0, -3.0]])

    configs = [([], 0.0), ([0.5], 0.0), ([0.5], 0.02), ([0.3, 0.7], 0.01)]
    worst = 0.0
    for alphas, delta in configs:
        ws = []
        for fits in (True, False):
            m = HPRO(fits_in_memory=fits, device=DEVICE)
            _, _, w = m(pts_t, vp_t, gamma=gamma, alphas=alphas, delta=delta, k=10)
            assert w.dtype == DTYPE, (
                f"alphas={alphas}, fits_in_memory={fits}: score dtype {w.dtype} "
                f"!= input dtype {DTYPE} (accumulator lost precision)"
            )
            ws.append(w.reshape(-1))
        diff = float((ws[0] - ws[1]).abs().max())
        worst = max(worst, diff)
        assert diff < 1e-12, (
            f"alphas={alphas}, delta={delta}: memory-efficient path disagrees "
            f"with batched path by {diff:.3g}"
        )
    print(f"  [ok] mem-path==batched over {len(configs)} alpha/delta configs: "
          f"max |Δw| = {worst:.2e}")


def check_batch_matches_loop(model, pts, gamma):
    """Batched (B=2) output must equal two separate B=1 calls (broadcast fix)."""
    poses = [
        ([0.0, 0.0, -1.5], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ([0.3, 0.2, -1.0], [0.2, -0.1, 1.0, 0.0, 1.0, 0.0]),
    ]
    pts_single = _to_tensor(pts.T[np.newaxis])         # (1, 3, N)

    # Looped single calls
    w_loop = []
    for vp, rot in poses:
        _, _, w = model(pts_single, _to_tensor([vp]), _to_tensor([rot]), gamma=gamma)
        w_loop.append(w[0])
    w_loop = torch.stack(w_loop, dim=0)                # (2, N)

    # Batched call
    pts_b = pts_single.repeat(2, 1, 1)                 # (2, 3, N)
    vp_b = _to_tensor([p[0] for p in poses])           # (2, 3)
    rot_b = _to_tensor([p[1] for p in poses])          # (2, 6)
    _, _, w_batch = model(pts_b, vp_b, rot_b, gamma=gamma)

    assert w_batch.shape == w_loop.shape, \
        f"batched shape {tuple(w_batch.shape)} != looped {tuple(w_loop.shape)}"
    max_diff = float((w_batch - w_loop).abs().max())
    assert max_diff < 1e-9, f"batched vs looped mismatch: max |Δ| = {max_diff:.3g}"
    print(f"  [ok] batch==loop: max |Δw| = {max_diff:.2e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = _frustum_cfg()
    gamma = -math.exp(-3.0)

    model = HPRO_limited(
        fits_in_memory=True,
        visibility_score_thresh=0.6,
        frustum_sharpness=50.0,
        device=DEVICE,
        **cfg,
    )

    pts = _make_cloud()
    pts_t = _to_tensor(pts.T[np.newaxis])
    vp_t = _to_tensor([[0.0, 0.0, -1.0]])
    rot_t = _to_tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])

    checks = [
        ("shapes", lambda: check_shapes(model, pts_t, vp_t, rot_t, gamma)),
        ("frustum-mask-vs-hard", lambda: check_frustum_mask_matches_hard(
            model, pts, np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), cfg)),
        ("behind-camera-zero", lambda: check_behind_camera_zero(model, cfg, gamma)),
        ("finite-gradients", lambda: check_finite_gradients(model, pts, cfg, gamma)),
        ("batch==loop", lambda: check_batch_matches_loop(model, pts, gamma)),
        ("mem-path==batched", lambda: check_memory_path_matches_batched(pts, gamma)),
        ("backbone-layer==HPRO_limited",
         lambda: check_backbone_layer_matches_hpro_limited(pts, cfg, gamma)),
        ("trajectory-losses", check_trajectory_losses),
        ("global-guide", check_global_guide),
    ]

    print(f"Running HPRO_limited smoke test on {DEVICE} (dtype={DTYPE})...\n")
    failures = []
    for name, fn in checks:
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failures.append(name)
        except Exception as e:  # noqa: BLE001 — surface any unexpected error
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failures.append(name)

    print()
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)}/{len(checks)}): {', '.join(failures)}")
        return 1
    print(f"SMOKE TEST PASSED ({len(checks)}/{len(checks)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
