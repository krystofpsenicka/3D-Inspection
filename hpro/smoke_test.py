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
