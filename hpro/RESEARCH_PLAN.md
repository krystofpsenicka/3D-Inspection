# Differentiable Inspection Planning with Visibility Surrogates — Audit, Evidence & Publication Plan

*Status as of 2026-07-16. Covers: implementation audit of the HPRO extension (Stages 1–2),
integration of the NVPS neural visibility model as an alternative backbone, diagnostic
experiments (all reproducible via `diagnostics/`), and the research strategy toward a
publishable paper.*

**TL;DR.** The core HPRO port and the Stage-1 frustum extension are correct and validated.
The Stage-2 multi-viewpoint optimizer is correctly *coded* but was methodologically
incomplete: with the HPRO backbone, gradient refinement *reduces* true coverage on the
concave wreck mesh while the soft objective climbs to ≈1.0 — the optimizer exploits the
miscalibrated visibility proxy. Swapping in the pretrained NVPS backbone flips the result:
**refinement beats oracle greedy on the wreck (0.941 vs 0.938 at V=5)** where HPRO lost
(0.915). The two operators fail in exactly complementary ways.

**⚠ Positioning corrected 2026-07-16 (see §5).** An earlier draft of this plan claimed that
*"nobody offers gradient-based joint refinement of camera positions and orientations
against a differentiable occlusion-aware visibility model on point clouds."* **That claim
is false.** [NeOF (Cao et al., RA-L 2024)](https://arxiv.org/abs/2412.08266) does exactly
that — on point clouds with normals, over full 6-DoF poses, with occlusion, FOV, a normal
gate, and a gradient/non-gradient hybrid — in this plan's own target venue. Static
differentiable camera *placement* is therefore **occupied ground**. What remains unclaimed
is **trajectories and online construction** (§6.4), which is now the headline; offline
placement follows as a second contribution, benchmarked against NeOF *and* against this
thesis's own combinatorial (set-cover + VRP) pipeline.

---

## Contents

1. [Environment](#1-environment)
2. [Implementation audit vs. the HPRO paper](#2-implementation-audit-vs-the-hpro-paper)
3. [Diagnostic experiments](#3-diagnostic-experiments)
4. [NVPS deep-dive](#4-nvps-deep-dive-what-were-plugging-in)
5. [Where a paper fits (related work)](#5-where-a-paper-fits)
6. [The strongest paper](#6-the-strongest-paper)
7. [Order of work](#7-order-of-work)
8. [Reproducing everything](#8-reproducing-everything)

---

## 1 · Environment

The `isaaclab` conda env runs everything in `hpro/`:

- Python 3.11, torch 2.7.0+cu128 (CUDA verified, RTX 3090, 24 GB), trimesh 4.5.1, rtree,
  scipy, matplotlib — already present.
- `embreex` 4.4.0 — installed 2026-07-10 (fast Embree ray-cast ground truth).
- `ocnn` 2.3.2 — installed 2026-07-10 (octree ops for the NVPS backbone).

**ocnn × triton incompatibility (found 2026-07-16).** ocnn 2.3.2's Triton conv kernels do
not compile under the triton 3.3 that ships with torch 2.7 — every NVPS forward pass died
with `AttributeError("'NoneType' object has no attribute 'type'")`, i.e. *all* NVPS
diagnostics in §8 were broken. The kernel guards its bias load with `if bias is not None`
but passes `bias` as a plain (non-`constexpr`) argument, so newer triton traces the branch
with `bias=None`; NVPS's convs are bias-free, so it always fires. `hpro/ocnn_compat.py`
sets ocnn's supported `OCNN_DISABLE_TRITON=1` fallback and **must be imported before
`ocnn`** — `backbones.py` and both NVPS diagnostics do this, so no manual env var is
needed. Consequences: the §3.1 numbers reproduce exactly, and the pure-PyTorch path turns
out to be *fast* (§4) — the old "~30 s" was triton's JIT compile, not the model.

```bash
alias ipy=/home/troja-lab-02/miniconda3/envs/isaaclab/bin/python
ipy hpro/smoke_test.py          # 5/5 checks pass
ipy hpro/eval_multi.py ...      # runs on GPU, ~1–5 s per optimization at N=3000
```

The heavier `inspection` env (isaacsim, cuOpt, triro/OptiX) is only needed for the legacy
pipeline and Isaac Sim replay, not for the HPRO/NVPS work.

NVPS assets live in `hpro/external/` (git-ignored):
`neural-visibility/` (clone of <https://github.com/octree-nn/neural-visibility>) and
`neuvis_00040.pth` (pretrained checkpoint). To re-fetch:

```bash
git clone --depth 1 https://github.com/octree-nn/neural-visibility.git hpro/external/neural-visibility
curl -sL -o hpro/external/neuvis_00040.pth \
  "https://huggingface.co/JKAlice/neuvis/resolve/main/depth_8/checkpoints/00040.model.pth"
```

---

## 2 · Implementation audit vs. the HPRO paper

Checked line-by-line against Katz & Tal, *HPRO: Direct Visibility of Point Clouds for
Optimization*, CGF 44(2), 2025 ([PDF](https://www.ee.technion.ac.il/~ayellet/Ps/25-KT.pdf)).

| Verdict | Finding |
|---|---|
| ✅ Correct | **Core operator (`HPRO.py`).** Exponential-inversion transform (Eq. 1–2), projection matrix R_ij = ⟨F(p_j)−C, d_i⟩ (Eq. 3; verified the `repeat`/`repeat_interleave` index order), top-k ELU score (Eq. 5, Alg. 1) all match. Batched and loop paths agree to 8×10⁻¹⁵ (smoke test). |
| ✅ Correct | **Frustum extension (`HPRO_limited.py`).** Six-sigmoid gate product, behind-camera depth clamp, Zhou-et-al. 6D rotation with Gram–Schmidt, `relu(w)·f` combination. Stage-1 validation (F1 0.911 ≥ best-tuned hard cull 0.907) is a real result. GT harness (`frustum_gt.py`) mirrors the camera convention exactly. |
| ⚠️ Unused | **δ noise offset (Eq. 7) and α second direction (Eq. 8)** are implemented but never enabled (`delta=0.0`, `alphas=[]`). The paper shows δ>0 helps *even on clean data* (optimal δ = noise + 1 %) and α adds ≈1 % accuracy precisely on silhouettes — HPRO's worst region and the wreck's dominant geometry. Free accuracy on the shelf. |
| ✅ Fixed | **`HPRO.py` — α-pass inconsistency in the memory-efficient path** (fixed 2026-07-16). With `alphas≠[]` and `fits_in_memory=False` the loop path was wrong in *two* ways, not the one originally logged: it projected against the **unshifted** points (`⟨F(p_j), d2_i⟩` instead of `⟨F(p_j)−C*, d2_i⟩`) *and* scored with ‖F(p_i)‖ instead of ‖F(p_i)−C*‖ — score and top-k reference in different frames. Measured divergence vs the batched path: **1.5–4.3** in a score of order 1. Both paths now verified against an independent NumPy reference of Eq. 8 (agreement 3×10⁻¹⁴) and against each other (bit-exact). Guarded by `smoke_test.py` check 6. |
| ✅ Fixed | **`HPRO.py` — float32 accumulator in the memory-efficient path** (found while fixing the above, 2026-07-16). `w`/`w_2` were allocated with `torch.zeros(...)` and no `dtype=`, so they were **always float32** and silently truncated float64 clouds. This is why the loop path disagreed with the batched path by ≈5×10⁻⁸ (= float32 eps) *even with `alphas=[]`* — a discrepancy that predates the α work and was never noticed because the batched path has no such accumulator. Now `dtype=centered_points.dtype`; the two paths agree bit-exactly. |
| 🐞 Method | **Fixed γ under free-position optimization.** γ=−e⁻⁷ was tuned at a fixed pose distance; the paper's own viewpoint application *fixes the radius and optimizes angles only* because the γ optimum shifts with distance and angular density. Stage 2 lets positions roam a 0.1–6.0 standoff band with one global γ. Diagnostics confirm viewpoints creep inward (standoff 2.7→2.0) because closer inflates the score. |
| 🐞 Method | **Soft coverage is an optimistic, exploitable proxy.** k=10 is chosen by the paper for *gradient flow*, not accuracy: any point ranking in its own top-k gets w∈[0,1] even when occluded. The soft set-cover 1−Π(1−w) over-counts, and Adam finds exactly those poses where the over-counting is worst. |
| ⚠️ Missing | **Baseline framing & scope.** `eval_multi.py`'s greedy baseline consumes oracle ray-cast visibility (legitimate upper reference, but the paper also needs non-oracle baselines, multi-mesh statistics, and CMA-ES — see §6.3). Stage 3 (trajectories) not started; no collision or path-length terms yet. |

---

## 3 · Diagnostic experiments

All numbers: N=3000 sampled points, FOV 30°×35°, near/far 0.1/6.0, γ=−e⁻⁷, k=10,
300 Adam steps, seed 17. GT = frustum ∩ Embree ray-cast. "Greedy" = oracle greedy over
50 Fibonacci-sphere candidates at radius 3.05. Meshes unit-sphere-normalized.

### 3.1 Backbone accuracy (full-sphere occlusion, no frustum) — `diagnostics/probe_neuvis.py`

Pretrained NVPS (class 1 = visible, threshold 0.5) vs plain HPRO (w>0.99), 12 random
viewpoints per radius:

| Mesh | Radius | NVPS F1 | NVPS prec | HPRO F1 | HPRO prec |
|---|---|---|---|---|---|
| wreck | R≈2.0 (train conv.) | **0.858** | 0.854 | 0.716 | 0.741 |
| wreck | R=1.5 | **0.846** | 0.849 | 0.719 | 0.694 |
| wreck | R=6.0 | **0.862** | 0.859 | 0.651 | 0.841 |
| lamp (thin) | R≈2.3 (train conv.) | 0.681 | 0.908 | **0.819** | 0.930 |
| lamp (thin) | R=1.5 | 0.488 | 0.871 | **0.825** | 0.911 |
| lamp (thin) | R=6.0 | **0.825** | 0.933 | 0.650 | 0.983 |

NVPS is flat across radii (structural — §4) and dominates on the scanned concave wreck;
HPRO dominates on the thin, non-watertight ModelNet lamp (NVPS's documented failure
class). Per-view cost after one-time feature extraction: NVPS 0.5–2 ms vs HPRO 3–6 ms at
N=3000; NVPS is O(N) memory vs HPRO's O(N²) — at N=100k HPRO is infeasible, NVPS routine.

### 3.2 Multi-view coverage optimization (frustum-gated soft set-cover) — `diagnostics/diag_stage2.py`, `diagnostics/diag_nvps_stage2.py`

GT coverage of final poses (↑/↓ = change vs the greedy init):

| Mesh / V | Greedy (oracle) | HPRO fib-init | HPRO greedy+refine | NVPS fib-init | NVPS greedy+refine |
|---|---|---|---|---|---|
| wreck, V=1 | 0.413 | 0.395 | **0.329 ↓** | 0.382 | 0.415 ↑ |
| wreck, V=5 | 0.938 | 0.880 | **0.915 ↓** | 0.920 | **0.941 ↑** |
| lamp, V=1 | 0.836 | 0.486 | 0.832 ↓ | 0.509 | **0.619 ↓** |
| lamp, V=5 | 0.959 | 0.958 | 0.959 → | 0.957 | 0.956 → |

Load-bearing observations:

1. **Surrogate exploitation is real and backbone-dependent.** HPRO refinement inflates its
   own score (soft C → 1.0) while true coverage falls, pulling standoffs 2.7→2.0. NVPS
   refinement is *conservative* (soft 0.902 vs GT 0.941 at wreck V=5) and standoffs drift
   *outward* — no collapse, because its score provably cannot be inflated by approaching
   (§4).
2. **With a well-calibrated backbone, refinement beats the oracle-greedy baseline** — the
   first positive result, obtained with zero tuning (single seed, no restarts, no normal
   gate, no calibration). Substantial headroom remains.
3. **Each backbone drags the optimizer into its own blind spots.** NVPS wrecked the lamp
   V=1 exactly where its recall is 0.11 (thin-structure blindness → refinement wanders,
   GT 0.836→0.619). No single frozen surrogate suffices — motivates §6.2/C2.
4. **Calibration is density-dependent for HPRO**: at N=1000 even the lamp miscalibrates
   badly (soft 0.62 vs GT 0.415); N≥3000 needed.
5. **Pure gradient from random init hits local minima at small V** (lamp V=1: 0.486 vs
   greedy 0.836) — local minima, not proxy error (calibration was fine there).
6. **The lamp saturates at V=5** (~0.958 for everything) — benchmark meshes must be
   concave/complex, or nothing differentiates.

---

## 4 · NVPS deep-dive: what we're plugging in

*NVPS = Wang et al., "Neural Visibility of Point Sets", SIGGRAPH Asia 2025*
([arXiv 2509.24150](https://arxiv.org/abs/2509.24150),
[code](https://github.com/octree-nn/neural-visibility), license: free for research use).

**Mechanics.** An octree U-Net (depth 8, feature `LP` = local coords + global position,
6 channels; cloud normalized into a [−0.8, 0.8] box) produces a view-independent 63-d
feature per point, **once per cloud**. *Corrected 2026-07-16: this costs ~24 ms at
N=3000, not the ~30 s recorded earlier — a 1000× error.* The old figure was `triton`'s
JIT/autotune compilation being charged to the model on the first call; with ocnn's
pure-PyTorch path (§1) the measured cost is 24 ms (N=3000), 29 ms (N=20k), 39 ms (N=100k),
CUDA-synchronised, after a one-time ~380 ms warm-up. NVPS therefore **beats** the paper's
reported 75 ms here rather than failing to reproduce it. This is load-bearing for §6.4:
re-featurising a growing cloud every control cycle is affordable, so the background-refresh
thread that item 7 was built around is unnecessary. Per (point, view): the feature is
multiplied elementwise with a 63-d NeRF-style
positional encoding (multires 10) of the **unit** direction point−viewpoint, and a 3-layer
MLP outputs visible/invisible logits (**class 1 = visible** — verified empirically).
Gradients flow to the viewpoint through the direction (verified). Checkpoint
`depth_8/checkpoints/00040.model.pth` from HF `JKAlice/neuvis` loads into
`MyNet(6, 63, 2)` with strict key match.

**The distance-invariance property (worth a proposition in the paper).** Each point's
prediction depends only on its own unit view direction, so NVPS visibility is *constant
along the ray from the point through the camera*: it cannot distinguish a camera 0.5 or
50 units away in the same direction. Consequences:

- *Shield:* the optimizer cannot inflate the occlusion score by approaching the surface —
  the radial component of the score gradient is identically zero. This is why the
  standoff collapse vanished in §3.2.
- *Blindness:* genuine distance-dependent visibility (camera inside a concavity vs outside
  along the same ray) is mis-modeled. Second-order for exterior viewpoints; not for
  close-in interior inspection.
- *Division of labor:* in the gated operator, **all** distance sensitivity must come from
  the analytic terms — frustum near/far and measurement-quality weighting (§6.2/C1).
  That's a clean, explainable architecture, not a workaround.

**Even the NVPS authors don't do free positions** — their view-selection demo re-projects
the viewpoint onto a radius-4 sphere every step (training viewpoints sit at a single
radius = bbox-diagonal norm). The 6-DoF frustum-gated free-pose setting here is beyond
both source papers.

**Retraining is feasible if needed.** Their training stack (`prepare_data.py` +
`train.py`/`thsolver`) is small; GT labeling is ray-casting we already do faster with
Embree. A distance-aware variant (feed log-distance alongside the direction PE; sample
training viewpoints in shells) is a contained, publishable extension — Phase-2 optional;
the pretrained model already clears the bar on the wreck.

---

## 5 · Where a paper fits

### 5.1 What the backbone papers do (and don't)

HPRO (CGF 2025) demonstrates only single-viewpoint, fixed-distance, angle-only
optimization of a full-360° operator. NVPS (SIGGRAPH Asia 2025) beats HPRO on raw
accuracy/speed but has **no frustum model, no multi-view coverage, no planning**, needs
synthetic training labels, and self-reports failures on thin structures and multi-layer
occlusion. Neither does planning.

### 5.2 The prior art that *does* occupy differentiable camera placement

This is the correction of 2026-07-16. Two papers must be cited, compared against, and
beaten — not ignored:

**[NeOF — Cao et al., "Neural Observation Field Guided Hybrid Optimization of Camera
Placement", RA-L 2024](https://arxiv.org/abs/2412.08266)**
([code](https://github.com/yhanCao/NeOF-HybridCamOpt)). Accepted in *this plan's target
venue*. What it already does:

| Claim we might have made | NeOF's status |
|---|---|
| Point-cloud target, no reconstruction | Target is a point cloud `{s_j, n_j}` with normals |
| Joint 6-DoF pose optimization | `P = ⟨p_i, r_i⟩`, positions **and** orientations in SO(3) |
| Differentiable visibility, gradients to the pose | Neural observation field, backprop to pose in PyTorch |
| Normal / back-face gate | Camera-to-object angle `φ^co` built from `n_j` |
| Frustum + measurement quality | Models FOV, image blur, occlusion; camera-to-camera angle `φ^cc` |
| Hybrid discrete–continuous | Gradient stage + non-gradient elite resampling of poor cameras |
| Periodic ray-cast audit / re-anchoring | `ShapeAnalyze` recomputes visibility and re-fits the field every iteration |

Their occlusion model is *the HPR family itself* — "spherical inverse flipping and convex
hull construction", i.e. Katz et al., HPRO's direct ancestor. Baselines: GA, SA, PSO, DE,
MIP, GNN. They also build a real six-camera capture rig.

**[Moraza et al., "Geometry-Based Differentiable Camera Placement for Optimal 3D
Coverage", VISAPP 2026](https://www.scitepress.org/Papers/2026/143358/143358.pdf).**
Differentiable rendering (PyTorch3D Z-buffer) on a **known mesh**; camera *positions* are
learnable while orientation is fixed by a look-at transform (3-DoF, not 6). Its
"At-Least-Once" coverage loss is `L = 1/N Σ_v (1 − P_v)` with
`P_v = 1 − Π_i (1 − P_{i,v})` — **algebraically identical to our soft set-cover `C`** — and
it already includes an incidence-angle term `σ(θ) = 1/(1+e^{−β(cos θ − cos τ)})`. So C1's
"photogrammetric terms none of the visibility papers model" is **already claimed**. It
beats NeOF on COG/OAQ on 4 meshes at M ∈ {3,5,7,9}.

### 5.3 What is genuinely left

1. **Trajectories and online construction (§6.4) — wide open.** Both NeOF and Moraza et al.
   solve *static placement* of a camera set. Neither has trajectories, receding-horizon
   replanning, warm-starting, demand-weighted coverage, or no-prior exploration. **This is
   the headline.**
2. **Surrogate exploitation as a named, characterized phenomenon (C2).** NeOF *works
   around* it — refitting its field against recomputed visibility every iteration is an
   implicit re-anchoring — but never names it, never shows the failure, never contrasts
   backbones. Our §3 evidence (soft C → 1.0 while GT coverage falls; standoff collapse
   2.7→2.0; the HPRO/NVPS inversion) is a real diagnosis. Note this *reduces* C2's novelty
   from "nobody does audits" to "nobody explains why audits are necessary" — still
   publishable, but as analysis, not invention.
3. **Direct differentiable operator vs. fitted field.** NeOF must *fit and continually
   refit a per-scene neural field* to obtain gradients; HPRO/NVPS differentiate the
   visibility operator directly, with no per-scene fitting. Defensible and architecturally
   cleaner — but a subtler claim than "we introduce gradients here", and it must be shown
   empirically (accuracy, wall-clock, scaling in V), not asserted.
4. **Comparison against a real combinatorial inspection pipeline.** Neither prior paper
   compares to a full sample→set-cover→VRP inspection stack. This thesis has one; that
   comparison is ours to make.

Classical inspection planning (structural-inspection planners, CPP surveys, QECI-CPP,
multi-UAV inspection) remains uniformly *sample-then-select* — the decomposition whose
suboptimality motivates the work.

Key references: [HPRO](https://www.ee.technion.ac.il/~ayellet/Ps/25-KT.pdf) ·
[NVPS](https://arxiv.org/abs/2509.24150) ·
[NeOF, RA-L 2024](https://arxiv.org/abs/2412.08266) ·
[Moraza et al., VISAPP 2026](https://www.scitepress.org/Papers/2026/143358/143358.pdf) ·
[QECI-CPP, Drones 2024](https://www.mdpi.com/2504-446X/8/8/394) ·
[Multi-UAV 3D inspection, 2022](https://arxiv.org/pdf/2204.10070) ·
[Multiobjective CPP, 2019](https://arxiv.org/pdf/1901.07272).

---

## 6 · The strongest paper

### 6.1 Headline claim

*Revised 2026-07-16 to sit in the ground §5.3 leaves open. The old headline ("joint
gradient-based optimization of 6-DoF poses beats sample-and-select") is, as stated, NeOF's
claim from 2024 — it cannot lead this paper.*

> *Differentiable point-cloud visibility surrogates turn inspection planning from a
> static pose-selection problem into a continuously re-optimizable **trajectory** problem:
> a receding-horizon planner that warm-starts across updates, repairs plans online when
> the world contradicts the prior model, and needs no surface reconstruction and no
> target-specific training. Gradient-based **placement** is the special case, where we
> also beat the discrete set-cover + VRP pipeline and match/exceed NeOF.*

The load-bearing word is **warm-starting**: discrete set-cover/VRP must re-solve from
scratch when the world changes, and NeOF must refit its neural field; a gradient optimizer
on a *fitting-free* surrogate keeps stepping from its current solution. That is a property
neither prior method has, and it is what makes the online setting ours.

### 6.2 Contribution stack

*Presentation order is now **C4 → C2 → C1 → C3 → C5** (§6.1). C1 and C3 are supporting
infrastructure, not headline claims — §5.2 shows both are substantially anticipated. They
are numbered as before to keep cross-references stable.*

- **C1 — A backbone-agnostic differentiable visibility layer for real cameras.**
  *(Infrastructure, not a novelty claim — NeOF has a normal gate and FOV model; Moraza et
  al. have an incidence-angle term. Present as engineering that enables C4, and compete on
  the fitting-free property and on accuracy, not on the ingredient list.)* The
  frustum-gated 6-DoF operator (done) with two interchangeable occlusion backbones —
  analytic HPRO and neural NVPS — behind one interface, plus a differentiable back-face
  gate σ(−s·⟨n, dir⟩) from the target normals (the HPR family itself never uses them
  *because it assumes none exist* — though NeOF, built on HPR, does) and a
  measurement-quality weight (incidence angle × 1/distance²; the 1/distance² term and the
  restoration of the distance sensitivity NVPS structurally lacks are the parts Moraza et
  al. do not have). Theory:
  limit-correctness of the gated operator (follows from HPRO's Lemmas 4.1–4.2 as
  sharpness→∞, γ→Γ); the NVPS distance-invariance proposition with its shield/blindness
  consequences.

- **C2 — Characterization and control of surrogate exploitation.** *(Now the paper's
  second pillar and its main analytical contribution.)* The §3 phenomenon — pose
  optimization adversarially exploits miscalibrated visibility — presented with the
  HPRO/NVPS contrast as evidence it is a property of the surrogate, not the planner.
  Controls, in escalating novelty: score sharpening with annealed temperature;
  **target-specific self-calibration** — model-based inspection *knows the mesh*, so
  ray-cast a few dozen calibration poses and fit a cheap correction (Platt scaling, or
  fine-tune the 3-layer VisNet head in minutes) before optimizing; periodic ray-cast
  audits during optimization whose disagreement re-anchors the objective.
  **Honesty check:** NeOF's per-iteration `ShapeAnalyze` + field refit is already an
  implicit audit/re-anchor loop, so "nobody does audits" is false. The defensible claim is
  narrower and still strong: *nobody names, isolates, or measures the exploitation
  phenomenon that makes such loops necessary*, and nobody shows it is backbone-dependent
  (HPRO collapses, NVPS does not, and §4 explains structurally why). Pitch C2 as
  diagnosis + a cheaper fitting-free control, not as inventing calibration.

- **C3 — Hybrid discrete–continuous coverage planning.** *(Anticipated by NeOF's
  gradient + elite-resampling hybrid — supporting contribution only.)* Greedy on
  candidates (keeps the (1−1/e) anchor) → joint 6-DoF gradient refinement of all poses
  (escapes candidate discretization; already +0.3 pts un-tuned on the wreck). Our
  distinction from NeOF: the discrete anchor is a *submodular greedy with an approximation
  guarantee*, not heuristic resampling, and it reappears at the trajectory level as the
  two-timescale hierarchy in §6.4/4 — that reuse is the interesting part. Optional
  backbone ensemble max(w_HPRO-gated, w_NVPS) to cover both failure classes — the
  lamp-vs-wreck table is the built-in ablation motivating it. Headline metric: viewpoints
  needed for target coverage, then path length after routing.

- **C4 — Differentiable inspection trajectories.** The "gradient step, harvest seen
  points, repeat" idea formalized two ways:
  1. *Receding-horizon gradient NBV:* optimize the next pose (or short horizon) by
     gradient with a motion cost from the current pose; commit; harvest covered points
     with a hard ray-cast audit (doubles as the C2 re-anchoring); remove; re-optimize the
     remainder.
  2. *Joint trajectory optimization:* each robot's path as a spline whose sampled poses
     enter the soft-OR coverage, with arc-length, inter-robot separation, and
     occupancy-grid SDF penalties — hundreds of variables, i.e. exactly the regime (§6.3)
     where gradients are the only option. Multi-robot allocation can still go through the
     existing VRP stage; MAPF post-processes feasibility.

  Both variants extend naturally to **online trajectory construction during the
  mission** — see §6.4, which is either C4's strongest instantiation or a standalone
  follow-up paper.

- **C5 — System-level evaluation on a real inspection pipeline.** ModelNet40 subset +
  TOSCA (mean±std, ≥5 seeds) for breadth; the wreck with realistic camera parameters for
  depth; baselines: random, Fibonacci, oracle-greedy, greedy-with-500-candidates (kills
  the "just add candidates" objection), operator-greedy (non-oracle), CMA-ES-on-raycast,
  **NeOF** (public code at <https://github.com/yhanCao/NeOF-HybridCamOpt> — mandatory now
  that §5.2 puts it in the same space; compare coverage, observation-angle quality, and
  wall-clock, and report their COG/OAQ metrics so numbers are commensurable), and the full
  thesis pipeline (coverage, path length, wall time). End with an Isaac Sim rollout of the
  optimized trajectories — few papers in this space close that loop.

### 6.3 The honest positioning questions (design the evaluation around them)

**Attack 0 (new, and now the most dangerous): "This is NeOF (RA-L 2024) with a different
visibility backbone."** §5.2 shows the overlap is extensive and in the same venue. There
is no evaluation trick that answers this — only positioning. The three defensible replies,
in order of strength: (i) *trajectories and online replanning*, which NeOF does not do at
all (§6.4 — hence the §7 re-ordering); (ii) *no per-scene fitting* — NeOF must fit and
repeatedly refit a neural observation field, so its "gradient" is through a learned proxy
of visibility, whereas HPRO/NVPS differentiate visibility directly, which is what makes
warm-started online replanning cheap; (iii) *the exploitation analysis* (C2) explaining
why NeOF's refit loop is load-bearing rather than incidental. Reply (ii) must be *measured*
— field-refit wall-clock vs our per-step cost as V and N grow — or it is just words. Run
NeOF's public code early (§7 step 6) so these numbers exist before the draft is written.

**Attack 1: "the mesh is known — why not optimize true ray-cast
coverage with a gradient-free method (CMA-ES, Bayesian optimization)?"** At V=5 (30 dims)
with 20 ms Embree evaluations, CMA-ES is genuinely competitive. The answer, and the
experimental design that follows: **put the evaluation in the regime where gradients are
indispensable.** Realistic inspection imaging (resolution/GSD limits) forces tight
standoff bands (e.g. near 0.5, far 1.5 on the unit wreck), where one view covers a few
percent of the surface and full coverage needs V ≈ 30–80 → 200–500 decision variables;
trajectory variants push to thousands. There, per-candidate ES sampling collapses and the
amortized-surrogate gradient method is the only joint optimizer available. **Today's
far=6.0 config (one view sees 40–80 % of the object) is a toy regime — tighten the camera
model before running the paper's experiments.** Include CMA-ES-on-raycast as a baseline
anyway: matching it at V=5 and dominating at V=40 *is* the story.

Gradients are indispensable in three regimes, and the paper should claim all three:
(1) **high dimensionality** (V ≈ 30–80 poses, trajectory splines — this section);
(2) **anytime warm-started re-planning** (online construction, §6.4 — discrete pipelines
must re-solve from scratch when the world changes, a gradient optimizer keeps stepping);
(3) **no mesh to ray-cast** (online exploration of an unknown structure, §6.4 — the
CMA-ES-on-raycast objection evaporates because there is nothing to ray-cast; the
point-cloud surrogate is the *only* available evaluator, not merely the fastest).

### 6.4 The headline: online trajectory construction

The differentiable trajectory optimizer (C4) adapts naturally from offline planning to
**building the trajectory during the mission** — an MPC-style loop that discrete
sample-and-select pipelines cannot replicate. The enabling property is
**warm-starting**: when the world state changes (points get covered, new sensor points
arrive, the prior mesh turns out to be wrong), set-cover/VRP must re-solve from scratch,
while a gradient optimizer keeps stepping on the updated objective from its current
solution — a handful of steps per control cycle suffice, making the planner *anytime* and
*incremental* by construction.

**Two settings, one mechanism:**

- *Model-based with deviations (fits the thesis pipeline directly):* a prior mesh exists,
  but localization drift, discovered defects needing closer imaging, or an outdated model
  (the wreck degraded since the survey) invalidate parts of the offline plan. Online
  refinement repairs the remaining trajectory on the fly.
- *No-prior exploration:* no mesh at all; the robot accumulates a point cloud from its
  sensor and plans views on the growing cloud. This is the setting where the point-cloud
  surrogate is not merely faster but the **only** evaluator — there is no surface to
  ray-cast (§6.3, pillar 3), and HPR-family operators were *designed* for exactly this
  "visibility without reconstruction" question.

**Architecture (receding-horizon differentiable MPC):**

1. **State.** Accumulated cloud P_t; per-point *residual demand* d_j = 1 − c_j, where the
   coverage mass c_j is updated from **actual execution feedback** (what the camera
   really captured at executed poses), not from surrogate predictions. The trajectory
   tail is a spline θ_t over the next horizon, warm-started by shifting the previous
   solution forward.
2. **Per-cycle re-optimization** (k budget-bounded gradient steps at control rate):
   `L(θ) = −Σ_j d_j·q_j·[1 − Π_{s∈tail}(1 − w_j(s))] + λ·(length + SDF collision +
   smoothness) + terminal cost`. Commit and execute the first segment; advance; repeat.
3. **Demand weighting, not point removal.** This is the "mark seen and remove" idea made
   differentiable: deleting points changes the loss discontinuously and breaks
   warm-starting, whereas demand weights d_j → 0 make covered points *smoothly* stop
   attracting the trajectory. Quality-aware demand (d_j decays only once incidence/GSD
   requirements are met, or after ≥k views) drops in for free via the C1 quality terms.
4. **Two-timescale hierarchy against myopia.** A cheap discrete global pass (greedy/TSP
   over cluster centroids of remaining demand, refreshed every ~10–30 s) supplies the
   visit order and the terminal cost; the differentiable optimizer continuously refines
   the local horizon against it. This is the C3 hybrid (discrete anchor + continuous
   refinement) reappearing at the trajectory level.
5. **Exploration term.** Coverage of *known* points never pulls the robot toward unseen
   space. Add frontier pseudo-points (boundary of observed free space / occupancy-unknown
   shell) into the demand set — to the operator they are just points, so no new machinery.
6. **Online self-calibration for free.** Execution feedback is exactly the audit signal
   of C2: discrepancies between predicted w_j and actually-observed coverage at executed
   poses feed recursive Platt scaling (or slow fine-tuning of the 3-layer NVPS head).
   Offline the audit is a ray-cast against the mesh; online it is the sensor itself —
   same mechanism, two instantiations. This symmetry is a selling point of the paper.
7. **Backbones.** *(Simplified 2026-07-16 — the premise was wrong.)* NVPS features cost
   **~24 ms per cloud, not ~30 s** (§4), so they can simply be recomputed every control
   cycle: no background thread, no stale-feature reasoning, no two-timescale machinery
   here. Delete that complexity from the design. What survives is the *distribution*
   argument: early in a no-prior mission the accumulated cloud is sparse and partial —
   out-of-distribution for NVPS (trained on complete ShapeNet objects) but natural for
   HPRO (any point set works, and O(N²) is cheap at small N). So: **HPRO backbone early /
   local, NVPS as the cloud completes** — the complementarity of §3 turned into a
   scheduling policy.
8. **Multi-robot.** Robots share the demand mass d_j (decremented by teammates'
   observations) and add separation penalties on predicted tails; since each planner is
   just gradient descent on shared soft demand, the scheme is asynchronous and
   decentralization-friendly. Slow-timescale VRP re-allocation on demand clusters stays
   as-is.

**Budget sanity check:** with NVPS at ~1 ms per pose for 3k points and a horizon of ~10
sampled poses, one gradient step costs ~20–30 ms → 5–10 steps per cycle at 4 Hz on the
3090. Real-time is comfortably feasible.

**Evaluation:** Isaac Sim (already in the pipeline) with a simulated depth camera on the
wreck. (i) *Outdated-mesh experiment* — deform/remove part of the wreck; the open-loop
offline plan misses the changed region, online refinement recovers it: the killer demo
tying model-based and online together. (ii) *No-prior exploration* vs frontier-based NBV
baselines: coverage vs mission time / path length. (iii) *Re-planning cost* vs a discrete
pipeline re-run per cycle, showing the discrete per-cycle cost blowing the real-time
budget as N grows.

**Scoping (revised 2026-07-16):** no longer optional. §5.2 makes this the paper's
headline — the receding-horizon variant in its online form, with the outdated-mesh demo,
goes *in* the RA-L paper. The re-planning-cost experiment (iii) is promoted from a nice
figure to a core result, because "warm-starting beats re-solving" is the claim that
distinguishes this work from both NeOF (refits its field) and the discrete pipeline
(re-solves from scratch). Add NeOF's refit cost as a third curve there.

### 6.5 Venue and scoping

**RA-L** (with ICRA/IROS presentation option) is the right target: the contribution mix
(operator + planning + system) fits, review is fast, and the thesis pipeline supplies the
system section. One paper, not two — and after §5.2, the trajectory + system layer is no
longer the *strongest* part of the paper but the **load-bearing** one: C1–C3 alone would
now be an incremental re-run of NeOF on NVPS's home turf, and would likely be rejected in
the venue where NeOF already sits. The receding-horizon variant of C4 is therefore
**mandatory, not optional**, ideally in its online form (§6.4) with the outdated-mesh
demo; the spline version stays future work unless receding-horizon results are strong.
Note the RA-L risk of drawing a NeOF author as reviewer is real — cite them generously
and compare head-to-head rather than working around them. The former fallback of an
operator-centric cut (C1+C2, heavier theory, ModelNet evaluation) for CGF/SGP/3DV is now
weaker but not dead: it would have to lead with C2's exploitation analysis and the
distance-invariance theory, which remain unclaimed.

### 6.6 Risks, stated plainly

- *Novelty risk is now the top risk, not an afterthought.* §5.2 removed the plan's
  original headline. If the trajectory/online results (§7 Stage B) do not materialize,
  there is **no paper** at RA-L — the offline half alone is too close to NeOF. Treat
  Stage B as go/no-go: if receding-horizon + warm-start cannot beat a discrete re-solve on
  wall-clock *and* match it on coverage, stop and reconsider the venue rather than padding
  the offline section.
- *The literature must be re-swept before drafting.* This plan asserted a false gap for
  months because nobody searched for "camera placement" (the term the graphics/vision side
  uses) as opposed to "inspection planning" (the robotics term). Sweep both vocabularies —
  and check RA-L/ICRA/IROS/VISAPP 2025–2026 — before the draft is frozen.
- *The +0.003 wreck win is one seed at a saturated V.* A green light, not a result. The
  paper's claim must be built at tight standoffs, high V, multi-seed.
- *NVPS generalization:* trained on ShapeNet at one radius; happens to transfer well to
  the wreck, poorly to thin ModelNet meshes. C2's self-calibration is the systematic
  answer; report calibration error per mesh.
- *Local minima remain* even with good surrogates (lamp V=1). Greedy init + restarts is
  the mitigation; report variance.
- *Licenses:* HPRO code CC BY-NC 4.0, NVPS "research use" — both fine for a paper; check
  before releasing derived code under a permissive license.

---

## 7 · Order of work

*Re-ordered 2026-07-16: **trajectories/online first** (§5.3's open ground, and the
headline), offline placement second (still a real contribution, but now a comparison
against NeOF and against this thesis's own combinatorial pipeline rather than a novelty
claim). Steps 1–3 are shared infrastructure needed under either framing, so they stay in
front.*

**Stage A — operator infrastructure (prerequisite for everything).**

1. Port `diagnostics/diag_nvps_stage2.py` into a proper `VisibilityBackbone` interface in
   `hpro/` (HPRO | NVPS | ensemble); fix `HPRO.py:110`; wire δ/α through `HPRO_limited`
   and ablate on the Stage-1 harness.
2. Add the normal gate + quality weighting; switch the eval camera to realistic tight
   standoffs. Re-establish greedy vs refine at V ∈ {10, 20, 40} on the wreck, 5 seeds —
   this validates the operator layer the trajectory work stands on.
3. Implement self-calibration (Platt on ~50 audit poses; optionally VisNet-head
   fine-tune); make the soft-vs-GT calibration gap a logged metric in `eval_multi.py`.
   This is C2's control and is needed before trusting any trajectory objective.

**Stage B — trajectories and online (the headline, §6.4).**

4. Receding-horizon trajectory variant; route with the existing VRP; compare
   coverage-per-path-meter against the thesis pipeline; Isaac Sim rollout figure.
5. Online construction (§6.4): demand-weighted warm-started re-optimization on top of
   step 4's receding-horizon code (demand updates from executed poses + shifted warm
   start + budgeted steps per cycle); run the **outdated-mesh experiment** in Isaac Sim.
   The re-planning-cost figure (warm-start vs discrete re-solve vs NeOF field refit, as a
   function of V/N) is the quantitative core of the headline claim.

**Stage C — offline placement, positioned as comparison.**

6. Hybrid + restarts + annealing; ModelNet/TOSCA sweep; baselines: CMA-ES,
   500-candidate-greedy, **NeOF** (public code), and the thesis's set-cover + VRP
   pipeline end-to-end (coverage, path length, wall time).
7. Optional Phase 2: distance-aware NVPS retraining (shell-sampled viewpoints,
   log-distance input) if close-range interior views prove important on the wreck.

---

## 8 · Reproducing everything

```bash
ipy=/home/troja-lab-02/miniconda3/envs/isaaclab/bin/python

# §3.2 HPRO-backbone diagnostics (lamp by default; pass a mesh path for the wreck)
$ipy hpro/diagnostics/diag_stage2.py
$ipy hpro/diagnostics/diag_stage2.py models/duke_of_lancaster_uk_clipped.glb

# §3.1 backbone accuracy probe (lamp + wreck, three radii, grad check)
$ipy hpro/diagnostics/probe_neuvis.py

# §3.2 NVPS-backbone go/no-go (wreck + lamp, V ∈ {1, 5})
$ipy hpro/diagnostics/diag_nvps_stage2.py
```

The NVPS scripts expect `hpro/external/neural-visibility/` and
`hpro/external/neuvis_00040.pth` (see §1 for download commands), overridable via the
`NEUVIS_DIR` / `NEUVIS_CKPT` env vars. Stage-1 results and reproduction commands for the
original frustum evaluation remain in [`RESULTS.md`](RESULTS.md).

### 8.1 Backbone interface (§7 step 1, added 2026-07-16)

```bash
$ipy hpro/smoke_test.py            # 7/7, includes backbone-equivalence checks
```

`backbones.py` exposes the occlusion models behind one interface, and
`visibility_layer.py` gates any of them with the shared camera model from `frustum.py`:

```python
from backbones import make_backbone
from visibility_layer import GatedVisibilityLayer

backbone = make_backbone("nvps", device="cuda")   # "hpro" | "nvps" | "ensemble"
backbone.prepare(pts_np, normals_np)              # one-time per cloud (~24 ms)
layer = GatedVisibilityLayer(backbone, fov_h=..., fov_v=..., near=..., far=...)
w = layer(pts_t, viewpoints, rot_6d)              # (V, N), differentiable
```

Correctness is pinned by equivalence rather than by re-deriving the maths: `HPROBackbone`
+ `GatedVisibilityLayer` reproduces the Stage-1-validated `HPRO_limited` **bit-exactly**
(max |Δ| = 0), and `NVPSBackbone` reproduces `diagnostics/diag_nvps_stage2.nvps_w_combined`
bit-exactly — so every number in §3 stands unchanged. `make_backbone` filters kwargs per
backbone, so one config dict can drive a sweep over all three.
