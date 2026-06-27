# Frustum-Restricted HPRO — Stage 1 Results

First results subsection for the paper. Numbers below are from `eval_frustum.py`
on the bundled `lamp_0001.off` (ModelNet-40), **12 poses on a sphere, 3000
surface points, CPU** (accuracy is device-independent; GPU is only needed for the
runtime numbers, see *Open items*). Reproduce: `python hpro/eval_frustum.py
--mesh_dir hpro --num_poses 12 --num_points 3000 --no_show`. Artifacts in
`hpro/results/` (`summary.txt`, `hpro_eval.csv`, `f1_vs_*.png`).

## What was built (Stage 1)

`HPRO_limited` extends the HPRO operator (Katz & Tal, CGF 2025) — full-360°
differentiable point-cloud visibility for a **single** viewpoint at **fixed
distance** — to a **depth-limited pyramid frustum**, differentiable in **both**
camera **position and orientation** (6D rotation, Zhou et al. 2019). The frustum
is a product of six smooth sigmoid gates (4 walls + near + far); the combined
score is `relu(w_HPRO) × f_frustum`. This is the building block the thesis's
Future Work named, and the prerequisite for Stage 2 (joint multi-viewpoint
coverage) and Stage 3 (differentiable coverage path planning).

## Headline numbers

Default config: `gamma=-e^-7≈-9.1e-4`, `sharpness=50`, `k=10`, `thresh=0.5`.
Ground truth = inside-frustum ∩ unoccluded (ray-cast vs. the mesh).

| Method | Precision | Recall | F1 | IoU | op time (CPU, 3k pts) |
|---|---|---|---|---|---|
| **HPRO_limited** (default) | 0.925 | 0.900 | **0.911** | 0.838 | ~107 ms |
| Baseline: HPRO + hard frustum cull (same γ) | 0.959 | 0.788 | 0.863 | 0.764 | ~94 ms |
| Best baseline over γ (γ=-e⁻⁸) | — | — | 0.907 | — | — |

**HPRO_limited (F1 0.911) beats even the best-tuned hard-cull baseline (0.907)** —
the soft frustum recovers true points near the boundary that the hard cull drops
(recall 0.900 vs 0.788), at a small precision cost. The headline result: the
**differentiable** frustum operator matches/exceeds the non-differentiable
reference at **no accuracy cost** — exactly what makes it usable for the
gradient-based viewpoint optimisation in Stages 2–3.

## γ is the critical hyper-parameter (`f1_vs_gamma.png`)

`gamma` must be negative and **close to 0**; F1 vs `-ln|γ|` is sharply peaked:

| `-ln\|γ\|` | 3 | 4 | 5 | 6 | **7** | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.15 | 0.38 | 0.66 | 0.81 | **0.89** | 0.87 | 0.77 | 0.74 | 0.73 |
| recall | 0.09 | 0.25 | 0.51 | 0.72 | 0.85 | 0.91 | 0.94 | 0.94 | 0.95 |

As |γ|→0, recall rises monotonically but precision falls, so F1 peaks at γ≈−e⁻⁷
(consistent with the paper's "γ should sit slightly closer to 0"). A naive
default (γ=−e⁻³) gives a useless F1 of 0.15 — **γ tuning is essential** and is the
main practitioner takeaway. Both methods peak together (see figure), confirming
the soft frustum tracks the hard cull at the optimum and degrades *more
gracefully* away from it.

## Other hyper-parameters (sweeps at the tuned γ)

- **frustum_sharpness**: F1 rises 0.41→0.91 over 10→200 (sharper ≈ hard cull).
  We keep **50** as a deliberate accuracy/optimizability compromise: higher
  sharpness = better accuracy but steeper sigmoids = smaller gradient basins for
  Stage-2 viewpoint optimisation.
- **thresh**: peaks at **0.5** (F1 0.911) — the natural midpoint of the [0,1]
  combined score.
- **k** (top-k): flat across 5–40 (0.86–0.89); **10** is fine.

## Framing

**Operator angle.** Frustum-restricted HPRO keeps HPRO's differentiability while
adding a depth-limited FOV, and matches a hard cull's accuracy (F1≈0.91). The γ
sweep characterises the recall/precision trade-off; sharpness controls the
accuracy↔gradient-smoothness trade-off.

**Inspection-planning angle.** Each pose's `w_combined` is a differentiable
per-point coverage signal — the quantity Stage 2 optimises jointly over a set of
viewpoints (soft set-cover `1 − Π_v (1 − w^{(v)})`) to replace the discrete
sample → ECSC stages of the current pipeline.

## Known caveats (carried into Stage 2)

- Expected lower recall on silhouettes / deep concavities (HPRO's documented
  failure mode); accuracy improves with denser clouds.
- Free-position optimisation can collapse the viewpoint into the cloud (the paper
  fixes the radius); Stage 1 only measures fixed poses — Stage 2 will add a
  stand-off/radius regulariser.
- `float64` throughout for CPU/GPU parity.

## Open items

- **Multi-mesh**: validated on `lamp_0001.off` so far; the harness takes
  `--mesh_dir`, so scale to a ModelNet-40 subset for the paper's mean±std table.
- **GPU runtime**: the ~107 ms/pose here is CPU at 3k points; rerun on a GPU node
  (`qsub -v CONDA_ENV=<gpu-torch-env> scripts/metacentrum/run_hpro_eval.sh`) at
  10k points for the timing numbers.
