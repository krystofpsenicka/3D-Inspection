# Frustum-Restricted HPRO — Stage 1 Results

Seed write-up for the first results subsection of the paper. Numbers are filled
from a MetaCentrum GPU run of `eval_frustum.py` (`results/summary.txt`,
`results/hpro_eval.csv`, and the `f1_vs_*.png` plots). Until then, cells marked
`TODO` are placeholders.

## What was built (Stage 1)

`HPRO_limited` extends the HPRO operator (Katz & Tal, CGF 2025) — which computes
**full-360°** differentiable point-cloud visibility for a **single** viewpoint at
**fixed distance** — to a **depth-limited pyramid frustum**, differentiable in
**both** camera **position and orientation** (6D rotation, Zhou et al. 2019). The
frustum is a product of six smooth sigmoid gates (4 walls + near + far); the
combined score is `relu(w_HPRO) × f_frustum`.

This is the building block the thesis's Future Work named ("extending HPRO"), and
the prerequisite for Stage 2 (joint multi-viewpoint coverage) and Stage 3
(differentiable coverage path planning).

## Methodology

- **Operator** vs. **ground truth** per (mesh, pose): GT = points **inside the
  frustum** AND **unoccluded** (ray-cast against the source mesh,
  `frustum_gt.compute_ground_truth`).
- **Poses**: `--num_poses` viewpoints on a sphere of radius `--radius` around each
  unit-normalised mesh, each looking at the centroid (deterministic, seeded).
- **Metrics**: precision, recall, F1, IoU, and per-pose operator wall-clock,
  aggregated as mean ± std over poses/meshes.
- **Baseline (ablation)**: plain HPRO (full 360° self-occlusion) followed by a
  **hard** (non-differentiable) frustum cull — isolates what the differentiable
  soft frustum mask costs in accuracy.
- **Sweeps** (one-at-a-time around defaults `gamma=-e^-3`, `sharpness=50`,
  `k=10`, `thresh=0.6`): `f1_vs_gamma.png`, `f1_vs_sharpness.png`,
  `f1_vs_thresh.png`, `f1_vs_k.png`.

Reproduce: `qsub scripts/metacentrum/run_hpro_eval.sh` (runs the CPU smoke test
gate, then the evaluation; copies `results/` back).

## Headline numbers (TODO — fill from `results/summary.txt`)

| Method | Precision | Recall | F1 | IoU | op time |
|---|---|---|---|---|---|
| HPRO_limited (default cfg) | TODO | TODO | TODO | TODO | TODO ms |
| Baseline: HPRO + hard cull | TODO | TODO | TODO | TODO | TODO ms |

F1 gap (baseline − HPRO_limited): **TODO** — the price of differentiability.

## Framing

**Operator angle.** Frustum-restricted HPRO retains HPRO's differentiability while
adding a depth-limited FOV; the sweeps quantify sensitivity to `gamma`/`sharpness`/`k`
and locate the accuracy/smoothness trade-off (lower `sharpness` = smoother gradients
but softer boundary). Expect lower recall on silhouettes / deep concavities (HPRO's
documented failure mode) and improvement with denser clouds.

**Inspection-planning angle.** Each pose's `w_combined` is a differentiable
per-point coverage signal — the quantity Stage 2 will optimise jointly over a set of
viewpoints (soft set-cover `1 − Π_v (1 − w^{(v)})`) to replace the discrete
sample → ECSC stages of the current pipeline. The per-pose runtime here is the unit
cost of that optimisation's inner loop.

## Known caveats (carried into Stage 2)

- Free-position optimisation can collapse the viewpoint into the cloud (the paper
  avoids this by fixing the radius); Stage 1 only measures fixed poses. Stage 2 will
  add a stand-off / radius regulariser.
- `float64` throughout for CPU/GPU parity.
