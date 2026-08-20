# Pending experiments — ITAT 2026 camera-ready

**Written 2026-08-20 with ~12 h to the resubmission deadline.** Ordered by
value-per-hour, not by completeness. Deep detail for E16 is in
[`E16_RUNBOOK.md`](E16_RUNBOOK.md); read this file first.

**Paper:** `~/thesis/itat/paper.tex`, branch `itat-camera-ready` in the thesis
repo (`f623aa5` = as-submitted, `ffd9288` = camera-ready).
`git diff HEAD~1 -- itat/paper.tex` shows the revision.

**Machine:** GPU box, `inspection` env.

```bash
conda activate inspection && cd /path/to/3D-Inspection
```

---

## Time budget

| # | task | cost | risk if skipped |
|---|---|---|---|
| 0 | make the repo public | minutes | **paper states a false claim** |
| 1 | soften the §5 relaxation claim | minutes | unsupported claim stands |
| ~~2~~ | ~~E16 smoke test~~ | **DONE** | — |
| ~~3~~ | ~~E16 `--arm root`~~ | **DONE** (`a0fb09f`) | — |
| ~~4~~ | ~~E12 `--mode orderings`~~ | **DONE** (`b8659bd`) | — |
| 5 | paste results, page count, rebuild | ~1 h | — |
| — | E16 `--arm close` / `--arm budget` | 2–16 h | **does not fit. Skip.** |

**All experiments are done.** The paper is complete (thesis `9c134a4`): no
placeholders, no unsupported claims, 13 pages, 0 undefined references. Nothing
below needs running. `--arm close` / `--arm budget` are unnecessary — the root
LP arm answered the question outright.

---

## 0. Make the repo public *(do this first — minutes)*

§5 *Setup* claims code, scripts and raw JSON "are at
`https://github.com/krystofpsenicka/3D-Inspection`". Anonymous fetch of that URL
returns **404** — the repo is private. R4.17 asked specifically about code
release, so this is the claim most likely to be checked. This branch is already
pushed; flipping visibility is the only step. Otherwise soften the sentence to
"available on request".

## 1. Soften the §5 relaxation claim *(minutes, zero compute)*

§5 *Ablation of the routing cuts* currently says the cuts

> neither **tighten the relaxation** nor improve incumbents measurably.

The second half is what E11/E15 measured. **The first half is not measured by
anything we have run** — it rests on B&B dual bounds at timeout, which came from
a misreported field (`E16_RUNBOOK.md` §0). If E16 does not get run, cut the
relaxation half and claim only the incumbent half. That is the zero-risk move
and it costs one sentence.

## 2–3. E16 root LP — DONE *(`a0fb09f`; first attempt `442f346` was buggy, see `158afe3`)*

100 LP solves (5 sizes x 5 seeds x 4 configs), all successful, `n_binaries > 0`
throughout. **The cuts are inert, and E16 says why.**

| wp | rows added by `pair` | arcs dropped by `reach` | LP bound spread across configs |
|---|---|---|---|
| 10 | 19 | 0 | < 1e-8 relative |
| 15 | 120 | 0 | < 1e-8 |
| 20 | 88 | 0 | < 1e-8 |
| 25 | 2 | 0 | < 1e-8 |
| **50** | **0** | **0** | **< 1e-8** |

- The β-aware filter drops **zero arcs at every size**: `T_tour_ub` is never
  tight enough for `c[d_v,i] + c[i,j] + c[j,d_v]` to exceed it.
- The forbidden-pair cuts fire only on small instances, and the count collapses
  with size.
- At 50 waypoints all four configs build the **bit-identical model** (12801
  vars, 12615 rows) — so E11/E15 were comparing one formulation four times, and
  their spread is solver nondeterminism.

Written into the paper (thesis `9c134a4`), which now answers the relaxation
question outright instead of leaving it open, and gives the mechanism. The
Variant A/B drafts in [`E16_RUNBOOK.md`](E16_RUNBOOK.md) §6 are superseded.

## 4. E12 orderings arm — DONE *(`b8659bd`, results in `results/e12_orderings_fixed/`)*

Written into the paper (thesis commit `8c4fa97`); the `%%% TODO camera-ready`
placeholder is gone. The result, 3 seeds on Duke, `n ∈ {1,5,10,20}`:

| n | objective (s) | makespan (s) | collision pairs | min sep (m) | seeds w/ collisions |
|---|---|---|---|---|---|
| 1 | 537.6 ± 26.7 | 190.3 ± 16.9 | 1.00 ± 1.00 | 0.52 ± 0.28 | 2/3 |
| 5 | 536.7 ± 26.3 | 190.0 ± 16.9 | 0.00 ± 0.00 | 0.74 ± 0.01 | 0/3 |
| 10 | 536.7 ± 26.3 | 190.0 ± 16.9 | 0.00 ± 0.00 | 0.76 ± 0.04 | 0/3 |
| 20 | 536.6 ± 26.2 | 190.0 ± 16.9 | 0.00 ± 0.00 | 0.74 ± 0.01 | 0/3 |

The orderings are a **feasibility** lever, not a cost lever: the objective is
flat (0.2% across the whole sweep) but a single ordering leaves residual
collisions in 2 of 3 seeds, below `d_safe = 0.70 m`. Everything saturates at
n = 5 — the n = 5/10/20 plans agree to three significant figures — so the
default of 20 costs 4x the MAPF wall-clock for nothing. `astar_fail = 0`
throughout.

Worth knowing for any future MAPF work: this is also independent evidence that
the coordination layer alone is not sufficient; it needs more than one ordering
to *find* the conflict-free assignment that E13 then audits.

## 5. Final pass

- Paper is **13 pages**; the allowance was +2 excluding citations and the body
  grew +2.5. Tasks 3 and 4 add ~0.3 page. Cheapest cuts, in order: the Isaac Sim
  figure (§5), the E09 paragraph, the E10 lower-bound derivation prose.
- Check the IWCIDM camera-ready mechanics — copyright form, and whether PDF/A is
  required. `paper.xmpdata` and `pdfa.xmpi` exist in `itat/` but the current
  build does not use them.
- Rebuild and confirm: `latexmk -pdf paper.tex`, then
  `grep -c undefined paper.log` must be 0.

---

## Already done — no action needed

All four reviews were addressed in `ffd9288`. Every
[`CAMERA_READY_PLAN.md`](CAMERA_READY_PLAN.md) item A–F has been run and written
up: E11/E15 (cut ablation), E12 collisions arm, E13 (post-smoothing audit), E14
(|C|/|P| scaling), E05/E06/E01 at 10 seeds, E10 at 5 seeds over 14 meshes, E08
gap-vs-timeout.

One standing caveat for any future analysis: the `gap` / `gap_reached` fields in
E08/E11/E15 raw JSON divide a **metres** incumbent by a **normalised** dual
bound and are not optimality gaps. `E16_RUNBOOK.md` §0 has the worked example —
the recomputed gap on E15/`both` is ~13%, not the ~99.7% logged. Never quote
them.
