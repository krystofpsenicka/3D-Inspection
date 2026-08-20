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
| 3 | E16 `--arm root` **re-run after the fix** | ~15 min | relaxation question stays open (acceptable) |
| ~~4~~ | ~~E12 `--mode orderings`~~ | **DONE** (`b8659bd`) | — |
| 5 | paste results, page count, rebuild | ~1 h | — |
| — | E16 `--arm close` / `--arm budget` | 2–16 h | **does not fit. Skip.** |

**The paper is submittable as it stands** (thesis `125bb7d`) — §5 now claims
only what is measured. The one remaining run is optional and cheap: re-running
`--arm root` after the fix in `158afe3` upgrades "we leave it open" to a
measured answer. ~15 min, because a true LP relaxation solves in seconds where
the buggy version was running 356 s MIP solves. Do not add
`--arm close`/`--arm budget`; they do not fit.

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

## 2–3. E16 root LP — ran, but the first attempt measured the wrong thing

**What happened.** `--arm root` completed at `wp=10` for 4 seeds (`442f346`),
but every row recorded `n_binaries = 0`. PuLP's `LpVariable` constructor
normalises `cat=LpBinary` to `cat=LpInteger` with bounds `[0,1]`, so the test
`v.cat == pulp.LpBinary` never matched, **nothing was relaxed**, and
`prob.solve()` ran branch-and-bound. The `lp_bound_*` columns in
`results/e16_vrp_cut_strength/` are MIP optima, not relaxation bounds. Fixed in
`158afe3`, which now also fails the row loudly instead of silently mislabelling.

**What the existing rows are still good for.** Within every seed the four
configs return a *bit-identical* optimum:

| seed | none | reach | pair | both |
|---|---|---|---|---|
| 7 | 53.086876 | 53.086876 | 53.086876 | 53.086876 |
| 42 | 53.963955 | 53.963955 | 53.963955 | 53.963955 |
| 123 | 54.092899 | 54.092899 | 54.092899 | 54.092899 |
| 2024 | 58.271470 | 58.271470 | 58.271470 | *(not run)* |

A valid cut cannot change the integer optimum, so this is a clean **validity
check** and both cuts pass it. That is now in the paper (thesis `125bb7d`),
replacing an assertion with evidence. It is *not* the tightness measurement.

**Re-run (optional, ~15 min).**

```bash
git pull   # need 158afe3
python -m experiments.e16_vrp_cut_strength --arm root \
    --waypoints 10 --seeds 42 -v          # smoke test: expect n_binaries > 0
```

If `n_binaries` is still 0 the row now reports `relaxation_failed` — stop, and
keep the paper as it is. Otherwise:

```bash
until python -m experiments.e16_vrp_cut_strength --arm root --resume \
    --waypoints 10 15 20 25 50 --seeds 42 123 7 2024 314; do echo restart; sleep 3; done
python -m experiments.e16_vrp_cut_strength --arm root --plots_only
```

**Note:** `--resume` will skip the existing (bad) `wp=10` rows. Delete them
first: `rm results/e16_vrp_cut_strength/raw/arm=root_wp=10_*.json`.

Then apply Variant A or B from [`E16_RUNBOOK.md`](E16_RUNBOOK.md) §6 — but note
both were written against the paper's *previous* wording. §5 now says "Whether
they tighten the LP relaxation is a separate question our experiments do not
settle, and we leave it open"; replace that sentence with the measured result.

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
