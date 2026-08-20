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
| 2 | E16 smoke test | ~10 min | untested code fails 40 min in |
| 3 | E16 `--arm root` | ~1 h | claim stays merely softened |
| ~~4~~ | ~~E12 `--mode orderings`~~ | **DONE** (`b8659bd`) | — |
| 5 | paste results, page count, rebuild | ~1 h | — |
| — | E16 `--arm close` / `--arm budget` | 2–16 h | **does not fit. Skip.** |

E12 is done, so **E16 `--arm root` (~1 h) is the only run left**, leaving ~10 h
of slack. Do not add `--arm close`/`--arm budget` with that slack; see §2–3.

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

## 2–3. E16 root LP *(smoke test, then ~1 h)*

E16 has never been executed. Smoke-test before committing an hour:

```bash
python -m experiments.e16_vrp_cut_strength --arm root \
    --waypoints 10 --seeds 42 -v
```

Expect 4 rows (`none`/`reach`/`pair`/`both`), each printing
`LP bound = ... (norm) = ... m`. If it crashes, **stop and fall back to task 1**
— do not debug new code on a deadline.

If it works:

```bash
until python -m experiments.e16_vrp_cut_strength --arm root --resume \
    --waypoints 10 15 20 25 50 --seeds 42 123 7 2024 314; do
  echo restart; sleep 3
done
python -m experiments.e16_vrp_cut_strength --arm root --plots_only
```

**Reading the result.** The root LP bound is where "does this cut tighten the
relaxation" is *defined* — no branch-and-bound, no gap closure needed.

- *All four agree within seed noise* → restore the full §5 sentence and cite the
  root-LP numbers instead of the dual bounds. Best case: the existing claim,
  properly evidenced.
- *A cut's bound is strictly higher* → the §5 sentence and the conclusion line
  ("The two routing cuts, however, do not measurably improve the solve at the
  budgets we ran") are both **wrong** and need rewriting to "the cuts tighten
  the root relaxation by X% but this does not translate into better incumbents
  within the budgets we can afford". Small, positive rewrite — but budget 30 min.

Do **not** start `--arm close` (2–6 h) or `--arm budget` (up to 16 h). They do
not fit, and Step 3 alone is a publishable answer.

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
