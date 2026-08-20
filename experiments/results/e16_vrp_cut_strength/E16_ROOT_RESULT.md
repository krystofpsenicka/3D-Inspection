# E16 arm=root — result (2026-08-20, corrected)

**Answer: Variant A — the routing cuts do NOT tighten the root LP relaxation.**
Now measured on the *true* LP relaxation (bug fixed in `158afe3`).

## Correction note
The first run (`442f346`) reported MIP optima, not LP bounds: PuLP normalises
`cat=LpBinary` to `cat=LpInteger`, so the old `v.cat == LpBinary` relax-test
matched nothing, `n_binaries = 0`, and `prob.solve()` ran branch-and-bound.
Fixed by relaxing every `!= LpContinuous` variable and failing loudly
(`relaxation_failed`) if nothing relaxes. This run has `n_binaries > 0` and
solves each LP in ~0 s (vs the ~356–600 s MIP solves before).

## What was run
Full sweep, no interruptions: `--arm root --waypoints 10 15 20 25 50
--seeds 42 123 7 2024 314` → 25 instances × 4 cut configs = **100 LP solves**,
all completed (exit 0). The prior buggy wp=10 rows were deleted first.

## Result — true root LP bound per cut config
Aggregated over 5 seeds at each size (every config over the identical seed set):

| wp | LP bound (m) | LP norm | binaries | rows (none→pair) | % vs none |
|---|---|---|---|---|---|
| 10 | 40.434 ± 3.366 | 0.69440 | 550   | 535 → 554     | +0.00% |
| 15 | 44.412 ± 2.851 | 0.72197 | 1200  | 1170 → 1290   | +0.00% |
| 20 | 47.269 ± 1.582 | 0.75708 | 2100  | 2055 → 2143   | +0.00% |
| 25 | 47.018 ± 2.757 | 0.76180 | 3250  | 3190 → 3192   | +0.00% |
| 50 | 57.305 ± 1.561 | 0.90897 | 12750 | 12615 → 12615 | +0.00% |

At **every** waypoint size, `none = reach = pair = both` to the printed
precision. `pair`/`both` add rows (e.g. +120 at wp=15) but move the LP bound by
**0.00%** — they remove no fractional mass the lifted-MTZ relaxation had not
already excluded. `reach` filters arcs without changing the bound either.

## Paper (§5 *Ablation of the routing cuts*)
Per the updated PENDING_EXPERIMENTS.md, §5 currently reads: *"Whether they
tighten the LP relaxation is a separate question our experiments do not settle,
and we leave it open."* Replace with the measured result (E16_RUNBOOK §6,
Variant A) — e.g.:

> Solving the root LP of each variant (E16) leaves the bound unchanged to the
> printed precision at every problem size we ran (10–50 waypoints): the `reach`
> and `pair` cuts move it by 0.00% relative to the uncut model. The cuts are
> valid and free to add, but at affordable budgets they neither tighten the
> relaxation nor improve incumbents measurably.

Do **not** use Variant B. Do not reintroduce the old `gap`/`gap_reached` fields.

## Also on record (still valid): cut-validity check
The earlier (mislabelled) run doubled as a validity check — within every seed
all four configs returned a bit-identical *integer* optimum, and a valid cut
cannot change the integer optimum. Both cuts pass. That is a separate fact from
the tightness result above.
