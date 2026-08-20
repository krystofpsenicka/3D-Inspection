# E16 arm=root — result (2026-08-20)

**Answer: Variant A holds — the routing cuts do NOT tighten the root LP relaxation.**

## What was run
- Smoke test (`--waypoints 10 --seeds 42`): passed, exit 0.
- Full sweep launched (`--waypoints 10 15 20 25 50 --seeds 42 123 7 2024 314`)
  but **stopped after the wp=10 block** by operator decision (see caveat below).
- Completed instances: wp=10, seeds **7, 42, 123** (all 4 cut configs each);
  seed 2024 partial (none/reach/pair, `both` interrupted); seed 314 not started.

## Result (the valid, within-instance comparison)
Within **every** completed wp=10 instance, all four cut configs give the
**identical** LP bound — spread = 0.0000 m:

| seed | none | reach | pair | both | spread (m) |
|---|---|---|---|---|---|
| 7   | 53.087 | 53.087 | 53.087 | 53.087 | 0.0000 |
| 42  | 53.964 | 53.964 | 53.964 | 53.964 | 0.0000 |
| 123 | 54.093 | 54.093 | 54.093 | 54.093 | 0.0000 |

Mean over the 3 complete instances (same seed set for every config):

| config | LP bound (m) | LP norm | % vs none | mean rows |
|---|---|---|---|---|
| none  | 53.715 ± 0.447 | 0.87490 | +0.00% | 535.0 |
| reach | 53.715 ± 0.447 | 0.87490 | +0.00% | 535.0 |
| pair  | 53.715 ± 0.447 | 0.87490 | −0.00% | 564.3 |
| both  | 53.715 ± 0.447 | 0.87490 | −0.00% | 564.3 |

`pair`/`both` add rows (up to 699 on some seeds) but remove no fractional mass:
the bound is unchanged to 4 decimals. This is the runbook's decisive Variant-A
case ("all four LP bounds agree to within seed noise → the cuts do not tighten
the relaxation").

> Note: the built-in `--plots_only` summary averages each config over whatever
> raw files exist, so it mixed the 4-seed none/reach/pair set with the 3-seed
> `both` set and printed a spurious −2.08% for `both`. That is a seed-composition
> artifact, not an effect. The table above compares configs on the **same**
> instances and is the number to quote.

## Caveat — why the sweep was stopped
`LP_TIME_LIMIT = 600` (code comment: "should finish in seconds"). The lifted-MTZ
root LP is numerically hard and hits the **full 600 s cap on essentially every
solve** (wp=10 seed 42 was a lucky 143 s; seeds 123/2024 hit 600 s). Bounds are
still valid (`status = Optimal`, all configs agree to 4 decimals), but the
100-solve sweep projects to **~14–15 h**, not the runbook's estimated ~1 h. With
the wp=10 answer already unambiguous, the sweep was stopped here rather than
blow the resubmission deadline. wp=15/20/25/50 not run.

## Paper — fill Variant A (E16_RUNBOOK.md §6)
Bounds of **53.7, 53.7, 53.7 and 53.7 m** for none, reach, pair and both, i.e.
within **0.0%** of each other. The existing §5 conclusion sentence stands
unchanged. Do **not** use Variant B.
