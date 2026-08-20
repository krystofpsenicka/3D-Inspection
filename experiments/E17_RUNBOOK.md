# E17 runbook — plus the one E16 arm that was never run

Two jobs for the GPU machine. Both answer reviewer objections that the current
camera-ready text can only *concede*, not close.

Run them in this order. Job 2 is cheap and decisive; job 1 is the one the paper
actually needs.

---

## Job 1 — E17: ablate the 4D GPU space-time A*

### Why

Reviewer 4 item 4 names two contributions the paper claims but never ablates:
the β-aware tour bound + forbidden-pair cuts, and the 4D GPU space-time A*.
Reviewers 1 and 2 say the same thing more generally ("the individual impact is
not sufficiently isolated"). E16 closed the first — the cuts are inert. Nothing
closes the second, and it is the contribution attached to the stage that eats
84% of end-to-end wall-clock.

What the paper has today is not an ablation:

* E10's timing breakdown shows MAPF is expensive. That is not evidence that
  *our* MAPF is good.
* E12 (coordinated vs independent) isolates the **continuous-time conflict
  filter**, which lives in `committed_motion.py` — a different contribution.

So the two claims attached to `space_time_search.py` are both untested:

* **C1** — parallel frontier expansion (Zhou & Zeng 2015, ported to a 4D
  space-time state) beats expanding one state at a time.
* **C2** — it pays for that with bounded suboptimality. `space_time_astar_gpu`
  expands every frontier cell with `f <= f_min + f_threshold_delta`, so paths
  are `f_threshold_delta`-suboptimal. The paper admits this in one sentence
  (§4.5, "the parallel expansion is nevertheless the usual bounded relaxation")
  and never quantifies it.

Until this runs, the camera-ready has to carry an explicit concession sentence
saying the GPU A* is evaluated only in aggregate. Getting numbers lets us delete
that sentence.

### What it does

`experiments/e17_stastar_ablation.py`, two arms.

**`--arm legs`** (decisive). Runs the real pipeline on Duke, then intercepts the
first `--max_legs` single-agent A* queries the route planner issues and solves
each one three ways on identical inputs — same grid, same start/goal/`t_offset`,
same *live* `CommittedMotion`:

| variant   | what it is                                                        |
|-----------|-------------------------------------------------------------------|
| `gpu_pf`  | production, `f_delta = 2.0` — expands a whole f-layer per iteration |
| `gpu_seq` | same code, same arrays, `f_delta = 0.0` — one f-layer at a time     |
| `cpu_seq` | sequential `heapq` A* over the same 4D state: same cost model, heuristic, obstacle test, continuous-time filter and holdable-goal rule. NumPy expands a state's 27 successors at once, so it is a competent CPU A*, not a strawman. |

`gpu_pf` vs `cpu_seq` is the "GPU-resident" number a reviewer looks for.
`gpu_pf` vs `gpu_seq` is the one that attributes the speedup to the *algorithm*
rather than to CUDA — nothing else changes between them. Costs from all three
make C2 measurable: `gpu_seq`/`cpu_seq` are optimal for the given filter, so
`cost(gpu_pf)/cost(gpu_seq) - 1` is the **realised** price of the relaxation, as
opposed to its `f_delta` bound. The summary also cross-checks that `cpu_seq` and
`gpu_seq` agree on cost, which validates both as the reference.

**`--arm fdelta`** (end-to-end consequence). Full Duke pipeline at K=5, sweeping
`f_threshold_delta ∈ {0, 0.5, 2, 8}`, reporting MAPF wall-clock, the β-blended
objective, makespan, true collision pairs and A* failures. Answers whether the
relaxation costs anything that survives to mission level, and whether turning it
up buys wall-clock.

No library code is touched. `f_threshold_delta` is already a parameter of
`space_time_astar_gpu`; `route_planner` just never passes it, so both arms wrap
`VRP.mapf.route_planner.space_time_astar_gpu` and restore it in a `finally`.

### How to run

```bash
# ~30-60 min depending on --cpu_timeout; run this first
python -m experiments.e17_stastar_ablation --arm legs --resume

# hours
python -m experiments.e17_stastar_ablation --arm fdelta --resume

# re-print the summary from raw JSON without recomputing
python -m experiments.e17_stastar_ablation --plots_only
```

Defaults: Duke, 3 seeds (42/123/7), K=5, 20 intercepted legs per seed, 120 s CPU
budget per leg. Useful knobs:

* `--cpu_timeout 300` if too many legs report `outcome: timeout`.
* `--no_cpu` for a fast GPU-parallel-vs-sequential-only pass.
* `--max_legs 40` for a bigger corpus (cost is roughly linear in it).

A CPU timeout is recorded as a result, not dropped: the reported speedup then
becomes a lower bound, and the summary says so.

### How to read the outcome

All three outcomes are publishable — E16 is the precedent for reporting the
unflattering one.

* `gpu_pf` ≫ `cpu_seq` **and** `gpu_pf` > `gpu_seq`, small cost inflation →
  the contribution stands; report the speedup and the realised price.
* `gpu_pf` ≈ `gpu_seq` → the parallel frontier is not what makes it fast, the
  GPU port is. Say so, and drop "parallel-frontier" from the claimed
  contribution.
* `gpu_pf` faster but materially worse paths → `f_delta = 2.0` is mistuned;
  the `fdelta` arm then says what to ship.

---

## Job 2 — E16 `--arm close`: do the cuts make the *solver* faster?

### Why

E16 so far only ran `--arm root`. That answered "do the cuts tighten the
relaxation" (no). It did **not** answer "do they make the solver faster", which
is the remaining way the cuts could still be a real contribution.

The `close` arm already exists and measures exactly that: branch-and-bound on
instances small enough that the 5% gap is actually reachable, reporting
`solve_time`, `terminated_early`, and a correctly recomputed `true_gap` (the
legacy `gap` field is invalid — it divides a metre incumbent by a normalised
dual bound). It has never been run.

### Read this before running it

The `root` results already constrain the answer, and the constraint is harsh.
Model sizes per configuration, from the raw JSON:

| \|W\| | n_vars (all cfgs) | rows `none`→`pair` | root LP bound |
|---|---|---|---|
| 10 | 561   | 535 → 554     | identical to 6 d.p. |
| 15 | 1216  | 1170 → 1290   | identical |
| 20 | 2121  | 2055 → 2143   | identical |
| 25 | 3276  | 3190 → 3192   | identical |
| 50 | 12801 | 12615 → 12615 | identical |

Two things follow:

1. **`reach` never removes a single arc at any size** — `n_vars` is 561/1216/…
   for every configuration. There is nothing for a speed benefit to come from.
2. **At |W| = 50 the four configurations build a byte-identical model.** So at
   our production size the cuts *cannot* make the solver faster, by
   construction. No `close` or `budget` result can rescue that.

The speed hypothesis is therefore only live at |W| ≤ 25, and only for `pair`,
and only through branching/propagation effects — because the root bound is
unchanged, so the extra rows buy no dual improvement while still costing
something in every node LP factorization. That prior is unfavourable but it is
not proof, and "time to proven optimality" is a genuinely different question
from "bound tightness". Worth the run.

### How to run

```bash
python -m experiments.e16_vrp_cut_strength --arm close --resume
```

Defaults: |W| ∈ {10, 15, 20, 25}, 5 seeds, 4 configs, cuOpt, 900 s cap per
solve. Worst case ~20 h; instances that close early stop early, so expect much
less. Trim with `--waypoints 10 15 20 --seeds 42 123 7` or `--time_limit 300`
if it drags.

Then:

```bash
python -m experiments.e16_vrp_cut_strength --plots_only
```

The `close` table prints `early/gap/runs` per configuration. A real result is
`both` proving optimality in materially less wall-clock than `none` **on the
same seeds**, with the two counts agreeing.

### What it can and cannot change in the paper

Even a clean win at |W| ≤ 25 does **not** overturn §5's conclusion, because that
conclusion is about production size, where the models are identical. It would
change the wording from "inert" to "inert at production sizes, though they close
small instances faster" — a narrower but honest claim, and one worth having.
