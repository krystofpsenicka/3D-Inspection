# E16 runbook — does the VRP cut relaxation actually help?

Machine: the GPU box with the `inspection` conda env (RTX 3090). Everything here
is run-only; the code is already on this branch.

```bash
conda activate inspection
cd /path/to/3D-Inspection
```

---

## 0. Why E11/E15 could not answer the question

Two independent problems, both fixed in `experiments/e16_vrp_cut_strength.py`.

**(a) Nothing ever terminated on the gap.**

| experiment | limit | actual solve time | terminated on |
|---|---|---|---|
| E08 | 120 s | `120.85 ± 0.25` s (59 runs) | wall clock |
| E11 | 120 s | `120.3–120.9` s (40 runs) | wall clock |
| E15 | 1800 s | `1801.0 ± 0.2` s (12 runs) | wall clock |

cuOpt *was* given `CUOPT_MIP_RELATIVE_GAP = 0.05`
(`VRP/vrp/mip_solver_gpu.py:124`), so it would have stopped itself had the gap
closed. It never did. Every cut config therefore returned "whatever B&B had
after N seconds", which is why all four looked identical.

**(b) The logged `gap` is not an optimality gap — it mixes units.**

`VRPResult.objective_value` is **metres** (`beta*T + (1-beta)*C`,
`VRP/vrp/vrp_solver.py:101`), while `best_bound` is cuOpt's bound on the
**normalised** model objective `beta*T/T_norm + (1-beta)*C/C_norm` with
`T_norm = T_lb`, `C_norm = K*T_lb` (`VRP/vrp/mip_model.py:171-172`). Dividing one
by the other pins `gap` at ~0.997 regardless of solve quality, and
`gap_reached` is therefore always `False`.

Worked example on the real E15 `both` means (`makespan=113.78`,
`total=484.35`, `bound=0.9117`, `K=5`, `beta=0.5`), assuming `T_lb = 100` m:

```
obj_mip  = 0.5*113.78 + 0.5*484.35/5 = 105.33 m     <- what cuOpt minimises
obj_norm = 105.33 / 100              = 1.0533
true gap = (1.0533 - 0.9117)/1.0533  = 13.4%        <- plausible
logged   = (299.06 - 0.9117)/299.06  = 99.7%        <- meaningless
```

So we are probably ~13–22% away, not ~100%. `T_lb` is not currently logged
anywhere, which is why the true number is still unknown — E16 computes it.

**(c) The ablation was ranked by the wrong objective.** E11/E15 compare
`beta*T + (1-beta)*C`; the model minimises `beta*T + (1-beta)*C/K`. At `K=5`
that over-weights total cost fivefold. Re-ranking the existing data under the
correct objective does **not** flip the null result (configs still overlap
within one σ), but E16 logs both columns so this is checkable.

---

## 1. Step 1 — root LP bound (cheap, decisive, run first)

"Does the cut tighten the relaxation" is *defined* by the root LP bound. This
needs no branch-and-bound and no gap closure, so it gives a clean answer even if
B&B never converges.

```bash
until python -m experiments.e16_vrp_cut_strength --arm root --resume \
    --waypoints 10 15 20 25 50 --seeds 42 123 7 2024 314; do
  echo "restart after OOM/crash"; sleep 3
done
python -m experiments.e16_vrp_cut_strength --arm root --plots_only
```

**Cost:** ~1 h (100 LP solves; the per-instance distance matrix dominates).
**Output:** `results/e16_vrp_cut_strength/raw/arm=root_*.json`, summary prints
LP bound in metres and normalised units, plus `% vs none`, model size
(binaries/rows) and LP time per config.

**How to read it**
- `reach` removes arcs → fewer binaries. If its LP bound rises, the filter is
  cutting off fractional mass, not just shrinking the model.
- `pair` adds rows → same binaries, more constraints. Its LP bound rising is a
  direct demonstration of cut strength.
- If **all four LP bounds agree to within seed noise, the cuts do not tighten
  the relaxation** — that is a complete and publishable answer, and Steps 2–3
  become optional. Say so in the paper and stop.

---

## 2. Step 2 — instances small enough that 5% is reachable

Only run if Step 1 shows a difference (or if you want the operational claim
"the cuts prove optimality faster"). A small instance the solver actually
closes is far more informative about cuts than a big one where all four time
out identically.

```bash
until python -m experiments.e16_vrp_cut_strength --arm close --resume \
    --waypoints 10 15 20 --seeds 42 123 7; do echo restart; sleep 3; done
python -m experiments.e16_vrp_cut_strength --arm close --plots_only
```

**Cost:** hard to predict — that is the point. 10–15 waypoints should close in
seconds to minutes; 20–25 may hit the 900 s cap. Budget 2–6 h for the command
above; add `--waypoints 25` and `--seeds 42 123 7 2024 314` only after you see
how fast the first sizes close.

**Output columns:** `obj_mip (m)`, `true_gap`, `solve_s`, and `early/gap/n` =
runs that stopped before the wall clock / runs whose recomputed gap is ≤ 5% /
total runs. Those first two counts should agree; a mismatch means the dual
bound came back unavailable (`best_bound == 0.0`).

**The result you want:** at a size where all configs close, compare
`solve_s`. Cuts that help show up as lower time-to-proof at equal objective.

---

## 3. Step 3 — full-size long run (optional, expensive)

Only if Steps 1–2 are suggestive and you want the 50-waypoint number.

```bash
until python -m experiments.e16_vrp_cut_strength --arm budget --resume \
    --waypoints 50 --seeds 42 123 --time_limit 7200; do echo restart; sleep 3; done
```

**Cost:** 2 seeds × 4 configs × up to 2 h = **up to 16 h**. Do not run five
seeds here. Expect it may still not close: lifted MTZ is a weak formulation and
50 customers × 5 vehicles min-max is exactly where it is weakest. If it does not
close, Step 1 is still your answer.

---

## 4. Step 4 — the other outstanding run: MAPF priority orderings

Unrelated to the cuts, but it is the remaining hole in the camera-ready. The
paper currently carries a two-line placeholder and a `%%% TODO camera-ready`
block at `itat/paper.tex:751`.

```bash
until python -m experiments.e12_mapf_ablation --resume --mode orderings \
    --seeds 42 123 7; do echo restart; sleep 3; done
python -m experiments.e12_mapf_ablation --plots_only
```

**Cost:** ~3 h on Duke at 3 seeds (1 trial ≈ 55 s, 20 ≈ 21 min per run).
**Then:** replace the placeholder sentence with the objective and MAPF
wall-clock per ordering budget, and say where the gain saturates.

---

## 5. Optional library fixes (not required by E16)

E16 recomputes everything itself, so none of these block the runs. Worth doing
before the next campaign so the raw JSONs are trustworthy on their own:

1. `VRP/core/types.py:24-31` — the `VRPResult.best_bound` docstring says
   "in meters". It is in normalised model units. Fix the docstring, or better,
   have the solvers also return `t_lb` so callers can convert.
2. `VRP/vrp/mip_model.py` — `build_vrp_mip` computes `T_lb` at line 154 and
   discards it. Attach it to the returned problem (`prob._t_lb = T_lb`) — this
   is non-breaking, both solvers unpack exactly two values.
3. `experiments/common/lb_sidecar.py:135` — `extract_cuopt_bound`'s docstring
   claims metres and the field it feeds is named `vrp_objective_best_bound_m`.
   Same unit error. **The paper does not use this field** (E10 uses the
   independent scipy 2-factor LP bound), so nothing published is affected.
4. `experiments/e11_vrp_cuts_ablation.py:119` and
   `experiments/e15_vrp_cuts_highbudget.py:110` — the `gap` / `gap_reached`
   computation. Either fix or drop the fields; leaving them invites the same
   mistake next time.

---

## 6. Drop-in paragraph variants for the paper

The camera-ready currently claims, in §5 *Ablation of the routing cuts*, that
the cuts "neither **tighten the relaxation** nor improve incumbents measurably".
The second half is what E11/E15 measured. The first half is not measured by
anything we have run.

Below are two ready-to-paste replacements. Both are the **same length or shorter
than the current paragraph** — the paper is at 13 pages against a +2-page
allowance, so do not let this section grow. Fill the `\PLACEHOLDER` marks from
`--arm root --plots_only`; the summary prints exactly the numbers needed
(`LP bound (m)` per config and `% vs none`).

Locate the paragraph with:

```bash
grep -n "Ablation of the routing cuts" ~/thesis/itat/paper.tex
```

---

### Variant A — root LP bounds agree (the null result holds)

Swap only the two marked spans; the rest of the paragraph is unchanged. This
replaces weak evidence (B&B dual bounds at timeout) with the right evidence (the
root relaxation), and the existing conclusion sentence stands as written.

Replace `with every mean inside one standard deviation of every other, dual
bounds agreeing to two decimals, and all 40 runs exhausting the budget.` with:

```latex
with every mean inside one standard deviation of every other and all 40 runs
exhausting the budget. The relaxation itself is unchanged: solving the root LP
of each variant (E16) gives bounds of \PLACEHOLDER, \PLACEHOLDER, \PLACEHOLDER
and \PLACEHOLDER\,m for \code{none}, \code{reach}, \code{pair} and \code{both},
i.e.\ within \PLACEHOLDER\% of each other --- neither cut removes fractional
mass the lifted MTZ formulation had not already excluded.
```

Everything after that, including "The cuts are valid and free to add, but at
affordable budgets they neither tighten the relaxation nor improve incumbents
measurably", stays. **No change to the conclusion.**

---

### Variant B — a cut's root LP bound is strictly higher

The current claim is then wrong. Replace the paragraph's last three sentences
(from "The cuts are valid and free to add" to "the natural one to make") with:

```latex
The relaxation, however, does tighten. Solving the root LP of each variant
(E16) raises the bound from \PLACEHOLDER\,m for \code{none} to \PLACEHOLDER\,m
for \code{\PLACEHOLDER} (${+}\PLACEHOLDER\%$), so the cut does remove fractional
mass that lifted MTZ admits. That strengthening simply does not reach the
incumbent within the budgets we can afford: branch-and-bound is still far from
closing after 1800\,s, and the $19$--$24\%$ the MIP gains over its warm start
comes from the search. The cuts are therefore a formulation improvement whose
practical value we cannot yet demonstrate --- on this model, at these sizes, the
binding constraint is the solver budget, not the relaxation.
```

**Also fix the conclusion** (§6). Replace:

```latex
The two routing cuts, however, do not measurably improve the solve at the
budgets we ran.
```

with:

```latex
The two routing cuts tighten the root relaxation but do not measurably improve
incumbents at the budgets we ran.
```

---

### If the result is mixed (one cut helps, the other does not)

Use Variant B but name only the cut that helps, and say plainly that the other
leaves the bound unchanged. Do not average them into a single claim — they are
independent additions and the paper presents them as such.

---

### Either way

Do not reintroduce the old `gap` / `gap_reached` numbers anywhere. Report solve
time hitting the limit (sound, already in the paper) and, if a gap is wanted,
the recomputed `true_gap` from E16. See §0.
