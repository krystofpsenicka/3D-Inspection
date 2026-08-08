# ITAT 2026 camera-ready — experiment & code runbook

Branch: `camera-ready/experiments` (off `main`).

The paper *"Multi-Robot 3D Inspection with Limited Perception"* was accepted for
the IWCIDM / ITAT 2026 proceedings (poster). To be published we must address the
reviewers' recommendations (or justify not doing so) and may add up to 2 pages.

This file tracks **only the non-manuscript work** — code changes and GPU
experiment runs. Pure manuscript edits (missing definitions, citations, notation,
limitations/related-work discussion, GPU-implementation write-up) are handled
separately in the paper repo. **Reviewer #10 / R3.5 external-baseline comparison
is deliberately NOT done** (we argue against it in the text). Item **F (more real
meshes)** is kept as the last, optional task if time permits.

Everything below is **already coded** on this branch. When the GPU box is
available, this is a run-only list: activate the env and launch the commands.

---

## 0. Environment & run conventions

```bash
conda activate inspection          # per README
cd /path/to/3D-Inspection
```

All experiment scripts share the same conventions:

- `--resume` skips already-finished rows (one JSON per row under `results/<exp>/raw/`)
  and **exits non-zero on CUDA OOM** so an outer loop can restart with a clean
  process. Always launch long runs inside a restart loop:

  ```bash
  until python -m experiments.e12_mapf_ablation --resume; do
    echo "restart after OOM/crash"; sleep 3
  done
  ```

- `--plots_only` / re-running with `--plots_only` reads the raw JSONs and prints
  the summary tables (and figures where present) without touching the GPU — use
  it afterwards to regenerate numbers for the paper.
- `--seeds` overrides the seed list. Config defines `SEEDS_3=[42,123,7]`,
  `SEEDS_5=[…,2024,314]`, `SEEDS_10` (see `experiments/common/config.py`).
- Results land in `experiments/results/<experiment>/raw/` as JSON.

**Measured per-run costs on the target box (RTX 3090, Duke, K=5), from existing
E10 output:** sampling+visibility+set-cover ≈ 2.4 s; VRP (incl. distance matrix)
≈ 3 min; **coordinated MAPF @ 20 orderings ≈ 18 min** (this dominates); full Duke
pipeline ≈ 21 min. A full TOSCA-mesh pipeline ≈ 3 min. MAPF time scales ~linearly
with the priority-ordering budget (1 trial ≈ 55 s, 5 ≈ 4.5 min, 10 ≈ 9 min).

---

## 1. Code changes already made on this branch

| File | Change | Enables |
|---|---|---|
| `VRP/vrp/mip_model.py` | `build_vrp_mip(..., beta_aware_filter=True, forbidden_pair_cuts=True)`; each cut can be disabled independently (both off = plain lifted MTZ) | E11 |
| `VRP/vrp/mip_solver_gpu.py`, `mip_solver_cpu.py`, `_solver_base.py` | thread the two toggles through `solve()` | E11 |
| `VRP/vrp/vrp_solver.py` | `solve_vrp(..., beta_aware_filter, forbidden_pair_cuts)` | E11 |
| `experiments/common/pipeline_run.py` | **new** shared "run pipeline to routes" helper (mirrors E10 up to VRP; `do_routing`, `num_points`, `num_candidates` knobs) | E12, E13, E14 |
| `experiments/e08_vrp_alpha_blending.py` | records solver `gap`, `best_bound`, `gap_reached` per run | #13 |
| `experiments/e11_vrp_cuts_ablation.py` | **new** — VRP cut ablation | R2.5 / #4 |
| `experiments/e12_mapf_ablation.py` | **new** — priority-orderings sweep + coordinated-vs-independent collisions | #4 / #11 / #23 |
| `experiments/e13_collision_coverage_audit.py` | **new** — post-smoothing collision & coverage audit | #12 / R2.2 |
| `experiments/e14_cp_scaling.py` | **new** — \|C\| and \|P\| scaling / discretization sensitivity | #21 / #22 |

No existing experiment behaviour was changed except the additive E08 logging.

---

## 2. Work items → reviewer comments

| Item | Addresses | Kind | Where |
|---|---|---|---|
| **A. E11 VRP cut ablation** | R2.5, R4.4 (β-aware bound + forbidden-pair cuts) | new exp | §A |
| **B. E12 MAPF ablation** | R4.4 (orderings), R4.11, R4.23 (collision-free claim) | new exp | §B |
| **C. E13 collision+coverage audit** | R4.12, R2.2 (smoothing safety) | new exp | §C |
| **D. E14 \|C\|/\|P\| scaling** | R4.21, R4.22 | new exp | §D |
| **E. Re-runs for variance & seeds** | R4.19 (≥5 seeds), R4.20 / R2 (variance), R4.14 (E10 gap sources) | re-run | §E |
| **F. More real meshes (optional)** | R4.2 | data + runs | §F |

Reviewer #13 (E08 gap-vs-timeout) is covered by the E08 logging change + re-run in §E.

---

## A. E11 — VRP cut ablation  *(R2.5, R4.4)*

**Why.** The β-aware per-tour bound (used as a Desrochers–Laporte reachability
arc filter) and the forbidden-pair cuts are listed as contributions but never
ablated. E11 solves the *same* seeded instance four ways — `none` (plain lifted
MTZ), `reach` (filter only), `pair` (cuts only), `both` (default) — at a fixed
120 s budget and reports objective, dual bound, optimality gap, %-gap-reached,
and solve time. The story: cuts tighten the bound / shrink the gap / speed the
solve at equal wall-clock.

**Run** (Duke, cuOpt):
```bash
until python -m experiments.e11_vrp_cuts_ablation --resume \
    --waypoints 30 50 --seeds 42 123 7 2024 314; do echo restart; sleep 3; done
python -m experiments.e11_vrp_cuts_ablation --plots_only    # summary table
```
**Output:** `results/e11_vrp_cuts_ablation/raw/*.json`; summary prints obj / gap% /
solve_s / makespan / gap-hit% per config.
**Est. time:** ~1.5–2 h (40 solves × ~120 s + 10 distance-matrix builds).

---

## B. E12 — MAPF ablation  *(R4.4 orderings; R4.11 / R4.23 collision claim)*

**Why.** Two things. (1) *orderings arm*: sweep the priority-ordering budget
`n_priority_trials ∈ {1,5,10,20}` and report the β-blended objective, quantifying
the value of the "number of orderings tried" (a listed contribution). (2)
*collisions arm*: for each fleet size, plan **coordinated** (shared reservation
table) vs **independent** (each robot planned against an empty reservation) and
count geometric inter-robot collisions on the final smoothed trajectories. This
makes explicit that the paper's "<1 collision per mission for K≤4" is the
*independent baseline*, and that the coordinated pipeline output is ~0 — resolving
R4.23's ambiguity and supporting (a softened) collision-free claim.

**Run** (Duke; orderings arm is the long pole):
```bash
# both arms, default Duke, 5 seeds
until python -m experiments.e12_mapf_ablation --resume --mode both \
    --seeds 42 123 7 2024 314; do echo restart; sleep 3; done
python -m experiments.e12_mapf_ablation --plots_only
```
To split, use `--mode orderings` / `--mode collisions`. `--coord_trials 5`
(default) keeps the collisions arm affordable; E13 does the full 20-trial audit.
Consider `--models duke_of_lancaster wolf0 cat0` for cross-mesh breadth.

**Output:** `results/e12_mapf_ablation/raw/*.json`; summary prints objective vs
ordering budget and coordinated-vs-independent colliding pairs per K.
**Est. time (Duke):** orderings ≈ 3 h @5 seeds (≈1.8 h @3); collisions ≈ 3–4 h
@5 seeds (≈2 h @3). **Recommend 3 seeds on Duke here**, 5 on cheaper TOSCA if run.

---

## C. E13 — Post-smoothing collision & coverage audit  *(R4.12, R2.2)*

**Why.** Smoothing runs *after* the reservation enforces collision avoidance, so a
B-spline shortcut could move the path off reserved cells. E13 runs the real
pipeline (20-trial MAPF) and audits the **final smoothed+densified** output:
environment-collision sample count (vs inflated fine OG), minimum inter-robot
separation over the overlapping active window + #pairs breaching
`d_safe = 2·ROBOT_RADIUS = 0.70 m`, and coverage survival (closest approach of
each robot to every inspection waypoint; a waypoint is "missed" if no sample lands
within 0.5 m). Expected: 0 env collisions, min separation ≥ d_safe, 0 missed
waypoints — which is the evidence R2.2/R4.12 ask for.

**Run** (Duke, 5 seeds; add meshes as available):
```bash
until python -m experiments.e13_collision_coverage_audit --resume \
    --seeds 42 123 7 2024 314; do echo restart; sleep 3; done
python -m experiments.e13_collision_coverage_audit --plots_only
```
**Output:** `results/e13_collision_coverage_audit/raw/*.json`; summary prints
env-collisions, min separation, breach pairs, missed waypoints.
**Est. time:** ~1.8 h (Duke, 5 seeds).
**Note:** if the audit surfaces real breaches, that's a genuine finding — the fix
(re-clamping smoothed paths against `d_safe`, or re-checking the reservation after
smoothing) becomes a code task, not just a manuscript note. Check
`min_separation_m` and `d_safe_breach_pairs`.

---

## D. E14 — |C| and |P| scaling / discretization sensitivity  *(R4.21, R4.22)*

**Why.** Scaling is only shown vs K. E14 sweeps candidate count |C| ∈
{500,1000,1500,2500,4000} and surface points |P| ∈ {50k,100k,200k,400k} on Duke,
one axis at a time, reporting per-stage timings and solution quality (viewpoints,
coverage). Answers both "how does it scale" (R4.21) and "is 200k / 1500 on a
plateau, not a cliff" (R4.22).

**Run** (fast — VRP skipped by default, since |C|/|P| scaling is about
sampling/visibility/set-cover; VRP cost is viewpoint-count-driven, see E07/E08):
```bash
until python -m experiments.e14_cp_scaling --resume --dim both \
    --seeds 42 123 7 2024 314; do echo restart; sleep 3; done
python -m experiments.e14_cp_scaling --plots_only
```
Add `--with_vrp` only if a routing-time-vs-|C|/|P| curve is wanted (adds ~3 min/point).
**Output:** `results/e14_cp_scaling/raw/*.json`; summary prints vps/cov/t_vis/t_opt
(and t_vrp) per swept value.
**Est. time:** ~15–30 min without VRP; ~2.5 h with `--with_vrp`.

---

## E. Re-runs for variance & seeds  *(R4.19, R4.20, R4.14, #13)*

No code changes needed beyond the E08 logging already added — just re-run existing
experiments with more seeds and regenerate tables with variance.

1. **E05 visibility variance (Table 1, R4.20).** Cheap (visibility only).
   ```bash
   until python -m experiments.e05_visibility_comparison --resume \
       --seeds 42 123 7 2024 314 999 55 8888 1337 2025; do echo restart; sleep 3; done
   ```
   Then report mean ± std in Table 1 (replaces bare numbers). ~0.5 h.

2. **E08 gap-vs-timeout (R4.13/#13) + variance.** Re-run to ≥5 seeds; the new
   `gap` / `gap_reached` fields quantify how often the 5% gap is met vs the 120 s
   budget being exhausted (existing runs already show solve_time ≈ 121 s → it
   usually times out). ~1 h (resume-add 2 seeds) / ~2 h (fresh 5 seeds).
   ```bash
   until python -m experiments.e08_vrp_alpha_blending --resume \
       --seeds 42 123 7 2024 314; do echo restart; sleep 3; done
   ```

3. **E10 to 5 seeds — variance + gap-band sources (R4.14, R4.19).** The 3×–6× gap
   band is currently unexplained; more seeds + the per-stage / per-mesh breakdown
   already in E10 lets us attribute it (lower-bound looseness vs mesh geometry).
   ```bash
   until python -m experiments.e10_cross_model --resume \
       --seeds 42 123 7 2024 314; do echo restart; sleep 3; done
   ```
   ~1.6 h to add 2 seeds to the existing 3 (Duke 2×21 min + 9 TOSCA ×2×3 min).

4. **Other cheap experiments to 5–10 seeds** (E01, E02, E06): re-run with
   `--seeds` SEEDS_5/SEEDS_10 as time permits; all are sub-minute per row.

**Seed policy** (compromise for R4.19 given MAPF cost): **≥5 seeds everywhere**;
10 for the cheap visibility/set-cover/sampling experiments; 3–5 for MAPF-heavy
E07/E10/E12/E13. State the per-experiment seed count in the paper.

---

## F. More real underwater meshes *(R4.2 — optional, last)*

Only if time remains. Duke is the sole real target-domain mesh. Source 1–2 more
real underwater structures (e.g. another wreck / a monopile / a pier photogrammetry
mesh in GLB/OBJ/STL), add a `ModelConfig` entry (mirror `duke_of_lancaster()` with
an appropriate `target_length`), then run E10 (+ optionally E13) on them:
```bash
python -m experiments.e10_cross_model --models duke_of_lancaster <new_mesh> --resume
```
Each new Duke-scale mesh ≈ 20 min/seed. Budget accordingly; treat as stretch.

---

## G. Suggested schedule (≈9 h attended today + 2 overnights)

Total new-experiment compute at the recommended seed counts ≈ **10–12 h**, almost
all of it MAPF-bound and fully unattended via `--resume` loops.

**Today (attended ~9 h):** launch the cheap/short jobs, verify they don't crash in
the first few rows, then hand off the long pole to run overnight.
- Kick off **E14** (~20 min) and **E05 re-run** (~30 min) first — quick wins,
  confirms env is healthy.
- Launch **E11** (~2 h) — watch the first config finish, then leave it.
- Before leaving, start the **overnight block**: `E12 --mode both` (Duke, 3 seeds)
  + `E13` (Duke, 5 seeds) chained in one restart loop (~6–8 h). See below.

**Overnight #1 (tonight → tomorrow):** E12 + E13 (the MAPF-heavy long pole).
```bash
until python -m experiments.e12_mapf_ablation --resume --mode both \
      --seeds 42 123 7; do echo r; sleep 3; done && \
until python -m experiments.e13_collision_coverage_audit --resume \
      --seeds 42 123 7 2024 314; do echo r; sleep 3; done
```

**Day 2 (from ~3 pm, attended):** review overnight results with `--plots_only`;
launch **E08 re-run** (~2 h) and any **E11** remainder; start **overnight #2**.

**Overnight #2 (day 2 → day 3):** **E10 to 5 seeds** (variance + gap band, ~1.6 h
to extend, or ~4 h fresh) + optional **E07** 5-seed re-run + optional **item F**
mesh runs.
```bash
until python -m experiments.e10_cross_model --resume --seeds 42 123 7 2024 314; do
  echo r; sleep 3; done
```

After each block, run the matching `--plots_only` and copy the summary numbers /
figures into the paper.

---

## H. Checklist

- [ ] A — E11 VRP cut ablation
- [ ] B — E12 MAPF ablation (orderings + collisions)
- [ ] C — E13 collision & coverage audit
- [ ] D — E14 |C|/|P| scaling & sensitivity
- [ ] E1 — E05 variance (Table 1 → mean ± std)
- [ ] E2 — E08 gap-vs-timeout + variance
- [ ] E3 — E10 to ≥5 seeds (variance + gap-band attribution)
- [ ] E4 — other cheap experiments to ≥5 seeds
- [ ] F — (optional) additional real meshes
