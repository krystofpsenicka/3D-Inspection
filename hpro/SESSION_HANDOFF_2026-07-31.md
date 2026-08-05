# Session handoff — 2026-07-31

Read this first on resume. It captures the strategic decision made this session
and the exact technical next step, so nothing is lost across the reboot.

> **RESOLVED 2026-07-31 (later session).** GPU blocker gone (reboot done,
> driver 580.173.02 both sides). Tasks 1–3 below are all executed; the
> **Option A go/no-go FAILED on 3/3 seeds → switch to Option B**. See §5 for
> results, which supersede the expectations in §2 and §3. §0–§4 are kept as
> written for the record.

## 0. IMMEDIATE BLOCKER (why we stopped)

The GPU is unusable: **NVIDIA driver/library version mismatch**.
- Running kernel module: `580.159.03` (see `/proc/driver/nvidia/version`)
- Userspace libs (apt-updated): `580.173.02` (`libnvidia-compute-580`, `nvidia-dkms-580`)
- Symptom: every CUDA call dies with
  `cudaErrorCompatNotSupportedOnDevice: forward compatibility was attempted on
  non supported HW` / `Driver/library version mismatch`. `nvidia-smi` fails with
  `Failed to initialize NVML: Driver/library version mismatch`.
- Cause: driver upgraded via apt in the last week; new kernel module never loaded.
- **Fix: reboot** (cleanest). After reboot `nvidia-smi` should work and the loaded
  module will match 580.173.02.
- All `joint_pilot.py` work is GPU-only (cupy + the OptiX ray-caster
  `RaycastingVisibilityQueryCuda`), so nothing runs until this is fixed.

**Verify recovery after reboot:** `nvidia-smi` prints the GPU table cleanly.

## 1. Strategic decision this session (supersedes RESEARCH_PLAN §9.7 "Next")

**Context:** Krystof has ~20 h of *technical* work left (writing not counted) and
wants the work to make the project **publishable at a real venue (ICRA/IROS)**.

**Krystof's critique of the current headline result (correct, accepted):** the
warm-started joint refiner (§9.7, Pareto-dominates the pipeline frontier 15/15)
is NOT a strong enough contribution on its own, because:
- it does not work standalone (cold start fails, §9.8: 0.759 vs pipeline 0.951);
- it consumes the pipeline's output, so the "5–18 s vs 174 s" wall-clock
  comparison is meaningless — the only honest claim is "given a plan, it improves
  it by ~2.4 cov pts / ~5.5% makespan in a few seconds." Marginal, dependent.

At the ICRA/IROS bar, only two outcomes clear it:
- **Option A (chosen, primary): make it a real standalone planner** — fix cold
  start so joint optimization from scratch matches the decomposition. Then the
  claim is "the staged set-cover+VRP decomposition is unnecessary." Best paper.
- **Option B (fallback): the two-sensor online / MPC story** — long-range sonar
  reveals *geometry* (demand); short-range camera coverage is the *inspection*
  deliverable. This resolves Krystof's "if you sensed it you inspected it"
  objection to the old online work (§3.6/§3.8 used ONE sensor that conflated
  discovery+inspection). Here "faster than the decomposition" is *legitimately*
  true (world/map grows → decomposition must re-solve; gradient method warm-steps).
  Lower risk, reuses `trajectory.py` receding-horizon machinery. No Isaac Sim
  needed — both sensors can be ray-cast sims (sonar = long-range geometry reveal,
  camera = existing short-range frustum+visibility coverage).

**Plan:** attack A first with a hard checkpoint (~6–8 h). Go/no-go: if cold start
moves from 0.759 to within ~2–3 pts of 0.951 at comparable makespan/clearance,
commit the rest to the standalone-planner paper. If it stalls below ~0.88, switch
to B.

## 2. Why cold start fails (measured, §9.8) and the planned fix

Diagnosis: the expand-phase capture reward is
`joint_pilot.py:1027`:
```python
gm = torch.sigmoid(args.eb_gain_sharp * (Mm[:, pool_act] - cap_req))  # eb_gain_sharp=12
gain = gm[view_mov].sum()
```
`Mm` = metric visibility margin (min over occlusion/near/far/frustum walls, in
metres; very negative for anything metres away or outside the frustum). At
sharpness 12 the reward and its gradient are ~0 unless a pool point is already
nearly captured → **no gradient toward distant unseen surface.** Local moves
therefore can't explore; the discrete `_reseed_pass` (k-means uncovered pts →
teleport poses onto them) is what lifted 0.595→0.759, then it plateaus (leftover
= scattered speckle below `reseed_min_cluster=40`, and pockets whose normal-offset
viewpoint is inside the hull — the latter the pipeline also can't cover; its
ceiling is 0.986, operating point 0.951).

**Planned attack (Option A), in `optimize_elastic` / `sweep_step`:**
1. **Continuation / graduated non-convexity:** anneal `eb_gain_sharp` low→high
   across expand sweeps (e.g. ~0.5–1.0 → 12). Low sharpness = wide basins = a
   real gradient toward marginally-outside points. (Won't reach points far
   outside the frustum — that's what #2 is for.)
2. **Long-range demand potential (the real lever):** add a position-space
   attraction term pulling each view pose toward the centroid of the nearest
   uncovered-point cluster (weighted by remaining demand), so a pose with no
   local prey still gets a nonzero, distance-based gradient to drift toward a
   region that has prey. This is the exploration signal the flat surrogate lacks.
   (This is the principled version of Krystof's "annealed exploration" idea —
   applied to objective shape + an explicit potential, NOT RL noise, which would
   just random-walk in a flat landscape.)
3. **Smarter global reseed:** lower `reseed_min_cluster` to mop up speckle; keep
   the net-delta + cheapest-slot rules already in `_reseed_pass`.

Realistic target: 0.759 → ~0.95 (NOT 1.0 — pocket points are infeasible for
everyone). A partial fix (e.g. 0.88) does NOT clear ICRA/IROS → fall back to B.

## 3. Exact resume steps

1. Reboot done, `nvidia-smi` OK.
2. Reproduce baseline (confirms env): 
   `ipy_inspection=/home/troja-lab-02/miniconda3/envs/inspection/bin/python`
   `$ipy_inspection hpro/joint_pilot.py --pipeline_dir outputs/pilot_baseline --modes cold --out hpro/results/cold_repro`
   Expect ~0.759 cold, ~163 s. (Prior run at 800 sweeps also gave 0.7594 —
   `hpro/results/cold_arm/joint_pilot.json`.)
3. Implement #1–#3 above in `hpro/joint_pilot.py`. New CLI flags for anneal
   schedule + demand-potential weight so it's ablatable.
4. Re-measure cold; apply the §9.3 / task-3 go/no-go.
5. Keep the project rules: score by the pipeline's ray-caster (never our
   surrogate), independent constraint audit at ≥8× resolution (see
   [[constraint-audit-rule]]), ≥5 seeds before any claim, honest negatives kept.

## 4. Repo state
- Branch `research/stage-a-operator-infra`, nothing merged, `main` untouched.
- `outputs/` is untracked (~1 MB/run binaries); `outputs/pilot_baseline` (the 0.95
  operating point) + all seed/robot baselines are present on disk. Regenerate with
  the `run_full_pipeline.py` invocation in RESEARCH_PLAN §9.7 if ever lost.
- Key files: `hpro/joint_pilot.py` (2001 lines — solver, `optimize_elastic`,
  `cold_start`, `_reseed_pass`, `evaluate`), `RESEARCH_PLAN.md` §9 (the pivot),
  `STATUS_OVERVIEW.md` §4d/§4e.
- Tasks open (harness task list): #1 reproduce cold baseline (in_progress),
  #2 implement continuation+demand potential, #3 go/no-go checkpoint.

## 5. RESULTS (executed 2026-07-31, after the reboot)

### 5a. The 0.759 baseline did not reproduce — two defects, since fixed

Cold start at HEAD gave **0.6006 in 9.3 s**, not 0.759/163 s. Neither arg drift
(`cold_expand_sweeps` 800→200, `sep_samples` 48→600) explained it. Root cause was
two defects in `optimize_elastic`, both exposed by commit `049230a` (which
post-dates the `cold_arm` run by 30 min):

1. **The separation gate reverted the WHOLE half-sweep** on any regression.
   Cold-start chains interleave, so one conflicting mover cancelled every
   capture its co-movers had earned (measured: violation 0.0000 → 0.4521 m,
   10–34 accepted moves dropped per half-sweep).
2. **`a0+a1==0` broke out of the expand loop** before `capt_stall` could reach
   `reseed_after`, so `_reseed_pass` — the only long-range mechanism — *never
   fired at all* in a cold run.

Fixed: per-mover separation reversion (same accept/reject pattern as the
coverage audit gate, least-valuable offender first), and a zero-move sweep now
hands its turn to the reseed instead of ending the phase. Cold start → **0.7361**.

Note the old 0.759 was recorded at `min_clearance` **−0.55 m** — clearance
*infeasible*. 0.7361 sits at 0.44 m against a 0.50 m constraint, so it is still
0.06 m short; the solver checks 10 samples/segment, the audit checks 40.

### 5b. Continuation helps, the demand potential does not

Single seed (`pilot_baseline`), 800 expand sweeps, pipeline = 0.9507 / 113.0 m:

| run | coverage | makespan | min-clear | wall |
|---|---|---|---|---|
| neither (control) | 0.7361 | 110.8 m | 0.44 m | 135 s |
| **continuation only** | **0.7816** | **102.9 m** | 0.44 m | 314 s |
| demand potential only | 0.7001 | 111.1 m | 0.51 m | 407 s |
| both | 0.7469 | 101.1 m | 0.50 m | 279 s |

The demand potential is a **recorded negative**: it is live (5–9 eligible movers
and 1–4 accepted drifts per half-sweep) and still loses coverage. Two eligibility
mistakes were found and fixed before that conclusion was trustworthy:

* requiring a pose to hold *nothing* unique left **1 of 65** movers eligible —
  the same trap `_reseed_pass` documents at its `net` rule;
* "starved" via `old_cap == 0` measures the **surrogate's false positives**:
  `old_cap` is a surrogate claim while `pool` is ray-cast-defined, so a nonzero
  `old_cap` means the surrogate sees what the ray-caster does not.

All new flags default OFF: `--eb_gain_sharp_lo`, `--eb_anneal_sweeps`,
`--eb_demand_w`, `--eb_demand_clusters/_min/_every/_max_uniq`.

### 5c. Go/no-go: Option A FAILS, 3/3 seeds → Option B

Best cold config (continuation on) vs each pipeline seed:

| pipeline seed | pipeline | cold | gap |
|---|---|---|---|
| `pilot_baseline` | 0.9507 | 0.7816 | −16.9 pts |
| `pilot_s1_tc90` | 0.9028 | 0.7749 | −12.8 pts |
| `pilot_s7_tc90` | 0.9038 | 0.7715 | −13.2 pts |

The bar was "within ~2–3 pts of the pipeline"; the fallback line was 0.88. Cold
lands at 0.77–0.78 — not close, and consistent across seeds.

**This is not a compute artifact.** The best arm converged at expand sweep **95
of 800**, with 4367 points still uncovered *and ~10 m of makespan budget
unspent* (102.9 of 113.0). It stalls; it does not saturate. Cold start's poses
are misplaced, and local sliding plus an occasional discrete reseed cannot
re-derive what set cover computes directly. Making the reseed aggressive enough
to close 13–17 points would amount to replacing set cover with another discrete
heuristic — which is not the "the decomposition is unnecessary" claim.

→ **Proceed with Option B** (two-sensor online / MPC, §1).

### 5d. The gate fixes are a NO-OP for the warm arm — §9.7 stands as published

Re-measured warm on `pilot_baseline` (pipeline 0.9507 / 113.0 m):

| warm arm | coverage | makespan | total | min-clear |
|---|---|---|---|---|
| `frontier/pilot_baseline` (pre-fix, 24 Jul 15:51) | 0.960800 | 106.4011 m | 201.2977 m | 0.4786 |
| + gate fixes (today) | 0.960800 | 106.4011 m | 201.2977 m | 0.4786 |
| + fixes + continuation | 0.961350 | **103.5220 m** | 197.5148 m | 0.4590 |

Bit-identical: **the §5a fixes change the cold arm only, and the stored §9.7
frontier does NOT need re-running.** That is the expected result in hindsight —
warm chains start from the pipeline's own well-separated poses, so the
half-sweep separation revert almost never triggers, and the expand-loop break
only matters when the reseed is in play, which is the cold path.

> **Trap for the next session:** `hpro/results/joint_pilot_eb/` (24 Jul 13:46,
> 0.9513 / 109.6 m) is a STALE run predating `eb_trust`, `eb_repair` and the
> whole `reseed_*` family — its `args` have them as `null`. It is *not* the
> §9.7 reference and comparing against it manufactures a ~1 pt / ~5 % gain that
> does not exist. The reference is `hpro/results/frontier*/`.

Continuation is a genuine but modest lever on warm: coverage flat (+0.0006),
makespan **106.4 → 103.5 m (−2.7 %)**, at one operating point on one seed. If
the refiner stays in the paper it is worth a frontier-wide ablation (~15 configs
× ~20 s), but it does not change the story.

### 5e. Open items

> **RESOLVED 2026-08-05 — both clearance items. See RESEARCH_PLAN §9.12 and
> STATUS_OVERVIEW §4f.** The cause was resolution, not a bad trade: the solver
> sampled segments at a fixed 10 points (up to 0.78 m apart on a 7.8 m
> segment) and stopped where its own samples read clear. Sampling is now
> spacing-bounded (`--col_spacing`, 5 cm), the audit runs at 640 samples/
> segment, and all 18 frontier runs were re-solved: worst clearance 0.4427 →
> **0.5200 m**, with the win unchanged (15/15, +2.42 ± 1.05 pts,
> −5.39 ± 2.75 %). The pipeline's own clearance is measured too — 0.33–1.20 m,
> under a looser 0.40 m grid-inflation rule of its own.

* **Structure clearance is below its constraint on the WARM arm too, and that
  is the arm the paper depends on.** Margin is 0.50 m
  (`robot_radius` 0.35 + `clearance_margin` 0.15); the 40-samples/segment audit
  reports 9 of the 18 stored frontier runs under it — 0.442, 0.468, 0.478,
  0.479, 0.481, 0.490, 0.494, 0.495, 0.497 m. The solver checks
  `col_samples=10`, so it under-resolves exactly what the audit catches. This
  is the [[constraint-audit-rule]] failure mode again and a reviewer will find
  it. Fix by raising the solver's collision resolution (or tightening
  `eb_slack`) and re-running the frontier.
* **The pipeline's own clearance is never measured** — every `pipeline` row
  reports `min_clearance = NaN`, so "we satisfy clearance better than the
  baseline" is currently unsupported in either direction. Note `warm-init`
  (0.31 m) is our straight-chain reconstruction, NOT the pipeline's ST-A* path,
  so it cannot stand in for it.
* The demand-potential code is retained, off, as a documented negative.
