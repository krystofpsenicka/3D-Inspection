# Session handoff — 2026-08-05

Read this first on resume. Work was stopped part-way through a re-run, so the
single most important thing here is **which numbers on disk are current**.

Full detail: `RESEARCH_PLAN.md` §9.12–§9.13; plain language:
`STATUS_OVERVIEW.md` §4f/§4g.

## 0. Nothing is committed

The working tree has 6 modified/new files plus ~4 MB of new results. `main` is
untouched; branch is still `research/stage-a-operator-infra`.

## 1. Direction, decided this session

Krystof reopened the publishability question. Option B (two-sensor online/MPC),
committed on 2026-07-31, was **never started and is now shelved** — starting a
fresh experiment stack from zero would have finished thinner and less verified
than what is already on disk. Instead the existing warm result is being
hardened and reframed as:

> an audit-gated differentiable **refinement layer** for multi-robot inspection
> plans — improves coverage AND makespan simultaneously, and holds a stricter
> safety constraint than the plans it refines.

The "it's just a post-processor" attack is answered by the safety column, which
turns it into a *certifying* refinement layer. That reads closer to an RA-L
contribution (rolling submission, no deadline) than an ICRA headline.

Agreed order was: (1) fix the clearance defect + re-run the frontier; (2)
measure the pipeline's own clearance; **then decide publish vs close with
verified numbers**; (3) the 50 m mesh. Steps 1–3 are done. **The publish-vs-
close decision is Krystof's and is now unblocked.**

## 2. What is trustworthy on disk

| results dir | state | trust |
|---|---|---|
| `hpro/results/frontier_clear/` | 18/18, current code | **YES — the reference** |
| `hpro/results/frontier_match/` | **1/18** — deleted, partly regenerated | **NO — re-run** |
| `hpro/results/full50/` | **empty** — deleted, not regenerated | **NO — re-run** |
| `hpro/results/clearance_audit/` | 18 files | pipeline column YES, *ours* column STALE |
| `hpro/results/frontier{,_seeds,_robots}/` | pre-2026-08-05 runs | history only |
| `outputs/full50_tc{78,90}` | 50 m baselines | YES — do NOT regenerate |

**The reference result** (`frontier_clear`, 3 seeds × 5 operating points, plus
R=3/4/5): 15/15 Pareto-dominated, **Δcov +2.45 ± 1.04 pts, Δmakespan
−5.39 ± 2.70 %**, worst clearance **0.5071 m** (constraint 0.50, audited at
640 samples/segment), worst separation **0.83 m** (constraint 0.80).

## 3. Resume steps (~35 min, unattended)

```bash
ipy=/home/troja-lab-02/miniconda3/envs/inspection/bin/python

# a) matched-margin control, 18 runs (~25 min)
./hpro/run_frontier.sh hpro/results/frontier_match --clearance_margin 0.05
$ipy hpro/frontier_summary.py --root hpro/results/frontier_match

# b) 50 m refiner runs, baselines already on disk (~3 min)
for t in tc78 tc90; do mkdir -p hpro/results/full50/$t
  $ipy hpro/joint_pilot.py --pipeline_dir outputs/full50_$t --modes warm \
       --out hpro/results/full50/$t --no_show; done

# c) refresh the clearance audit against the CURRENT chains (~8 min)
for d in outputs/pilot_*; do n=$(basename $d); grp=frontier
  case $n in pilot_s*) grp=frontier_seeds;; pilot_r*) grp=frontier_robots;; esac
  $ipy hpro/audit_clearance.py $d \
     --chains hpro/results/frontier_clear/$grp/$n/chains.npz \
     --json_out hpro/results/clearance_audit/$n.json; done
```

Then update the marked-`PRE-FIX` tables in RESEARCH_PLAN §9.12.3 and §9.12.5
and the matching ones in STATUS_OVERVIEW §4f/§4g.

## 4. What was fixed this session (three defects, one theme)

All three are the same failure: **the constraint was fine, the test was not.**

1. **Clearance sampled by count, not spacing** (§9.12). 10 samples/segment on
   segments up to 7.8 m = a check every 0.78 m; a straight line grazing a hull
   dips ~5 cm between checks. 9 of 18 published runs sat inside a limit the
   solver believed it was meeting (true worst 0.4725 m vs 0.50). Fix:
   `_col_per_seg` bounds sample *spacing* (`--col_spacing`, 5 cm), applied to
   every solver-side clearance test. Audit raised 40 → 640 samples/segment;
   results now record the whole ladder and dump `chains.npz`.
   `--col_spacing 0` reproduces the pre-fix solver bit-for-bit.
2. **Vias inserted without checking they were feasible** (§9.12.3). The seed is
   pushed out along the ESDF gradient, but inside a thin structure the field is
   flat (every occupied voxel reads the same −1 voxel), so the push does
   nothing — and the via was inserted anyway, *adding* a waypoint inside the
   hull, then the loop re-found the same segment until the budget was gone.
   Fix: verify before inserting, deterministic spherical probe as fallback,
   abandon unrescuable seeds (keyed by world position, since insertion
   renumbers segments).
3. **"Feasible" meant "within a millimetre"** (§9.12.4). The acceptance test
   admits violation ≤ `tol` = 1e-3, so a plan riding the constraint settles
   1 mm inside it. Invisible at pilot scale; on the 50 m mesh with correct
   sampling it showed as 0.4992 m against 0.50. Fix: the solver defends
   `col_margin = margin + tol` everywhere, so an accepted move is genuinely
   feasible. Also `col_samples_max` 400 → 2000, and it now **warns when the cap
   binds**, since a silently weakened spacing guarantee is defect 1 again.

## 5. Two measurements worth keeping regardless

**The ESDF is biased, opposite to the obvious guess** (§9.12.1). Asked what it
reads at 200 000 points on the true mesh surface — where a perfect field reads
0 — it answers **−0.108 m** (median −0.121), because the distance transform
measures voxel centre to centre, so the voxelized hull bulges ~1 voxel past the
real one. The field **under**-reports clearance. Consequences:

* **The pipeline's paths are NOT infeasible** and must not be described that
  way. Its worst executed trajectory of 18 carries ~0.44 m of *true* clearance,
  above the 0.35 m robot radius. Its rule is 0.40 m of grid inflation
  (`INFLATION_VOXELS = int(0.35/0.10)+1`), looser than our 0.50 m by design.
  An earlier draft of §9.12.1 implied a defect in their planner; that was
  wrong and is corrected.
* The bias is a constant on both arms, so it cancels in comparisons and matters
  only for absolute feasibility verdicts.

**A hard ceiling in the baseline** (§9.12.5). At 50 m the 0.95 operating point
cannot be generated at all: `run_full_pipeline.py` segfaults (exit 139) inside
cuOpt's VRP, twice out of two, at 213 viewpoints; 0.93 also crashes at 155;
126 succeeds. Threshold is between **126 and 155 viewpoints**. Cuts both ways —
a real scaling limit of the combinatorial pipeline, and a bound on us too,
since the refiner needs its output as a warm start. It is the strongest
argument yet for the standalone direction that §9.8 could not deliver.

## 6. Open, in priority order

1. Finish the three re-runs in §3.
2. **Decide publish vs close.** Everything the decision needs is measured.
3. Continuation ablation across the frontier (§5d: −2.7 % makespan at one
   operating point; ~15 configs × ~20 s).
4. SCP / coordinated wave moves — deliberately out of scope; the measured
   front-advance limitation makes it a strong future-work paragraph.
5. MPC/receding-horizon framing, then writing.
