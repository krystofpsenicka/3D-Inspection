# Where the project stands — plain-language overview (2026-07-17)

> **⚠ 2026-07-23 evening: direction pivot — read §4d below and RESEARCH_PLAN.md §9.**
> The paper is now *joint differentiable multi-robot trajectory optimization as
> optimal control*, measured against (and optionally warm-started by) the thesis
> pipeline. The online belief-update experiments described below (§4.3–§4c) are
> parked, not retracted. §4d also records what the first pilot built and the one
> open problem it left.
>
> **⚠ 2026-07-24: that open problem is solved and the headroom question is
> answered — read §4e below and RESEARCH_PLAN.md §9.7.** The joint refiner now
> Pareto-dominates the pipeline's own coverage-vs-makespan frontier at every
> measured operating point.

*Written for reading alongside the work. The formal evidence document is
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md); this file explains the same state in prose,
spelling out what the methods actually are. Nothing here is new — every number is
sourced from a section of the research plan.*

---

## 1 · What we are trying to do, in one paragraph

Inspection planning asks: given a structure (here: a shipwreck), find a short camera
trajectory that *sees all of its surface*. The classical way — and the way your thesis
pipeline does it — is **sample-and-select**: generate a few hundred candidate camera
poses, compute exactly which surface points each one sees (by casting rays against the
mesh), pick a small subset that together covers everything (set cover), then connect the
chosen poses with a short route (a travelling-salesman-style step). This works, but it is
rigid: the candidates are fixed, the whole thing must be re-solved from scratch whenever
anything changes, and it cannot "slide" a camera pose a little to the left to see around
a corner. Our idea is to make visibility itself **differentiable** — a smooth function
where you can ask "if I nudge the camera, does coverage go up?" — and then optimize
camera *trajectories* directly by gradient descent, the same way one trains a neural
network. The pay-off is not offline planning (a good discrete solver is hard to beat when
the world is frozen); it is **planning that keeps re-optimizing while the world changes
under it** — which is what today's headline experiment finally demonstrated.

## 2 · The building blocks, spelled out

**HPRO — "Hidden Point Removal Operator", Katz & Tal, 2025.** An analytic (formula-based,
no learning) trick that estimates which points of a point cloud are visible from a given
viewpoint *without any mesh or ray-casting*. It applies a spherical inversion that turns
"visible from the viewpoint" into "lies near a convex hull", and softens that test so it
has gradients. Its weakness, which we measured extensively: its scores are **miscalibrated
in exactly the way a gradient optimizer loves to exploit** — moving the camera closer
inflates the score without seeing more. An optimizer using HPRO happily reports coverage
≈ 1.0 while true coverage *falls*. We call this **surrogate exploitation** and it is one
of the paper's contributions (diagnosing it, showing it is a property of the visibility
model rather than the planner).

**NVPS — "Neural Visibility of Point Sets", SIGGRAPH Asia 2025.** A small neural network
(an octree-based U-Net) pretrained on synthetic shapes that answers the same question —
"is this point visible from this direction?" — learned rather than derived. We plug in the
authors' pretrained checkpoint, no retraining. Two properties matter: it is much better
calibrated on complex concave geometry (the wreck), and it is structurally *incapable* of
the "come closer to inflate the score" cheat, because its prediction depends only on the
viewing **direction**, not the distance. That same property is also its blind spot (it
cannot distinguish being inside vs outside a cavity along the same ray). HPRO and NVPS
fail in complementary ways — HPRO is better on thin structures, NVPS on concave ones —
which motivates having both behind one interface.

**The visibility layer (our Stage A work).** We built a common interface
(`backbones.py`, `visibility_layer.py`) where either occlusion model (or an ensemble)
is combined with a real camera model: field of view, near/far range, and a smooth
"frustum gate" so the whole thing stays differentiable. Along the way we found and fixed
two real bugs in the HPRO reference implementation and validated everything against
brute-force ray-casting. Stage A is finished and defensible.

**The receding-horizon planner (Stage B, `trajectory.py`).** Instead of choosing all
camera poses at once, the robot plans a short **horizon** of ~6 future poses by gradient
descent (maximize soft coverage, minimize path length, stay in a safe standoff band),
executes only the *first* pose, then observes what it actually saw — a hard ray-cast
"audit" that stands in for the real sensor. Seen points stop attracting the trajectory
(their "demand" smoothly drops to zero), the horizon slides forward, and — crucially —
the next optimization **warm-starts** from the previous solution, so a handful of
gradient steps per cycle suffice. Warm-starting is the property the whole paper leans on:
a discrete pipeline must re-solve from scratch when anything changes; a gradient
optimizer just keeps stepping.

## 3 · What happened before today (the honest history)

- **The original headline died in review of the literature.** We had planned to claim
  "first gradient-based joint optimization of camera poses on point clouds". A 2024
  paper (NeOF, in our own target venue RA-L) already does that for *static* camera
  placement. The claim was repositioned: **trajectories and online replanning** are the
  open ground; static placement is now a supporting comparison, not the novelty.
- **The planner used to stall.** Once it had seen everything nearby, the coverage
  gradient vanished and the robot crawled at ~5 cm per cycle — the last 15 of 40 cycles
  bought nothing. Offline, the discrete baseline beat it.
- **A result had to be retracted.** A promising number (0.933 coverage) turned out to be
  a 1-in-8 lottery draw: the same configuration and seed gave 0.563 in seven processes
  and 0.933 in one. Tiny GPU floating-point differences get amplified by the closed
  planning loop into two completely different outcomes. Since then the iron rule is:
  **no Stage B number from fewer than ~5 runs, always mean ± std.**

## 4 · What was done today (2026-07-17)

### 4.1 Environment: everything is green now

The one missing piece of the `inspection` environment — the GPU ray-casting library
`triro`, which needs NVIDIA's OptiX SDK — is installed and working. The old failure was
an ABI mismatch (OptiX 9.1 is too new for the driver); rebuilding against OptiX 8.0.0
fixed it. **All 126 tests of the full pipeline now pass.** The exact recipe (including a
pip caching trap) is in RESEARCH_PLAN.md §1.

### 4.2 The stall is fixed — a "two-timescale" planner (§3.5)

The fix has a discrete, slow-thinking half and a continuous, fast-thinking half:

- **Slow timescale (new, `GlobalGuide`):** every cycle, cluster the not-yet-seen surface
  points into a handful of regions (k-means), order those regions into a sensible tour
  from the robot's position (nearest-neighbour + 2-opt — deliberately the *same* routing
  logic the classical baseline uses), and commit to the first region as the current
  **target**. The gradient optimizer's "go somewhere useful next" term is pointed at
  that region only.
- **Fast timescale (as before):** the local horizon keeps being refined by gradient
  descent and harvests anything it happens to see on the way.

Two non-obvious lessons made it actually work:

1. **A stall detector must know the difference between "travelling" and "stuck".**
   Judging progress only by "did we see new points this cycle" fires constantly during
   transit legs. Progress is now *either* seeing new points *or* getting closer to the
   target; a target that yields neither for a few cycles is shelved for a while (some
   pockets genuinely cannot be seen from the allowed distance band — without shelving
   they are a tar pit).
2. **When the target changes, the old plan is the trap.** Warm-starting is right while
   pursuing one target, but across a target switch the inherited horizon is precisely
   the local minimum we're escaping — 30 gradient steps cannot unfold it against the
   path-length penalty. On a switch, the *plan* (never the robot) is re-seeded as a
   straight line of poses toward the new target, which the next optimization bends
   around obstacles.

Knob values were then **re-derived over 5 seeds** (never from a single promising run —
that is exactly how the retracted number happened). Results, wreck, tight camera, n=5:

| | coverage | path |
|---|---|---|
| planner without the guide | 0.768 ± 0.144 | 9.5 m |
| **planner with the guide (tuned)** | **0.942 ± 0.014** | 13.0 m |
| oracle greedy baseline, interpolated at 13.0 m | ≈ 0.929 | — |

So offline, against a baseline that *cheats* (it selects using exact ray-cast visibility
we only estimate), the planner is now at statistical **parity, slightly above**. Bonus:
the fixed planner turned out to be **bit-for-bit reproducible across 8 separate
processes** at both configurations we probed — the old bimodal lottery disappeared,
most likely because the plan re-seeding discards the accumulated floating-point drift.

### 4.3 The go/no-go experiment — cleared (§3.6)

The plan's own bar said: if warm-started replanning cannot beat a discrete re-solve on
wall-clock while matching its coverage *in a changing world*, there is no paper. That
experiment (`eval_outdated.py`) now exists and passes:

**Setup.** The robot is given an **outdated survey mesh**, but flies over a reality where
one region has caved in. A simulated depth sensor tells each arm, per executed pose,
what is really there: points confirmed seen, believed points *refuted* (the sensor looks
straight through where the old surface used to be), and newly discovered surface near
what was seen. All three arms get exactly the same sensor, pose budget and stop rule:

- **open-loop** — the classical pipeline planned once on the outdated mesh, flown blind;
- **adaptive** — the classical pipeline, re-solved every time reality contradicts the
  plan; each re-solve must **re-ray-cast** its candidate poses (that is its honest cost —
  and we even grant it the true mesh for this, which flatters it);
- **ours** — the receding-horizon planner, whose demand map just shifts smoothly as
  refutations and discoveries come in, and whose warm start survives every change.

**Result, 5 seeds:**

| arm | true-surface coverage | path | total planning time | reaction time per world change |
|---|---|---|---|---|
| open-loop | 0.957 ± 0.005 | 12.2 m | 3.3 s | — |
| adaptive (re-solve) | 0.964 ± 0.012 | 13.1 m | 28.9 s | **3.2 s** |
| **ours (warm-started)** | 0.962 ± 0.006 | **7.2 m** | **5.2 s** | **0.20 s** |

Same coverage, **half the flight path, 16× faster reaction to a changed world, ~6× less
total planning compute**. In the coverage-vs-metres figure, all five of our runs sit
above both baselines at every path length. This was the single experiment the entire
repositioned paper hinged on, and it works.

*Stated limitation:* the "blind plan misses the damage" half of the story is weak in the
current configuration — the dent is shallow enough that the blind plan still sees most of
it by accident (0.913 vs our 0.949 in the changed region). The latency/path numbers don't
depend on that; a deeper or topological change (a hole into the interior) would sharpen
the demo.

## 4b · Update (written while you read): the NeOF head-to-head is done, and we win

This happened after the overview above was first written. NeOF is the 2024 prior-art
paper whose existence forced the repositioning; a reviewer's first question will be "how
is this not NeOF with a different visibility model?" We ran their released code on our
wreck benchmark (their full method at their published defaults, camera adapted to our
tight optics, everyone scored by the same exact ray-cast oracle):

| method (static placement) | 10 cameras | 20 cameras | compute |
|---|---|---|---|
| NeOF (their released hybrid) | 0.623 ± 0.085 | 0.809 ± 0.027 | 41–64 s |
| classical greedy (cheating oracle) | 0.886 ± 0.019 | 0.966 ± 0.006 | ~1 s |
| **greedy + our gradient refinement** | **0.938 ± 0.006** | **0.972 ± 0.002** | ~2 s |

Giving NeOF a 4.6× denser working set does not rescue it (0.779, at 165 s). The
mechanistic reason is satisfying: NeOF's own training labels come from *classical
hidden-point removal* — a proxy for visibility that is badly wrong on a concave scanned
wreck — so its neural field faithfully learns wrong answers. That is precisely the
"surrogate exploitation" phenomenon our paper diagnoses, showing up in the published
state of the art. We also measured the cost structure: NeOF must re-fit its neural field
every epoch, and that refit grows **superlinearly** with scene resolution (1.7 s → 191 s
per epoch as the working set grows 18×), while our fitting-free pipeline stays flat
(~30 ms to prepare a cloud, ~5–15 ms per optimization step). Fairness caveats are
recorded in RESEARCH_PLAN.md §3.7 (their demos target wide-lens tabletop scenes; this
regime is ours; their released code needed two crash fixes). With this, all three
planned defenses against the "it's just NeOF" attack are measurements, not arguments.

## 4c · Update 2026-07-23: two more items closed

**No-prior exploration works (plan §3.8).** The robot starts with *nothing* — no mesh,
no prior cloud — and builds its belief purely from the depth sensor (first observation
seeds it; the discovery halo grows it). This is the setting where a point-cloud
visibility model is the *only possible* evaluator: there is no mesh to ray-cast, so the
"why not just optimize true ray-cast coverage" objection structurally disappears.
Results (wreck, tight camera, 5 seeds):

| method | coverage | path |
|---|---|---|
| frontier NBV (cheap heuristic) | 0.630 ± 0.139 | 5.5 m |
| oracle NBV (peeks at the hidden mesh) | 0.954 ± 0.004 | 17.9 m |
| **ours (receding horizon on the sensor cloud)** | 0.894 ± 0.091 | **6.2 m** |

At any matched path length ours dominates both. The honest tail: one seed in five
stalls at 0.715 — and the belief-growth pictures show why: it had *discovered the
entire surface* but the leftover uncovered points are scattered speckle that yields a
trickle per view, and the global guide keeps shelving those targets (the same trickle
phenomenon §3.5 tuned away offline). It is a cleanup problem, not an exploration
problem — the natural fix is the measurement-quality demand weighting that is already
next on the operator-layer list. Also measured: the planned "HPRO early, NVPS later"
backbone switch changes nothing (0.896 vs 0.894) because discovery crosses the switch
threshold within two poses — one less mechanism to defend in the paper.

**The "make the open-loop arm visibly fail" idea is dead on this mesh, and that is an
answer (plan §3.6 addendum).** The wreck is a thin shell ~0.3 thick; any collapse
deeper than that punches *through* the hull and sticks out the far side as a fin —
which quietly breaks the sensor model's assumption that refuted ghost points float in
empty space. The summary tables looked fine; only rendering the mutation showed the
fin. At the deepest *clean* dent (0.25), the blind open-loop plan still covers 0.905
of the changed region by accident. Silver lining: across all four dent configs the
headline result is stable — matched coverage at roughly half the path and ~16× faster
reaction — so the online claim is robust to how badly outdated the survey is.

## 4d · The pivot, in plain language (2026-07-23 evening) — READ THIS FIRST

After reviewing everything above, Krystof re-set the direction. The reasoning: the
online experiments (§4.3, §4c) rest on a synthetic belief-sensor model (the discovery
halo, point-level refutation) that he doesn't find convincing as a mission story — and
the original goal was always to make the *thesis pipeline itself* better with
differentiable methods. So the paper is now:

> **Multi-robot inspection planning as one optimal control problem.** Robot
> trajectories under a simple velocity-bounded motion model, optimized jointly by
> gradients through a differentiable visibility model, with collision and inter-robot
> separation as constraints *inside the objective* (no post-hoc repair — explicitly
> rejected). The thesis pipeline (sample → set cover → VRP → ST-A*) is the
> combinatorial state of the art it must beat on its own metrics — coverage, makespan,
> planning time — and also serves as an optional warm start. The same problem solved
> receding-horizon is the online/anytime mode, which is where the §4.2 machinery gets
> reused, framed as MPC (an interest of Krystof's — Zeilinger-style learning-based
> MPC supplies the vocabulary: terminal costs, real-time iterations, safety layers).

The killer argument for this framing: the pipeline's staged decomposition is exactly
what differentiability removes — set cover can't slide a viewpoint to save a detour,
VRP can't trade coverage against makespan. Everything already measured stays relevant
(NeOF loss + scaling table = why fitted fields can't enter at this scale; surrogate
exploitation = the analysis; guide/warm-start = the MPC mode's internals).

**What the same-evening pilot established (plan §9.5, all single-seed diagnostics):**

- The whole stack works end-to-end against the real pipeline: baseline at pilot scale
  (25 m wreck, 20k points, 2 AUVs → 49 viewpoints, 95.1 % coverage, 113 m makespan),
  conventions verified exactly, differentiable signed-distance collision field built
  from the pipeline's own occupancy grid.
- **A negative finding that matters for the paper:** the pretrained NVPS network is
  *uninformative* at real inspection scale (F1 ≈ 0.2 — it cannot tell visible from
  occluded when the camera is 3 m from a 25 m structure; its training saw objects
  from 1–4 object-radii away). The unit-sphere successes above do not transfer.
- **A new third backbone that does work there:** a splatted spherical z-buffer —
  geometric, training-free, scale-free, O(N) — validated at F1 0.76 against the
  pipeline's ray-caster, with its own "come closer to cheat" exploit found and
  structurally eliminated (points occlude an angular footprint that grows on
  approach, like real surface).
- **The open problem is the inner solver.** Five optimizer formulations were tried
  and each failure is understood and written down (plan §9.5 has the table — the
  short version: Adam erases force hierarchies; sigmoid-based constraints lose their
  restoring gradient once violated; per-pose anchors deadlock; per-point max-margin
  chatters). Shortening a 49-pose two-robot tour around a concave hull needs
  *coordinated* moves that naive simultaneous gradient descent cannot find. The
  designed fix (next session, plan §9.6): elastic-band-style coordinate descent over
  poses, or TrajOpt-style sequential convexification — the constraints are already
  in metric form, which is what both need.

**The headroom question — can joint optimization actually beat the decomposition — is
therefore still open**, and it is the go/no-go for this whole direction.

## 4e · Update 2026-07-24: the inner solver works, and the headroom is real

The §4d open problem — an optimizer that can actually move a feasible
multi-robot tour — was solved by the design queued last session:
**elastic-band coordinate descent**. Poses are improved in alternating
odd/even half-sweeps, so each moving pose's neighbours stand still: the
gradient tug-of-war that sank all five §4d formulations simply cannot occur.
Each move then has to pass a per-pose *acceptance test* before it is kept.

Three further mechanisms made it honest, each forced by a measured failure:

- **Via waypoints.** Route-only points (not viewpoints — the analogue of the
  pipeline's ST-A* detours) bend chain segments around the hull. With them,
  every result is collision-clear to 0.50 m — stricter than the pipeline's
  own executed trajectory manages (0.415 m) on the same leg.
- **The audit gate.** The surrogate's job is gradients; the pipeline's own
  ray-caster gets the last word: every accepted move is re-audited and the
  least valuable reverted until true coverage stays above the baseline
  floor. This became necessary when surrogate-only invariants leaked real
  coverage (0.9507 → 0.9483) — the z-buffer *claims* 831 of the 985
  uncovered points are already visible, so they are invisible to both its
  gradients and its constraints. It costs 30–310 audited poses per run; the
  pipeline's own planning ray-casts 600.
- **Small steps, verified trades.** A trust region keeps each move a small
  lean (big dives abandon old coverage and get reverted wholesale), and a
  move may only be kept if its *net* ray-cast effect — new points captured
  minus sole-covered points dropped — is non-negative, with route spending
  budget-capped per robot at the baseline's own executed lengths.

**The result** (each row: the pipeline run at its own coverage target, then
the joint refiner warm-started from it — same pose count, same route budget,
scored by the pipeline's ray-caster):

| V | pipeline | joint refinement | Δ |
|---|---|---|---|
| 15 | 0.662 @ 54.0 m | 0.673 @ 53.2 m | +1.1 pts, −1.5 % |
| 20 | 0.781 @ 67.4 m | 0.810 @ 61.8 m | +2.9 pts, −8.3 % |
| 28 | 0.859 @ 78.1 m | 0.882 @ 74.5 m | +2.4 pts, −4.6 % |
| 35 | 0.901 @ 86.2 m | 0.925 @ 79.5 m | +2.4 pts, −7.8 % |
| 49 | 0.951 @ 113.0 m | 0.960 @ 108.7 m | +0.9 pts, −3.8 % |

Every point improves **both** metrics at once; at matched coverage the
makespan savings read ≈10–20 % through the middle of the frontier, and the
V=49 result (0.960) is above anything the pipeline reached at any cost. Each
solve takes 4–7 seconds against the pipeline's ~174 s. A repeat run
reproduced its numbers bit-for-bit. **The §9.3 go/no-go bar is met.**

The honest fine print: pure route-shortening at a fixed coverage is
near-tight (~3 % — greedy's poses each hold unique points, so nothing can be
dropped); the wins come from the coverage-for-metres trade the staged
pipeline cannot express. Coordinated "front advance" moves are beyond
coordinate descent (the concrete motivation for a sequential-convexification
follow-up), all of this is one mesh / one pipeline seed / two robots at
pilot scale, and the robustness sweep is the first item below.

## 5 · Next steps (rewritten for the pivot, priority order)

1. ~~Inner solver for the warm arm~~ — **done 2026-07-24 (§4e): bar met.**
2. **Robustness sweep**: ≥3 pipeline seeds × the frontier, then R > 2 and the
   50 m mesh — nothing in §4e is a claim until this passes (the §3 rule).
3. **Cold start** with the same solver (+ adaptive term weighting).
4. **SCP / coordinated wave moves** — the measured front-advance limitation
   says exactly what a second-order method could still win.
5. Then: MPC/receding-horizon mode, theory framing, and the paper skeleton.
   The parked online results (§4.3/§4c) stay citable as history.

**Overall state in one sentence (2026-07-24, end of day):** the joint-OCP
direction has passed its go/no-go — an elastic-band coordinate-descent
refiner with ray-cast-audited acceptance Pareto-dominates the thesis
pipeline's own coverage/makespan frontier at every measured operating point
(matched pose and route budgets, their scorer, collision-clear, 4–7 s per
solve, bit-reproducible) — and the road ahead is robustness seeds, the cold
arm, and an SCP-style upgrade for the coordinated moves coordinate descent
provably cannot make.
