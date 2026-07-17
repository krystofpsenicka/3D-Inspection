# Where the project stands — plain-language overview (2026-07-17)

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

## 5 · Next steps, broadly, in priority order

1. **Head-to-head against NeOF (the 2024 prior-art paper) — the biggest open risk.**
   A reviewer's first question will be "how is this not NeOF with a different visibility
   model?" Our answers (trajectories/online, no per-scene field fitting, the
   exploitation analysis) must be backed by running their public code on our scenes:
   compare coverage, their own quality metrics, and — most importantly — their
   field-refitting cost vs our per-step cost as the number of cameras and points grows.
   **This is what I am starting on now.**
2. **Sharpen the outdated-mesh demo** (optional but cheap): deeper collapse or a breach
   into the interior, so the open-loop arm visibly fails rather than just being slower.
3. **No-prior exploration variant**: start with *no* mesh at all and build the plan from
   the accumulating sensor cloud. This is where a point-cloud visibility model is the
   *only* possible evaluator (nothing to ray-cast), i.e. the strongest motivation, and
   most of the machinery (discovery halo, belief updates) now exists.
4. **Operator-layer upgrades that the trajectory objective still lacks** (Stage A steps
   2–3): a back-face/normal gate (don't count surface seen edge-on or from behind),
   measurement-quality weighting (incidence angle, distance), and cheap per-scene
   self-calibration of the visibility model against a few ray-cast audit poses.
5. **System/paper work**: multi-mesh statistics (ModelNet/TOSCA), CMA-ES and
   more-candidates baselines, VRP routing integration, an Isaac Sim rollout of an
   optimized trajectory as the closing figure, then the draft. Venue: RA-L.

**Overall state in one sentence:** the environment is fully reproducible (126/126 tests),
the operator layer (Stage A) is done, the trajectory planner (Stage B) now ties the
oracle offline and decisively beats the classical pipeline when the world changes — the
paper's headline is measured, and the remaining risk is the direct comparison with NeOF.
