# Test & Analysis Report

Date: 2026-04-03

Full test run of all pytest tests, visibility scripts, VRP scripts, and the
end-to-end pipeline.  This document records findings, bugs discovered, fixes
applied, and remaining issues.

---

## 1  Pytest Results

**75 passed, 9 skipped, 0 failed** (after fixes below).

| Test file | Tests | Result | Notes |
|-----------|-------|--------|-------|
| `test_astar.py` | 10 | pass | After removing dead class + fixing args |
| `test_occupancy_grid.py` | 12 | pass | |
| `test_sampling.py` | 5 | pass | Requires GPU (CuPy) |
| `test_serialization.py` | 4 | pass | |
| `test_vrp.py` | 36 | 27 pass, 9 skip | `mip_gpu` skipped (RAPIDS env absent) |
| `test_vrp_solver.py` | 8 | pass | |

---

## 2  Bugs Found and Fixed

### 2.1  `tests/test_astar.py` -- dead test class

`TestDistanceMatrixCPU` referenced `compute_distance_matrix_cpu`, a function
that no longer exists (removed during GPU-only refactor).  The three tests
always failed with `NameError`.

**Fix:** Removed the entire class.

### 2.2  `tests/test_astar.py` -- wrong `astar_path` arguments

`TestExtractAstarPath` called `extract_astar_path(small_og, pt, pt)` passing
an `OccupancyGrid` object, but the function signature is
`astar_path(grid, start_world, goal_world, origin, resolution)`.

**Fix:** Changed calls to
`extract_astar_path(small_og.grid, pt, pt, small_og.origin, small_og.resolution)`.

### 2.3  `tests/test_vrp.py` -- CuPy/NumPy interop in Space-Time A* tests

Six tests (`TestSpaceTimeAStar` x 4, `TestCoordinateTransforms` x 2) passed
NumPy arrays to functions expecting CuPy arrays, causing `TypeError: Implicit
conversion to a NumPy array is not allowed`.

**Fix:** Converted test inputs to CuPy (`cp.array(...)`, `cp.zeros(...)`) and
outputs back to NumPy (`.get()`) for assertions.

### 2.4  `run_full_pipeline.py:543` -- undefined `visibility_map`

Line 543 stored `visibility_map` in the pipeline dict, but the variable was
never assigned.  Should reference `opt_result.visibility_map`.

**Fix:** `"visibility_map": cp.asnumpy(opt_result.visibility_map)`.

### 2.5  `shared/__init__.py` -- eager `open3d` import breaks subprocess

The `__init__.py` eagerly imported `SurfacePointSampler` which pulls in
`open3d`.  This broke the cuGraph subprocess (runs in the RAPIDS conda env
without `open3d`).

**Fix:** Removed `from .surface_sampler import SurfacePointSampler` from
`__init__.py`.  All callers already use `from shared.surface_sampler import
SurfacePointSampler` directly.

### 2.6  `VRP/mapf/route_executor.py:118` -- keyword mismatch

Called `plan_robot_route_st(..., fine_og=self.og)` but the parameter is named
`fine_occupancy_grid`.

**Fix:** Changed to `fine_occupancy_grid=self.og`.

### 2.7  `VRP/mapf/route_planner.py` -- NumPy grid passed to GPU function

`plan_robot_route_st` received a NumPy `coarse_grid` from `route_executor`
but `space_time_astar_gpu` indexes it as a CuPy array.

**Fix:** Added `coarse_grid = cp.asarray(coarse_grid)` at function entry.

### 2.8  `VRP/mapf/route_executor.py` -- CuPy arrays in `np.interp`

Trajectory densification used `np.interp` on CuPy arrays from
`plan_robot_route_st`, causing
`NotImplementedError: Only python scalars or ndarrays are supported for v`.

**Fix:** Added `.get()` to convert `w_xyz` and `c_t` to NumPy before
densification.

### 2.9  `visibility/methods/epsilon_cuda.py:538` -- `cp.lexsort` API

`cp.lexsort((arr_a, arr_b))` passes a Python tuple.  Unlike NumPy, CuPy
requires a stacked 2D array.

**Fix:** `cp.lexsort(cp.stack([front_pair_distances, front_pair_vp_indices]))`.

---

## 3  Script Results

### 3.1  Visibility scripts

| Script | Status | Notes |
|--------|--------|-------|
| `compare_gpu_vs_cpu.py` | Pass | Epsilon 3.0x, Raycast 2.0x GPU speedup |
| `check_visibility.py` | Pass | Computation OK; hangs on Open3D GUI |
| `compare_vs_raycast.py` | Pass | F1 = 1.000 (epsilon matches raycast) |
| `compare_optimizers.py` | Timeout | Heavy benchmark (expected) |
| `compare_visibility_methods.py` | Timeout | Heavy benchmark (expected) |
| `visualize_esdf.py --mode 2d` | Pass | Saved `esdf_slice_Z1.5.png` |
| `visualize_sampling.py` | Fail | API mismatch (see 4.1) |

### 3.2  VRP scripts

| Script | Status | Notes |
|--------|--------|-------|
| `run_vrp.py` | Pass | 2 robots, 5 WPs, 0 failures, 25.8 s makespan |
| `evaluate_vrp.py` | Pass | Figures saved to `/tmp/vrp_eval` |
| `visualize_esdf.py --mode 2d` | Fail | Missing `raw_grid` attribute (see 4.2) |
| `clip_ship_mesh.py` | Skipped | Needs unclipped source mesh |

### 3.3  Full pipeline (`run_full_pipeline.py`)

Completed end-to-end with reduced parameters
(`--num_surface_points 5000 --num_candidates 50 --target_coverage 0.80
--num_robots 2 --solver ortools`).

- 5000 surface points, 50 candidates, 32 selected viewpoints
- 2 robots, 0 failures, 103.3 s actual makespan
- Output saved: `test_pipeline_data.{npz,json}` + `_exec.{npz,json}`
- Runtime: ~58 min (dominated by 20-trial priority-order search in route
  execution)

---

## 4  Remaining Issues (Not Fixed)

### 4.1  `visualize_sampling.py` -- `TargetedViewpointSampler.sample()` API

The script calls `sampler.sample(num_candidates=..., side=...)` but
`TargetedViewpointSampler.sample()` requires `uncovered_indices` as its first
positional argument.  This was likely broken when the sampler API was
refactored to support targeted resampling.

### 4.2  `VRP/scripts/visualize_esdf.py` -- `raw_grid` attribute

The script calls `og.raw_grid` but `VRP.core.occupancy_grid.build_occupancy_grid`
returns a plain `OccupancyGrid` which has no `raw_grid` attribute.  Only
`SamplingOccupancyGrid` (from the visibility module) carries this field.

### 4.3  OptimizingSampler (CMA-ES) fails to find covering viewpoints

See dedicated analysis in section 5.

---

## 5  OptimizingSampler Analysis

### Problem

When `run_full_pipeline.py` is run with `--resampling_strategy optimal`, the
`OptimizingSampler` (CMA-ES backend) typically finds zero viewpoints that
cover any under-covered surface points.  It logs:

    [OptimizingSampler] Backend found no useful viewpoint -- stopping.

and returns an empty array.

### Root cause: random initialization in a degenerate landscape

The failure chain:

1. **Random CMA-ES init.**
   `CMAESBackend.optimize()` creates the CMAES searcher without a
   `center_init` (`cmaes.py:54`).  EvoTorch defaults to a random point in
   `[0,1]^6`.

2. **Flat fitness landscape.**
   The objective (`optimizing.py:221-288`) computes
   `f_obs = V @ under_k_mask / n_under_k` -- the fraction of under-covered
   points visible.  A random camera pose almost always sees **zero**
   under-covered points, so `f_obs = 0` for the entire initial population.

3. **No gradient signal.**
   With `f_obs = 0` everywhere, the only nonzero term is
   `travel_weight * travel_cost` (~0.05-0.1), which is nearly flat.
   CMA-ES cannot learn a useful covariance.

4. **Early stopping.**
   After the backend returns, `_optimize_one` recomputes the best solution's
   actual coverage (`optimizing.py:313-318`).  If `best_score == 0`, the
   outer loop immediately breaks (`optimizing.py:185-187`).

5. **Small budget.**
   With `popsize=15` and `maxiter=20`, only 300 evaluations are made in a
   6-D space.  This is far too few to randomly stumble onto a covering pose.

### Why warm-starting would fix it

The infrastructure already exists but is unused at initialization time.
`sample_optimized()` receives `existing_pos_gpu` / `existing_rot_gpu` (the
uniform-phase viewpoints), stores them in `all_pos/all_rot`, but only uses
them to compute an **average pose for the travel penalty**.  The CMA-ES
initial mean is still random.

A warm-start would:

- Normalize the best existing viewpoint (highest coverage of under-covered
  points) into `[0,1]^6` and pass it as `center_init` to the CMAES
  constructor.
- Guarantee the initial population is centered on a region with known nonzero
  coverage.
- Let CMA-ES refine from a good starting point rather than searching blindly.

### Relevant code locations

| File | Lines | What |
|------|-------|------|
| `visibility/sampling/samplers/optimizing.py` | 155-208 | Main loop; calls `_optimize_one` per round |
| `visibility/sampling/samplers/optimizing.py` | 180-183 | Calls backend **without** init guidance |
| `visibility/sampling/samplers/optimizing.py` | 185-187 | Early stop on `score == 0` |
| `visibility/sampling/samplers/optimizing.py` | 221-288 | Objective: `-f_obs + travel * cost` |
| `visibility/sampling/samplers/optimization_backends/cmaes.py` | 54 | `CMAES(...)` -- missing `center_init` |
| `visibility/core/constants.py` | 39-42 | `popsize=15, maxiter=20` |
| `run_full_pipeline.py` | 316-333 | Pipeline call with `existing_pos_gpu` |
