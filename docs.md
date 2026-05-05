# 3D-Inspection — Documentation

Setup and quickstart in [`README.md`](README.md).

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Repository layout](#2-repository-layout)
3. [`VRP/`](#3-vrp)
4. [`VRP/core/constants.py`](#4-vrpcoreconstantspy)
5. [`visibility/`](#5-visibility)
6. [`shared/`](#6-shared)
7. [`visualization_isaac/`](#7-visualization_isaac)
8. [Scripts](#8-scripts)
9. [Experiments](#9-experiments)
10. [Tests](#10-tests)
11. [Coordinate-frame note](#11-coordinate-frame-note)

---

## 1. Pipeline overview

End-to-end flow ([`scripts/run_full_pipeline.py`](scripts/run_full_pipeline.py)):

```
mesh (.glb)
   │  shared/mesh_loader.py:load_and_transform_mesh         (1)
   ▼  target points + outward normals
   │  shared/surface_sampler.py:SurfacePointSampler         (2)
   ▼  sampling occupancy grid
   │  visibility/sampling/utils/sampling_grid_builder.py    (2b)
   ▼  candidate viewpoints (positions + rotmats)
   │  visibility/sampling/                                  (3)
   ▼  visibility matrix V (K×M)
   │  visibility/visibility/                                (4)
   ▼  selected viewpoints
   │  visibility/set_cover/                                 (5)
   ▼  VRP waypoints + depots + distance matrix
   │  VRP/core/distance_matrix.py:compute_distance_matrix    (7)
   ▼  VRP routes
   │  VRP/vrp/vrp_solver.py:solve_vrp                        (8)
   ▼  collision-free space-time trajectories
   │  VRP/mapf/mapf_planner.py:MultiAgentPathPlanner         (8)
   ▼  serialised pipeline data
   │  VRP/utils/serialization.py:save_pipeline               (9)
   ▼  Isaac Sim replay  →  scripts/visualize_full_pipeline.py
```

## 2. Repository layout

| Path | Purpose |
|------|---------|
| `VRP/` | VRP MIP, MAPF, occupancy-grid A* |
| `visibility/` | Viewpoint generation, visibility queries, set cover |
| `shared/` | Mesh, surface sampling, occupancy grid, geometry |
| `visualization_isaac/` | Isaac Sim wrapper + per-phase visualisers + replay |
| `scripts/` | Entry points |
| `tests/` | pytest unit tests |
| `experiments/` | `e01..e10` parametric studies (thesis figures) |
| `models/` | Default ship mesh + TOSCA |
| `assets/` | Optional BROV USD |
| `outputs/` | Pipeline artefacts |

---

## 3. `VRP/`

### `VRP/core/`

- **`VRPBackend`** (`core/types.py:11`) — Enum selecting the MIP backend used by `solve_vrp`: `HIGHS` (CPU via PuLP) or `CUOPT` (GPU).
- **`VRPResult`** (`core/types.py:19`) — Dataclass returned by both MIP backends. Holds per-vehicle `routes` (customer sequences, no depots), `total_cost`, `makespan`, `per_vehicle_costs`, the `alpha` blend used, the blended `objective_value`, the dual `best_bound` (0.0 if unavailable), `solver`, and `status`.
- **`PlanningStats`** (`core/types.py:45`) — Per-robot diagnostics from Space-Time A*: `wait_steps`, `detour_ratio` (actual / straight-line), `astar_failures`, `astar_retries`. Used by `MultiAgentPathPlanner` to reorder priorities on retry.
- **`ExecutionResult`** (`core/types.py:59`) — Final dense per-robot output: `all_traj_positions` / `all_traj_velocities` (each `(T_i, 6)` = `[x, y, z, yaw, cam_pitch, cam_roll]`), `all_waypoints` (`[x, y, z, qw, qx, qy, qz]`), `initial_positions`, `fail_counts`, `actual_makespan`, and per-vehicle / per-leg times.
- **`compute_distance_matrix(og, waypoints_xyz)`** (`core/distance_matrix.py:186`) — Builds the `(N, N)` float32 collision-free distance matrix in metres via cuGraph SSSP on the 26-connected occupancy graph. Runs in a subprocess to isolate RMM allocations, with optional `.npy` cache.
- **`compute_start_grid(num_robots, bmin, bmax, ...)`** (`core/geometry.py:13`) — Lays out depot positions in a regular grid offset from the mesh AABB so that all robots start collision-free above the structure.

### `VRP/vrp/`

- **`solve_vrp(dist_matrix, num_vehicles, depots, alpha=1.0, backend=HIGHS, time_limit=120, mip_gap=0.05) -> VRPResult`** (`vrp/vrp_solver.py:32`) — Single entry point for VRP. Generates a nearest-neighbour warm-start, dispatches to the chosen backend, and returns a `VRPResult`. `alpha ∈ [0, 1]` blends makespan (1) and total distance (0).
- **`MIPSolverCPU`** (`vrp/mip_solver_cpu.py:16`) — Wraps the shared `MIPModel` and solves it with HiGHS through PuLP. Reports the dual bound from the underlying solver when available.
- **`MIPSolverGPU`** (`vrp/mip_solver_gpu.py:64`) — Same MIP, solved in-process by NVIDIA cuOpt. Exports the PuLP model to MPS, runs cuOpt MILP, and reads the dual bound from cuOpt's MILP stats.
- **`MIPModel`** (`vrp/mip_model.py`) — Shared α-aware VRP formulation reused by both backends. Adds Desrochers–Laporte (1991) reachability filtering, lifted MTZ subtour elimination, per-vehicle tour upper bound, and forbidden-pair cuts on infeasible 3-arc paths.

### `VRP/mapf/`

- **`MultiAgentPathPlanner(start_positions, og)`** (`mapf/mapf_planner.py`) — Priority-based sequential MAPF planner. `execute(routes, waypoint_positions, waypoint_rotmats, home_indices, dist_matrix, alpha)` runs Space-Time A* per robot, tries multiple priority orderings on conflicts, densifies trajectories to replay resolution, and returns an `ExecutionResult`.
- **`space_time_astar_gpu(coarse_og, start_ijk, goal_ijk, t_offset, reservation, ...)`** (`mapf/space_time_search.py:28`) — GPU 4-D parallel-frontier A* (Zhou & Zeng 2015) over `(t, x, y, z)` with 27-connected neighbours (26 spatial + wait). Expands all frontier cells within `f_threshold_delta` of `f_min` and avoids committed robots via the `ReservationTable`. Returns `(path_ijk, t_steps)` or `None`.
- **`plan_robot_route_st(coarse_og, reservation, route, ...)`** (`mapf/route_planner.py:35`) — Plans one robot's full route leg-by-leg: per-leg ST-A* with a budget proportional to leg distance, optional OMPL smoothing, and commits the trajectory back into the reservation table. Returns `(traj, t_indices, waypoints, PlanningStats)`.
- **`simplify_path_ompl(...)`** (`mapf/path_smoother.py:27`) — OMPL `PathSimplifier` (shortcut + B-spline) on the fine grid, with optional two-stage space-time validity (any-time reservation rejection, then margin-based time check) so smoothed legs remain conflict-free.
- **`arc_length_resample(path, target_count)`** (`mapf/path_smoother.py:164`) — Resamples a 3-D path to a uniform arc-length grid; used to restore the ST-A* sample count after smoothing.
- **`apply_heading_orientation(...)`** (`mapf/orientation.py`) — Sets body yaw and camera pitch on a dense trajectory by reading forward axes from waypoint rotation matrices and easing between dwells with a cosine blend.
- **`ReservationTable(shape_xyzt)`** (`mapf/reservation_table.py:18`) — Dense 4-D GPU occupancy grid for committed robot trajectories, inflated by the collision radius. Provides `is_reserved`, `is_reserved_batch`, and `commit_trajectory`; queried by ST-A* and the smoother.

### `VRP/utils/`

- **`save_solution` / `load_solution`** (`utils/serialization.py:19`, `:47`) — Serialise an `ExecutionResult` as ragged NPZ (positions / velocities + offset arrays) plus a JSON sidecar (waypoints, fail counts).
- **`save_pipeline(pipeline_data, output_dir)`** (`utils/serialization.py:96`) — Wipes the directory and writes the full pipeline state for replay: `manifest.json`, point cloud, candidates, optimization result, VRP solution, and the `ExecutionResult` via `save_solution`.
- **`find_trajectory_collisions(...)`** (`utils/collision.py`) — Post-execution check: pads all robot trajectories to equal length and broadcasts a pairwise distance test on GPU, returning `(step, robot_a, robot_b, penetration_depth)` tuples.

---

## 4. `VRP/core/constants.py`

| Constant | Value | Description |
|---|---:|---|
| `MESH_PATH` | `models/duke_of_lancaster_uk_clipped.glb` | Default mesh. |
| `MESH_POSE` | `[0, 0, 1.5, 0, 1, 0, 0]` | `[x,y,z,qw,qx,qy,qz]`. |
| `MESH_TARGET_LENGTH` | `50.0` | Longest-axis (m). |
| `VOXEL_RESOLUTION` | `0.10` | OG cell (m). |
| `ROBOT_RADIUS` | `0.35` | Collision sphere (m). |
| `INFLATION_VOXELS` | `4` | `int(ROBOT_RADIUS / VOXEL_RESOLUTION) + 1`. |
| `SPACE_TIME_RESOLUTION` | `0.50` | Coarse OG for ST-A* (m). |
| `SPACE_TIME_DT` | `0.25` | ST-A* timestep (s). |
| `SPACE_TIME_MAX_HORIZON_S` | `400.0` | Max horizon (s). |
| `SPACE_TIME_DWELL_S` | `2.0` | Waypoint dwell (s). |
| `SPACE_TIME_SAFETY_FACTOR` | `3` | Velocity safety. |
| `SPACE_TIME_MIN_LEG_STEPS` | `20` | Min steps/leg. |
| `SPLINE_SAFETY_VOXELS` | `1` | OMPL clearance. |
| `TRAJ_DT` | `0.02` | Replay timestep (s). |
| `OMPL_SIMPLIFY_MAX_TIME` | `0.5` | OMPL budget (s). |
| `PATH_SMOOTHER_RESERVATION_MARGIN` | `10` | Smoother time-step margin. |
| `CAMERA_OFFSET_FORWARD` / `_UP` | `0.30` / `0.05` | Camera offsets (m). |
| `AUV_CRUISE_SPEED` | `2.0` | m/s. |
| `VRP_ALPHA` | `1.0` | Default blend (1=makespan, 0=dist). |
| `MIP_TIME_LIMIT` / `MIP_GAP` | `120` / `0.05` | MIP limits. |
| `GPU_SEARCH_MAX_ITERATIONS` | `100_000` | A* cap. |
| `OFFSETS_26 / WEIGHTS_26 / OFFSETS_27 / WEIGHTS_27` | CuPy | Neighbour offsets + Euclidean weights. |

---

## 5. `visibility/`

### `visibility/core/`

- **`FrustumParams(fov_y, aspect, near, far)`** (`core/types.py:19`) — Pinhole camera frustum (radians) passed to every visibility query.
- **`OptimizationResult`** (`core/types.py:40`) — Output of set-cover: `positions` (`K, 3`), `rotations` (`K, 3, 3`), `visibility_map` (`K, M` uint8), `total_coverage`, `num_viewpoints`, `redundancy`, `optimization_time`, `selected_indices`. All CuPy.
- **`normalize_vector(v)`** (`core/types.py:10`) — Safe unit-vector; returns zero if `‖v‖ < NORM_EPS`.
- **`compute_redundancy(visibility_map)`** (`core/utils.py:6`) — Mean number of selected viewpoints that see each covered point; 0.0 for empty / uncovered matrices.
- **Constants** (`core/constants.py`) — Tuning knobs: `DEFAULT_MAX_DISTANCE_OFFSET`, `DEFAULT_K_COVERAGE`, `NORM_EPS`, `RAYCAST_TOLERANCE`, `PROXIMITY_KNN_FRACTION`, `TARGETED_PROXIMITY_SIGMA_FACTOR`, `OPT_SAMPLER_MAXITER` / `_POPSIZE`, `OPT_SAMPLER_TRAVEL_*`, `DEFAULT_MAX_VIEWPOINTS`, `DEFAULT_TARGET_COVERAGE`.

### `visibility/sampling/`

All samplers return `(positions (N, 3), rotmats (N, 3, 3))` as CuPy arrays.

- **`ViewpointSamplerBase`** (`sampling/samplers/base.py:19`) — Abstract base that owns the GPU occupancy grid, SDF, and feasible-region cache; subclasses produce candidate viewpoints. Exposes the feasible region in OUTSIDE / INSIDE modes with optional curvature weighting.
- **`WeightedViewpointSampler`** (`sampling/samplers/weighted.py:18`) — Uniform free-space sampling weighted by `sdf²` distance, with KNN-derived viewing directions and optional curvature bias toward complex surface regions.
- **`TargetedViewpointSampler`** (`sampling/samplers/targeted.py:24`) — Weighted sampler biased toward currently uncovered points via proximity-weighted KNN; supports k-coverage and refreshes coverage between batches when a visibility query is provided.
- **`OptimizingSampler`** (`sampling/samplers/optimizing.py:57`) — Iterative resampler that drives an `OptimizationBackend` to search 6-DoF poses (`x, y, z, θ, φ, roll`), maximising deficit-weighted coverage minus a travel-distance penalty.
- **`OptimizationBackend`** (`sampling/samplers/optimizing.py:28`) — Abstract black-box optimiser interface: `optimize(objective_fn, n_dims, popsize, maxiter, ...) -> cp.ndarray` minimising a CuPy objective in `[0, 1]^D`.
- **`CMAESBackend`** (`sampling/samplers/optimization_backends/cmaes.py:35`) — EvoTorch-backed CMA-ES on CUDA with optional warm-start (`center_init`); the default backend behind `OptimizingSampler`.

### `visibility/visibility/`

All implementations expose `compute_visibility(viewpoint, rotation)` and `compute_visibility_batch(positions, rotations) -> (V (K, M) uint8, time)`.

- **`VisibilityQueryBase`** (`visibility/base.py:26`) — Abstract API; stores `frustum_params` and `num_points`.
- **`VisibilityQuery`** (`visibility/base.py:51`) — CPU base providing KD-tree frustum culling and a per-viewpoint batch loop (reference implementation, slow at scale).
- **`RaycastingVisibilityQuery`** (`visibility/raycast.py:15`) — Ground-truth visibility on CPU via Open3D `RaycastingScene`; a target is visible iff its ray hits within `RAYCAST_TOLERANCE` before any occluder.
- **`RaycastingVisibilityQueryCuda`** (`visibility/raycast_cuda.py`) — Same semantics on GPU using NVIDIA OptiX through `triro`, with a CUDA batch frustum-cull kernel up front. This is what the full pipeline runs.
- **`EpsilonVisibilityQuery{,Cuda}`** (`visibility/epsilon*.py`) — Approximate ε-visibility (Lien 2009): radial-binned occlusion test with auto-estimated ε from k-NN sample density. The CUDA variant ports the binning and back-face checks to custom kernels for batch speed.
- **`get_frustum_bounding_sphere(...)`** (`visibility/base.py:13`) — Bounding sphere for a frustum, used for spatial frustum culling via KD-tree queries.

### `visibility/set_cover/`

All optimisers expose `optimize(target_coverage=0.95, max_viewpoints=100) -> OptimizationResult`.

- **`IterativeSetCoverOptimizer`** (`set_cover/base.py:17`) — Abstract greedy-style driver. Subclasses override `select_next` and `commit_selection`; the base loops until coverage or viewpoint limit is hit, then assembles the `OptimizationResult`.
- **`GreedySetCover`** (`set_cover/greedy.py:13`) — CPU NumPy greedy: each step picks the candidate with the largest gain on uncovered points (matvec + argmax).
- **`GreedySetCoverCuda`** (`set_cover/greedy_cuda.py`) — Same algorithm fully on GPU via CuPy; substantially faster for large candidate / point counts.
- **`LazyGreedySetCover`** (`set_cover/lazy_greedy.py:16`) — Minoux 1978 lazy greedy with a max-heap of stale-checkable gains. Same solution quality as plain greedy but far fewer marginal-gain evaluations; default in the full pipeline.
- **`ExpansionIterativeSetCover`** (`set_cover/expansion.py`) — Wraps an inner optimiser with a local refinement step: every selection is locally re-sampled (probabilistic or CMA-ES) and only kept if it strictly improves uncovered coverage.

---

## 6. `shared/`

- **`Side`** (`types.py:6`) — Enum `OUTSIDE` / `INSIDE` selecting the inspection surface; controls grid fill semantics and feasible sampling region.
- **`OccupancyGrid`** (`occupancy_grid.py:21`) — GPU-resident 3-D bool grid (CuPy) with `origin` and `resolution`. Methods cover the world ↔ voxel ↔ flat-index round trip and free-space queries: `world_to_voxel`, `voxel_to_world`, `is_valid_voxel`, `is_free_world{,_batch}` (out-of-bounds is conservatively treated as occupied), `world_to_flat_index`, `flat_index_to_world`.
- **`build_occupancy_grid(mesh, padding, inflation_voxels, resolution, fill_interior, complement_fill, extra_free_points, extra_margin_voxels)`** (`grid_builder_utils.py:162`) — End-to-end builder: derives bounds, voxelises the mesh, optionally fills the interior or inverts (`complement_fill` for inside inspection), inflates by `inflation_voxels`, and returns a planning-ready `OccupancyGrid`.
- **`compute_grid_bounds(mesh, ...)`** (`grid_builder_utils.py:107`) — World-AABB-derived `(origin, shape)` for the grid; can grow the box to contain `extra_free_points` with margin.
- **`voxelize_mesh(mesh, resolution)`** (`grid_builder_utils.py:71`) — Surface-only voxelisation on GPU; returns `(raw_grid, filled_grid)` CuPy bool arrays (filled only when `fill_interior=True`).
- **`downsample_occupancy_grid(og, coarse_res)`** (`grid_utils.py:44`) — Any-occupied downsampling: a coarse cell is occupied if any fine cell within it is. Preserves "no false free space"; used to build the ST-A* coarse grid.
- **`inflate_grid(grid, inflation_voxels)`** (`grid_utils.py:21`) — GPU spherical dilation so planners can treat the robot as a point.
- **`load_and_transform_mesh(mesh_path, target_length, pose)`** (`mesh_loader.py:14`) — Loads `.glb` / `.obj` / `.stl`, scales the longest axis to `target_length`, applies the SE(3) `pose` `[x, y, z, qw, qx, qy, qz]`, and stores the scale factor in `mesh.metadata["scale_factor"]`.
- **`SurfacePointSampler.sample(mesh, num_points, ...)`** (`surface_sampler.py:39`) — Open3D Poisson-disk sampling + normal estimation + tangent-plane consistency, then outward orientation. Disk-cached under `.cache/surface_samples/` keyed by mesh hash and parameters; repeat calls return instantly.
- **`orient_normals_outward(points, normals)`** (`geometry.py:11`) — Flips normals globally if their mean dot with radial vectors from the centroid is negative; ensures outward-pointing surface normals.
- **`direction_to_rotmat(direction)`** (`geometry.py:21`) — Single-direction Rodrigues rotation that maps local `+X` (camera forward) onto `direction`; handles the 180° degenerate case.
- **`directions_rolls_to_rotmats` / inverse `rotmats_to_directions_rolls`** (`geometry.py:75`, `:101`) — Batch GPU conversions between `(N, 3)` directions + `(N,)` rolls and `(N, 3, 3)` rotation matrices, with stable up-vector fallback when the direction aligns with `±Z`.

---

## 7. `visualization_isaac/`

| Symbol | File | Notes |
|---|---|---|
| `IsaacApp(headless=False, renderer="RayTracedLighting", physics_dt=1/60, rendering_dt=1/60, low_quality=True)` | `app.py:61` | Context manager, yields `IsaacContext`. |
| `IsaacContext` | `app.py:28` | `app`, `world`, `stage`, `default_prim_path`, `is_isaac_45`, `extras`; `step`, `update`, `is_running`, `close`. |
| Stage helpers | `_helpers.py`, `_usd_primitives.py` | `add_ground_plane`, `add_dome_light`, `add_distant_light`, `add_zero_gravity`, `frame_viewport`, `set_camera_lookat`, `generate_tab20_colors`, `rotmat_to_quat_wxyz`. |
| `Phase`, `PhaseController` | `phases.py` | Named phase + keyboard-driven controller. |
| Visibility visualisers | `visibility/` | `ModelVisualizer`, `SamplingVisualizer`, `VisibilityVisualizer`, `SetCoverVisualizer`, `EsdfVisualizer`, `add_frustum_lineset`, `add_viewpoint_geometry`. |
| VRP visualisers | `vrp/` | `VRPVisualizer`, `ReplayVisualizer` (`convert_trajectories`, `traj8_to_pose`), `add_brov_robots`, `ROBOT_COLORS`. |

---

## 8. Scripts

### `scripts/run_full_pipeline.py`

| Flag | Default | Description |
|---|---|---|
| `--output, -o` | `outputs/full_pipeline` | Output dir (overwritten). |
| `--num_robots, -n` | `5` | AUVs. |
| `--mesh_target_length` | `50.0` | Longest-axis (m). |
| `--num_surface_points` | `200_000` | |
| `--num_candidates` | `1500` | |
| `--target_coverage` | `0.95` | |
| `--frustum_near / _far / _fov_deg / _aspect` | `0.1 / 6.0 / 40.0 / 1.0` | |
| `--solver` | `cuopt` | `cuopt` / `highs`. |
| `--alpha` | `0.5` | 1 = makespan, 0 = total dist. |
| `--seed` | `42` | |
| `--curvature_weighting` | off | Bias toward complex regions. |
| `--resample_fraction` | `0.0` | Targeted-resampling fraction. |
| `--resampling_strategy` | `optimal` | `random` / `optimal` (CMA-ES). |
| `--k_coverage` | `1` | Per-target redundancy. |
| `--side` | `outside` | `outside` / `inside`. |
| `-v, --verbose` | off | |

### `scripts/visualize_full_pipeline.py`

Positional: pipeline output directory.

| Flag | Description |
|---|---|
| `--brov-usd` | Path (default `assets/robot/brov/BROV_high.usd`). |
| `--use-brov-usd` | Spawn BROV USD references (slow; default = cuboids). |

Keys: `N`/→ next, `P`/← prev, `Q`/`Esc` quit.

### Other

- `scripts/generate_dataset_figures.py` — thesis dataset figures.
- `scripts/measure_mapf_conflict_density.py` — MAPF conflict rates vs. pure-VRP; writes `results/mapf_conflict_density.csv`.
- `visibility/scripts/visualize_sampling.py` — single sampler stage viz.

---

## 9. Experiments

Under `experiments/`, results into `experiments/results/`.

| Script | Topic |
|---|---|
| `e01_sampling_strategy.py` | Three sampling strategies (thesis-final). |
| `e02_candidate_scaling.py` | Candidate count vs. coverage / runtime. |
| `e03_iterative_sampler_params.py` | Iterative-sampler param sweep. |
| `e04_curvature_sensitivity.py` | Curvature-aware weighting sensitivity. |
| `e05_visibility_comparison.py` | Raycast vs. ε-visibility (CPU/GPU). |
| `e06_set_cover_optimizers.py` | Greedy / lazy-greedy / expansion. |
| `e07_vrp_fleet_scaling.py` | VRP runtime vs. fleet size. |
| `e08_vrp_alpha_blending.py` | Impact of `--alpha` on routing. |
| `e09_sampler_routing_impact.py` | Sampler choice → makespan / cost? |
| `e10_cross_model.py` | Generalisation across mesh complexity (TOSCA). |
| `run_all.py` | Orchestrator. |

---

## 10. Tests

```bash
python -m pytest tests/ -v
```

| Test | Coverage |
|---|---|
| `test_distance_matrix.py` | cuGraph 26-conn. Dijkstra. |
| `test_geometry.py` | `direction_to_rotmat`, frustum maths. |
| `test_mapf_factoradic.py` | Permutation encoding. |
| `test_occupancy_grid.py` | Inflate, downsample, world↔voxel. |
| `test_orientation.py` | Heading interpolation. |
| `test_path_smoother.py` | OMPL + B-spline. |
| `test_proximity_kernel.py` | Proximity-weighted kernel. |
| `test_reservation_table.py` | 4D conflict checks. |
| `test_route_planner.py` | Per-robot ST-A*. |
| `test_sampling.py` | Sampler validity. |
| `test_sampling_space.py` | Feasible region. |
| `test_serialization.py` | NPZ + JSON round-trip. |
| `test_set_cover.py` | Greedy / lazy / expansion. |
| `test_space_time_astar.py` | GPU parallel-frontier A*. |
| `test_trajectory_collisions.py` | Post-execution collisions. |
| `test_visibility_query.py` | Raycast + ε (CPU/GPU). |
| `test_vrp.py` | Full solver (HiGHS + cuOpt). |
| `test_vrp_helpers.py` | Cost helpers, transforms. |

---

## 11. Coordinate-frame note

Default mesh is GLB (Y-up). Isaac's GLB→USD prepends Y-up→Z-up (`+90°` X) before any user `xformOps`, but `trimesh` loads raw. To align, `scripts/run_full_pipeline.py:48–65` composes:

```
R_combined = R_mesh_pose(180° X) @ R_y2z(+90° X) = R_x(270°) = R_x(−90°)
quat (qw,qx,qy,qz) = (cos(−45°), sin(−45°), 0, 0) = (√2/2, −√2/2, 0, 0)
```

If you replace the mesh or change `MESH_POSE` (`VRP/core/constants.py:13`), reconcile this rotation manually — otherwise Isaac viz will be flipped relative to planned trajectories.
