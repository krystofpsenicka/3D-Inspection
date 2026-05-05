# 3D-Inspection — Documentation

This document is the full reference for the codebase. For setup and a quickstart, see [`README.md`](README.md).

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Repository layout](#2-repository-layout)
3. [`VRP/` — vehicle routing & multi-agent path-finding](#3-vrp--vehicle-routing--multi-agent-path-finding)
4. [`VRP/core/constants.py` — full configuration table](#4-vrpcoreconstantspy--full-configuration-table)
5. [`visibility/` — viewpoint sampling, visibility, set cover](#5-visibility--viewpoint-sampling-visibility-set-cover)
6. [`shared/` — mesh, sampling, occupancy grid](#6-shared--mesh-sampling-occupancy-grid)
7. [`visualization_isaac/` — Isaac Sim integration](#7-visualization_isaac--isaac-sim-integration)
8. [Scripts](#8-scripts)
9. [Experiments](#9-experiments)
10. [Tests](#10-tests)
11. [Coordinate-frame note (Y-up GLB ↔ Z-up Isaac)](#11-coordinate-frame-note-y-up-glb--z-up-isaac)

---

## 1. Pipeline overview

End-to-end flow as implemented in [`scripts/run_full_pipeline.py`](scripts/run_full_pipeline.py):

```
mesh (.glb)
   │  shared/mesh_loader.py:load_and_transform_mesh         (stage 1)
   ▼
target points + outward normals
   │  shared/surface_sampler.py:SurfacePointSampler         (stage 2)
   ▼
sampling occupancy grid
   │  visibility/sampling/utils/sampling_grid_builder.py    (stage 2b)
   ▼
candidate viewpoints (positions + rotation matrices)
   │  visibility/sampling/  (Targeted / Weighted / Optimizing)  (stage 3)
   ▼
visibility matrix V (K candidates × M targets)
   │  visibility/visibility/  (Raycasting | Epsilon × CPU/CUDA)  (stage 4)
   ▼
selected viewpoints (subset achieving target coverage)
   │  visibility/set_cover/  (LazyGreedy | Greedy | Expansion)   (stage 5)
   ▼
VRP waypoints + depots + distance matrix
   │  VRP/core/distance_matrix.py:compute_distance_matrix         (stage 7)
   ▼
VRP routes
   │  VRP/vrp/vrp_solver.py:solve_vrp  (HiGHS | cuOpt)            (stage 8)
   ▼
collision-free space-time trajectories per robot
   │  VRP/mapf/mapf_planner.py:MultiAgentPathPlanner              (stage 8)
   ▼
serialised pipeline data (NPZ + JSON)
   │  VRP/utils/serialization.py:save_pipeline                    (stage 9)
   ▼
Isaac Sim phase-by-phase replay  →  scripts/visualize_full_pipeline.py
```

## 2. Repository layout

| Path | Purpose |
|------|---------|
| `VRP/` | Vehicle Routing Problem solving (MIP), multi-agent path finding, occupancy-grid Dijkstra |
| `visibility/` | GPU-accelerated viewpoint generation, visibility queries, set-cover optimisation |
| `shared/` | Mesh loading, surface sampling, occupancy grid, geometry utilities |
| `visualization_isaac/` | Isaac Sim wrapper, per-phase visualisers (visibility / VRP), trajectory replay |
| `scripts/` | Entry points: full pipeline, Isaac replay, dataset figure generation, MAPF benchmark |
| `tests/` | pytest unit tests for every module (18 files) |
| `experiments/` | `e01..e10` parametric studies that produced the thesis figures |
| `models/` | Default mesh (`duke_of_lancaster_uk_clipped.glb`) + optional TOSCA dataset |
| `assets/` | Optional BROV USD for Isaac Sim replay (`assets/robot/brov/BROV_high.usd`) |
| `outputs/` | Default destination for pipeline artefacts |
| `Bachelor-Thesis/` | LaTeX sources, figures, primary-source PDFs |

---

## 3. `VRP/` — vehicle routing & multi-agent path-finding

### `VRP/core/`

| Symbol | File | Purpose |
|--------|------|---------|
| `VRPBackend` (enum) | `core/types.py:11` | MIP backend selector (`HIGHS` / `CUOPT`). |
| `VRPResult` (dataclass) | `core/types.py:18` | `routes`, `total_cost`, `makespan`, `per_vehicle_costs`, `alpha`, `objective_value`, `best_bound`, `solver`, `status`. |
| `PlanningStats` (dataclass) | `core/types.py:44` | `wait_steps`, `detour_ratio`, `astar_failures`, `astar_retries`. |
| `ExecutionResult` (dataclass) | `core/types.py:58` | `all_traj_positions`, `all_traj_velocities`, `all_waypoints`, `initial_positions`, `fail_counts`, `actual_makespan`, `actual_per_vehicle_times`, `actual_per_leg_times`. |
| `compute_distance_matrix(occupancy_grid, waypoints_xyz)` | `core/distance_matrix.py:24` | GPU 26-connected Dijkstra (cuGraph) → `(N, N)` float32 distance matrix in metres. |
| `compute_start_grid(num_robots, mesh_bmin, mesh_bmax)` | `core/geometry.py` | Lays out depots on a grid offset from the mesh AABB. |

### `VRP/vrp/`

| Symbol | File | Purpose |
|--------|------|---------|
| `solve_vrp(dist_matrix, num_vehicles, depots, alpha=1.0, backend=VRPBackend.HIGHS, time_limit=120, mip_gap=0.05) -> VRPResult` | `vrp/vrp_solver.py:32` | Unified MIP entry point. `alpha ∈ [0,1]` blends makespan (`1.0`) vs. total distance (`0.0`). |
| `MIPSolverCPU` | `vrp/mip_solver_cpu.py` | HiGHS-backed solver via PuLP. |
| `MIPSolverGPU` | `vrp/mip_solver_gpu.py` | cuOpt-backed solver. |
| `MIPModel` | `vrp/mip_model.py` | Shared β-aware MIP formulation (per-vehicle tour upper bound, forbidden-pair cuts). |

### `VRP/mapf/`

| Symbol | File | Purpose |
|--------|------|---------|
| `MultiAgentPathPlanner(start_positions, og)` | `mapf/mapf_planner.py` | Priority-based sequential Space-Time A*. `execute(routes, waypoint_positions, waypoint_rotmats, home_indices, dist_matrix, alpha) -> ExecutionResult`. |
| `space_time_astar_gpu(coarse_og, start_ijk, goal_ijk, t_offset, reservation, time_step_cost=0.01, max_time_steps=0, max_iterations=100_000, f_threshold_delta=2.0)` | `mapf/space_time_search.py:28` | GPU parallel-frontier 4D A* (Zhou & Zeng extension). Returns `(path_ijk, t_steps)` or `None`. |
| `plan_robot_route_st(coarse_og, reservation, route, waypoint_positions, dwell_s=2.0, dt=0.25, fine_occupancy_grid=None, robot_radius=0.35, cruise_speed=2.0)` | `mapf/route_planner.py:35` | Plan one robot's full route: legs + waypoint dwells. Returns `(traj, t_indices, waypoints, PlanningStats)`. |
| `simplify_path_ompl(path_xyz, occupancy_grid, robot_radius=0.35, max_time=0.5, reservation=None, time_steps=None, coarse_og=None, reservation_margin=10)` | `mapf/path_smoother.py:27` | OMPL shortcut + B-spline smoothing with optional space-time constraint preservation. |
| `arc_length_resample(path, target_count)` | `mapf/path_smoother.py` | Uniform arc-length resampling. |
| `apply_heading_orientation(positions, target_orientations, dt)` | `mapf/orientation.py` | Heading-blended quaternion track. |
| `ReservationTable(shape_xyzt)` | `mapf/reservation_table.py:18` | Dense 4D GPU reservation grid. Methods: `is_reserved(x,y,z,t)`, `is_reserved_batch()`, `commit_trajectory()`. |

### `VRP/utils/`

| Symbol | File | Purpose |
|--------|------|---------|
| `save_solution(result, path)` / `load_solution(path)` | `utils/serialization.py:19, 47` | NPZ + JSON serialisation of `ExecutionResult`. |
| `save_pipeline(pipeline_data, output_dir)` | `utils/serialization.py` | Saves the full pipeline state used by `scripts/visualize_full_pipeline.py`. |
| `find_trajectory_collisions(trajectories, occupancy_grid, robot_radius)` | `utils/collision.py` | Post-execution collision validation. |

---

## 4. `VRP/core/constants.py` — full configuration table

| Constant | Value | Description |
|----------|------:|-------------|
| `MESH_PATH` | `models/duke_of_lancaster_uk_clipped.glb` | Default mesh path. |
| `MESH_POSE` | `[0, 0, 1.5, 0.0, 1.0, 0.0, 0.0]` | Mesh pose `[x, y, z, qw, qx, qy, qz]`. |
| `MESH_TARGET_LENGTH` | `50.0` | Target length (m) of the mesh's longest axis. |
| `VOXEL_RESOLUTION` | `0.10` | Occupancy-grid cell size (m). |
| `ROBOT_RADIUS` | `0.35` | Collision-sphere radius (m). |
| `INFLATION_VOXELS` | `4` | `int(ROBOT_RADIUS / VOXEL_RESOLUTION) + 1`. |
| `SPACE_TIME_RESOLUTION` | `0.50` | Coarse occupancy-grid resolution for ST-A* (m). |
| `SPACE_TIME_DT` | `0.25` | ST-A* time step (s). |
| `SPACE_TIME_MAX_HORIZON_S` | `400.0` | Max ST-A* planning horizon (s). |
| `SPACE_TIME_DWELL_S` | `2.0` | Dwell time at each waypoint (s). |
| `SPACE_TIME_SAFETY_FACTOR` | `3` | Velocity safety multiplier. |
| `SPACE_TIME_MIN_LEG_STEPS` | `20` | Minimum time steps per leg. |
| `SPLINE_SAFETY_VOXELS` | `1` | Clearance voxels for OMPL smoothing. |
| `TRAJ_DT` | `0.02` | Replay trajectory timestep (s). |
| `OMPL_SIMPLIFY_MAX_TIME` | `0.5` | OMPL budget (s). |
| `PATH_SMOOTHER_RESERVATION_MARGIN` | `10` | Time-step margin for reservation checks during smoothing. |
| `CAMERA_OFFSET_FORWARD` | `0.30` | Camera offset ahead of robot (m). |
| `CAMERA_OFFSET_UP` | `0.05` | Camera offset above robot (m). |
| `AUV_CRUISE_SPEED` | `2.0` | AUV cruise speed (m/s). |
| `VRP_ALPHA` | `1.0` | Default objective blending (1 = makespan, 0 = total distance). |
| `MIP_TIME_LIMIT` | `120` | MIP solver time limit (s). |
| `MIP_GAP` | `0.05` | MIP relative optimality gap (5 %). |
| `GPU_SEARCH_MAX_ITERATIONS` | `100_000` | Parallel-frontier A* iteration cap. |
| `OFFSETS_26 / WEIGHTS_26 / OFFSETS_27 / WEIGHTS_27` | CuPy arrays | 26-/27-connectivity neighbour offsets and Euclidean step weights. |

---

## 5. `visibility/` — viewpoint sampling, visibility, set cover

### `visibility/core/`

| Symbol | File | Purpose |
|--------|------|---------|
| `FrustumParams(fov_y, aspect, near, far)` | `core/types.py:18` | Pinhole frustum description (radians). |
| `OptimizationResult` | `core/types.py:39` | `positions (K,3)`, `rotations (K,3,3)`, `visibility_map (K,M)`, `total_coverage`, `num_viewpoints`, `redundancy`, `optimization_time`, `selected_indices`. |
| `normalize_vector(v)` | `core/types.py:10` | Safe unit-vector normalisation. |
| `compute_redundancy(visibility_map)` | `core/__init__.py` | Mean coverage count per target. |

Selected constants in `visibility/core/constants.py`: `DEFAULT_MAX_DISTANCE_OFFSET`, `DEFAULT_K_COVERAGE`, `NORM_EPS`, `RAYCAST_TOLERANCE`, `PROXIMITY_KNN_FRACTION`, `TARGETED_PROXIMITY_SIGMA_FACTOR`, `OPT_SAMPLER_MAXITER`, `OPT_SAMPLER_POPSIZE`, `OPT_SAMPLER_TRAVEL_ROT_FRACTION`, `OPT_SAMPLER_TRAVEL_WEIGHT`, `DEFAULT_MAX_VIEWPOINTS`, `DEFAULT_TARGET_COVERAGE`.

### `visibility/sampling/`

All samplers return `(positions: cp.ndarray (N,3), rotmats: cp.ndarray (N,3,3))`.

| Class | File | Notes |
|-------|------|-------|
| `ViewpointSamplerBase` | `sampling/samplers/base.py:19` | Abstract base; holds shared GPU state (occupancy grid, SDF, feasible-region cache). |
| `WeightedViewpointSampler` | `sampling/samplers/weighted.py:18` | Distance-weighted uniform sampling with optional curvature bias. `sample(num_candidates, side=Side.OUTSIDE, min_distance=None, max_distance_offset=…, curvature_weighting=False, max_dir_noise_rad=…)`. |
| `TargetedViewpointSampler` | `sampling/samplers/targeted.py:24` | Biased toward uncovered regions; supports k-coverage. `sample(uncovered_indices, num_candidates, side=…, curvature_weighting=False, visibility_query=None, k_coverage=3, coverage_count_gpu=None, …)`. |
| `OptimizingSampler` | `sampling/samplers/optimizing.py:57` | Wraps an `OptimizationBackend` for CMA-ES targeted resampling. `sample_optimized(n_targeted, coverage_count_gpu, raycast_query, existing_pos_gpu, existing_rot_gpu, k_coverage)`. |
| `OptimizationBackend` | `sampling/samplers/optimizing.py:28` | Abstract: `optimize(objective_fn, n_dims, popsize, maxiter, verbose=False, center_init=None) -> cp.ndarray`. |
| `CMAESBackend` | `sampling/samplers/optimization_backends/cmaes.py:35` | EvoTorch CMA-ES implementation. |

### `visibility/visibility/`

All visibility queries expose `compute_visibility(viewpoint, rotation)` and `compute_visibility_batch(positions, rotations) -> (V, time)` where `V` is `(K, M)` uint8.

| Class | File | Description |
|-------|------|-------------|
| `VisibilityQueryBase` | `visibility/base.py:26` | Abstract base. |
| `VisibilityQuery` | `visibility/base.py:51` | CPU base with KD-tree frustum culling. |
| `RaycastingVisibilityQuery` | `visibility/raycast.py:15` | Open3D RaycastingScene (CPU). |
| `RaycastingVisibilityQueryCuda` | `visibility/raycast_cuda.py` | OptiX BVH via `triro` (GPU). Used in `run_full_pipeline.py`. |
| `EpsilonVisibilityQuery` / `EpsilonVisibilityQueryCuda` | `visibility/epsilon*.py` | Approximate ε-visibility (Lien 2009 + GPU batch adaptation). |
| `get_frustum_bounding_sphere(viewpoint, direction, params)` | `visibility/base.py:13` | Bounding sphere for spatial queries. |

### `visibility/set_cover/`

All optimisers expose `optimize(target_coverage=0.95, max_viewpoints=100) -> OptimizationResult`.

| Class | File | Notes |
|-------|------|-------|
| `IterativeSetCoverOptimizer` | `set_cover/base.py:17` | Abstract base. Subclasses override `select_next` / `commit_selection`. |
| `GreedySetCover` | `set_cover/greedy.py:13` | Standard greedy (CPU, NumPy). |
| `GreedySetCoverCuda` | `set_cover/greedy_cuda.py` | GPU variant. |
| `LazyGreedySetCover` | `set_cover/lazy_greedy.py:16` | Minoux-1978 lazy greedy (default in `run_full_pipeline.py`). |
| `ExpansionIterativeSetCover` | `set_cover/expansion.py` | Generates new candidates per iteration (re-sampling-aware). |

---

## 6. `shared/` — mesh, sampling, occupancy grid

| Symbol | File | Purpose |
|--------|------|---------|
| `Side` (enum) | `types.py:6` | `OUTSIDE` / `INSIDE` — drives normal direction and grid filling. |
| `OccupancyGrid` (dataclass) | `occupancy_grid.py:20` | GPU-resident 3D bool grid. Methods: `world_to_voxel`, `voxel_to_world`, `is_valid_voxel`, `is_free_world`, `is_free_world_batch`, `world_to_flat_index`, `flat_index_to_world`. |
| `build_occupancy_grid(mesh, padding, inflation_voxels, resolution, fill_interior, complement_fill, extra_free_points, extra_margin_voxels)` | `grid_builder_utils.py` | Voxelises a mesh, inflates obstacles, optionally fills the interior. |
| `compute_grid_bounds(mesh)` | `grid_builder_utils.py` | World-frame AABB. |
| `voxelize_mesh(mesh, resolution)` | `grid_builder_utils.py` | Surface-only voxelisation. |
| `downsample_occupancy_grid(og, coarse_res)` | `grid_utils.py:44` | Any-occupied downsampling for ST-A* and distance matrix. |
| `inflate_grid(grid, inflation_voxels)` | `grid_utils.py:21` | GPU spherical morphological dilation. |
| `load_and_transform_mesh(mesh_path, target_length, pose)` | `mesh_loader.py:14` | Loads, scales (longest axis = `target_length`), applies `pose`. Returns `trimesh.Trimesh` with `metadata["scale_factor"]`. |
| `SurfacePointSampler.sample(mesh, num_points, normal_radius=0.5, normal_max_nn=30, tangent_plane_k=15, seed=None)` | `surface_sampler.py:23` | Poisson-disk surface sampling with normal estimation; disk-cached under `.cache/surface_samples/`. |
| `orient_normals_outward(points, normals)` | `geometry.py:11` | Flips inward-pointing normals using mesh centroid. |
| `direction_to_rotmat(direction)` | `geometry.py:21` | Maps local +X to `direction`; `(3,3)` rotation matrix. |
| `directions_rolls_to_rotmats(directions, rolls)` | `geometry.py` | Combines look-direction + roll into `(N,3,3)`. |
| `rotmats_to_directions_rolls(rotmats)` | `geometry.py` | Inverse of the above. |

---

## 7. `visualization_isaac/` — Isaac Sim integration

| Symbol | File | Purpose |
|--------|------|---------|
| `IsaacApp(headless=False, renderer="RayTracedLighting", physics_dt=1/60, rendering_dt=1/60, low_quality=True)` | `app.py:61` | Context manager that boots `SimulationApp`. Yields an `IsaacContext`. |
| `IsaacContext` (dataclass) | `app.py:28` | `app`, `world`, `stage`, `default_prim_path`, `is_isaac_45`, `extras`. Methods: `step(render=True)`, `update()`, `is_running()`, `close()`. |
| Stage helpers | `_helpers.py`, `_usd_primitives.py` | `add_ground_plane`, `add_dome_light`, `add_distant_light`, `add_zero_gravity`, `frame_viewport`, `set_camera_lookat`, `generate_tab20_colors`, `rotmat_to_quat_wxyz`. |
| `Phase`, `PhaseController` | `phases.py` | Named phase + keyboard-driven controller for the replay UI. |
| Visibility visualisers | `visibility/` | `ModelVisualizer`, `SamplingVisualizer`, `VisibilityVisualizer`, `SetCoverVisualizer`, `EsdfVisualizer`, `add_frustum_lineset`, `add_viewpoint_geometry`. |
| VRP visualisers | `vrp/` | `VRPVisualizer`, `ReplayVisualizer` (`convert_trajectories`, `traj8_to_pose`), `add_brov_robots(stage, base_path, num_robots, brov_usd_path)`, `ROBOT_COLORS`. |

---

## 8. Scripts

### `scripts/run_full_pipeline.py` — full pipeline (computation only)

| Flag | Default | Description |
|------|---------|-------------|
| `--output, -o` | `outputs/full_pipeline` | Output directory (overwritten each run). |
| `--num_robots, -n` | `5` | Number of AUV robots. |
| `--mesh_target_length` | `50.0` | Target mesh length along longest axis (m). |
| `--num_surface_points` | `200_000` | Surface points sampled. |
| `--num_candidates` | `1500` | Candidate viewpoints generated. |
| `--target_coverage` | `0.95` | Greedy set-cover target. |
| `--frustum_near` | `0.1` | Near plane (m). |
| `--frustum_far` | `6.0` | Far plane (m). |
| `--frustum_fov_deg` | `40.0` | Vertical FOV (degrees). |
| `--frustum_aspect` | `1.0` | Width / height. |
| `--solver` | `cuopt` | `cuopt` (GPU MIP) or `highs` (CPU MIP). |
| `--alpha` | `0.5` | Objective blend (1 = makespan, 0 = total distance). |
| `--seed` | `42` | RNG seed. |
| `--curvature_weighting` | off | Bias sampling toward complex regions. |
| `--resample_fraction` | `0.0` | Fraction of candidates from targeted resampling. |
| `--resampling_strategy` | `optimal` | `random` (proximity-weighted) or `optimal` (CMA-ES). |
| `--k_coverage` | `1` | Per-target coverage redundancy. |
| `--side` | `outside` | `outside` or `inside` inspection. |
| `--verbose, -v` | off | Debug logging. |

### `scripts/visualize_full_pipeline.py` — Isaac Sim phase-by-phase replay

Positional argument: pipeline output directory (e.g. `outputs/full_pipeline`).

| Flag | Description |
|------|-------------|
| `--brov-usd` | Path to `BROV_high.usd` (default: `assets/robot/brov/BROV_high.usd`). |
| `--use-brov-usd` | Spawn full BROV USD references for replay (slow; default uses lightweight cuboids). |

Keyboard controls inside the viewer: `N` / right arrow → next phase, `P` / left arrow → previous, `Q` / `Esc` → quit.

### Other scripts

- `scripts/generate_dataset_figures.py` — render the dataset figures used in the thesis (`Bachelor-Thesis/img/datasets/`).
- `scripts/measure_mapf_conflict_density.py` — benchmark MAPF conflict rates against pure-VRP trajectories; writes `results/mapf_conflict_density.csv`.
- `visibility/scripts/visualize_sampling.py` — standalone visualisation of a single sampler stage.

---

## 9. Experiments

All under `experiments/`. Each is a self-contained script that runs the pipeline (or a subset of it) across a parameter sweep and writes results into `experiments/results/`.

| Script | Topic |
|--------|-------|
| `e01_sampling_strategy.py` | Compare three candidate-sampling strategies (thesis-final config). |
| `e02_candidate_scaling.py` | Effect of candidate count on coverage and runtime. |
| `e03_iterative_sampler_params.py` | Parameter sweep for the iterative sampler. |
| `e04_curvature_sensitivity.py` | Sensitivity of curvature-aware weighting. |
| `e05_visibility_comparison.py` | Compare raycasting vs. ε-visibility (CPU/GPU). |
| `e06_set_cover_optimizers.py` | Greedy vs. lazy-greedy vs. expansion. |
| `e07_vrp_fleet_scaling.py` | VRP runtime vs. fleet size. |
| `e08_vrp_alpha_blending.py` | Impact of `--alpha` on routing. |
| `e09_sampler_routing_impact.py` | Does the sampler choice affect final makespan / cost? |
| `e10_cross_model.py` | Pipeline generalisation across mesh complexity (TOSCA dataset). |
| `run_all.py` | Orchestrator that runs the full sweep. |

---

## 10. Tests

```bash
python -m pytest tests/ -v
```

| Test | Coverage |
|------|----------|
| `test_distance_matrix.py` | cuGraph 26-connected Dijkstra. |
| `test_geometry.py` | `direction_to_rotmat`, frustum maths. |
| `test_mapf_factoradic.py` | Permutation encoding for ST-A*. |
| `test_occupancy_grid.py` | Inflation, downsampling, world↔voxel. |
| `test_orientation.py` | Heading interpolation. |
| `test_path_smoother.py` | OMPL shortcut + B-spline smoother. |
| `test_proximity_kernel.py` | Proximity-weighted sampling kernel. |
| `test_reservation_table.py` | 4D reservation conflict checks. |
| `test_route_planner.py` | Per-robot ST-A* route planning. |
| `test_sampling.py` | Sampler positions/rotations validity. |
| `test_sampling_space.py` | Feasible region construction. |
| `test_serialization.py` | NPZ + JSON round-trip. |
| `test_set_cover.py` | Greedy / lazy-greedy / expansion. |
| `test_space_time_astar.py` | GPU parallel-frontier A*. |
| `test_trajectory_collisions.py` | Post-execution collision detection. |
| `test_visibility_query.py` | Raycast & ε-visibility (CPU/GPU). |
| `test_vrp.py` | Full VRP solver (HiGHS + cuOpt). |
| `test_vrp_helpers.py` | Cost helpers, coordinate transforms. |

---

## 11. Coordinate-frame note (Y-up GLB ↔ Z-up Isaac)

The default ship mesh is a glTF/GLB file, which stores vertices in Y-up. Isaac Sim's GLB→USD converter implicitly prepends a Y-up→Z-up rotation (`+90°` about X) before any user `xformOps`, but `trimesh` loads raw coordinates. To keep the trimesh-loaded mesh aligned with what Isaac shows, `scripts/run_full_pipeline.py:48–65` composes:

```
R_combined = R_mesh_pose(180° X) @ R_y2z(+90° X) = R_x(270°) = R_x(−90°)
quaternion (qw, qx, qy, qz) = (cos(−45°), sin(−45°), 0, 0) = (√2/2, −√2/2, 0, 0)
```

If you replace the mesh or change `MESH_POSE` (`VRP/core/constants.py:13`), reconcile this rotation manually or your Isaac visualisation will be flipped relative to the planned trajectories.
