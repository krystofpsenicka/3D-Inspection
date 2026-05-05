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

| Symbol | File | Notes |
|---|---|---|
| `VRPBackend` enum | `core/types.py:11` | `HIGHS` / `CUOPT`. |
| `VRPResult` | `core/types.py:18` | `routes`, `total_cost`, `makespan`, `per_vehicle_costs`, `alpha`, `objective_value`, `best_bound`, `solver`, `status`. |
| `PlanningStats` | `core/types.py:44` | `wait_steps`, `detour_ratio`, `astar_failures`, `astar_retries`. |
| `ExecutionResult` | `core/types.py:58` | `all_traj_positions`, `all_traj_velocities`, `all_waypoints`, `initial_positions`, `fail_counts`, `actual_makespan`, `actual_per_vehicle_times`, `actual_per_leg_times`. |
| `compute_distance_matrix(og, waypoints_xyz)` | `core/distance_matrix.py:24` | GPU 26-conn. cuGraph Dijkstra → `(N,N)` float32 metres. |
| `compute_start_grid(num_robots, bmin, bmax)` | `core/geometry.py` | Depots offset from mesh AABB. |

### `VRP/vrp/`

| Symbol | File | Notes |
|---|---|---|
| `solve_vrp(dist_matrix, num_vehicles, depots, alpha=1.0, backend=HIGHS, time_limit=120, mip_gap=0.05) -> VRPResult` | `vrp/vrp_solver.py:32` | `alpha ∈ [0,1]`: 1 = makespan, 0 = total dist. |
| `MIPSolverCPU` | `vrp/mip_solver_cpu.py` | HiGHS via PuLP. |
| `MIPSolverGPU` | `vrp/mip_solver_gpu.py` | cuOpt. |
| `MIPModel` | `vrp/mip_model.py` | Shared β-aware formulation (per-vehicle tour UB, forbidden-pair cuts). |

### `VRP/mapf/`

| Symbol | File | Notes |
|---|---|---|
| `MultiAgentPathPlanner(start_positions, og)` | `mapf/mapf_planner.py` | Priority-based ST-A*. `execute(routes, waypoint_positions, waypoint_rotmats, home_indices, dist_matrix, alpha) -> ExecutionResult`. |
| `space_time_astar_gpu(coarse_og, start_ijk, goal_ijk, t_offset, reservation, ...)` | `mapf/space_time_search.py:28` | GPU 4D parallel-frontier A* (Zhou & Zeng). Returns `(path_ijk, t_steps)` or `None`. |
| `plan_robot_route_st(coarse_og, reservation, route, ...)` | `mapf/route_planner.py:35` | Per-robot full route (legs + dwells). Returns `(traj, t_indices, waypoints, PlanningStats)`. |
| `simplify_path_ompl(...)` | `mapf/path_smoother.py:27` | OMPL shortcut + B-spline, optional ST preservation. |
| `arc_length_resample(path, target_count)` | `mapf/path_smoother.py` | Uniform arc-length resample. |
| `apply_heading_orientation(...)` | `mapf/orientation.py` | Heading-blended quaternion track. |
| `ReservationTable(shape_xyzt)` | `mapf/reservation_table.py:18` | Dense 4D GPU grid: `is_reserved`, `is_reserved_batch`, `commit_trajectory`. |

### `VRP/utils/`

| Symbol | File | Notes |
|---|---|---|
| `save_solution` / `load_solution` | `utils/serialization.py:19, 47` | NPZ + JSON `ExecutionResult`. |
| `save_pipeline(pipeline_data, output_dir)` | `utils/serialization.py` | Full pipeline state for replay. |
| `find_trajectory_collisions(...)` | `utils/collision.py` | Post-execution collision check. |

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

| Symbol | File | Notes |
|---|---|---|
| `FrustumParams(fov_y, aspect, near, far)` | `core/types.py:18` | Pinhole frustum (radians). |
| `OptimizationResult` | `core/types.py:39` | `positions`, `rotations`, `visibility_map`, `total_coverage`, `num_viewpoints`, `redundancy`, `optimization_time`, `selected_indices`. |
| `normalize_vector(v)` | `core/types.py:10` | Safe unit-vector. |
| `compute_redundancy(visibility_map)` | `core/__init__.py` | Mean coverage count. |

`core/constants.py`: `DEFAULT_MAX_DISTANCE_OFFSET`, `DEFAULT_K_COVERAGE`, `NORM_EPS`, `RAYCAST_TOLERANCE`, `PROXIMITY_KNN_FRACTION`, `TARGETED_PROXIMITY_SIGMA_FACTOR`, `OPT_SAMPLER_MAXITER`, `OPT_SAMPLER_POPSIZE`, `OPT_SAMPLER_TRAVEL_ROT_FRACTION`, `OPT_SAMPLER_TRAVEL_WEIGHT`, `DEFAULT_MAX_VIEWPOINTS`, `DEFAULT_TARGET_COVERAGE`.

### `visibility/sampling/`

All return `(positions (N,3), rotmats (N,3,3))` CuPy.

| Class | File | Notes |
|---|---|---|
| `ViewpointSamplerBase` | `sampling/samplers/base.py:19` | Abstract; shared GPU state (OG, SDF, feasible-region cache). |
| `WeightedViewpointSampler` | `sampling/samplers/weighted.py:18` | Distance-weighted uniform, optional curvature bias. |
| `TargetedViewpointSampler` | `sampling/samplers/targeted.py:24` | Biased toward uncovered; supports k-coverage. |
| `OptimizingSampler` | `sampling/samplers/optimizing.py:57` | Wraps `OptimizationBackend` for CMA-ES targeted resampling. |
| `OptimizationBackend` | `sampling/samplers/optimizing.py:28` | Abstract: `optimize(objective_fn, n_dims, popsize, maxiter, ...) -> cp.ndarray`. |
| `CMAESBackend` | `sampling/samplers/optimization_backends/cmaes.py:35` | EvoTorch CMA-ES. |

### `visibility/visibility/`

All expose `compute_visibility(viewpoint, rotation)` and `compute_visibility_batch(positions, rotations) -> (V (K,M) uint8, time)`.

| Class | File | Notes |
|---|---|---|
| `VisibilityQueryBase` | `visibility/base.py:26` | Abstract. |
| `VisibilityQuery` | `visibility/base.py:51` | CPU base, KD-tree frustum culling. |
| `RaycastingVisibilityQuery` | `visibility/raycast.py:15` | Open3D RaycastingScene (CPU). |
| `RaycastingVisibilityQueryCuda` | `visibility/raycast_cuda.py` | OptiX BVH via `triro`. Used by full pipeline. |
| `EpsilonVisibilityQuery{,Cuda}` | `visibility/epsilon*.py` | Approximate ε-visibility (Lien 2009 + GPU batch). |
| `get_frustum_bounding_sphere(...)` | `visibility/base.py:13` | For spatial queries. |

### `visibility/set_cover/`

All expose `optimize(target_coverage=0.95, max_viewpoints=100) -> OptimizationResult`.

| Class | File | Notes |
|---|---|---|
| `IterativeSetCoverOptimizer` | `set_cover/base.py:17` | Abstract; subclasses override `select_next` / `commit_selection`. |
| `GreedySetCover` | `set_cover/greedy.py:13` | CPU NumPy greedy. |
| `GreedySetCoverCuda` | `set_cover/greedy_cuda.py` | GPU. |
| `LazyGreedySetCover` | `set_cover/lazy_greedy.py:16` | Minoux 1978 (default in full pipeline). |
| `ExpansionIterativeSetCover` | `set_cover/expansion.py` | Re-sampling-aware. |

---

## 6. `shared/`

| Symbol | File | Notes |
|---|---|---|
| `Side` enum | `types.py:6` | `OUTSIDE` / `INSIDE`. |
| `OccupancyGrid` | `occupancy_grid.py:20` | GPU 3D bool. `world_to_voxel`, `voxel_to_world`, `is_valid_voxel`, `is_free_world{,_batch}`, `world_to_flat_index`, `flat_index_to_world`. |
| `build_occupancy_grid(mesh, padding, inflation_voxels, resolution, fill_interior, complement_fill, extra_free_points, extra_margin_voxels)` | `grid_builder_utils.py` | Voxelise + inflate, optionally fill interior. |
| `compute_grid_bounds(mesh)` | `grid_builder_utils.py` | World AABB. |
| `voxelize_mesh(mesh, resolution)` | `grid_builder_utils.py` | Surface-only. |
| `downsample_occupancy_grid(og, coarse_res)` | `grid_utils.py:44` | Any-occupied downsampling. |
| `inflate_grid(grid, inflation_voxels)` | `grid_utils.py:21` | GPU spherical dilation. |
| `load_and_transform_mesh(mesh_path, target_length, pose)` | `mesh_loader.py:14` | Load, scale longest axis, apply pose; sets `metadata["scale_factor"]`. |
| `SurfacePointSampler.sample(mesh, num_points, ...)` | `surface_sampler.py:23` | Poisson-disk + normals; disk-cached `.cache/surface_samples/`. |
| `orient_normals_outward(points, normals)` | `geometry.py:11` | Flips inward via centroid. |
| `direction_to_rotmat(direction)` | `geometry.py:21` | Local +X → `direction`; (3,3). |
| `directions_rolls_to_rotmats` / inverse | `geometry.py` | Look-direction + roll ↔ (N,3,3). |

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
