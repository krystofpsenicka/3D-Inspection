# Codebase Review Checklist

Files ordered by dependency depth (leaves first). ~11,400 lines to review + 2 deprecated files to skim.

---

## Day 1: Shared Utilities + Visibility Core (~1,210 lines)

### Leaves (no internal imports)
- [ x ] 1. `shared/geometry.py` (57 lines) — pose transforms, quaternion helpers
- [ x ] 2. `shared/grid_utils.py` (25 lines) — grid indexing, neighbor queries
- [ x ] 3. `shared/mesh_loader.py` (60 lines) — trimesh I/O, normal estimation, inflation
- [ x ] 4. `shared/occupancy_grid.py` (257 lines) — ESDF computation, surface-only OG
- [ x ] 5. `shared/__init__.py` (1 line)
- [ ] 6. `visibility/core/constants.py` (54 lines) — magic numbers, default params
- [ x ] 7. `visibility/core/utils.py` (11 lines) — small helpers

### Tests for shared
- [ x ] 8. `tests/conftest.py` (35 lines) — shared fixtures
- [ x ] 11. `tests/test_occupancy_grid.py` (206 lines)

### Visibility core (types -> base -> base_cuda -> sampling engine)
- [ x ] 13. `visibility/core/types.py` (58 lines) — depends on constants.py
- [ x ] 15. `visibility/core/base.py` (136 lines) — CPU visibility base class
- [ x ] 16. `visibility/core/base_cuda.py` (126 lines) — GPU visibility base class
- [ x ] 17. `visibility/core/sampling.py` (~10 lines) — **DEPRECATED** (stale copy) skim header, likely delete
- [ x ] 18. `visibility/core/__init__.py` (5 lines)

---

## Day 2: Visibility Sampling + Methods + Optimizers + Viz (~3,520 lines)

### Sampling
- [ x ] 19. `visibility/sampling/curvature.py` (57 lines) — depends on constants.py
- [ x ] 20. `visibility/sampling/targeted.py` (189 lines) — depends on constants.py
- [ x ] 21. `visibility/sampling/base.py` (788 lines) — main GPU viewpoint sampler
- [ x ] 22. `visibility/sampling/__init__.py` (1 line)
- [ x ] 23. `tests/test_sampling.py` (228 lines)

### Visibility methods (CPU then GPU for each)
- [ x ] 24. `visibility/methods/raycast.py` (62 lines) — CPU raycast
- [ x ] 25. `visibility/methods/raycast_cuda.py` (85 lines) — GPU raycast
- [ x ] 26. `visibility/methods/epsilon.py` (202 lines) — CPU epsilon
- [ x ] 27. `visibility/methods/epsilon_cuda.py` (239 lines) — GPU epsilon
- [ x ] 28. `visibility/methods/__init__.py` (4 lines)

### Optimizers (read CPU/GPU pairs together)
- [ x ] 29. `visibility/optimizers/greedy.py` (117 lines) — CPU greedy set-cover
- [ x ] 30. `visibility/optimizers/greedy_cuda.py` (117 lines) — GPU greedy set-cover
- [ x ] 31. `visibility/optimizers/lazy_greedy.py` (155 lines) — CPU lazy greedy
- [ x ] 32. `visibility/optimizers/lazy_greedy_cuda.py` (156 lines) — GPU lazy greedy
- [ x ] 33. `visibility/optimizers/kernel_greedy.py` (138 lines) — CPU kernel greedy
- [ x ] 34. `visibility/optimizers/kernel_greedy_cuda.py` (155 lines) — GPU kernel greedy
- [ x ] 35. `visibility/optimizers/__init__.py` (6 lines)

### Visualization + top-level init
- [ ] 36. `visibility/visualization.py` (808 lines)
- [ x ] 37. `visibility/__init__.py` (15 lines)

---

## Day 3: VRP Foundation + Core + Solver + Routing (~3,570 lines)

### VRP leaves (no internal imports)
- [ ] 38. `VRP/config.py` (112 lines) — RAPIDS paths, solver config
- [ ] 39. `VRP/solver/cuopt_subprocess.py` (174 lines) — subprocess wrapper for NVIDIA cuOpt
- [ ] 40. `VRP/solver/mip_makespan_solver.py` (642 lines) — OR-Tools MIP makespan formulation
- [ ] 41. `VRP/solver/mip_makespan_subprocess.py` (124 lines) — subprocess wrapper for MIP

### VRP utilities + solver interface
- [ ] 42. `VRP/utils.py` (175 lines) — depends on config.py
- [ ] 43. `VRP/solver/vrp_solver.py` (530 lines) — main solver dispatch
- [ ] 44. `VRP/solver/__init__.py` (4 lines)

### VRP solver tests
- [ ] 45. `tests/test_vrp.py` (766 lines) — main VRP test suite
- [ ] 46. `tests/test_vrp_solver.py` (85 lines)

### VRP core infrastructure
- [ ] 47. `VRP/core/gpu_distance_matrix.py` (551 lines) — cuGraph Dijkstra
- [ ] 48. `VRP/core/occupancy_grid.py` (338 lines) — VRP-specific OG wrapper
- [ ] 49. `VRP/core/waypoint_loader.py` (431 lines) — viewpoint to pose conversion
- [ ] 50. `VRP/core/__init__.py` (5 lines)

### A* tests
- [ ] 51. `tests/test_astar.py` (103 lines)

### VRP routing
- [ ] 52. `VRP/routing/traffic_light.py` (~10 lines) — **DEPRECATED** (superseded by Space-Time A*) skim header, likely delete
- [ ] 53. `VRP/routing/space_time_astar.py` (659 lines) — collision-free trajectory planning

---

## Day 4: VRP Routing Execution + Scripts + Viz + Pipeline (~3,100 lines)

### VRP route execution
- [ ] 54. `VRP/routing/route_executor.py` (553 lines) — route command sequencing
- [ ] 55. `VRP/routing/__init__.py` (10 lines)
- [ x ] 56. `tests/test_serialization.py` (107 lines)

### VRP visualization
- [ ] 58. `VRP/viz/visualize_solution.py` (106 lines) — standalone solution viewer
- [ ] 59. `VRP/viz/visualization.py` (353 lines) — main VRP visualization
- [ ] 60. `VRP/viz/__init__.py` (1 line)

### VRP scripts
- [ ] 61. `VRP/scripts/clip_ship_mesh.py` (95 lines)
- [ ] 62. `VRP/scripts/vrp_planner.py` (348 lines) — full VRP planning orchestration
- [ ] 63. `VRP/scripts/run_vrp.py` (154 lines) — main VRP CLI entry point
- [ ] 64. `VRP/scripts/evaluate_vrp.py` (621 lines) — benchmarking
- [ ] 65. `VRP/scripts/__init__.py` (1 line)
- [ ] 66. `VRP/__init__.py` (51 lines)

### Visibility scripts (comparison/benchmarking tools)
- [ ] 67. `visibility/scripts/check_visibility.py` (107 lines)
- [ ] 68. `visibility/scripts/compare_vs_raycast.py` (201 lines)
- [ ] 69. `visibility/scripts/compare_optimizers.py` (261 lines)
- [ ] 70. `visibility/scripts/compare_gpu_vs_cpu.py` (277 lines)
- [ ] 71. `visibility/scripts/compare_visibility_methods.py` (310 lines)
- [ ] 72. `visibility/scripts/visualize_sampling.py` (284 lines)
- [ ] 73. `visibility/scripts/visualize_esdf.py` (306 lines) — standalone ESDF renderer

### Main pipeline (the capstone)
- [ ] 73. `run_full_pipeline.py` (565 lines) — end-to-end orchestration
