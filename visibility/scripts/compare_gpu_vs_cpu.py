"""
Benchmark GPU (CUDA) vs CPU visibility methods.

Compares:
  - Frustum culling: CPU KDTree vs GPU brute-force
  - Visibility: CPU Epsilon vs GPU Epsilon, CPU Raycast vs GPU Raycast (OptiX)
  - Optimization: CPU Greedy vs GPU Greedy
"""
import cupy as cp
import numpy as np
import open3d as o3d
import os
import sys
from time import time as get_time
from typing import List, Tuple

from visibility.core.types import FrustumParams
from visibility.sampling import WeightedViewpointSampler
from visibility.core import orient_normals_outward

# CPU methods
from visibility.methods.raycast import RaycastingVisibilityQuery
from visibility.methods.epsilon import EpsilonVisibilityQuery
from visibility.optimizers.greedy import GreedyOptimizer
from visibility.optimizers.kernel_greedy import KernelGreedyOptimizer

# GPU methods
from visibility.methods.epsilon_cuda import EpsilonVisibilityQueryCuda
from visibility.optimizers.greedy_cuda import GreedyOptimizerCuda
from visibility.optimizers.kernel_greedy_cuda import KernelGreedyOptimizerCuda

# Optionally import GPU raycast (requires Triro/OptiX)
try:
    from visibility.methods.raycast_cuda import RaycastingVisibilityQueryCuda
    HAS_TRIRO = True
except ImportError:
    HAS_TRIRO = False
    print("[WARNING] Triro not available — skipping GPU raycast benchmarks")


# ===========================================================================
# CONFIGURATION
# ===========================================================================
NUM_TARGET_POINTS = 100000
NUM_CANDIDATE_VPs = 1500
TARGET_COVERAGE = 0.90
MAX_VIEWPOINTS = 100
NUM_WARMUP_VPS = 5      # warm up GPU before timing
NUM_TIMED_VPS = 50       # viewpoints to time per-query benchmark


def load_scene(mesh_path=None):
    """Load mesh and sample target points."""
    if mesh_path is not None:
        print(f"Loading mesh from: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    else:
        print("Using default sphere mesh")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5.0, resolution=100)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(number_of_points=NUM_TARGET_POINTS)
    target_points = np.asarray(pcd.points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals)
    normals = orient_normals_outward(target_points, normals)

    frustum_params = FrustumParams(fov_y=np.deg2rad(45), aspect=1.0, near=0.01, far=7)

    return mesh, target_points, normals, frustum_params


def benchmark_single_visibility(query, positions, rotmats, label, is_gpu=False):
    """Benchmark per-viewpoint visibility computation."""
    # Warm up
    for i in range(NUM_WARMUP_VPS):
        query.compute_visibility(positions[i], rotmats[i])

    times = []
    total_visible = 0
    for i in range(NUM_WARMUP_VPS, NUM_WARMUP_VPS + NUM_TIMED_VPS):
        vis, t = query.compute_visibility(positions[i], rotmats[i])
        times.append(t)
        total_visible += len(vis)

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_visible = total_visible / len(times)

    print(f"  [{label}] Per-viewpoint: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms "
          f"(avg {avg_visible:.0f} visible points)")
    return avg_time, std_time


def benchmark_frustum_culling(query_cpu, query_gpu, positions_cpu, rotmats_cpu,
                              positions_gpu, rotmats_gpu):
    """Compare frustum culling strategies."""
    print("\n" + "=" * 70)
    print("FRUSTUM CULLING BENCHMARK")
    print("=" * 70)

    # CPU KDTree
    times_cpu = []
    for i in range(NUM_TIMED_VPS):
        t0 = get_time()
        idx = query_cpu.points_in_frustum_with_kdtree(positions_cpu[i], rotmats_cpu[i])
        times_cpu.append(get_time() - t0)
    print(f"  CPU KDTree:        {np.mean(times_cpu)*1000:.2f} ± {np.std(times_cpu)*1000:.2f} ms")

    # GPU brute-force — warm up
    for i in range(3):
        query_gpu.points_in_frustum_gpu(positions_gpu[i], rotmats_gpu[i])
    times_gpu = []
    for i in range(NUM_TIMED_VPS):
        t0 = get_time()
        idx = query_gpu.points_in_frustum_gpu(positions_gpu[i], rotmats_gpu[i])
        times_gpu.append(get_time() - t0)
    print(f"  GPU Brute-force:   {np.mean(times_gpu)*1000:.2f} ± {np.std(times_gpu)*1000:.2f} ms")

    speedup = np.mean(times_cpu) / np.mean(times_gpu)
    print(f"  Speedup: {speedup:.1f}x")


def benchmark_visibility_methods(queries, positions_cpu, rotmats_cpu,
                                 positions_gpu, rotmats_gpu):
    """Benchmark all visibility methods."""
    print("\n" + "=" * 70)
    print("VISIBILITY COMPUTATION BENCHMARK")
    print("=" * 70)

    results = {}
    for label, query, is_gpu in queries:
        if is_gpu:
            avg_t, std_t = benchmark_single_visibility(
                query, positions_gpu, rotmats_gpu, label, is_gpu=True)
        else:
            avg_t, std_t = benchmark_single_visibility(
                query, positions_cpu, rotmats_cpu, label, is_gpu=False)
        results[label] = avg_t

    # Print speedups
    print("\n  Speedups:")
    if "CPU Epsilon" in results and "GPU Epsilon" in results:
        s = results["CPU Epsilon"] / results["GPU Epsilon"]
        print(f"    Epsilon GPU speedup: {s:.1f}x")
    if "CPU Raycast" in results and "GPU Raycast (OptiX)" in results:
        s = results["CPU Raycast"] / results["GPU Raycast (OptiX)"]
        print(f"    Raycast GPU speedup: {s:.1f}x")

    return results


def benchmark_optimization(optimizer_configs, positions, rotmats):
    """Benchmark optimizers given as (label, optimizer) pairs."""
    print("\n" + "=" * 70)
    print("GREEDY OPTIMIZER BENCHMARK")
    print("=" * 70)

    subset_pos = positions[:min(500, len(positions))]
    subset_rot = rotmats[:min(500, len(rotmats))]

    for label, optimizer in optimizer_configs:
        print(f"\n  --- {label} ---")

        t0 = get_time()
        result = optimizer.optimize(
            positions=subset_pos,
            rotmats=subset_rot,
            target_coverage=TARGET_COVERAGE,
            max_viewpoints=MAX_VIEWPOINTS,
        )
        total_t = get_time() - t0

        print(f"  [{label}] {result.num_viewpoints} VPs, "
              f"coverage={result.total_coverage*100:.1f}%, "
              f"vis_time={result.visibility_computation_time:.2f}s, "
              f"opt_time={result.optimization_time:.2f}s, "
              f"total={total_t:.2f}s")


def main():
    mesh_path = None
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]

    mesh, target_points, normals, frustum_params = load_scene(mesh_path)

    print(f"\nScene: {len(target_points)} target points, "
          f"mesh with {len(np.asarray(mesh.triangles))} triangles")

    # Generate candidate viewpoints (GPU arrays from sampler)
    sampler = WeightedViewpointSampler(mesh, target_points, normals, frustum_params.far, collision_radius=0.5)
    pos_gpu, rot_gpu = sampler.sample(num_candidates=NUM_CANDIDATE_VPs, side="outside")

    # Transfer to CPU for CPU benchmarks
    positions = cp.asnumpy(pos_gpu)
    rotmats = cp.asnumpy(rot_gpu)

    # =====================================================================
    # Initialize queries
    # =====================================================================
    print("\n[INIT] Setting up visibility queries...")

    # CPU
    t0 = get_time()
    q_raycast_cpu = RaycastingVisibilityQuery(mesh, target_points, normals, frustum_params)
    print(f"  CPU Raycast init: {get_time()-t0:.2f}s")

    t0 = get_time()
    q_epsilon_cpu = EpsilonVisibilityQuery(target_points, normals, frustum_params)
    print(f"  CPU Epsilon init: {get_time()-t0:.2f}s")

    # GPU Epsilon
    t0 = get_time()
    q_epsilon_gpu = EpsilonVisibilityQueryCuda(
        target_points, normals, frustum_params,
    )
    print(f"  GPU Epsilon init: {get_time()-t0:.2f}s")

    # GPU Raycast (if Triro available)
    q_raycast_gpu = None
    if HAS_TRIRO:
        t0 = get_time()
        q_raycast_gpu = RaycastingVisibilityQueryCuda(
            mesh, target_points, normals, frustum_params
        )
        print(f"  GPU Raycast (OptiX) init: {get_time()-t0:.2f}s")

    # =====================================================================
    # Benchmarks
    # =====================================================================

    # 1. Frustum culling
    benchmark_frustum_culling(q_epsilon_cpu, q_epsilon_gpu,
                              positions, rotmats, pos_gpu, rot_gpu)

    # 2. Visibility computation
    queries = [
        ("CPU Epsilon", q_epsilon_cpu, False),
        ("GPU Epsilon", q_epsilon_gpu, True),
        ("CPU Raycast", q_raycast_cpu, False),
    ]
    if q_raycast_gpu is not None:
        queries.append(("GPU Raycast (OptiX)", q_raycast_gpu, True))

    benchmark_visibility_methods(queries, positions, rotmats, pos_gpu, rot_gpu)

    # 3. Optimization benchmark
    opt_configs = [
        ("CPU Epsilon + CPU Greedy",       GreedyOptimizer(q_epsilon_cpu)),
        ("GPU Epsilon + GPU Greedy",       GreedyOptimizerCuda(q_epsilon_gpu)),
        ("CPU Epsilon + KernelGreedy",     KernelGreedyOptimizer(q_epsilon_cpu)),
        ("GPU Epsilon + KernelGreedy GPU", KernelGreedyOptimizerCuda(q_epsilon_gpu)),
        ("CPU Raycast + CPU Greedy",       GreedyOptimizer(q_raycast_cpu)),
    ]
    if q_raycast_gpu is not None:
        opt_configs.append(("GPU Raycast + GPU Greedy", GreedyOptimizerCuda(q_raycast_gpu)))
    benchmark_optimization(opt_configs, positions, rotmats)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
