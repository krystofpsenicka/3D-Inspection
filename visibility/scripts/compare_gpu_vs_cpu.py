"""
Benchmark GPU (CUDA) vs CPU visibility methods.

Compares:
  - Frustum culling: CPU KDTree vs GPU brute-force vs GPU KDTree-hybrid
  - Visibility: CPU Epsilon vs GPU Epsilon, CPU Raycast vs GPU Raycast (OptiX)
  - Optimization: CPU Greedy vs GPU Greedy
"""
import numpy as np
import open3d as o3d
import os
import sys
from time import time as get_time
from typing import List, Tuple

from visibility.core.types import FrustumParams
from visibility.core.sampling import ViewpointSampler
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


def benchmark_single_visibility(query, viewpoints, label):
    """Benchmark per-viewpoint visibility computation."""
    # Warm up
    for vp, d in viewpoints[:NUM_WARMUP_VPS]:
        query.compute_visibility(vp, d)

    times = []
    total_visible = 0
    for vp, d in viewpoints[NUM_WARMUP_VPS:NUM_WARMUP_VPS + NUM_TIMED_VPS]:
        vis, t = query.compute_visibility(vp, d)
        times.append(t)
        total_visible += len(vis)

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_visible = total_visible / len(times)

    print(f"  [{label}] Per-viewpoint: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms "
          f"(avg {avg_visible:.0f} visible points)")
    return avg_time, std_time


def benchmark_frustum_culling(query_cpu, query_gpu_bf, query_gpu_kd, viewpoints):
    """Compare frustum culling strategies."""
    print("\n" + "=" * 70)
    print("FRUSTUM CULLING BENCHMARK")
    print("=" * 70)

    test_vps = viewpoints[:NUM_TIMED_VPS]

    # CPU KDTree
    times_cpu = []
    for vp, d in test_vps:
        t0 = get_time()
        idx = query_cpu.points_in_frustum_with_kdtree(vp, d)
        times_cpu.append(get_time() - t0)
    print(f"  CPU KDTree:        {np.mean(times_cpu)*1000:.2f} ± {np.std(times_cpu)*1000:.2f} ms")

    # GPU brute-force
    for vp, d in test_vps[:3]:
        query_gpu_bf.points_in_frustum_bruteforce_gpu(vp, d)
    times_gpu_bf = []
    for vp, d in test_vps:
        t0 = get_time()
        idx = query_gpu_bf.points_in_frustum_bruteforce_gpu(vp, d)
        times_gpu_bf.append(get_time() - t0)
    print(f"  GPU Brute-force:   {np.mean(times_gpu_bf)*1000:.2f} ± {np.std(times_gpu_bf)*1000:.2f} ms")

    # GPU KDTree hybrid
    for vp, d in test_vps[:3]:
        query_gpu_kd.points_in_frustum_kdtree_gpu(vp, d)
    times_gpu_kd = []
    for vp, d in test_vps:
        t0 = get_time()
        idx = query_gpu_kd.points_in_frustum_kdtree_gpu(vp, d)
        times_gpu_kd.append(get_time() - t0)
    print(f"  GPU KDTree+GPU:    {np.mean(times_gpu_kd)*1000:.2f} ± {np.std(times_gpu_kd)*1000:.2f} ms")

    speedup_bf = np.mean(times_cpu) / np.mean(times_gpu_bf)
    speedup_kd = np.mean(times_cpu) / np.mean(times_gpu_kd)
    print(f"  Speedup (brute-force): {speedup_bf:.1f}x")
    print(f"  Speedup (KDTree+GPU):  {speedup_kd:.1f}x")


def benchmark_visibility_methods(queries, viewpoints):
    """Benchmark all visibility methods."""
    print("\n" + "=" * 70)
    print("VISIBILITY COMPUTATION BENCHMARK")
    print("=" * 70)

    results = {}
    for label, query in queries:
        avg_t, std_t = benchmark_single_visibility(query, viewpoints, label)
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


def benchmark_optimization(optimizer_configs, candidates):
    """Benchmark optimizers given as (label, optimizer) pairs."""
    print("\n" + "=" * 70)
    print("GREEDY OPTIMIZER BENCHMARK")
    print("=" * 70)

    subset = candidates[:min(500, len(candidates))]

    for label, optimizer in optimizer_configs:
        print(f"\n  --- {label} ---")

        t0 = get_time()
        result = optimizer.optimize(
            candidates=subset,
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

    # Generate candidate viewpoints
    sampler = ViewpointSampler(mesh, target_points, normals, frustum_params.far)
    candidates = sampler.sample_outside_mesh(
        num_candidates=NUM_CANDIDATE_VPs,
        offset_scale=0.95,
        pos_noise_std=0.05,
        dir_noise_std=0.05,
    )

    # =====================================================================
    # Initialize queries
    # =====================================================================
    print("\n[INIT] Setting up visibility queries...")

    # CPU
    t0 = get_time()
    q_raycast_cpu = RaycastingVisibilityQuery(mesh, target_points, normals, frustum_params)
    print(f"  CPU Raycast init: {get_time()-t0:.2f}s")

    t0 = get_time()
    q_epsilon_cpu = EpsilonVisibilityQuery(mesh, target_points, normals, frustum_params)
    print(f"  CPU Epsilon init: {get_time()-t0:.2f}s")

    # GPU Epsilon (two frustum strategies) — both compute per-viewpoint epsilon
    t0 = get_time()
    q_epsilon_gpu_bf = EpsilonVisibilityQueryCuda(
        mesh, target_points, normals, frustum_params,
        frustum_method="bruteforce"
    )
    print(f"  GPU Epsilon (bruteforce) init: {get_time()-t0:.2f}s")

    t0 = get_time()
    q_epsilon_gpu_kd = EpsilonVisibilityQueryCuda(
        mesh, target_points, normals, frustum_params,
        frustum_method="kdtree"
    )
    print(f"  GPU Epsilon (kdtree) init: {get_time()-t0:.2f}s")

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
    benchmark_frustum_culling(q_epsilon_cpu, q_epsilon_gpu_bf, q_epsilon_gpu_kd, candidates)

    # 2. Visibility computation
    queries = [
        ("CPU Epsilon", q_epsilon_cpu),
        ("GPU Epsilon (bruteforce)", q_epsilon_gpu_bf),
        ("GPU Epsilon (kdtree)", q_epsilon_gpu_kd),
        ("CPU Raycast", q_raycast_cpu),
    ]
    if q_raycast_gpu is not None:
        queries.append(("GPU Raycast (OptiX)", q_raycast_gpu))

    benchmark_visibility_methods(queries, candidates)

    # 3. Optimization benchmark
    opt_configs = [
        ("CPU Epsilon + CPU Greedy",       GreedyOptimizer(q_epsilon_cpu)),
        ("GPU Epsilon + GPU Greedy",       GreedyOptimizerCuda(q_epsilon_gpu_bf)),
        ("CPU Epsilon + KernelGreedy",     KernelGreedyOptimizer(q_epsilon_cpu)),
        ("GPU Epsilon + KernelGreedy GPU", KernelGreedyOptimizerCuda(q_epsilon_gpu_bf)),
        ("CPU Raycast + CPU Greedy",       GreedyOptimizer(q_raycast_cpu)),
    ]
    if q_raycast_gpu is not None:
        opt_configs.append(("GPU Raycast + GPU Greedy", GreedyOptimizerCuda(q_raycast_gpu)))
    benchmark_optimization(opt_configs, candidates)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
