"""
Compare Greedy vs KernelGreedy optimizers using EpsilonVisibilityQuery,
cross-validated with RaycastingVisibilityQuery for ground truth coverage.
"""
import numpy as np
import open3d as o3d
import os
from time import time as get_time
from typing import Dict, Any
import matplotlib.pyplot as plt

from visibility.core.types import FrustumParams, OptimizationResult
from visibility.sampling import UniformViewpointSampler
from visibility.methods.raycast import RaycastingVisibilityQuery
from visibility.methods.epsilon import EpsilonVisibilityQuery
from visibility.optimizers.greedy import GreedyOptimizer
from visibility.optimizers.kernel_greedy import KernelGreedyOptimizer
from visibility.visualization import Visualizer


# ===========================================================================
# CONFIGURATION
# ===========================================================================
NUM_TARGET_POINTS = 200000
NUM_CANDIDATE_VPs = 2000
TARGET_COVERAGES = [0.85, 0.90, 0.925, 0.95, 0.97, 0.98]
MAX_VIEWPOINTS = 1000


# ===========================================================================
# ANALYSIS FUNCTIONS
# ===========================================================================

def check_solution_with_raycast(raycast_query: RaycastingVisibilityQuery,
                                result: OptimizationResult) -> float:
    """Computes actual coverage using ground-truth Raycasting."""
    num_target_points = len(raycast_query.target_points)
    total_visible_mask = np.zeros(num_target_points, dtype=bool)

    for vp_result in result.viewpoints:
        visible_indices_rc, _ = raycast_query.compute_visibility(
            viewpoint=vp_result.position,
            orientation=vp_result.orientation
        )
        total_visible_mask[visible_indices_rc] = True

    return np.sum(total_visible_mask) / num_target_points


def generate_comparison_graphs(data: Dict[float, Dict[str, Dict[str, Any]]],
                               mesh_name: str, output_dir: str = "optimizer_comparison_results"):
    """Generates bar graphs comparing Greedy vs KernelGreedy optimizers."""
    print(f"\n[GRAPHS] Generating bar graphs and saving to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)

    target_coverages = sorted(data.keys())
    methods = ["Greedy", "KernelGreedy"]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

    def get_metric_list(metric_name):
        res = {m: [] for m in methods}
        for tc in target_coverages:
            for m in methods:
                if m in data[tc]:
                    res[m].append(data[tc][m][metric_name])
                else:
                    res[m].append(0)
        return res

    def create_grouped_bar_chart(metric_data, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(target_coverages))
        width = 0.35

        colors = {'Greedy': 'tab:blue', 'KernelGreedy': 'tab:orange'}

        rects1 = ax.bar(x - width / 2, metric_data[methods[0]], width,
                        label=methods[0], color=colors[methods[0]])
        rects2 = ax.bar(x + width / 2, metric_data[methods[1]], width,
                        label=methods[1], color=colors[methods[1]])
        ax.bar_label(rects1, padding=3, fmt='%.1f')
        ax.bar_label(rects2, padding=3, fmt='%.1f')

        ax.set_ylabel(ylabel)
        ax.set_xlabel('Target Coverage (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{tc * 100:.1f}%" for tc in target_coverages])
        ax.legend(loc='best')

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close(fig)
        print(f"  - {filename} saved.")

    vp_counts = get_metric_list("Num_Viewpoints")
    create_grouped_bar_chart(vp_counts,
                             f'Viewpoints Selected ({mesh_name})',
                             'Number of Viewpoints',
                             f"{mesh_name}_Optimizer_VP_Count.png")

    total_times = get_metric_list("Total_Time")
    create_grouped_bar_chart(total_times,
                             f'Total Time ({mesh_name})',
                             'Time (seconds)',
                             f"{mesh_name}_Optimizer_Total_Time.png")

    act_coverage = get_metric_list("Actual_Coverage")
    for m in methods:
        act_coverage[m] = [val * 100 for val in act_coverage[m]]

    create_grouped_bar_chart(act_coverage,
                             f'Actual Coverage (Raycast-verified) ({mesh_name})',
                             'Actual Coverage (%)',
                             f"{mesh_name}_Optimizer_Coverage.png")

    redundancy = get_metric_list("Redundancy")
    create_grouped_bar_chart(redundancy,
                             f'Coverage Redundancy ({mesh_name})',
                             'Avg Viewpoints per Point',
                             f"{mesh_name}_Optimizer_Redundancy.png")

    print(f"[GRAPHS] All graphs saved to {os.path.abspath(output_dir)}")


# ===========================================================================
# MAIN COMPARISON PIPELINE
# ===========================================================================

def create_mock_data(num_points=1000, num_candidates=100, mesh_path=None):
    """Creates a basic mesh, target points, and candidates for testing."""
    if mesh_path is not None:
        print(f"Loading mesh from: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    else:
        print("Using default mesh (Sphere)")
        mesh = o3d.geometry.TriangleMesh.create_sphere()
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    target_points = np.asarray(pcd.points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)

    frustum_params = FrustumParams(fov_y=np.deg2rad(45), aspect=1.0, near=0.01, far=7)

    from shared.geometry import direction_roll_to_quaternion
    candidate_pos = target_points[:num_candidates] + normals[:num_candidates] * 1.5
    candidate_dir = -normals[:num_candidates]
    candidates = [(pos, direction_roll_to_quaternion(d)) for pos, d in zip(candidate_pos, candidate_dir)]

    return mesh, target_points, normals, frustum_params, candidates


def run_comparison_pipeline():
    """
    Compares Greedy vs KernelGreedy optimizers using EpsilonVisibilityQuery,
    cross-validated with RaycastingVisibilityQuery.
    """
    print("=" * 80)
    print("STARTING OPTIMIZER COMPARISON PIPELINE (Greedy vs KernelGreedy)")
    print("=" * 80)

    viz_output_dir = "optimizer_snapshots"
    os.makedirs(viz_output_dir, exist_ok=True)

    print(f"\n[SETUP] Loading mesh and sampling target points ({NUM_TARGET_POINTS} points)")

    mesh, target_points, normals, frustum_params, candidates = create_mock_data(
        NUM_TARGET_POINTS,
        NUM_CANDIDATE_VPs,
    )

    visualizer = Visualizer(mesh, target_points, normals, frustum_params)
    sampler = UniformViewpointSampler(mesh, target_points, normals, frustum_params.far, collision_radius=0.5)

    print(f"[SAMPLING] Generating {NUM_CANDIDATE_VPs} candidate viewpoints...")
    candidates = sampler.sample(num_candidates=NUM_CANDIDATE_VPs, side="outside")

    # Initialize Epsilon query (used by both optimizers)
    visibility_query_epsilon = EpsilonVisibilityQuery(
        target_points=target_points,
        normals=normals, frustum_params=frustum_params
    )

    # Initialize Raycast query for ground-truth validation
    visibility_query_raycast = RaycastingVisibilityQuery(
        mesh=mesh, target_points=target_points,
        normals=normals, frustum_params=frustum_params
    )

    comparison_data: Dict[float, Dict[str, Dict[str, Any]]] = {}

    for TARGET_COVERAGE in TARGET_COVERAGES:
        print("\n" + "=" * 80)
        print(f"RUNNING COMPARISON FOR TARGET COVERAGE: {TARGET_COVERAGE * 100:.1f}%")
        print("=" * 80)

        comparison_data[TARGET_COVERAGE] = {}

        # Greedy Optimizer (standard)
        print("\n--- Greedy Optimizer ---")
        optimizer_greedy = GreedyOptimizer(visibility_query_epsilon)
        result_greedy = optimizer_greedy.optimize(
            candidates=candidates,
            target_coverage=TARGET_COVERAGE,
            max_viewpoints=MAX_VIEWPOINTS
        )

        actual_coverage_greedy = check_solution_with_raycast(
            visibility_query_raycast, result_greedy
        )

        comparison_data[TARGET_COVERAGE]["Greedy"] = {
            "Total_Time": result_greedy.total_time,
            "Num_Viewpoints": result_greedy.num_viewpoints,
            "Reported_Coverage": result_greedy.total_coverage,
            "Actual_Coverage": actual_coverage_greedy,
            "Redundancy": result_greedy.redundancy,
        }

        # KernelGreedy Optimizer
        print("\n--- KernelGreedy Optimizer ---")
        optimizer_kernel = KernelGreedyOptimizer(visibility_query_epsilon)
        result_kernel = optimizer_kernel.optimize(
            candidates=candidates,
            target_coverage=TARGET_COVERAGE,
            max_viewpoints=MAX_VIEWPOINTS
        )

        actual_coverage_kernel = check_solution_with_raycast(
            visibility_query_raycast, result_kernel
        )

        comparison_data[TARGET_COVERAGE]["KernelGreedy"] = {
            "Total_Time": result_kernel.total_time,
            "Num_Viewpoints": result_kernel.num_viewpoints,
            "Reported_Coverage": result_kernel.total_coverage,
            "Actual_Coverage": actual_coverage_kernel,
            "Redundancy": result_kernel.redundancy,
        }

        print(f"\n[SUMMARY for {TARGET_COVERAGE * 100:.1f}% Target]")
        print(f"  Greedy:       VPs={result_greedy.num_viewpoints}, "
              f"Actual={actual_coverage_greedy * 100:.2f}%, "
              f"Time={result_greedy.total_time:.2f}s")
        print(f"  KernelGreedy: VPs={result_kernel.num_viewpoints}, "
              f"Actual={actual_coverage_kernel * 100:.2f}%, "
              f"Time={result_kernel.total_time:.2f}s")

    print("\n" + "=" * 80)
    print("OPTIMIZER COMPARISON PIPELINE COMPLETE")
    print("=" * 80)

    generate_comparison_graphs(comparison_data, "Duke_Of_Lancaster")


if __name__ == "__main__":
    run_comparison_pipeline()
