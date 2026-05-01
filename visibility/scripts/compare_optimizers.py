"""
Compare Greedy vs KernelGreedy optimizers using EpsilonVisibilityQuery,
cross-validated with RaycastingVisibilityQuery for ground truth coverage.
"""

import os
from typing import Any

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from shared.surface_sampler import SurfacePointSampler
from shared.types import Side
from visibility.core.types import FrustumParams, OptimizationResult
from visibility.sampling import WeightedViewpointSampler
from visibility.set_cover import GreedySetCover
from visibility.visibility.epsilon import EpsilonVisibilityQuery
from visibility.visibility.raycast import RaycastingVisibilityQuery

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


def check_solution_with_raycast(
    raycast_query: RaycastingVisibilityQuery, result: OptimizationResult
) -> float:
    """Computes actual coverage using ground-truth Raycasting."""
    num_target_points = len(raycast_query.target_points)
    total_visible_mask = np.zeros(num_target_points, dtype=bool)

    positions_np = result.positions.get()
    rotations_np = result.rotations.get()
    for i in range(result.num_viewpoints):
        visible_indices_rc, _ = raycast_query.compute_visibility(
            viewpoint=positions_np[i], rotation=rotations_np[i]
        )
        total_visible_mask[visible_indices_rc] = True

    return np.sum(total_visible_mask) / num_target_points


def generate_comparison_graphs(
    data: dict[float, dict[str, dict[str, Any]]],
    mesh_name: str,
    output_dir: str = "optimizer_comparison_results",
):
    """Generates bar graphs comparing Greedy vs KernelGreedy optimizers."""
    print(f"\n[GRAPHS] Generating bar graphs and saving to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)

    target_coverages = sorted(data.keys())
    methods = ["Greedy", "KernelGreedy"]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})

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

        colors = {"Greedy": "tab:blue", "KernelGreedy": "tab:orange"}

        rects1 = ax.bar(
            x - width / 2,
            metric_data[methods[0]],
            width,
            label=methods[0],
            color=colors[methods[0]],
        )
        rects2 = ax.bar(
            x + width / 2,
            metric_data[methods[1]],
            width,
            label=methods[1],
            color=colors[methods[1]],
        )
        ax.bar_label(rects1, padding=3, fmt="%.1f")
        ax.bar_label(rects2, padding=3, fmt="%.1f")

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Target Coverage (%)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{tc * 100:.1f}%" for tc in target_coverages])
        ax.legend(loc="best")

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close(fig)
        print(f"  - {filename} saved.")

    vp_counts = get_metric_list("Num_Viewpoints")
    create_grouped_bar_chart(
        vp_counts,
        f"Viewpoints Selected ({mesh_name})",
        "Number of Viewpoints",
        f"{mesh_name}_Optimizer_VP_Count.png",
    )

    total_times = get_metric_list("Total_Time")
    create_grouped_bar_chart(
        total_times,
        f"Total Time ({mesh_name})",
        "Time (seconds)",
        f"{mesh_name}_Optimizer_Total_Time.png",
    )

    act_coverage = get_metric_list("Actual_Coverage")
    for m in methods:
        act_coverage[m] = [val * 100 for val in act_coverage[m]]

    create_grouped_bar_chart(
        act_coverage,
        f"Actual Coverage (Raycast-verified) ({mesh_name})",
        "Actual Coverage (%)",
        f"{mesh_name}_Optimizer_Coverage.png",
    )

    redundancy = get_metric_list("Redundancy")
    create_grouped_bar_chart(
        redundancy,
        f"Coverage Redundancy ({mesh_name})",
        "Avg Viewpoints per Point",
        f"{mesh_name}_Optimizer_Redundancy.png",
    )

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

    surface_sampler = SurfacePointSampler()
    target_points, normals = surface_sampler.sample(
        mesh,
        num_points,
        normal_radius=0.1,
        tangent_plane_k=10,
    )

    frustum_params = FrustumParams(fov_y=np.deg2rad(45), aspect=1.0, near=0.01, far=7)

    from shared.geometry import direction_roll_to_rotmat

    candidate_pos = target_points[:num_candidates] + normals[:num_candidates] * 1.5
    candidate_dir = -normals[:num_candidates]
    positions = candidate_pos.astype(np.float32)
    rotmats = np.array([direction_roll_to_rotmat(d) for d in candidate_dir])

    return mesh, target_points, normals, frustum_params, positions, rotmats


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

    mesh, target_points, normals, frustum_params, positions, rotmats = create_mock_data(
        NUM_TARGET_POINTS,
        NUM_CANDIDATE_VPs,
    )

    sampler = WeightedViewpointSampler(
        mesh, target_points, normals, frustum_params.far, collision_radius=0.5
    )

    print(f"[SAMPLING] Generating {NUM_CANDIDATE_VPs} candidate viewpoints...")
    pos_gpu, rot_gpu = sampler.sample(num_candidates=NUM_CANDIDATE_VPs, side=Side.OUTSIDE)
    positions = cp.asnumpy(pos_gpu)
    rotmats = cp.asnumpy(rot_gpu)

    # Initialize Epsilon query (used by both optimizers)
    visibility_query_epsilon = EpsilonVisibilityQuery(
        target_points=target_points, normals=normals, frustum_params=frustum_params
    )

    # Initialize Raycast query for ground-truth validation
    visibility_query_raycast = RaycastingVisibilityQuery(
        mesh=mesh, target_points=target_points, normals=normals, frustum_params=frustum_params
    )

    comparison_data: dict[float, dict[str, dict[str, Any]]] = {}

    for TARGET_COVERAGE in TARGET_COVERAGES:
        print("\n" + "=" * 80)
        print(f"RUNNING COMPARISON FOR TARGET COVERAGE: {TARGET_COVERAGE * 100:.1f}%")
        print("=" * 80)

        comparison_data[TARGET_COVERAGE] = {}

        # Greedy Set Cover
        print("\n--- Greedy Set Cover ---")
        V_eps, _ = visibility_query_epsilon.compute_visibility_batch(positions, rotmats)
        optimizer_greedy = GreedySetCover(len(target_points), positions, rotmats, V_eps)
        result_greedy = optimizer_greedy.optimize(
            target_coverage=TARGET_COVERAGE, max_viewpoints=MAX_VIEWPOINTS
        )

        actual_coverage_greedy = check_solution_with_raycast(
            visibility_query_raycast, result_greedy
        )

        comparison_data[TARGET_COVERAGE]["Greedy"] = {
            "Optimization_Time": result_greedy.optimization_time,
            "Num_Viewpoints": result_greedy.num_viewpoints,
            "Reported_Coverage": result_greedy.total_coverage,
            "Actual_Coverage": actual_coverage_greedy,
            "Redundancy": result_greedy.redundancy,
        }

        print(f"\n[SUMMARY for {TARGET_COVERAGE * 100:.1f}% Target]")
        print(
            f"  Greedy: VPs={result_greedy.num_viewpoints}, "
            f"Actual={actual_coverage_greedy * 100:.2f}%, "
            f"Time={result_greedy.optimization_time:.2f}s"
        )

    print("\n" + "=" * 80)
    print("OPTIMIZER COMPARISON PIPELINE COMPLETE")
    print("=" * 80)

    generate_comparison_graphs(comparison_data, "Duke_Of_Lancaster")


if __name__ == "__main__":
    run_comparison_pipeline()
