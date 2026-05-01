"""
Compare Raycast vs Epsilon visibility methods using the same GreedyOptimizer
across multiple coverage targets.
"""

import os
import time
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
from visualization import SetCoverVisualizer

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


def check_epsilon_solution_with_raycast(
    raycast_query: RaycastingVisibilityQuery, epsilon_result: OptimizationResult
) -> float:
    """
    Computes the 'actual' coverage of a set of viewpoints (from Epsilon query)
    using the ground-truth Raycasting query.
    """
    target_points = raycast_query.target_points
    num_target_points = len(target_points)
    total_visible_mask = np.zeros(num_target_points, dtype=bool)

    positions_np = epsilon_result.positions.get()
    rotations_np = epsilon_result.rotations.get()
    for i in range(epsilon_result.num_viewpoints):
        visible_indices_rc, _ = raycast_query.compute_visibility(
            viewpoint=positions_np[i], rotation=rotations_np[i]
        )
        total_visible_mask[visible_indices_rc] = True

    actual_coverage = np.sum(total_visible_mask) / num_target_points
    return actual_coverage


def generate_comparison_graphs(
    data: dict[float, dict[str, dict[str, Any]]],
    mesh_name: str,
    output_dir: str = "comparison_results",
):
    """Generates and saves bar graphs comparing metrics across target coverages."""
    print(f"\n[GRAPHS] Generating bar graphs and saving to '{output_dir}/'...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_coverages = sorted(data.keys())
    methods = ["Raycast Visibility", "Epsilon Visibility"]

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

    def create_grouped_bar_chart(
        metric_data, title, ylabel, filename, is_stacked_time=False, second_metric_data=None
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(target_coverages))
        width = 0.35

        colors = {"Raycast Visibility": "tab:blue", "Epsilon Visibility": "tab:red"}

        if not is_stacked_time:
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
        else:
            ax.bar(
                x - width / 2,
                metric_data[methods[0]],
                width,
                label=f"{methods[0]} (Opt)",
                color=colors[methods[0]],
                alpha=0.6,
            )
            ax.bar(
                x - width / 2,
                second_metric_data[methods[0]],
                width,
                bottom=metric_data[methods[0]],
                label=f"{methods[0]} (Vis)",
                color=colors[methods[0]],
            )
            ax.bar(
                x + width / 2,
                metric_data[methods[1]],
                width,
                label=f"{methods[1]} (Opt)",
                color=colors[methods[1]],
                alpha=0.6,
            )
            ax.bar(
                x + width / 2,
                second_metric_data[methods[1]],
                width,
                bottom=metric_data[methods[1]],
                label=f"{methods[1]} (Vis)",
                color=colors[methods[1]],
            )

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
        f"Viewpoints Selected vs. Target Coverage ({mesh_name})",
        "Number of Viewpoints",
        f"{mesh_name}_Viewpoint_Count_Bar.png",
    )

    opt_times = get_metric_list("Optimization_Time")
    vis_times = get_metric_list("Visibility_Time")

    create_grouped_bar_chart(
        opt_times,
        f"Total Time Breakdown ({mesh_name})",
        "Time (seconds)",
        f"{mesh_name}_Time_Stacked_Bar.png",
        is_stacked_time=True,
        second_metric_data=vis_times,
    )

    create_grouped_bar_chart(
        vis_times,
        f"Visibility Computation Time ({mesh_name})",
        "Time (seconds)",
        f"{mesh_name}_Visibility_Time_Bar.png",
    )

    create_grouped_bar_chart(
        opt_times,
        f"Optimization Time ({mesh_name})",
        "Time (seconds)",
        f"{mesh_name}_Optimization_Time_Bar.png",
    )

    act_coverage = get_metric_list("Actual_Coverage")
    for m in methods:
        act_coverage[m] = [val * 100 for val in act_coverage[m]]

    create_grouped_bar_chart(
        act_coverage,
        f"Actual Coverage Achieved ({mesh_name})",
        "Actual Coverage (%)",
        f"{mesh_name}_Coverage_Bar.png",
    )

    redundancy = get_metric_list("Redundancy")
    create_grouped_bar_chart(
        redundancy,
        f"Coverage Redundancy ({mesh_name})",
        "Avg Viewpoints per Point",
        f"{mesh_name}_Redundancy_Bar.png",
    )

    print(f"[GRAPHS] All bar graphs saved to {os.path.abspath(output_dir)}")


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
    Compares Epsilon and Raycast visibility methods across all TARGET_COVERAGES.
    """
    print("=" * 80)
    print("STARTING VIEWPOINT COVERAGE COMPARISON PIPELINE (Multi-Coverage)")
    print("=" * 80)

    viz_output_dir = "solution_snapshots"
    if not os.path.exists(viz_output_dir):
        os.makedirs(viz_output_dir)

    print(f"\n[SETUP] Loading mesh and sampling target points ({NUM_TARGET_POINTS} points)")

    mesh, target_points, normals, frustum_params, positions, rotmats = create_mock_data(
        NUM_TARGET_POINTS,
        NUM_CANDIDATE_VPs,
    )

    set_cover_viz = SetCoverVisualizer(mesh, target_points, frustum_params)
    sampler = WeightedViewpointSampler(
        mesh, target_points, normals, frustum_params.far, collision_radius=0.5
    )

    print(f"[SAMPLING] Generating {NUM_CANDIDATE_VPs} candidate viewpoints...")
    pos_gpu, rot_gpu = sampler.sample(num_candidates=NUM_CANDIDATE_VPs, side=Side.OUTSIDE)
    positions = cp.asnumpy(pos_gpu)
    rotmats = cp.asnumpy(rot_gpu)

    start_time_rc_init = time.perf_counter()
    visibility_query_raycast = RaycastingVisibilityQuery(
        mesh=mesh, target_points=target_points, normals=normals, frustum_params=frustum_params
    )
    time.perf_counter() - start_time_rc_init

    start_time_eps_init = time.perf_counter()
    visibility_query_epsilon = EpsilonVisibilityQuery(
        target_points=target_points, normals=normals, frustum_params=frustum_params
    )
    time.perf_counter() - start_time_eps_init
    if visibility_query_epsilon.fixed_epsilon is not None:
        print(f"  Fixed Epsilon: {visibility_query_epsilon.fixed_epsilon:.4f} radians")
    else:
        print(
            f"  Estimated δ (sampling density): {visibility_query_epsilon.delta:.4f} (ε computed per viewpoint)"
        )

    comparison_data: dict[float, dict[str, dict[str, Any]]] = {}

    for TARGET_COVERAGE in TARGET_COVERAGES:
        print("\n" + "=" * 80)
        print(f"RUNNING COMPARISON FOR TARGET COVERAGE: {TARGET_COVERAGE * 100:.1f}%")
        print("=" * 80)

        comparison_data[TARGET_COVERAGE] = {}

        # Raycasting Visibility (Ground Truth)
        print("\n--- Raycast Visibility ---")
        V_rc, _ = visibility_query_raycast.compute_visibility_batch(positions, rotmats)
        optimizer_raycast = GreedySetCover(len(target_points), positions, rotmats, V_rc)
        optimization_result_raycast = optimizer_raycast.optimize(
            target_coverage=TARGET_COVERAGE, max_viewpoints=MAX_VIEWPOINTS
        )

        snap_name = os.path.join(viz_output_dir, f"Raycast_{int(TARGET_COVERAGE * 100)}cov.gif")
        set_cover_viz.save_animation(optimization_result_raycast, snap_name, frames=210)

        comparison_data[TARGET_COVERAGE]["Raycast Visibility"] = {
            "Optimization_Time": optimization_result_raycast.optimization_time,
            "Num_Viewpoints": optimization_result_raycast.num_viewpoints,
            "Reported_Coverage": optimization_result_raycast.total_coverage,
            "Actual_Coverage": optimization_result_raycast.total_coverage,
            "Redundancy": optimization_result_raycast.redundancy,
        }

        # Epsilon Visibility
        print("\n--- Epsilon Visibility ---")
        V_eps, _ = visibility_query_epsilon.compute_visibility_batch(positions, rotmats)
        optimizer_epsilon = GreedySetCover(len(target_points), positions, rotmats, V_eps)
        optimization_result_epsilon = optimizer_epsilon.optimize(
            target_coverage=TARGET_COVERAGE, max_viewpoints=MAX_VIEWPOINTS
        )

        snap_name = os.path.join(viz_output_dir, f"Epsilon_{int(TARGET_COVERAGE * 100)}cov.gif")
        set_cover_viz.save_animation(optimization_result_epsilon, snap_name, frames=210)

        actual_coverage_eps = check_epsilon_solution_with_raycast(
            raycast_query=visibility_query_raycast, epsilon_result=optimization_result_epsilon
        )

        comparison_data[TARGET_COVERAGE]["Epsilon Visibility"] = {
            "Optimization_Time": optimization_result_epsilon.optimization_time,
            "Num_Viewpoints": optimization_result_epsilon.num_viewpoints,
            "Reported_Coverage": optimization_result_epsilon.total_coverage,
            "Actual_Coverage": actual_coverage_eps,
            "Redundancy": optimization_result_epsilon.redundancy,
        }

        print(f"\n[SUMMARY for {TARGET_COVERAGE * 100:.1f}% Target]")
        print(
            f"  Raycast: VPs={optimization_result_raycast.num_viewpoints}, "
            f"Coverage={optimization_result_raycast.total_coverage * 100:.2f}%, "
            f"Time={optimization_result_raycast.optimization_time:.2f}s"
        )
        print(
            f"  Epsilon: VPs={optimization_result_epsilon.num_viewpoints}, "
            f"Reported={optimization_result_epsilon.total_coverage * 100:.2f}%, "
            f"Actual={actual_coverage_eps * 100:.2f}%, "
            f"Time={optimization_result_epsilon.optimization_time:.2f}s"
        )

    print("\n" + "=" * 80)
    print("FULL COMPARISON PIPELINE COMPLETE")
    print("=" * 80)

    generate_comparison_graphs(comparison_data, "Duke Of Lancaster")


if __name__ == "__main__":
    run_comparison_pipeline()
