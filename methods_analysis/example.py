import numpy as np
import open3d as o3d
import os
from time import time as get_time
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

# --- Import all modules from the created library ---
from utils import * # type: ignore
from visualization import Visualizer # type: ignore
from sampling import ViewpointSampler # type: ignore
from raycast_visibility import RaycastingVisibilityQuery # type: ignore
from greedy_optimizer_with_kernel import GreedyOptimizer # type: ignore
from epsilon_visibility import EpsilonVisibilityQuery # type: ignore


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
# --- Configuration for TOSCA dataset download and selection ---
TOSCA_DATA_DIR = "TOSCA_dataset"
# Select a single mesh sample from the dataset
MESH_CATEGORY = "Cat" 
MESH_NAME = "cat0" 
# ==============================================================

# Experiment Parameters
NUM_TARGET_POINTS = 200000
NUM_CANDIDATE_VPs = 1800
TARGET_COVERAGES = [0.85, 0.90, 0.925, 0.95, 0.97, 0.98]
MAX_VIEWPOINTS = 1000


# ===========================================================================
# 2. ANALYSIS & GRAPHING FUNCTIONS
# ===========================================================================

def check_epsilon_solution_with_raycast(raycast_query: RaycastingVisibilityQuery, epsilon_result: OptimizationResult) -> float:
    """
    Computes the 'actual' coverage of a set of viewpoints (from Epsilon query)
    using the ground-truth Raycasting query.
    """
    target_points = raycast_query.target_points
    num_target_points = len(target_points)
    total_visible_mask = np.zeros(num_target_points, dtype=bool)
    
    # Accumulate coverage from Epsilon's selected VPs using Raycast logic
    for vp_result in epsilon_result.viewpoints:
        # Use the Raycasting query object to compute visibility for the selected viewpoint
        visible_indices_rc, _ = raycast_query.compute_visibility(
            viewpoint=vp_result.position, 
            direction=vp_result.direction
        )
        total_visible_mask[visible_indices_rc] = True
        
    actual_coverage = np.sum(total_visible_mask) / num_target_points
    return actual_coverage

def generate_comparison_graphs(data: Dict[float, Dict[str, Dict[str, Any]]], mesh_name: str, output_dir: str = "comparison_results"):
    """
    Generates and saves Multi-Group Bar Graphs comparing metrics across TARGET_COVERAGES.
    
    Args:
        data: Nested dictionary: {target_coverage: {method: {metric: value}}}
    """
    print(f"\n[GRAPHS] Generating bar graphs and saving to '{output_dir}/'...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_coverages = sorted(data.keys())
    methods = ["Raycast Visibility", "Epsilon Visibility"]
    
    # Set Matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 
                         'axes.titlesize': 14, 
                         'axes.labelsize': 12})
    
    # Helper to extract data lists for plotting
    def get_metric_list(metric_name):
        res = {m: [] for m in methods}
        for tc in target_coverages:
            for m in methods:
                if m in data[tc]:
                    res[m].append(data[tc][m][metric_name])
                else:
                    res[m].append(0)
        return res

    # Helper function to create a grouped bar chart
    def create_grouped_bar_chart(metric_data, title, ylabel, filename, is_stacked_time=False, second_metric_data=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(target_coverages))  # Label locations
        width = 0.35  # Width of the bars
        
        # Colors
        colors = {'Raycast Visibility': 'tab:blue', 'Epsilon Visibility': 'tab:red'}
        
        if not is_stacked_time:
            # Standard Grouped Bar
            rects1 = ax.bar(x - width/2, metric_data[methods[0]], width, label=methods[0], color=colors[methods[0]])
            rects2 = ax.bar(x + width/2, metric_data[methods[1]], width, label=methods[1], color=colors[methods[1]])
            
            # Add labels on top of bars
            ax.bar_label(rects1, padding=3, fmt='%.1f')
            ax.bar_label(rects2, padding=3, fmt='%.1f')
            
        else:
            # Stacked Bar (for Time: Optimization + Visibility)
            # metric_data is Optimization Time, second_metric_data is Visibility Time
            # We shift them like grouped bars, but stack the two time components
            
            # Raycast Bar
            p1 = ax.bar(x - width/2, metric_data[methods[0]], width, label=f'{methods[0]} (Opt)', color=colors[methods[0]], alpha=0.6)
            p2 = ax.bar(x - width/2, second_metric_data[methods[0]], width, bottom=metric_data[methods[0]], label=f'{methods[0]} (Vis)', color=colors[methods[0]])
            
            # Epsilon Bar
            p3 = ax.bar(x + width/2, metric_data[methods[1]], width, label=f'{methods[1]} (Opt)', color=colors[methods[1]], alpha=0.6)
            p4 = ax.bar(x + width/2, second_metric_data[methods[1]], width, bottom=metric_data[methods[1]], label=f'{methods[1]} (Vis)', color=colors[methods[1]])
            
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Target Coverage (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{tc*100:.1f}%" for tc in target_coverages])
        ax.legend(loc='best')
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close(fig)
        print(f"  - {filename} saved.")

    # --- 1. Viewpoint Count Bar Chart ---
    vp_counts = get_metric_list("Num_Viewpoints")
    create_grouped_bar_chart(vp_counts, 
                             f'Viewpoints Selected vs. Target Coverage ({mesh_name})', 
                             'Number of Viewpoints', 
                             f"{mesh_name}_Viewpoint_Count_Bar.png")

    # --- 2. Computation Time Charts ---
    opt_times = get_metric_list("Optimization_Time")
    vis_times = get_metric_list("Visibility_Time")

    # 2a. Stacked (Total)
    create_grouped_bar_chart(opt_times, 
                             f'Total Time Breakdown ({mesh_name})', 
                             'Time (seconds)', 
                             f"{mesh_name}_Time_Stacked_Bar.png",
                             is_stacked_time=True,
                             second_metric_data=vis_times)
    
    # 2b. Visibility Time Only
    create_grouped_bar_chart(vis_times, 
                             f'Visibility Computation Time ({mesh_name})', 
                             'Time (seconds)', 
                             f"{mesh_name}_Visibility_Time_Bar.png")

    # 2c. Optimization Time Only
    create_grouped_bar_chart(opt_times, 
                             f'Optimization Time ({mesh_name})', 
                             'Time (seconds)', 
                             f"{mesh_name}_Optimization_Time_Bar.png")

    # --- 3. Actual Coverage Bar Chart ---
    # Convert coverage to percentage for display
    act_coverage = get_metric_list("Actual_Coverage")
    for m in methods:
        act_coverage[m] = [val * 100 for val in act_coverage[m]]
        
    create_grouped_bar_chart(act_coverage, 
                             f'Actual Coverage Achieved ({mesh_name})', 
                             'Actual Coverage (%)', 
                             f"{mesh_name}_Coverage_Bar.png")
    
    # --- 4. Redundancy Bar Chart ---
    redundancy = get_metric_list("Redundancy")
    create_grouped_bar_chart(redundancy, 
                             f'Coverage Redundancy ({mesh_name})', 
                             'Avg Viewpoints per Point', 
                             f"{mesh_name}_Redundancy_Bar.png")

    print(f"[GRAPHS] All bar graphs saved to {os.path.abspath(output_dir)}")


# ===========================================================================
# 3. MAIN COMPARISON PIPELINE
# ===========================================================================

def run_comparison_pipeline():
    """
    Performs the comparison between Epsilon and Raycast visibility methods 
    across all TARGET_COVERAGES.
    """
    
    print("="*80)
    print("STARTING VIEWPOINT COVERAGE COMPARISON PIPELINE (Multi-Coverage)")
    print("="*80)
    
    # Create output directory for visualization snapshots
    viz_output_dir = "solution_snapshots"
    if not os.path.exists(viz_output_dir):
        os.makedirs(viz_output_dir)
    
    # 2. Setup Data and Environment (using the downloaded mesh)
    print(f"\n[SETUP] Loading mesh and sampling target points ({NUM_TARGET_POINTS} points)")
    
    # Ensure correct mesh path is used
    mesh, target_points, normals, frustum_params, candidates = create_mock_data(
        NUM_TARGET_POINTS,
        NUM_CANDIDATE_VPs,
        "models/duke_of_lancaster_uk.glb"
    )
        
    visualizer = Visualizer(mesh, target_points, normals, frustum_params)
    sampler = ViewpointSampler(mesh, target_points, normals, frustum_params.far)
    
    # Candidate viewpoints are generated once and used for both methods
    print(f"[SAMPLING] Generating {NUM_CANDIDATE_VPs} candidate viewpoints...")
    candidates = sampler.sample_outside_mesh(
        num_candidates=NUM_CANDIDATE_VPs, 
        offset_scale=0.95, 
        pos_noise_std=0.1,
        dir_noise_std=0.02,
    )
    
    # Initialize Raycast Query (Ground Truth)
    start_time_rc_init = get_time()
    visibility_query_raycast = RaycastingVisibilityQuery(
        mesh=mesh, 
        target_points=target_points, 
        normals=normals, 
        frustum_params=frustum_params
    )
    rc_vis_init_time = get_time() - start_time_rc_init 
    
    # Initialize Epsilon Query
    start_time_eps_init = get_time()
    visibility_query_epsilon = EpsilonVisibilityQuery( 
        mesh=mesh, 
        target_points=target_points, 
        normals=normals, 
        frustum_params=frustum_params
    )
    eps_vis_init_time = get_time() - start_time_eps_init
    print(f"  Estimated Epsilon for mesh: {visibility_query_epsilon.epsilon:.4f} radians")
    
    # Container for all results: {target_coverage: {method: {metric: value}}}
    comparison_data: Dict[float, Dict[str, Dict[str, Any]]] = {}

    # --- 3. Run Comparison for all Target Coverages ---
    for TARGET_COVERAGE in TARGET_COVERAGES:
        print("\n" + "="*80)
        print(f"RUNNING COMPARISON FOR TARGET COVERAGE: {TARGET_COVERAGE*100:.1f}%")
        print("="*80)
        
        comparison_data[TARGET_COVERAGE] = {}
        
        # --- 3a. Raycasting Visibility (Ground Truth) ---
        print("\n--- Raycast Visibility ---")
        optimizer_raycast = GreedyOptimizer(visibility_query_raycast)
        optimization_result_raycast: OptimizationResult = optimizer_raycast.optimize(
            candidates=candidates,
            target_coverage=TARGET_COVERAGE,
            max_viewpoints=MAX_VIEWPOINTS
        )
        
        # SAVE VISUALIZATION ANIMATION
        snap_name = os.path.join(viz_output_dir, f"Raycast_{int(TARGET_COVERAGE*100)}cov.gif")
        visualizer.save_solution_animation(optimization_result_raycast, snap_name, frames=210)

        # Store Results
        rc_vis_time = rc_vis_init_time + optimization_result_raycast.visibility_computation_time 
        
        comparison_data[TARGET_COVERAGE]["Raycast Visibility"] = {
            "Total_Time": optimization_result_raycast.total_time,
            "Optimization_Time": optimization_result_raycast.total_time - optimization_result_raycast.visibility_computation_time,
            "Visibility_Time": optimization_result_raycast.visibility_computation_time,
            "Num_Viewpoints": optimization_result_raycast.num_viewpoints,
            "Reported_Coverage": optimization_result_raycast.total_coverage,
            "Actual_Coverage": optimization_result_raycast.total_coverage, # Raycast is ground truth
            "Redundancy": optimization_result_raycast.redundancy,
        }
        
        # --- 3b. Epsilon Visibility ---
        print("\n--- Epsilon Visibility (with Kernel Expansion) ---")

        # The EpsilonQuery
        optimizer_epsilon = GreedyOptimizer(visibility_query_epsilon)

        optimization_result_epsilon: OptimizationResult = optimizer_epsilon.optimize(
            candidates=candidates,
            target_coverage=TARGET_COVERAGE,
            max_viewpoints=MAX_VIEWPOINTS
        )

        # SAVE VISUALIZATION ANIMATION
        snap_name = os.path.join(viz_output_dir, f"Epsilon_{int(TARGET_COVERAGE*100)}cov.gif")
        visualizer.save_solution_animation(optimization_result_epsilon, snap_name, frames=210)

        # Check Actual Coverage of Epsilon Solution (THE KEY STEP)
        actual_coverage_eps = check_epsilon_solution_with_raycast(
            raycast_query=visibility_query_raycast, 
            epsilon_result=optimization_result_epsilon
        )

        # Store Results
        eps_vis_time = eps_vis_init_time + optimization_result_epsilon.visibility_computation_time
        
        comparison_data[TARGET_COVERAGE]["Epsilon Visibility"] = {
            "Total_Time": optimization_result_epsilon.total_time,
            "Optimization_Time": optimization_result_epsilon.total_time - optimization_result_epsilon.visibility_computation_time,
            "Visibility_Time": optimization_result_epsilon.visibility_computation_time,
            "Num_Viewpoints": optimization_result_epsilon.num_viewpoints,
            "Reported_Coverage": optimization_result_epsilon.total_coverage,
            "Actual_Coverage": actual_coverage_eps,
            "Redundancy": optimization_result_epsilon.redundancy,
        }
        
        print(f"\n[SUMMARY for {TARGET_COVERAGE*100:.1f}% Target]")
        print(f"  Raycast: VPs={optimization_result_raycast.num_viewpoints}, Coverage={optimization_result_raycast.total_coverage*100:.2f}%, Time={optimization_result_raycast.total_time:.2f}s")
        print(f"  Epsilon: VPs={optimization_result_epsilon.num_viewpoints}, Reported={optimization_result_epsilon.total_coverage*100:.2f}%, Actual={actual_coverage_eps*100:.2f}%, Time={optimization_result_epsilon.total_time:.2f}s")


    # --- 4. Final Summary and Graphs ---
    print("\n" + "="*80)
    print(f"FULL COMPARISON PIPELINE COMPLETE")
    print("="*80)

    # Generate and save graphs
    generate_comparison_graphs(comparison_data, "Duke Of Lancaster")

# Execute the comparison
if __name__ == "__main__":
    run_comparison_pipeline()