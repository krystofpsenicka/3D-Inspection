import numpy as np
import open3d as o3d
import os
from time import time as get_time
from typing import List
import matplotlib.pyplot as plt

# --- Import all modules from the created library ---
# Note: Assuming all referenced classes and functions are available in your local file structure.
# Import new download utility
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
NUM_TARGET_POINTS = 100000 
NUM_CANDIDATE_VPs = 1
TARGET_COVERAGE = 0.99 
MAX_VIEWPOINTS = 100


# ===========================================================================
# 2. ANALYSIS & GRAPHING FUNCTIONS
# ===========================================================================

def check_epsilon_solution_with_raycast(raycast_query: RaycastingVisibilityQuery, epsilon_result: OptimizationResult) -> float:
    """
    Computes the 'actual' coverage of a set of viewpoints (from Epsilon query)
    using the ground-truth Raycasting query.
    """
    print("\n[VERIFICATION] Checking Epsilon Solution's Actual Coverage with Raycasting...")
    start_time = get_time()
    
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
    end_time = get_time()
    print(f"  Verification Complete. Actual Coverage: {actual_coverage*100:.2f}%. Time: {end_time - start_time:.2f}s")
    
    return actual_coverage

def generate_comparison_graphs(data: dict, mesh_name: str, output_dir: str = "comparison_results"):
    """
    Generates and saves grouped bar graphs comparing metrics between the two methods.
    """
    print(f"\n[GRAPHS] Generating comparison graphs and saving to '{output_dir}/'...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    methods = data["Method"]
    x = np.arange(len(methods))  # the label locations
    width = 0.35  # the width of the bars
    mesh_title = mesh_name.replace("_", " / ")

    # --- 1. Timing Comparison Graph (Stacked and Combined) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # visibility_times = data["Visibility_Time"]
    # optimization_times = data["Optimization_Time"]
    
    # Plot 1: Stacked Bar for Visibility + Optimization Time (Group 1)
    # rects1 = ax.bar(x - width/2, visibility_times, width, label='Visibility Comp. Time', color='skyblue')
    # rects2 = ax.bar(x - width/2, optimization_times, width, bottom=visibility_times, label='Optimization Time', color='salmon')

    # Plot 2: Single Bar for Total Time (Group 2)
    rects_total = ax.bar(x + width/2, data["Total_Time"], width, label='Total Time (Combined)', color='darkgreen')

    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Computation Time Comparison for Mesh: {mesh_title}')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mesh_name}_Time_Comparison.png"))
    plt.close(fig)
    print(f"  - Time Comparison graph saved.")


    # --- 2. Viewpoint Count and Coverage Comparison Graph ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for Viewpoint Count (Primary Y-axis)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Number of Viewpoints Selected', color='tab:blue')
    ax1.bar(x, data["Num_Viewpoints"], width, label='Num. Viewpoints', color='tab:blue', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Line plot for Coverage (Secondary Y-axis)
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Coverage (%)', color='tab:red')  
    
    # Plot Reported Coverage
    reported_y = np.array(data["Reported_Coverage"]) * 100
    ax2.plot(x, reported_y, color='tab:red', marker='o', linestyle='-', linewidth=2, label='Reported Coverage')

    # Plot Actual Coverage (Raycast check)
    actual_y = np.array(data["Actual_Coverage"]) * 100
    ax2.plot(x, actual_y, color='tab:green', marker='x', linestyle='--', linewidth=2, label='Actual Coverage (Raycast Check)')
    
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # Set Y-limit to focus on the high coverage range
    ax2.set_ylim(min(90, np.min(actual_y) - 2), 100) 
    
    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center')

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    plt.title(f'Viewpoint Count & Coverage Comparison for Mesh: {mesh_title}')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mesh_name}_Coverage_Viewpoints_Comparison.png"))
    plt.close(fig)
    print(f"  - Coverage and Viewpoint Count graph saved.")

    # --- 3. Redundancy Graph ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, data["Redundancy"], width=0.5, color=['skyblue', 'salmon'])
    ax.set_ylabel('Average Coverage Redundancy (views/point)')
    ax.set_title(f'Coverage Redundancy for Mesh: {mesh_title}')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mesh_name}_Redundancy_Comparison.png"))
    plt.close(fig)
    print(f"  - Redundancy graph saved.")
    
    print(f"[GRAPHS] All graphs saved to {os.path.abspath(output_dir)}")


# ===========================================================================
# 3. MAIN COMPARISON PIPELINE
# ===========================================================================

def run_comparison_pipeline():
    """
    Performs the comparison between Epsilon and Raycast visibility methods.
    """
    
    print("="*80)
    print("STARTING VIEWPOINT COVERAGE COMPARISON PIPELINE")
    print("="*80)
    

    # 2. Setup Data and Environment (using the downloaded mesh)
    print(f"\n[SETUP] Loading mesh and sampling target points ({NUM_TARGET_POINTS} points)")
    
    mesh, target_points, normals, frustum_params, _ = create_mock_data(
        num_points=NUM_TARGET_POINTS, 
        mesh_path="models/duke_of_lancaster_uk.glb" # Pass the base path to the vert/tri files
    )

    visualizer = Visualizer(mesh, target_points, normals, frustum_params)
    sampler = ViewpointSampler(mesh, target_points, normals, frustum_params.far)
    
    # Candidate viewpoints are generated once and used for both methods
    print(f"[SAMPLING] Generating {NUM_CANDIDATE_VPs} candidate viewpoints...")
    candidates = sampler.sample_outside_mesh(
        num_candidates=NUM_CANDIDATE_VPs, 
        offset_scale=0.95, 
        pos_noise_std=0.1
    )
    
    # Container for results
    comparison_results = {
        "Method": [],
        # "Visibility_Time": [],
        "Optimization_Time": [],
        "Total_Time": [],
        "Num_Viewpoints": [],
        "Reported_Coverage": [],
        "Actual_Coverage": [], 
        "Redundancy": [],
    }

    # --- 3. Raycasting Visibility (Ground Truth) ---
    print("\n" + "="*80)
    print("RUNNING: RAYCASTING VISIBILITY OPTIMIZATION")
    print("="*80)
    
    # 3a. Initialize Visibility Query
    start_time_rc_init = get_time()
    visibility_query_raycast = RaycastingVisibilityQuery(
        mesh=mesh, 
        target_points=target_points, 
        normals=normals, 
        frustum_params=frustum_params
    )
    # BVH initialization is part of the Visibility setup
    rc_vis_init_time = get_time() - start_time_rc_init 
    
    # 3b. Initialize and Run Optimization
    optimizer_raycast = GreedyOptimizer(visibility_query_raycast)
    optimization_result_raycast: OptimizationResult = optimizer_raycast.optimize(
        candidates=candidates,
        target_coverage=TARGET_COVERAGE,
        max_viewpoints=MAX_VIEWPOINTS
    )

    # 3c. Store Results
    # rc_vis_time = rc_vis_init_time + optimization_result_raycast.visibility_computation_time 
    
    comparison_results["Method"].append("Raycast Visibility")
    # comparison_results["Visibility_Time"].append(rc_vis_time)
    # comparison_results["Optimization_Time"].append(optimization_result_raycast.optimization_time)
    comparison_results["Total_Time"].append(optimization_result_raycast.total_time)
    comparison_results["Num_Viewpoints"].append(optimization_result_raycast.num_viewpoints)
    comparison_results["Reported_Coverage"].append(optimization_result_raycast.total_coverage)
    comparison_results["Actual_Coverage"].append(optimization_result_raycast.total_coverage) # Raycast is ground truth
    comparison_results["Redundancy"].append(optimization_result_raycast.redundancy)
    
    # 3d. Visualization (Raycast Solution)
    print("\n[VISUALIZATION] Launching Raycast solution visualization (Close window to continue)...")
    visualizer.visualize_solution_pcd(
        optimization_result_raycast
    )
    print("NOTE: For permanent files, save the screenshot during the interactive Open3D visualization.")
    
    # --- 4. Epsilon Visibility ---
    print("\n" + "="*80)
    print("RUNNING: EPSILON VISIBILITY OPTIMIZATION")
    print("="*80)

    # 4a. Initialize Visibility Query
    start_time_eps_init = get_time()
    visibility_query_epsilon = EpsilonVisibilityQuery( 
        mesh=mesh, 
        target_points=target_points, 
        normals=normals, 
        frustum_params=frustum_params
    )
    eps_vis_init_time = get_time() - start_time_eps_init
    print(f"  Estimated Epsilon for mesh: {visibility_query_epsilon.epsilon:.4f} radians")


    # 4b. Initialize and Run Optimization
    optimizer_epsilon = GreedyOptimizer(visibility_query_epsilon)
    optimization_result_epsilon: OptimizationResult = optimizer_epsilon.optimize(
        candidates=candidates,
        target_coverage=TARGET_COVERAGE,
        max_viewpoints=MAX_VIEWPOINTS
    )

    # 4c. Check Actual Coverage of Epsilon Solution (THE KEY STEP)
    actual_coverage_eps = check_epsilon_solution_with_raycast(
        raycast_query=visibility_query_raycast, # Reuse the Raycast query object
        epsilon_result=optimization_result_epsilon
    )

    # 4d. Store Results
    # eps_vis_time = eps_vis_init_time + optimization_result_epsilon.visibility_computation_time
    
    comparison_results["Method"].append("Epsilon Visibility")
    # comparison_results["Visibility_Time"].append(eps_vis_time)
    # comparison_results["Optimization_Time"].append(optimization_result_epsilon.optimization_time)
    comparison_results["Total_Time"].append(optimization_result_epsilon.total_time)
    comparison_results["Num_Viewpoints"].append(optimization_result_epsilon.num_viewpoints)
    comparison_results["Reported_Coverage"].append(optimization_result_epsilon.total_coverage)
    comparison_results["Actual_Coverage"].append(actual_coverage_eps)
    comparison_results["Redundancy"].append(optimization_result_epsilon.redundancy)

    # 4e. Visualization (Epsilon Solution)
    print("\n[VISUALIZATION] Launching Epsilon solution visualization (Close window to exit)...")
    visualizer.visualize_solution_pcd(
        optimization_result_epsilon,
    )
    print("NOTE: For permanent files, save the screenshot during the interactive Open3D visualization.")

    # --- 5. Final Summary and Graphs ---
    print("\n" + "="*80)
    print(f"COMPARISON SUMMARY: Epsilon vs Raycast for {MESH_CATEGORY}/{MESH_NAME}")
    print("="*80)
    
    # Display Summary Table
    print("Metric Comparison Table:")
    print("--------------------------------------------------------------------------------")
    print(f"{'Metric':<25} | {'Raycast':<20} | {'Epsilon':<20}")
    print("--------------------------------------------------------------------------------")
    print(f"{'Num. Viewpoints':<25} | {comparison_results['Num_Viewpoints'][0]:<20} | {comparison_results['Num_Viewpoints'][1]:<20}")
    print(f"{'Total Time (s)':<25} | {comparison_results['Total_Time'][0]:<20.4f} | {comparison_results['Total_Time'][1]:<20.4f}")
    # print(f"{'  - Visibility Time (s)':<25} | {comparison_results['Visibility_Time'][0]:<20.4f} | {comparison_results['Visibility_Time'][1]:<20.4f}")
    # print(f"{'  - Optimization Time (s)':<25} | {comparison_results['Optimization_Time'][0]:<20.4f} | {comparison_results['Optimization_Time'][1]:<20.4f}")
    print(f"{'Reported Coverage (%)':<25} | {comparison_results['Reported_Coverage'][0]*100:<20.2f} | {comparison_results['Reported_Coverage'][1]*100:<20.2f}")
    print(f"{'Actual Coverage (%)':<25} | {comparison_results['Actual_Coverage'][0]*100:<20.2f} | {comparison_results['Actual_Coverage'][1]*100:<20.2f}")
    print(f"{'Redundancy (views/point)':<25} | {comparison_results['Redundancy'][0]:<20.4f} | {comparison_results['Redundancy'][1]:<20.4f}")
    print("--------------------------------------------------------------------------------")

    # Generate and save graphs
    generate_comparison_graphs(comparison_results, "Bunny")

# Execute the comparison
if __name__ == "__main__":
    run_comparison_pipeline()