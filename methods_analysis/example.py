import numpy as np
import open3d as o3d
import sys
import os

# --- Import all modules from the created library ---
from utils import create_mock_data, ViewpointResult, OptimizationResult # type: ignore
from visualization import Visualizer # type: ignore
from sampling import ViewpointSampler # type: ignore
from raycast_visibility import RaycastingVisibilityQuery # type: ignore
from greedy_optimizer_with_kernel import GreedyOptimizer # type: ignore
from epsilon_visibility import EpsilonVisibilityQuery

def run_raycasting_greedy_pipeline():
    """
    Implements a full visibility pipeline:
    1. Generates mock mesh/points.
    2. Samples viewpoints outside the mesh.
    3. Uses Raycasting for ground-truth visibility query.
    4. Optimizes viewpoint placement using the Greedy approach.
    5. Visualizes the final result.
    """
    print("="*80)
    print("STARTING VIEWPOINT COVERAGE PIPELINE: Raycasting + Greedy")
    print("="*80)
    
    # --- Configuration ---
    NUM_TARGET_POINTS = 5000
    NUM_CANDIDATE_VPs = 1
    TARGET_COVERAGE = 0.90 # 90% coverage
    MAX_VIEWPOINTS = 20
    
    # 1. Setup Data and Environment
    print(f"\n[SETUP] Generating mock data ({NUM_TARGET_POINTS} target points)...")
    mesh, target_points, normals, frustum_params, _ = create_mock_data(
        num_points=NUM_TARGET_POINTS, 
        num_candidates=NUM_CANDIDATE_VPs,
        mesh_path="models/duke_of_lancaster_uk.glb"
    )

    # 2. Viewpoint Sampling (External/Outside)
    print("\n[STEP 1] Generating candidate viewpoints (Outside Mesh)...")
    sampler = ViewpointSampler(mesh, target_points, normals, frustum_params.far)
    candidates = sampler.sample_outside_mesh(NUM_CANDIDATE_VPs, offset_scale=0.1)
    
    # 3. Visibility Query Setup (Raycasting/BVH)
    print("\n[STEP 2] Initializing Raycasting (Ground-Truth) Visibility Query...")
    visibility_query = EpsilonVisibilityQuery(mesh, target_points, normals, frustum_params)

    # =========================================================================
    # NEW STEP: Pre-compute and Visualize Candidate Visibility
    # =========================================================================
    print("\n[STEP 3] Pre-computing visibility for ALL candidates...")
    
    # Pre-compute visibility for all candidates
    visibility_map = visibility_query.compute_visibility_for_all_candidates(candidates)
    
    # Initialize Visualizer
    visualizer = Visualizer(mesh, target_points, normals, frustum_params)
    
    # Visualize the 1st candidate's visibility
    print("[STEP 3.1] Launching visualization for the FIRST candidate (index 0)...")
    # You can change the index (e.g., candidate_index=5) to see a different one
    visualizer.visualize_visibility_results(visibility_map, candidate_index=0)
    # =========================================================================

    # 4. Optimization (Greedy Set Cover)
    print("\n[STEP 4] Running Greedy Optimizer (Set Cover)...")
    
    # The GreedyOptimizer is passed the query object, which will use the 
    # pre-computed map internally if available, or compute on demand.
    # NOTE: The GreedyOptimizer class in your pipeline must be adjusted to 
    # accept and use the pre-computed visibility_map for efficiency, 
    # but for simplicity here we just use the query object.
    optimizer = GreedyOptimizer(visibility_query)
    
    optimization_result: OptimizationResult = optimizer.optimize(
        candidates=candidates, 
        target_coverage=TARGET_COVERAGE, 
        max_viewpoints=MAX_VIEWPOINTS
    )

    # 5. Output Results
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - FINAL RESULTS")
    print("="*80)
    print(f"Method: {optimization_result.method_name}")
    print(f"Total Time: {optimization_result.total_time:.2f} seconds")
    print(f"Viewpoints Selected: {optimization_result.num_viewpoints} / {MAX_VIEWPOINTS}")
    print(f"Total Coverage Achieved: {optimization_result.total_coverage*100:.2f}%")
    print(f"Average Coverage Redundancy: {optimization_result.redundancy:.2f} views/point")
    print("Selected Viewpoints (Position, New Coverage):")
    for i, vp in enumerate(optimization_result.viewpoints):
        # Calculate new coverage count for printing simplicity
        print(f"  VP {i+1}: Pos={vp.position}, Coverage={len(vp.visible_indices)} points")

    # 6. Visualization
    print("\n[VISUALIZATION] Launching final result visualization (requires GUI)...")
    # Re-use the existing visualizer instance
    visualizer.visualize_solution(optimization_result)
    
    print("\nPipeline finished.")

if __name__ == "__main__":
    run_raycasting_greedy_pipeline()