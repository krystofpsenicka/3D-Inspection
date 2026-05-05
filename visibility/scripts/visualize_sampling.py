"""Visualise sampler variants in Isaac Sim.

Each sampler variant is run on a small fresh problem (or, if ``--input
<pipeline_dir>`` is given, on the saved candidates) and shown as four phases:

  1. mesh + target pointcloud
  2. free-space sampling heatmap (outside + inside)
  3. emitted candidates with frustums
  4. visibility-coloured points

Press ``N`` to advance, ``Q`` to quit.

Usage::

    python -m visibility.scripts.visualize_sampling --sampler weighted
    python -m visibility.scripts.visualize_sampling --sampler optimizing --num-candidates 30
    python -m visibility.scripts.visualize_sampling --input outputs/full_pipeline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cupy as cp
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh_path", nargs="?", default=None, help="Optional mesh override.")
    p.add_argument(
        "--sampler",
        choices=["weighted", "targeted", "optimizing"],
        default="weighted",
    )
    p.add_argument("--num-candidates", type=int, default=30)
    p.add_argument("--num-points", type=int, default=20000)
    p.add_argument("--side", choices=["outside", "inside"], default="outside")
    p.add_argument("--collision-radius", type=float, default=0.5)
    p.add_argument("--frustum-far", type=float, default=7.0)
    p.add_argument("--frustum-near", type=float, default=0.01)
    p.add_argument("--frustum-fov-deg", type=float, default=45.0)
    p.add_argument("--frustum-aspect", type=float, default=1.0)
    p.add_argument("--curvature-weighting", action="store_true")
    p.add_argument("--phase-duration", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--mesh-target-length",
        type=float,
        default=50.0,
        help="Scale mesh so longest axis equals this length (metres).",
    )
    p.add_argument(
        "--mesh-pose",
        type=float,
        nargs=7,
        default=[0.0, 0.0, 1.5, 0.7071067811865476, -0.7071067811865476, 0.0, 0.0],
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="World-frame pose applied to mesh after scaling.",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Pipeline directory written by run_full_pipeline (load instead of fresh run).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _trimesh_to_o3d(mesh_tm):
    import open3d as o3d

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_tm.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_tm.faces))
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


def _load_or_build_mesh(args) -> tuple:
    """Return (mesh_o3d, mesh_trimesh, saved_data) using shared.mesh_loader for scale/pose."""
    from shared.mesh_loader import load_and_transform_mesh

    if args.input:
        from VRP.utils.serialization import load_pipeline

        data = load_pipeline(args.input)
        mesh_tm = load_and_transform_mesh(
            data["mesh_path"], data["mesh_target_length"], data["mesh_pose"]
        )
        return _trimesh_to_o3d(mesh_tm), mesh_tm, data

    mesh_path = args.mesh_path or os.path.join(
        REPO_ROOT, "models", "duke_of_lancaster_uk_clipped.glb"
    )
    mesh_tm = load_and_transform_mesh(mesh_path, args.mesh_target_length, args.mesh_pose)
    return _trimesh_to_o3d(mesh_tm), mesh_tm, None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    from shared.surface_sampler import SurfacePointSampler
    from shared.types import Side
    from visibility.core.types import FrustumParams
    from visibility.sampling import (
        CMAESBackend,
        OptimizingSampler,
        TargetedViewpointSampler,
        WeightedViewpointSampler,
    )
    from visibility.visibility.raycast_cuda import RaycastingVisibilityQueryCuda

    mesh_o3d, mesh_tm, saved = _load_or_build_mesh(args)
    side = Side(args.side)

    # Surface sampling -- either reuse saved or generate fresh.
    if saved is not None and args.input:
        target_points = saved["target_points"]
        normals = saved["normals"]
    else:
        target_points, normals = SurfacePointSampler().sample(mesh_o3d, args.num_points)

    target_points_gpu = cp.asarray(target_points, dtype=cp.float32)
    normals_gpu = cp.asarray(normals, dtype=cp.float32)
    if side == Side.INSIDE:
        normals_gpu = -normals_gpu

    frustum_params = FrustumParams(
        fov_y=np.deg2rad(args.frustum_fov_deg),
        aspect=args.frustum_aspect,
        near=args.frustum_near,
        far=args.frustum_far,
    )

    # Build sampler
    sampler_kwargs = dict(
        mesh=mesh_o3d,
        target_points=target_points_gpu,
        normals=normals_gpu,
        frustum_far=args.frustum_far,
        collision_radius=args.collision_radius,
    )
    if args.sampler == "weighted":
        sampler = WeightedViewpointSampler(**sampler_kwargs)
    elif args.sampler == "targeted":
        sampler = TargetedViewpointSampler(**sampler_kwargs)
    else:  # optimizing
        sampler = OptimizingSampler(
            backend=CMAESBackend(),
            random_sampler=WeightedViewpointSampler(**sampler_kwargs),
            **sampler_kwargs,
        )

    # Sample candidates
    if args.sampler == "weighted":
        pos_gpu, rot_gpu = sampler.sample(
            args.num_candidates, side=side, curvature_weighting=args.curvature_weighting,
        )
    elif args.sampler == "targeted":
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points_gpu)),
            args.num_candidates,
            side=side,
            curvature_weighting=args.curvature_weighting,
        )
    else:
        # Optimising sampler: warm-start from a small uniform pool, then optimise.
        warm_sampler = sampler.random_sampler
        warm_n = max(args.num_candidates // 2, 5)
        warm_pos, warm_rot = warm_sampler.sample(warm_n, side=side)
        query_pre = RaycastingVisibilityQueryCuda(
            mesh=mesh_o3d,
            target_points=target_points_gpu,
            normals=normals_gpu,
            frustum_params=frustum_params,
        )
        V_warm, _ = query_pre.compute_visibility_batch(warm_pos, warm_rot)
        coverage_count = V_warm.astype(cp.int32).sum(axis=0)
        opt_pos, opt_rot, _ = sampler.sample_optimized(
            args.num_candidates - warm_n,
            coverage_count,
            query_pre,
            existing_pos_gpu=warm_pos,
            existing_rot_gpu=warm_rot,
            side=side,
        )
        if len(opt_pos) > 0:
            pos_gpu = cp.concatenate([warm_pos, opt_pos])
            rot_gpu = cp.concatenate([warm_rot, opt_rot])
        else:
            pos_gpu, rot_gpu = warm_pos, warm_rot

    # Visibility for the candidates
    vis_query = RaycastingVisibilityQueryCuda(
        mesh=mesh_o3d,
        target_points=target_points_gpu,
        normals=normals_gpu,
        frustum_params=frustum_params,
    )
    V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    V_np = cp.asnumpy(V)

    # Free-space heatmap inputs (both sides)
    outside_pos_gpu, outside_w_gpu, _ = sampler.get_feasible_sampling_data(
        side=Side.OUTSIDE, curvature_weighting=args.curvature_weighting
    )
    inside_pos_gpu, inside_w_gpu, _ = sampler.get_feasible_sampling_data(
        side=Side.INSIDE, curvature_weighting=args.curvature_weighting
    )
    outside_pos = cp.asnumpy(outside_pos_gpu)
    outside_w = cp.asnumpy(outside_w_gpu)
    inside_pos = cp.asnumpy(inside_pos_gpu)
    inside_w = cp.asnumpy(inside_w_gpu)

    # Free GPU memory used by the sampler / query before launching Isaac.
    del sampler, vis_query
    cp.get_default_memory_pool().free_all_blocks()

    # ── Isaac Sim ──────────────────────────────────────────────────────
    from visualization_isaac import (
        IsaacApp,
        ModelVisualizer,
        Phase,
        PhaseController,
        SamplingVisualizer,
        add_dome_light,
    )

    target_points_np = np.asarray(target_points)
    normals_np = np.asarray(normals)

    with IsaacApp(headless=args.headless) as ctx:
        add_dome_light(ctx.stage)

        model_vis = ModelVisualizer(mesh_tm, target_points_np, normals_np)
        sampling_vis = SamplingVisualizer(mesh_tm, target_points_np, normals_np, frustum_params)

        def enter_mesh(stage, parent):
            model_vis.add_mesh(stage, f"{parent}/mesh")
            model_vis.add_points(stage, f"{parent}/points", color=(0.85, 0.85, 0.85))

        def enter_freespace(stage, parent):
            sampling_vis.add_free_space(
                stage, parent, outside_pos, outside_w, inside_pos, inside_w,
                point_size=0.05,
            )

        def enter_candidates(stage, parent):
            sampling_vis.add_candidates(stage, parent, pos_np, rot_np)

        def enter_visibility(stage, parent):
            sampling_vis.add_candidates(stage, parent, pos_np, rot_np, visibility_map=V_np)

        D = args.phase_duration
        phases = [
            Phase("mesh_and_points", enter=enter_mesh, duration=D),
            Phase("free_space_heatmap", enter=enter_freespace, duration=D),
            Phase("candidates", enter=enter_candidates, duration=D),
            Phase("visibility_coloured", enter=enter_visibility, duration=D),
        ]
        controller = PhaseController(ctx, phases)
        if args.headless:
            controller.headless_play()
        else:
            controller.run()


if __name__ == "__main__":
    main()
