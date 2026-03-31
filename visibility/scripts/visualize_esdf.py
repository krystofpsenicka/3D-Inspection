#!/usr/bin/env python3
"""
Visualize ESDF voxel grid — 2D slices (matplotlib) or interactive 3D (Open3D).

Usage — 2D slice:
    python -m visibility.scripts.visualize_esdf
    python -m visibility.scripts.visualize_esdf --z_slice 2.0
    python -m visibility.scripts.visualize_esdf --z_slice 1.5 --axis 1

Usage — 3D interactive:
    python -m visibility.scripts.visualize_esdf --mode 3d
    python -m visibility.scripts.visualize_esdf --mode 3d --esdf_band 2.0

Colour convention (both modes):
  Red     = inside obstacle  (positive ESDF)
  White   = surface          (ESDF ≈ 0)
  Blue    = free space       (negative ESDF)
"""

from __future__ import annotations

import argparse

import numpy as np

from visualization import EsdfVisualizer


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_og_and_esdf():
    """Build sampling occupancy grid and compute ESDF.  Returns (og, raw, esdf_3d)."""
    import open3d as o3d
    from shared.mesh_loader import load_and_transform_mesh
    from visibility.sampling.utils.sampling_grid_builder import (
        build_sampling_occupancy_grid, build_sdf_grid,
    )
    from VRP.config import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH

    print("Loading mesh …")
    tm = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))

    print("Building sampling occupancy grid …")
    og = build_sampling_occupancy_grid(o3d_mesh, frustum_far=6.0, min_clearance=1.0)
    raw = og.raw_grid
    print(f"Grid shape: {raw.shape}  origin: {og.origin}  res: {og.resolution}m")

    print("Computing SDF …")
    esdf = build_sdf_grid(og)
    print(f"SDF range: [{esdf.min():.3f}, {esdf.max():.3f}]")
    return og, raw, esdf


def _load_scaled_mesh():
    """Load the ship mesh with the same scale + pose applied in occupancy_grid."""
    import trimesh
    from scipy.spatial.transform import Rotation as R
    from VRP.config import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH

    raw_mesh = trimesh.load(MESH_PATH, force="mesh")
    if isinstance(raw_mesh, trimesh.Scene):
        raw_mesh = trimesh.util.concatenate(list(raw_mesh.geometry.values()))
    longest = float(raw_mesh.extents.max())
    if longest > 0:
        raw_mesh.apply_scale(MESH_TARGET_LENGTH / longest)
    T = np.eye(4)
    T[:3, 3] = MESH_POSE[:3]
    T[:3, :3] = R.from_quat(MESH_POSE[3:7], scalar_first=True).as_matrix()
    raw_mesh.apply_transform(T)
    return raw_mesh


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ESDF visualization — 2D matplotlib slice or 3D Open3D")
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d",
                        help="2d = matplotlib heatmap slice  |  3d = Open3D interactive")

    # ── 2D options ────────────────────────────────────────────────────
    parser.add_argument("--z_slice", type=float, default=1.5,
                        help="Slice position in world-frame metres (2D)")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                        help="Slice axis  0=X  1=Y  2=Z  (2D)")
    parser.add_argument("--vmin", type=float, default=-3.0,
                        help="Colour-bar lower bound in metres (2D)")
    parser.add_argument("--vmax", type=float, default=1.0,
                        help="Colour-bar upper bound in metres (2D)")

    # ── 3D options ────────────────────────────────────────────────────
    parser.add_argument("--esdf_band", type=float, default=2.0,
                        help="Render voxels where |ESDF| < band  metres (3D)")
    parser.add_argument("--show_occupied", action="store_true", default=False,
                        help="Overlay raw occupied voxels as red cloud (3D)")
    parser.add_argument("--show_inflated", action="store_true", default=False,
                        help="Overlay inflation shell as blue cloud (3D)")
    parser.add_argument("--max_points", type=int, default=300_000,
                        help="Max points per layer — subsampled if exceeded (3D)")

    # ── Shared ────────────────────────────────────────────────────────
    parser.add_argument("--show_mesh", action="store_true", default=True,
                        help="Render ship mesh (both modes)")
    parser.add_argument("--no_mesh", dest="show_mesh", action="store_false")

    args = parser.parse_args()

    og, raw, esdf = _build_og_and_esdf()
    scaled_mesh = _load_scaled_mesh() if args.show_mesh else None
    viz = EsdfVisualizer(og, raw, esdf, scaled_mesh=scaled_mesh)

    if args.mode == "3d":
        viz.visualize_3d(
            esdf_band=args.esdf_band,
            show_occupied=args.show_occupied,
            show_inflated=args.show_inflated,
            show_mesh=args.show_mesh,
            max_points=args.max_points,
        )
    else:
        viz.visualize_2d(
            axis=args.axis,
            slice_pos=args.z_slice,
            vmin=args.vmin,
            vmax=args.vmax,
            show_mesh=args.show_mesh,
        )


if __name__ == "__main__":
    main()
