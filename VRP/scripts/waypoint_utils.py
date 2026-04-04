"""
VRP Planner – Waypoint Loader

Supports loading waypoints from multiple sources:
  1. Random free-space sampling (via OccupancyGrid)
  2. JSON / NPZ manual specification
  3. 3D-Inspection ViewpointResult objects
  4. CSV file (x,y,z per row)

All loaders return ``(positions, rotmats)`` — positions as (N, 3) and
rotation matrices as (N, 3, 3) numpy arrays. No quaternions.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple

import cupy as cp
import numpy as np

from VRP.core.constants import (
    MESH_PATH,
    MESH_TARGET_LENGTH,
    PROJECT_ROOT,
    ROBOT_RADIUS,
)


# ── Mesh proximity filter (cached) ───────────────────────────────────────────

_SCALED_MESH = None   # trimesh.Trimesh, lazily loaded once


def _get_scaled_mesh():
    """Return the ship mesh with the same scale + pose as the occupancy grid."""
    global _SCALED_MESH
    if _SCALED_MESH is not None:
        return _SCALED_MESH

    from shared.mesh_loader import load_and_transform_mesh
    from VRP.core.constants import MESH_POSE

    _SCALED_MESH = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    return _SCALED_MESH


# ── Rotation matrix helpers ──────────────────────────────────────────────────

def _identity_rotmats(n: int) -> np.ndarray:
    """Return (N, 3, 3) identity rotation matrices."""
    return np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))


def _yaw_pitch_to_rotmats(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Compute (N, 3, 3) rotation matrices from yaw + pitch angles.

    R = Rz(yaw) @ Ry(pitch). Camera looks along local +X.
    """
    N = len(yaw)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp_, sp = np.cos(pitch), np.sin(pitch)

    R = np.zeros((N, 3, 3), dtype=np.float32)
    # Rz(yaw) @ Ry(pitch)
    R[:, 0, 0] = cy * cp_
    R[:, 0, 1] = -sy
    R[:, 0, 2] = cy * sp
    R[:, 1, 0] = sy * cp_
    R[:, 1, 1] = cy
    R[:, 1, 2] = sy * sp
    R[:, 2, 0] = -sp
    R[:, 2, 1] = 0.0
    R[:, 2, 2] = cp_
    return R


# ── Source 1: Random free-space sampling ─────────────────────────────────────

def load_random_waypoints(
    n: int,
    og,  # OccupancyGrid (GPU-resident)
    seed: Optional[int] = 42,
    z_min: float = 0.5,
    z_max: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    mesh_clearance: Optional[float] = None,
    max_attempts: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` random free-space waypoints via rejection sampling.

    Returns ``(positions (N,3), rotmats (N,3,3))`` as numpy arrays.
    """
    rng = np.random.RandomState(seed)

    # Get grid bounds (GPU → CPU for sampling)
    grid_origin_np = cp.asnumpy(og.origin)
    grid_shape_np = np.array(og.grid.shape)
    grid_np = cp.asnumpy(og.grid)

    grid_min = grid_origin_np
    grid_max = grid_origin_np + grid_shape_np * og.resolution
    if x_min is None:
        x_min = float(grid_min[0])
    if x_max is None:
        x_max = float(grid_max[0])
    if y_min is None:
        y_min = float(grid_min[1])
    if y_max is None:
        y_max = float(grid_max[1])
    if z_max is None:
        z_max = float(grid_max[2])
    if mesh_clearance is None:
        mesh_clearance = ROBOT_RADIUS * 2.0

    print(f"[WaypointLoader] Sampling {n} waypoints  "
          f"x=[{x_min:.1f},{x_max:.1f}] y=[{y_min:.1f},{y_max:.1f}] "
          f"z=[{z_min:.1f},{z_max:.1f}]  clearance={mesh_clearance:.2f}m")

    prox = None
    if mesh_clearance > 0 and os.path.isfile(MESH_PATH):
        import trimesh
        mesh = _get_scaled_mesh()
        prox = trimesh.proximity.ProximityQuery(mesh)

    collected: List[np.ndarray] = []
    batch_size = max(n * 50, 500)
    total_tried = 0

    while len(collected) < n and total_tried < max_attempts:
        pts = np.column_stack([
            rng.uniform(x_min, x_max, batch_size),
            rng.uniform(y_min, y_max, batch_size),
            rng.uniform(z_min, z_max, batch_size),
        ])
        total_tried += batch_size

        # Check occupancy on CPU grid
        ijk = np.floor((pts - grid_origin_np) / og.resolution).astype(int)
        in_bounds = np.all(ijk >= 0, axis=1) & np.all(ijk < grid_shape_np, axis=1)
        pts = pts[in_bounds]
        ijk = ijk[in_bounds]
        free_mask = ~grid_np[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        pts = pts[free_mask]

        if prox is not None and len(pts) > 0:
            _, dists, _ = prox.on_surface(pts)
            pts = pts[dists >= mesh_clearance]

        for p in pts:
            if len(collected) >= n:
                break
            collected.append(p)

    if len(collected) < n:
        raise ValueError(
            f"Only {len(collected)} valid points found after {total_tried} "
            f"attempts; requested {n}."
        )

    positions = np.array(collected[:n], dtype=np.float32)
    print(f"[WaypointLoader] Sampled {n} waypoints in {total_tried} attempts")

    # Random yaw + pitch → rotation matrices
    yaw = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    pitch = rng.uniform(-np.pi / 4, np.pi / 4, n).astype(np.float32)
    rotmats = _yaw_pitch_to_rotmats(yaw, pitch)

    return positions, rotmats


# ── Source 2: JSON / NPZ file ─────────────────────────────────────────────────

def load_waypoints_from_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load waypoints from a JSON file.

    Expected format: dict with 'positions' (N×3) and 'rotmats' (N×3×3) keys.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        positions = np.array(data["positions"], dtype=np.float32)
        rotmats = np.array(data["rotmats"], dtype=np.float32)
    else:
        # Legacy: list of [x, y, z] — identity rotmats
        positions = np.array(data, dtype=np.float32)[:, :3]
        rotmats = _identity_rotmats(len(positions))
    return positions, rotmats


def load_waypoints_from_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load waypoints from an NPZ file.

    Expects 'positions' (N×3) and 'rotmats' (N×3×3) keys.
    """
    data = np.load(path, allow_pickle=True)
    positions = np.asarray(data["positions"], dtype=np.float32)
    if "rotmats" in data:
        rotmats = np.asarray(data["rotmats"], dtype=np.float32)
    else:
        rotmats = _identity_rotmats(len(positions))
    return positions, rotmats


def load_waypoints_from_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load waypoints from a CSV file (x,y,z per row)."""
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    positions = arr[:, :3]
    rotmats = _identity_rotmats(len(positions))
    return positions, rotmats


# ── Source 3: 3D-Inspection ViewpointResult ───────────────────────────────────

def load_waypoints_from_inspection(
    models_dir: str,
    result_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load waypoints from 3D-Inspection optimisation results.

    ViewpointResult objects have ``position`` and ``orientation`` (rotation
    matrix) attributes — used directly, no quaternion conversion.
    """
    inspection_root = os.path.join(PROJECT_ROOT, "3D-Inspection", "methods_analysis")
    if inspection_root not in sys.path:
        sys.path.insert(0, inspection_root)

    try:
        from utils import ViewpointResult, OptimizationResult  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Cannot import 3D-Inspection utils from {inspection_root}: {e}"
        )

    if result_file:
        result_files = [result_file]
    else:
        result_files = []
        for fname in os.listdir(models_dir):
            if fname.endswith(".npz") or fname.endswith(".pkl"):
                result_files.append(os.path.join(models_dir, fname))

    if not result_files:
        raise FileNotFoundError(
            f"No NPZ/PKL result files found in {models_dir}."
        )

    all_viewpoints = []
    for rpath in result_files:
        try:
            data = np.load(rpath, allow_pickle=True)
            if "viewpoints" in data:
                all_viewpoints.extend(data["viewpoints"].tolist())
            elif "result" in data:
                result_obj = data["result"].item()
                if hasattr(result_obj, "viewpoints"):
                    all_viewpoints.extend(result_obj.viewpoints)
        except Exception as e:
            print(f"[WaypointLoader] Warning: could not load {rpath}: {e}")

    if not all_viewpoints:
        raise ValueError("No ViewpointResult objects found in the result files.")

    positions = np.array([vp.position for vp in all_viewpoints], dtype=np.float32)
    rotmats = np.array([vp.orientation for vp in all_viewpoints], dtype=np.float32)

    print(f"[WaypointLoader] Loaded {len(positions)} viewpoints from "
          f"{len(result_files)} file(s).")
    return positions, rotmats


# ── Universal loader ──────────────────────────────────────────────────────────

def load_waypoints(
    source: str,
    n_random: Optional[int] = None,
    og=None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Universal dispatcher for waypoint loading.

    Returns ``(positions (N, 3), rotmats (N, 3, 3))`` as numpy arrays.
    """
    if source == "random":
        if og is None or n_random is None:
            raise ValueError("'random' source requires og and n_random.")
        return load_random_waypoints(n_random, og, seed=random_seed)
    if source.endswith(".json"):
        return load_waypoints_from_json(source)
    if source.endswith(".npz"):
        return load_waypoints_from_npz(source)
    if source.endswith(".csv"):
        return load_waypoints_from_csv(source)
    if os.path.isdir(source):
        return load_waypoints_from_inspection(source)
    raise ValueError(
        f"Cannot determine waypoint source type from: '{source}'. "
        "Use 'random', a .json/.npz/.csv file, or a directory."
    )
