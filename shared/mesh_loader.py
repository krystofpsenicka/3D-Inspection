"""
Mesh loading: load, uniform-scale, and pose-transform a mesh.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def load_and_transform_mesh(
    mesh_path: str,
    target_length: float,
    pose: list,
) -> trimesh.Trimesh:
    """Load a mesh, scale so longest axis == *target_length*, apply *pose*.

    Parameters
    ----------
    mesh_path : str
        Path to ``.glb`` / ``.obj`` / ``.stl`` mesh file.
    target_length : float
        Desired length (metres) of the mesh's longest axis.
    pose : list
        ``[x, y, z, qw, qx, qy, qz]`` world-frame pose.

    Returns
    -------
    trimesh.Trimesh
        Scaled and transformed mesh.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    scene_or_mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh

    longest = float(mesh.extents.max())
    scale_factor: float | None = None
    if longest > 0:
        scale_factor = target_length / longest
        mesh.apply_scale(scale_factor)

    T_pose = np.eye(4)
    T_pose[:3, 3] = pose[:3]
    quat_wxyz = pose[3:7]
    rot = R.from_quat(quat_wxyz, scalar_first=True)
    T_pose[:3, :3] = rot.as_matrix()
    mesh.apply_transform(T_pose)

    # Stash the scale factor on the mesh for downstream use
    mesh.metadata["scale_factor"] = scale_factor
    return mesh
