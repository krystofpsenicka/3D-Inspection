#!/usr/bin/env python3
"""Validate TOSCA models: check that normals are non-degenerate.

Usage:
    conda run -n isaaclab python -m experiments.common.validate_models
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import open3d as o3d

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import TOSCA_ALL, TOSCA_DIR, ModelConfig
from experiments.common.pipeline_setup import _check_normals

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def validate_all_tosca(models: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Validate normals for TOSCA models.

    Returns (valid_models, invalid_models).
    """
    if models is None:
        models = TOSCA_ALL

    valid = []
    invalid = []

    for name in models:
        try:
            cfg = ModelConfig.tosca(name)
        except FileNotFoundError:
            logger.warning("SKIP %s: file not found", name)
            invalid.append(name)
            continue

        logger.info("Validating %s ...", name)

        try:
            import trimesh
            mesh = trimesh.load(cfg.mesh_path, force="mesh")
            longest = float(mesh.extents.max())
            if longest > 0:
                mesh.apply_scale(cfg.target_length / longest)

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
            o3d_mesh.compute_vertex_normals()

            from shared.surface_sampler import SurfacePointSampler
            sampler = SurfacePointSampler()
            pts, norms = sampler.sample(o3d_mesh, 2000, seed=42)

            if _check_normals(norms, threshold_deg=30.0):
                logger.info("  OK: %s (%d verts, %d faces)", name, len(mesh.vertices), len(mesh.faces))
                valid.append(name)
            else:
                logger.warning("  INVALID: %s (degenerate normals)", name)
                invalid.append(name)

        except Exception as e:
            logger.error("  ERROR: %s: %s", name, e)
            invalid.append(name)

    logger.info("\n=== VALIDATION RESULTS ===")
    logger.info("Valid:   %s", valid)
    logger.info("Invalid: %s", invalid)
    return valid, invalid


if __name__ == "__main__":
    validate_all_tosca()
