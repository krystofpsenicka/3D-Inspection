"""Surface point sampler with disk caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from shared.geometry import orient_normals_outward

logger = logging.getLogger(__name__)

# Repo root: two levels up from this file (shared/surface_sampler.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "surface_samples"


class SurfacePointSampler:
    """Poisson-disk surface sampling with normal estimation and disk caching.

    Wraps the standard Open3D pipeline:
      1. ``sample_points_poisson_disk``
      2. ``estimate_normals`` (KDTreeSearchParamHybrid)
      3. ``orient_normals_consistent_tangent_plane`` (MST propagation)
      4. ``orient_normals_outward`` (global centroid flip)

    Results are cached as NPZ + JSON so repeated runs with the same mesh
    and params return instantly.
    """

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR

    def sample(
        self,
        mesh: o3d.geometry.TriangleMesh,
        num_points: int,
        normal_radius: float = 0.5,
        normal_max_nn: int = 30,
        tangent_plane_k: int = 15,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample surface points and estimate outward normals.

        Returns ``(target_points, normals)`` as numpy arrays of shape ``(N, 3)``.
        Uses disk cache when available.
        """
        cache_key = self._build_cache_key(
            mesh, num_points, normal_radius, normal_max_nn, tangent_plane_k, seed,
        )

        # Try cache
        result = self._load_cache(cache_key)
        if result is not None:
            logger.info("Surface sample cache HIT (%s)", cache_key[:16])
            return result

        logger.info("Surface sample cache MISS — computing …")
        t0 = time.perf_counter()

        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn,
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=tangent_plane_k)

        target_points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        normals = orient_normals_outward(target_points, normals)

        elapsed = time.perf_counter() - t0
        logger.info("  Sampled %d points in %.1f s", len(target_points), elapsed)

        self._save_cache(
            cache_key, target_points, normals,
            params=dict(
                num_points=num_points,
                normal_radius=normal_radius,
                normal_max_nn=normal_max_nn,
                tangent_plane_k=tangent_plane_k,
                seed=seed,
            ),
        )
        return target_points, normals

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mesh_hash(mesh: o3d.geometry.TriangleMesh) -> str:
        h = hashlib.sha256()
        h.update(np.asarray(mesh.vertices).tobytes())
        h.update(np.asarray(mesh.triangles).tobytes())
        return h.hexdigest()

    @staticmethod
    def _build_cache_key(
        mesh: o3d.geometry.TriangleMesh,
        num_points: int,
        normal_radius: float,
        normal_max_nn: int,
        tangent_plane_k: int,
        seed: int | None,
    ) -> str:
        mesh_hash = SurfacePointSampler._mesh_hash(mesh)[:16]
        seed_str = str(seed) if seed is not None else "none"
        raw = (
            f"{mesh_hash}_{num_points}_{normal_radius}_{normal_max_nn}"
            f"_{tangent_plane_k}_{seed_str}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        return (
            self._cache_dir / f"surface_{key}.npz",
            self._cache_dir / f"surface_{key}.json",
        )

    def _load_cache(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        npz_path, json_path = self._cache_paths(key)
        if not npz_path.exists() or not json_path.exists():
            return None
        try:
            data = np.load(npz_path)
            return data["target_points"], data["normals"]
        except Exception:
            logger.warning("Corrupt cache entry %s — recomputing", key[:16])
            return None

    def _save_cache(
        self,
        key: str,
        target_points: np.ndarray,
        normals: np.ndarray,
        params: dict,
    ) -> None:
        npz_path, json_path = self._cache_paths(key)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, target_points=target_points, normals=normals)
        meta = {
            "cache_key": key,
            "num_points": len(target_points),
            "params": params,
            "timestamp": time.time(),
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("  Cached to %s", npz_path)
