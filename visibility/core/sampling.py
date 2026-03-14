import numpy as np
import open3d as o3d
from typing import List, Tuple

from .types import FrustumParams


class ViewpointSampler:
    """
    Handles generating candidate viewpoints and directions, supporting
    both external (outside) and internal (inside) mesh sampling.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_far: float):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.frustum_far = frustum_far
        self.num_points = len(target_points)

        if self.num_points == 0:
            print("[ViewpointSampler] Warning: No target points available.")

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        print("[ViewpointSampler] Created O3D RaycastingScene for internal checks.")

    def sample_outside_mesh(self, num_candidates: int, offset_scale: float = 0.95,
                            pos_noise_std: float = 0.05,
                            dir_noise_std: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Samples candidates outside the mesh by offsetting target points along the normal.
        """
        if self.num_points == 0:
            return []

        print(f"\n[ViewpointSampler] Sampling {num_candidates} viewpoints from OUTSIDE mesh...")

        num_candidates = min(num_candidates, self.num_points)
        indices = np.random.choice(self.num_points, num_candidates, replace=False)

        selected_points = self.target_points[indices]
        selected_normals = self.normals[indices]

        # Position/Offset Noise
        base_offset = self.frustum_far * offset_scale
        pos_noise = np.random.normal(0.0, pos_noise_std, num_candidates)
        final_offsets = base_offset + pos_noise
        final_offsets[final_offsets < (base_offset * 0.1)] = base_offset * 0.1

        candidate_pos = selected_points + selected_normals * final_offsets[:, np.newaxis]

        # Direction Noise
        base_dir = -selected_normals
        dir_noise_vectors = np.random.normal(0.0, dir_noise_std, size=base_dir.shape)
        noisy_dir = base_dir + dir_noise_vectors
        candidate_dir = noisy_dir / np.linalg.norm(noisy_dir, axis=1)[:, np.newaxis]

        candidates = list(zip(candidate_pos, candidate_dir))
        print(f"[ViewpointSampler] Generated {len(candidates)} outside candidates with noise.")

        return candidates

    def sample_inside_mesh(self, num_candidates: int,
                           max_dist_factor: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Samples candidates INSIDE the mesh.
        """
        if self.num_points == 0 or self.scene is None:
            print("[ViewpointSampler] Skipping internal sampling (no points or no scene).")
            return []

        print(f"\n[ViewpointSampler] Sampling {num_candidates} viewpoints from INSIDE mesh...")

        candidates = []
        attempts = 0
        max_attempts = num_candidates * 5

        while len(candidates) < num_candidates and attempts < max_attempts:
            idx = np.random.randint(0, self.num_points)
            p = self.target_points[idx]
            n = self.normals[idx]

            dist = np.random.uniform(0.1, self.frustum_far * max_dist_factor)
            vp_pos = p - n * dist
            vp_dir = n

            rays = o3d.core.Tensor(
                [[vp_pos[0], vp_pos[1], vp_pos[2], 0.0, 0.0, 1.0]],
                dtype=o3d.core.DType.Float32,
            )
            ans = self.scene.cast_rays(rays)

            if dist < self.frustum_far * 0.9:
                candidates.append((vp_pos, vp_dir))

            attempts += 1

        print(f"[ViewpointSampler] Generated {len(candidates)} inside candidates (after {attempts} attempts).")
        return candidates
