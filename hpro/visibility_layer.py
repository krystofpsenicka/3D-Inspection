"""Backbone-agnostic differentiable visibility layer for real cameras (C1).

Composes an interchangeable occlusion backbone (``backbones.py``) with the
shared camera model (``frustum.py``)::

    w[v, j] = w_occlusion(backbone)[v, j] · f_frustum[v, j]

Both factors are differentiable w.r.t. the camera position and the 6D
orientation, so ``w`` can be fed straight into the soft set-cover objective in
``multi_viewpoint.py``.

This generalises ``HPRO_limited``, which hard-codes HPRO as the occlusion model.
With ``HPROBackbone`` the two are numerically identical (asserted by
``smoke_test.py``); the point of the indirection is that NVPS and ensembles drop
in without touching the camera model, which is what makes the §3 backbone
comparison controlled.

The normal gate and measurement-quality weighting (§7 step 2) will be added here
as further factors, so that every backbone inherits them at once.
"""

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import frustum as _frustum                      # noqa: E402
from backbones import VisibilityBackbone        # noqa: E402


class GatedVisibilityLayer(nn.Module):
    """Frustum-gated visibility over an arbitrary occlusion backbone.

    Args:
        backbone: the occlusion model. Call ``backbone.prepare(...)`` before use
            if it needs per-cloud state (NVPS does).
        fov_h: horizontal field-of-view in radians.
        fov_v: vertical field-of-view in radians.
        near: near-plane distance (> 0).
        far: far-plane distance (> near).
        frustum_sharpness: sigmoid steepness of the frustum walls. Higher is a
            sharper boundary; too high starves the gradient outside the frustum.
        device: compute device; defaults to the backbone's.
    """

    def __init__(
        self,
        backbone: VisibilityBackbone,
        fov_h: float = math.pi / 2,
        fov_v: float = math.pi / 2,
        near: float = 0.1,
        far: float = 100.0,
        frustum_sharpness: float = 50.0,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.near = near
        self.far = far
        self.frustum_sharpness = frustum_sharpness
        self.device = device or getattr(backbone, "device", None) or (
            "cuda" if torch.cuda.is_available() else "cpu")

    def prepare(self, pts_np, normals_np=None) -> None:
        """Forward per-cloud precomputation to the backbone."""
        self.backbone.prepare(pts_np, normals_np)

    def forward(
        self,
        pts_t: torch.Tensor,
        viewpoints: torch.Tensor,
        rot_6d: torch.Tensor,
        fov_h: Optional[float] = None,
        fov_v: Optional[float] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        frustum_sharpness: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            pts_t: (3, N) point cloud on the compute device.
            viewpoints: (V, 3) camera positions (learnable).
            rot_6d: (V, 6) camera orientations, Zhou et al. 6D form (learnable).
            fov_h, fov_v, near, far, frustum_sharpness: per-call overrides of the
                values set at construction.

        Returns:
            (V, N) combined visibility scores, >= 0, differentiable w.r.t.
            ``viewpoints`` and ``rot_6d``.
        """
        fov_h = fov_h if fov_h is not None else self.fov_h
        fov_v = fov_v if fov_v is not None else self.fov_v
        near = near if near is not None else self.near
        far = far if far is not None else self.far
        sharp = frustum_sharpness if frustum_sharpness is not None else self.frustum_sharpness

        V = viewpoints.shape[0]

        w_occ = self.backbone(pts_t, viewpoints)                 # (V, N)

        look, up, right = _frustum.six_d_to_rotation_matrix(rot_6d)
        f = _frustum.compute_frustum_mask(
            pts_t.unsqueeze(0).expand(V, -1, -1),                 # (V, 3, N)
            viewpoints.unsqueeze(2),                              # (V, 3, 1)
            look, up, right,
            fov_h, fov_v, near, far, sharp,
        )                                                         # (V, N)

        return w_occ * f
