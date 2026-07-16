"""Differentiable camera-frustum geometry, independent of any visibility backbone.

These are the pieces of ``HPRO_limited`` that describe *the camera* rather than
*the occlusion model*: the 6D rotation parameterisation of Zhou et al. (2019)
and the soft pyramid frustum mask.  They live here so that any occlusion
backbone (analytic HPRO, neural NVPS, an ensemble) can be gated by the same
camera model without depending on ``HPRO_limited`` -- previously the NVPS
diagnostics had to instantiate a throwaway ``HPRO_limited`` purely to reach
these two functions.

``HPRO_limited`` delegates to this module, so its behaviour is unchanged and
Stage-1 results remain valid.
"""

import math

import torch
import torch.nn.functional as F


def six_d_to_rotation_matrix(rot_6d: torch.Tensor):
    """
    Convert a 6D rotation representation (Zhou et al. 2019) to an orthonormal
    camera frame via Gram-Schmidt.

    The 6D vector encodes the first two columns of the rotation matrix::

        rot_6d = [a1_x, a1_y, a1_z,  a2_x, a2_y, a2_z]

    where *a1* approximates the **look** (forward/gaze) axis and *a2*
    approximates the **up** axis.  After ortho-normalisation:

    * ``look  = normalize(a1)``                   -- forward axis (exact)
    * ``up    = normalize(a2 - (a2·look) look)``  -- up axis (orthogonal to look)
    * ``right = cross(look, up)``                 -- right axis

    This parameterisation is singularity-free and unconstrained, which is what
    makes it suitable for gradient-based pose optimisation.

    Args:
        rot_6d (torch.Tensor): Shape ``(B, 6)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(look, up, right)``,
        each of shape ``(B, 3)``.
    """
    a1 = rot_6d[:, :3]   # (B, 3) -- approximate look direction
    a2 = rot_6d[:, 3:6]  # (B, 3) -- approximate up direction

    look = F.normalize(a1, dim=-1)                              # (B, 3)
    up = a2 - (a2 * look).sum(dim=-1, keepdim=True) * look      # orthogonalise
    up = F.normalize(up, dim=-1)                                 # (B, 3)
    right = torch.linalg.cross(look, up)                        # (B, 3)

    return look, up, right


def compute_frustum_mask(
    pts: torch.Tensor,
    viewpoint: torch.Tensor,
    look: torch.Tensor,
    up: torch.Tensor,
    right: torch.Tensor,
    fov_h: float,
    fov_v: float,
    near: float,
    far: float,
    sharpness: float,
) -> torch.Tensor:
    """
    Compute a differentiable per-point frustum membership score in [0, 1].

    Each frustum wall (left, right, top, bottom, near, far) contributes a
    sigmoid gate.  The final score is the product of all six gates::

        f = σ(sharpness·(tan_h − |h_tan|))
          × σ(sharpness·(tan_v − |v_tan|))
          × σ(sharpness·(depth − near))
          × σ(sharpness·(far  − depth))

    Points fully inside the frustum receive scores close to 1; points outside or
    near the boundary decay smoothly toward 0.  Points behind the camera are
    suppressed by the near-plane gate.

    Args:
        pts (torch.Tensor): Point cloud, shape ``(B, 3, N)``.
        viewpoint (torch.Tensor): Camera position, shape ``(B, 3, 1)``.
        look (torch.Tensor): Unit gaze vector, shape ``(B, 3)``.
        up (torch.Tensor): Unit camera-up vector, shape ``(B, 3)``.
        right (torch.Tensor): Unit camera-right vector, shape ``(B, 3)``.
        fov_h (float): Horizontal FOV in radians.
        fov_v (float): Vertical FOV in radians.
        near (float): Near-plane distance (must be > 0).
        far (float): Far-plane distance (must be > near).
        sharpness (float): Sigmoid steepness.

    Returns:
        torch.Tensor: Frustum scores, shape ``(B, N)``, values in (0, 1).
    """
    eps = 1e-6

    # Vectors from viewpoint to each point: (B, 3, N)
    v = pts - viewpoint  # viewpoint already (B, 3, 1) -- broadcasts

    # Project onto camera axes -> (B, N)
    depth = (v * look.unsqueeze(-1)).sum(dim=1)     # along gaze (+forward)
    h_coord = (v * right.unsqueeze(-1)).sum(dim=1)  # along right
    v_coord = (v * up.unsqueeze(-1)).sum(dim=1)     # along up

    # Tangent-space angular coordinates (perspective divide).
    # Clamp the denominator to a positive floor so points at or behind the
    # camera (depth <= 0) cannot produce an exploding / div-by-zero tangent
    # (which would yield NaN gradients). Such points are out of the frustum
    # anyway and are suppressed by the f_near gate below, so this clamp does
    # not affect the score of any genuinely in-frustum point.
    depth_pos = torch.clamp(depth, min=eps)   # (B, N)
    h_tan = h_coord / depth_pos   # (B, N)
    v_tan = v_coord / depth_pos   # (B, N)

    # FOV half-angle limits in tangent space
    tan_h = math.tan(fov_h / 2.0)
    tan_v = math.tan(fov_v / 2.0)

    # Sigmoid gates -- positive argument = inside frustum
    f_h    = torch.sigmoid(sharpness * (tan_h - h_tan.abs()))
    f_v    = torch.sigmoid(sharpness * (tan_v - v_tan.abs()))
    f_near = torch.sigmoid(sharpness * (depth - near))
    f_far  = torch.sigmoid(sharpness * (far - depth))

    return f_h * f_v * f_near * f_far  # (B, N)
