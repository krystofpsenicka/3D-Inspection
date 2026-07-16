import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from HPRO import HPRO
import frustum as _frustum


class HPRO_limited(nn.Module):
    """
    Frustum-constrained differentiable visibility operator (HPRO_limited).

    Extends the HPRO operator with a soft pyramid frustum mask so that only
    points inside the camera frustum contribute to the visibility score.  The
    mask is implemented as a product of sigmoid gates — one for each frustum
    wall (left/right/top/bottom) plus near and far depth planes — making the
    full operator differentiable with respect to the viewpoint position *and*
    the camera orientation.

    Camera orientation is parameterised using the **6D rotation representation**
    of Zhou et al. (2019), i.e. the first two columns of the target rotation
    matrix concatenated into a 6-vector.  Gram–Schmidt ortho-normalisation is
    applied inside the forward pass to recover a full orthonormal camera frame,
    yielding a singularity-free, unconstrained SO(3) parameterisation that is
    well-suited for gradient-based optimisation.

    Attributes:
        fits_in_memory (bool): Passed through to the internal HPRO instance.
        visibility_score_thresh (float): Threshold applied to the combined
            score (HPRO * frustum) to produce the binary visible mask.
        device (str): Compute device.
        fov_h (float): Default horizontal field-of-view in radians.
        fov_v (float): Default vertical field-of-view in radians.
        near (float): Default near-plane distance.
        far (float): Default far-plane distance.
        frustum_sharpness (float): Steepness of the sigmoid gates.  Higher
            values give a sharper frustum boundary (less smooth transition);
            lower values give a wider, softer transition.

    Example — optimising viewpoint and frustum orientation::

        import torch, torch.nn as nn, math
        from HPRO_limited import HPRO_limited

        model = HPRO_limited(fov_h=math.radians(60), fov_v=math.radians(45),
                             near=0.1, far=50.0, frustum_sharpness=20.0,
                             device='cpu')

        # Learnable parameters
        viewpoint = nn.Parameter(torch.tensor([[0.0, 0.0, -4.0]]))  # (1, 3)
        # Identity orientation: look along +Z, up along +Y
        # rot_6d = [a1 | a2] where a1 is the look-direction column and
        # a2 is the up-direction column, each of length 3.
        rot_6d = nn.Parameter(
            torch.tensor([[0.0, 0.0, 1.0,   # a1: look
                           0.0, 1.0, 0.0]])  # a2: up (approx)
        )  # (1, 6)

        optimizer = torch.optim.Adam([viewpoint, rot_6d], lr=1e-3)

        pts = ...  # (1, 3, N) point cloud

        for step in range(1000):
            optimizer.zero_grad()
            vis_pts, vis_idx, w_combined = model(pts, viewpoint, rot_6d,
                                                  gamma=-1e-4)
            loss = -w_combined.sum()   # maximise number of visible points
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        fits_in_memory: bool = True,
        visibility_score_thresh: float = 0.5,
        fov_h: float = math.pi / 2,
        fov_v: float = math.pi / 2,
        near: float = 0.1,
        far: float = 100.0,
        frustum_sharpness: float = 10.0,
        device: str = None,
    ):
        """
        Args:
            fits_in_memory (bool): If True, visibility computation is fully
                parallelised (faster, higher memory).  If False, a loop over
                points is used (slower, lower memory).
            visibility_score_thresh (float): Threshold on the combined
                (HPRO × frustum) score to classify a point as visible.
                Note: because the frustum mask attenuates the raw HPRO score,
                a lower threshold than the plain HPRO default of 0.99 is
                recommended (default: 0.5).
            fov_h (float): Horizontal field-of-view in radians (default π/2).
            fov_v (float): Vertical field-of-view in radians (default π/2).
            near (float): Near-plane distance along the gaze axis (default 0.1).
            far (float): Far-plane distance along the gaze axis (default 100.0).
            frustum_sharpness (float): Sigmoid steepness for frustum walls.
                Higher = sharper boundary, lower = softer transition.
            device (str): Compute device ('cuda' or 'cpu').
        """
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Internal HPRO operator — we call detect_max_in_direction directly
        # so we can combine the raw w scores with the frustum mask before
        # applying the threshold.
        self.hpro = HPRO(
            fits_in_memory=fits_in_memory,
            visiblity_score_thresh=visibility_score_thresh,
            device=device,
        )

        self.visibility_score_thresh = visibility_score_thresh
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.near = near
        self.far = far
        self.frustum_sharpness = frustum_sharpness
        self.device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _6d_to_rotation_matrix(rot_6d: torch.Tensor):
        """
        Convert a 6D rotation representation (Zhou et al. 2019) to an
        orthonormal camera frame via Gram–Schmidt.

        The 6D vector encodes the first two columns of the rotation matrix::

            rot_6d = [a1_x, a1_y, a1_z,  a2_x, a2_y, a2_z]

        where *a1* is an approximation of the **look** (forward/gaze) axis and
        *a2* is an approximation of the **up** axis.  After ortho-normalisation:

        * ``look  = normalize(a1)``          — forward axis (exact)
        * ``up    = normalize(a2 − (a2·look) look)`` — up axis (orthogonal to look)
        * ``right = cross(look, up)``         — right axis (orthogonal to both)

        Args:
            rot_6d (torch.Tensor): Shape ``(B, 6)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ``(look, up, right)`` each of shape ``(B, 3)``.

        Note:
            Thin wrapper kept for backward compatibility; the implementation
            lives in :func:`frustum.six_d_to_rotation_matrix` so that non-HPRO
            backbones can share it.
        """
        return _frustum.six_d_to_rotation_matrix(rot_6d)

    def compute_frustum_mask(
        self,
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

        Points fully inside the frustum receive scores close to 1; points
        outside or near the boundary receive scores smoothly decaying toward 0.
        Points behind the camera (depth < near ≤ 0) are naturally suppressed
        by the near-plane gate.

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

        Note:
            Thin wrapper kept for backward compatibility; the implementation
            lives in :func:`frustum.compute_frustum_mask` so that non-HPRO
            backbones can share it.
        """
        return _frustum.compute_frustum_mask(
            pts, viewpoint, look, up, right,
            fov_h, fov_v, near, far, sharpness,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        pts: torch.Tensor,
        viewpoint: torch.Tensor,
        rot_6d: torch.Tensor,
        gamma: float,
        fov_h: float = None,
        fov_v: float = None,
        near: float = None,
        far: float = None,
        frustum_sharpness: float = None,
        alphas: list = [],
        k: int = 10,
        delta: float = 0.0,
        use_linear_kernel: bool = False,
    ):
        """
        Forward pass for frustum-constrained differentiable visibility.

        The combined visibility score is::

            w_combined = relu(w_hpro)  ×  f_frustum

        where ``w_hpro`` is the raw HPRO score (ELU output, range (-1, +∞))
        and ``f_frustum`` is the differentiable frustum membership score.
        The ``relu`` clamp is essential: occluded points receive ``w_hpro < 0``
        and must be zeroed out before multiplication so the optimiser cannot
        reduce the loss by moving the frustum *away* from the cloud.
        ``w_combined`` is always ≥ 0 and differentiable with respect to
        ``viewpoint`` and ``rot_6d`` wherever ``w_hpro > 0``.

        Args:
            pts (torch.Tensor): Point cloud, shape ``(B, 3, N)``.
            viewpoint (torch.Tensor): Camera position, shape ``(B, 3)`` or
                ``(B, 3, 1)``.  This is a learnable parameter for optimisation.
            rot_6d (torch.Tensor): 6D camera orientation (Zhou et al. 2019),
                shape ``(B, 6)``, containing the first two (unnormalised)
                columns of the desired rotation matrix ``[a1 | a2]`` where
                *a1* ≈ look direction and *a2* ≈ up direction.  This is a
                learnable parameter for optimisation.
            gamma (float): HPRO radial transformation parameter.  Negative
                values are typical for point clouds (see HPRO docs).
            fov_h (float, optional): Horizontal FOV override in radians.
                Defaults to the value set at construction time.
            fov_v (float, optional): Vertical FOV override in radians.
                Defaults to the value set at construction time.
            near (float, optional): Near-plane distance override.
                Defaults to the value set at construction time.
            far (float, optional): Far-plane distance override.
                Defaults to the value set at construction time.
            frustum_sharpness (float, optional): Sigmoid steepness override.
                Defaults to the value set at construction time.
            alphas (list): Secondary virtual-centre alphas forwarded to HPRO.
            k (int): Top-k parameter forwarded to HPRO.
            delta (float): Noise-offset parameter forwarded to HPRO.
            use_linear_kernel (bool): Use linear instead of power kernel.

        Returns:
            tuple:
                * **visible_pts** (*torch.Tensor*) — coordinates of visible
                  points, shape ``(B, 3, M)`` where ``M ≤ N``.
                * **visible_indices** (*torch.Tensor*) — indices of visible
                  points into the original ``pts`` array, shape ``(M,)``.
                * **w_combined** (*torch.Tensor*) — combined differentiable
                  visibility scores, shape ``(B, N)``.  Use this tensor for
                  gradient-based optimisation.
        """
        # Resolve per-call overrides
        fov_h = fov_h if fov_h is not None else self.fov_h
        fov_v = fov_v if fov_v is not None else self.fov_v
        near = near if near is not None else self.near
        far = far if far is not None else self.far
        sharpness = frustum_sharpness if frustum_sharpness is not None else self.frustum_sharpness

        # Move inputs to device
        pts = pts.to(self.device)
        viewpoint = viewpoint.to(self.device)
        rot_6d = rot_6d.to(self.device)

        batch_size = pts.shape[0]
        n_pts = pts.shape[2]

        # Ensure viewpoint is (B, 3, 1) for broadcasting
        if viewpoint.dim() == 2:
            viewpoint_3d = viewpoint.unsqueeze(2)   # (B, 3, 1)
        else:
            viewpoint_3d = viewpoint                # already (B, 3, 1)

        # ---- HPRO raw scores ----------------------------------------
        centered_points = pts - viewpoint_3d.expand_as(pts)          # (B, 3, N)
        directions = F.normalize(centered_points, dim=1)              # (B, 3, N)

        w = self.hpro.detect_max_in_direction(
            batch_size, centered_points, directions,
            gamma, n_pts, use_linear_kernel,
            alphas=alphas, k=k, delta=delta,
        )  # (B, N) — note: keepdim broadcast inside HPRO may yield (B, 1, N); normalise:
        w = w.reshape(batch_size, n_pts)  # (B, N)

        # ---- Frustum mask -------------------------------------------
        look, up, right = self._6d_to_rotation_matrix(rot_6d)        # each (B, 3)

        f = self.compute_frustum_mask(
            pts, viewpoint_3d, look, up, right,
            fov_h, fov_v, near, far, sharpness,
        )  # (B, N)

        # ---- Combine ------------------------------------------------
        # Clamp w to non-negative before multiplying: ELU gives w ∈ (-1, +∞),
        # so occluded points (w < 0) must not contribute negative values that
        # the optimiser could exploit by moving the frustum away from the cloud.
        w_combined = torch.clamp(w, min=0.0) * f   # (B, N) — differentiable w.r.t. viewpoint & rot_6d

        # ---- Threshold to get visible set ---------------------------
        w1 = w_combined > self.visibility_score_thresh    # (B, N) bool

        if batch_size == 1:
            mask = w1[0]                                            # (N,)
            visible_pts = pts[:, :, mask].to(self.device)          # (1, 3, M)
            visible_indices = torch.nonzero(mask).squeeze(-1).to(self.device)  # (M,)
        else:
            # For batched input, return per-viewpoint lists: a boolean mask of
            # shape (B, N) cannot index a single tensor axis, and different
            # viewpoints see different numbers of points. ``w_combined`` (the
            # differentiable score used for optimisation) is always (B, N).
            visible_pts = [pts[b, :, w1[b]].to(self.device) for b in range(batch_size)]
            visible_indices = [torch.nonzero(w1[b]).squeeze(-1).to(self.device)
                               for b in range(batch_size)]

        return visible_pts, visible_indices, w_combined
