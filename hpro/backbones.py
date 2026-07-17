"""Interchangeable differentiable occlusion backbones.

A *backbone* answers one question, differentiably: **given a camera position,
how visible is each point of the cloud, ignoring the camera's field of view?**
Everything that describes the camera rather than the occlusion -- the frustum,
and later the normal gate and measurement-quality weighting -- lives in
``frustum.py`` / ``visibility_layer.py`` and is shared by every backbone.

That split is what makes the backbones comparable: §3 of ``RESEARCH_PLAN.md``
shows analytic HPRO and neural NVPS failing in complementary ways (HPRO wins on
thin non-watertight meshes, NVPS wins on scanned concave ones), and the only way
to make that a controlled comparison is to hold the camera model fixed while
swapping the occlusion model.

Backbones
---------
* :class:`HPROBackbone` -- analytic, training-free, O(N²) memory, works on any
  point set. Score is the ELU output of the HPRO operator, clamped to >= 0.
* :class:`NVPSBackbone` -- pretrained octree U-Net (Wang et al., SIGGRAPH Asia
  2025). O(N) memory, per-cloud feature extraction is a one-time cost, but the
  score is *distance-invariant* by construction (see §4 of the plan): it depends
  only on the unit direction from point to camera, so all distance sensitivity
  must come from the analytic terms layered on top.
* :class:`EnsembleBackbone` -- elementwise max over backbones, to cover both
  failure classes at once.

Example::

    from backbones import make_backbone
    from visibility_layer import GatedVisibilityLayer

    backbone = make_backbone("nvps", device="cuda")
    backbone.prepare(pts_np, normals_np)          # one-time per cloud
    layer = GatedVisibilityLayer(backbone, fov_h=..., fov_v=..., near=..., far=...)
    w = layer(pts_t, viewpoints, rot_6d)          # (V, N), differentiable
"""

import math
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import ocnn_compat  # noqa: F401,E402  -- must precede any ocnn import; see module docstring
from HPRO import HPRO  # noqa: E402


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class VisibilityBackbone(nn.Module, ABC):
    """Differentiable, frustum-agnostic occlusion model.

    Subclasses must return per-(viewpoint, point) visibility scores that are
    differentiable w.r.t. the viewpoint positions.  Scores are >= 0 and are
    interpreted as "1 = certainly visible, 0 = certainly occluded"; they are
    not required to be calibrated probabilities (they are not, for HPRO -- see
    C2 in ``RESEARCH_PLAN.md``).

    Attributes:
        name (str): Short identifier used in logs and CSV rows.
        requires_normals (bool): Whether :meth:`prepare` needs point normals.
    """

    name: str = "base"
    requires_normals: bool = False

    def prepare(self, pts_np: np.ndarray, normals_np: Optional[np.ndarray] = None) -> None:
        """Precompute any view-independent state for this cloud (one-time).

        Args:
            pts_np: (N, 3) surface points.
            normals_np: (N, 3) normals, required iff ``requires_normals``.
        """
        return None

    @abstractmethod
    def forward(self, pts_t: torch.Tensor, viewpoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts_t: (3, N) point cloud on the compute device.
            viewpoints: (V, 3) camera positions (differentiable).

        Returns:
            (V, N) visibility scores, >= 0, differentiable w.r.t. ``viewpoints``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Analytic HPRO
# ---------------------------------------------------------------------------

class HPROBackbone(VisibilityBackbone):
    """Analytic HPRO operator (Katz & Tal, CGF 2025) as an occlusion backbone.

    Training-free and applicable to any point set, but the score is *not*
    calibrated: raising the score by moving the camera closer is possible, which
    is the mechanism behind the standoff collapse documented in §3.2 of the
    plan.  Memory is O(V·N²) with ``fits_in_memory=True``.

    Args:
        gamma: HPRO radial-transform parameter (negative, close to 0).
        k: top-k parameter for the ELU score (paper default 10).
        alphas: secondary virtual-centre alphas (Eq. 8); ``[]`` disables.
        delta: noise offset (Eq. 7); ``0.0`` disables.
        fits_in_memory: parallel (True) vs low-memory loop (False).
        use_linear_kernel: linear instead of power kernel.
        device: compute device.
    """

    name = "hpro"
    requires_normals = False

    def __init__(
        self,
        gamma: float = -math.exp(-7.0),
        k: int = 10,
        alphas: Sequence[float] = (),
        delta: float = 0.0,
        fits_in_memory: bool = True,
        use_linear_kernel: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.gamma = gamma
        self.k = k
        self.alphas = list(alphas)
        self.delta = delta
        self.use_linear_kernel = use_linear_kernel
        self.hpro = HPRO(fits_in_memory=fits_in_memory, device=device)

    def forward(self, pts_t: torch.Tensor, viewpoints: torch.Tensor) -> torch.Tensor:
        V = viewpoints.shape[0]
        N = pts_t.shape[1]

        # (V, 3, N) -- .expand shares storage; the subtraction below makes it
        # contiguous per viewpoint.
        pts_b = pts_t.unsqueeze(0).expand(V, -1, -1)
        centered = pts_b - viewpoints.unsqueeze(2)              # (V, 3, N)
        directions = torch.nn.functional.normalize(centered, dim=1)

        # k must not exceed the cloud size — tiny clouds occur legitimately in
        # the no-prior online setting, where the belief starts near-empty.
        w = self.hpro.detect_max_in_direction(
            V, centered, directions, self.gamma, N, self.use_linear_kernel,
            alphas=self.alphas, k=min(self.k, N), delta=self.delta,
        ).reshape(V, N)

        # ELU output is in (-1, +inf); occluded points score < 0 and must be
        # clamped, otherwise the optimiser can lower the loss by pushing points
        # further into "very occluded" territory rather than by seeing them.
        return torch.clamp(w, min=0.0)


# ---------------------------------------------------------------------------
# Neural NVPS
# ---------------------------------------------------------------------------

class NVPSBackbone(VisibilityBackbone):
    """Pretrained Neural Visibility of Point Sets (Wang et al., SIGGRAPH Asia 2025).

    An octree U-Net produces a view-independent 63-d feature per point once per
    cloud (:meth:`prepare`, ~30 s at N=3000 on a 3090); per view, the feature is
    multiplied elementwise with a positional encoding of the unit direction
    point->camera and a 3-layer MLP outputs visible/invisible logits (class 1 =
    visible).

    Because the MLP only ever sees the *unit* direction, the score is invariant
    to camera distance along a ray.  That is a shield (the optimiser cannot
    inflate the score by approaching) and a blindness (genuine distance-dependent
    visibility is not modelled) -- see §4 of the plan.

    Requires the external assets (git-ignored); override via ``NEUVIS_DIR`` /
    ``NEUVIS_CKPT``.

    Args:
        nv_dir: clone of octree-nn/neural-visibility.
        ckpt: path to the pretrained checkpoint.
        device: compute device.
    """

    name = "nvps"
    requires_normals = True

    def __init__(
        self,
        nv_dir: Optional[str] = None,
        ckpt: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.nv_dir = nv_dir or os.environ.get(
            "NEUVIS_DIR", os.path.join(_DIR, "external", "neural-visibility"))
        self.ckpt_path = ckpt or os.environ.get(
            "NEUVIS_CKPT", os.path.join(_DIR, "external", "neuvis_00040.pth"))

        # Imported lazily and kept module-local: ocnn is only needed for this
        # backbone, and the HPRO path must stay usable without it.
        if self.nv_dir not in sys.path:
            sys.path.insert(0, self.nv_dir)
        try:
            from models import MyNet, get_embedder
        except ImportError as e:                                  # pragma: no cover
            raise ImportError(
                f"NVPS backbone needs the neural-visibility clone at {self.nv_dir!r} "
                f"and the ocnn package. See RESEARCH_PLAN.md §1 for fetch commands."
            ) from e

        net = MyNet(6, 63, 2).to(device)
        state = torch.load(self.ckpt_path, map_location=device, weights_only=True)
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        net.load_state_dict(state)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.embedder, _ = get_embedder(10)
        self.feature: Optional[torch.Tensor] = None   # (N, 63), set by prepare()

    @staticmethod
    def _input_feature(octree):
        """Octree input feature 'LP': local coords + global position (6 ch)."""
        depth = octree.depth
        local_points = octree.points[depth].frac() - 0.5
        scale = 2 ** (1 - depth)
        global_points = octree.points[depth] * scale - 1.0
        return torch.cat([local_points, global_points], dim=1)

    def prepare(self, pts_np: np.ndarray, normals_np: Optional[np.ndarray] = None) -> None:
        if normals_np is None:
            raise ValueError("NVPSBackbone.prepare() requires normals_np")
        from ocnn.octree import Octree, Points

        xyz = torch.tensor(pts_np, dtype=torch.float32, device=self.device)
        nrm = torch.tensor(normals_np, dtype=torch.float32, device=self.device)
        points = Points(xyz.clone(), nrm.clone())
        bbmin, bbmax = points.bbox()
        # The network was trained on clouds normalised into a [-0.8, 0.8] box;
        # feeding it any other scale is out of distribution.
        points.normalize(bbmin, bbmax, scale=0.8)
        octree = Octree(8, 2, device=self.device)
        octree.build_octree(points)
        octree.construct_all_neigh()
        feat = self._input_feature(octree)
        bid = torch.zeros(points.points.shape[0], 1, device=self.device)
        qp = torch.cat([points.points, bid], dim=1)
        with torch.no_grad():
            self.feature = self.net.UNet(feat, octree, octree.depth, qp)   # (N, 63)

    def forward(self, pts_t: torch.Tensor, viewpoints: torch.Tensor) -> torch.Tensor:
        if self.feature is None:
            raise RuntimeError(
                "NVPSBackbone.prepare(pts_np, normals_np) must be called before forward()"
            )
        V = viewpoints.shape[0]
        N = pts_t.shape[1]
        if self.feature.shape[0] != N:
            raise ValueError(
                f"prepare() was called with {self.feature.shape[0]} points but "
                f"forward() received {N}; features are cloud-specific."
            )

        # Unit direction from each point to each viewpoint. Gradients reach the
        # viewpoint only through this direction -- hence distance-invariance.
        vd = pts_t.T.unsqueeze(0) - viewpoints.unsqueeze(1)      # (V, N, 3)
        vd = vd / vd.norm(dim=2, keepdim=True)
        vd_e = self.embedder(vd.reshape(-1, 3))                  # (V*N, 63)
        feat = self.feature.unsqueeze(0).expand(V, -1, -1).reshape(-1, 63)
        logits = self.net.VisNet(feat * vd_e).view(-1, 2)
        return torch.softmax(logits, dim=1)[:, 1].view(V, N)     # class 1 = visible


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleBackbone(VisibilityBackbone):
    """Elementwise max over several backbones.

    Motivated by the complementary failure modes in §3.1: max is the natural
    combiner when each member is confidently right in its own regime and
    under-confident (not wrongly confident) outside it.  Note the max inherits
    each member's *optimistic* bias, so an ensemble is not a calibration fix --
    see C2.

    Args:
        backbones: members to combine.
    """

    name = "ensemble"

    def __init__(self, backbones: Sequence[VisibilityBackbone]):
        super().__init__()
        if not backbones:
            raise ValueError("EnsembleBackbone needs at least one member")
        self.members = nn.ModuleList(backbones)
        self.name = "+".join(b.name for b in backbones)
        self.requires_normals = any(b.requires_normals for b in backbones)

    def prepare(self, pts_np: np.ndarray, normals_np: Optional[np.ndarray] = None) -> None:
        for b in self.members:
            b.prepare(pts_np, normals_np)

    def forward(self, pts_t: torch.Tensor, viewpoints: torch.Tensor) -> torch.Tensor:
        out = self.members[0](pts_t, viewpoints)
        for b in self.members[1:]:
            out = torch.max(out, b(pts_t, viewpoints))
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Constructor keys understood by each backbone. ``make_backbone`` routes a
#: single flat config to the right members using these, so that a sweep can do
#: ``make_backbone(name, **cfg)`` for every name with one unchanged ``cfg``.
HPRO_KEYS = ("gamma", "k", "alphas", "delta", "fits_in_memory", "use_linear_kernel")
NVPS_KEYS = ("nv_dir", "ckpt")


def make_backbone(name: str, device: Optional[str] = None, **kwargs) -> VisibilityBackbone:
    """Construct a backbone by name: ``'hpro'``, ``'nvps'`` or ``'ensemble'``.

    Keyword arguments are *filtered per backbone*: passing an HPRO-only key such
    as ``gamma`` to the NVPS backbone is silently ignored rather than raising, so
    that a single config dict can drive a sweep across all three backbones.
    Keys understood by no backbone still raise, to catch typos.

    Args:
        name: backbone identifier (case-insensitive).
        device: compute device.
        **kwargs: backbone options; see :data:`HPRO_KEYS` and :data:`NVPS_KEYS`.

    Returns:
        The constructed backbone.

    Raises:
        TypeError: if a key is understood by no backbone.
        ValueError: if ``name`` is not a known backbone.
    """
    name = name.lower()
    unknown = set(kwargs) - set(HPRO_KEYS) - set(NVPS_KEYS)
    if unknown:
        raise TypeError(
            f"make_backbone({name!r}) got kwargs no backbone understands: "
            f"{sorted(unknown)}"
        )
    hpro_kw = {k: v for k, v in kwargs.items() if k in HPRO_KEYS}
    nvps_kw = {k: v for k, v in kwargs.items() if k in NVPS_KEYS}

    if name == "hpro":
        return HPROBackbone(device=device, **hpro_kw)
    if name == "nvps":
        return NVPSBackbone(device=device, **nvps_kw)
    if name == "ensemble":
        return EnsembleBackbone([
            HPROBackbone(device=device, **hpro_kw),
            NVPSBackbone(device=device, **nvps_kw),
        ])
    raise ValueError(f"unknown backbone {name!r}; expected 'hpro', 'nvps' or 'ensemble'")
