"""Shared ground-truth and frustum-geometry helpers for the frustum-restricted
HPRO operator (HPRO_limited).

These functions were originally inlined in ``demo_limited.py``; they are factored
here so both the demo and the quantitative evaluation harness (``eval_frustum.py``)
use exactly the same camera-frame convention, hard frustum test, and ray-cast
ground truth.

Conventions (matching ``HPRO_limited._6d_to_rotation_matrix``):
    * ``look``  — unit forward / gaze axis (normalised ``a1``).
    * ``up``    — unit up axis, orthogonalised against ``look``.
    * ``right`` = ``cross(look, up)``.
    * Horizontal angle uses ``right``; vertical angle uses ``up``; depth uses ``look``.
"""

import math

import numpy as np
import trimesh

from GroundTruthGenerator import GroundTruthGenerator


# ---------------------------------------------------------------------------
# Camera frame
# ---------------------------------------------------------------------------

def build_camera_frame(look_approx_np, up_approx_np):
    """
    Return an orthonormal (look, up, right) frame via Gram-Schmidt, mirroring
    the logic inside ``HPRO_limited._6d_to_rotation_matrix``.

    Args:
        look_approx_np: (3,) approximate look (forward/gaze) direction.
        up_approx_np:   (3,) approximate up direction.

    Returns:
        (look, up, right) — each a unit 1-D numpy array of shape (3,).
    """
    look  = look_approx_np / np.linalg.norm(look_approx_np)
    up    = up_approx_np - np.dot(up_approx_np, look) * look
    up    = up / np.linalg.norm(up)
    right = np.cross(look, up)
    return look, up, right


# ---------------------------------------------------------------------------
# Hard (exact) frustum test
# ---------------------------------------------------------------------------

def points_inside_frustum(pts_np, viewpoint_np, look, up, right,
                          fov_h, fov_v, near, far):
    """
    Exact (hard) geometric frustum test — no sigmoid approximation.

    Args:
        pts_np      : (N, 3) point positions.
        viewpoint_np: (3,)   camera position.
        look/up/right: orthonormal camera-frame axes, each (3,).
        fov_h, fov_v: horizontal / vertical FOV in radians.
        near, far   : depth clip distances.

    Returns:
        Bool array (N,) — True if the point is inside the frustum.
    """
    v = pts_np - viewpoint_np[np.newaxis, :]

    depth   = v @ look
    h_coord = v @ right
    v_coord = v @ up

    tan_h_lim = math.tan(fov_h / 2.0)
    tan_v_lim = math.tan(fov_v / 2.0)

    eps      = 1e-8
    in_depth = (depth > near) & (depth < far)
    in_h     = np.abs(h_coord) < tan_h_lim * (depth + eps)
    in_v     = np.abs(v_coord) < tan_v_lim * (depth + eps)

    return in_depth & in_h & in_v


# ---------------------------------------------------------------------------
# Ray-cast ground truth (frustum membership + occlusion)
# ---------------------------------------------------------------------------

def compute_ground_truth(mesh, viewpoint_np, pts_np,
                         look, up, right,
                         fov_h, fov_v, near, far,
                         intersector=None):
    """
    Ground-truth: points inside the frustum AND not occluded (ray-cast vs. the
    mesh). Ray-cast runs only on the frustum subset to save time.

    Args:
        mesh        : trimesh.Trimesh the points were sampled from.
        viewpoint_np: (3,) camera position.
        pts_np      : (N, 3) sampled surface points.
        look/up/right, fov_h, fov_v, near, far: frustum definition.
        intersector : optional pre-built RayMeshIntersector (reused across poses
                      on the same mesh to avoid rebuilding the BVH every call).

    Returns:
        gt_indices      — list[int], frustum-visible AND unoccluded indices.
        frustum_indices — list[int], all indices inside the frustum (pre-occlusion).
    """
    in_frustum      = points_inside_frustum(
        pts_np, viewpoint_np, look, up, right, fov_h, fov_v, near, far
    )
    frustum_indices = list(np.where(in_frustum)[0])

    if intersector is None:
        # mesh.ray auto-selects the Embree BVH backend (ray_pyembree) when
        # embreex is installed, falling back to the pure-Python ray_triangle.
        intersector = mesh.ray

    gt_indices = [
        idx for idx in frustum_indices
        if GroundTruthGenerator.singleRayIntersection(
            idx, intersector, viewpoint_np, pts_np
        )
    ]
    return gt_indices, frustum_indices


def compute_ground_truth_batched(mesh, viewpoint_np, pts_np,
                                 look, up, right,
                                 fov_h, fov_v, near, far,
                                 intersector=None, tol=1e-6):
    """
    Vectorised equivalent of :func:`compute_ground_truth`.

    Casts **all** frustum rays in a single ``intersects_location`` call instead
    of one Python call per point, then marks a point visible iff the nearest
    ray–mesh hit coincides with the point itself (``< tol``). This matches the
    semantics of :meth:`GroundTruthGenerator.singleRayIntersection` exactly but
    is orders of magnitude faster for dense clouds / many meshes — essential for
    multi-mesh evaluation. Falls back to no-hit = not-visible.

    Returns:
        gt_indices, frustum_indices  (both list[int]).
    """
    in_frustum = points_inside_frustum(
        pts_np, viewpoint_np, look, up, right, fov_h, fov_v, near, far
    )
    frustum_indices = np.where(in_frustum)[0]
    if len(frustum_indices) == 0:
        return [], []

    if intersector is None:
        # mesh.ray auto-selects the Embree BVH backend (ray_pyembree) when
        # embreex is installed, falling back to the pure-Python ray_triangle.
        intersector = mesh.ray

    targets = pts_np[frustum_indices]                       # (M, 3)
    origins = np.repeat(viewpoint_np[np.newaxis], len(frustum_indices), axis=0)
    directions = targets - viewpoint_np[np.newaxis]         # (M, 3), unnormalised

    locs, idx_ray, _ = intersector.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )

    visible = np.zeros(len(frustum_indices), dtype=bool)
    if len(idx_ray) > 0:
        # Nearest hit per ray = first after sorting by (ray, distance-to-viewpoint).
        d_hit = np.linalg.norm(locs - viewpoint_np[np.newaxis], axis=1)
        order = np.lexsort((d_hit, idx_ray))
        rays_s = idx_ray[order]
        locs_s = locs[order]
        first = np.ones(len(rays_s), dtype=bool)
        first[1:] = rays_s[1:] != rays_s[:-1]
        nearest_ray = rays_s[first]                         # ray (local) indices
        nearest_loc = locs_s[first]                         # their nearest hit
        # Visible iff that nearest hit is the target point itself.
        dd = np.linalg.norm(nearest_loc - targets[nearest_ray], axis=1)
        visible[nearest_ray[dd < tol]] = True

    gt_indices = list(frustum_indices[visible])
    return gt_indices, list(frustum_indices)


# ---------------------------------------------------------------------------
# Frustum wireframe helpers (visualisation only)
# ---------------------------------------------------------------------------

def frustum_corners(viewpoint_np, look, up, right, fov_h, fov_v, near, far):
    """
    Compute the 8 corners of the rectangular frustum.

    Near/far plane corner order:
        0: (+h, +v)  top-right
        1: (-h, +v)  top-left
        2: (-h, -v)  bottom-left
        3: (+h, -v)  bottom-right

    Returns:
        near_corners : (4, 3) numpy array
        far_corners  : (4, 3) numpy array
    """
    th = math.tan(fov_h / 2.0)
    tv = math.tan(fov_v / 2.0)

    offsets = np.array([
        [ th,  tv],
        [-th,  tv],
        [-th, -tv],
        [ th, -tv],
    ])  # (4, 2)  — (h_scale, v_scale)

    def plane_corners(dist):
        centre = viewpoint_np + dist * look
        return np.array([
            centre + dist * (h * right + v * up)
            for h, v in offsets
        ])  # (4, 3)

    return plane_corners(near), plane_corners(far)


def draw_frustum(ax, viewpoint_np, look, up, right,
                 fov_h, fov_v, near, far,
                 color='orange', linewidth=1.2, alpha=0.85):
    """
    Draw a frustum wireframe on a Matplotlib 3D axis using individual plot3D
    calls (one per edge). Using plot3D — rather than Line3DCollection — ensures
    that the frustum corner world coordinates are registered in the axis data
    limits, which is required for ``set_axes_equal_3d`` to produce correct
    equal-scale rendering of tilted frustums.

    Draws:
      - 4 edges of the near rectangle
      - 4 edges of the far  rectangle
      - 4 lateral edges: viewpoint → near corner → far corner
    """
    nc, fc = frustum_corners(viewpoint_np, look, up, right, fov_h, fov_v, near, far)

    def seg(p0, p1):
        ax.plot3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                  color=color, linewidth=linewidth, alpha=alpha)

    # Near-plane rectangle
    for i in range(4):
        seg(nc[i], nc[(i + 1) % 4])

    # Far-plane rectangle
    for i in range(4):
        seg(fc[i], fc[(i + 1) % 4])

    # Lateral edges: viewpoint → near corner → far corner
    for i in range(4):
        seg(viewpoint_np, nc[i])
        seg(nc[i], fc[i])


def set_axes_equal_3d(ax):
    """Set equal data-range on all three axes so tilted frustums are not distorted."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
