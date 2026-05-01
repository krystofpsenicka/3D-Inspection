"""Low-level USD prim builders for Isaac Sim visualization.

Every function takes ``(stage, path, ...)`` and returns the prim path string.
All ``pxr`` / ``omni`` imports are lazy (inside function bodies) so the module
can be imported without Isaac Sim running.
"""

from __future__ import annotations

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


def create_mesh_prim(
    stage,
    path: str,
    mesh: trimesh.Trimesh,
    color: tuple = (0.8, 0.8, 0.8),
    opacity: float = 1.0,
) -> str:
    """Create a ``UsdGeom.Mesh`` prim from a trimesh.

    Parameters
    ----------
    stage : Usd.Stage
    path : USD prim path.
    mesh : Source triangle mesh.
    color : Uniform (r, g, b) display colour.
    opacity : 0-1 opacity.

    Returns
    -------
    The prim path string.
    """
    from pxr import Gf, UsdGeom, Vt

    usd_mesh = UsdGeom.Mesh.Define(stage, path)

    vertices = mesh.vertices.astype(np.float64)
    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vertices]))

    faces = mesh.faces.astype(int)
    usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
        normals = mesh.vertex_normals.astype(np.float64)
        usd_mesh.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
        usd_mesh.SetNormalsInterpolation("vertex")

    usd_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    if opacity < 1.0:
        usd_mesh.GetDisplayOpacityAttr().Set(Vt.FloatArray([opacity]))

    return path


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


def create_points_prim(
    stage,
    path: str,
    positions: np.ndarray,
    colors: np.ndarray | tuple | None = None,
    point_size: float = 4.0,
) -> str:
    """Create a ``UsdGeom.Points`` prim.

    Parameters
    ----------
    stage : Usd.Stage
    path : USD prim path.
    positions : (N, 3) array of point positions.
    colors : (N, 3) per-point colours, or a single (r, g, b) tuple for
        uniform colour.  ``None`` defaults to white.
    point_size : Display width of each point.

    Returns
    -------
    The prim path string.
    """
    from pxr import Gf, UsdGeom, Vt

    pts_prim = UsdGeom.Points.Define(stage, path)

    positions = np.asarray(positions, dtype=np.float64)
    pts_prim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in positions]))

    widths = Vt.FloatArray([point_size] * len(positions))
    pts_prim.GetWidthsAttr().Set(widths)

    if colors is None:
        colors_arr = Vt.Vec3fArray([Gf.Vec3f(1, 1, 1)] * len(positions))
    elif (
        isinstance(colors, (tuple, list)) and len(colors) == 3 and not hasattr(colors[0], "__len__")
    ):
        colors_arr = Vt.Vec3fArray([Gf.Vec3f(*colors)] * len(positions))
    else:
        colors = np.asarray(colors, dtype=np.float64)
        colors_arr = Vt.Vec3fArray([Gf.Vec3f(*c) for c in colors])

    pts_prim.GetDisplayColorAttr().Set(colors_arr)

    return path


# ---------------------------------------------------------------------------
# LineSet (BasisCurves)
# ---------------------------------------------------------------------------


def create_lineset_prim(
    stage,
    path: str,
    points: np.ndarray,
    lines: np.ndarray,
    color: tuple = (1.0, 1.0, 1.0),
    width: float = 2.0,
) -> str:
    """Create a ``UsdGeom.BasisCurves`` prim for line segments.

    Parameters
    ----------
    stage : Usd.Stage
    path : USD prim path.
    points : (P, 3) array of vertices referenced by *lines*.
    lines : (L, 2) array of index pairs  --  each row is one segment.
    color : Uniform (r, g, b) colour.
    width : Line display width.

    Returns
    -------
    The prim path string.
    """
    from pxr import Gf, UsdGeom, Vt

    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.GetTypeAttr().Set("linear")

    lines = np.asarray(lines, dtype=int)
    points = np.asarray(points, dtype=np.float64)

    # Flatten: for each segment, emit the two endpoint positions.
    seg_pts = []
    for i0, i1 in lines:
        seg_pts.append(Gf.Vec3f(*points[i0]))
        seg_pts.append(Gf.Vec3f(*points[i1]))

    curves.GetPointsAttr().Set(Vt.Vec3fArray(seg_pts))
    curves.GetCurveVertexCountsAttr().Set(Vt.IntArray([2] * len(lines)))

    curves.GetWidthsAttr().Set(Vt.FloatArray([width] * len(seg_pts)))

    curves.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    return path


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------


def create_sphere_prim(
    stage,
    path: str,
    position: np.ndarray,
    radius: float = 0.15,
    color: tuple = (1.0, 1.0, 0.0),
) -> str:
    """Create a visual sphere prim.

    Tries ``omni.isaac.core.objects.VisualSphere`` first; falls back to raw
    ``UsdGeom.Sphere`` with a translate xformOp.

    Returns
    -------
    The prim path string.
    """
    position = np.asarray(position, dtype=np.float64)
    try:
        from omni.isaac.core.objects import VisualSphere

        VisualSphere(
            path,
            position=position,
            radius=radius,
            color=np.array(color, dtype=np.float32),
        )
    except Exception:
        from pxr import Gf, UsdGeom

        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.GetRadiusAttr().Set(float(radius))
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))

    return path


# ---------------------------------------------------------------------------
# Coordinate frame (3 coloured axis lines)
# ---------------------------------------------------------------------------


def create_coordinate_frame_prim(
    stage,
    path: str,
    size: float = 2.0,
) -> str:
    """Create a coordinate-frame indicator (RGB = XYZ axes) under *path*.

    Returns
    -------
    The parent prim path.
    """
    origin = np.zeros(3)
    axes = [
        ("X", np.array([size, 0, 0]), (1, 0, 0)),
        ("Y", np.array([0, size, 0]), (0, 1, 0)),
        ("Z", np.array([0, 0, size]), (0, 0, 1)),
    ]
    for name, endpoint, color in axes:
        create_lineset_prim(
            stage,
            f"{path}/{name}",
            points=np.array([origin, endpoint]),
            lines=np.array([[0, 1]]),
            color=color,
            width=3.0,
        )
    return path


# ---------------------------------------------------------------------------
# Wireframe from trimesh
# ---------------------------------------------------------------------------


def set_prim_pose(prim, xyz: np.ndarray, qwxyz: np.ndarray) -> None:
    """Teleport a USD prim by setting xformOp:translate and xformOp:orient.

    Parameters
    ----------
    prim : Usd.Prim
    xyz : (3,) position array.
    qwxyz : (4,) quaternion [w, x, y, z].
    """
    from pxr import Gf

    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    qw, qx, qy, qz = float(qwxyz[0]), float(qwxyz[1]), float(qwxyz[2]), float(qwxyz[3])

    translate_attr = prim.GetAttribute("xformOp:translate")
    orient_attr = prim.GetAttribute("xformOp:orient")

    if translate_attr and translate_attr.IsValid():
        try:
            translate_attr.Set(Gf.Vec3f(x, y, z))
        except Exception:
            try:
                translate_attr.Set(Gf.Vec3d(x, y, z))
            except Exception:
                pass
    else:
        from pxr import UsdGeom

        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3f(x, y, z))

    if orient_attr and orient_attr.IsValid():
        try:
            orient_attr.Set(Gf.Quatf(qw, qx, qy, qz))
        except Exception:
            try:
                orient_attr.Set(Gf.Quatd(qw, qx, qy, qz))
            except Exception:
                pass
    else:
        from pxr import UsdGeom

        UsdGeom.Xformable(prim).AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))


# ---------------------------------------------------------------------------
# Cuboid
# ---------------------------------------------------------------------------


def create_cuboid_prim(
    stage,
    path: str,
    position: np.ndarray,
    orientation: np.ndarray,
    color: tuple = (1.0, 1.0, 1.0),
    size: float = 0.1,
) -> str:
    """Create a visual cuboid prim.

    Tries ``omni.isaac.core.objects.cuboid.VisualCuboid`` first; falls back
    to raw ``UsdGeom.Cube`` with xformOps.

    Returns
    -------
    The prim path string.
    """
    position = np.asarray(position, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    try:
        from omni.isaac.core.objects import cuboid as cuboid_mod

        cuboid_mod.VisualCuboid(
            path,
            position=position,
            orientation=orientation,
            color=np.array(color, dtype=np.float32),
            size=float(size),
        )
    except Exception:
        from pxr import Gf, UsdGeom

        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(float(size))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))
        if len(orientation) == 4:
            qw, qx, qy, qz = orientation
            xformable.AddOrientOp().Set(Gf.Quatd(float(qw), float(qx), float(qy), float(qz)))

    return path


# ---------------------------------------------------------------------------
# Wireframe from trimesh
# ---------------------------------------------------------------------------


def create_wireframe_from_trimesh(
    stage,
    path: str,
    mesh: trimesh.Trimesh,
    color: tuple = (0.7, 0.7, 0.7),
    width: float = 1.0,
) -> str:
    """Create a wireframe ``BasisCurves`` prim from the unique edges of a trimesh.

    Returns
    -------
    The prim path string.
    """
    edges = mesh.edges_unique
    return create_lineset_prim(
        stage,
        path,
        points=mesh.vertices,
        lines=edges,
        color=color,
        width=width,
    )
