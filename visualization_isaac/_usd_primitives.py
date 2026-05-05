"""Low-level USD prim builders for Isaac Sim visualization.

Every function takes ``(stage, path, ...)`` and returns the prim path string. All ``pxr`` /
``omni`` imports are lazy so the module can be imported without Isaac Sim running.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _vec3f_array(arr):
    """Build a Vt.Vec3fArray in one C++ copy. The legacy per-row Gf.Vec3f path blocks Kit's main
    loop for several seconds on 100k+ row meshes/point-clouds."""
    from pxr import Vt

    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    if hasattr(Vt.Vec3fArray, "FromNumpy"):
        return Vt.Vec3fArray.FromNumpy(arr)
    return Vt.Vec3fArray(arr)


def _float_array(arr):
    from pxr import Vt

    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    if hasattr(Vt.FloatArray, "FromNumpy"):
        return Vt.FloatArray.FromNumpy(arr)
    return Vt.FloatArray(arr)


def _int_array(arr):
    from pxr import Vt

    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.int32))
    if hasattr(Vt.IntArray, "FromNumpy"):
        return Vt.IntArray.FromNumpy(arr)
    return Vt.IntArray(arr)


def create_mesh_prim(
    stage,
    path: str,
    mesh: trimesh.Trimesh,
    color: tuple = (0.55, 0.55, 0.55),
    opacity: float = 1.0,
) -> str:
    from pxr import Gf, UsdGeom, Vt

    usd_mesh = UsdGeom.Mesh.Define(stage, path)

    vertices = np.asarray(mesh.vertices)
    usd_mesh.GetPointsAttr().Set(_vec3f_array(vertices))

    faces = np.asarray(mesh.faces, dtype=np.int32)
    usd_mesh.GetFaceVertexCountsAttr().Set(_int_array(np.full(len(faces), 3, dtype=np.int32)))
    usd_mesh.GetFaceVertexIndicesAttr().Set(_int_array(faces.reshape(-1)))

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
        usd_mesh.GetNormalsAttr().Set(_vec3f_array(mesh.vertex_normals))
        usd_mesh.SetNormalsInterpolation("vertex")

    usd_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    if opacity < 1.0:
        usd_mesh.GetDisplayOpacityAttr().Set(Vt.FloatArray([opacity]))

    return path


def create_points_prim(
    stage,
    path: str,
    positions: np.ndarray,
    colors: np.ndarray | tuple | None = None,
    point_size: float = 0.02,
) -> str:
    """``colors``: (N,3) per-point or (r,g,b) uniform; ``None`` defaults to white."""
    from pxr import UsdGeom, Vt

    pts_prim = UsdGeom.Points.Define(stage, path)

    positions = np.asarray(positions)
    n = len(positions)
    pts_prim.GetPointsAttr().Set(_vec3f_array(positions))

    pts_prim.GetWidthsAttr().Set(_float_array(np.full(n, point_size, dtype=np.float32)))

    if colors is None:
        colors_np = np.broadcast_to(np.array([1.0, 1.0, 1.0], dtype=np.float32), (n, 3))
    elif (
        isinstance(colors, (tuple, list)) and len(colors) == 3 and not hasattr(colors[0], "__len__")
    ):
        colors_np = np.broadcast_to(np.asarray(colors, dtype=np.float32), (n, 3))
    else:
        colors_np = np.asarray(colors, dtype=np.float32)

    pts_prim.GetDisplayColorAttr().Set(_vec3f_array(colors_np))
    pts_prim.GetDisplayOpacityAttr().Set(Vt.FloatArray([1.0]))

    return path


def create_lineset_prim(
    stage,
    path: str,
    points: np.ndarray,
    lines: np.ndarray,
    color: tuple = (1.0, 1.0, 1.0),
    width: float = 0.02,
) -> str:
    """BasisCurves prim for line segments. ``lines``: (L,2) index pairs into ``points``."""
    from pxr import Gf, UsdGeom, Vt

    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.GetTypeAttr().Set("linear")

    lines = np.asarray(lines, dtype=np.int32)
    points = np.asarray(points)

    seg_pts = points[lines.reshape(-1)]
    n_seg_pts = len(seg_pts)

    curves.GetPointsAttr().Set(_vec3f_array(seg_pts))
    curves.GetCurveVertexCountsAttr().Set(_int_array(np.full(len(lines), 2, dtype=np.int32)))

    curves.GetWidthsAttr().Set(_float_array(np.full(n_seg_pts, width, dtype=np.float32)))
    curves.GetWidthsAttr().SetMetadata("interpolation", "vertex")

    curves.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    # Force opaque so RTX doesn't apply alpha-coverage softening at the silhouette
    # (the source of the "halo through mesh" look).
    curves.GetDisplayOpacityAttr().Set(Vt.FloatArray([1.0]))

    return path


def create_sphere_prim(
    stage,
    path: str,
    position: np.ndarray,
    radius: float = 0.15,
    color: tuple = (1.0, 1.0, 0.0),
) -> str:
    """Tries omni.isaac.core.objects.VisualSphere; falls back to UsdGeom.Sphere."""
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


def create_coordinate_frame_prim(
    stage,
    path: str,
    size: float = 2.0,
) -> str:
    """RGB = XYZ axis indicator."""
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


def set_prim_pose(prim, xyz: np.ndarray, qwxyz: np.ndarray) -> None:
    """Teleport a USD prim by setting xformOp:translate and xformOp:orient."""
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


def create_cuboid_prim(
    stage,
    path: str,
    position: np.ndarray,
    orientation: np.ndarray,
    color: tuple = (1.0, 1.0, 1.0),
    size: float = 0.1,
) -> str:
    """Tries omni.isaac.core.objects.cuboid.VisualCuboid; falls back to UsdGeom.Cube."""
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


def create_wireframe_from_trimesh(
    stage,
    path: str,
    mesh: trimesh.Trimesh,
    color: tuple = (0.7, 0.7, 0.7),
    width: float = 0.003,
) -> str:
    edges = mesh.edges_unique
    return create_lineset_prim(
        stage,
        path,
        points=mesh.vertices,
        lines=edges,
        color=color,
        width=width,
    )
