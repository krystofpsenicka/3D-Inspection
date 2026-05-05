"""Isaac Sim application lifecycle helper.

Use this as a context manager to launch ``SimulationApp`` once and obtain a
ready-to-use stage:

    from visualization_isaac import IsaacApp

    with IsaacApp(headless=False) as ctx:
        # build prims via the visualizer classes on ctx.stage ...
        ctx.run_until_quit()

The class defers every ``isaacsim`` / ``omni`` / ``pxr`` import to
``__enter__`` because ``SimulationApp`` must be constructed before any of
those modules can be imported safely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IsaacContext:
    """Runtime handles handed to scripts inside the ``IsaacApp`` context."""

    app: Any
    world: Any
    stage: Any
    default_prim_path: str = "/World"
    is_isaac_45: bool = False
    extras: dict = field(default_factory=dict)

    def step(self, render: bool = True) -> None:
        """Advance the world by one physics step."""
        self.world.step(render=render)

    def update(self) -> None:
        """Render-only frame tick (no physics)."""
        self.app.update()

    def is_running(self) -> bool:
        return bool(self.app.is_running())

    def close(self) -> None:
        try:
            self.world.stop()
        except Exception:
            pass
        try:
            self.app.close()
        except Exception:
            pass


class IsaacApp:
    """Context manager that launches ``SimulationApp`` and yields an :class:`IsaacContext`.

    Parameters
    ----------
    headless : Disable rendering / GUI when ``True``.
    renderer : Renderer name (e.g. ``"RayTracedLighting"``).
    physics_dt : World physics timestep.
    rendering_dt : World rendering timestep.
    """

    def __init__(
        self,
        headless: bool = False,
        renderer: str = "RayTracedLighting",
        physics_dt: float = 1.0 / 60.0,
        rendering_dt: float = 1.0 / 60.0,
        low_quality: bool = True,
    ):
        self.headless = headless
        self.renderer = renderer
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.low_quality = low_quality
        self._ctx: IsaacContext | None = None

    def __enter__(self) -> IsaacContext:
        from isaacsim import SimulationApp  # noqa: WPS433

        app = SimulationApp({"headless": self.headless, "renderer": self.renderer})

        # Detect Isaac Sim version (4.5 vs 5.x); uses the same heuristic as
        # ``visualization_isaac.vrp.replay``: presence of the legacy URDF
        # importer module ``omni.importer.urdf``.
        is_isaac_45 = False
        try:
            import omni.importer.urdf  # type: ignore  # noqa: F401
        except ImportError:
            is_isaac_45 = True

        # World creation must happen after SimulationApp.
        try:
            from isaacsim.core.api import World  # type: ignore
        except Exception:
            from omni.isaac.core import World  # type: ignore

        import omni.usd  # type: ignore

        world = World(physics_dt=self.physics_dt, rendering_dt=self.rendering_dt)
        stage = omni.usd.get_context().get_stage()

        # Ensure /World exists and is the default prim, so `Sdf.Reference` /
        # `OverridePrim` paths resolve consistently across helpers.
        from pxr import UsdGeom

        if not stage.GetPrimAtPath("/World").IsValid():
            UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

        if self.low_quality:
            try:
                import carb  # type: ignore

                s = carb.settings.get_settings()
                # Balanced preset: keep shadows + AO + indirect diffuse for
                # depth cues, but skip the heavyweight effects we don't need
                # (AA, reflections, translucency, DLSS upscaling).
                s.set("/rtx/post/aa/op", 0)
                s.set("/rtx/translucency/enabled", False)
                s.set("/rtx/reflections/enabled", False)
                s.set("/rtx/post/dlss/execMode", 0)
                s.set("/rtx/raytracing/lightTrees/enabled", False)
                s.set("/rtx/shadows/enabled", True)
                s.set("/rtx/ambientOcclusion/enabled", True)
                s.set("/rtx/indirectDiffuse/enabled", True)

                # Kill temporal accumulation + motion blur. These are what
                # cause the "trails when panning the camera" and the soft
                # halo around line-set / curve edges occluded by the mesh.
                s.set("/rtx/post/aa/enabled", False)
                s.set("/rtx/post/taa/enabled", False)
                s.set("/rtx/post/motionblur/enabled", False)
                s.set("/rtx/temporalAccumulation/enabled", False)
                s.set("/rtx/raytracing/temporalAccumulation/enabled", False)
                s.set("/rtx/post/temporalAA/enabled", False)
                s.set("/rtx/post/dof/enabled", False)
                s.set("/rtx/post/chromaticAberration/enabled", False)
                s.set("/rtx/post/bloom/enabled", False)

                # Denoising/spatial-filter halo around small features (lines,
                # cubes silhouetted against the mesh). Turn the RTX denoiser
                # off so the renderer doesn't blur sub-pixel edges.
                s.set("/rtx/raytracing/showLights", False)
                s.set("/rtx/raytracing/spatialFilter/enabled", False)
                s.set("/rtx/post/denoising/enabled", False)
                s.set("/rtx/raytracing/denoiser/enabled", False)
                s.set("/rtx/raytracing/aa/op", 0)
            except Exception as exc:  # pragma: no cover - rendering settings best-effort
                logger.warning("[IsaacApp] could not apply quality preset: %s", exc)

        ctx = IsaacContext(
            app=app, world=world, stage=stage, is_isaac_45=is_isaac_45,
        )
        # Convenience method bindings (keep IsaacContext stateless of references).
        ctx.run_until_quit = lambda: _run_until_quit(ctx)  # type: ignore[attr-defined]
        self._ctx = ctx
        logger.info(
            "[IsaacApp] Launched SimulationApp (headless=%s, renderer=%s, isaac45=%s)",
            self.headless,
            self.renderer,
            is_isaac_45,
        )
        return ctx

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None


def _run_until_quit(ctx: IsaacContext) -> None:
    """Loop ``app.update()`` until the user closes the window."""
    while ctx.is_running():
        ctx.update()


def add_dome_light(
    stage,
    path: str = "/World/DomeLight",
    intensity: float = 500.0,
    color: tuple = (1.0, 1.0, 1.0),
    guide_radius: float = 1.0,
) -> str:
    """Convenience: add a dome light prim to *stage* (mirrors ReplayVisualizer.add_dome_light).

    DomeLight is conceptually at infinity, but Hydra renders a guide sphere
    proportional to ``guide_radius`` (the visible "dome" you see in the
    viewport). USD doesn't expose this on the schema in every version, so
    we set it via the raw attribute path.
    """
    from pxr import Gf, Sdf, UsdLux

    dome_light = UsdLux.DomeLight.Define(stage, path)
    dome_light.GetIntensityAttr().Set(intensity)
    dome_light.GetColorAttr().Set(Gf.Vec3f(*color))
    prim = dome_light.GetPrim()
    # Best-effort: write whichever guide-radius attr the kit recognises.
    for attr_name in ("guideRadius", "inputs:guideRadius", "radius", "inputs:radius"):
        attr = prim.GetAttribute(attr_name)
        if not attr:
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        try:
            attr.Set(float(guide_radius))
        except Exception:
            pass
    return path


def add_distant_light(
    stage,
    path: str = "/World/DistantLight",
    intensity: float = 1500.0,
    angle_deg: float = 1.5,
    color: tuple = (1.0, 1.0, 1.0),
    rotate_xyz: tuple = (-45.0, 0.0, 30.0),
) -> str:
    """Add a directional 'sun' light for diffuse shading."""
    from pxr import Gf, UsdGeom, UsdLux

    light = UsdLux.DistantLight.Define(stage, path)
    light.GetIntensityAttr().Set(intensity)
    light.GetAngleAttr().Set(angle_deg)
    light.GetColorAttr().Set(Gf.Vec3f(*color))
    xf = UsdGeom.Xformable(light.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))
    return path


def add_ground_plane(
    ctx_or_stage,
    path: str = "/World/defaultGroundPlane",
    size: float = 200.0,
    color: tuple | None = None,
    z: float = 0.0,
) -> str:
    """Add Isaac Sim's canonical blue+grid ground plane.

    Pass an :class:`IsaacContext` (preferred) so the signature default
    ground plane environment can be installed via ``world.scene``. Falls
    back to a plain ``GroundPlane`` quad and finally to a raw USD mesh.
    """
    from pxr import Gf, UsdGeom, Vt

    # Distinguish IsaacContext from a bare USD stage.
    ctx = ctx_or_stage if hasattr(ctx_or_stage, "world") else None
    stage = ctx.stage if ctx is not None else ctx_or_stage

    if ctx is not None and color is None:
        try:
            ctx.world.scene.add_default_ground_plane(z_position=z)
            # The bundled environment is ~25 m across by default. We scale
            # every immediate non-light child of the env reference; the
            # transform inherits down to all geometry, while sibling
            # SphereLight prims are left at their original size.
            from pxr import UsdLux

            scale_factor = max(size / 25.0, 1.0)
            light_schemas = (
                UsdLux.SphereLight,
                UsdLux.DomeLight,
                UsdLux.DistantLight,
                UsdLux.RectLight,
                UsdLux.DiskLight,
                UsdLux.CylinderLight,
            )

            scaled_any = False
            for guess in (path, "/World/defaultGroundPlane", "/defaultGroundPlane"):
                root = stage.GetPrimAtPath(guess)
                if not (root and root.IsValid()):
                    continue
                for child in root.GetChildren():
                    if any(child.IsA(s) for s in light_schemas):
                        continue
                    xf = UsdGeom.Xformable(child)
                    if not xf:
                        continue
                    ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
                    scale_op = ops.get(UsdGeom.XformOp.TypeScale)
                    if scale_op is None:
                        scale_op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                    scale_op.Set(Gf.Vec3d(scale_factor, scale_factor, 1.0))
                    scaled_any = True
                break
            if not scaled_any:
                logger.warning("[add_ground_plane] default env had no scalable children")
            return path
        except Exception as exc:
            logger.warning("[add_ground_plane] default env unavailable: %s", exc)

    try:
        from isaacsim.core.api.objects.ground_plane import GroundPlane  # type: ignore

        kwargs = {"prim_path": path, "z_position": z, "size": size}
        if color is not None:
            kwargs["color"] = np.array(color)
        GroundPlane(**kwargs)
        return path
    except Exception:
        pass

    # Fallback colour if even the simple quad path is taken.
    if color is None:
        color = (0.18, 0.40, 0.85)

    # Fallback: simple quad mesh.
    half = size / 2.0
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-half, -half, z),
                Gf.Vec3f(half, -half, z),
                Gf.Vec3f(half, half, z),
                Gf.Vec3f(-half, half, z),
            ]
        )
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return path


def frame_viewport(
    stage,
    target: tuple = (0.0, 0.0, 0.0),
    distance: float = 80.0,
    azimuth_deg: float = -60.0,
    elevation_deg: float = 25.0,
    path: str = "/World/MainCamera",
    focal_length: float = 18.0,
) -> str:
    """Create a camera looking at *target* and make it the active viewport camera.

    Isaac Sim's default perspective camera sits at ~(5, 5, 5) looking at the
    origin -- inside any large scene. This helper places a camera at
    *distance* metres from *target* (spherical coords) and switches the
    viewport to it so the user actually sees something on first frame.
    """
    import math

    from pxr import Gf, UsdGeom

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = (
        target[0] + distance * math.cos(el) * math.cos(az),
        target[1] + distance * math.cos(el) * math.sin(az),
        target[2] + distance * math.sin(el),
    )

    cam = UsdGeom.Camera.Define(stage, path)
    cam.GetFocalLengthAttr().Set(float(focal_length))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, max(distance * 10.0, 1000.0)))

    # Build a look-at transform: place at *eye*, look at *target*, with +Z up.
    mat = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0)
    ).GetInverse()
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(mat)

    try:
        from omni.kit.viewport.utility import get_active_viewport  # type: ignore

        vp = get_active_viewport()
        if vp is not None:
            vp.set_active_camera(path)
    except Exception as exc:  # pragma: no cover - viewport API differs across versions
        logger.warning("[frame_viewport] could not switch active camera: %s", exc)

    return path


def set_camera_lookat(
    stage,
    path: str,
    eye: tuple,
    target: tuple,
    up: tuple = (0.0, 0.0, 1.0),
) -> None:
    """Re-aim a camera prim with a fresh look-at transform."""
    import math  # noqa: F401  (re-used by callers building eye/target)

    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    mat = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)
    ).GetInverse()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(mat)


def add_zero_gravity(stage, scene_path: str = "/physicsScene") -> str:
    """Convenience: configure a zero-gravity physics scene."""
    from pxr import Gf, UsdPhysics

    ps = UsdPhysics.Scene.Get(stage, scene_path)
    if not ps:
        ps = UsdPhysics.Scene.Define(stage, scene_path)
    ps.GetGravityDirectionAttr().Set(Gf.Vec3f(0, 0, 0))
    ps.GetGravityMagnitudeAttr().Set(0.0)
    return scene_path
