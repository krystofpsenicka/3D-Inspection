#!/usr/bin/env python3
"""
Full Pipeline Visualization – Isaac Sim + OceanSim
===================================================

Loads ``pipeline_data.pkl`` produced by :pymod:`run_full_pipeline` and replays
the entire 3D-Inspection → VRP pipeline inside Isaac Sim with OceanSim,
advancing through five keyboard-triggered phases.

Phases
------
1  Ship mesh + surface pointcloud (default blue)
2  All candidate viewpoints (frustum wireframes, gray)
3  Selected viewpoints – neutral gray frustums + coverage-coloured pointcloud
4  Selected viewpoints – VRP robot-coloured frustums + coverage-coloured pointcloud
5  VRP trajectory preview – trajectory paths + viewpoint cube markers per robot
6  VRP trajectory replay – AUVs move, coloured frustums disappear on visit

Controls
--------
**Press PLAY** in Isaac Sim's timeline panel, then:

* **N** – advance to the next phase
* **O** – toggle orbit camera (auto-rotates around the scene)
* **R** – reset Phase 6 replay

Usage
-----
::

    python visualize_full_pipeline_oceansim.py                             # defaults
    python visualize_full_pipeline_oceansim.py --data my_pipeline.pkl      # custom data
    python visualize_full_pipeline_oceansim.py --headless                  # no GUI
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in [REPO_ROOT, os.path.join(REPO_ROOT, "3D-Inspection", "methods_analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Frustum geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

_FRUSTUM_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # near quad
    (4, 5), (5, 6), (6, 7), (7, 4),   # far quad
    (0, 4), (1, 5), (2, 6), (3, 7),   # connecting
]


def _compute_frustum_corners(
    position: np.ndarray,
    direction: np.ndarray,
    fov_y: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Return (8, 3) array: near [TL TR BR BL] + far [TL TR BR BL]."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    up_ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(d[2]) < 0.99
        else np.array([0.0, 1.0, 0.0])
    )
    right = np.cross(d, up_ref)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, d)

    h_n, w_n = near * np.tan(fov_y / 2), near * np.tan(fov_y / 2) * aspect
    h_f, w_f = far  * np.tan(fov_y / 2), far  * np.tan(fov_y / 2) * aspect

    cn = position + d * near
    cf = position + d * far

    return np.array([
        cn + up * h_n - right * w_n,  cn + up * h_n + right * w_n,
        cn - up * h_n + right * w_n,  cn - up * h_n - right * w_n,
        cf + up * h_f - right * w_f,  cf + up * h_f + right * w_f,
        cf - up * h_f + right * w_f,  cf - up * h_f - right * w_f,
    ])


def _frustum_edge_points(corners: np.ndarray) -> np.ndarray:
    """Flatten 8 corners into 24 edge-endpoint vertices (12 edges × 2)."""
    pts = np.zeros((24, 3))
    for i, (a, b) in enumerate(_FRUSTUM_EDGES):
        pts[2 * i]     = corners[a]
        pts[2 * i + 1] = corners[b]
    return pts


# ═══════════════════════════════════════════════════════════════════════════════
# Arrival-step computation for Phase 5
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_arrival_steps(
    traj_positions_list: list,
    robot_insp_indices: list[list[int]],
    selected_wp: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """Map selected-VP index → (robot_idx, trajectory step of closest approach)."""
    arrivals: dict[int, tuple[int, int]] = {}
    for ri, traj in enumerate(traj_positions_list):
        search_from = 0
        for wp_idx in robot_insp_indices[ri]:
            wp_pos = np.asarray(selected_wp[wp_idx][:3], dtype=np.float64)
            best_step, best_dist = search_from, float("inf")
            for step in range(search_from, len(traj)):
                d = np.linalg.norm(np.asarray(traj[step][:3], dtype=np.float64) - wp_pos)
                if d < best_dist:
                    best_dist = d
                    best_step = step
                elif d > best_dist + 2.0:
                    break
            arrivals[wp_idx] = (ri, best_step)
            search_from = best_step
    return arrivals


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise the full inspection pipeline in Isaac Sim + OceanSim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="pipeline_data.pkl",
                   help="Path to pipeline_data.pkl from run_full_pipeline.py.")
    p.add_argument("--headless", action="store_true",
                   help="Run without GUI window.")
    return p.parse_args()


def _load_ship_mesh_trimesh(stage, data, Gf, Vt, UsdGeom):
    """Create the ship mesh as a UsdGeom.Mesh from trimesh-loaded vertices.

    This bypasses the Omniverse GLB→USD converter, which applies an implicit
    up-axis rotation (glTF Y-up → Isaac Sim Z-up) that would desynchronise
    the rendered mesh from the pointcloud / viewpoint data (all generated in
    the trimesh coordinate frame).
    """
    import trimesh as _tm
    from scipy.spatial.transform import Rotation as _Rot

    raw = _tm.load(data["mesh_path"], force="mesh")
    if isinstance(raw, _tm.Scene):
        raw = _tm.util.concatenate(list(raw.geometry.values()))

    longest = float(raw.extents.max())
    ms = data["mesh_target_length"] / longest if longest > 0 else 1.0
    raw.apply_scale(ms)

    mp = data["mesh_pose"]
    T = np.eye(4)
    T[:3, 3] = mp[:3]
    T[:3, :3] = _Rot.from_quat([mp[4], mp[5], mp[6], mp[3]]).as_matrix()
    raw.apply_transform(T)

    ship_path = "/World/DukeOfLancaster"
    mesh_prim = UsdGeom.Mesh.Define(stage, ship_path)

    verts = np.asarray(raw.vertices, dtype=np.float32)
    faces = np.asarray(raw.faces)

    mesh_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    )
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
    mesh_prim.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.55, 0.55, 0.6)]))
    mesh_prim.GetDoubleSidedAttr().Set(True)

    logger.info("Ship mesh loaded via trimesh: %d verts, %d faces, scale=%.4f",
                len(verts), len(faces), ms)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    # ── Load pipeline data ────────────────────────────────────────────
    logger.info("Loading pipeline data from %s …", args.data)
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    

    # ══════════════════════════════════════════════════════════════════
    # Unpack data
    # ══════════════════════════════════════════════════════════════════
    target_points  = data["target_points"]       # (N_pts, 3)
    all_candidates = data["all_candidates"]      # List[(pos, dir)]
    opt_result     = data["optimization_result"] # OptimizationResult
    frustum_cfg    = data["frustum_params"]       # dict
    selected_wp    = data["selected_waypoints_7dof"]  # (N_sel, 7)
    exec_result    = data["exec_result"]          # ExecutionResult
    num_robots     = data["num_robots"]
    robot_insp_idx = data["robot_inspection_wp_indices"]

    fov_y   = frustum_cfg["fov_y_rad"]
    aspect  = frustum_cfg["aspect"]
    f_near  = frustum_cfg["near"]
    f_far   = frustum_cfg["far"]

    N_pts       = len(target_points)
    N_cand      = len(all_candidates)
    N_sel       = opt_result.num_viewpoints
    total_steps = max(len(t) for t in exec_result.all_traj_positions)

    # ── Patch VRP config to match saved mesh scale + pose ───────────────
    import VRP.config as _cfg
    _cfg.MESH_TARGET_LENGTH = data["mesh_target_length"]
    _cfg.MESH_POSE = data["mesh_pose"]

    import VRP.visualization_oceansim as _viz
    _viz.MESH_TARGET_LENGTH = data["mesh_target_length"]

    # ── Bootstrap Isaac Sim ───────────────────────────────────────────
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp(
        {"headless": args.headless, "width": "1920", "height": "1080"}
    )

    # ── Enable OceanSim ───────────────────────────────────────────────
    _viz._try_enable_oceansim_extension()
    if not _viz._ensure_oceansim_importable():
        logger.error("OceanSim extension could not be loaded.  Aborting.")
        sim_app.close()
        return

    # ── Late imports (need SimulationApp alive) ───────────────────────
    try:
        from isaacsim.core.api import World  # type: ignore
    except ImportError:
        from omni.isaac.core import World  # type: ignore

    from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Vt  # type: ignore
    import carb          # type: ignore
    import carb.input    # type: ignore
    import omni.appwindow  # type: ignore

    # ── World + physics ───────────────────────────────────────────────
    my_world = World(stage_units_in_meters=1.0)
    stage    = my_world.stage

    try:
        ps = UsdPhysics.Scene.Get(stage, "/physicsScene")
        if not ps:
            ps = UsdPhysics.Scene.Define(stage, "/physicsScene")
        ps.GetGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        ps.GetGravityMagnitudeAttr().Set(0.0)
    except Exception as e:
        logger.warning("Could not set zero gravity: %s", e)

    # ── Renderer + caustics ───────────────────────────────────────────
    try:
        carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
        logger.info("Renderer set to RTX RaytracedLighting.")
    except Exception:
        pass
    _viz._enable_rtx_caustics()

    # ── Lighting + scene ──────────────────────────────────────────────
    _viz._add_ambient_dome(stage)

    # Load ship mesh directly from trimesh instead of _load_ship_mesh.
    # The GLB→USD converter in add_reference_to_stage applies an implicit
    # up-axis rotation (Y-up → Z-up) that trimesh does not.  Since the
    # pointcloud and all viewpoint data were generated from the trimesh-
    # loaded mesh, creating the USD mesh from the same vertices ensures
    # perfect alignment.
    _load_ship_mesh_trimesh(stage, data, Gf, Vt, UsdGeom)

    _viz._load_static_obstacles(stage)

    mesh_pose = data["mesh_pose"]
    my_world.scene.add_default_ground_plane(
        z_position=float(mesh_pose[2]) - f_far - 4.0  # ensure ground is below the far plane of all frustums
    )

    # Map selected-VP index → owning robot
    wp_to_robot: dict[int, int] = {}
    for ri, idxs in enumerate(robot_insp_idx):
        for j in idxs:
            wp_to_robot[j] = ri

    COLORS = _viz._COLORS
    logger.info(
        "Data: %d pts, %d candidates, %d selected, %d robots, %d traj steps.",
        N_pts, N_cand, N_sel, num_robots, total_steps,
    )

    # ══════════════════════════════════════════════════════════════════
    # Helper: numpy → Vt.Vec3fArray
    # ══════════════════════════════════════════════════════════════════
    def _to_vt3f(arr: np.ndarray) -> Vt.Vec3fArray:
        return Vt.Vec3fArray(
            [Gf.Vec3f(float(r[0]), float(r[1]), float(r[2])) for r in arr]
        )

    # ══════════════════════════════════════════════════════════════════
    # Phase 1 geometry: Pointcloud (200 K points)
    # ══════════════════════════════════════════════════════════════════
    logger.info("Building pointcloud prim (%d pts) …", N_pts)
    PC_PATH = "/World/Pointcloud"
    pc_prim = UsdGeom.Points.Define(stage, PC_PATH)
    pc_prim.GetPointsAttr().Set(_to_vt3f(target_points))
    pc_prim.GetWidthsAttr().Set(Vt.FloatArray([0.04] * N_pts))
    DEFAULT_PT_COLOR = Gf.Vec3f(0.7, 0.85, 1.0)
    pc_prim.GetDisplayColorAttr().Set(
        Vt.Vec3fArray([DEFAULT_PT_COLOR] * N_pts)
    )
    UsdGeom.Imageable(pc_prim.GetPrim()).MakeInvisible()

    # ══════════════════════════════════════════════════════════════════
    # Phase 2 geometry: Candidate frustum wireframes (batched)
    # ══════════════════════════════════════════════════════════════════
    logger.info("Building candidate frustum wireframes (%d) …", N_cand)
    cand_edge_pts: list[np.ndarray] = []
    for pos, d in all_candidates:
        c = _compute_frustum_corners(pos, d, fov_y, aspect, f_near, f_far)
        cand_edge_pts.append(_frustum_edge_points(c))
    cand_pts_flat = np.concatenate(cand_edge_pts, axis=0)

    CAND_PATH = "/World/CandidateFrustums"
    cand_curves = UsdGeom.BasisCurves.Define(stage, CAND_PATH)
    cand_curves.GetPointsAttr().Set(_to_vt3f(cand_pts_flat))
    cand_curves.GetCurveVertexCountsAttr().Set(Vt.IntArray([2] * (N_cand * 12)))
    cand_curves.GetTypeAttr().Set("linear")
    cand_curves.GetWidthsAttr().Set(Vt.FloatArray([0.02] * len(cand_pts_flat)))
    cand_curves.GetDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(0.45, 0.45, 0.45)])
    )
    UsdGeom.Imageable(cand_curves.GetPrim()).MakeInvisible()

    # ══════════════════════════════════════════════════════════════════
    # Phases 3–6 geometry: Selected frustum wireframes (individual prims)
    # ══════════════════════════════════════════════════════════════════
    logger.info("Building selected frustum prims (%d) …", N_sel)
    SEL_GROUP = "/World/SelectedFrustums"
    UsdGeom.Xform.Define(stage, SEL_GROUP)
    UsdGeom.Imageable(stage.GetPrimAtPath(SEL_GROUP)).MakeInvisible()

    _NEUTRAL_FRUSTUM = Gf.Vec3f(0.85, 0.85, 0.85)
    sel_frustum_prims: list = []
    sel_frustum_vrp_colors: list = []  # per-frustum VRP robot color
    for si, vp in enumerate(opt_result.viewpoints):
        corners = _compute_frustum_corners(
            vp.position, vp.direction, fov_y, aspect, f_near, f_far,
        )
        edge_pts = _frustum_edge_points(corners)

        fp = f"{SEL_GROUP}/f_{si}"
        c = UsdGeom.BasisCurves.Define(stage, fp)
        c.GetPointsAttr().Set(_to_vt3f(edge_pts))
        c.GetCurveVertexCountsAttr().Set(Vt.IntArray([2] * 12))
        c.GetTypeAttr().Set("linear")
        c.GetWidthsAttr().Set(Vt.FloatArray([0.03] * 24))

        # Store VRP color; start neutral gray (coloring applied in phase 4+)
        ri = wp_to_robot.get(si, 0)
        clr = COLORS[ri % len(COLORS)]
        vrp_color = Gf.Vec3f(float(clr[0]), float(clr[1]), float(clr[2]))
        sel_frustum_vrp_colors.append(vrp_color)
        c.GetDisplayColorAttr().Set(Vt.Vec3fArray([_NEUTRAL_FRUSTUM]))
        sel_frustum_prims.append(stage.GetPrimAtPath(fp))

    # ══════════════════════════════════════════════════════════════════
    # Phase 5 geometry: Robot models
    # ══════════════════════════════════════════════════════════════════
    logger.info("Spawning %d BROV models …", num_robots)
    try:
        rob_prims = _viz._spawn_brov_models(stage, num_robots)
    except FileNotFoundError as e:
        logger.error("BROV model error: %s", e)
        sim_app.close()
        return

    for rp in rob_prims:
        UsdGeom.Imageable(rp).MakeInvisible()

    # Pre-convert trajectories to (N, 7) pose arrays
    traj_poses = _viz._convert_trajectories(exec_result.all_traj_positions)

    # ══════════════════════════════════════════════════════════════════
    # Phase 5 geometry: Trajectory paths + viewpoint cube markers
    # ══════════════════════════════════════════════════════════════════
    logger.info("Building trajectory path prims (%d robots) …", num_robots)
    TRAJ_GROUP = "/World/TrajectoryPaths"
    UsdGeom.Xform.Define(stage, TRAJ_GROUP)
    traj_curve_prims: list = []
    for i, poses in enumerate(traj_poses):
        pts = poses[::10, :3]  # downsample: every 10th step
        clr = COLORS[i % len(COLORS)]
        tp = f"{TRAJ_GROUP}/robot_{i}"
        tc = UsdGeom.BasisCurves.Define(stage, tp)
        tc.GetPointsAttr().Set(_to_vt3f(pts))
        tc.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))
        tc.GetTypeAttr().Set("linear")
        tc.GetWidthsAttr().Set(Vt.FloatArray([0.05] * len(pts)))
        tc.GetDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(clr[0]), float(clr[1]), float(clr[2]))])
        )
        traj_curve_prims.append(stage.GetPrimAtPath(tp))
    UsdGeom.Imageable(stage.GetPrimAtPath(TRAJ_GROUP)).MakeInvisible()

    logger.info("Building viewpoint cube markers (%d) …", N_sel)
    VP_CUBES_GROUP = "/World/ViewpointCubes"
    UsdGeom.Xform.Define(stage, VP_CUBES_GROUP)
    vp_cube_prims: list = []
    for si in range(N_sel):
        pos = selected_wp[si][:3]
        ri = wp_to_robot.get(si, 0)
        clr = COLORS[ri % len(COLORS)]
        cp = f"{VP_CUBES_GROUP}/cube_{si}"
        cube = UsdGeom.Cube.Define(stage, cp)
        cube.GetSizeAttr().Set(0.3)
        cube.GetDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(clr[0]), float(clr[1]), float(clr[2]))])
        )
        UsdGeom.XformCommonAPI(cube.GetPrim()).SetTranslate(
            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
        )
        vp_cube_prims.append(stage.GetPrimAtPath(cp))
    UsdGeom.Imageable(stage.GetPrimAtPath(VP_CUBES_GROUP)).MakeInvisible()

    # Pre-compute when each frustum should disappear
    arrivals = _compute_arrival_steps(
        exec_result.all_traj_positions, robot_insp_idx, selected_wp,
    )
    logger.info("Arrival steps computed for %d waypoints.", len(arrivals))

    # ══════════════════════════════════════════════════════════════════
    # Pre-compute coverage colouring for Phase 3+
    # ══════════════════════════════════════════════════════════════════
    logger.info("Computing coverage colouring …")
    covered_set: set[int] = set()
    for vp in opt_result.viewpoints:
        covered_set.update(int(i) for i in vp.visible_indices)

    GREEN = Gf.Vec3f(0.1, 1.0, 0.2)
    RED   = Gf.Vec3f(1.0, 0.15, 0.1)
    coverage_colors = Vt.Vec3fArray(
        [GREEN if i in covered_set else RED for i in range(N_pts)]
    )
    logger.info(
        "Coverage: %d / %d pts (%.1f%%).",
        len(covered_set), N_pts,
        100.0 * len(covered_set) / max(N_pts, 1),
    )

    # ══════════════════════════════════════════════════════════════════
    # Initialise physics
    # ══════════════════════════════════════════════════════════════════
    sim_app.update()
    sim_app.update()
    my_world.initialize_physics()

    # ── Orbit camera setup ────────────────────────────────────────────
    ORBIT_CENTER    = np.array(data["mesh_pose"][:3], dtype=np.float64)
    ORBIT_RADIUS    = max(f_far * 3.5, 50.0)   # distance from ship centre (m)
    ORBIT_HEIGHT    = ORBIT_RADIUS * 0.35       # elevation above scene centre
    ORBIT_SPEED_DEG = 0.4                       # degrees per render frame

    orbit_cam_path = "/World/OrbitCamera"
    orbit_cam_prim = UsdGeom.Camera.Define(stage, orbit_cam_path)
    orbit_cam_prim.GetHorizontalApertureAttr().Set(20.955)
    orbit_cam_prim.GetVerticalApertureAttr().Set(15.291)
    orbit_cam_prim.GetFocalLengthAttr().Set(18.147)
    orbit_cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, ORBIT_RADIUS * 4.0))

    orbit_enabled     = [False]
    orbit_angle       = [0.0]
    _default_cam_path = ["/OmniverseKit_Persp"]
    try:
        import omni.kit.viewport.utility as _vpu
        _default_cam_path[0] = _vpu.get_active_viewport().camera_path
    except Exception:
        pass

    def _update_orbit_camera(angle: float) -> None:
        """Reposition the orbit camera to the given azimuth angle."""
        px = ORBIT_CENTER[0] + ORBIT_RADIUS * math.cos(angle)
        py = ORBIT_CENTER[1] + ORBIT_RADIUS * math.sin(angle)
        pz = ORBIT_CENTER[2] + ORBIT_HEIGHT
        pos = np.array([px, py, pz])
        fwd = ORBIT_CENTER - pos
        fwd /= np.linalg.norm(fwd)
        up_w = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_w)
        if np.linalg.norm(right) < 1e-6:
            up_w = np.array([0.0, 1.0, 0.0])
            right = np.cross(fwd, up_w)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        # USD row-vector convention: each row is the world direction of a local axis.
        # Camera -Z is forward, so row2 = -fwd.
        mat = Gf.Matrix4d(
            float(right[0]),  float(right[1]),  float(right[2]),  0.0,
            float(cam_up[0]), float(cam_up[1]), float(cam_up[2]), 0.0,
            float(-fwd[0]),   float(-fwd[1]),   float(-fwd[2]),   0.0,
            float(px),        float(py),        float(pz),        1.0,
        )
        xf = UsdGeom.Xformable(orbit_cam_prim.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)

    # ══════════════════════════════════════════════════════════════════
    # Phase state machine + keyboard handler
    # ══════════════════════════════════════════════════════════════════
    REPLAY_SPEED  = 2         # trajectory steps advanced per render frame
    phase         = [0]      # mutable for closure
    phase_changed = [True]
    replay_idx    = [0]
    hidden_wp     = [set()]  # VP indices already hidden in Phase 6

    def _on_keyboard(event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        if event.input == carb.input.KeyboardInput.N:
            if phase[0] < 6:
                phase[0] += 1
                phase_changed[0] = True
                logger.info("▶ Phase %d", phase[0])
        elif event.input == carb.input.KeyboardInput.O:
            orbit_enabled[0] = not orbit_enabled[0]
            try:
                import omni.kit.viewport.utility as _vpu
                vp = _vpu.get_active_viewport()
                if orbit_enabled[0]:
                    _default_cam_path[0] = vp.camera_path
                    _update_orbit_camera(orbit_angle[0])
                    vp.camera_path = orbit_cam_path
                else:
                    vp.camera_path = _default_cam_path[0]
            except Exception as _e:
                logger.warning("Could not switch viewport camera: %s", _e)
            logger.info("Orbit camera %s.", "ENABLED" if orbit_enabled[0] else "DISABLED")
        elif event.input == carb.input.KeyboardInput.R:
            replay_idx[0] = 0
            hidden_wp[0] = set()
            # Re-show all selected frustums with VRP colors
            for si in range(N_sel):
                UsdGeom.Imageable(sel_frustum_prims[si]).MakeVisible()
                UsdGeom.BasisCurves(sel_frustum_prims[si]).GetDisplayColorAttr().Set(
                    Vt.Vec3fArray([sel_frustum_vrp_colors[si]])
                )
            logger.info("▶ Phase 6 replay reset.")

    app_window = omni.appwindow.get_default_app_window()
    keyboard   = app_window.get_keyboard()
    input_iface = carb.input.acquire_input_interface()
    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard)

    # ── Phase-transition logic ────────────────────────────────────────
    def _apply_phase() -> None:
        p = phase[0]

        # ── Pointcloud ──────────────────────────────────────────────
        pc_img = UsdGeom.Imageable(stage.GetPrimAtPath(PC_PATH))
        if p in (1, 2):
            pc_img.MakeVisible()
            pc_prim.GetDisplayColorAttr().Set(Vt.Vec3fArray([DEFAULT_PT_COLOR] * N_pts))
        elif p in (3, 4, 6):
            pc_img.MakeVisible()
            pc_prim.GetDisplayColorAttr().Set(coverage_colors)
        else:  # phase 0, 5
            pc_img.MakeInvisible()

        # ── Candidate frustums (Phase 2 only) ───────────────────────
        cand_img = UsdGeom.Imageable(cand_curves.GetPrim())
        if p == 2:
            cand_img.MakeVisible()
        else:
            cand_img.MakeInvisible()

        # ── Selected frustums (Phases 3, 4, 6) ──────────────────────
        sel_img = UsdGeom.Imageable(stage.GetPrimAtPath(SEL_GROUP))
        if p in (3, 4, 6):
            sel_img.MakeVisible()
            for si in range(N_sel):
                UsdGeom.Imageable(sel_frustum_prims[si]).MakeVisible()
                curves = UsdGeom.BasisCurves(sel_frustum_prims[si])
                if p == 3:
                    # Neutral gray — no robot assignment hint
                    curves.GetDisplayColorAttr().Set(
                        Vt.Vec3fArray([_NEUTRAL_FRUSTUM])
                    )
                else:
                    # VRP robot colors (phases 4 and 6)
                    curves.GetDisplayColorAttr().Set(
                        Vt.Vec3fArray([sel_frustum_vrp_colors[si]])
                    )
        else:
            sel_img.MakeInvisible()

        # ── Trajectory-preview geometry (Phase 5 only) ───────────────
        traj_img  = UsdGeom.Imageable(stage.GetPrimAtPath(TRAJ_GROUP))
        cubes_img = UsdGeom.Imageable(stage.GetPrimAtPath(VP_CUBES_GROUP))
        if p == 5:
            traj_img.MakeVisible()
            cubes_img.MakeVisible()
        else:
            traj_img.MakeInvisible()
            cubes_img.MakeInvisible()

        # ── Robots (Phase 6 only) ────────────────────────────────────
        if p == 6:
            for i, rp in enumerate(rob_prims):
                UsdGeom.Imageable(rp).MakeVisible()
                _viz._set_robot_pose(
                    rp, traj_poses[i][0, :3], traj_poses[i][0, 3:],
                )
            replay_idx[0] = 0
            hidden_wp[0] = set()
        else:
            for rp in rob_prims:
                UsdGeom.Imageable(rp).MakeInvisible()

        # Force the renderer to pick up all visibility changes immediately
        sim_app.update()

    # ══════════════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════════════
    logger.info(
        "Ready.  Press PLAY in Isaac Sim, then N to advance phases (1→6).  "
        "O toggles orbit camera.  R resets Phase 6 replay."
    )

    try:
        while sim_app.is_running():
            my_world.step(render=True)

            if not my_world.is_playing():
                continue

            # Apply phase transitions
            if phase_changed[0]:
                _apply_phase()
                phase_changed[0] = False

            # Orbit camera tick
            if orbit_enabled[0]:
                orbit_angle[0] = (
                    orbit_angle[0] + math.radians(ORBIT_SPEED_DEG)
                ) % (2.0 * math.pi)
                _update_orbit_camera(orbit_angle[0])

            # Phase 6: trajectory replay
            if phase[0] == 6:
                idx = replay_idx[0]
                if idx < total_steps:
                    # Teleport robots
                    for i, rp in enumerate(rob_prims):
                        safe = min(idx, len(traj_poses[i]) - 1)
                        _viz._set_robot_pose(
                            rp,
                            traj_poses[i][safe, :3],
                            traj_poses[i][safe, 3:],
                        )

                    # Hide frustums upon arrival
                    for wp_idx, (_, arr_step) in arrivals.items():
                        if idx >= arr_step and wp_idx not in hidden_wp[0]:
                            UsdGeom.Imageable(
                                sel_frustum_prims[wp_idx]
                            ).MakeInvisible()
                            hidden_wp[0].add(wp_idx)

                    replay_idx[0] += REPLAY_SPEED
                    if replay_idx[0] % 500 == 0:
                        logger.info(
                            "Replay step %d / %d", replay_idx[0], total_steps,
                        )
                elif idx == total_steps:
                    logger.info(
                        "Replay finished (%d steps).  Press R to restart.",
                        total_steps,
                    )
                    replay_idx[0] = total_steps + 1  # sentinel

    finally:
        input_iface.unsubscribe_to_keyboard_events(keyboard, kb_sub)
        sim_app.close()


if __name__ == "__main__":
    main()
