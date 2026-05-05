#!/usr/bin/env python3
"""Visualise a saved pipeline run end-to-end in Isaac Sim.

Loads the directory written by ``scripts.run_full_pipeline`` and walks 9 phases
(mesh+pointcloud, normals, candidates, set-cover visibility, selected, VRP depots+waypoints,
assignment, trajectories, replay). Keys: N/Right next, P/Left prev, Q/Esc quit.
``--phase-duration`` adds an auto-advance fallback.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import trimesh

DEFAULT_BROV = os.path.join(REPO_ROOT, "assets", "robot", "brov", "BROV_high.usd")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default=os.path.join(REPO_ROOT, "outputs", "full_pipeline"),
        help="Pipeline directory written by run_full_pipeline.",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--phase-duration",
        type=float,
        default=None,
        help="Auto-advance every phase after this many seconds (still keypress-interruptible).",
    )
    p.add_argument("--replay-dt", type=float, default=1.0 / 30.0,
        help="Wall-clock seconds per trajectory frame in the replay phase.")
    p.add_argument("--brov-usd", default=DEFAULT_BROV, help=f"BROV USD asset path (default: {DEFAULT_BROV}).")
    p.add_argument(
        "--max-candidates",
        type=int,
        default=300,
        help="Subsample candidate viewpoints in phase 3 to this many for performance.",
    )
    p.add_argument(
        "--mesh-z-offset",
        type=float,
        default=15.0,
        help="Lift the inspection mesh by this many metres so it doesn't intersect the floor.",
    )
    p.add_argument(
        "--use-brov-usd",
        action="store_true",
        help="Spawn replay robots as full BROV USD references (slow; default uses light cuboids).",
    )
    p.add_argument(
        "--orbit-period",
        type=float,
        default=20.0,
        help="Seconds for one full camera orbit during replay. 0 to disable.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    # Load pipeline data BEFORE launching Isaac Sim (no Isaac deps yet).
    from VRP.utils.serialization import load_pipeline

    data = load_pipeline(args.input)

    # Lift the mesh so it doesn't intersect the floor; shift depots / targets / trajectories
    # by the same offset so the scene stays internally consistent.
    if args.mesh_z_offset:
        dz = float(args.mesh_z_offset)
        pose = list(data["mesh_pose"])
        pose[2] = float(pose[2]) + dz
        data["mesh_pose"] = pose
        for key in ("target_points", "all_positions", "selected_positions", "robot_start_xyzs"):
            arr = data.get(key)
            if arr is not None and len(arr) > 0:
                arr = np.asarray(arr, dtype=np.float64).copy()
                arr[..., 2] += dz
                data[key] = arr
        exec_res = data.get("exec_result")
        if exec_res is not None and getattr(exec_res, "all_traj_positions", None) is not None:
            shifted = []
            for traj in exec_res.all_traj_positions:
                t = np.asarray(traj, dtype=np.float64).copy()
                if t.size:
                    t[:, 2] += dz
                shifted.append(t)
            exec_res.all_traj_positions = shifted

        opt_res = data.get("optimization_result")
        if opt_res is not None and getattr(opt_res, "positions", None) is not None:
            pos = np.asarray(opt_res.positions, dtype=np.float64).copy()
            if pos.size:
                pos[..., 2] += dz
            opt_res.positions = pos

    from shared.mesh_loader import load_and_transform_mesh
    from visibility.core.types import FrustumParams

    mesh = load_and_transform_mesh(
        data["mesh_path"], data["mesh_target_length"], data["mesh_pose"]
    )
    fp = data["frustum_params"]
    frustum_params = FrustumParams(
        fov_y=float(fp["fov_y_rad"]),
        aspect=float(fp["aspect"]),
        near=float(fp["near"]),
        far=float(fp["far"]),
    )

    from visualization_isaac import (
        IsaacApp,
        ModelVisualizer,
        Phase,
        PhaseController,
        SamplingVisualizer,
        SetCoverVisualizer,
        VRPVisualizer,
        add_distant_light,
        add_dome_light,
        add_ground_plane,
        convert_trajectories,
        frame_viewport,
        set_camera_lookat,
    )

    with IsaacApp(headless=args.headless) as ctx:
        # Dome only — directional sun light produces hard contrasty shadows that read as
        # "weird lighting"; pure dome gives soft uniform diffuse, closer to a CAD-style look.
        add_dome_light(ctx.stage, intensity=750.0)
        # Pass IsaacContext (not just stage) for the canonical blue+grid Isaac default ground plane.
        add_ground_plane(ctx, size=1500.0)

        # Frame the viewport on the mesh; default Isaac camera at ~(5,5,5) is inside
        # any tens-of-metres-scale scene.
        cam_path = "/World/MainCamera"
        cam_target = (0.0, 0.0, 0.0)
        cam_distance = 50.0
        try:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            cam_target = tuple(((verts.min(axis=0) + verts.max(axis=0)) * 0.5).tolist())
            extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
            cam_distance = max(extent * 1.4, 5.0)
            cam_path = frame_viewport(ctx.stage, target=cam_target, distance=cam_distance)
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).warning("frame_viewport failed: %s", exc)

        model_vis = ModelVisualizer(mesh, data["target_points"], data["normals"])
        sampling_vis = SamplingVisualizer(
            mesh, data["target_points"], data["normals"], frustum_params
        )
        sc_vis = SetCoverVisualizer(mesh, data["target_points"], frustum_params)

        traj_poses = convert_trajectories(data["exec_result"].all_traj_positions)
        vrp_vis = VRPVisualizer(num_robots=int(data["num_robots"]), traj_poses=traj_poses)

        def enter_mesh_pc(stage, parent):
            model_vis.add_mesh(stage, f"{parent}/mesh")
            model_vis.add_points(stage, f"{parent}/points", color=(0.9, 0.9, 0.9))

        def enter_normals(stage, parent):
            model_vis.add_mesh(stage, f"{parent}/mesh")
            model_vis.add_points(stage, f"{parent}/points", color=(0.5, 0.5, 0.5))
            model_vis.add_normals(stage, f"{parent}/normals", normal_scale=0.3)

        # Subsample candidates: rendering 2000 frustums craters interactivity even on RTX 3090.
        all_pos = np.asarray(data["all_positions"])
        all_R = np.asarray(data["all_rotmats"])
        if len(all_pos) > args.max_candidates:
            sel = np.random.default_rng(0).choice(
                len(all_pos), size=args.max_candidates, replace=False
            )
            cand_pos = all_pos[sel]
            cand_R = all_R[sel]
        else:
            cand_pos = all_pos
            cand_R = all_R

        def enter_candidates(stage, parent):
            sampling_vis.add_candidates(
                stage,
                parent,
                positions=cand_pos,
                rotmats=cand_R,
            )

        # Reuse the subsampled candidate set + matching rows of the visibility map.
        if len(all_pos) > args.max_candidates:
            vis_pos = cand_pos
            vis_R = cand_R
            full_vis_full = data.get("full_visibility_map")
            full_vis_sub = (
                np.asarray(full_vis_full)[sel] if full_vis_full is not None else None
            )
        else:
            vis_pos = all_pos
            vis_R = all_R
            full_vis_sub = data.get("full_visibility_map")

        def enter_visibility(stage, parent):
            if full_vis_sub is None:
                sc_vis.add_phase_selected(stage, parent, data["optimization_result"])
                return
            sc_vis.add_phase_visibility(
                stage,
                parent,
                all_positions=vis_pos,
                all_rotmats=vis_R,
                full_visibility_map=full_vis_sub,
            )

        def enter_selected(stage, parent):
            sc_vis.add_phase_selected(stage, parent, data["optimization_result"])

        def enter_depots(stage, parent):
            vrp_vis.add_inspection_mesh(
                stage,
                parent,
                data["mesh_path"],
                data["mesh_pose"],
                data["mesh_target_length"],
            )
            vrp_vis.add_depots(stage, parent, data["robot_start_xyzs"])
            vrp_vis.add_waypoints(
                stage,
                parent,
                vp_positions=np.asarray(data["selected_positions"]),
                vp_rotmats=np.asarray(data["selected_rotmats"]),
            )

        def enter_assignment(stage, parent):
            vrp_vis.add_inspection_mesh(
                stage,
                parent,
                data["mesh_path"],
                data["mesh_pose"],
                data["mesh_target_length"],
            )
            vrp_vis.add_depots(stage, parent, data["robot_start_xyzs"])
            vrp_vis.add_assignment(
                stage,
                parent,
                vp_positions=np.asarray(data["selected_positions"]),
                vp_rotmats=np.asarray(data["selected_rotmats"]),
                routes=data["vrp_routes"],
                home_indices=data["home_indices"],
            )

        def enter_trajectories(stage, parent):
            vrp_vis.add_inspection_mesh(
                stage,
                parent,
                data["mesh_path"],
                data["mesh_pose"],
                data["mesh_target_length"],
            )
            vrp_vis.add_depots(stage, parent, data["robot_start_xyzs"])
            vrp_vis.add_assignment(
                stage,
                parent,
                vp_positions=np.asarray(data["selected_positions"]),
                vp_rotmats=np.asarray(data["selected_rotmats"]),
                routes=data["vrp_routes"],
                home_indices=data["home_indices"],
            )
            vrp_vis.add_trajectories(stage, parent)

        replay_state: dict = {
            "prims": [],
            "last_t": None,
            "frame": 0,
            "enter_t": None,
        }

        def enter_replay(stage, parent):
            from visualization_isaac._usd_primitives import create_cuboid_prim

            vrp_vis.add_inspection_mesh(
                stage,
                parent,
                data["mesh_path"],
                data["mesh_pose"],
                data["mesh_target_length"],
            )
            vrp_vis.add_depots(stage, parent, data["robot_start_xyzs"])
            vrp_vis.add_trajectories(stage, parent)
            if args.use_brov_usd:
                replay_state["prims"] = vrp_vis.add_brov_robots(stage, parent, args.brov_usd)
            else:
                # 345 MB BROV USD * 3 references is too heavy for replay; use a coloured
                # cuboid per robot. Animation hook is identical.
                prims = []
                for i in range(int(data["num_robots"])):
                    path = f"{parent}/simple_robot_{i}"
                    color = tuple(vrp_vis.colors[i % len(vrp_vis.colors)])
                    create_cuboid_prim(
                        stage,
                        path,
                        position=np.zeros(3),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                        color=color,
                        size=0.6,
                    )
                    prims.append(stage.GetPrimAtPath(path))
                replay_state["prims"] = prims
            replay_state["last_t"] = time.perf_counter()
            replay_state["enter_t"] = time.perf_counter()
            replay_state["frame"] = 0

        import math as _math

        orbit_period = float(args.orbit_period)
        orbit_elev_rad = _math.radians(25.0)

        # Single global orbit clock so the camera keeps moving smoothly across phase transitions.
        orbit_state = {"start_t": time.perf_counter()}

        def step_orbit(stage, parent, frame_idx, _now):
            if orbit_period <= 0:
                return
            dt = time.perf_counter() - orbit_state["start_t"]
            az = (dt / orbit_period) * 2.0 * _math.pi
            eye = (
                cam_target[0] + cam_distance * _math.cos(orbit_elev_rad) * _math.cos(az),
                cam_target[1] + cam_distance * _math.cos(orbit_elev_rad) * _math.sin(az),
                cam_target[2] + cam_distance * _math.sin(orbit_elev_rad),
            )
            set_camera_lookat(stage, cam_path, eye=eye, target=cam_target)

        def step_replay(stage, parent, frame_idx, now):
            t = time.perf_counter()
            if replay_state["last_t"] is None:
                replay_state["last_t"] = t
            if (t - replay_state["last_t"]) >= args.replay_dt:
                replay_state["frame"] += 1
                replay_state["last_t"] = t
            if replay_state["prims"]:
                vrp_vis.step(replay_state["prims"], replay_state["frame"])
            step_orbit(stage, parent, frame_idx, now)

        D = args.phase_duration
        phases = [
            Phase("mesh_and_points", enter=enter_mesh_pc, on_step=step_orbit, duration=D),
            Phase("surface_normals", enter=enter_normals, on_step=step_orbit, duration=D),
            Phase("all_candidates", enter=enter_candidates, on_step=step_orbit, duration=D),
            Phase("setcover_visibility", enter=enter_visibility, on_step=step_orbit, duration=D),
            Phase("setcover_selected", enter=enter_selected, on_step=step_orbit, duration=D),
            Phase("vrp_depots_waypoints", enter=enter_depots, on_step=step_orbit, duration=D),
            Phase("vrp_assignment", enter=enter_assignment, on_step=step_orbit, duration=D),
            Phase("vrp_trajectories", enter=enter_trajectories, on_step=step_orbit, duration=D),
            Phase(
                "vrp_replay",
                enter=enter_replay,
                on_step=step_replay,
                duration=D,
            ),
        ]

        controller = PhaseController(ctx, phases)
        if args.headless:
            controller.headless_play()
        else:
            controller.run()


if __name__ == "__main__":
    main()
