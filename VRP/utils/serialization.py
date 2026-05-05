"""Solution & pipeline persistence (NPZ + JSON format)."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from VRP.core.types import ExecutionResult


# ────────────────────────────────────────────────────────────────────────────
# ExecutionResult round-trip (single-file NPZ + JSON)
# ────────────────────────────────────────────────────────────────────────────


def save_solution(result: ExecutionResult, path: str) -> None:  # noqa: F821
    """Save an ExecutionResult to NPZ + JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base = path.rsplit(".", 1)[0] if "." in path else path

    arrays = {}
    for field_name in ("all_traj_positions", "all_traj_velocities"):
        robot_trajs = getattr(result, field_name)
        all_steps = []
        offsets = [0]
        for robot_steps in robot_trajs:
            if len(robot_steps) > 0:
                all_steps.append(robot_steps)
            offsets.append(offsets[-1] + len(robot_steps))
        arrays[field_name] = np.concatenate(all_steps) if all_steps else np.empty((0, 6))
        arrays[f"{field_name}_offsets"] = np.array(offsets, dtype=np.int64)

    arrays["initial_positions"] = np.array(result.initial_positions)
    np.savez_compressed(base + ".npz", **arrays)

    meta = {
        "all_waypoints": result.all_waypoints,
        "fail_counts": result.fail_counts,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f)


def load_solution(path: str) -> ExecutionResult:  # noqa: F821
    """Load an ExecutionResult from NPZ + JSON.

    Raises FileNotFoundError if the files do not exist.
    """
    from VRP.core.types import ExecutionResult

    base = path.rsplit(".", 1)[0] if "." in path else path
    npz_path = base + ".npz"
    json_path = base + ".json"

    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        raise FileNotFoundError(f"Solution files not found: {npz_path} and/or {json_path}")

    data = np.load(npz_path)
    with open(json_path) as f:
        meta = json.load(f)

    def unpack_ragged(name):
        flat = data[name]
        offsets = data[f"{name}_offsets"]
        return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]

    return ExecutionResult(
        all_traj_positions=unpack_ragged("all_traj_positions"),
        all_traj_velocities=unpack_ragged("all_traj_velocities"),
        all_waypoints=meta["all_waypoints"],
        initial_positions=list(data["initial_positions"]),
        fail_counts=meta["fail_counts"],
    )


# ────────────────────────────────────────────────────────────────────────────
# Directory-based pipeline round-trip
# ────────────────────────────────────────────────────────────────────────────


_SCHEMA_VERSION = 1


def _to_np(x):
    """Convert a CuPy / NumPy array (or scalar) to a NumPy array."""
    if x is None:
        return None
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def save_pipeline(pipeline_data: dict, dir_path: str) -> None:
    """Persist the full pipeline output into *dir_path*.

    The directory is wiped (mkdir -p, then remove children) before writing so
    each run produces a clean snapshot.

    Layout::

        manifest.json              -- schema, mesh meta, frustum, args, file index
        pointcloud.npz             -- target_points, normals
        candidates.npz             -- all_positions, all_rotmats, full_visibility_map
        optimization_result.npz    -- positions, rotations, visibility_map, selected_indices
        optimization_result.json   -- scalars (total_coverage, num_viewpoints, ...)
        vrp.json                   -- routes, home_indices, robot_start_xyzs, ...
        exec_result.npz            -- via save_solution()
        exec_result.json           -- via save_solution()

    Required keys in *pipeline_data*:
      ``mesh_path``, ``mesh_pose``, ``mesh_target_length``, ``mesh_scale``,
      ``mesh_bounds_min``, ``mesh_bounds_max``, ``frustum_params``,
      ``target_points``, ``normals``, ``all_positions``, ``all_rotmats``,
      ``full_visibility_map``, ``optimization_result``, ``selected_positions``,
      ``selected_rotmats``, ``vrp_routes``, ``vrp_routes_with_homes``,
      ``home_indices``, ``robot_start_xyzs``, ``robot_inspection_wp_indices``,
      ``num_robots``, ``exec_result``, ``args``.
    """
    dir_path = os.path.abspath(dir_path)
    if os.path.isdir(dir_path):
        for entry in os.listdir(dir_path):
            full = os.path.join(dir_path, entry)
            if os.path.isfile(full) or os.path.islink(full):
                os.unlink(full)
            elif os.path.isdir(full):
                shutil.rmtree(full)
    os.makedirs(dir_path, exist_ok=True)

    # 1. Pointcloud
    np.savez_compressed(
        os.path.join(dir_path, "pointcloud.npz"),
        target_points=_to_np(pipeline_data["target_points"]).astype(np.float32),
        normals=_to_np(pipeline_data["normals"]).astype(np.float32),
    )

    # 2. Candidates (positions + rotmats + full visibility)
    cand_kwargs = {
        "all_positions": _to_np(pipeline_data["all_positions"]).astype(np.float32),
        "all_rotmats": _to_np(pipeline_data["all_rotmats"]).astype(np.float32),
    }
    fvm = pipeline_data.get("full_visibility_map")
    if fvm is not None:
        cand_kwargs["full_visibility_map"] = _to_np(fvm).astype(np.uint8)
    np.savez_compressed(os.path.join(dir_path, "candidates.npz"), **cand_kwargs)

    # 3. Optimization result
    opt = pipeline_data["optimization_result"]
    opt_arrays = {
        "positions": _to_np(opt.positions).astype(np.float32),
        "rotations": _to_np(opt.rotations).astype(np.float32),
        "visibility_map": _to_np(opt.visibility_map).astype(np.uint8),
    }
    if getattr(opt, "selected_indices", None) is not None:
        opt_arrays["selected_indices"] = _to_np(opt.selected_indices).astype(np.int64)
    np.savez_compressed(os.path.join(dir_path, "optimization_result.npz"), **opt_arrays)

    with open(os.path.join(dir_path, "optimization_result.json"), "w") as f:
        json.dump(
            {
                "total_coverage": float(opt.total_coverage),
                "num_viewpoints": int(opt.num_viewpoints),
                "redundancy": float(opt.redundancy),
                "optimization_time": float(opt.optimization_time),
            },
            f,
        )

    # 4. VRP metadata
    vrp_meta = {
        "routes": [list(r) for r in pipeline_data["vrp_routes"]],
        "routes_with_homes": [list(r) for r in pipeline_data["vrp_routes_with_homes"]],
        "home_indices": list(pipeline_data["home_indices"]),
        "robot_start_xyzs": [list(map(float, xyz)) for xyz in pipeline_data["robot_start_xyzs"]],
        "robot_inspection_wp_indices": [
            list(map(int, idxs)) for idxs in pipeline_data["robot_inspection_wp_indices"]
        ],
        "num_robots": int(pipeline_data["num_robots"]),
        "vrp_status": pipeline_data.get("vrp_status"),
        "vrp_total_cost": pipeline_data.get("vrp_total_cost"),
        "vrp_makespan": pipeline_data.get("vrp_makespan"),
        "vrp_solver": pipeline_data.get("vrp_solver"),
        "alpha": pipeline_data.get("alpha"),
    }
    with open(os.path.join(dir_path, "vrp.json"), "w") as f:
        json.dump(vrp_meta, f, default=float)

    # 5. ExecutionResult
    save_solution(pipeline_data["exec_result"], os.path.join(dir_path, "exec_result"))

    # 6. Manifest (mesh, frustum, args, file index)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "mesh": {
            "path": pipeline_data["mesh_path"],
            "pose": list(pipeline_data["mesh_pose"]),
            "target_length": float(pipeline_data["mesh_target_length"]),
            "scale_factor": float(pipeline_data.get("mesh_scale", 1.0)),
            "bounds_min": list(map(float, pipeline_data["mesh_bounds_min"])),
            "bounds_max": list(map(float, pipeline_data["mesh_bounds_max"])),
        },
        "frustum": dict(pipeline_data["frustum_params"]),
        "args": pipeline_data.get("args", {}),
        "files": {
            "pointcloud": "pointcloud.npz",
            "candidates": "candidates.npz",
            "optimization_result_npz": "optimization_result.npz",
            "optimization_result_json": "optimization_result.json",
            "vrp": "vrp.json",
            "exec_result_npz": "exec_result.npz",
            "exec_result_json": "exec_result.json",
        },
        "has_full_visibility_map": fvm is not None,
    }
    with open(os.path.join(dir_path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def load_pipeline(dir_path: str) -> dict:
    """Load a pipeline directory written by :func:`save_pipeline`.

    Returns a dict with the same keys as the original ``pipeline_data``
    (NumPy arrays, except for ``optimization_result`` and ``exec_result``
    which are reconstructed dataclasses).
    """
    from visibility.core.types import OptimizationResult

    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Pipeline directory not found: {dir_path}")

    with open(os.path.join(dir_path, "manifest.json")) as f:
        manifest = json.load(f)

    pc = np.load(os.path.join(dir_path, "pointcloud.npz"))
    target_points = pc["target_points"]
    normals = pc["normals"]

    cand = np.load(os.path.join(dir_path, "candidates.npz"))
    all_positions = cand["all_positions"]
    all_rotmats = cand["all_rotmats"]
    full_visibility_map = cand["full_visibility_map"] if "full_visibility_map" in cand.files else None

    opt_arrays = np.load(os.path.join(dir_path, "optimization_result.npz"))
    with open(os.path.join(dir_path, "optimization_result.json")) as f:
        opt_meta = json.load(f)
    opt_result = OptimizationResult(
        positions=opt_arrays["positions"],
        rotations=opt_arrays["rotations"],
        visibility_map=opt_arrays["visibility_map"],
        total_coverage=float(opt_meta["total_coverage"]),
        num_viewpoints=int(opt_meta["num_viewpoints"]),
        redundancy=float(opt_meta["redundancy"]),
        optimization_time=float(opt_meta["optimization_time"]),
        selected_indices=opt_arrays["selected_indices"]
        if "selected_indices" in opt_arrays.files
        else None,
    )

    with open(os.path.join(dir_path, "vrp.json")) as f:
        vrp_meta = json.load(f)

    exec_result = load_solution(os.path.join(dir_path, "exec_result"))

    pipeline_data = {
        # Mesh
        "mesh_path": manifest["mesh"]["path"],
        "mesh_pose": manifest["mesh"]["pose"],
        "mesh_target_length": manifest["mesh"]["target_length"],
        "mesh_scale": manifest["mesh"]["scale_factor"],
        "mesh_bounds_min": np.asarray(manifest["mesh"]["bounds_min"], dtype=np.float32),
        "mesh_bounds_max": np.asarray(manifest["mesh"]["bounds_max"], dtype=np.float32),
        # Pointcloud
        "target_points": target_points,
        "normals": normals,
        # Candidates
        "all_positions": all_positions,
        "all_rotmats": all_rotmats,
        "full_visibility_map": full_visibility_map,
        # Frustum
        "frustum_params": manifest["frustum"],
        # Set cover
        "optimization_result": opt_result,
        "selected_positions": opt_result.positions,
        "selected_rotmats": opt_result.rotations,
        "visibility_map": opt_result.visibility_map,
        # VRP
        "vrp_routes": vrp_meta["routes"],
        "vrp_routes_with_homes": vrp_meta["routes_with_homes"],
        "num_robots": vrp_meta["num_robots"],
        "home_indices": vrp_meta["home_indices"],
        "robot_start_xyzs": [np.asarray(xyz, dtype=np.float32) for xyz in vrp_meta["robot_start_xyzs"]],
        "robot_inspection_wp_indices": vrp_meta["robot_inspection_wp_indices"],
        "vrp_status": vrp_meta.get("vrp_status"),
        "vrp_total_cost": vrp_meta.get("vrp_total_cost"),
        "vrp_makespan": vrp_meta.get("vrp_makespan"),
        "vrp_solver": vrp_meta.get("vrp_solver"),
        "alpha": vrp_meta.get("alpha"),
        # Trajectories
        "exec_result": exec_result,
        # Manifest passthrough
        "args": manifest.get("args", {}),
    }
    return pipeline_data
