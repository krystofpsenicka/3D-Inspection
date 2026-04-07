"""Solution persistence (NPZ + JSON format)."""

from __future__ import annotations

import json
import os

import numpy as np


def save_solution(result: "ExecutionResult", path: str) -> None:  # noqa: F821
    """Save an ExecutionResult to NPZ + JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base = path.rsplit(".", 1)[0] if "." in path else path

    arrays = {}
    for field_name in ("all_traj_positions", "all_traj_velocities"):
        robot_trajs = getattr(result, field_name)
        all_steps = []
        offsets = [0]
        for robot_steps in robot_trajs:
            if robot_steps:
                all_steps.append(np.stack(robot_steps))
            offsets.append(offsets[-1] + len(robot_steps))
        arrays[field_name] = np.concatenate(all_steps) if all_steps else np.empty((0, 8))
        arrays[f"{field_name}_offsets"] = np.array(offsets, dtype=np.int64)

    arrays["initial_positions"] = np.array(result.initial_positions)
    np.savez_compressed(base + ".npz", **arrays)

    meta = {
        "all_waypoints": result.all_waypoints,
        "fail_counts": result.fail_counts,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f)


def load_solution(path: str) -> "ExecutionResult":  # noqa: F821
    """Load an ExecutionResult from NPZ + JSON.

    Raises FileNotFoundError if the files do not exist.
    """
    from VRP.core.types import ExecutionResult

    base = path.rsplit(".", 1)[0] if "." in path else path
    npz_path = base + ".npz"
    json_path = base + ".json"

    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Solution files not found: {npz_path} and/or {json_path}"
        )

    data = np.load(npz_path)
    with open(json_path) as f:
        meta = json.load(f)

    def unpack_ragged(name):
        flat = data[name]
        offsets = data[f"{name}_offsets"]
        return [list(flat[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)]

    return ExecutionResult(
        all_traj_positions=unpack_ragged("all_traj_positions"),
        all_traj_velocities=unpack_ragged("all_traj_velocities"),
        all_waypoints=meta["all_waypoints"],
        initial_positions=list(data["initial_positions"]),
        fail_counts=meta["fail_counts"],
    )
