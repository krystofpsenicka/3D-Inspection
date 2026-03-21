"""
VRP Planner – Shared Utilities

  • load_local_robot_config()
  • find_trajectory_collisions()
  • save_solution() / load_solution()
"""

from __future__ import annotations
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import yaml

from VRP.config import (
    ASSETS_PATH,
    BROV_CUBOID_DIMS,
    CONFIGS_PATH,
    ROBOT_CFG_DIR,
)

logger = logging.getLogger(__name__)


# ── Robot config ─────────────────────────────────────────────────────────────

def load_local_robot_config(robot_file: str = "brov.yml") -> dict:
    """Load a robot YAML config from the configs/robot directory.

    Patches ``external_asset_path`` and ``collision_spheres`` to absolute
    paths so the returned dict is self-contained.
    """
    config_path = os.path.join(ROBOT_CFG_DIR, robot_file)
    with open(config_path) as f:
        robot_cfg = yaml.safe_load(f)["robot_cfg"]
    robot_cfg["kinematics"]["external_asset_path"] = ASSETS_PATH
    spheres_file = robot_cfg["kinematics"].get("collision_spheres", "spheres/brov.yml")
    robot_cfg["kinematics"]["collision_spheres"] = os.path.join(
        ROBOT_CFG_DIR, spheres_file
    )
    return robot_cfg


# ── AABB collision check ──────────────────────────────────────────────────────

def find_trajectory_collisions(
    all_traj_positions: List[List[np.ndarray]],
    dims: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int, float]]:
    """Scan replay trajectories for AABB inter-robot collisions.

    Returns
    -------
    collisions : list of (step, robot_a, robot_b, penetration_depth)
    """
    if dims is None:
        dims = np.array(BROV_CUBOID_DIMS, dtype=np.float32)
    dims = np.asarray(dims, dtype=np.float32)
    half = dims / 2.0

    num_robots  = len(all_traj_positions)
    total_steps = max(len(t) for t in all_traj_positions)
    collisions  = []

    for step in range(total_steps):
        for a in range(num_robots):
            if step >= len(all_traj_positions[a]):
                pos_a = all_traj_positions[a][-1][:3]
            else:
                pos_a = all_traj_positions[a][step][:3]
            for b in range(a + 1, num_robots):
                if step >= len(all_traj_positions[b]):
                    pos_b = all_traj_positions[b][-1][:3]
                else:
                    pos_b = all_traj_positions[b][step][:3]
                diff = np.abs(pos_a - pos_b)
                gap  = diff - 2.0 * half
                if np.all(gap < 0):
                    penetration = float(-np.max(gap))
                    collisions.append((step, a, b, penetration))
    return collisions


# ---------------------------------------------------------------------------
# Solution persistence
# ---------------------------------------------------------------------------

def save_solution(result: "ExecutionResult", path: str) -> None:  # noqa: F821
    """Save an :class:`~route_executor.ExecutionResult` to NPZ+JSON.

    Parameters
    ----------
    result:
        The completed execution result returned by
        :class:`~route_executor.RouteExecutor`.
    path:
        Destination file path (extension is replaced with .npz/.json).
    """
    import json

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base = path.rsplit(".", 1)[0] if "." in path else path

    # Pack ragged trajectory arrays
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
        "joint_names": result.joint_names,
        "fail_counts": result.fail_counts,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f)


def load_solution(path: str) -> "ExecutionResult":  # noqa: F821
    """Load an :class:`~route_executor.ExecutionResult` from NPZ+JSON.

    Falls back to pickle for old cached files.

    Parameters
    ----------
    path:
        Path to the saved file (extension is replaced with .npz/.json).

    Returns
    -------
    ExecutionResult
    """
    import json
    from VRP.routing.route_executor import ExecutionResult

    base = path.rsplit(".", 1)[0] if "." in path else path
    npz_path = base + ".npz"
    json_path = base + ".json"

    if os.path.exists(npz_path) and os.path.exists(json_path):
        data = np.load(npz_path)
        with open(json_path) as f:
            meta = json.load(f)

        def unpack_ragged(name):
            flat = data[name]
            offsets = data[f"{name}_offsets"]
            return [list(flat[offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)]

        return ExecutionResult(
            all_traj_positions=unpack_ragged("all_traj_positions"),
            all_traj_velocities=unpack_ragged("all_traj_velocities"),
            all_waypoints=meta["all_waypoints"],
            initial_positions=list(data["initial_positions"]),
            joint_names=meta["joint_names"],
            fail_counts=meta["fail_counts"],
        )
    # Backward compat: pickle
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)
