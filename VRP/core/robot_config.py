"""Robot YAML configuration loader."""

from __future__ import annotations

import os

import yaml

from .constants import ASSETS_PATH, ROBOT_CFG_DIR


def load_local_robot_config(robot_file: str = "brov.yml") -> dict:
    """Load a robot YAML config and patch asset paths to absolute."""
    config_path = os.path.join(ROBOT_CFG_DIR, robot_file)
    with open(config_path) as f:
        robot_cfg = yaml.safe_load(f)["robot_cfg"]
    robot_cfg["kinematics"]["external_asset_path"] = ASSETS_PATH
    spheres_file = robot_cfg["kinematics"].get("collision_spheres", "spheres/brov.yml")
    robot_cfg["kinematics"]["collision_spheres"] = os.path.join(
        ROBOT_CFG_DIR, spheres_file
    )
    return robot_cfg
