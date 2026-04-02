"""VRP constants — all tuneable parameters in one place.

Organized by subsystem following the pattern in
``visibility/core/constants.py``.
"""

import math
import os

# ── Paths ────────────────────────────────────────────────────────────────────

VRP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(VRP_ROOT)
ASSETS_PATH = os.path.join(PROJECT_ROOT, "assets")
CONFIGS_PATH = os.path.join(PROJECT_ROOT, "configs")
ROBOT_CFG_DIR = os.path.join(CONFIGS_PATH, "robot")

MESH_PATH = os.path.join(PROJECT_ROOT, "models", "duke_of_lancaster_uk_clipped.glb")
MESH_POSE = [0, 0, 1.5, 0.0, 1.0, 0.0, 0.0]
MESH_TARGET_LENGTH = 50.0

# ── Occupancy grid ───────────────────────────────────────────────────────────

VOXEL_RESOLUTION = 0.10        # metres per voxel edge
ROBOT_RADIUS = 0.35            # collision sphere radius (brov.yml)
INFLATION_VOXELS = int(ROBOT_RADIUS / VOXEL_RESOLUTION) + 1

# ── Robot physical constants ─────────────────────────────────────────────────

BROV_CUBOID_DIMS = [0.7, 0.5, 0.35]
STATIC_OBSTACLES = {}

# ── RAPIDS / cuGraph subprocess ──────────────────────────────────────────────


def _find_rapids_python() -> str:
    """Search common conda/mamba prefixes for a rapids_solver environment."""
    search_roots = [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/mambaforge"),
        os.path.expanduser("~/.conda"),
    ]
    for root in search_roots:
        candidate = os.path.join(root, "envs", "rapids_solver", "bin", "python")
        if os.path.isfile(candidate):
            return candidate
    return os.path.expanduser("~/miniconda3/envs/rapids_solver/bin/python")


RAPIDS_PYTHON = os.environ.get("RAPIDS_PYTHON", _find_rapids_python())

# ── Space-Time A* collision avoidance ────────────────────────────────────────

SPACE_TIME_RESOLUTION = 0.50
SPACE_TIME_DT = SPACE_TIME_RESOLUTION / 2.0          # 0.25 s
SPACE_TIME_MAX_HORIZON_S = 400.0
SPACE_TIME_MAX_WAIT = 200
SPACE_TIME_DWELL_S = 2.0
SPLINE_SAFETY_VOXELS = 1

# ── Trajectory replay ────────────────────────────────────────────────────────

TRAJ_DT = 0.02

# ── Space-Time A* tuning ─────────────────────────────────────────────────────

ST_ASTAR_MAX_EXPANSIONS = 500_000
OMPL_SIMPLIFY_MAX_TIME = 0.5
SPACE_TIME_HOP_DISTANCE = 2.0
SNAP_TO_FREE_MAX_RADIUS = 10

# ── Camera geometry (from URDF kinematic chain) ──────────────────────────────

CAMERA_OFFSET_FORWARD = 0.30
CAMERA_OFFSET_UP = 0.05

# ── AUV dynamics ─────────────────────────────────────────────────────────────

AUV_CRUISE_SPEED = 2.0
AUV_MAX_ACCEL = 1.5

# ── MIP VRP solver ───────────────────────────────────────────────────────────

VRP_ALPHA = 1.0
MIP_TIME_LIMIT = 120
MIP_GAP = 0.05

# ── VRP ↔ path-planning feedback loop ───────────────────────────────────────

VRP_FEEDBACK_ITERATIONS = 3
VRP_FEEDBACK_THRESHOLD = 0.20

# ── GPU search (parallel A*) ────────────────────────────────────────────────

CUDA_BLOCK_SIZE = 256
GPU_SEARCH_MAX_ITERATIONS = 100_000
