"""VRP constants"""

import math
import os

import cupy as cp
import numpy as np

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
ROBOT_RADIUS = 0.35            # collision sphere radius
INFLATION_VOXELS = int(ROBOT_RADIUS / VOXEL_RESOLUTION) + 1

# ── Robot physical constants ─────────────────────────────────────────────────


# ── Space-Time A* collision avoidance ────────────────────────────────────────

SPACE_TIME_RESOLUTION = 0.50
SPACE_TIME_DT = SPACE_TIME_RESOLUTION / 2.0          # 0.25 s
SPACE_TIME_MAX_HORIZON_S = 400.0
SPACE_TIME_MAX_WAIT = 200
SPACE_TIME_DWELL_S = 2.0
SPACE_TIME_SAFETY_FACTOR = 3
SPACE_TIME_MIN_LEG_STEPS = 20
SPLINE_SAFETY_VOXELS = 1

# ── Trajectory replay ────────────────────────────────────────────────────────

TRAJ_DT = 0.02

# ── Space-Time A* tuning ─────────────────────────────────────────────────────

ST_ASTAR_MAX_EXPANSIONS = 500_000
OMPL_SIMPLIFY_MAX_TIME = 0.5
# Minimum gap (in time steps) between the estimated traversal time and
# any reserved time step at the same voxel during OMPL path smoothing.
PATH_SMOOTHER_RESERVATION_MARGIN = 10

# ── Camera offset ────────────────────────────────────────────────────────────

CAMERA_OFFSET_FORWARD = 0.30
CAMERA_OFFSET_UP = 0.05

# ── AUV dynamics ─────────────────────────────────────────────────────────────

AUV_CRUISE_SPEED = 2.0
AUV_MAX_ACCEL = 1.5

# ── MIP VRP solver ───────────────────────────────────────────────────────────

VRP_ALPHA = 1.0
MIP_TIME_LIMIT = 120
# Relative optimality gap: the MIP solver stops when
# (best_bound - incumbent) / incumbent <= MIP_GAP.
# 0.05 = within 5% of optimal. Lower = better solution, longer solve.
MIP_GAP = 0.05

# ── GPU search (parallel A*) ────────────────────────────────────────────────

CUDA_BLOCK_SIZE = 256
GPU_SEARCH_MAX_ITERATIONS = 100_000

# ── 26/27-connected offsets and weights (CuPy, GPU-resident) ───────────────

OFFSETS_26 = cp.array([
    (di, dj, dk)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
], dtype=cp.int32)

WEIGHTS_26 = cp.sqrt((OFFSETS_26.astype(cp.float32) ** 2).sum(axis=1))

OFFSETS_27 = cp.concatenate([OFFSETS_26, cp.zeros((1, 3), dtype=cp.int32)])
WEIGHTS_27 = cp.concatenate([WEIGHTS_26, cp.zeros(1, dtype=cp.float32)])
