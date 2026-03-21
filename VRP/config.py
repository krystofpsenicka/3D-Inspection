"""
VRP Planner – Central Configuration
All tuneable constants and paths in one place.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
VRP_ROOT      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(VRP_ROOT)
ASSETS_PATH   = os.path.join(PROJECT_ROOT, "assets")
CONFIGS_PATH  = os.path.join(PROJECT_ROOT, "configs")
ROBOT_CFG_DIR = os.path.join(CONFIGS_PATH, "robot")

MESH_PATH = os.path.join(PROJECT_ROOT, "models", "duke_of_lancaster_uk_clipped.glb")
# Pose [x, y, z, qw, qx, qy, qz] for the mesh in the Isaac Sim stage
# 180° rotation about X-axis (qw=0, qx=1) to flip the GLB mesh right-side up.
MESH_POSE = [0, 0, 1.5, 0.0, 1.0, 0.0, 0.0]
# Target length (metres) of the ship along its longest axis.
# The mesh is uniformly scaled at load time so that
# mesh.extents.max() == MESH_TARGET_LENGTH.
MESH_TARGET_LENGTH = 50.0

# ── Occupancy grid ─────────────────────────────────────────────────────────────
VOXEL_RESOLUTION = 0.10        # metres per voxel edge
ROBOT_RADIUS     = 0.35        # collision sphere radius (brov.yml)
# Inflate occupancy by this many voxels on each side
INFLATION_VOXELS = int(ROBOT_RADIUS / VOXEL_RESOLUTION) + 1   # ≥ 4

# ── Robot physical constants ──────────────────────────────────────────────────
BROV_CUBOID_DIMS   = [0.7, 0.5, 0.35]

# Static obstacles – empty dict; callers iterate over it, so no-op.
STATIC_OBSTACLES = {}

# ── cuOpt / cuGraph subprocess ─────────────────────────────────────────────────
# Path to the Python binary inside the RAPIDS conda env.
# Override at runtime with env-var RAPIDS_PYTHON if conda path differs.
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
    # fall back to the miniconda3 path; will produce a clear error at runtime
    return os.path.expanduser("~/miniconda3/envs/rapids_solver/bin/python")

RAPIDS_PYTHON = os.environ.get("RAPIDS_PYTHON", _find_rapids_python())

# Service time per waypoint for cuOpt temporal separation (seconds)
CUOPT_SERVICE_TIME = 2

# ── OR-Tools fallback settings ─────────────────────────────────────────────────
ORTOOLS_TIME_LIMIT_S = 30
ORTOOLS_FIRST_SOLUTION = "PARALLEL_CHEAPEST_INSERTION"
ORTOOLS_LOCAL_SEARCH   = "GUIDED_LOCAL_SEARCH"

# ── Space-Time A* collision avoidance ──────────────────────────────────────
# Coarse spatial resolution for the 4-D reservation table (metres).
# Chosen so the dense 4-D array fits comfortably in RAM (~80 MB).
SPACE_TIME_RESOLUTION = 0.50
# Time step for the coarse Space-Time A* search (seconds).
# At 2 m/s cruise the robot moves exactly 1 coarse voxel per step.
SPACE_TIME_DT = SPACE_TIME_RESOLUTION / 2.0          # 0.25 s
# Maximum planning horizon (seconds).  Constrains reservation table size.
SPACE_TIME_MAX_HORIZON_S = 400.0
# Maximum contiguous wait steps before declaring a deadlock
SPACE_TIME_MAX_WAIT = 200
# Inspection-dwell hold time at each waypoint (seconds)
SPACE_TIME_DWELL_S = 2.0

# Extra reservation‐table inflation (in coarse voxels) to compensate for
# cubic-spline overshoot during smooth interpolation.  At 0.5 m resolution a
# margin of 1 adds 0.5 m clearance on each side – well above the ~0.15 m
# worst-case spline deviation.
SPLINE_SAFETY_VOXELS = 1

# ── Trajectory replay ─────────────────────────────────────────────────────────
TRAJ_DT = 0.02                   # seconds – replay sample period

# ── Space-Time A* tuning ─────────────────────────────────────────────────────
ST_ASTAR_MAX_EXPANSIONS = 500_000   # default expansion budget
OMPL_SIMPLIFY_MAX_TIME = 0.5       # seconds – OMPL path simplifier budget
SPACE_TIME_HOP_DISTANCE = 2.0      # metres – stride for straight-line hops
SNAP_TO_FREE_MAX_RADIUS = 10       # BFS shells for snap-to-free

# ── Camera geometry (from URDF kinematic chain) ───────────────────────────────
# Total offset of the camera_optical_frame from base_link along the robot's
# forward (+X) axis at zero joint angles (0.20 + 0.02 + 0.02 + 0.06 m).
CAMERA_OFFSET_FORWARD = 0.30    # metres – camera is this far ahead of body centre
CAMERA_OFFSET_UP      = 0.05   # metres – camera is this far above body centre

# ── AUV dynamics ──────────────────────────────────────────────────────────────
AUV_CRUISE_SPEED = 2.0          # m/s nominal cruise speed
AUV_MAX_ACCEL    = 1.5          # m/s² (for future trapezoidal profile)

# ── MIP makespan solver ──────────────────────────────────────────────────────
VRP_OBJECTIVE     = "makespan"    # "makespan" or "total_distance"
MIP_TIME_LIMIT    = 120           # seconds – wall-clock limit for MIP solver
MIP_GAP           = 0.05          # relative optimality gap (5 %)

# ── VRP ↔ path-planning feedback loop ────────────────────────────────────────
VRP_FEEDBACK_ITERATIONS = 3       # max re-solve iterations
VRP_FEEDBACK_THRESHOLD  = 0.20    # accept if actual makespan within 20 % of VRP estimate

