"""Constants for the visibility package."""

import cupy as cp
import numpy as np

# Numerical stability
NORM_EPS = 1e-12

# Raycast visibility
RAYCAST_TOLERANCE = 1e-4

# CUDA kernel configuration
CUDA_BLOCK_SIZE = 256

# Set-cover optimization defaults
DEFAULT_TARGET_COVERAGE = 0.95
DEFAULT_MAX_VIEWPOINTS = 50

# Expansion sampling
EXPANSION_N_SAMPLES = 20
EXPANSION_RADIUS = 0.5

# Viewpoint sampling
DEFAULT_MAX_DIR_NOISE_RAD = np.deg2rad(25.0)
KNN_DIRECTION_K = 10
GPU_NN_CHUNK_SIZE = 500 # chunk size to cap GPU memory usage during nearest-neighbor queries

# Curvature-weighted sampling
CURVATURE_KNN_K = 20
CURVATURE_POSITION_WEIGHT = 5.0

# Targeted resampling
TARGETED_PROXIMITY_SIGMA_FACTOR = 2.0
PROXIMITY_KNN_FRACTION = 0.01
RESAMPLE_FRACTION = 0.25
DEFAULT_K_COVERAGE = 4  # coverage-redundancy k (Glorieux 2020)

# Optimal resampling (optimizing sampler)
OPT_SAMPLER_POPSIZE = 40       # population size per generation
OPT_SAMPLER_MAXITER = 40       # max generations per optimization run
OPT_SAMPLER_TRAVEL_WEIGHT = 0.1  # weight of travel cost vs coverage in objective
OPT_SAMPLER_TRAVEL_ROT_FRACTION = 0.1  # fraction of travel cost from rotation vs position

# Epsilon-visibility estimation
DELTA_DEFAULT = 0.1           # fallback sampling density when estimation fails
DELTA_SAMPLE_SIZE = 1000      # max points sampled for delta estimation
GAMMA_FALLBACK_DIVISOR = 4.0  # gamma = frustum_far / this when no front-facing points

# Epsilon-visibility aggregation functions
DELTA_AGG_FUNCS = {
    "max": np.max,
    "p99": lambda x: np.percentile(x, 99),
    "p95": lambda x: np.percentile(x, 95),
    "p90": lambda x: np.percentile(x, 90),
}

DELTA_AGG_FUNCS_CP = {
    "max": cp.max,
    "p99": lambda x, **kw: cp.percentile(x, 99, **kw),
    "p95": lambda x, **kw: cp.percentile(x, 95, **kw),
    "p90": lambda x, **kw: cp.percentile(x, 90, **kw),
}

GAMMA_AGG_FUNCS = {
    "median": np.median,
    "mean": np.mean,
    "p10": lambda x: np.percentile(x, 10),
    "p25": lambda x: np.percentile(x, 25),
    "p30": lambda x: np.percentile(x, 30),
    "p40": lambda x: np.percentile(x, 40),
    "p60": lambda x: np.percentile(x, 60),
    "p75": lambda x: np.percentile(x, 75),
    "p90": lambda x: np.percentile(x, 90),
}

GAMMA_AGG_FUNCS_CP = {
    "median": cp.median,
    "mean": cp.mean,
    "p10": lambda x: float(cp.percentile(x, 10)),
    "p25": lambda x: float(cp.percentile(x, 25)),
    "p30": lambda x: float(cp.percentile(x, 30)),
    "p40": lambda x: float(cp.percentile(x, 40)),
    "p60": lambda x: float(cp.percentile(x, 60)),
    "p75": lambda x: float(cp.percentile(x, 75)),
    "p90": lambda x: float(cp.percentile(x, 90)),
}

GAMMA_PERCENTILE = {
    "median": 0.5,
    "p10": 0.1, "p25": 0.25, "p30": 0.3, "p40": 0.4,
    "p60": 0.6, "p75": 0.75, "p90": 0.9,
}
