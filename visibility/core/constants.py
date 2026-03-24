"""Constants for the visibility package."""

import numpy as np

# Numerical stability
NORM_EPS = 1e-12

# Raycast visibility
RAYCAST_TOLERANCE = 1e-4

# CUDA kernel configuration
CUDA_BLOCK_SIZE = 256

# Kernel greedy expansion
KERNEL_N_SAMPLES = 20
KERNEL_RADIUS = 0.5

# Viewpoint sampling
DEFAULT_MAX_DIR_NOISE_RAD = np.deg2rad(25.0)
KNN_DIRECTION_K = 10
GPU_NN_CHUNK_SIZE = 500

# Curvature-weighted sampling
CURVATURE_KNN_K = 20
CURVATURE_POSITION_WEIGHT = 5.0

# Targeted resampling
TARGETED_PROXIMITY_SIGMA_FACTOR = 2.0
PROXIMITY_KNN_FRACTION = 0.01
RESAMPLE_FRACTION = 0.25

# Optimal resampling (Differential Evolution)
DE_POPSIZE = 15       # scipy multiplier (actual pop = DE_POPSIZE * n_dims = 90)
DE_MAXITER = 20       # max generations per DE run
DE_TRAVEL_WEIGHT = 0.1  # weight of travel cost vs coverage in DE objective

# Epsilon-visibility aggregation functions
DELTA_AGG_FUNCS = {
    "max": np.max,
    "p99": lambda x: np.percentile(x, 99),
    "p95": lambda x: np.percentile(x, 95),
    "p90": lambda x: np.percentile(x, 90),
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
