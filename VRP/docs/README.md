# RAPIDS GPU Setup

The GPU-accelerated distance matrix computation (cuGraph Dijkstra) and the
VRP solver (NVIDIA cuOpt) run directly in the main conda environment.

---

## 1. Prerequisites

| Requirement | Version |
|-------------|---------|
| CUDA Toolkit | >= 12.0 |
| Conda / Mamba | any recent |
| GPU with CC >= 7.0 (Volta+) | -- |

---

## 2. Install RAPIDS packages (pip)

All RAPIDS packages are installed via pip in the main environment alongside
IsaacSim.  No separate conda environment is needed.

```bash
# Install cuGraph, cuDF, cuPy, and cuOpt from NVIDIA's PyPI index
pip install \
  --extra-index-url https://pypi.nvidia.com \
  cugraph-cu12 \
  cudf-cu12 \
  cupy-cuda12x \
  cuopt-cu12

# Verify
python -c "import cugraph; print('cugraph OK')"
python -c "from cuopt.linear_programming import Solve; print('cuOpt OK')"
```

---

## 3. Fallback behaviour

| Stage | Primary (GPU) | Fallback (CPU) |
|-------|--------------|----------------|
| Distance matrix | cuGraph Dijkstra (in-process) | -- |
| VRP solve | cuOpt MILP (in-process) | HiGHS via PuLP |

---

## 4. Quick smoke test

```bash
cd /home/troja_robot_lab/Desktop/Krystof/3D-Inspection
python VRP/scripts/run_vrp.py \
  --num_robots 2 \
  --random_waypoints 4 \
  --solver cuopt \
  --verbose
```

This will:
1. Build (or load cached) occupancy grid from the wreck mesh.
2. Sample 4 random collision-free waypoints.
3. Compute the 4x4 GPU distance matrix via cuGraph (in-process).
4. Solve the 2-vehicle VRP with cuOpt (or HiGHS with `--solver highs`).
5. Resolve traffic-light conflicts.
6. Plan trajectories via Space-Time A*.
7. Print per-robot summary.
