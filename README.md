# 3D Inspection

GPU-accelerated 3D inspection path planning: visibility analysis, vehicle routing (VRP), and Isaac Lab simulation.

## Prerequisites

| Requirement | Version |
|-------------|---------|
| CUDA Toolkit / driver | >= 12.0 |
| Conda / Mamba | any recent |
| GPU compute capability | >= 7.0 (Volta+) |
| NVIDIA OptiX SDK | >= 7.7 (for `triro` raycasting) |

**OptiX SDK**: Download from the [NVIDIA developer portal](https://developer.nvidia.com/optix). After installing, set the environment variable before running the setup script:
```bash
export OptiX_INSTALL_DIR=/home/troja_robot_lab/NVIDIA-OptiX-SDK-8.0.0  # default used by setup_env.sh
```

---

## Setup

### Option B — Fresh setup (no existing IsaacSim)

```bash
conda create -n inspection python=3.11 -y
conda activate inspection

# 1. PyTorch with CUDA 12.8
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 2. IsaacSim 5.0 + IsaacLab 2.2 (~12.6 GB from NVIDIA PyPI)
pip install "isaacsim[all]==5.0.0.0" isaaclab==2.2.0 \
    --extra-index-url https://pypi.nvidia.com

# 3. RAPIDS — cuGraph (GPU Dijkstra), cuOpt (GPU MIP), cuDF, cuPy
pip install cupy-cuda12x cudf-cu12 cugraph-cu12 cuopt-cu12 \
    --extra-index-url https://pypi.nvidia.com

# 4. triro (not on PyPI — install from GitHub)
pip install "git+https://github.com/lcp29/trimesh-ray-optix.git"

# 5. Install this project (pulls in numpy, scipy, trimesh, evotorch, etc.)
pip install -e .
```

---

## Running the pipeline

```bash
conda activate inspection

# Full 3D inspection pipeline (visibility + VRP + simulation)
python run_full_pipeline.py --solver cuopt

# VRP only (random waypoints)
python VRP/scripts/run_vrp.py --num_robots 2 --random_waypoints 5 --solver cuopt

# Use HiGHS (CPU) instead of cuOpt (GPU)
python VRP/scripts/run_vrp.py --num_robots 2 --random_waypoints 5 --solver highs

# Tests
python -m pytest tests/ -v
```

---

## Package groups

| Group | Install command | Contents |
|-------|----------------|----------|
| Core (auto) | `pip install -e .` | numpy, scipy, trimesh, open3d, matplotlib, pulp, highspy, evotorch, triro |
| `rapids` | `pip install -e ".[rapids]"` | cupy-cuda12x, cudf-cu12, cugraph-cu12, cuopt-cu12 |
| `isaac` | `pip install -e ".[isaac]"` | torch, torchvision, torchaudio, isaacsim, isaaclab |
| `test` | `pip install -e ".[test]"` | pytest |

---

## Use of AI tools

This framework was developed as part of a Bachelor thesis at MFF UK (Charles University). In accordance with Article 4 of Dean's Directive 26/2023, I disclose that I used **Claude Code**, Anthropic's AI coding assistant (powered primarily by the Claude Opus 4 model family, with occasional use of related Anthropic models), as a tool throughout the implementation of this codebase.

AI assistance was used across essentially the entire framework — visibility kernels, the routing and mixed-integer programming layer, the priority-based multi-agent path-finding stage, the Isaac Sim integration, and the experiment scripts under `experiments/e00`–`experiments/e16`. The work was driven by detailed specifications I authored, refined over many iterations, and reviewed line by line; every module was specified, reviewed, tested and edited by me, and AI output was never accepted unmodified.

The algorithmic contributions claimed in the thesis (the GPU batch adaptation of Lien's ε-visibility, the β-aware per-vehicle tour upper bound with forbidden-pair cuts, and the 4D space-time extension of Zhou and Zeng's GPU parallel-frontier A*) are my own designs; AI was used as an implementation aid following those designs.

A more detailed disclosure covering thesis text and literature review is provided in the "Use of AI tools" section of the thesis Preface.
