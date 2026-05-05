# 3D Inspection

GPU-accelerated 3D inspection path planning: visibility analysis, vehicle routing (VRP), space-time multi-agent path-finding, and Isaac Sim simulation.

The full pipeline — from mesh to executed multi-AUV trajectories — is in [`scripts/run_full_pipeline.py`](scripts/run_full_pipeline.py); the matching Isaac Sim replay is in [`scripts/visualize_full_pipeline.py`](scripts/visualize_full_pipeline.py). For a complete module reference, configuration tables, and per-script CLI flags, see [`docs.md`](docs.md).

## Prerequisites

| Requirement | Version |
|-------------|---------|
| NVIDIA driver | recent enough for CUDA 12.8 wheels (e.g. ≥ 535) |
| GPU compute capability | ≥ 7.0 (Volta+) |
| Conda / Mamba | any recent |
| Disk free | ~15 GB (IsaacSim wheels + RAPIDS) |
| NVIDIA OptiX SDK | ≥ 7.7 (used by `triro` raycasting) |

**OptiX SDK** is gated behind a free NVIDIA developer account; do this *before* starting the conda setup below, because step 5 (`triro`) will not build without `OptiX_INSTALL_DIR` pointing at a real install:

1. Create / sign in at <https://developer.nvidia.com/login> (free account; approval is usually instant).
2. Go to <https://developer.nvidia.com/designworks/optix/download>, accept the license, and download the Linux 64-bit installer (`NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh` or newer; the repo was developed against 8.0.0).
3. Run the installer to unpack it, e.g.:
   ```bash
   chmod +x NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
   ./NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
   # accept the license, choose an install dir (default $HOME/NVIDIA-OptiX-SDK-8.0.0)
   ```
4. Note the resulting directory — you'll point `OptiX_INSTALL_DIR` at it in step 5.

OptiX is a header-only / library bundle; no kernel module or driver beyond what CUDA already ships.

## Setup

The canonical path. All steps must be run inside a single conda environment.

```bash
# 1. Create and activate the env (Python 3.11)
conda create -n inspection python=3.11 -y
conda activate inspection

# 2. PyTorch with CUDA 12.8 — install BEFORE triro, which depends on torch at build time
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. IsaacSim 5.0 + IsaacLab 2.2 (~12.6 GB from NVIDIA's PyPI mirror)
pip install "isaacsim[all]==5.0.0.0" isaaclab==2.2.0 \
    --extra-index-url https://pypi.nvidia.com

# 4. RAPIDS — cuPy, cuDF, cuGraph (GPU Dijkstra), cuOpt (GPU MIP).
#    Pinned to 26.2 because RAPIDS 26.4 ships numba 0.64 which conflicts with
#    isaacsim-core's numba==0.59.1 pin (and needs a newer `coverage`).
pip install cupy-cuda12x "cudf-cu12==26.2.*" "cugraph-cu12==26.2.*" "cuopt-cu12==26.2.*" \
    --extra-index-url https://pypi.nvidia.com

# 5. triro (OptiX raycasting bindings) — not on PyPI, install from GitHub.
#    Point OptiX_INSTALL_DIR at the directory you unpacked the OptiX SDK into
#    (see "Prerequisites" above); the install fails without it.
export OptiX_INSTALL_DIR=$HOME/NVIDIA-OptiX-SDK-8.0.0   # adjust to your path
pip install "git+https://github.com/lcp29/trimesh-ray-optix.git"

# 6. Install this project (pulls in numpy, scipy, trimesh, open3d, pulp,
#    highspy, evotorch, ompl, ...). triro is listed as a dep but is already
#    satisfied by step 5.
pip install -e .
```

`pip install -e ".[test]"` adds `pytest` if you want to run the test suite.

## External assets

The following files are not in git and must be supplied separately.

### Mesh (default scene)

The default ship mesh lives at `models/duke_of_lancaster_uk_clipped.glb` and is loaded by `VRP/core/constants.py:MESH_PATH`. The pipeline rescales it to 50 m along its longest axis (`--mesh_target_length`). To use a different mesh, either drop a `.glb` / `.obj` / `.stl` into `models/` and pass it via `MESH_PATH`, or pass an explicit path through code that calls `shared/mesh_loader.py:load_and_transform_mesh`. The optional TOSCA dataset used by `experiments/e10_cross_model` is expected at `models/TOSCA-dataset/`.

### BROV USD (optional, Isaac Sim replay only)

`scripts/visualize_full_pipeline.py --use-brov-usd` references `assets/robot/brov/BROV_high.usd` (~345 MB). Without the file or the flag, the replay falls back to lightweight cuboids, so this asset is **optional**.

To obtain the USD: follow the README of NVIDIA's OceanSim project (<https://github.com/umfieldrobotics/OceanSim>), which hosts the BROV USD on Google Drive. Place the downloaded file at `assets/robot/brov/BROV_high.usd`.

## Quickstart

```bash
conda activate inspection

# Full pipeline (default 5 robots, 95% coverage, GPU MIP, outside inspection).
# Saves all intermediate data to outputs/full_pipeline/.
python scripts/run_full_pipeline.py --solver cuopt

# CPU-only fallback (HiGHS instead of cuOpt).
python scripts/run_full_pipeline.py --solver highs

# Isaac Sim phase-by-phase replay of the saved pipeline.
# Keys: N / right arrow → next phase, P / left arrow → previous, Q / Esc → quit.
python scripts/visualize_full_pipeline.py outputs/full_pipeline

# Tests
python -m pytest tests/ -v
```

For the full CLI reference (`--num_robots`, `--target_coverage`, `--side`, `--resample_fraction`, `--curvature_weighting`, `--k_coverage`, `--alpha`, frustum parameters, …), see the *Scripts* section of [`docs.md`](docs.md).

## Documentation

- [`docs.md`](docs.md) — architecture overview, repository layout, full module reference (every public class and function in `VRP/`, `visibility/`, `shared/`, `visualization_isaac/`), configuration constants, all script flags, experiment index, test index.
- [`VRP/docs/README.md`](VRP/docs/README.md) — RAPIDS GPU stack deep-dive (cuGraph Dijkstra, cuOpt MIP, HiGHS fallback).

## Use of AI tools

This framework was developed as part of a Bachelor thesis at MFF UK (Charles University). In accordance with Article 4 of Dean's Directive 26/2023, I disclose that I used **Claude Code**, Anthropic's AI coding assistant (powered primarily by the Claude Opus 4 model family, with occasional use of related Anthropic models), as a tool throughout the implementation of this codebase.

AI assistance was used across essentially the entire framework — visibility kernels, the routing and mixed-integer programming layer, the priority-based multi-agent path-finding stage, the Isaac Sim integration, and the experiment scripts under `experiments/e00`–`experiments/e16`. The work was driven by detailed specifications I authored, refined over many iterations, and reviewed line by line; every module was specified, reviewed, tested and edited by me, and AI output was never accepted unmodified.

The algorithmic contributions claimed in the thesis (the GPU batch adaptation of Lien's ε-visibility, the β-aware per-vehicle tour upper bound with forbidden-pair cuts, and the 4D space-time extension of Zhou and Zeng's GPU parallel-frontier A*) are my own designs; AI was used as an implementation aid following those designs.

A more detailed disclosure covering thesis text and literature review is provided in the "Use of AI tools" section of the thesis Preface.
