#!/usr/bin/env bash
# Setup conda environment for 3D-Inspection on MetaCentrum.
# Run once after cloning the repo:
#   bash scripts/metacentrum/setup_env.sh
set -euo pipefail

ENV_NAME="inspection"

# MetaCentrum modules -- load CUDA toolkit and conda
module load cuda/cuda-12.6.3-gcc
module load mambaforge

# Create conda env
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists — skipping create."
else
    mamba create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"

# PyTorch with CUDA 12
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# RAPIDS (CuPy, cuDF, cuGraph, cuOpt) -- needs NVIDIA PyPI
pip install cupy-cuda12x cudf-cu12 cugraph-cu12 cuopt-cu12 \
    --extra-index-url https://pypi.nvidia.com

# triro (OptiX raycasting) -- not on PyPI
pip install "git+https://github.com/lcp29/trimesh-ray-optix.git"

# Install this project in editable mode (pulls numpy, scipy, trimesh, open3d, etc.)
pip install -e ".[test]"

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
