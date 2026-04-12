#!/usr/bin/env bash
# Creates the `inspection` conda environment for the 3D-Inspection project.
#
# The heavy torch / IsaacSim / IsaacLab / Omniverse packages (~14 GB) are
# symlinked from the existing `isaaclab` conda environment rather than
# re-downloaded.  All other dependencies are installed fresh via pip.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Prerequisites:
#   - CUDA 12.x driver
#   - conda / mamba in PATH
#   - GPU with compute capability >= 7.0 (Volta+)
#   - NVIDIA OptiX SDK >= 7.7 (for triro raycasting)
#     https://developer.nvidia.com/optix
#     After install: export OptiX_INSTALL_DIR=/path/to/optix

set -euo pipefail

ENV_NAME="inspection"
ISAACLAB_SP="/home/troja_robot_lab/miniconda3/envs/isaaclab/lib/python3.11/site-packages"
NVIDIA_INDEX="https://pypi.nvidia.com"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# OptiX SDK path — required to build triro. Override with:
#   export OptiX_INSTALL_DIR=/path/to/optix
export OptiX_INSTALL_DIR="${OptiX_INSTALL_DIR:-/home/troja_robot_lab/NVIDIA-OptiX-SDK-8.0.0}"

# ---------------------------------------------------------------------------
echo "=== Creating conda env: ${ENV_NAME} (Python 3.11) ==="
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "  '${ENV_NAME}' already exists — skipping creation."
else
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

# ---------------------------------------------------------------------------
echo "=== Installing pip dependencies ==="
conda run -n "${ENV_NAME}" pip install \
    "numpy>=2.0,<2.3" \
    "scipy>=1.10" \
    "trimesh" \
    "open3d" \
    "matplotlib" \
    "Pillow" \
    "pyyaml" \
    "pulp" \
    "highspy" \
    "evotorch"

# ---------------------------------------------------------------------------
echo "=== Installing RAPIDS packages (cuDF, cuGraph, cuOpt, cuPy) ==="
conda run -n "${ENV_NAME}" pip install \
    --extra-index-url "${NVIDIA_INDEX}" \
    "cupy-cuda12x" \
    "cudf-cu12" \
    "cugraph-cu12" \
    "cuopt-cu12"

# ---------------------------------------------------------------------------
echo "=== Symlinking torch / IsaacSim / IsaacLab from isaaclab env ==="
DST="$(conda run -n "${ENV_NAME}" python -c 'import site; print(site.getsitepackages()[0])')"

symlink_glob() {
    local pattern="$1"
    for item in "${ISAACLAB_SP}"/${pattern}; do
        [ -e "${item}" ] || continue
        ln -sfn "${item}" "${DST}/$(basename "${item}")"
    done
}

# PyTorch (~2.1 GB)
symlink_glob "torch"
symlink_glob "torch-*.dist-info"
symlink_glob "functorch"
symlink_glob "torchgen"
symlink_glob "torchaudio"
symlink_glob "torchaudio-*.dist-info"
symlink_glob "torchvision"
symlink_glob "torchvision.libs"
symlink_glob "torchvision-*.dist-info"

# IsaacSim + IsaacLab + Omniverse (~12.6 GB)
symlink_glob "isaacsim"
symlink_glob "isaacsim_*"
symlink_glob "isaaclab"
symlink_glob "isaaclab-*.dist-info"
symlink_glob "omni"
symlink_glob "omniverse_kit-*.dist-info"

# jaxtyping (triro dependency, already present in isaaclab)
symlink_glob "jaxtyping"
symlink_glob "jaxtyping-*.dist-info"

# ---------------------------------------------------------------------------
echo "=== Installing triro (OptiX raycasting — torch must be symlinked first) ==="
# triro is not on PyPI; install from GitHub at the version used in isaaclab.
# --no-build-isolation is required because setup.py imports torch at build time,
# and pip's isolated build subprocess cannot see the symlinked torch.
conda run -n "${ENV_NAME}" pip install --no-build-isolation \
    "git+https://github.com/lcp29/trimesh-ray-optix.git@004e08f99768844e81fb436881ec99c493f095fc"

# ---------------------------------------------------------------------------
echo "=== Installing project in editable mode ==="
conda run -n "${ENV_NAME}" pip install -e "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
echo ""
echo "=== Done! ==="
echo "  conda activate ${ENV_NAME}"
echo ""
echo "NOTE: triro (OptiX raycasting) requires NVIDIA OptiX SDK >= 7.7."
echo "      Download: https://developer.nvidia.com/optix"
echo "      Then set: export OptiX_INSTALL_DIR=/path/to/optix"
