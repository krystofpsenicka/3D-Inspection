#!/usr/bin/env bash
#PBS -N 3d-inspection
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=20gb:cuda_version=12.0
#PBS -l walltime=4:00:00
#PBS -q gpu
#PBS -m abe
#
# Submit with:  qsub scripts/metacentrum/run_pipeline.sh
#
# Output goes to $PBS_O_WORKDIR/pipeline_data.*
# Logs go to $PBS_O_WORKDIR/3d-inspection.{o,e}<jobid>
set -euo pipefail

# ── Modules ──────────────────────────────────────────────────────────
module load cuda/cuda-12.6.3-gcc
module load mambaforge
conda activate inspection

# ── Copy data to fast scratch ────────────────────────────────────────
REPO_DIR="${PBS_O_WORKDIR}"
WORK_DIR="${SCRATCHDIR}/3d-inspection"
mkdir -p "${WORK_DIR}"
cp -a "${REPO_DIR}/." "${WORK_DIR}/"
cd "${WORK_DIR}"

# OptiX -- set if the SDK is installed at a shared path on the cluster.
# Check with:  module avail optix  or  ls /software/optix
# export OptiX_INSTALL_DIR=/software/optix/NVIDIA-OptiX-SDK-8.0.0

# ── Run ──────────────────────────────────────────────────────────────
echo "=== GPU info ==="
nvidia-smi

echo "=== Running pipeline ==="
python run_full_pipeline.py \
    --solver cuopt \
    --num_robots 5 \
    --output pipeline_data.pkl \
    --verbose

# ── Copy results back ───────────────────────────────────────────────
cp -v pipeline_data* "${REPO_DIR}/"

# ── Cleanup scratch ─────────────────────────────────────────────────
clean_scratch
