#!/usr/bin/env bash
#PBS -N 3d-insp-test
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb:cuda_version=12.0
#PBS -l walltime=1:00:00
#PBS -q gpu
#PBS -m abe
#
# Submit with:  qsub scripts/metacentrum/run_tests.sh
set -euo pipefail

module load cuda/cuda-12.6.3-gcc
module load mambaforge
conda activate inspection

REPO_DIR="${PBS_O_WORKDIR}"
WORK_DIR="${SCRATCHDIR}/3d-inspection"
mkdir -p "${WORK_DIR}"
cp -a "${REPO_DIR}/." "${WORK_DIR}/"
cd "${WORK_DIR}"

nvidia-smi
python -m pytest tests/ -v 2>&1 | tee "${REPO_DIR}/test_results.log"

clean_scratch
