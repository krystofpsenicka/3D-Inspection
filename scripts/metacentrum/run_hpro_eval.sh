#!/usr/bin/env bash
#PBS -N hpro-eval
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb:cuda_version=12.0
#PBS -l walltime=2:00:00
#PBS -q gpu
#PBS -m abe
#
# Frustum-restricted HPRO (HPRO_limited) Stage-1 evaluation.
#
# Submit with:  qsub scripts/metacentrum/run_hpro_eval.sh
#
# Optional knobs (override at submit time), e.g.:
#   qsub -v MESH_DIR=models,NUM_POSES=40,NUM_POINTS=10000 scripts/metacentrum/run_hpro_eval.sh
#
# Results (hpro_eval.csv, *.png, summary.txt) and the smoke-test log are copied
# back to  $PBS_O_WORKDIR/hpro/results/  and  $PBS_O_WORKDIR/hpro_smoke.log .
set -euo pipefail

# ── Tunables (env-overridable via qsub -v) ───────────────────────────
MESH_DIR="${MESH_DIR:-hpro}"
NUM_POSES="${NUM_POSES:-20}"
NUM_POINTS="${NUM_POINTS:-10000}"
CONDA_ENV="${CONDA_ENV:-inspection}"   # override: qsub -v CONDA_ENV=hpro ...

# ── Modules ──────────────────────────────────────────────────────────
module load cuda/cuda-12.6.3-gcc
module load mambaforge
conda activate "${CONDA_ENV}"

# ── Copy repo to fast scratch ────────────────────────────────────────
REPO_DIR="${PBS_O_WORKDIR}"
WORK_DIR="${SCRATCHDIR}/3d-inspection"
mkdir -p "${WORK_DIR}"
cp -a "${REPO_DIR}/." "${WORK_DIR}/"
cd "${WORK_DIR}/hpro"

echo "=== GPU info ==="
nvidia-smi

# ── 1. CPU smoke test (correctness gate) ─────────────────────────────
echo "=== Smoke test ==="
python smoke_test.py 2>&1 | tee "${REPO_DIR}/hpro_smoke.log"

# ── 2. Quantitative evaluation ───────────────────────────────────────
echo "=== Evaluation ==="
python eval_frustum.py \
    --mesh_dir "${MESH_DIR}" \
    --num_poses "${NUM_POSES}" \
    --num_points "${NUM_POINTS}" \
    --out results \
    --no_show 2>&1 | tee "${REPO_DIR}/hpro_eval.log"

# ── Copy results back ────────────────────────────────────────────────
mkdir -p "${REPO_DIR}/hpro/results"
cp -v results/* "${REPO_DIR}/hpro/results/"

# ── Cleanup scratch ──────────────────────────────────────────────────
clean_scratch
