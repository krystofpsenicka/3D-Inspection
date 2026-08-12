#!/usr/bin/env bash
# Camera-ready experiment orchestration. Fully resumable: safe to Ctrl-C and
# re-launch at any time -- each experiment's --resume skips finished rows.
# Runs experiments sequentially (no GPU contention) in priority order, each in a
# bounded restart loop so a rare cuOpt segfault/OOM restarts with a clean process
# but a deterministically-failing row cannot infinite-loop.
#
#   bash experiments/run_camera_ready.sh              # run all, in order
#   bash experiments/run_camera_ready.sh e11 e13      # run only these stages
#
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate inspection      # conda activate.d scripts reference unbound vars; keep set -u off until after
set -u
export PYTHONPATH="$PWD"

LOGDIR="experiments/results/_camera_ready_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MAX_RETRIES=15      # per stage: guards against an infinite segfault loop
ATTEMPT_TIMEOUT=28800  # 8h hard cap per attempt; a true hang is killed and --resume retries

run_stage () {
  local name="$1"; shift
  local log="$LOGDIR/${name}_${STAMP}.log"
  echo "========================================================================"
  echo ">>> STAGE $name  |  $(date)  |  log: $log"
  echo ">>> cmd: python -m experiments.$* --resume"
  echo "========================================================================"
  local try=0
  until timeout -k 60 "$ATTEMPT_TIMEOUT" python -u -m "experiments.$@" --resume >>"$log" 2>&1; do
    rc=$?
    try=$((try+1))
    echo "!!! $name exited rc=$rc (attempt $try/$MAX_RETRIES) $(date) -- see $log"
    if [ "$try" -ge "$MAX_RETRIES" ]; then
      echo "!!! $name hit retry cap; moving on. Re-run later to finish remaining rows."
      break
    fi
    sleep 5
  done
  echo "<<< STAGE $name done (or capped)  |  $(date)"
  # final summary table into its own log (reuse full args so a custom
  # --output_dir is honoured; --plots_only ignores the run-only flags)
  python -u -m "experiments.$@" --plots_only >>"$log" 2>&1 || true
  echo ">>> summary appended to $log"
}

# ---- stage table: name -> module + args (priority order) --------------------
declare -A STAGES
STAGES[e14]="e14_cp_scaling --dim both --seeds 42 123 7 2024 314"
STAGES[e05]="e05_visibility_comparison --seeds 42 123 7 2024 314 999 55 8888 1337 2025"
STAGES[e11]="e11_vrp_cuts_ablation --waypoints 30 50 --seeds 42 123 7 2024 314 --backend cuopt"
STAGES[e13]="e13_collision_coverage_audit --seeds 42 123 7 2024 314"
STAGES[e08]="e08_vrp_alpha_blending --seeds 42 123 7 2024 314"
STAGES[e10]="e10_cross_model --seeds 42 123 7 2024 314"
STAGES[e12]="e12_mapf_ablation --mode both --seeds 42 123 7"
# E4 in the plan: cheap no-VRP re-runs to 10 seeds (R4.19 broad seed coverage).
SEEDS10="42 123 7 2024 314 999 55 8888 1337 2025"
STAGES[e01]="e01_sampling_strategy --seeds $SEEDS10"
STAGES[e02]="e02_candidate_scaling --seeds $SEEDS10"
STAGES[e06]="e06_set_cover_optimizers --seeds $SEEDS10"
# E15: high-budget cut ablation. Runs LAST; a single solve can take the full
# budget, so cost is unpredictable (~4 configs x 3 seeds x up to time_limit).
STAGES[e15]="e15_vrp_cuts_highbudget --time_limit 1800 --waypoints 50 --seeds 42 123 7"
# Stage F (plan item F): additional real high-complexity meshes, run VERY LAST.
# Separate output_dir so it stays isolated from the main Duke+TOSCA E10 analysis.
STAGES[eF]="e10_cross_model --models armadillo dragon happy_buddha --seeds 42 123 7 --output_dir experiments/results/e10_extra_real"

# Post-fix collision re-audit (SPLINE_SAFETY_VOXELS 1->3). Fresh output dirs so
# --resume does not skip the old pre-fix rows; E12 collisions at the full 20-trial
# budget across K, E13 with the new margin+penetration split.
STAGES[e12c]="e12_mapf_ablation --mode collisions --coord_trials 20 --seeds 42 123 7 --output_dir experiments/results/e12_collisions_fixed"
STAGES[e13f]="e13_collision_coverage_audit --seeds 42 123 7 2024 314 --output_dir experiments/results/e13_fixed"

ORDER=(e14 e05 e01 e02 e06 e11 e13 e08 e10 e12 e12c e13f e15 eF)

# If args given, run only those stages (in the given order); else full ORDER.
if [ "$#" -gt 0 ]; then
  SELECT=("$@")
else
  SELECT=("${ORDER[@]}")
fi

echo "### Camera-ready run START $(date)  stages: ${SELECT[*]}"
for s in "${SELECT[@]}"; do
  if [ -z "${STAGES[$s]:-}" ]; then echo "unknown stage: $s"; continue; fi
  run_stage "$s" ${STAGES[$s]}
done
echo "### Camera-ready run END $(date)"
