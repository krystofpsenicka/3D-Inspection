#!/bin/bash
# Smoke-tests each experiment by running `--help`, which forces a full import
# of the module. Any syntax error, missing import, typo, or broken default
# constant will fail here before you start long runs.
cd "$(dirname "$0")" || exit 1

EXPS=(
  e01_sampling_strategy
  e02_candidate_scaling
  e03_visibility_comparison
  e04_set_cover_optimizers
  e06_vrp_fleet_scaling
  e10_sta_resolution
  e15_cross_model
  e17_sampler_routing_impact
  e19_vrp_time_limit_sweep
)

for exp in "${EXPS[@]}"; do
  printf "%-35s " "$exp"
  if conda run -n isaaclab python -m experiments.$exp --help >/dev/null 2>/tmp/e_err; then
    echo OK
  else
    echo FAIL
    cat /tmp/e_err
    exit 1
  fi
done

echo "=== ALL OK ==="
