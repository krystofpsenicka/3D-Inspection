#!/usr/bin/env bash
# Re-run the whole §9.7/§9.9/§9.11 frontier through the warm arm.
#
# The 18 runs are: 5 operating points on pipeline seed 0 (`frontier`), the same
# sweep on seeds 1 and 7 (`frontier_seeds`), and R=3,4,5 robots at the 0.90
# coverage target (`frontier_robots`). They differ ONLY in --pipeline_dir and
# --out; every solver knob is a default, which is what makes the sweep a
# robustness check rather than 18 separate tunings.
#
#   ./hpro/run_frontier.sh <results-root> [extra joint_pilot.py args...]
#
# e.g. matched-margin sweep (same clearance rule the pipeline enforces):
#   ./hpro/run_frontier.sh hpro/results/frontier_match --clearance_margin 0.05
#
set -euo pipefail
ROOT="${1:-hpro/results/frontier_clear}"
shift || true
EXTRA=("$@")
IPY=/home/troja-lab-02/miniconda3/envs/inspection/bin/python

run () {   # run <group> <pipeline-dir-basename>
  local out="$ROOT/$1/$2"
  mkdir -p "$out"
  echo "=== $1/$2 ==="
  "$IPY" hpro/joint_pilot.py --pipeline_dir "outputs/$2" --modes warm \
      --out "$out" --no_show "${EXTRA[@]}" 2>&1 | tee "$out/run.log" \
    | grep -E "clearance after|baseline:|warm-init|-> cov" || true
}

for tc in pilot_baseline pilot_tc65 pilot_tc78 pilot_tc85 pilot_tc90; do
  run frontier "$tc"
done
for s in s1 s7; do
  for tc in 65 78 85 90 95; do
    run frontier_seeds "pilot_${s}_tc${tc}"
  done
done
for r in r3 r4 r5; do
  run frontier_robots "pilot_${r}_tc90"
done
echo "ALL DONE -> $ROOT"
