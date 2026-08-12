#!/usr/bin/env bash
# Auto-push camera-ready results: after each pipeline STAGE completes, commit the
# new results and push to origin/camera-ready/experiments. Exits once no
# experiment/master process is running (with a final push). Self-contained;
# survives master relaunches (counts completion markers across all master logs).
cd "$(dirname "$0")/.."
BRANCH=camera-ready/experiments
LOGDIR=experiments/results/_camera_ready_logs
ALOG="$LOGDIR/auto_push.log"
mkdir -p "$LOGDIR"

commit_and_push () {
  local msg="$1"
  git add -A
  if git diff --cached --quiet; then
    return 0                       # nothing new
  fi
  git commit -q -m "$msg" && git push origin "$BRANCH" >>"$ALOG" 2>&1 \
    && echo "$(date +%F_%T) pushed: $msg" >>"$ALOG" \
    || echo "$(date +%F_%T) PUSH FAILED: $msg" >>"$ALOG"
}

echo "$(date +%F_%T) auto-push daemon started" >>"$ALOG"
seen=$(cat $LOGDIR/MASTER_*.log 2>/dev/null | grep -c "<<< STAGE")
while true; do
  sleep 180
  cur=$(cat $LOGDIR/MASTER_*.log 2>/dev/null | grep -c "<<< STAGE")
  if [ "${cur:-0}" -gt "${seen:-0}" ]; then
    commit_and_push "auto: camera-ready results after stage completion ($(date +%F_%T))"
    seen=$cur
  fi
  # If nothing is running any more, do a final push and stop.
  if ! pgrep -f "run_camera_ready.sh" >/dev/null && ! pgrep -f "python -u -m experiments.e" >/dev/null; then
    commit_and_push "auto: final camera-ready results ($(date +%F_%T))"
    echo "$(date +%F_%T) no experiments running; daemon exiting" >>"$ALOG"
    break
  fi
done
