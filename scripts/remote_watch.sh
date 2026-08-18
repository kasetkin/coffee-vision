#!/usr/bin/env bash
# Live-tail a remote log over SSH, filtered to the lines worth a notification,
# reconnecting automatically on a dropped connection. For watching a
# coffeecv.run_folds sweep launched detached on a remote machine (setsid/nohup,
# so a dropped connection here never touches the training job itself) --
# replays the log from the start on first connect, then only new lines on any
# reconnect, so a network blip doesn't re-emit history.
#
# Usage: scripts/remote_watch.sh [ssh_host] [remote_log] [grep_pattern]
set -uo pipefail

HOST="${1:-powervpsssh}"
LOG="${2:-~/sweep.log}"
PATTERN="${3:-^===|^epoch|finished in|Archived to|SWEEP_DONE_EXIT|Traceback|Error|FAILED|Killed|OOM|SystemExit|[Rr]efusing}"

first=1
while true; do
  if [ "$first" = "1" ]; then
    ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=6 -o ConnectTimeout=15 "$HOST" "tail -n +1 -F $LOG"
    first=0
  else
    ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=6 -o ConnectTimeout=15 "$HOST" "tail -n0 -F $LOG"
  fi
  echo "[remote_watch] connection lost, reconnecting in 5s..." >&2
  sleep 5
done | grep -E --line-buffered "$PATTERN"
