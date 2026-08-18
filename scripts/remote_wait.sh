#!/usr/bin/env bash
# Block until a remote command sweep signals completion (a sentinel string
# appended to a status file on the remote host), reconnecting automatically on
# a dropped SSH connection. Prints the final status and a log tail once done.
# Companion to remote_watch.sh; both assume the sweep was launched detached
# from its SSH session (setsid/nohup) so a dropped connection here never
# touches the training job itself.
#
# Usage: scripts/remote_wait.sh [ssh_host] [remote_status_file] [remote_log] [sentinel]
set -uo pipefail

HOST="${1:-powervpsssh}"
STATUS="${2:-~/sweep_status.log}"
LOG="${3:-~/sweep.log}"
SENTINEL="${4:-SWEEP_DONE_EXIT}"

while true; do
  if ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=6 -o ConnectTimeout=15 "$HOST" "
    while ! grep -q $SENTINEL $STATUS 2>/dev/null; do sleep 30; done
    echo DONE_MARKER
  "; then
    break
  fi
  echo "[remote_wait] ssh dropped, reconnecting in 10s..." >&2
  sleep 10
done

echo "=== FINAL STATUS ==="
ssh "$HOST" "cat $STATUS; echo ---LAST150---; tail -150 $LOG"
