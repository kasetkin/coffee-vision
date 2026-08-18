#!/usr/bin/env bash
# Launch a command on a remote host, detached from the SSH session
# (setsid + nohup + disown), so it survives a dropped connection and keeps
# running unattended for a multi-hour sweep. Pairs with remote_watch.sh
# (tails the log) and remote_wait.sh (blocks on the status file this writes).
#
# The payload is base64-encoded before being sent: `ssh host arg1 arg2 ...`
# does NOT preserve local argument-quoting boundaries -- it flattens multiple
# arguments into one string that the remote shell re-parses, so a payload
# containing spaces or `&&` gets split apart on the wire and part of it runs
# as a separate top-level command on the SSH session itself instead of inside
# the detached job. Base64 has no shell-special characters, so it survives
# that flattening intact regardless.
#
# The payload runs verbatim in a nested `bash -c`, so chain steps yourself
# with && -- a failing step stops the chain (mirrors run_folds.py exiting
# non-zero on failure) and reports as SWEEP_DONE_EXIT=<its exit code>.
#
# Usage: scripts/remote_launch.sh <payload> [ssh_host] [remote_workdir] [remote_venv] [remote_log] [remote_status]
# Example:
#   scripts/remote_launch.sh \
#     'python -m coffeecv.run_folds --arm beans --epochs 80 --seed 42 --mixup-alpha 0.2 --tag mixup02 --start-exp 106'
set -uo pipefail

PAYLOAD="${1:?usage: remote_launch.sh <payload> [ssh_host] [remote_workdir] [remote_venv] [remote_log] [remote_status]}"
HOST="${2:-powervpsssh}"
WORKDIR="${3:-~/coffee-vision}"
VENV="${4:-~/coffee-vision-venv}"
LOG="${5:-~/sweep.log}"
STATUS="${6:-~/sweep_status.log}"

PAYLOAD_B64="$(printf '%s' "$PAYLOAD" | base64 -w0)"

ssh "$HOST" bash -s -- "$WORKDIR" "$VENV" "$LOG" "$STATUS" "$PAYLOAD_B64" <<'REMOTE'
WORKDIR="$1"; VENV="$2"; LOG="$3"; STATUS="$4"
PAYLOAD="$(printf '%s' "$5" | base64 -d)"
rm -f "$LOG" "$STATUS"
setsid bash -c "
  source $VENV/bin/activate
  cd $WORKDIR
  echo \"=== launch \$(date) ===\"
  $PAYLOAD
  ec=\$?
  echo \"=== exit=\$ec \$(date) ===\"
  echo \"SWEEP_DONE_EXIT=\$ec\" >> $STATUS
" > "$LOG" 2>&1 < /dev/null &
disown
sleep 2
echo LAUNCHED
REMOTE
