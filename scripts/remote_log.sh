#!/usr/bin/env bash
# Fetch a remote log file and render it the way a terminal would: a \r means
# "return to line start, overwrite" rather than a real newline, so a
# tqdm-style progress bar collapses to its final state per line instead of
# one line per tick (a raw `cat` of a training log is otherwise one
# multi-hundred-KB line per epoch, unreadable and unwieldy to hand off).
# Companion to remote_launch.sh/remote_wait.sh/remote_watch.sh.
#
# Usage: scripts/remote_log.sh [ssh_host] [remote_log] [local_out]
#   local_out: if given, also write the rendered log there (still prints to stdout)
set -uo pipefail

HOST="${1:-powervpsssh}"
LOG="${2:-~/sweep.log}"
OUT="${3:-}"

render() {
  python3 -c '
import sys
text = sys.stdin.read()
out = []
for chunk in text.split("\n"):
    # Keep only the last \r-delimited segment of each real line -- that is
    # what would actually be on screen once the line is done being redrawn.
    out.append(chunk.split("\r")[-1])
print("\n".join(out))
'
}

if [ -n "$OUT" ]; then
  ssh "$HOST" "cat $LOG" | render | tee "$OUT"
else
  ssh "$HOST" "cat $LOG" | render
fi
