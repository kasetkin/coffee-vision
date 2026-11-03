#!/usr/bin/env bash
# ML-19 P1 effort sweep on one machine: bash efforts.sh LABEL "SCOPE_PREFIX" PHOTO...
set -uo pipefail
cd /tmp/ml19p1
export UV_CACHE_DIR=/tmp/ml19p1/uvcache UV_PYTHON_INSTALL_DIR=/tmp/ml19p1/python PYTHONUNBUFFERED=1
label=$1; scope=$2; shift 2
mkdir -p out/eff_$label
for f in "$@"; do for e in ${EFFORTS:-3 4 5 6 7 8 9}; do
  $scope uv run -q --script pipeline.py canon "$f" --out out/eff_$label/e$e --effort $e --threads 2 >> out/eff_$label.jsonl
  status=$?   # taken before anything else runs: 137 is a kill by the OOM killer
  if [ $status -ne 0 ]; then
    echo "{\"photo\": \"$(basename "$f")\", \"effort\": $e, \"failed\": $status}" >> out/eff_$label.jsonl
  fi
  grep MemAvailable /proc/meminfo >> out/eff_$label.mem
done; done
echo DONE >> out/eff_$label.jsonl
