#!/usr/bin/env bash
# ML-14 P5 (D11): the six dry runs of the ticket's Activity table, one at a time, on 2026-10-30. Nothing under
# dataset/ is written (--dry-run). Before running, as was done then:
#   git worktree add --detach /tmp/ml14-before 413f7b4            # the code before ML-14
#   mkdir /tmp/ml14_fullstrip && cp -r dataset/segmenter_positives /tmp/ml14_fullstrip/
#   exiftool -q -q -overwrite_original -GPSLatitude=48.1 -GPSLatitudeRef=N -GPSLongitude=11.5 -GPSLongitudeRef=E \
#       -Artist=Someone /tmp/ml14_fullstrip/segmenter_positives  # the test suite's private tags, for the full strip
# Usage: analysis/ml14/run_all.sh > analysis/ml14/results.txt; afterwards delete the copy and the worktree.
set -u
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
M=$ROOT/analysis/ml14/measure.py
CLEAN=$ROOT/dataset/segmenter_positives
FULL=/tmp/ml14_fullstrip/segmenter_positives
cd /tmp/ml14-before
python "$M" "before 413f7b4, clean path" -- python -m coffeecv.strip_metadata "$CLEAN" --dry-run
python "$M" "before 413f7b4, full strip" -- python -m coffeecv.strip_metadata "$FULL" --dry-run
cd "$ROOT"
python "$M" "after, --jobs 1, clean path" -- python -m coffeecv.strip_metadata "$CLEAN" --dry-run --jobs 1
python "$M" "after, --jobs 1, full strip" -- python -m coffeecv.strip_metadata "$FULL" --dry-run --jobs 1
python "$M" "after, default --jobs, clean path" -- python -m coffeecv.strip_metadata "$CLEAN" --dry-run
python "$M" "after, default --jobs, full strip" -- python -m coffeecv.strip_metadata "$FULL" --dry-run
echo ALL DONE
