#!/usr/bin/env bash
# ML-7 P4: the comparison run on yappy (plan P4). Run from ~/ml7/, detached, with the site stopped and nothing
# else running (ticket D11). It reads P2's layout (p2/inputs.npz, p2/refs_emb.npz, p2/onnx/, p2/tflite/,
# p2/SHA256SUMS), the venvs, rt_vulkan/, and the release (/opt/coffee-cv/current, read-only) for the torch cell.
# Outputs to p4/out/r<round>/: <cell>__<graph>.json, <cell>__<graph>__stability.json, <cell>__replay.{json,npz},
# and logs/, every process's whole output (a crash leaves its exit code in this log and its output there).
#
# Each round runs every cell in turn, the order rotated between rounds so that clock speed and heat fall on all
# of them alike; every graph and every replay is a fresh process (p4_graph.py, p4_replay.py). Every round, per
# cell: the photo replay (PASSES passes) at the cell's faster DINOv3 batch, which decides D15. Round 1 only (Q6,
# answered by the owner 2026-10-14), before each replay: the decoder, the encoder, DINOv3 at batch 8 and 40
# (WARMUP warm-ups, TIMED timed calls), which choose that batch, and the stability run, STAB calls at batch 8 for
# every cell; the CPU cell's DINOv3 ORT-fused too; and, after the cells, the information-only variants (fp16
# GPU copies, the int8 encoder).
# OpenVINO's cache is emptied before each of its processes, so each cold start compiles.
#
# The cells come from backends.P4_CELLS and P4_VARIANTS. The replay's DINOv3 batch is the faster one in round 1,
# kept for the later rounds (OUT/batch_<cell>). The protocol's numbers can be overridden from the environment,
# for a rehearsal:
#   ROUNDS=1 WARMUP=1 TIMED=2 STAB=3 PASSES=1 PHOTOS=2 OUT=p4/rehearsal ./p4_run.sh
set -u
cd "$(dirname "$0")"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json PYTHONUNBUFFERED=1
P=p2; THREADS=4; RELEASE=/opt/coffee-cv/current
ROUNDS=${ROUNDS:-3}; WARMUP=${WARMUP:-3}; TIMED=${TIMED:-20}; STAB=${STAB:-100}; PASSES=${PASSES:-2}
PHOTOS=${PHOTOS:-}; OUT=${OUT:-p4/out}
from_backends() { venv_litert/bin/python -c "import backends; print(*$1)"; }   # the lists live in backends.py
read -r -a CELLS <<< "$(from_backends backends.P4_CELLS)"
read -r -a VARIANTS <<< "$(from_backends backends.P4_VARIANTS)"

python_of() {
  case $1 in
    torch) echo "$RELEASE/.venv/bin/python" ;;
    litert*) echo venv_litert/bin/python ;;
    ov*) echo venv_ort_openvino/bin/python ;;
    *) echo venv_ort_webgpu/bin/python ;;
  esac
}
checkpoint=$(sed -n 's/^COFFEE_CV_CHECKPOINT=//p' "$RELEASE/release.env")
common() {   # common <cell>: the arguments every process of the cell takes
  local a=(--cell "$1" --model-dir $P --inputs $P/inputs.npz --threads $THREADS --runtime-dir rt_vulkan)
  if [[ $1 == ov* ]]; then rm -rf cache/openvino; mkdir -p cache/openvino; a+=(--cache-dir cache/openvino); fi
  printf '%s\n' "${a[@]}"
}
run() {   # run <cell> <record> <script> <args...>: one process, logged; the torch cell runs the release's code.
  # A failure leaves <cell>__<record>.failed beside the records, its exit code and the log's tail (p4_results.py).
  local cell=$1 label=$2; shift 2
  local env=() args; mapfile -t args < <(common "$cell")
  [[ $cell == torch ]] && env=(ML7_REPO="$RELEASE" COFFEE_CV_CHECKPOINT="$checkpoint")
  local log="$OUT/r$R/logs/${cell}__${label// /_}.log" rc
  mkdir -p "$OUT/r$R/logs"
  echo "=== r$R $cell $label $(date -Is)"
  # The whole output kept, per process; its progress lines (a line per call, per loaded replay, per photo)
  # passed on to this log as they come.
  env "${env[@]}" "$(python_of "$cell")" "$@" "${args[@]}" 2>&1 | tee "$log" \
    | grep --line-buffered -E '^(call|loaded|photo) '
  rc=${PIPESTATUS[0]}
  grep -E '^\{|Traceback|Error:|error:' "$log" | grep -v kLiteRtStatusErrorNotFound | tail -n 5
  if (( rc != 0 )); then
    echo "=== r$R $cell $label FAILED exit=$rc$( ((rc > 128)) && echo " (signal $((rc - 128)))") $log"
    { echo "exit=$rc log=$log"; grep -v kLiteRtStatusErrorNotFound "$log" | tail -n 20; } \
      > "$OUT/r$R/${cell}__${label}.failed"
  fi
}
graph() {   # graph <cell> <graph> <timed> <suffix>
  run "$1" "$2$4" p4_graph.py --graph "$2" --refs $P/refs_emb.npz --warmup "$WARMUP" --timed "$3" \
    --out "$OUT/r$R/$1__$2$4.json"
}
faster() {   # faster <cell>: 8 or 40, the batch with the lower time per patch in this round
  python3 - "$OUT/r$R/$1__dino_b8.json" "$OUT/r$R/$1__dino_b40.json" <<'PY'
import json, sys
per = {}
for b, f in zip((8, 40), sys.argv[1:]):
    try:
        per[b] = json.load(open(f))["timing"]["warm"]["median"] / b
    except (OSError, KeyError, TypeError, ValueError):
        pass
print(min(per, key=per.get) if per else 8)
PY
}
cell() {   # cell <cell>: in round 1 the four graphs and the stability run (Q6); every round, the replay
  local c=$1 g b
  if (( R == 1 )); then
    for g in decoder encoder dino_b8 dino_b40; do graph "$c" "$g" "$TIMED" ""; done
    graph "$c" dino_b8 "$STAB" "__stability"
    if [[ $c == ort_cpu ]]; then
      for g in dino_b8 dino_b40; do graph ort_cpu_fused "$g" "$TIMED" ""; done
    fi
  fi
  # The replay batch is chosen once, in round 1, so every round replays at it (D15's spread between rounds).
  [[ -s "$OUT/batch_$c" ]] || faster "$c" > "$OUT/batch_$c"
  b=$(< "$OUT/batch_$c")
  run "$c" "replay" p4_replay.py --batch "$b" --passes "$PASSES" ${PHOTOS:+--photos "$PHOTOS"} \
    --out "$OUT/r$R/${c}__replay"
}

mkdir -p "$OUT"
echo "=== start $(date -Is) rounds=$ROUNDS warmup=$WARMUP timed=$TIMED stab=$STAB passes=$PASSES photos=${PHOTOS:-all}"
sha256sum backends.py cellrun.py p4_graph.py p4_replay.py run_litert.py run_ort.py p4_run.sh
(cd $P && sha256sum -c --quiet SHA256SUMS) && echo "=== inputs verified" || echo "=== inputs NOT verified"
uname -r; cat /sys/class/drm/card0/gt/gt0/rps_max_freq_mhz 2>/dev/null
for ((R = 1; R <= ROUNDS; R++)); do
  mkdir -p "$OUT/r$R"
  n=${#CELLS[@]}; shift_by=$(( (R - 1) * (n / ROUNDS > 0 ? n / ROUNDS : 1) ))
  for ((i = 0; i < n; i++)); do cell "${CELLS[$(( (i + shift_by) % n ))]}"; done
  if (( R == 1 )); then
    for v in "${VARIANTS[@]}"; do
      for g in $(venv_litert/bin/python -c "import backends; print(*backends.CELLS['$v'].graphs)"); do
        graph "$v" "$g" "$TIMED" ""
      done
    done
  fi
done
echo "=== end $(date -Is)"
