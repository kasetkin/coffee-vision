#!/usr/bin/env bash
# ML-7 P7: does DINOv3 in fp16 on LiteRT's GPU change an answer, over the ~780 photos of p7_answer_photos.txt
# (p7_answers.py; plan P7). On yappy, from ~/ml7/, alone on the host, the photos under PHOTOS/<path in the list>.
# Steps, in the order given (default: patches fp32 fp16 compare); each is resumable, so a step cut short is run
# again with the same command:
#   patches  the release's venv and code: serving's preprocessing to OUT/patches/<sha256>.npy (TORCH=1: PyTorch's
#            readouts too, ~22 s a photo more)
#   fp32     the litert_gpu cell's DINOv3 b8 (FP32_MODELS/tflite/dino_b8.tflite) over every photo's patches
#   fp16     the litert_gpu_fp16 cell's (FP16_MODELS: the fp16 forms)
#   split    the litert_gpu_split cell's (SPLIT_MODELS), once backends.py has the cell
#   compare  the release's venv: answers per set, flips, cosines; OUT/p7_answers.json and a summary
# Each step logs to OUT/logs/<step>.log (tail -f it); this prints a line as each starts and ends.
#   ./p7_answers.sh                          # the whole check, ~5.5 h
#   N=3 OUT=p7/answers_rehearsal ./p7_answers.sh   # a rehearsal on the first 3 photos
#   ./p7_answers.sh split compare            # later, the split cell
set -u
cd "$(dirname "$0")"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json PYTHONUNBUFFERED=1
RELEASE=/opt/coffee-cv/current; CKPT=models/allrigs_dino3b16_segxl0v3_country_s42.pt
OUT=${OUT:-p7/answers}; PHOTOS=${PHOTOS:-p7/photos}; N=${N:-}; TORCH=${TORCH:-}
FP32_MODELS=${FP32_MODELS:-p7/whole}; FP16_MODELS=${FP16_MODELS:-p7/fp16}; SPLIT_MODELS=${SPLIT_MODELS:-p7/split}
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(patches fp32 fp16 compare)
limit=(); [ -n "$N" ] && limit=(--limit "$N")
release_py() { ML7_REPO=$RELEASE $RELEASE/.venv/bin/python p7_answers.py "$@"; }
cell() {   # cell <name> <model dir>
  (cd "$2/tflite" && sha256sum dino_b8.tflite)
  venv_litert/bin/python p7_answers.py run --cell "$1" --model-dir "$2" --runtime-dir rt_vulkan --out "$OUT" \
    --threads 4 "${limit[@]}"
}
mkdir -p "$OUT/logs"
echo "=== start $(date -Is) steps=${STEPS[*]} out=$OUT n=${N:-all} torch=${TORCH:-0}"
sha256sum p7_answers.py p7_answer_photos.txt backends.py cellrun.py run_litert.py parity.py make_refs.py graphs.py
for step in "${STEPS[@]}"; do
  log=$OUT/logs/$step.log
  echo "=== $step $(date -Is)"
  case $step in
    patches) release_py patches --checkpoint $CKPT --root "$PHOTOS" --out "$OUT" --threads 4 "${limit[@]}" \
               ${TORCH:+--torch} >> "$log" 2>&1 ;;
    fp32)    cell litert_gpu "$FP32_MODELS" >> "$log" 2>&1 ;;
    fp16)    cell litert_gpu_fp16 "$FP16_MODELS" >> "$log" 2>&1 ;;
    split)   cell litert_gpu_split "$SPLIT_MODELS" >> "$log" 2>&1 ;;
    compare) release_py compare --checkpoint $CKPT --out "$OUT" --base litert_gpu --json "$OUT/p7_answers.json" \
               > "$log" 2>&1 ;;
    *)       echo "unknown step $step"; continue ;;
  esac
  rc=$?
  echo "$step exit=$rc $(grep -E '^\{' "$log" | tail -1) $(grep -c ' ERROR ' "$log") error lines"
  [ "$step" = compare ] && grep -vE 'Warning|warn' "$log" | tail -40
done
echo "=== end $(date -Is)"
