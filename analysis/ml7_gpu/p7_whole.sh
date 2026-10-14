#!/usr/bin/env bash
# ML-7 P7: the LiteRT GPU cell with DINOv3 and the decoder converted in their whole-on-the-GPU forms (plan P7,
# graphs.litert_gpu_forms: DINOv3's prefix tokens tiled, SAM's box prompt without writes), on yappy, as P4 ran
# it: the decoder and DINOv3 at batch 8 per graph (3 warm-ups, 20 timed; the delegate's report), then the photo
# replay (2 passes), then its summary and parity against refs (p4_results.py, the release's venv and code).
# MODELS holds tflite/{dino_b8,decoder,encoder}.tflite; run from ~/ml7/, alone on the host. CELL and GRAPHS for
# the fp16 forms (graphs.fp16_forms, P7):
#   MODELS=p7/fp16 OUT=p7/fp16_out CELL=litert_gpu_fp16 GRAPHS="decoder encoder dino_b8" ./p7_whole.sh
# and the precision split (backends.litert_gpu_split: the encoder and decoder in fp32, DINOv3 in fp16):
#   MODELS=p7/split OUT=p7/split_out CELL=litert_gpu_split GRAPHS="decoder encoder dino_b8" ./p7_whole.sh
set -u
cd "$(dirname "$0")"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json PYTHONUNBUFFERED=1
MODELS=${MODELS:-p7/whole}; OUT=${OUT:-p7/whole_out}; R=$OUT/r1; RELEASE=/opt/coffee-cv/current
CELL=${CELL:-litert_gpu}; GRAPHS=${GRAPHS:-decoder dino_b8}
common=(--cell "$CELL" --model-dir "$MODELS" --inputs p2/inputs.npz --threads 4 --runtime-dir rt_vulkan)
mkdir -p "$R"
echo "=== start $(date -Is) models=$MODELS cell=$CELL graphs=$GRAPHS"
(cd "$MODELS/tflite" && sha256sum ./*.tflite)
for g in $GRAPHS; do
  venv_litert/bin/python p4_graph.py "${common[@]}" --graph $g --refs p2/refs_emb.npz --warmup 3 --timed 20 \
    --out "$R/${CELL}__$g.json" > "$R/${CELL}__$g.log" 2>&1
  echo "$g exit=$? $(grep -E '^\{' "$R/${CELL}__$g.log")"
done
venv_litert/bin/python p4_replay.py "${common[@]}" --batch 8 --passes 2 --out "$R/${CELL}__replay" \
  > "$R/${CELL}__replay.log" 2>&1
echo "replay exit=$? $(grep -E '^\{' "$R/${CELL}__replay.log")"
ML7_REPO=$RELEASE $RELEASE/.venv/bin/python p4_results.py --runs "$OUT" --refs refs_noemb.npz \
  --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt --out "$OUT/results.json"
echo "=== end $(date -Is)"
