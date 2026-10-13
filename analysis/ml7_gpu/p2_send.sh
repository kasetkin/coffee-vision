#!/usr/bin/env bash
# ML-7 P2: P1's files to yappy's ~/ml7/p2/ over its slow link (about 216 KiB/s), in the order p2_cells.sh needs
# them, while it runs. Run here, from the repo root, detached:  setsid nohup analysis/ml7_gpu/p2_send.sh &
#
#   - Each file is compressed here with zstd, level 19 on 8 threads (rsync's own zstd is single-threaded and
#     cannot reach level 19's ratio on the inputs: 0.21 against zlib's 0.29; model weights stay near 0.93 either
#     way), sent by rsync (resumable: --partial), decompressed on the host to a temporary name and renamed, so
#     p2_cells.sh never reads a half-written file. It checks every file against scratch/p2/SHA256SUMS anyway.
#   - DINOv3's batch-40 files share their weights with batch 8: the .onnx.data twins are byte-identical and are
#     copied on the host; dino_b40.tflite (99.9% the same blocks) and the blanket fp16 copy are seeded with their
#     twin there and sent as rsync deltas, compressed with zstd.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/analysis/ml7_gpu"
HOST=webappremote; R=ml7/p2
A=artifacts; S=scratch/p2; Z=scratch/p2_zst
ZSTD=(env UV_PROJECT_ENVIRONMENT= uv run --no-project --quiet --python 3.12 --with zstandard python -c '
import sys, zstandard as z
p = z.ZstdCompressionParameters.from_level(19, threads=8)
with open(sys.argv[1], "rb") as i, open(sys.argv[2] + ".part", "wb") as o:
    z.ZstdCompressor(compression_params=p).copy_stream(i, o)
import os; os.replace(sys.argv[2] + ".part", sys.argv[2])')

src() { [[ -e "$S/$1" ]] && echo "$S/$1" || echo "$A/$1"; }

send() {   # send <path under p2>: compress, rsync, decompress there
  local f=$1 z="$Z/$1.zst"
  mkdir -p "$(dirname "$z")"
  [[ -e "$z" ]] || "${ZSTD[@]}" "$(src "$f")" "$z" || return 1
  ssh "$HOST" "mkdir -p $R/zst/$(dirname "$f") $R/$(dirname "$f")"
  until rsync -t --partial "$z" "$HOST:$R/zst/$f.zst"; do echo "rsync $f failed; retrying in 60 s"; sleep 60; done
  ssh "$HOST" "cd $R && zstd -d -q -f zst/$f.zst -o $f.part && mv $f.part $f && rm zst/$f.zst"
  echo "sent $f $(date -Is)"
}

seed() {   # seed <twin> <path>: copy the twin on the host, then send <path> as an rsync delta against it
  ssh "$HOST" "cd $R && cp $1 $2.seed && mv $2.seed $2"
  until rsync -t --partial --compress --compress-choice=zstd "$A/$2" "$HOST:$R/$2"; do sleep 60; done
  echo "sent $2 (delta on $1) $(date -Is)"
}

if [[ $# -gt 0 ]]; then   # these files only, after the run: p2_send.sh onnx/encoder.int8.onnx onnx/encoder.int8.onnx.data
  until rsync -t "$S/SHA256SUMS" "$HOST:$R/SHA256SUMS"; do sleep 60; done
  for f; do send "$f"; done; exit
fi
echo "start $(date -Is)"
until rsync -t "$S/SHA256SUMS" "$HOST:$R/SHA256SUMS"; do sleep 60; done
send refs_emb.npz
send inputs.npz
for g in decoder encoder dino_b8; do
  send "tflite/$g.tflite"; send "onnx/$g.onnx"; send "onnx/$g.onnx.data"
done
seed onnx/dino_b8.onnx.data onnx/dino_b40.onnx.data
send onnx/dino_b40.onnx
seed tflite/dino_b8.tflite tflite/dino_b40.tflite
for g in decoder encoder dino_b8; do
  send "onnx/$g.fp16.onnx"; send "onnx/$g.fp16.onnx.data"
done
seed onnx/dino_b8.fp16.onnx.data onnx/dino_b40.fp16.onnx.data
send onnx/dino_b40.fp16.onnx
seed onnx/dino_b8.fp16.onnx.data onnx/dino_b8.fp16_blanket.onnx.data
send onnx/dino_b8.fp16_blanket.onnx
ssh "$HOST" "cd $R && sha256sum -c --quiet SHA256SUMS && echo 'all files verified'"
echo "end $(date -Is)"
