#!/usr/bin/env bash
# ML-7 P2: the Vulkan paths on yappy, every file of P1 through LiteRT's and ORT's Vulkan path, one process per
# graph, path and precision (plan P2). Run from ~/ml7/ on the host, detached (scripts/remote_launch.sh), with the
# site stopped (ticket D11). Layout: p2/inputs.npz, p2/refs_emb.npz (refs.npz's PyTorch embeddings, all the decoder
# cells read), p2/onnx/, p2/tflite/, p2/SHA256SUMS, venv_litert, venv_ort_webgpu, rt_vulkan/ (the wheel's
# libLiteRt.so and libLiteRtWebGpuAccelerator.so only); outputs to p2/out/.
#
# yappy's link is slow (about 216 KiB/s), so the files arrive while this runs: each cell first waits until its
# files match their sha256 in p2/SHA256SUMS (the P1 files, hashed before the copy), then runs.
#
# Every run sees only Intel's Vulkan driver (VK_ICD_FILENAMES), so llvmpipe is not on offer; each run's record
# proves the GPU computed (cellrun.gpu_proof). An ORT cell that refuses a graph with CPU fallback off is run once
# more with --fallback, as <name>__diag, for diagnosis only: never a cell's result (plan P2).
set -u
cd "$(dirname "$0")"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export PYTHONUNBUFFERED=1
P=p2; OUT=$P/out; mkdir -p "$OUT"
THREADS=4
graph_of() { case "$1" in dino*) echo dino ;; encoder*) echo encoder ;; decoder*) echo decoder ;; esac; }
need() {   # need <path under p2>...: wait until each matches its line in SHA256SUMS
  local f
  for f in "$@"; do
    until (cd $P && grep -q " $f\$" SHA256SUMS && grep " $f\$" SHA256SUMS | sha256sum -c --status 2>/dev/null); do
      sleep 60
    done
  done
}

ort() {   # ort <file> <name>
  local g; g=$(graph_of "$1")
  need inputs.npz refs_emb.npz "onnx/$1" "onnx/$1.data"
  echo "=== ort webgpu $1 $(date -Is)"
  if ! venv_ort_webgpu/bin/python run_ort.py --onnx-dir $P/onnx --inputs $P/inputs.npz --refs $P/refs_emb.npz \
      --graph "$g" --file "$1" --provider webgpu --threads $THREADS --out "$OUT/$2.npz"; then
    echo "=== ort webgpu $1 FAILED with fallback off; diagnosis run with fallback on"
    venv_ort_webgpu/bin/python run_ort.py --onnx-dir $P/onnx --inputs $P/inputs.npz --refs $P/refs_emb.npz \
      --graph "$g" --file "$1" --provider webgpu --threads $THREADS --fallback --out "$OUT/$2__diag.npz" \
      || echo "=== ort webgpu $1 FAILED with fallback on too"
  fi
}

litert() {   # litert <file> <precision> <name>
  need inputs.npz refs_emb.npz "tflite/$1"
  echo "=== litert vulkan $1 $2 $(date -Is)"
  venv_litert/bin/python run_litert.py --model-dir $P/tflite --inputs $P/inputs.npz --refs $P/refs_emb.npz \
    --graph "$(graph_of "$1")" --file "$1" --accelerator gpu --runtime-dir rt_vulkan --precision "$2" \
    --threads $THREADS --out "$OUT/$3.npz" || echo "=== litert vulkan $1 $2 FAILED"
}

echo "=== start $(date -Is) $(git -C . rev-parse --short HEAD 2>/dev/null || true)"
sha256sum cellrun.py run_ort.py run_litert.py
for f in decoder encoder dino_b8 dino_b40; do      # the order the files arrive in
  litert "$f.tflite" fp32 "litert_vulkan_fp32_$f"
  ort "$f.onnx" "ort_webgpu_fp32_$f"
done
for f in decoder encoder dino_b8 dino_b40; do
  litert "$f.tflite" fp16 "litert_vulkan_fp16_$f"
  ort "$f.fp16.onnx" "ort_webgpu_fp16_$f"
done
ort dino_b8.fp16_blanket.onnx ort_webgpu_fp16blanket_dino_b8
echo "=== end $(date -Is)"
