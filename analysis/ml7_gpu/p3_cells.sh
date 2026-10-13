#!/usr/bin/env bash
# ML-7 P3: ORT's OpenVINO execution provider on yappy, on the GPU (Intel's GPU plugin on NEO's OpenCL, the
# OpenCL path) and on the CPU, one process per graph, device and precision (plan P3). Run from ~/ml7/ on the host,
# detached, after p2_cells.sh, with the site stopped (ticket D11). It reads P2's layout (p2/inputs.npz,
# p2/refs_emb.npz, p2/onnx/, p2/SHA256SUMS) and venv_ort_openvino; outputs to p3/out/.
#
#   - GPU, fp32 (INFERENCE_PRECISION_HINT f32), each P2 .onnx: OpenVINO's compiled-model cache is emptied first,
#     so the cold start compiles every kernel; the cell then runs once more on the filled cache, as
#     <name>__cached, for its cold start only.
#   - GPU, fp16 (the hint at f16, on the fp32 file): information only (ticket D12, Q4).
#   - CPU, fp32, 4 threads, each .onnx, and the int8 encoder from P1 for information (ticket D4).
#
# OpenCL's only ICD on yappy is NEO's (/etc/OpenCL/vendors/intel.icd), so no CPU OpenCL device is on offer; each
# GPU run's record proves the GPU computed (cellrun.gpu_proof, path opencl: libigdrcl.so). A cell refused with
# CPU fallback off is run once more with --fallback, as <name>__diag, for diagnosis only (plan P2).
#
# With arguments, one cell only: p3_cells.sh ov <device> <precision> <file> <name>.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
P=p2; OUT=p3/out; CACHE=cache/openvino; mkdir -p "$OUT"
THREADS=4
graph_of() { case "$1" in dino*) echo dino ;; encoder*) echo encoder ;; decoder*) echo decoder ;; esac; }
need() {   # need <path under p2>...: each must match its line in SHA256SUMS (P2 brought them)
  local f
  for f in "$@"; do
    (cd $P && grep " $f\$" SHA256SUMS | sha256sum -c --status) || { echo "=== $f missing or changed"; return 1; }
  done
}

run() {   # run <device> <precision> <file> <out name> [extra args]
  local dev=$1 prec=$2 file=$3 name=$4; shift 4
  venv_ort_openvino/bin/python run_ort.py --onnx-dir $P/onnx --inputs $P/inputs.npz --refs $P/refs_emb.npz \
    --graph "$(graph_of "$file")" --file "$file" --provider "openvino_$dev" --precision "$prec" \
    --threads $THREADS --out "$OUT/$name.npz" "$@"
}

ov() {   # ov <gpu|cpu> <fp32|fp16> <file> <name>
  local dev=$1 prec=$2 file=$3 name=$4 cache=()
  need inputs.npz refs_emb.npz "onnx/$file" "onnx/$file.data" || return
  echo "=== openvino $dev $prec $file $(date -Is)"
  if [[ $dev == gpu ]]; then rm -rf "$CACHE"; mkdir -p "$CACHE"; cache=(--cache-dir "$CACHE"); fi
  if ! run "$dev" "$prec" "$file" "$name" "${cache[@]}"; then
    echo "=== openvino $dev $prec $file FAILED with fallback off; diagnosis run with fallback on"
    run "$dev" "$prec" "$file" "${name}__diag" "${cache[@]}" --fallback \
      || echo "=== openvino $dev $prec $file FAILED with fallback on too"
    return
  fi
  if [[ $dev == gpu ]]; then
    echo "=== openvino $dev $prec $file, cached $(date -Is)"
    run "$dev" "$prec" "$file" "${name}__cached" "${cache[@]}" || echo "=== cached run FAILED"
  fi
}

if [[ $# -gt 0 ]]; then
  echo "=== rerun $* $(date -Is)"; sha256sum cellrun.py run_ort.py; "$@"; exit
fi
echo "=== start $(date -Is)"
sha256sum cellrun.py run_ort.py
venv_ort_openvino/bin/python -c "from onnxruntime.capi import _pybind_state as C; \
  print('openvino devices', C.get_available_openvino_device_ids())" 2>&1 || true
for f in decoder encoder dino_b8 dino_b40; do
  ov gpu fp32 "$f.onnx" "ort_ovgpu_fp32_$f"
done
for f in decoder encoder dino_b8 dino_b40; do
  ov cpu fp32 "$f.onnx" "ort_ovcpu_fp32_$f"
done
ov cpu fp32 encoder.int8.onnx ort_ovcpu_int8_encoder
for f in decoder encoder dino_b8 dino_b40; do
  ov gpu fp16 "$f.onnx" "ort_ovgpu_fp16_$f"
done
echo "=== end $(date -Is)"
