#!/bin/sh
# OPS-23: ONNX Runtime 1.31.0 on the Pi 5 -- ML-7 P4's per-graph timing for the CPU provider (ort_cpu)
# and the WebGPU plugin (webgpu, onnxruntime-ep-webgpu 0.4.0, Dawn on V3DV through the pishim layer; the container
# mounts a fake /sys/class/drm so ORT's device discovery lists the non-PCI V3D). Outputs to out/ort/.
cd /b
O=out/ort; mkdir -p $O; P=$O/progress.log
echo "start $(date -Is) cells=$CELLS" >> $P
for c in $CELLS; do
  for g in decoder encoder dino_b8 dino_b40; do
    echo "== $c $g start $(date -Is)" >> $P
    python -B code/p4_graph.py --cell $c --graph $g --model-dir p2 --inputs p2/inputs.npz --refs refs_emb.npz \
      --threads 4 --warmup ${WARMUP:-2} --timed ${TIMED:-8} --out $O/${c}__$g.json > $O/${c}__$g.log 2>&1
    echo "== $c $g end rc=$? $(date -Is)" >> $P
    grep -E "^\{" $O/${c}__$g.log | tail -1 >> $P
  done
done
echo "end $(date -Is)" >> $P
