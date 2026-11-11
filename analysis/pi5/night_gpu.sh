#!/bin/sh
# OPS-23. Runs in the coffee-bench:gpu-mesa262 container with /b mounted: /b/code holds analysis/ml7_gpu's
# backends.py, cellrun.py, run_litert.py, p4_graph.py, p4_replay.py and graphs.py; /b/p_gpu/tflite the release's
# encoder, decoder and dino_b8 (models/allrigs_dino3b16_segxl0v3_country_s42.litert); /b/p2/inputs.npz and
# /b/refs_emb.npz (refs.npz's embeddings); /b/shim the built pishim layer. The docker run is in the ticket.
# Pi 5 GPU night run: LiteRT fp32 on V3D through Dawn/Vulkan (with the pishim layer). Each graph compiles and is
# timed in its own process (p4_graph.py), then, if all three ran, the photo replay at DINOv3 batch 8. Mesa's shader
# cache persists in cache/mesa; V3D_DEBUG=shaderdb prints one SHADER-DB line per compiled shader.
cd /b
S=/usr/local/lib/python3.12/site-packages/ai_edge_litert
mkdir -p /rt && ln -sf $S/libLiteRt.so $S/libLiteRtWebGpuAccelerator.so /rt/
O=out/night; mkdir -p $O; P=$O/progress.log
echo "start $(date -Is)" >> $P
echo none > $O/current

watch() {   # one line a minute: the step, shaders compiled in it, seconds since the last one, GPU compute jobs
  while :; do
    cur=$(cat $O/current); f=$O/gpu__$cur.json.stderr; [ -f "$f" ] || f=$O/gpu__$cur.log
    [ "$cur" = replay ] && { f=$O/gpu__replay.stderr; [ -f "$f" ] || f=$O/gpu__replay.log; }
    n=$(grep -c "SHADER-DB" "$f" 2>/dev/null); n=${n:-0}
    last=$( [ -f "$f" ] && echo $(( $(date +%s) - $(stat -c %Y "$f") )) || echo -)
    csd=$(awk '/^csd/{print $3}' /sys/class/drm/card0/device/gpu_stats 2>/dev/null)
    tmp=$(cat /sys/class/thermal/thermal_zone0/temp)
    echo "$(date +%T) step=$cur shaders=$n idle_s=$last csd_jobs=$csd temp=$((tmp/1000))C $(grep -E '^(call|photo|loaded) ' $O/gpu__$cur.log 2>/dev/null | tail -1)" >> $P
    sleep 60
  done
}
watch & W=$!

ok=1
for g in decoder encoder dino_b8; do
  echo $g > $O/current
  echo "== $g start $(date -Is)" >> $P
  python -B code/p4_graph.py --cell litert_gpu --graph $g --model-dir p_gpu --inputs p2/inputs.npz \
    --refs refs_emb.npz --threads 4 --warmup 1 --timed 3 --runtime-dir /rt --out $O/gpu__$g.json > $O/gpu__$g.log 2>&1
  rc=$?
  echo "== $g end rc=$rc shaders=$(grep -c SHADER-DB $O/gpu__$g.log) $(date -Is)" >> $P
  grep -E "operations will run|^\{" $O/gpu__$g.log | tail -2 >> $P
  [ $rc -eq 0 ] || ok=0
done
if [ $ok = 1 ]; then
  echo replay > $O/current
  echo "== replay start $(date -Is)" >> $P
  python -B code/p4_replay.py --cell litert_gpu --batch 8 --model-dir p_gpu --inputs p2/inputs.npz --threads 4 \
    --passes 2 --runtime-dir /rt --out $O/gpu__replay > $O/gpu__replay.log 2>&1
  echo "== replay end rc=$? $(date -Is)" >> $P
  grep -E "^\{" $O/gpu__replay.log | tail -1 >> $P
fi
kill $W
echo "end $(date -Is)" >> $P
