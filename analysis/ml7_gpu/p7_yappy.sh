#!/usr/bin/env bash
# ML-7 P7: the main track's serving-shaped numbers on yappy, with P2's files (plan P7, the yappy task). Run from
# ~/ml7/, alone on the host, the site down. Three parts, each a fresh process of P4's replay for the LiteRT GPU
# cell at batch 8 (p4_replay.py), or of the release's venv:
#   torch  torch's own footprint in the serving worker (ticket D5 (a)): the release's imports without the models,
#          their time, peak RSS and the MemAvailable drop. Added to the replay's memory for R5.
#   cold   the cold start with Mesa's shader cache empty, then warm, COLD_TRIALS times each (2 photos, one pass):
#          the restart cost after a deploy. The cache lives in OUT, not in the user's ~/.cache.
#   soak   the replay for SOAK_PASSES passes over the 13 photos; MemAvailable and the process's RSS every 30 s
#          (OUT/soak_mem.tsv), so memory growth, a GPU fault (an exit or NaN) and the times' drift show.
# Outputs to OUT (p7/out): <part>_*.json/.npz/.log, as p4_replay.py writes them.
#   COLD_TRIALS=1 SOAK_PASSES=1 OUT=p7/rehearsal ./p7_yappy.sh
set -u
cd "$(dirname "$0")"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json PYTHONUNBUFFERED=1
P=p2; RELEASE=/opt/coffee-cv/current; PY=venv_litert/bin/python
COLD_TRIALS=${COLD_TRIALS:-3}; SOAK_PASSES=${SOAK_PASSES:-40}; OUT=${OUT:-p7/out}
REPLAY=(p4_replay.py --cell litert_gpu --batch 8 --model-dir $P --inputs $P/inputs.npz --threads 4
        --runtime-dir rt_vulkan)
mkdir -p "$OUT"
echo "=== start $(date -Is) cold_trials=$COLD_TRIALS soak_passes=$SOAK_PASSES"
sha256sum p4_replay.py backends.py cellrun.py run_litert.py p7_yappy.sh
(cd $P && sha256sum -c --quiet SHA256SUMS) && echo "=== inputs verified" || echo "=== inputs NOT verified"

echo "=== torch $(date -Is)"
(cd "$RELEASE" && .venv/bin/python - <<'PY') > "$OUT/torch.json"
import json, resource, time
def avail():
    return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 1024
a0, t0 = avail(), time.perf_counter()
import torch, torchvision, timm                                    # noqa: E401,F401  the worker's imports, no models
import coffeecv.infer                                              # noqa: F401
print(json.dumps({"imports": ["torch", "torchvision", "timm", "coffeecv.infer"], "torch": torch.__version__,
                  "import_s": round(time.perf_counter() - t0, 2),
                  "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                  "mem_available_drop_mb": round(a0 - avail())}))
PY
cat "$OUT/torch.json"

cache=$OUT/mesa_cache
for ((t = 1; t <= COLD_TRIALS; t++)); do
  rm -rf "$cache"
  for state in empty warm; do
    echo "=== cold $state $t $(date -Is)"
    MESA_SHADER_CACHE_DIR=$cache $PY "${REPLAY[@]}" --photos 2 --passes 1 --out "$OUT/cold_${state}_$t" \
      > "$OUT/cold_${state}_$t.log" 2>&1
    echo "exit=$? $(grep -E '^\{' "$OUT/cold_${state}_$t.log") cache $(du -sk "$cache" | cut -f1) KiB"
  done
done

echo "=== soak $(date -Is)"
$PY "${REPLAY[@]}" --passes "$SOAK_PASSES" --out "$OUT/soak" > "$OUT/soak.log" 2>&1 &
pid=$!
echo -e "epoch_s\tmem_available_kb\trss_kb" > "$OUT/soak_mem.tsv"
while kill -0 $pid 2>/dev/null; do
  echo -e "$(date +%s)\t$(awk '/MemAvailable/ {print $2}' /proc/meminfo)\t$(awk '/VmRSS/ {print $2}' /proc/$pid/status 2>/dev/null)"
  sleep 30
done >> "$OUT/soak_mem.tsv"
wait $pid
echo "soak exit=$? $(grep -E '^\{' "$OUT/soak.log")"
echo "=== end $(date -Is)"
