#!/usr/bin/env bash
# The GPU option test on yappy (research_quantization.md §5.4): every cell of gpu_quantized_ops.py, each in its own
# process, ROUNDS times with the order rotated, with the live service stopped so nothing else uses the GPU.
# The service is started again when the script ends, however it ends.
#
#   gpu_quantized_ops.sh [ROUNDS] [PHOTOS] [OUT]      run from the directory that holds gpu_quantized_ops.py
set -uo pipefail
ROUNDS="${1:-3}"; PHOTOS="${2:-13}"; OUT="${3:-out}"
RELEASE=/opt/coffee-cv/current
INPUTS="$HOME/ml7/p2/inputs.npz"
PY="$RELEASE/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
set -a; . "$RELEASE/release.env"; set +a          # VK_ICD_FILENAMES, as the service has it

restart() {
  sudo systemctl start coffee-cv-web && echo "SERVICE started $(date -u +%FT%TZ)" \
    || echo "WARNING: coffee-cv-web did not start: sudo systemctl start coffee-cv-web"
}
trap restart EXIT
echo "RELEASE $(readlink -f $RELEASE)"
sudo systemctl stop coffee-cv-web && echo "SERVICE stopped $(date -u +%FT%TZ)"
sleep 5
free -m | sed -n 2p

DINO=(base share quant quant_only)
ENC=(base quant)
for r in $(seq 1 "$ROUNDS"); do
  for i in "${!DINO[@]}"; do
    c="${DINO[$(( (i + r - 1) % ${#DINO[@]} ))]}"
    "$PY" gpu_quantized_ops.py --release "$RELEASE" --inputs "$INPUTS" --graph dino --cell "$c" \
      --photos "$PHOTOS" --out "$OUT/r${r}_dino_$c" || echo "FAILED r$r dino $c (exit $?)"
    sleep 3
  done
  for i in "${!ENC[@]}"; do
    c="${ENC[$(( (i + r - 1) % ${#ENC[@]} ))]}"
    "$PY" gpu_quantized_ops.py --release "$RELEASE" --inputs "$INPUTS" --graph encoder --cell "$c" \
      --photos "$PHOTOS" --out "$OUT/r${r}_encoder_$c" || echo "FAILED r$r encoder $c (exit $?)"
    sleep 3
  done
done
echo "ALL CELLS DONE"
