#!/bin/sh
# OPS-23 fp16 experiment on the Pi: dino16.py over each DINOv3 variant, fp32 and FORCE_FP16. Outputs in /b/out/p16/.
cd /b; O=out/p16; mkdir -p $O; P=$O/progress.log
cool() { while [ $(cat /sys/class/thermal/thermal_zone0/temp) -ge 60000 ]; do sleep 10; done; }
run() {   # run <tag> <runtime> <file> <flags>
  cool; echo "== $1 start $(date -Is) temp=$(cat /sys/class/thermal/thermal_zone0/temp)" >> $P
  python -B code/dino16.py $2 $3 $4 $O/$1 > $O/$1.log 2>&1; echo "== $1 end rc=$? $(date -Is)" >> $P
  tail -1 $O/$1.log >> $P
}
for v in ${VARIANTS:-plain:p2/tflite/dino_b8.tflite release:p_gpu/tflite/dino_b8.tflite prescaled:p16/tflite/dino_b8_prescaled.tflite split128:p16/tflite/dino_b8_split128.tflite}; do
  tag=${v%%:*}; f=${v#*:}
  for fl in ${FLAGS:-0 4}; do run ${tag}_f$fl litert $f $fl; done
done
for v in $ORT; do tag=${v%%:*}; f=${v#*:}; run ort_$tag ort $f -; done
echo "end $(date -Is)" >> $P
