"""OPS-23 fp16 experiment: one DINOv3 batch-8 graph over ML-7's 13 photos x 40 patches (65 calls), on the CPU.
Prints a line per photo; writes <out>.npz (readouts [13, 40, 1536], for parity_now.py) and <out>.json (every call's
time, the warm median and p90 with the first call left out, load time, non-finite count, peak RSS).
    python dino16.py litert FILE.tflite FLAGS OUT      # FLAGS: 0 = fp32, 4 = XNNPACK FORCE_FP16
    python dino16.py ort FILE.onnx - OUT               # ONNX Runtime CPU provider"""

import json
import resource
import statistics
import sys
import time
import numpy as np

rt, path, flags, out = sys.argv[1:5]
inp = np.load("/b/p2/inputs.npz")
P = inp["patches"]  # [13, 40, 3, 224, 224]
t0 = time.perf_counter()
if rt == "litert":
    from ai_edge_litert.compiled_model import (
        CompiledModel,
        CpuOptions,
        HardwareAccelerator,
        Options,
    )

    kw = dict(num_threads=4)
    if int(flags):
        kw["xnnpack_flags"] = int(flags)
    m = CompiledModel.from_file(
        path,
        options=Options(
            hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(**kw)
        ),
    )
    ib, ob = m.create_input_buffers(0), m.create_output_buffers(0)

    def call(x):
        ib[0].write(np.ascontiguousarray(x, np.float32))
        m.run_by_index(0, ib, ob)
        return ob[0].read(8 * 1536, np.float32).reshape(8, 1536)
else:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name

    def call(x):
        return s.run(None, {name: np.ascontiguousarray(x, np.float32)})[0].astype(
            np.float32
        )


load_s = time.perf_counter() - t0
R, times = np.zeros((13, 40, 1536), np.float32), []
for ph in range(13):
    for b in range(5):
        t = time.perf_counter()
        R[ph, b * 8 : (b + 1) * 8] = call(P[ph, b * 8 : (b + 1) * 8])
        times.append(time.perf_counter() - t)
    print(f"photo {ph + 1}/13 last call {times[-1]:.3f} s", flush=True)
warm = sorted(times[1:])
rec = dict(
    runtime=rt,
    file=path,
    flags=flags,
    load_s=round(load_s, 3),
    call_s=[round(t, 4) for t in times],
    warm_median_s=statistics.median(warm),
    warm_p90_s=warm[int(0.9 * (len(warm) - 1) + 0.5)],
    per_photo_s=5 * statistics.median(warm),
    nonfinite=int((~np.isfinite(R)).sum()),
    peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
)
np.savez(out + ".npz", readouts=R)
json.dump(rec, open(out + ".json", "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k != "call_s"}))
