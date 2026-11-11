import glob
import json
import statistics
import time
import numpy as np
from ai_edge_litert.compiled_model import (
    CompiledModel,
    CpuOptions,
    HardwareAccelerator,
    Options,
)

x = np.load("/b/p16/micro/micro_x.npy")
res = {}


def load(path, flags):
    kw = dict(num_threads=4)
    if flags:
        kw["xnnpack_flags"] = flags
    m = CompiledModel.from_file(
        path,
        options=Options(
            hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(**kw)
        ),
    )
    key = list(m.get_signature_list())[0]
    sig = m.get_signature_by_index(0)
    det = m.get_output_tensor_details(key)
    shp = [int(d) for d in det[sig["outputs"][0]]["shape"]]
    ib, ob = m.create_input_buffers(0), m.create_output_buffers(0)

    def call():
        ib[0].write(x)
        m.run_by_index(0, ib, ob)
        return ob[0].read(int(np.prod(shp)), np.float32).astype(np.float64)

    return call


for p in sorted(glob.glob("/b/p16/micro/*.tflite")):
    name = p.split("/")[-1][:-7]
    outs = {}
    for flags in (0, 4):
        call = load(p, flags)
        for _ in range(3):
            y = call()
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            y = call()
            ts.append(time.perf_counter() - t)
        outs[flags] = (statistics.median(ts), y)
    a, b = outs[0][1], outs[4][1]
    fin = np.isfinite(b).all()
    cos = float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)) if fin else None
    res[name] = dict(
        fp32_ms=outs[0][0] * 1e3,
        fp16_ms=outs[4][0] * 1e3,
        speedup=outs[0][0] / outs[4][0],
        cos=cos,
        nonfinite=int((~np.isfinite(b)).sum()),
    )
    print(
        f"{name:14s} fp32 {outs[0][0] * 1e3:8.1f} ms  fp16 {outs[4][0] * 1e3:8.1f} ms  x{outs[0][0] / outs[4][0]:.2f}  cos {cos}",
        flush=True,
    )
json.dump(res, open("/b/out/p16/micro.json", "w"), indent=1)
