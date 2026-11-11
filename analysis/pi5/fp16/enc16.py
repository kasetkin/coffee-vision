"""OPS-23 fp16 experiment on the Pi: an encoder .tflite at fp32 and FORCE_FP16 on 3 photos -- embedding cosine
against PyTorch (refs_emb.npz), non-finite count, time per call; and small probe graphs fp32 vs fp16."""

import sys
import time
import json
import numpy as np
from ai_edge_litert.compiled_model import (
    CompiledModel,
    CpuOptions,
    HardwareAccelerator,
    Options,
)


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
    ib, ob = m.create_input_buffers(0), m.create_output_buffers(0)
    shp = [int(d) for d in det[sig["outputs"][0]]["shape"]]

    def call(x):
        ib[0].write(np.ascontiguousarray(x, np.float32))
        t = time.perf_counter()
        m.run_by_index(0, ib, ob)
        return ob[0].read(int(np.prod(shp)), np.float32).reshape(shp).astype(
            np.float64
        ), time.perf_counter() - t

    return call


inp = np.load("/b/p2/inputs.npz")
emb = np.load("/b/refs_emb.npz")["embeddings"]
out = {}
for path in sys.argv[1].split(","):
    for flags in (0, 4):
        call = load(path, flags)
        rows = []
        for i in range(3):
            e, dt = call(inp["images"][i])
            r = emb[i].ravel().astype(np.float64)
            e = e.ravel()
            fin = np.isfinite(e).all()
            rows.append(
                dict(
                    cos=float(np.dot(e, r) / np.linalg.norm(e) / np.linalg.norm(r))
                    if fin
                    else None,
                    nonfinite=int((~np.isfinite(e)).sum()),
                    s=round(dt, 3),
                )
            )
        out[f"{path.split('/')[-1]} f{flags}"] = rows
        print(path.split("/")[-1], "flags", flags, rows, flush=True)
for name, xin in (("p5_unsplit", "b0"), ("p4_sc", "b0"), ("p4_pc", "h1")):
    x = np.load(f"/b/p16/probe/{xin}.npy")
    a, _ = load(f"/b/p16/probe/{name}.tflite", 0)(x)
    b, _ = load(f"/b/p16/probe/{name}.tflite", 4)(x)
    c = float(np.dot(a.ravel(), b.ravel()) / np.linalg.norm(a) / np.linalg.norm(b))
    out[name] = c
    print(
        f"probe {name}: fp16 vs fp32 cos {c:.6f} nonfinite {int((~np.isfinite(b)).sum())}",
        flush=True,
    )
json.dump(out, open("/b/out/p16/enc16.json", "w"), indent=1)
