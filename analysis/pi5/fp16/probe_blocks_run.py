import numpy as np
from ai_edge_litert.compiled_model import (
    CompiledModel,
    CpuOptions,
    HardwareAccelerator,
    Options,
)

x = np.ascontiguousarray(np.load("/b/p2/inputs.npz")["patches"][0, :8], np.float32)


def run(flags):
    kw = dict(num_threads=4)
    if flags:
        kw["xnnpack_flags"] = flags
    m = CompiledModel.from_file(
        "/b/p16/tflite/dino_b8_allpre_blocks.tflite",
        options=Options(
            hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(**kw)
        ),
    )
    sig = m.get_signature_by_index(0)
    ib, ob = m.create_input_buffers(0), m.create_output_buffers(0)
    ib[0].write(x)
    m.run_by_index(0, ib, ob)
    return [
        b.read(8 * 201 * 768, np.float32).reshape(8, 201, 768).astype(np.float64)
        for name, b in zip(sig["outputs"], ob)
    ], sig["outputs"]


a, names = run(0)
b, _ = run(4)
order = sorted(
    range(12), key=lambda i: int("".join(c for c in names[i] if c.isdigit()) or 0)
)
for k, i in enumerate(order):
    A, B = a[i], b[i]
    bad = ~np.isfinite(B)
    toks = sorted(set(np.argwhere(bad)[:, 1].tolist()))
    fin = np.isfinite(B).all(-1)
    cos = (
        (A * np.where(np.isfinite(B), B, 0)).sum(-1)
        / np.linalg.norm(A, axis=-1)
        / np.linalg.norm(np.where(np.isfinite(B), B, 0), axis=-1)
    )
    print(
        f"out {names[i]:10s} fp32 max|x| {np.abs(A).max():9.1f} (prefix {np.abs(A[:, :5]).max():9.1f}, patches {np.abs(A[:, 5:]).max():8.1f})  fp16 nonfinite {int(bad.sum()):7d} tokens {toks[:8]}{'...' if len(toks) > 8 else ''}  least cos (finite tokens) {np.nanmin(np.where(fin, cos, np.nan)):.5f}"
    )
