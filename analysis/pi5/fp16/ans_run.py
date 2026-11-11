"""On the Pi: each packed photo's patches rebuilt bitwise (checked against the patch file's sha256), one DINOv3
.tflite over them (5 calls of 8), readouts saved as p7_answers.py run does: OUT/<cell>/<sha>.npy and
OUT/<cell>.jsonl (sha256, input_sha256, dino_s, nonfinite, s). Resumable.
    python ans_run.py PACKDIR MODEL.tflite FLAGS CELL OUT"""

import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from ai_edge_litert.compiled_model import (
    CompiledModel,
    CpuOptions,
    HardwareAccelerator,
    Options,
)

pack, model, flags, cell, out = (
    Path(sys.argv[1]),
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4],
    Path(sys.argv[5]),
)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
kw = dict(num_threads=4)
if flags:
    kw["xnnpack_flags"] = flags
m = CompiledModel.from_file(
    model,
    options=Options(
        hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(**kw)
    ),
)
ib, ob = m.create_input_buffers(0), m.create_output_buffers(0)
dest = out / cell
dest.mkdir(parents=True, exist_ok=True)
rows = [json.loads(ln) for ln in (pack / "manifest.jsonl").read_text().splitlines()]
w0 = time.perf_counter()
with (out / f"{cell}.jsonl").open("a") as f:
    for k, r in enumerate(rows, 1):
        if (dest / f"{r['sha256']}.npy").exists():
            continue
        t = time.perf_counter()
        u = np.load(pack / f"{r['sha256']}.npz")["u"]
        P = (u.astype(np.float32) / np.float32(255) - MEAN) / STD
        buf = io.BytesIO()
        np.save(buf, P, allow_pickle=False)
        input_sha = hashlib.sha256(buf.getvalue()).hexdigest()
        row = {"sha256": r["sha256"], "input_sha256": input_sha}
        if input_sha != r["patch_sha256"]:
            row["error"] = "rebuilt patches differ from the originals"
        else:
            tc = time.perf_counter()
            Y = np.zeros((40, 1536), np.float32)
            for b in range(5):
                ib[0].write(np.ascontiguousarray(P[b * 8 : (b + 1) * 8]))
                m.run_by_index(0, ib, ob)
                Y[b * 8 : (b + 1) * 8] = (
                    ob[0].read(8 * 1536, np.float32).reshape(8, 1536)
                )
            row |= {
                "dino_s": round(time.perf_counter() - tc, 3),
                "nonfinite": not bool(np.isfinite(Y).all()),
            }
            b2 = io.BytesIO()
            np.save(b2, Y, allow_pickle=False)
            tmp = dest / f"{r['sha256']}.npy.tmp"
            tmp.write_bytes(b2.getvalue())
            os.replace(tmp, dest / f"{r['sha256']}.npy")
        row["s"] = round(time.perf_counter() - t, 3)
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(
            f"photo {k}/{len(rows)} {r['role']} {row.get('dino_s')} s {row.get('error', '')} (elapsed {time.perf_counter() - w0:.0f} s)",
            flush=True,
        )
