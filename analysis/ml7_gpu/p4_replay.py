"""ML-7 P4: the photo replay of one cell (plan P4), in a fresh process that loads the encoder, the decoder and
DINOv3 together, at the DINOv3 batch the cell ran faster. It prints a line when the graphs are loaded and one per
photo as it is answered. For each photo, as serving runs it: encoder and decoder
(the /crop pass), encoder and decoder again (/classify; OPS-7 would remove one), then the photo's 40 patches.
--passes passes over the photos. The decoder reads the replay's own embedding, so the outputs are the cell's end
to end. Pre- and post-processing stay out: they are the same for every cell, and P0 has their times.

Writes <out>.npz, the first pass's outputs for parity.py (low_masks and ious of the /classify pass, readouts),
and <out>.json: the time from the start of loading to the first photo's answer (cold start, kernel compilation
included); every photo's wall time and their median and p90 (the first photo, cold, left out); peak RSS and the
drop in MemAvailable with the three graphs loaded; the calls with a NaN or Inf; the counters across the replay;
and on a GPU cell the GPU proof. Run by p4_run.sh with the cell's venv:

    <venv>/bin/python p4_replay.py --cell webgpu --batch 8 --model-dir p2 --inputs p2/inputs.npz --threads 4 \\
        --passes 2 --out p4/out/r1/webgpu__replay
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backends  # noqa: E402
import cellrun  # noqa: E402
from run_litert import CapturedStderr, delegate_report  # noqa: E402


def replay_photo(enc, dec, dino, image, box, patches, batch: int) -> tuple:
    """One photo as serving runs it: (low_masks, ious) of the second decoder pass, readouts [40, 1536]."""
    dec(enc(image), box)                                       # /crop
    low, ious = dec(enc(image), box)                           # /classify
    readouts = np.concatenate([dino(patches[i:i + batch]) for i in range(0, len(patches), batch)])
    return low, ious, readouts


def main() -> None:
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=sorted(backends.CELLS), required=True)
    ap.add_argument("--batch", type=int, choices=(8, 40), required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--passes", type=int, required=True, help="2 in plan P4 (p4_run.sh holds the numbers)")
    ap.add_argument("--photos", type=int, default=None, help="the first N photos only (a rehearsal)")
    ap.add_argument("--runtime-dir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inputs = dict(np.load(args.inputs, allow_pickle=False))
    if args.photos:
        inputs = {k: v[:args.photos] for k, v in inputs.items()}
    cell = backends.CELLS[args.cell]
    args.out.parent.mkdir(parents=True, exist_ok=True)          # the capture's file lives beside the outputs
    sampler = cellrun.Sampler()
    sampler.start()
    probe = cellrun.Probe()
    mem0 = cellrun.mem_available_mb()
    try:
        with CapturedStderr(args.out.parent / (args.out.name + ".stderr")) as err:
            t_load = time.perf_counter()
            mods = backends._torch_modules(args.threads) if cell.runtime == "torch" else None
            enc, dec, dino = (backends.open_graph(args.cell, g, args.model_dir, args.threads, args.runtime_dir,
                                                  args.cache_dir, _torch=mods)
                              for g in ("encoder", "decoder", f"dino_b{args.batch}"))
            loaded_s = time.perf_counter() - t_load
            print(f"loaded {args.cell} (encoder, decoder, dino_b{args.batch}) in {loaded_s:.1f} s", flush=True)
            photo_s, outs, nonfinite, first_answer_s = [], [], 0, None
            total = args.passes * len(inputs["photos"])
            w0, cpu0, rc6_0 = time.perf_counter(), cellrun.cpu_seconds(), probe.rc6_ms()
            for p in range(args.passes):
                for i in range(len(inputs["photos"])):
                    t = time.perf_counter()
                    out = replay_photo(enc, dec, dino, inputs["images"][i], inputs["boxes"][i],
                                       inputs["patches"][i], args.batch)
                    photo_s.append(time.perf_counter() - t)
                    if first_answer_s is None:
                        first_answer_s = time.perf_counter() - t_load
                    print(f"photo {len(photo_s)}/{total} pass {p + 1} {inputs['photos'][i]} {photo_s[-1]:.2f} s "
                          f"(elapsed {time.perf_counter() - t_load:.0f} s)", flush=True)
                    nonfinite += not cellrun.finite(out)
                    if p == 0:
                        outs.append(out)
            w1, cpu1, rc6_1 = time.perf_counter(), cellrun.cpu_seconds(), probe.rc6_ms()
    finally:
        sampler.stop()
    counters = sampler.between(w0, w1)
    if rc6_0 is not None and rc6_1 is not None:
        counters["rc6_ms_delta"] = rc6_1 - rc6_0
    warm = sorted(photo_s[1:])
    seen = cellrun.loaded()
    mem_min = counters.get("mem_available_mb_min")
    record = {
        "cell": args.cell, "batch": args.batch, "passes": args.passes, "threads": args.threads,
        "graphs": {"encoder": enc.info, "decoder": dec.info, "dino": dino.info},
        "load_s": round(loaded_s, 3), "first_answer_s": round(first_answer_s, 3),
        "process_to_first_answer_s": round(t_load - t_start + first_answer_s, 3),
        "photo_s": [round(s, 3) for s in photo_s],
        "photo": {"n": len(warm), "median": statistics.median(warm),
                  "p90": warm[min(len(warm) - 1, int(0.9 * (len(warm) - 1) + 0.5))]},
        "nonfinite_photos": nonfinite, "cpu_over_wall": round((cpu1 - cpu0) / (w1 - w0), 2),
        "peak_rss_mb": cellrun.peak_rss_mb(), "mem_available_mb_before": mem0,
        "mem_available_drop_mb": round(mem0 - mem_min, 1) if mem0 is not None and mem_min is not None else None,
        "counters": counters, "loaded": seen,
    }
    if cell.path is not None:
        record["gpu_proof"] = cellrun.gpu_proof(seen, counters, cell.path)
    if cell.runtime == "litert" and cell.device == "gpu":
        record["delegate"] = delegate_report(err.text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out.with_suffix(".npz"), low_masks=np.stack([o[0][0] for o in outs]),
             ious=np.stack([o[1][0] for o in outs]), readouts=np.stack([o[2] for o in outs]))
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"cell": args.cell, "batch": args.batch, "first_answer_s": record["first_answer_s"],
                      "photo_median_s": record["photo"]["median"], "photo_p90_s": record["photo"]["p90"],
                      "nonfinite_photos": nonfinite, "peak_rss_mb": record["peak_rss_mb"],
                      "gpu_proof": record.get("gpu_proof", {}).get("ok")}))


if __name__ == "__main__":
    main()
