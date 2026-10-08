"""ML-5 P4 (D18, R6): a segmenter variant's time per photo, and the fine-tune cache it implies. Reported, no limit.

Per photo, median of --runs: decode, resize, encoder, decoder, upsample (segment_beans' serving path: whole-image
box, multi3). Then the seg_embed_cache work for one photo, timed with seg_finetune's own pieces: load once, then
per view (seg_ft.views) the dihedral transform, the encoder, the fp16 store and the label in the encoder's frame.
The estimate for --photos photos x the views: size = emb (256 x 64 x 64 fp16) + label (frame x frame uint8) per
photo and view; build time = photos x (load + views x per-view median), at the photos' mean.

Photos: analysis/ml2_p0/timing_photos.txt (ML-2 P0's latency pair, one 12 MP pixel, one 19 MP sony), so the
numbers line up with ML-2's L0 table (analysis/ml2_p0/README.md). On the VM, idle, at the pinned 4 threads:

    PYTHONPATH=. python analysis/ml5_xl0_timing/xl0_timing.py --variant xl0 --threads 4 \
        --out analysis/ml5_xl0_timing/timing_vm_xl0_t4.json

Needs the photos and models_pretrained/efficientvit_sam/efficientvit_sam_<variant>.pt (dvc pull those targets).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from coffeecv import seg_finetune as ft
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.sam_loader import VARIANTS, weights_for
from coffeecv.segment_beans import BeanSegmenter, SegParams, segment_and_crop

PHOTO_LIST = REPO_ROOT / "analysis" / "ml2_p0" / "timing_photos.txt"
EMB_BYTES = 2 * int(np.prod(ft.EMB_SHAPE))       # fp16


def _med(rows: list[dict]) -> dict:
    return {k: round(statistics.median(r[k] for r in rows), 1) for k in rows[0]}


@torch.inference_mode()
def cache_view_ms(seg: BeanSegmenter, rgb: np.ndarray, views: tuple[str, ...]) -> dict:
    """One photo through build_cache's per-view loop body (the label is the photo's own mask; its content does
    not change the cost). ms per view, keyed by view."""
    model, frame = seg.model, ft.encoder_frame(seg.model)
    mask = np.zeros(rgb.shape[:2], bool)
    mask[rgb.shape[0] // 4: 3 * rgb.shape[0] // 4] = True
    out = {}
    for v in views:
        t0 = time.perf_counter()
        im = np.ascontiguousarray(ft.VIEWS[v](rgb))
        m = np.ascontiguousarray(ft.VIEWS[v](mask))
        model.image_encoder(model.transform(im).unsqueeze(0))[0].numpy().astype(np.float16)
        ft.label_in_frame(m, frame)
        out[v] = (time.perf_counter() - t0) * 1e3
    return out


def run(variant: str, runs: int, n_photos: int, views: tuple[str, ...]) -> dict:
    seg = BeanSegmenter(SegParams(mask_select="multi3", prompt="box", weights=weights_for(variant)))
    frame = ft.encoder_frame(seg.model)
    photos = [REPO_ROOT / s for line in PHOTO_LIST.read_text().splitlines() if (s := line.split("#", 1)[0].strip())]
    report = {"variant": variant, "frame": frame, "threads": torch.get_num_threads(), "runs": runs,
              "weights_sha256": seg.weights_sha256, "photos": {}}
    loads, per_view = [], []
    for path in photos:
        serve, cache = [], []
        for _ in range(runs):
            t0 = time.perf_counter()
            rgb = load_rgb_image(path)
            decode = (time.perf_counter() - t0) * 1e3
            crop = segment_and_crop(rgb, seg)
            serve.append({"decode": decode, **{k: crop.info["timing_ms"][k]
                                                for k in ("resize", "encoder", "decoder", "upsample")}})
            cache.append(cache_view_ms(seg, rgb, views))
        s, c = _med(serve), _med(cache)
        loads.append(s["decode"])
        per_view.append(statistics.mean(c.values()))
        report["photos"][str(path.relative_to(REPO_ROOT))] = {"shape": list(rgb.shape[:2]), "serving_median_ms": s,
                                                              "cache_per_view_median_ms": c}
        print(f"{path.name} {rgb.shape[1]}x{rgb.shape[0]}  " + "  ".join(f"{k} {v:.0f}" for k, v in s.items())
              + "  | cache per view " + "  ".join(f"{k} {v:.0f}" for k, v in c.items()) + "  (ms)", flush=True)
    per_photo_s = (statistics.mean(loads) + len(views) * statistics.mean(per_view)) / 1e3
    size = n_photos * len(views) * (EMB_BYTES + frame * frame)
    report["cache_estimate"] = {"photos": n_photos, "views": list(views), "bytes": size,
                                "gib": round(size / 2 ** 30, 2), "build_seconds_per_photo": round(per_photo_s, 2),
                                "build_minutes": round(n_photos * per_photo_s / 60, 1)}
    print(json.dumps(report["cache_estimate"]))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--threads", type=int, required=True, help="torch.set_num_threads (the VM's stages pin 4)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--photos", type=int, default=410, help="photos in the cache estimate (ticket ML-5: ~410)")
    ap.add_argument("--out", type=Path, help="write the report as JSON")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    views = ft.FtParams.from_config(RunConfig.from_params_yaml()).views
    report = run(args.variant, args.runs, args.photos, views)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
