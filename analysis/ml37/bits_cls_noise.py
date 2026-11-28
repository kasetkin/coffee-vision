"""ML-37 bits for the classifier, level 1: the quantization steps against the noise already in the picture.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/bits_cls_noise.py [--n 12] [--threads 8]

On --n photos of exp290's test split, one or more per session (sessions in turn, originals at most --max-mp MP):
1. The lossy coding error of the canonical photo (ADR 0036's store setting, q95 e7). The original is decoded and
   converted exactly as the store's encode does (canonical.decode, Source.peak, canonical.to_rec2020 with
   STORE_SETTINGS): `pre`, the uint16 picture libjxl was given. On the first --verify photos `pre` is encoded again
   (jxl_encode.encode_rec2020, q95 e7) and the bytes compared with the stored canonical photo, to check that `pre` is
   what the store encoded. The error is read-back(stored) - read-back(pre), float sRGB, on the crop's bean pixels.
   Also the front's canonical photo of the same original (canonical.canonicalize_with, FRONT_SETTINGS: q90 e3, the
   integer colour path, ADR 0036), what a served upload is made of, against `pre` the same way (where its size is
   the store's, i.e. the original is at most 18 MP).
2. A second generation: the stored photo's decoded samples re-encoded at q95 e7 and decoded; its error against the
   stored photo, the same way (the "re-encode at the same quality" yardstick).
3. The fine-scale noise of the bean pixels: Immerkaer's (1996) noise estimator per 16 x 16 block wholly on beans,
   on each channel of the float crop; the 10th percentile of the blocks (the flattest) and the median, and D10's
   robust second-difference sigma for comparison.
4. Each arm's own rounding error on the same pixels (RMS, largest).
All in 8-bit levels (x 255). Then, for level 2, the model inputs and DINOv3 features (f32 arm) of the photo's exp290
test boxes from `pre` (no lossy coding) and from the second generation, saved to tmp/ml37/bits_cls/noise/<rank>.npz
beside the input RMS differences; bits_cls_predict.py compares them with bits_cls_embed.py's f32 features.
Writes analysis/ml37/bits_cls_noise.txt.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import imagecodecs as ic
import numpy as np
import torch
from scipy.signal import convolve2d

from analysis.ml37 import bits_cls_common as bc
from analysis.ml37.bits_cls_check_boxes import exp290_test
from analysis.ml37.bits_cls_embed import embed
from coffeecv import canonical as cn
from coffeecv.backbones import build_backbone
from coffeecv.config import REPO_ROOT
from coffeecv.jxl_encode import encode_rec2020

OUT = Path(__file__).with_suffix(".txt")
OUT_DIR = bc.SCRATCH / "noise"
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
BLOCK = 16


def original_of(e: dict) -> Path:
    return REPO_ROOT / e["canonical"].replace("data/canonical/", "dataset/", 1)[:-len(".jxl")]


def pre_encode(path: Path) -> np.ndarray:
    """The uint16 Rec.2020 picture the store's encode hands libjxl (canonical.encode's region, whole)."""
    src = cn.decode(path.read_bytes(), path.name)
    return cn.to_rec2020(src.pixels, src.full, src.profile, settings=cn.STORE_SETTINGS, linearize=src.linearize,
                         peak=src.peak())


def jxl(rec: np.ndarray, threads: int) -> bytes:
    return encode_rec2020(rec.shape[0], rec.shape[1], lambda y, x, ys, xs: rec[y:y + ys, x:x + xs],
                          level=cn.QUALITY, effort=cn.EFFORT, numthreads=threads, transfer="BT709", intensity_target=255)


def err_stats(d: np.ndarray, mask: np.ndarray | None) -> tuple[float, float, float, float]:
    """(RMS, mean, 99th percentile of |d|, max |d|) of a float difference on bean pixels, in 8-bit levels."""
    v = (d[mask] if mask is not None else d.reshape(-1, 3)).ravel().astype(np.float64) * 255
    a = np.abs(v)
    return float(np.sqrt(np.mean(v ** 2))), float(v.mean()), float(np.percentile(a, 99)), float(a.max())


def immerkaer_blocks(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Immerkaer's sigma (sqrt(pi/2) / (6 (W-2)(H-2)) sum |I * N|) of each 16 x 16 block wholly on beans, per
    channel, in 8-bit levels."""
    h, w = x.shape[:2]
    hb, wb = h // BLOCK, w // BLOCK
    out = []
    for c in range(3):
        r = np.abs(convolve2d(x[..., c].astype(np.float64) * 255, IMMERKAER, mode="same"))
        r = r[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK)[:, 1:-1, :, 1:-1]
        s = np.sqrt(np.pi / 2) * r.sum(axis=(1, 3)) / (6 * (BLOCK - 2) ** 2)
        if mask is not None:
            full = mask[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).all(axis=(1, 3))
            s = s[full]
        out.append(s.ravel())
    return np.concatenate(out)


def fine_sigma(green: np.ndarray, mask: np.ndarray | None) -> float:
    """D10's measure: robust sigma (1.4826 MAD / sqrt 6) of the horizontal second difference, bean pixels, levels."""
    d = green[:, 2:] - 2 * green[:, 1:-1] + green[:, :-2]
    if mask is not None:
        d = d[mask[:, 1:-1] & mask[:, 2:] & mask[:, :-2]]
    d = d.ravel()
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6) * 255)


def pick(test: list[dict], entries: dict, n: int, max_mp: float) -> list[tuple[int, dict]]:
    by = defaultdict(list)
    for rank, t in enumerate(test):
        e = entries[t["photo"]]
        if e["size"][0] * e["size"][1] / 1e6 <= max_mp and original_of(e).exists():
            by[e["session"]].append((rank, t))
    out, k = [], 0
    while len(out) < n and any(by.values()):
        for s in sorted(by):
            if by[s] and len(out) < n:
                out.append(by[s].pop(k % len(by[s])))
        k += 7
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--verify", type=int, default=3)
    ap.add_argument("--max-mp", type=float, default=16.0, help="crop megapixels at most (memory)")
    ap.add_argument("--threads", type=int, default=bc.THREADS)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geo = bc.Geometry.from_params()
    entries = bc.pool_entries()
    test = exp290_test()
    chosen = pick(test, entries, a.n, a.max_mp)
    bb = build_backbone("dinov3_vitb16")
    arms = ("f16", "u12", "u10", "u8")
    lines = [f"{len(chosen)} photos of exp290's test split, {a.threads} threads; all figures in 8-bit levels "
             "(1/255 of [0, 1] sRGB), on the crop's bean pixels, all three channels pooled",
             "",
             "photo | session | crop MP | coding error q95e7: RMS, mean, p99, max | second generation: RMS, p99, max | front q90e3: RMS, p99 "
             "| Immerkaer block sigma p10, median | D10 sigma | arm rounding RMS: " + ", ".join(arms)]
    agg = defaultdict(list)
    scale = (bc.STD * 255).astype(np.float32)
    for i, (rank, t) in enumerate(chosen):
        e = entries[t["photo"]]
        stored_bytes = (REPO_ROOT / e["canonical"]).read_bytes()
        stored = cn.decode_canonical(stored_bytes, a.threads)
        pre = pre_encode(original_of(e))
        if pre.shape != stored.shape:
            print(f"skip {t['photo']}: pre {pre.shape} stored {stored.shape}")
            continue
        verified = ""
        if i < a.verify:
            verified = " bytes==stored" if jxl(pre, a.threads) == stored_bytes else " BYTES DIFFER"
        gen2 = ic.jpegxl_decode(jxl(stored, a.threads), numthreads=a.threads)
        front = cn.decode_canonical(cn.canonicalize_with(original_of(e).read_bytes(), original_of(e).name,
                                                         cn.FRONT_SETTINGS).canonical, a.threads)
        if front.shape != pre.shape:
            front = None
        x_st, mask = bc.float_crop(e, a.threads, rec=stored)
        x_pre, _ = bc.float_crop(e, a.threads, rec=pre)
        x_g2, _ = bc.float_crop(e, a.threads, rec=gen2)
        x_fr = bc.float_crop(e, a.threads, rec=front)[0] if front is not None else None
        stored = pre = gen2 = front = None
        coding = err_stats(x_st - x_pre, mask)
        second = err_stats(x_g2 - x_st, mask)
        fr = err_stats(x_fr - x_pre, mask) if x_fr is not None else (np.nan,) * 4
        if x_fr is not None:
            agg["front_rms"].append(fr[0])
        blocks = immerkaer_blocks(x_st, mask)
        p10, med = float(np.percentile(blocks, 10)), float(np.median(blocks))
        fs = fine_sigma(x_st[..., 1], mask)
        rnd = [err_stats(bc.quantize(x_st, arm) - x_st, mask) for arm in arms]
        agg["coding_rms"].append(coding[0])
        agg["coding_mean"].append(coding[1])
        agg["second_rms"].append(second[0])
        agg["p10"].append(p10)
        agg["med"].append(med)
        agg["fs"].append(fs)
        for arm, r in zip(arms, rnd):
            agg[f"rnd_{arm}"].append(r[0])
        lines.append(f"{t['photo'][:44]:44s} | {e['session'][:24]:24s} | {x_st.shape[0] * x_st.shape[1] / 1e6:5.1f} | "
                     f"{coding[0]:.3f} {coding[1]:+.4f} {coding[2]:.2f} {coding[3]:.2f}{verified} | "
                     f"{second[0]:.3f} {second[2]:.2f} {second[3]:.2f} | {fr[0]:.3f} {fr[2]:.2f} | {p10:.2f} {med:.2f} | {fs:.2f} | "
                     + " ".join(f"{r[0]:.4f}" for r in rnd))
        print(lines[-1], flush=True)
        # level 2 yardsticks: the f32 arm without lossy coding and after a second generation, on exp290's boxes
        u8 = np.rint(x_st * 255).astype(np.uint8)
        boxes, _ = geo.boxes(u8, mask, 123, t["class_idx"], t["photo_idx"], "test", t["n"])
        forms = {"f32": x_st, "pre": x_pre, "gen2": x_g2, "u8": x_st}
        if x_fr is not None:
            forms["front"] = x_fr
        inp = {k: [bc.model_input(bc.cut(v, b), "u8" if k == "u8" else "f32") for b, _, _ in boxes]
               for k, v in forms.items()}
        ref = np.stack(inp["f32"])
        rms = {k: float(np.sqrt(np.mean(((np.stack(v) - ref) * scale) ** 2))) for k, v in inp.items()}
        ks = [k for k in ("pre", "gen2", "front") if k in inp]
        feats = embed(bb, [v for k in ks for v in inp[k]]).reshape(len(ks), len(boxes), -1)
        np.savez(OUT_DIR / f"{rank:04d}.npz", **{k: feats[j] for j, k in enumerate(ks)}, rank=rank,
                 input_rms=np.array([rms["pre"], rms["gen2"], rms["u8"], rms.get("front", np.nan)]))
        x_st = x_pre = x_g2 = x_fr = None
    lines += ["", "medians over photos (min-max):"]
    for k, label in [("coding_rms", "coding error RMS (q95 e7, store)"), ("coding_mean", "coding error mean"),
                     ("second_rms", "second-generation error RMS"), ("front_rms", "front (q90 e3) error RMS"),
                     ("p10", "Immerkaer sigma, flattest 10% of blocks"), ("med", "Immerkaer sigma, median block"),
                     ("fs", "D10 second-difference sigma, green")] + [(f"rnd_{arm}", f"rounding RMS {arm}")
                                                                       for arm in arms]:
        v = np.array(agg[k])
        lines.append(f"  {label:44s} {np.median(v):.4f} ({v.min():.4f}-{v.max():.4f})")
    lines.append(f"peak RSS {bc.rss_mb():.0f} MB")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
