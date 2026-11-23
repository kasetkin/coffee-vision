"""ML-37 D11: float resizers against today's 8-bit PIL bilinear, on canonical photos read back as 8-bit and as float.

Research only (analysis/ml37/research_d11_float_resize_and_augment.md); imports nothing it changes. Writes
analysis/ml37/d11_resize_parity.txt.

Inputs, per photo (a deterministic sample of <= 13 canonical photos, one per capture folder):
  u8   canonical.load_canonical's picture: decode_canonical (uint16 Rec.2020, BT.709 curve), LittleCMS to uint8 sRGB
  f32  the same decode, the same LittleCMS transform (REC2020 -> SRGB, relative, NOOPTIMIZE) with float32 output,
       clipped to [0, 1]: today's canonical photo read back without the 8-bit step (a stand-in for D2's read-back;
       the linear half-float photo does not exist yet)

Resizers on float32 [0, 1] HxWx3:
  pil_f      PIL "F" mode, channel by channel, Image.BILINEAR (D11 (a))
  cv_linear  cv2.resize INTER_LINEAR on float32 (D11 (b))
  cv_area    cv2.resize INTER_AREA on float32 (D11 (b))
  torch_aa   F.interpolate bilinear, align_corners=False, antialias=True on float32 NCHW (D11 (c))
Today: pil_u8, PIL RGB Image.BILINEAR on uint8 (model_io.sam_resize, eval_resize, the patch store).

Sizes: the SAM side (1024, XL0; model_io.sam_preprocess_shape) and the classifier's two steps, a crop of the photo
to patch_store_size 448 (the patch store; sides drawn log-uniform 224-1100 px so both down- and upscales occur) and
448 -> patch_resize 224 (eval_resize).

Differences are in 8-bit levels (x 255), against today's uint8 output (float - uint8, no rounding), and between the
float paths. "u8 as float" feeds the uint8 picture / 255 to the float path: the filter alone, no read-back change.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffeecv import canonical, model_io

torch.set_num_threads(2)
cv2.setNumThreads(2)

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".txt")
SAM_SIDE = 1024
STORE, RESIZE = 448, 224
N_PHOTOS = 13
PATCHES_PER_PHOTO = 12


def sample_photos() -> list[Path]:
    store = REPO / "data" / "canonical"
    files = sorted(p for p in store.rglob("*.jxl") if "ood_negatives" not in p.parts)
    groups: dict[str, list[Path]] = {}
    for p in files:
        groups.setdefault(p.relative_to(store).parts[0], []).append(p)
    rng = random.Random(37011)
    names = sorted(groups)
    rng.shuffle(names)
    return [rng.choice(groups[n]) for n in names[:N_PHOTOS]]


def read_back(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import imagecodecs as ic
    rec = canonical.decode_canonical(path.read_bytes(), numthreads=2)
    u8 = canonical.rec2020_to_srgb(rec, numthreads=2)
    f32 = np.empty(rec.shape, np.float32)
    for y in range(0, rec.shape[0], 128):
        ic.cms_transform(rec[y:y + 128], canonical.REC2020, canonical.SRGB, intent=canonical.RELATIVE,
                         flags=canonical.NOOPTIMIZE, outdtype=np.float32, out=f32[y:y + 128])
    del rec
    np.clip(f32, 0.0, 1.0, out=f32)
    return u8, f32


# ---- resizers, size as (w, h) ----

def pil_u8(a: np.ndarray, size) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize(size, Image.BILINEAR))


def pil_f(a: np.ndarray, size) -> np.ndarray:
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c])).resize(size, Image.BILINEAR))
                     for c in range(a.shape[2])], axis=2)


def cv_linear(a, size):
    return cv2.resize(a, size, interpolation=cv2.INTER_LINEAR)


def cv_area(a, size):
    return cv2.resize(a, size, interpolation=cv2.INTER_AREA)


def torch_aa(a, size):
    t = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None]
    o = F.interpolate(t, size=(size[1], size[0]), mode="bilinear", align_corners=False, antialias=True)
    return o[0].permute(1, 2, 0).numpy()


FLOAT = {"pil_f": pil_f, "cv_linear": cv_linear, "cv_area": cv_area, "torch_aa": torch_aa}


class Acc:
    """max |d|, sum |d|, sum d, n, and the share of pixels-channels with |d| > 1 level."""

    def __init__(self):
        self.mx, self.sa, self.s, self.n, self.over1 = 0.0, 0.0, 0.0, 0, 0

    def add(self, a: np.ndarray, b: np.ndarray):
        d = a.astype(np.float64) - b.astype(np.float64)
        ad = np.abs(d)
        self.mx = max(self.mx, float(ad.max()))
        self.sa += float(ad.sum())
        self.s += float(d.sum())
        self.n += d.size
        self.over1 += int((ad > 1.0).sum())

    def row(self, name: str) -> str:
        n = max(self.n, 1)
        return (f"  {name:<44} max {self.mx:7.3f}  mean|d| {self.sa / n:7.4f}  mean d {self.s / n:+8.4f}  "
                f">1 level {100 * self.over1 / n:6.3f}%")


def lv(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) * 255.0


def impulse_taps() -> list[str]:
    """How many input pixels each resizer reads per output pixel on a 4x downscale (a 1-D impulse test)."""
    out = []
    n_in, n_out = 64, 16
    for name, fn in FLOAT.items():
        widths = []
        for pos in range(20, 44):
            a = np.zeros((4, n_in, 3), np.float32)
            a[:, pos, :] = 1.0
            r = fn(a, (n_out, 4))[1, :, 0]
            widths.append(int((np.abs(r) > 1e-7).sum()))
        # read in reverse: which inputs feed output 8
        reach = []
        for pos in range(n_in):
            a = np.zeros((4, n_in, 3), np.float32)
            a[:, pos, :] = 1.0
            if abs(fn(a, (n_out, 4))[1, 8, 0]) > 1e-7:
                reach.append(pos)
        out.append(f"  {name:<10} output 8 of 16 reads inputs {reach} ({len(reach)} taps)")
    return out


def main() -> None:
    photos = sample_photos()
    lines = [f"ML-37 D11 resize parity. Pillow {Image.__version__}, OpenCV {cv2.__version__}, torch {torch.__version__}",
             f"photos: {len(photos)}; patches {PATCHES_PER_PHOTO} per photo, side log-uniform 224-1100 px", ""]
    lines.append("Impulse test, 64 -> 16 px (4x down), float32:")
    lines += impulse_taps()
    lines.append("")

    keys = ["readback f32 vs u8"]
    for stage in ("sam", "store", "eval", "chain"):
        keys += [f"{stage}: pil_f(u8 as float) vs pil_u8"]
        keys += [f"{stage}: {m}(f32) vs pil_u8" for m in FLOAT]
        keys += [f"{stage}: {m}(f32) vs pil_f(f32)" for m in FLOAT if m != "pil_f"]
    acc = {k: Acc() for k in keys}
    store_down, store_up = Acc(), Acc()
    store_down_tc, store_up_tc = Acc(), Acc()

    prng = np.random.default_rng(37011)
    for path in photos:
        u8, f32 = read_back(path)
        h, w = u8.shape[:2]
        lines.append(f"  {path.relative_to(REPO / 'data' / 'canonical').parts[0]:<36} {w}x{h}")
        print(path, w, h, file=sys.stderr, flush=True)
        acc["readback f32 vs u8"].add(lv(f32), u8)
        nh, nw = model_io.sam_preprocess_shape(h, w, SAM_SIDE)
        size = (nw, nh)
        ref = pil_u8(u8, size)
        assert np.array_equal(ref, model_io.sam_resize(u8, SAM_SIDE))
        acc["sam: pil_f(u8 as float) vs pil_u8"].add(lv(pil_f(u8.astype(np.float32) / 255, size)), ref)
        outs = {m: fn(f32, size) for m, fn in FLOAT.items()}
        for m, o in outs.items():
            acc[f"sam: {m}(f32) vs pil_u8"].add(lv(o), ref)
            if m != "pil_f":
                acc[f"sam: {m}(f32) vs pil_f(f32)"].add(lv(o), lv(outs["pil_f"]))
        del outs, ref

        for _ in range(PATCHES_PER_PHOTO):
            side = int(round(np.exp(prng.uniform(np.log(224), np.log(1100)))))
            side = min(side, h, w)
            y0, x0 = int(prng.integers(0, h - side + 1)), int(prng.integers(0, w - side + 1))
            cu, cf = u8[y0:y0 + side, x0:x0 + side], f32[y0:y0 + side, x0:x0 + side]
            s_ref = pil_u8(cu, (STORE, STORE))
            e_ref = np.asarray(model_io.eval_resize(Image.fromarray(s_ref), RESIZE))
            s_par = pil_f(cu.astype(np.float32) / 255, (STORE, STORE))
            acc["store: pil_f(u8 as float) vs pil_u8"].add(lv(s_par), s_ref)
            acc["eval: pil_f(u8 as float) vs pil_u8"].add(lv(pil_f(s_ref.astype(np.float32) / 255, (RESIZE, RESIZE))),
                                                         e_ref)
            acc["chain: pil_f(u8 as float) vs pil_u8"].add(lv(pil_f(s_par, (RESIZE, RESIZE))), e_ref)
            (store_down if side > STORE else store_up).add(lv(pil_f(cf, (STORE, STORE))), s_ref)
            (store_down_tc if side > STORE else store_up_tc).add(lv(torch_aa(cf, (STORE, STORE))),
                                                                 lv(pil_f(cf, (STORE, STORE))))
            st = {m: fn(cf, (STORE, STORE)) for m, fn in FLOAT.items()}
            # eval: every float resizer from the same float 448 patch (pil_f's), against today's eval on s_ref
            ev = {m: fn(st["pil_f"], (RESIZE, RESIZE)) for m, fn in FLOAT.items()}
            ch = {m: fn(st[m], (RESIZE, RESIZE)) for m, fn in FLOAT.items()}
            for stage, outs, ref in (("store", st, s_ref), ("eval", ev, e_ref), ("chain", ch, e_ref)):
                for m, o in outs.items():
                    acc[f"{stage}: {m}(f32) vs pil_u8"].add(lv(o), ref)
                    if m != "pil_f":
                        acc[f"{stage}: {m}(f32) vs pil_f(f32)"].add(lv(o), lv(outs["pil_f"]))
        del u8, f32

    lines.append("")
    lines.append("Differences in 8-bit levels (float x 255 minus uint8, or float minus float x 255):")
    for k in keys:
        lines.append(acc[k].row(k))
    lines.append("")
    lines.append("The patch store split by direction (pil_f(f32) vs pil_u8; torch_aa(f32) vs pil_f(f32)):")
    lines.append(store_down.row("store downscale (side > 448): pil_f vs pil_u8"))
    lines.append(store_up.row("store upscale (side <= 448): pil_f vs pil_u8"))
    lines.append(store_down_tc.row("store downscale: torch_aa vs pil_f"))
    lines.append(store_up_tc.row("store upscale: torch_aa vs pil_f"))
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
