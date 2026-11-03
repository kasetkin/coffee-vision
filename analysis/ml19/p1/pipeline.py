# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "imagecodecs==2026.10.10",
#     "numpy==2.5.3",
#     "opencv-python-headless==5.0.0.93",
#     "pillow==12.3.0",
#     "pillow-heif==1.8.0",
#     "rawpy==0.27.1",
# ]
# ///
"""ML-19 P1: a prototype of canonicalization (D44's recipe), to run the same on the three machines.

    uv run --script pipeline.py canon PHOTO... --out DIR [--effort E] [--strip ROWS] [--threads N]
    uv run --script pipeline.py srgb JXL... --out DIR
    uv run --script pipeline.py compare DIR_A DIR_B
    uv run --script pipeline.py info

`canon` writes DIR/<name>.jxl and one JSON line per photo (sha256, bytes, seconds per step, peak memory). `srgb`
decodes canonical files to 8-bit sRGB, DIR/<name>.png, and prints each array's sha256. `compare` reports the
largest level difference between two `srgb` outputs. Throwaway, not the P3 module: it has no tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import cv2
import imagecodecs as ic
import numpy as np
import pillow_heif
import rawpy
from PIL import Image

HERE = Path(__file__).resolve().parent
REC2020 = (HERE / "rec2020_bt709.icc").read_bytes()   # libjxl's own profile for RGB_D65_202_Rel_709 (D46)
REC2020_SHA256 = "877c5b36808954f322af187b56c3b3ebfd1cc8bad569f27f67f1a0c450d11b9c"
SRGB = ic.cms_profile("srgb")                          # LittleCMS's built-in sRGB, for untagged photos (D16)
RELATIVE = ic.CMS.INTENT.RELATIVE_COLORIMETRIC
NOOPTIMIZE = ic.CMS.FLAGS.NOOPTIMIZE
KEEP_MAX, CAP_TO = 60_000_000, 50_000_000              # D27
FILL_SRGB = (124, 116, 104)                            # segment_beans.py's fill, for transparent pixels (D13)
RAW_SUFFIXES = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2"}
LINEAR_REC2020 = ic.cms_profile("rgb", whitepoint=(0.3127, 0.3290),
                                primaries=(0.708, 0.292, 0.170, 0.797, 0.131, 0.046), gamma=1.0)

assert hashlib.sha256(REC2020).hexdigest() == REC2020_SHA256


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def orient(a: np.ndarray, orientation: int) -> np.ndarray:
    """EXIF orientation 1-8 applied as a view, not a copy (the colour step reads it a strip at a time); anything
    else as stored (D28)."""
    t = a.swapaxes(0, 1)
    ops = {2: lambda: a[:, ::-1], 3: lambda: a[::-1, ::-1], 4: lambda: a[::-1], 5: lambda: t,
           6: lambda: t[:, ::-1], 7: lambda: t[::-1, ::-1], 8: lambda: t[::-1]}
    return ops[orientation]() if orientation in ops else a


def decode(path: Path) -> tuple[np.ndarray, float, bytes | None, str]:
    """The main image at its decoder's depth, upright (maybe as a view), the value that means full scale, its
    profile (None for untagged) and what decoded it. One array per photo: no PIL image is converted, no copy is made
    to turn or widen it. Grey, grey with alpha, RGB and RGBA come back as decoded; CMYK is refused."""
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:   # D36: 16-bit, camera white balance, linear Rec.2020, the profile below
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(output_bps=16, use_camera_wb=True, output_color=rawpy.ColorSpace.Rec2020,
                                  gamma=(1, 1))
        return rgb, 65535.0, LINEAR_REC2020, "rawpy16"
    if suffix in {".heic", ".heif", ".hif"}:   # upright as decoded (D28), main image only (D20)
        heif = pillow_heif.open_heif(path, convert_hdr_to_8bit=False)
        icc, depth = heif.info.get("icc_profile") or None, heif.info.get("bit_depth", 8)
        rgb = np.asarray(heif)
        del heif
        # pillow-heif shifts a 10- or 12-bit sample into the top of 16 bits: 10-bit white is 1023 << 6 = 65472
        full = float((2**depth - 1) << (16 - depth)) if rgb.dtype == np.uint16 else 255.0
        return rgb, full, icc, f"heif{depth}"
    with Image.open(path) as im:   # opened for its metadata only: Pillow never loads the pixels here
        icc = im.info.get("icc_profile") or None
        orientation = im.getexif().get(0x0112, 1)
        fmt, mode = im.format, im.mode
    if mode == "CMYK":
        raise ValueError(f"{path.name}: CMYK is not handled by this prototype")
    data = path.read_bytes()
    if fmt in ("JPEG", "MPO"):     # the same pixels as Pillow's decode (5 of 5 test JPEGs, 2026-11-03), 3 bytes a pixel
        rgb = ic.jpeg8_decode(data)
    elif fmt == "PNG":             # Pillow reads 16-bit RGB PNG as 8 bits; imagecodecs keeps the depth
        rgb = ic.png_decode(data)
    else:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGBA" if "A" in im.getbands() else "RGB"))
    del data
    return orient(rgb, orientation), float(np.iinfo(rgb.dtype).max), icc, f"{fmt}{rgb.dtype.itemsize * 8}"


def cap(rgb: np.ndarray) -> np.ndarray:
    """Over 60,000,000 pixels to at most 50,000,000, by area averaging at the size given (D27, D43)."""
    h, w = rgb.shape[:2]
    if h * w <= KEEP_MAX:
        return rgb
    s = (CAP_TO / (h * w)) ** 0.5
    nw, nh = int(w * s), int(h * s)
    while nw * nh > CAP_TO:
        nw, nh = nw - 1, nh - 1
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def to_rec2020(rgb: np.ndarray, full: float, icc: bytes | None, strip: int) -> np.ndarray:
    """Source values through LittleCMS on float32, relative colorimetric, into the Rec.2020 profile, clipped to
    [0, 1] and rounded to uint16, a strip of rows at a time (D31, D38). Transparent pixels over the fill (D13).
    `rgb` may be a turned view, grey (2-D, or 1 or 2 channels with alpha) or RGB(A): each strip is copied out of it
    once, as float32, and every later step works in place, so a strip's temporaries are two float32 strips. `full`
    is the sample value that means 1.0."""
    h, w = rgb.shape[:2]
    scale = full
    out = np.empty((h, w, 3), np.uint16)
    profile = icc or SRGB
    channels = 1 if rgb.ndim == 2 else rgb.shape[2]
    colour = 1 if channels <= 2 else 3                 # grey is spread over R, G and B by broadcasting
    alpha_at = {2: 1, 4: 3}.get(channels)
    fill = None
    if alpha_at is not None:
        fill = ic.cms_transform(np.array([[FILL_SRGB]], np.float32) / 255, SRGB, REC2020, intent=RELATIVE,
                                outdtype=np.float32)[0, 0]
    for y in range(0, h, strip):
        part = rgb[y:y + strip]
        if part.ndim == 2:
            part = part[..., None]
        f = np.empty(part.shape[:2] + (3,), np.float32)
        f[...] = part[..., :colour]         # the one copy out of the (maybe turned, maybe grey) source
        f /= scale
        c = ic.cms_transform(f, profile, REC2020, intent=RELATIVE, outdtype=np.float32)
        np.clip(c, 0, 1, out=c)
        if fill is not None:
            f[...] = part[..., alpha_at:alpha_at + 1]   # the strip's alpha, reusing f: c * alpha + fill * (1 - alpha)
            f /= scale
            c *= f
            np.subtract(1, f, out=f)
            f *= fill
            c += f
        c *= 65535
        np.rint(c, out=c)
        out[y:y + strip] = c
    return out


def canonicalize(path: Path, effort: int, strip: int, threads: int) -> tuple[bytes, dict]:
    t0 = time.perf_counter()
    rgb, full, icc, kind = decode(path)
    t1 = time.perf_counter()
    rgb = cap(rgb)
    t2 = time.perf_counter()
    rec = to_rec2020(rgb, full, icc, strip)
    del rgb
    t3 = time.perf_counter()
    before = peak_mb()
    jxl = ic.jpegxl_encode(rec, level=95, effort=effort, primaries="BT2100", transfer="BT709", numthreads=threads)
    t4 = time.perf_counter()
    return jxl, {"decoded_as": kind, "profile": "own" if icc else "sRGB (untagged)",
                 "size": [rec.shape[1], rec.shape[0]], "decode_s": round(t1 - t0, 2), "cap_s": round(t2 - t1, 2),
                 "colour_s": round(t3 - t2, 2), "encode_s": round(t4 - t3, 2), "total_s": round(t4 - t0, 2),
                 "peak_before_encode_mb": round(before), "rec2020_sha256": hashlib.sha256(rec.data).hexdigest()}


def to_srgb(jxl: bytes, strip: int = 128) -> np.ndarray:
    """A canonical file to 8-bit sRGB: decoded at 16 bits, LittleCMS without its lossy optimization, relative
    colorimetric, clipped and rounded (D7)."""
    rec = ic.jpegxl_decode(jxl)
    if rec.dtype != np.uint16:
        raise ValueError(f"decoded as {rec.dtype}, not uint16")
    out = np.empty(rec.shape, np.uint8)
    for y in range(0, rec.shape[0], strip):   # written straight into its rows of `out`
        ic.cms_transform(rec[y:y + strip], REC2020, SRGB, intent=RELATIVE, flags=NOOPTIMIZE, outdtype=np.uint8,
                         out=out[y:y + strip])
    return out


def machine() -> dict:
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "")
    return {"host": platform.node(), "cpu": cpu, "glibc": platform.libc_ver()[1], "python": platform.python_version(),
            "imagecodecs": ic.__version__, "lcms": ic.cms_version(), "libjxl": ic.jpegxl_version(),
            "numpy": np.__version__, "opencv": cv2.__version__, "cv2_cpu": cv2.getCPUFeaturesLine()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("canon")
    c.add_argument("photos", nargs="+", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--effort", type=int, default=7)
    c.add_argument("--strip", type=int, default=128)
    c.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("srgb")
    s.add_argument("files", nargs="+", type=Path)
    s.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("compare")
    m.add_argument("a", type=Path)
    m.add_argument("b", type=Path)
    sub.add_parser("info")
    sub.add_parser("resize")
    fl = sub.add_parser("floor")
    fl.add_argument("npy", type=Path)
    fl.add_argument("--effort", type=int, default=7)
    fl.add_argument("--threads", type=int, default=2)
    fi = sub.add_parser("fidelity")
    fi.add_argument("photo", type=Path)
    fi.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    Image.MAX_IMAGE_PIXELS = 2**28
    if args.cmd == "info":
        print(json.dumps(machine()))
    elif args.cmd == "resize":   # D43's INTER_AREA, the same on every CPU? Fixed-seed images, all under the cap's
        # 60 MP so that the front fits them in 500M: cap() itself is the same call at a computed size.
        for dtype, (h, w) in ((np.uint8, (6000, 7000)), (np.uint16, (5000, 4000)), (np.uint8, (8165, 6145))):
            a = np.random.default_rng(19).integers(0, np.iinfo(dtype).max + 1, (h, w, 3), dtype=dtype)
            b = cap(a) if h * w > KEEP_MAX else cv2.resize(a, (w * 4 // 5, h * 4 // 5), interpolation=cv2.INTER_AREA)
            print(json.dumps({"input": [w, h, np.dtype(dtype).name], "output": [b.shape[1], b.shape[0]],
                              "sha256": hashlib.sha256(b.data).hexdigest(), "peak_mb": round(peak_mb())}))
            del a, b
    elif args.cmd == "floor":   # the least a whole-frame encode needs: the Rec.2020 frame loaded, then encoded
        rec = np.load(args.npy)
        loaded = peak_mb()
        t0 = time.perf_counter()
        jxl = ic.jpegxl_encode(rec, level=95, effort=args.effort, primaries="BT2100", transfer="BT709",
                               numthreads=args.threads)
        print(json.dumps({"megapixels": round(rec.shape[0] * rec.shape[1] / 1e6, 1), "effort": args.effort,
                          "threads": args.threads, "frame_mb": round(rec.nbytes / 2**20), "peak_loaded_mb": round(loaded),
                          "peak_mb": round(peak_mb()), "encode_s": round(time.perf_counter() - t0, 2), "bytes": len(jxl)}))
    elif args.cmd == "fidelity":   # each file's sRGB against the photo taken to sRGB directly, without JPEG XL
        # An analysis tool, not memory-lean: it holds the source, the reference and full-frame int16 differences.
        rgb, full, icc, _ = decode(args.photo)
        rgb = cap(rgb)
        ref = np.empty(rgb.shape[:2] + (3,), np.uint8)
        for y in range(0, rgb.shape[0], 512):
            strip = rgb[y:y + 512, :, :3].astype(np.float32) / full
            c = ic.cms_transform(strip, icc or SRGB, SRGB, intent=RELATIVE, outdtype=np.float32)
            ref[y:y + 512] = np.round(np.clip(c, 0, 1) * 255)
        del rgb
        for path in args.files:
            d = np.abs(to_srgb(path.read_bytes()).astype(np.int16) - ref).ravel()
            print(json.dumps({"file": str(path), "mean_level": round(float(d.mean()), 3),
                              "p99_level": int(np.percentile(d, 99)), "max_level": int(d.max())}))
    elif args.cmd == "canon":
        args.out.mkdir(parents=True, exist_ok=True)
        for photo in args.photos:
            jxl, record = canonicalize(photo, args.effort, args.strip, args.threads)
            (args.out / f"{photo.stem}.jxl").write_bytes(jxl)
            print(json.dumps({"photo": photo.name, "effort": args.effort, "threads": args.threads, "bytes": len(jxl),
                              "original_bytes": photo.stat().st_size, "sha256": hashlib.sha256(jxl).hexdigest(),
                              **record, "peak_mb": round(peak_mb())}), flush=True)
    elif args.cmd == "srgb":
        args.out.mkdir(parents=True, exist_ok=True)
        for f in args.files:
            t0 = time.perf_counter()
            a = to_srgb(f.read_bytes())
            dt = time.perf_counter() - t0
            cv2.imwrite(str(args.out / f"{f.stem}.png"), a[..., ::-1])
            print(json.dumps({"file": f.name, "sha256": hashlib.sha256(a.data).hexdigest(),
                              "seconds": round(dt, 2), "peak_mb": round(peak_mb())}), flush=True)
    else:
        worst = 0
        for pa in sorted(args.a.glob("*.png")):
            a = cv2.imread(str(pa)).astype(np.int16)
            b = cv2.imread(str(args.b / pa.name)).astype(np.int16)
            d = np.abs(a - b)
            worst = max(worst, int(d.max()))
            print(json.dumps({"file": pa.name, "max_level": int(d.max()), "pixels_differing": int((d.max(2) > 0).sum()),
                              "pixels": int(d.shape[0] * d.shape[1])}))
        print(json.dumps({"worst_level": worst, "within_1": worst <= 1}))
        sys.exit(0 if worst <= 1 else 1)


if __name__ == "__main__":
    main()
