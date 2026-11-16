"""ML-29 P7's checks of coffeecv.gain_map (the plan's P7; ticket R5, D14).

    python analysis/ml29/gain_map_check.py [--per-brand 20]

1. R5: on the dataset photos that carry two forms of map (ISO 21496-1 and the vendor's: Samsung's Ultra HDR JPEGs,
   Apple's HEICs), the rendition each form asks for, at the map's resolution, from the main image's light shrunk to
   it: the difference in stops (log2 of ISO's light over the vendor's) where the main image is above 1% of white.
2. libultrahdr: on Ultra HDR JPEGs (`--per-brand` of each brand), this module's rendition at full size against the
   one libultrahdr itself decodes (imagecodecs.ultrahdr_decode, linear, white 1.0), where the main image is above 5%.
3. The public samples (tmp/ml29_samples, libavif's test files; D29: not committed): libavif's map up from a
   standard-range AVIF against Photoshop's PQ rendition of the same photo, its map down from a PQ AVIF against the
   standard-range one, and Adobe's JPEG (per-channel hdrgm, which libultrahdr refuses) read. The two AVIFs carry the
   same map, but the PQ one is not that map applied (its light reaches 3.7, the map's headroom 1.3 stops, 2.5): they
   are two renditions made apart, so the difference is theirs, and it is the same up and down.

The main images are read as their decoder gives them, the sRGB curve undone (the JPEGs' and HEICs' profiles all use
it); a map applies per pixel and, with one channel and equal offsets, the same in any primaries. Reads
tmp/ml29_scan/files.csv (scan_dataset.py); writes nothing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import sys
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv import colour_declaration as cd  # noqa: E402
from coffeecv import gain_map as gm  # noqa: E402

SAMPLES = REPO / "tmp" / "ml29_samples"
Image.MAX_IMAGE_PIXELS = None


def srgb_decode(u: np.ndarray) -> np.ndarray:
    return np.where(u > 0.04045, ((u + 0.055) / 1.055) ** 2.4, u / 12.92).astype(np.float32)


def main_light(data: bytes, size: tuple[int, int] | None = None) -> np.ndarray:
    """The main image's linear light as its decoder gives it (a JPEG as stored, a HEIC turned), at `size` (w, h)."""
    if data[:3] == b"\xff\xd8\xff":
        im = Image.open(io.BytesIO(data))
        if size:
            im.draft("RGB", size)
        im = im.convert("RGB")
    else:
        import pillow_heif
        im = Image.fromarray(np.asarray(pillow_heif.open_heif(io.BytesIO(data))))
    if size:
        im = im.resize(size, Image.BILINEAR)
    return srgb_decode(np.asarray(im, np.float32) / 255)


def stops(a: np.ndarray, b: np.ndarray, where: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(a[where], 1e-6) / np.maximum(b[where], 1e-6))


def both_forms(rows: list[dict]) -> None:
    print("1. R5: ISO 21496-1 against the vendor's map, on photos carrying both (stops, at the map's size)")
    by_brand = collections.defaultdict(list)
    for r in rows:
        if "ISO 21496-1" not in r["gain_map"] or ("hdrgm" not in r["gain_map"] and "Apple" not in r["gain_map"]):
            continue
        data = Path(r["file"]).read_bytes()
        forms = {g.form: g for g in gm.read_all(data)}
        iso, vendor = forms.get("ISO 21496-1"), forms.get("Ultra HDR") or forms.get("Apple")
        if iso is None or vendor is None:
            by_brand[r["make"]].append({"missing": sorted(forms)})
            continue
        h, w = iso.pixels.shape[:2]
        base = main_light(data, (w, h))
        lit = base.max(axis=2) > 0.01
        a, b = gm.apply(base, iso), gm.apply(base, vendor)
        d = np.abs(stops(a, b, lit))
        by_brand[r["make"]].append({"mean": float(d.mean()), "p99": float(np.percentile(d, 99)),
                                    "iso": float(stops(a, base, lit).mean()), "vendor": float(stops(b, base, lit).mean())})
    for brand, found in by_brand.items():
        ok = [f for f in found if "mean" in f]
        missing = collections.Counter(str(f["missing"]) for f in found if "missing" in f)
        if ok:
            print(f"   {brand:10s} {len(ok):4d} photos: |ISO - vendor| mean {np.median([f['mean'] for f in ok]):.3f} "
                  f"(median photo; worst {max(f['mean'] for f in ok):.3f}), p99 {np.median([f['p99'] for f in ok]):.3f} "
                  f"(worst {max(f['p99'] for f in ok):.3f}); boost ISO {np.median([f['iso'] for f in ok]):.2f}, "
                  f"vendor {np.median([f['vendor'] for f in ok]):.2f}"
                  + (f"; not read: {dict(missing)}" if missing else ""))
        else:
            print(f"   {brand:10s} none read: {dict(missing)}")


def libultrahdr(rows: list[dict], per_brand: int) -> None:
    print(f"2. libultrahdr against this module, Ultra HDR JPEGs, {per_brand} per brand, full size "
          "(|log2 ours / libultrahdr's| where the main image is above 5% of white)")
    taken, beyond = collections.Counter(), collections.Counter()
    by_brand = collections.defaultdict(list)
    for r in rows:
        if r["type"] != "JPEG" or "hdrgm" not in r["gain_map"] or taken[r["make"]] >= per_brand:
            continue
        data = Path(r["file"]).read_bytes()
        try:
            theirs = ic.ultrahdr_decode(data).astype(np.float32)[..., :3]
        except ic.UltrahdrError as exc:              # e.g. a 200 MP photo: this libultrahdr stops at 8192 x 8192
            beyond[r["make"]] += 1
            print(f"   (libultrahdr cannot: {Path(r['file']).name}: {str(exc)[:70]})")
            continue
        taken[r["make"]] += 1
        g = gm.read(data)
        base = srgb_decode(ic.jpeg8_decode(data).astype(np.float32) / 255)
        ours = gm.apply(base, g)
        lit = base.max(axis=2) > 0.05
        d = np.abs(stops(ours, theirs, lit))
        by_brand[r["make"]].append((g.form, float(d.mean()), float(np.percentile(d, 99)), float(theirs.max())))
    for brand, found in by_brand.items():
        forms = collections.Counter(f[0] for f in found)
        print(f"   {brand:22s} {len(found):3d} ({dict(forms)}): mean {np.median([f[1] for f in found]):.4f} "
              f"(worst {max(f[1] for f in found):.4f}), p99 {np.median([f[2] for f in found]):.4f} "
              f"(worst {max(f[2] for f in found):.4f}); libultrahdr's brightest {max(f[3] for f in found):.2f}"
              + (f"; {beyond[brand]} beyond libultrahdr's size" if beyond[brand] else ""))


def avif_light(data: bytes) -> np.ndarray:
    """An AVIF's linear light from its nclx (sRGB or PQ here), relative to standard-range white."""
    rgb = ic.avif_decode(data)
    depth = 8 if rgb.dtype == np.uint8 else cd.primary_bit_depth(data)
    a = rgb[..., :3].astype(np.float64) / (2**depth - 1)
    cd.CURVE_DECODERS[cd.declare(data, "avif").curve](a)
    return a.astype(np.float32)


def samples() -> None:
    print("3. The public samples (libavif's test files): stops, where the reference is above 1% of white")
    if not SAMPLES.is_dir():
        print(f"   {SAMPLES} missing")
        return
    pairs = (("seine_sdr_gainmap_srgb.avif", "seine_hdr_srgb.avif", "map up from standard range, against "
              "Photoshop's PQ rendition"),
             ("seine_hdr_gainmap_srgb.avif", "seine_sdr_gainmap_srgb.avif", "map down from PQ, against the "
              "standard-range image"))
    for name, ref_name, what in pairs:
        data = (SAMPLES / name).read_bytes()
        g = gm.read(data)
        out = gm.apply(avif_light(data), g)
        ref = avif_light((SAMPLES / ref_name).read_bytes())
        lit = ref.max(axis=2) > 0.01
        d = stops(out, ref, lit)
        print(f"   {name}: {g.form}, {'down' if g.metadata.leads_down else 'up'}, {g.pixels.shape[2]} channels; "
              f"{what}: mean {d.mean():+.3f}, |mean| {np.abs(d).mean():.3f}, p99 |{np.percentile(np.abs(d), 99):.3f}|; "
              f"brightest {out.max():.2f} (reference {ref.max():.2f})")
    jpeg = (SAMPLES / "seine_sdr_gainmap_srgb.jpg").read_bytes()
    g = gm.read(jpeg)
    up = gm.apply(srgb_decode(ic.jpeg8_decode(jpeg).astype(np.float32) / 255), g)
    avif = gm.apply(avif_light((SAMPLES / "seine_sdr_gainmap_srgb.avif").read_bytes()),
                    gm.read((SAMPLES / "seine_sdr_gainmap_srgb.avif").read_bytes()))
    lit = avif.max(axis=2) > 0.01
    print(f"   seine_sdr_gainmap_srgb.jpg: {g.form}, gain max {tuple(round(x, 3) for x in g.metadata.gain_max)}; "
          f"against the AVIF made from it: |mean| {np.abs(stops(up, avif, lit)).mean():.3f} stops")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-brand", type=int, default=20)
    ap.add_argument("--sections", default="123", help="which of the three checks to run")
    ap.add_argument("--scan", type=Path, default=REPO / "tmp" / "ml29_scan" / "files.csv")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(a.scan.open()) if r["gain_map"]]
    if "1" in a.sections:
        both_forms(rows)
    if "2" in a.sections:
        libultrahdr(rows, a.per_brand)
    if "3" in a.sections:
        samples()


if __name__ == "__main__":
    main()
