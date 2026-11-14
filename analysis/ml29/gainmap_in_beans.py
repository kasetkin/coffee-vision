"""How much a photo's gain map would brighten its bean region, and the rest of it (ticket ML-29, Q15's first step).

    python analysis/ml29/gainmap_in_beans.py [--masks data/seg_masks_ml19] [--out tmp/ml29_scan]

For every hand-checked full-photo bean mask whose original carries a gain map, the boost the map asks for at full
HDR, in stops (log2 of HDR light over standard-range light), measured at the map's own resolution, inside the mask
and outside it. Nothing is re-canonicalized and no model runs.

- JPEG (Ultra HDR): the second image's hdrgm parameters, as the Ultra HDR format defines the boost:
  log2 boost = GainMapMin + (GainMapMax - GainMapMin) * map ** (1 / Gamma); HDR = (SDR + OffsetSDR) * 2**boost - OffsetHDR.
- HEIC (Apple): Apple's published formula: HDR = SDR * (1 + (headroom - 1) * linear(map)), the map linearised with the
  sRGB curve, the headroom from the file's HDRGainMapHeadroom. The ISO 21496-1 item most of these files also carry is
  not read here.
The main image is read small (about the map's size) only to say how bright the boosted bean pixels are.

Writes <out>/gainmap_in_beans.csv, one row per photo, and prints the summary by camera.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv import canonical as cn  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
THRESHOLDS = (0.1, 0.25, 0.5, 1.0)


def srgb_to_linear(u: np.ndarray) -> np.ndarray:
    return np.where(u > 0.04045, ((u + 0.055) / 1.055) ** 2.4, u / 12.92)


def first(value: str | None, default: float) -> float:
    return float(value.split(",")[0]) if value else default


def jpeg_boost(d: bytes, name: str, detail: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(log2 boost, the main image's linear luminance) at the map's size, upright, and the two offsets."""
    at = d.find(b"hdrgm:GainMapM")
    soi = d.rfind(b"\xff\xd8\xff", 0, at)
    turn = cn.applied_orientation(d, name)
    with Image.open(io.BytesIO(d[soi:])) as im:
        rec = np.asarray(im.convert("L"), np.float32) / 255
    lo, hi = first(detail.get("GainMapMin"), 0.0), first(detail.get("GainMapMax"), 1.0)
    boost = lo + (hi - lo) * rec ** (1 / first(detail.get("Gamma"), 1.0))
    with Image.open(io.BytesIO(d)) as im:
        im.draft("RGB", (rec.shape[1], rec.shape[0]))
        main = np.asarray(im.convert("L").resize((rec.shape[1], rec.shape[0]), Image.BILINEAR), np.float32) / 255
    return (np.ascontiguousarray(cn.orient(boost, turn)), np.ascontiguousarray(cn.orient(srgb_to_linear(main), turn)),
            first(detail.get("OffsetSDR"), 1 / 64), first(detail.get("OffsetHDR"), 1 / 64))


def heic_boost(d: bytes, detail: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    import pillow_heif
    heif = pillow_heif.open_heif(io.BytesIO(d), convert_hdr_to_8bit=True)
    ids = next(v for k, v in heif.info["aux"].items() if k.endswith("hdrgainmap"))
    rec = np.asarray(heif.get_aux_image(ids[0]), np.float32) / 255
    boost = np.log2(1 + (float(detail["headroom"]) - 1) * srgb_to_linear(rec))
    main = Image.fromarray(np.asarray(heif)).convert("L").resize((rec.shape[1], rec.shape[0]), Image.BILINEAR)
    return boost, srgb_to_linear(np.asarray(main, np.float32) / 255), 0.0, 0.0


def stats(b: np.ndarray, prefix: str) -> dict:
    if b.size == 0:
        return {}
    q = np.percentile(b, [5, 50, 95, 99])
    out = {f"{prefix}_mean": b.mean(), f"{prefix}_p5": q[0], f"{prefix}_p50": q[1], f"{prefix}_p95": q[2], f"{prefix}_p99": q[3]}
    out.update({f"{prefix}_over_{t}": float((b > t).mean()) for t in THRESHOLDS})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", type=Path, default=REPO / "data" / "seg_masks_ml19")
    ap.add_argument("--out", type=Path, default=REPO / "tmp" / "ml29_scan")
    a = ap.parse_args()
    scanned = {"__".join(Path(r["file"].split("dataset/")[1]).with_suffix("").parts): r
               for r in csv.DictReader((a.out / "files.csv").open())}
    rows, skipped = [], collections.Counter()
    for mask_path in sorted(a.masks.rglob("*.png")):
        r = scanned.get(mask_path.stem)
        if r is None or not r["gain_map_detail"]:
            skipped["no gain map" if r else "no scanned photo"] += 1
            continue
        detail, path = json.loads(r["gain_map_detail"]), Path(r["file"])
        d = path.read_bytes()
        try:
            if r["type"] == "JPEG":
                boost, lum, k_sdr, k_hdr = jpeg_boost(d, path.name, detail)
            elif detail.get("headroom") is None:
                skipped["HEIC without a headroom"] += 1
                continue
            else:
                boost, lum, k_sdr, k_hdr = heic_boost(d, detail)
        except Exception as exc:  # noqa: BLE001 -- a measurement: one unreadable map is counted, not fatal
            skipped[f"{type(exc).__name__}"] += 1
            continue
        with Image.open(mask_path) as im:
            if abs(im.width / im.height - boost.shape[1] / boost.shape[0]) > 0.02:
                skipped["mask and map differ in shape"] += 1
                continue
            mask = np.asarray(im.convert("L").resize((boost.shape[1], boost.shape[0]), Image.BILINEAR)) > 127
        if not mask.any():
            skipped["empty mask"] += 1
            continue
        # The boost a pixel really gets, offsets included: what log2(HDR / SDR) is for its own light.
        real = np.log2(np.maximum((lum + k_sdr) * 2 ** boost - k_hdr, 1e-6) / np.maximum(lum, 1e-6)) if k_sdr else boost
        bean, rest = boost[mask], boost[~mask]
        strong = mask & (boost > 0.25)
        rows.append({"file": str(path), "type": r["type"], "make": r["make"], "map_size": f"{boost.shape[1]}x{boost.shape[0]}",
                     "map_max_stops": first(detail.get("GainMapMax"), 0) if r["type"] == "JPEG" else float(np.log2(float(detail["headroom"]))),
                     "bean_frac": float(mask.mean()), **stats(bean, "bean"), **stats(rest, "rest"),
                     "bean_real_mean": float(real[mask].mean()),
                     "bean_lum_all": float(lum[mask].mean()), "bean_lum_boosted": float(lum[strong].mean()) if strong.any() else ""})
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "gainmap_in_beans.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} photos measured; skipped {dict(skipped)}")
    med = lambda xs: float(np.median([x for x in xs if x != ""])) if any(x != "" for x in xs) else float("nan")  # noqa: E731
    groups = collections.defaultdict(list)
    for r in rows:
        groups[f"{r['type']} {r['make']}"].append(r)
    groups["ALL"] = rows
    print("\nBoost in stops at full HDR. Per photo, then the median over photos (and the worst photo).")
    print(f"{'camera':28s} {'n':>4s} {'map max':>8s} | {'beans mean':>10s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'worst mean':>10s} | {'rest mean':>9s} {'p95':>6s}")
    for name, g in groups.items():
        print(f"{name:28s} {len(g):4d} {med([r['map_max_stops'] for r in g]):8.2f} | {med([r['bean_mean'] for r in g]):10.3f} "
              f"{med([r['bean_p50'] for r in g]):6.3f} {med([r['bean_p95'] for r in g]):6.3f} {med([r['bean_p99'] for r in g]):6.3f} "
              f"{max(r['bean_mean'] for r in g):10.3f} | {med([r.get('rest_mean', '') for r in g]):9.3f} {med([r.get('rest_p95', '') for r in g]):6.3f}")
    print("\nShare of bean-region pixels boosted by more than a threshold: median photo (worst photo); and photos where over 5% of the region passes it.")
    for name, g in groups.items():
        parts = []
        for t in THRESHOLDS:
            xs = [r[f"bean_over_{t}"] for r in g]
            parts.append(f">{t}: {100 * med(xs):5.2f}% ({100 * max(xs):5.1f}%), {sum(x > 0.05 for x in xs):3d} photos")
        print(f"{name:28s} " + " | ".join(parts))
    print("\nWithin one bean region: the spread of the boost (p95 - p5), median photo and worst; mean linear light of all bean pixels and of those boosted over 0.25 stops.")
    for name, g in groups.items():
        spread = [r["bean_p95"] - r["bean_p5"] for r in g]
        print(f"{name:28s} spread {med(spread):.3f} (worst {max(spread):.3f}) | light: all {med([r['bean_lum_all'] for r in g]):.3f}, boosted {med([r['bean_lum_boosted'] for r in g]):.3f}"
              f" | boost with offsets, mean {med([r['bean_real_mean'] for r in g]):.3f}")


if __name__ == "__main__":
    main()
