"""What the dataset's originals declare about colour, bit depth, HDR and second images (ticket ML-29).

    python analysis/ml29/scan_dataset.py [--root dataset] [--out tmp/ml29_scan]

Reads every photo under --root (nothing is decoded, nothing is written beside --out) with two instruments:
- exiftool, one run over the tree, for what it names well: the ICC profile, the EXIF colour tags, the camera,
  MPF, the gain-map and depth XMP, a DNG's or CR3's raw fields;
- this file's own parsers, for what exiftool does not tie to an image: which HEIF item is the primary one and
  which colour boxes, sizes and auxiliary types belong to it and to each other item; a JPEG gain map's own
  header and its hdrgm parameters (they sit in the second image's XMP, which exiftool does not open); a PNG's
  colour chunks; a JPEG XL's header (coffeecv.jxl_header).

Writes <out>/files.csv, one row per photo (local scratch: paths, no pixels), and prints the summary that
analysis/ml29/README.md records. Counts only; no GPS, no free-text metadata.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv.jxl_header import read_header  # noqa: E402

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".hif", ".avif", ".jxl", ".gif", ".bmp",
             ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
PRIMARIES = {1: "BT.709/sRGB", 2: "unspecified", 9: "BT.2020", 11: "DCI-P3", 12: "Display P3"}
TRANSFERS = {1: "BT.709", 2: "unspecified", 8: "linear", 13: "sRGB", 14: "BT.2020-10", 16: "PQ", 18: "HLG"}
EXIF_TAGS = ["File:FileType", "IFD0:Make", "IFD0:Model", "File:ImageWidth", "File:ImageHeight",
             "ICC_Profile:ProfileDescription", "ICC-header:ColorSpaceData", "ExifIFD:ColorSpace",
             "InteropIFD:InteropIndex", "File:BitsPerSample", "File:EncodingProcess", "File:YCbCrSubSampling",
             "MPF0:NumberOfImages", "XMP-hdrgm:Version", "XMP-GContainer:ContainerDirectory",
             "XMP-HDRGainMap:HDRGainMapVersion", "XMP-HDRGainMap:HDRGainMapHeadroom", "XMP-GCamera:MotionPhoto",
             "XMP-OPMedia:IsHDRActive", "JSON:Hdr", "Canon:HDR", "XMP-apdi:AuxiliaryImageType",
             "SubIFD:BitsPerSample", "IFD0:BitsPerSample", "SubIFD:ProfileGainTableMap", "IFD0:ProfileName",
             "IFD0:ProfileToneCurve", "Canon:ColorSpace", "PNG:BitDepth", "PNG:ColorType"]


def exiftool(root: Path) -> dict[str, dict]:
    out = subprocess.run(["exiftool", "-r", "-json", "-G1", "-a", "-struct", *(f"-{t}" for t in EXIF_TAGS), str(root)],
                         capture_output=True, text=True, check=False).stdout
    return {r["SourceFile"]: r for r in json.loads(out or "[]")}


# ----- ISO BMFF (HEIC, HEIF, AVIF) -----

def _boxes(d: bytes, s: int, e: int):
    while s + 8 <= e:
        n, t = struct.unpack(">I4s", d[s:s + 8])
        h = 8
        if n == 1:
            n, h = struct.unpack(">Q", d[s + 8:s + 16])[0], 16
        if n == 0:
            n = e - s
        if n < h:
            return
        yield t, s + h, s + n
        s += n


def _find(d: bytes, s: int, e: int, kind: bytes):
    return next(((a, b) for t, a, b in _boxes(d, s, e) if t == kind), None)


def _colr(b: bytes) -> str:
    if b[:4] == b"nclx":
        pr, tf, _ = struct.unpack(">HHH", b[4:10])
        return f"nclx {PRIMARIES.get(pr, pr)}/{TRANSFERS.get(tf, tf)}"
    return "ICC"


def heif(d: bytes) -> dict:
    """The primary item and every other item: its type, role, size, bit depth and colour boxes."""
    meta = _find(d, 0, len(d), b"meta")
    if not meta:
        return {}
    ms, me = meta[0] + 4, meta[1]
    primary, types, roles, props, assoc = None, {}, {}, [], collections.defaultdict(list)
    for t, a, b in _boxes(d, ms, me):
        if t == b"pitm":
            primary = struct.unpack(">H" if d[a] == 0 else ">I", d[a + 4:a + (6 if d[a] == 0 else 8)])[0]
        elif t == b"iinf":
            for t2, a2, _ in _boxes(d, a + (6 if d[a] == 0 else 8), b):
                if t2 == b"infe":
                    v = d[a2]
                    item = struct.unpack(">H", d[a2 + 4:a2 + 6])[0] if v < 3 else struct.unpack(">I", d[a2 + 4:a2 + 8])[0]
                    types[item] = d[a2 + (8 if v < 3 else 10):a2 + (12 if v < 3 else 14)].decode("latin1")
        elif t == b"iref":
            wide = d[a] != 0
            for t2, a2, b2 in _boxes(d, a + 4, b):
                f = ">I" if wide else ">H"
                w = 4 if wide else 2
                src = struct.unpack(f, d[a2:a2 + w])[0]
                roles[src] = t2.decode("latin1")     # auxl, thmb, cdsc, dimg (a grid's or tone-mapped item's parts)
        elif t == b"iprp":
            for t2, a2, b2 in _boxes(d, a, b):
                if t2 == b"ipco":
                    props = [(t3, d[a3:b3]) for t3, a3, b3 in _boxes(d, a2, b2)]
                elif t2 == b"ipma":
                    v, flags = d[a2], d[a2 + 3]
                    p = a2 + 8
                    for _ in range(struct.unpack(">I", d[a2 + 4:a2 + 8])[0]):
                        item = struct.unpack(">H" if v < 1 else ">I", d[p:p + (2 if v < 1 else 4)])[0]
                        p += 2 if v < 1 else 4
                        n = d[p]
                        p += 1
                        for _ in range(n):
                            if flags & 1:
                                idx = struct.unpack(">H", d[p:p + 2])[0] & 0x7FFF
                                p += 2
                            else:
                                idx = d[p] & 0x7F
                                p += 1
                            assoc[item].append(idx)
    items = {}
    for item, kind in types.items():
        rec = {"type": kind, "role": roles.get(item, ""), "colr": [], "aux": "", "size": "", "bits": ""}
        for idx in assoc.get(item, []):
            if not 1 <= idx <= len(props):
                continue
            t, b = props[idx - 1]
            if t == b"colr":
                rec["colr"].append(_colr(b))
            elif t == b"auxC":
                rec["aux"] = b[4:].split(b"\0")[0].decode("latin1")
            elif t == b"ispe":
                rec["size"] = "%dx%d" % struct.unpack(">II", b[4:12])
            elif t == b"pixi":
                rec["bits"] = "/".join(str(x) for x in b[5:5 + b[4]])
        items[item] = rec
    return {"primary": items.get(primary, {}), "others": [r for i, r in items.items() if i != primary]}


# ----- JPEG gain map (Ultra HDR: MPF second image, hdrgm XMP) -----

HDRGM = re.compile(rb'hdrgm:(\w+)="([^"]*)"')
ISO21496 = b"urn:iso:std:iso:ts:21496:-1"


def jpeg_gainmap(d: bytes) -> dict:
    at = d.find(b"hdrgm:GainMapM")       # GainMapMin / GainMapMax: only in the gain map's own XMP
    if at < 0:
        return {"iso21496": ISO21496 in d} if ISO21496 in d else {}
    soi = d.rfind(b"\xff\xd8\xff", 0, at)
    out = {k.decode(): v.decode() for k, v in HDRGM.findall(d[at - 2000:at + 3000])}
    out["iso21496"] = ISO21496 in d
    p = soi + 2
    while p + 4 < len(d) and d[p] == 0xFF:
        m, n = d[p + 1], struct.unpack(">H", d[p + 2:p + 4])[0]
        if m in (0xC0, 0xC1, 0xC2):
            h, w, comps = struct.unpack(">HHB", d[p + 5:p + 10])
            out.update(size=f"{w}x{h}", components=comps, bytes=len(d) - soi)
            break
        if m == 0xDA:
            break
        p += 2 + n
    return out


# ----- PNG colour chunks -----

def png(d: bytes) -> dict:
    s, out = 8, {}
    while s + 8 <= len(d):
        n, t = struct.unpack(">I4s", d[s:s + 8])
        b = d[s + 8:s + 8 + n]
        if t == b"gAMA":
            out["gAMA"] = round(100000 / struct.unpack(">I", b)[0], 4)
        elif t == b"cHRM":
            out["cHRM"] = tuple(x / 100000 for x in struct.unpack(">8I", b))
        elif t == b"cICP":
            out["cICP"] = f"{PRIMARIES.get(b[0], b[0])}/{TRANSFERS.get(b[1], b[1])}"
        elif t in (b"iCCP", b"sRGB"):
            out[t.decode()] = True
        elif t == b"IDAT":
            break
        s += 12 + n
    return out


SRGB_CHRM = (0.3127, 0.329, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06)


def png_declares(c: dict) -> str:
    if c.get("iCCP"):
        return "ICC"
    if c.get("cICP"):
        return "cICP " + c["cICP"]
    if c.get("sRGB"):
        return "sRGB chunk"
    parts = []
    if "gAMA" in c:
        parts.append("gamma 2.2 (sRGB-like)" if abs(c["gAMA"] - 2.2) < 0.01 else f"gamma {c['gAMA']}")
    if "cHRM" in c:
        same = all(abs(a - b) < 0.0002 for a, b in zip(c["cHRM"], SRGB_CHRM))
        parts.append("sRGB chromaticities" if same else f"chromaticities {c['cHRM']}")
    return " + ".join(parts) or "nothing"


# ----- one row per photo -----

def row(path: Path, x: dict) -> dict:
    d = path.read_bytes()
    kind = x.get("File:FileType", path.suffix.lstrip(".").upper())
    icc = x.get("ICC_Profile:ProfileDescription")
    icc_rgb = x.get("ICC-header:ColorSpaceData", "").strip() == "RGB"
    r = {"file": str(path), "type": kind, "make": str(x.get("IFD0:Make", "")).strip(), "model": str(x.get("IFD0:Model", "")).strip(),
         "megapixels": round(x.get("File:ImageWidth", 0) * x.get("File:ImageHeight", 0) / 1e6, 1) if "File:ImageWidth" in x else "",
         "icc": icc or "", "exif_colorspace": x.get("ExifIFD:ColorSpace", ""), "interop": str(x.get("InteropIFD:InteropIndex", ""))[:3],
         "bits": "", "declared_without_icc": "", "read_today_as": "", "gain_map": "", "gain_map_detail": "",
         "second_images": "", "vendor_hdr_flag": "", "motion_photo": bool(x.get("XMP-GCamera:MotionPhoto")), "notes": ""}
    second = []
    if kind == "JPEG":
        r["bits"] = f"{x.get('File:BitsPerSample', '?')}-bit {x.get('File:YCbCrSubSampling', '').split(' (')[0]}"
        g = jpeg_gainmap(d)
        if g.get("size"):
            r["gain_map"] = "Ultra HDR (hdrgm)" + (" + ISO 21496-1" if g.get("iso21496") else "")
            r["gain_map_detail"] = json.dumps(g, sort_keys=True)
        elif g.get("iso21496"):
            r["gain_map"] = "ISO 21496-1"
        for item in x.get("XMP-GContainer:ContainerDirectory") or []:
            sem = (item.get("Item") or {}).get("Semantic", "")
            if sem not in ("Primary", "GainMap"):
                second.append(f"container:{sem}")
        second += [f"apple:{a}" for a in ([x["XMP-apdi:AuxiliaryImageType"]] if isinstance(x.get("XMP-apdi:AuxiliaryImageType"), str) else x.get("XMP-apdi:AuxiliaryImageType") or [])]
        if not icc:
            if r["exif_colorspace"] == "Uncalibrated":
                r["declared_without_icc"] = "EXIF Uncalibrated" + (" + R03 (Adobe RGB)" if r["interop"] == "R03" else "")
            elif r["interop"] == "R03":
                r["declared_without_icc"] = "R03 (Adobe RGB)"
    elif kind in ("HEIC", "HEIF", "AVIF"):
        h = heif(d)
        p = h.get("primary", {})
        r["bits"] = f"{p.get('bits', '?')}-bit {p.get('type', '')}"
        nclx = [c for c in p.get("colr", []) if c.startswith("nclx") and "unspecified/unspecified" not in c]
        r["notes"] = "primary colr: " + ", ".join(p.get("colr", [])) if p else "no meta box"
        icc = icc or ("(ICC on primary item)" if "ICC" in p.get("colr", []) else "")
        icc_rgb = icc_rgb or bool(icc)
        r["icc"] = icc
        if nclx and "ICC" not in p.get("colr", []):
            r["declared_without_icc"] = nclx[0]
        for o in h.get("others", []):
            if o["aux"].endswith("hdrgainmap"):
                r["gain_map"] = "Apple (auxiliary image)"
                r["gain_map_detail"] = json.dumps({"size": o["size"], "bits": o["bits"], "colr": o["colr"], "primary_size": p.get("size"),
                                                   "headroom": x.get("XMP-HDRGainMap:HDRGainMapHeadroom"),
                                                   "version": x.get("XMP-HDRGainMap:HDRGainMapVersion")}, sort_keys=True)
            elif o["type"] == "tmap":
                r["gain_map"] = (r["gain_map"] + " + " if r["gain_map"] else "") + "ISO 21496-1 (tmap item)"
            elif o["aux"]:
                second.append(o["aux"].split(":")[-1] + ("" if o["aux"] != "urn:mpeg:hevc:2015:auxid:2" else " (depth/disparity)"))
            elif o["role"] == "thmb":
                second.append("thumbnail")
    elif kind == "PNG":
        c = png(d)
        r["bits"] = f"{x.get('PNG:BitDepth', '?')}-bit {x.get('PNG:ColorType', '')}"
        dec = png_declares(c)
        r["notes"] = "chunks: " + dec
        if dec not in ("ICC", "nothing"):
            r["declared_without_icc"] = dec
    elif kind.startswith("JXL"):
        h = read_header(d)
        c = h.colour
        r["bits"] = f"{h.bits_per_sample}-bit{' float' if h.float_sample else ''}"
        r["notes"] = f"header: {'ICC' if c.want_icc else f'{c.primaries}/{c.transfer or c.gamma}'}; boxes {h.boxes or 'none'}"
        r["read_today_as"] = "embedded ICC: refused" if c.want_icc else f"enumeration {c.primaries}/{c.transfer}"
        if "jhgm" in h.boxes:
            r["gain_map"] = "JPEG XL (jhgm box)"
    elif kind in ("DNG", "CR3", "CR2", "NEF", "ARW", "RAF", "ORF", "RW2", "PEF", "SRW"):
        # A DNG names its raw depth in the SubIFD; a CR3's IFD0 describes its preview, so its depth is left unsaid.
        r["bits"] = f"raw {x.get('SubIFD:BitsPerSample', x.get('IFD0:BitsPerSample', '?'))}" if kind == "DNG" else "raw (depth not read)"
        r["read_today_as"] = "developed by rawpy into linear Rec.2020"
        if x.get("SubIFD:ProfileGainTableMap"):
            second.append("DNG ProfileGainTableMap (local tone map)")
        if x.get("IFD0:ProfileToneCurve"):
            second.append("DNG ProfileToneCurve")
        r["notes"] = str(x.get("IFD0:ProfileName", "") or x.get("Canon:ColorSpace", ""))
    else:
        r["bits"] = "8-bit palette" if kind == "GIF" else ""
    if not r["read_today_as"]:
        r["read_today_as"] = f"ICC: {icc}" if icc and icc_rgb else ("sRGB (ICC not RGB)" if icc else "sRGB (no ICC)")
    flags = [f"OPMedia IsHDRActive={x['XMP-OPMedia:IsHDRActive']}" if "XMP-OPMedia:IsHDRActive" in x else "",
             f"Xiaomi Hdr={x['JSON:Hdr']}" if "JSON:Hdr" in x else "", f"Canon HDR={x['Canon:HDR']}" if "Canon:HDR" in x else ""]
    r["vendor_hdr_flag"] = "; ".join(f for f in flags if f)
    r["second_images"] = "; ".join(sorted(set(second)))
    return r


def table(title: str, rows: list[dict], *keys: str, top: int = 40) -> None:
    c = collections.Counter(tuple(str(r[k]) for k in keys) for r in rows)
    print(f"\n## {title}  ({' | '.join(keys)})")
    for k, n in c.most_common(top):
        print(f"{n:6d}  " + " | ".join(k))
    if len(c) > top:
        print(f"        ... {len(c) - top} more")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "dataset")
    ap.add_argument("--out", type=Path, default=REPO / "tmp" / "ml29_scan")
    a = ap.parse_args()
    meta = exiftool(a.root)
    paths = sorted(p for p in a.root.rglob("*") if p.is_file() and p.suffix.lower() in PHOTO_EXT)
    rows = [row(p, meta.get(str(p), {})) for p in paths]
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "files.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} photos under {a.root}; rows in {a.out / 'files.csv'}")
    table("Formats and bit depth", rows, "type", "bits")
    table("What canonicalization takes each photo's colour from today", rows, "type", "read_today_as")
    table("Colour declared some other way, with no ICC profile (read as sRGB today)", rows, "type", "declared_without_icc", "exif_colorspace", "interop")
    table("Gain maps by camera", rows, "gain_map", "type", "make")
    table("Other second images", rows, "type", "second_images")
    table("Vendor HDR flags", rows, "make", "vendor_hdr_flag")
    table("Motion photos (a video after the image)", rows, "type", "motion_photo")
    gm = [json.loads(r["gain_map_detail"]) | {"type": r["type"], "make": r["make"]} for r in rows if r["gain_map_detail"]]
    print("\n## Gain map parameters")
    for kind in ("JPEG", "HEIC"):
        g = [x for x in gm if x["type"] == kind]
        if not g:
            continue
        if kind == "JPEG":
            mx = sorted(float(x.get("GainMapMax", "nan").split(",")[0]) for x in g if x.get("GainMapMax"))
            cap = sorted(float(x["HDRCapacityMax"]) for x in g if x.get("HDRCapacityMax"))
            print(f"JPEG: {len(g)} maps; GainMapMax (log2 stops) min {mx[0]:.2f} median {mx[len(mx) // 2]:.2f} max {mx[-1]:.2f}; "
                  f"HDRCapacityMax median {cap[len(cap) // 2]:.2f}; components {collections.Counter(x['components'] for x in g)}; "
                  f"base rendition {collections.Counter(x.get('BaseRenditionIsHDR', 'absent') for x in g)}; "
                  f"ISO 21496-1 too: {sum(x['iso21496'] for x in g)}")
            print("      sizes: " + ", ".join(f"{k} x{n}" for k, n in collections.Counter(x["size"] for x in g).most_common(8)))
        else:
            hr = sorted(float(x["headroom"]) for x in g if x.get("headroom") is not None)
            print(f"HEIC: {len(g)} maps; headroom (linear) min {hr[0]:.2f} median {hr[len(hr) // 2]:.2f} max {hr[-1]:.2f} ({len(hr)} with the tag); "
                  f"versions {collections.Counter(x['version'] for x in g)}; bits {collections.Counter(x['bits'] for x in g)}; "
                  f"colour boxes {collections.Counter(', '.join(x['colr']) for x in g)}")
            print("      sizes (map of primary): " + ", ".join(f"{k} x{n}" for k, n in collections.Counter(f"{x['size']} of {x['primary_size']}" for x in g).most_common(8)))


if __name__ == "__main__":
    main()
