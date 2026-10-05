"""Lossless removal of private metadata from dataset photos (ticket ML-3, D13).

    python -m coffeecv.strip_metadata DIR... --manifest labels/ml3/strip_manifest.csv
    python -m coffeecv.strip_metadata DIR... --dry-run [--report FILE]   # what would go; nothing is written

D13.1 as a deny list: GPS (EXIF, XMP and any GPS* tag), maker notes (Sony face info, Apple, Google HDR+
maker note), C2PA/JUMBF (its signing certificate carries a device-stable id), extended XMP, embedded
thumbnails and previews, device and lens serial numbers, owner/artist fields. Everything else stays,
Orientation and the ICC profile included: they are not private and they describe the photo.

exiftool edits metadata without re-encoding image data. Each photo is written to a new file beside it,
checked, and only then moved over the original (a new inode: a DVC cache file linked from the working
tree is never written through). The checks, all of which must pass:

  - pixels: the decoded image is equal before and after -- `dataset.load_rgb_image` (rawpy for a RAW
    file), and Pillow's own decode in the file's mode, so a PNG's alpha counts too; for HEIC every
    top-level image and every auxiliary image (Apple's gain map, depth); for RAW the sensor data as well;
  - every other tag survives byte for byte: binary tags (an MPF gain map, the ICC profile) are compared
    as bytes, and the kept fields D13 names -- Orientation, ICC profile, Make, Model, DateTimeOriginal,
    the exposure tags -- must be present after if they were before. load_rgb_image ignores Orientation
    and ICC, so the pixel check alone could not see them go. Only file-layout tags (offsets, lengths,
    sizes) may change;
  - audit: `exiftool -a -G0:1 -ee -u` of the new file finds no deny-listed tag.

The manifest gets one row per photo: path, sha256_before, sha256_after, pixel_sha256, removed (the
deny-listed tags that went). It is the re-key map for ticket ML-3 P3 and the record of D13. A photo with
nothing to remove gets a row with the same hash on both sides.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pillow_heif
import rawpy
from PIL import Image

from coffeecv.config import REPO_ROOT
from coffeecv.dataset import RAW_EXTENSIONS, load_rgb_image

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"} | RAW_EXTENSIONS
HEIF_EXTENSIONS = {".heic", ".heif"}

# What exiftool is asked to delete. The audit below, not this list, is the guarantee: a deny-listed tag
# these miss fails the photo.
DELETE_ARGS = [
    "-gps:all=",                          # the EXIF GPS IFD
    "-xmp:gps*=",                         # GPS in XMP (XMP-exif)
    "-makernotes:all=",                   # Sony (face info), Apple, Google, ...
    "-xmp-gcamera:hdrplusmakernote=",     # Google's HDR+ maker note, in extended XMP
    "-xmp-xmpnote:all=",                  # HasExtendedXMP: the pointer to the extended XMP segments
    "-jumbf:all=",                        # C2PA
    "-ifd1:all=",                         # a JPEG's thumbnail IFD
    "-previewimage=",
    "-*serialnumber=",                    # SerialNumber, LensSerialNumber, InternalSerialNumber, ...
    "-artist=", "-copyright=", "-ownername=", "-xpauthor=", "-xmp-dc:creator=", "-xmp-dc:rights=",
    "-iptc:by-line=", "-iptc:by-linetitle=", "-iptc:copyrightnotice=",
]

DENY_GROUPS = {"GPS", "JUMBF", "CBOR", "IFD1"}                 # family-1 groups; IFD1 is a JPEG's thumbnail
DENY_TAGS = {"Artist", "Copyright", "OwnerName", "CameraOwnerName", "XPAuthor", "Creator", "Rights",
             "By-line", "By-lineTitle", "CopyrightNotice", "HdrPlusMakernote", "HasExtendedXMP", "ThumbnailImage", "ThumbnailOffset",
             "ThumbnailLength", "PreviewImage", "PreviewImageStart", "PreviewImageLength"}
_DENY_TAG_RE = re.compile(r"^GPS|SerialNumber")

# D13.1's "keep" list: must be there after if it was there before, byte for byte.
KEPT_TAGS = {"Orientation", "ICC_Profile", "Make", "Model", "DateTimeOriginal", "CreateDate",
             "ExposureTime", "FNumber", "ISO", "ExposureProgram", "ExposureCompensation", "FocalLength",
             "Flash", "WhiteBalance", "MeteringMode", "OffsetTimeOriginal", "SubSecTimeOriginal"}
# Groups exiftool derives or describes the file with (either family); never compared.
VOLATILE_GROUPS = {"System", "ExifTool", "Composite"}
# Where things sit in the file: they move when a segment before them goes. A HEIC's MediaData is its
# mdat box, which holds the Exif item as well as the images; the images are compared decoded instead.
# exiftool also keeps EXIF and IPTC well-formed on its own: YCbCrPositioning goes with an IFD0 left with
# nothing else, and EnvelopeRecordVersion and the IPTC digests come with rewritten IPTC.
LAYOUT_TAGS = {"MPImageStart", "MPImageLength", "FileSize", "ExifByteOrder", "XMPToolkit",
               "CurrentIFDPosition", "MediaDataOffset", "MediaDataSize", "MediaData", "TileOffsets",
               "StripOffsets", "YCbCrPositioning", "EnvelopeRecordVersion", "CurrentIPTCDigest", "IPTCDigest"}

MANIFEST_FIELDS = ["path", "sha256_before", "sha256_after", "pixel_sha256", "removed"]


class StripError(Exception):
    """A photo failed a check; its original is untouched."""


def exiftool_version() -> str:
    return subprocess.run(["exiftool", "-ver"], capture_output=True, text=True, check=True).stdout.strip()


def read_tags(path: Path) -> dict[str, object]:
    """Every tag of `path` as {"Family0:Family1:Tag": value}, embedded documents and unknown tags included,
    binary values as "base64:..." so they compare as bytes, and the ICC profile's raw bytes as their own tag."""
    # -ICC_Profile adds the profile as one binary block (by default only its decoded fields are listed).
    proc = subprocess.run(["exiftool", "-j", "-b", "-a", "-G0:1", "-ee", "-u", "-q", "-q", "-all", "-ICC_Profile",
                           str(path)],
                          capture_output=True, text=True)
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        raise StripError(f"exiftool could not read {path.name}: {proc.stderr.strip()}")
    doc = json.loads(proc.stdout)[0]
    doc.pop("SourceFile", None)
    return doc


def _parts(key: str) -> tuple[str, str, str]:
    groups, _, tag = key.rpartition(":")
    fam0, _, fam1 = groups.partition(":")
    return fam0, fam1 or fam0, tag


def is_denied(key: str) -> bool:
    fam0, fam1, tag = _parts(key)
    if fam0 in VOLATILE_GROUPS - {"Composite"} or fam1 in VOLATILE_GROUPS - {"Composite"}:
        return False
    return (fam0 == "MakerNotes" or fam1 in DENY_GROUPS or tag in DENY_TAGS
            or bool(_DENY_TAG_RE.search(tag)))


def is_compared(key: str) -> bool:
    fam0, fam1, tag = _parts(key)
    return (fam0 not in VOLATILE_GROUPS and fam1 not in VOLATILE_GROUPS and tag not in LAYOUT_TAGS
            and not is_denied(key))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(h, arr: np.ndarray) -> None:
    h.update(f"{arr.dtype}{arr.shape}".encode())
    h.update(np.ascontiguousarray(arr).tobytes())


def pixel_digest(path: Path) -> str:
    """sha256 over every decoded image in the file: what the strip must leave alone."""
    h = hashlib.sha256()
    _digest(h, load_rgb_image(path))               # what training and serving decode
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            _digest(h, raw.raw_image_visible)
    elif ext in HEIF_EXTENSIONS:
        heif = pillow_heif.open_heif(str(path), convert_hdr_to_8bit=False)
        for img in heif:
            _digest(h, np.asarray(img))
            for depth in img.info.get("depth_images", []):
                _digest(h, np.asarray(depth))
            for ids in (img.info.get("aux") or {}).values():
                for aux_id in ids:
                    _digest(h, np.asarray(img.get_aux_image(aux_id)))
    else:
        with Image.open(path) as im:                # the file's own mode: a PNG's alpha counts
            _digest(h, np.asarray(im))
    return h.hexdigest()


def compare(before: dict, after: dict) -> tuple[list[str], list[str]]:
    """(denied tags that went, problems). A problem is a denied tag still there, or a compared tag that
    changed or went."""
    problems = [f"deny-listed tag survived: {k}" for k in after if is_denied(k)]
    for k, v in before.items():
        if not is_compared(k):
            continue
        if k not in after:
            problems.append(f"tag lost: {k}")
        elif after[k] != v:
            problems.append(f"tag changed: {k}")
    kept_before = {k for k in before if _parts(k)[2] in KEPT_TAGS and is_compared(k)}
    problems += [f"kept field lost: {k}" for k in sorted(kept_before - set(after))]
    problems += [f"tag added: {k}" for k in after if k not in before and is_compared(k)]
    # Composite tags are exiftool's own, derived from the real ones: they go with them, and are not listed.
    removed = sorted(k for k in before if is_denied(k) and k not in after and _parts(k)[0] != "Composite")
    return removed, problems


def strip_one(path: Path, dry_run: bool) -> dict:
    """Strip `path` (or, with dry_run, only check what stripping would do). Returns its manifest row
    plus "problems"; raises nothing for a failed check -- the row says so and the original is kept."""
    before = read_tags(path)
    sha_before = sha256_file(path)
    pix_before = pixel_digest(path)
    row = {"path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
           "sha256_before": sha_before, "sha256_after": sha_before, "pixel_sha256": pix_before,
           "removed": "", "problems": [], "notes": []}
    if path.suffix.lower() in HEIF_EXTENSIONS:
        # A HEIF thumbnail is an image item, not a tag: exiftool can neither list nor remove it.
        n = sum(len(img.info.get("thumbnails") or []) for img in pillow_heif.open_heif(str(path)))
        if n:
            row["notes"].append("HEIF thumbnail image item stays (exiftool cannot remove it)")
    if not any(is_denied(k) for k in before):
        return row
    # Beside the photo for a real strip (os.replace is atomic within one filesystem); a dry run writes
    # nothing under the photo's folder.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix,
                                    dir=None if dry_run else path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.unlink()                                 # exiftool -o refuses to overwrite
        proc = subprocess.run(["exiftool", "-q", "-q", *DELETE_ARGS, "-o", str(tmp), str(path)],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.exists():
            row["problems"].append(f"exiftool failed: {(proc.stderr or proc.stdout).strip()}")
            return row
        after = read_tags(tmp)
        removed, problems = compare(before, after)
        if pixel_digest(tmp) != pix_before:
            problems.append("decoded pixels differ")
        row["removed"] = " ".join(removed)
        row["problems"] = problems
        if problems:
            return row
        row["sha256_after"] = sha256_file(tmp)
        if not dry_run:
            shutil.copymode(path, tmp)
            os.replace(tmp, path)
        return row
    finally:
        tmp.unlink(missing_ok=True)


def photos_under(dirs: list[Path]) -> tuple[list[Path], list[Path]]:
    """(photos, other files) under `dirs` (a file stands for itself), sorted; symlinks followed (DVC's
    working tree may link)."""
    photos, other = [], []
    for d in dirs:
        for p in [Path(d)] if Path(d).is_file() else sorted(Path(d).rglob("*")):
            if p.is_dir() or p.name.startswith("."):
                continue
            (photos if p.suffix.lower() in PHOTO_EXTENSIONS else other).append(p)
    return photos, other


def read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith("#"))
    return {r["path"]: r for r in rows}


def append_manifest(path: Path, rows: list[dict], header: list[str]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        if new:
            f.writelines(f"# {line}\n" for line in header)
            csv.writer(f).writerow(MANIFEST_FIELDS)
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writerows(rows)


def report(rows: list[dict], other: list[Path], out) -> None:
    """What went (or would go), per top-level folder."""
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        parts = Path(r["path"]).parts
        by_folder["/".join(parts[:2]) if parts[0] in ("dataset", "dataset_new_ignored") else parts[0]].append(r)
    for folder, rs in sorted(by_folder.items()):
        n_strip = sum(1 for r in rs if r["removed"])
        n_bad = sum(1 for r in rs if r["problems"])
        print(f"\n{folder}: {len(rs)} photos, {n_strip} with private tags, {n_bad} failing a check", file=out)
        groups = Counter(t.rsplit(":", 1)[0] for r in rs for t in r["removed"].split())
        for g, n in sorted(groups.items()):
            print(f"  {g:<28} {n:>6} tag(s)", file=out)
        problems = Counter(p for r in rs for p in r["problems"])
        for p, n in problems.most_common():
            print(f"  FAIL x{n}: {p}", file=out)
        for note, n in Counter(x for r in rs for x in r.get("notes", [])).most_common():
            print(f"  NOTE x{n}: {note}", file=out)
    if other:
        print(f"\nnot photos, left alone: {len(other)} ({', '.join(sorted({p.suffix for p in other}))})", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="folders (searched recursively) or photos")
    ap.add_argument("--manifest", type=Path, help="CSV to append one row per photo to (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="strip copies and check them; change nothing")
    ap.add_argument("--report", type=Path, help="also write the per-folder report here")
    ap.add_argument("--header", action="append", default=[], help="extra '# ...' line for a new manifest")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.manifest:
        ap.error("--manifest is required unless --dry-run")

    photos, other = photos_under([d.resolve() if not d.is_absolute() else d for d in args.dirs])
    done = read_manifest(args.manifest) if args.manifest else {}
    version = exiftool_version()
    rows = []
    for i, p in enumerate(photos, 1):
        key = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
        if key in done and not args.dry_run:
            if sha256_file(p) != done[key]["sha256_after"]:
                print(f"  {key}: in the manifest, but its bytes are not the recorded sha256_after", file=sys.stderr)
                return 1
            continue
        row = strip_one(p, args.dry_run)
        rows.append(row)
        status = "FAIL " + "; ".join(row["problems"]) if row["problems"] else (
            f"{len(row['removed'].split())} tags" if row["removed"] else "clean")
        print(f"[{i}/{len(photos)}] {key}: {status}", flush=True)
    good = [r for r in rows if not r["problems"]]
    if args.manifest and not args.dry_run:
        header = [f"ML-3 D13 strip manifest: coffeecv/strip_metadata.py, exiftool {version}",
                  f"started {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                  "exiftool " + " ".join(DELETE_ARGS), *args.header]
        append_manifest(args.manifest, good, header)
    out_files = [sys.stdout] + ([open(args.report, "w")] if args.report else [])
    for out in out_files:
        print(f"{'DRY RUN: ' if args.dry_run else ''}exiftool {version}; {len(rows)} photos checked, "
              f"{sum(1 for r in good if r['removed'])} {'would be ' if args.dry_run else ''}stripped, "
              f"{len(rows) - len(good)} failing", file=out)
        report(rows, other, out)
    for out in out_files[1:]:
        out.close()
    return 1 if len(good) != len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
