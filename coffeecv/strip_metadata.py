"""Lossless removal of private metadata from dataset photos (ticket ML-3, D13).

    python -m coffeecv.strip_metadata DIR... --manifest labels/ml3/strip_manifest.csv [--jobs N] [--max-memory GB]
    python -m coffeecv.strip_metadata DIR... --dry-run [--report FILE]   # what would go; nothing is written

D13.1 as a deny list: GPS (EXIF, XMP and any GPS* tag), maker notes (Sony face info, Apple, Google HDR+
maker note), C2PA/JUMBF (its signing certificate carries a device-stable id), extended XMP, embedded
thumbnails and previews, device and lens serial numbers, owner/artist fields, and a Samsung JPEG's SEF
trailer (MCC_Data, the network's country; Image_UTC_Data), cut from the file's end as exiftool cannot
delete it (samsung_trailer_start, 2026-10-23). Everything else stays,
Orientation and the ICC profile included: they are not private and they describe the photo. A RAW file's
preview and a HEIF's thumbnail item are image data exiftool cannot remove; the owner keeps them (2026-10-05).

exiftool edits metadata without re-encoding image data. Each photo is written to a new file beside it,
checked, and only then moved over the original (a new inode: a DVC cache file linked from the working
tree is never written through). The checks, all of which must pass:

  - pixels: the decoded image is equal before and after -- `photo_io.load_rgb_image` (rawpy for a RAW
    file; up to photo_io's DATASET_MAX_PIXELS, 537 MP, as training decodes), and Pillow's own decode in the
    file's mode, so a PNG's alpha counts too; for HEIC every top-level image and every auxiliary image
    (Apple's gain map, depth), and an auxiliary image pillow-heif cannot decode (10-bit, ML-6 D10) by its
    stored bytes: coded data, properties and tiles; for RAW the sensor data as well;
  - every other tag survives byte for byte: binary tags (an MPF gain map, the ICC profile) are compared
    as bytes, and the kept fields D13 names -- Orientation, ICC profile, Make, Model, DateTimeOriginal,
    the exposure tags -- must be present after if they were before. load_rgb_image ignores Orientation
    and ICC, so the pixel check alone could not see them go. Only file-layout tags (offsets, lengths,
    sizes) may change;
  - audit: `exiftool -a -G0:1 -ee -u` of the new file finds no deny-listed tag.

The manifest gets one row per photo: path, sha256_before, sha256_after, pixel_sha256, removed (the
deny-listed tags that went). It is the re-key map for ticket ML-3 P3 and the record of D13. A photo with
nothing to remove gets a row with the same hash on both sides.

A run (ticket ML-14):

  - the manifest first: a photo in it whose bytes are not its recorded sha256_after exits 1 before any photo
    is touched; the photos in it with their recorded bytes are left alone (D6);
  - up to --jobs photos at once, each in a spawned process of its own; by default min(CPU cores, 10, GB of memory
    and swap free at the start), about 1 GB a worker (D15). --jobs 1, or one photo to do, runs in this process;
  - a photo whose decode would need more memory than the budget (estimate_decode_bytes, from its header) fails
    before it is decoded, "too big: about N GB to decode, budget M GB", its original untouched; the budget is
    --max-memory GB, by default the memory available at the start less 1 GB (D13);
  - an exception for one photo is a failing row, "error: <type>: <message>", and the run goes on (D5). A
    failing row counts in the report and the exit code (1), never in the manifest;
  - each good row is appended to the manifest, flushed, as soon as its photo and every photo before it are
    done: the progress lines, the rows and the report are in photo order at any --jobs (D3, D9). A new
    manifest gets its header with its first row, with the run's start time (D10);
  - a Ctrl-C cancels the photos not yet handed to the workers, waits for the others and writes every row done,
    then the report, and exits 130; a rerun does the rest (D4). A second Ctrl-C stops at once: the photos then
    in progress keep their originals (one its worker replaced in the instant before is left without its row),
    each with a temporary copy (.<name>.*) beside it: a rerun skips it, and it is deleted by hand before the
    folder is added to DVC. At --jobs 1, likewise, a Ctrl-C in the instant between a photo's replacement and
    its row loses that row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pillow_heif
import rawpy
from PIL import Image

from coffeecv.config import REPO_ROOT
from coffeecv.photo_io import DATASET_MAX_PIXELS, RAW_EXTENSIONS, load_rgb_image
from coffeecv.repo_files import sha256_file

# The dataset's limit, not Pillow's: the 200 MP photos must decode. Set here, not by the caller, so a worker that
# imports this module gets it too; photo_io, not coffeecv.dataset, so no process here imports torch (ML-14 D8).
Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS

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

# Deny-listed, but kept by the owner's decision of 2026-10-05: a RAW file's embedded preview is image data
# exiftool cannot remove without rewriting the file's structure (HEIF thumbnails, likewise kept, are image
# items exiftool does not even list). Noted in the report, not failed.
ACCEPTED_IN_RAW = {"PreviewImage", "PreviewImageStart", "PreviewImageLength"}

# D13.1's "keep" list: must be there after if it was there before, byte for byte.
KEPT_TAGS = {"Orientation", "ICC_Profile", "Make", "Model", "DateTimeOriginal", "CreateDate",
             "ExposureTime", "FNumber", "ISO", "ExposureProgram", "ExposureCompensation", "FocalLength",
             "Flash", "WhiteBalance", "MeteringMode", "OffsetTimeOriginal", "SubSecTimeOriginal"}
# Groups exiftool derives or describes the file with (either family); never compared.
VOLATILE_GROUPS = {"System", "ExifTool", "Composite"}
# Where things sit in the file: they move when a segment before them goes. A HEIC's MediaData is its
# mdat box, which holds the Exif item as well as the images; the images are compared decoded instead (or by
# their stored bytes, an auxiliary image pillow-heif cannot decode).
# exiftool also keeps EXIF and IPTC well-formed on its own: YCbCrPositioning goes with an IFD0 left with
# nothing else, and EnvelopeRecordVersion and the IPTC digests come with rewritten IPTC.
LAYOUT_TAGS = {"MPImageStart", "MPImageLength", "FileSize", "ExifByteOrder", "XMPToolkit",
               "CurrentIFDPosition", "MediaDataOffset", "MediaDataSize", "MediaData", "TileOffsets",
               "StripOffsets", "YCbCrPositioning", "EnvelopeRecordVersion", "CurrentIPTCDigest", "IPTCDigest"}

MANIFEST_FIELDS = ["path", "sha256_before", "sha256_after", "pixel_sha256", "removed"]

GB = 10**9
# pixel_digest's peak memory above the imports, per pixel, measured 2026-10-30 (ticket ML-14 P3; each photo in its
# own process, 12-200 MP, the same at every size): a JPEG 10.0 (RGB or grey), an RGBA PNG 12.0, a HEIC 11.0 (per
# pixel of its top-level images, Apple's auxiliary images included), a DNG 13.0 (per pixel of rawpy's output size).
# Rounded up. The stripped copy is decoded after the photo, never beside it.
DECODE_BYTES_PER_PIXEL = {".jpg": 11, ".jpeg": 11, ".png": 13, ".heic": 12, ".heif": 12,
                          **dict.fromkeys(RAW_EXTENSIONS, 14)}
MAX_DEFAULT_JOBS = 10                       # the owner's, 2026-10-30 (ML-14 D15)


class StripError(Exception):
    """A photo failed a check; its original is untouched."""


def exiftool(*args: str) -> subprocess.CompletedProcess:
    """exiftool, in a process group of its own: a Ctrl-C at the terminal reaches the run, which decides what stops
    (ML-14 D4), and not exiftool, whose own SIGINT handler would fail a photo the run is waiting to finish."""
    return subprocess.run(["exiftool", *args], capture_output=True, text=True, process_group=0)


def exiftool_version() -> str:
    proc = exiftool("-ver")
    proc.check_returncode()
    return proc.stdout.strip()


def read_tags(path: Path) -> dict[str, object]:
    """Every tag of `path` as {"Family0:Family1:Tag": value}, embedded documents and unknown tags included,
    binary values as "base64:..." so they compare as bytes, and the ICC profile's raw bytes as their own tag."""
    # -ICC_Profile adds the profile as one binary block (by default only its decoded fields are listed).
    proc = exiftool("-j", "-b", "-a", "-G0:1", "-ee", "-u", "-q", "-q", "-all", "-ICC_Profile", str(path))
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


def _digest(h, arr: np.ndarray) -> None:
    h.update(f"{arr.dtype}{arr.shape}".encode())
    h.update(np.ascontiguousarray(arr).tobytes())


def _boxes(b: bytes, start: int, end: int):
    """(type, payload start, end) of each ISO-BMFF box in b[start:end]."""
    while start < end:
        size, typ = struct.unpack_from(">I4s", b, start)
        head = 8
        if size == 1:
            size, head = struct.unpack_from(">Q", b, start + 8)[0], 16
        elif size == 0:
            size = end - start
        if size < head or start + size > end:
            raise StripError(f"malformed {typ!r} box at byte {start}")
        yield typ.decode("latin-1"), start + head, start + size
        start += size


def _uint(b: bytes, pos: int, n: int) -> tuple[int, int]:
    return int.from_bytes(b[pos:pos + n], "big"), pos + n


def _heif_meta(b: bytes) -> dict:
    """A HEIF file's item table, from its meta box: primary item, iloc extents, idat, the ipco properties
    and their ipma associations, and the dimg references (a grid to its tiles)."""
    meta = next(((s, e) for t, s, e in _boxes(b, 0, len(b)) if t == "meta"), None)
    if meta is None:
        raise StripError("no meta box")
    sub = list(_boxes(b, meta[0] + 4, meta[1]))                 # meta is a full box: version and flags first
    box = {t: (s, e) for t, s, e in sub}
    out = {"primary": None, "extents": {}, "idat": b"", "props": [], "assoc": {}, "dimg": {}}
    if "pitm" in box:
        s = box["pitm"][0]
        out["primary"] = _uint(b, s + 4, 2 if b[s] == 0 else 4)[0]
    if "idat" in box:
        out["idat"] = b[box["idat"][0]:box["idat"][1]]
    s = box["iloc"][0]
    version, p = b[s], s + 4
    off_size, len_size, base_size = b[p] >> 4, b[p] & 15, b[p + 1] >> 4
    index_size = b[p + 1] & 15 if version in (1, 2) else 0
    id_size = 2 if version < 2 else 4
    count, p = _uint(b, p + 2, id_size)
    for _ in range(count):
        iid, p = _uint(b, p, id_size)
        method = 0
        if version in (1, 2):
            method, p = b[p + 1] & 15, p + 2
        ref, p = _uint(b, p, 2)
        base, p = _uint(b, p, base_size)
        n, p = _uint(b, p, 2)
        extents = []
        for _ in range(n):
            p += index_size
            off, p = _uint(b, p, off_size)
            length, p = _uint(b, p, len_size)
            extents.append((base + off, length))
        out["extents"][iid] = (method, ref, extents)
    if "iprp" in box:
        for t, s, e in _boxes(b, *box["iprp"]):
            if t == "ipco":
                out["props"] += [t2.encode() + b[s2:e2] for t2, s2, e2 in _boxes(b, s, e)]
            elif t == "ipma":
                version, flags, p = b[s], _uint(b, s + 1, 3)[0], s + 4
                n, p = _uint(b, p, 4)
                for _ in range(n):
                    iid, p = _uint(b, p, 2 if version < 1 else 4)
                    k, p = b[p], p + 1
                    for _ in range(k):
                        v, p = _uint(b, p, 2 if flags & 1 else 1)
                        out["assoc"].setdefault(iid, []).append(v & (0x7FFF if flags & 1 else 0x7F))
    if "iref" in box:
        s, e = box["iref"]
        id_size = 2 if b[s] == 0 else 4
        for t, s2, _ in _boxes(b, s + 4, e):
            src, p = _uint(b, s2, id_size)
            n, p = _uint(b, p, 2)
            if t == "dimg":
                out["dimg"].setdefault(src, []).extend(_uint(b, p + i * id_size, id_size)[0] for i in range(n))
    return out


def heif_stored_digest(path: Path, item_id: int) -> str:
    """sha256 of a HEIF image item as stored: its coded data (in the file, or in the meta box's idat), the
    properties ipma associates with it (its decoder configuration among them), and the same for every item
    it is derived from (dimg: a grid's tiles), recursively. For an auxiliary image pillow-heif cannot
    decode (Apple's 10-bit ones, ticket ML-6 D10): exiftool never re-encodes image data, so equal stored
    bytes are an equal image."""
    b = path.read_bytes()
    meta = _heif_meta(b)
    h = hashlib.sha256()

    def add(iid: int, chain: tuple) -> None:
        if iid in chain:
            raise StripError(f"item {iid} is derived from itself")
        method, ref, extents = meta["extents"].get(iid, (0, 0, []))
        if ref or method not in (0, 1):
            raise StripError(f"item {iid}: data outside the file or construction method {method}")
        src = b if method == 0 else meta["idat"]
        h.update(f"item {iid}".encode())
        for off, length in extents:
            h.update(src[off:off + length] if length else src[off:])
        for k in meta["assoc"].get(iid, []):
            if k:
                h.update(meta["props"][k - 1])
        for tile in meta["dimg"].get(iid, []):
            add(tile, chain + (iid,))

    add(item_id, ())
    return h.hexdigest()


def pixel_digest(path: Path) -> str:
    """sha256 over every decoded image in the file: what the strip must leave alone. A HEIF auxiliary image
    pillow-heif cannot decode counts by its stored bytes instead (heif_stored_digest)."""
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
                    try:
                        aux = np.asarray(img.get_aux_image(aux_id))
                    except NotImplementedError:      # pillow-heif decodes 8-bit auxiliary images only
                        h.update(f"stored {heif_stored_digest(path, aux_id)}".encode())
                        continue
                    _digest(h, aux)
    else:
        with Image.open(path) as im:                # the file's own mode: a PNG's alpha counts
            _digest(h, np.asarray(im))
    return h.hexdigest()


def accepted(key: str, ext: str) -> bool:
    return ext.lower() in RAW_EXTENSIONS and _parts(key)[2] in ACCEPTED_IN_RAW


def compare(before: dict, after: dict, ext: str = "") -> tuple[list[str], list[str]]:
    """(denied tags that went, problems). A problem is a denied tag still there (unless accepted for this
    file type), or a compared tag that changed or went."""
    problems = [f"deny-listed tag survived: {k}" for k in after if is_denied(k) and not accepted(k, ext)]
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


def samsung_trailer_start(data: bytes) -> int | None:
    """Where a Samsung SEF trailer begins in `data`, or None if it has none. Samsung appends it after a JPEG's
    last image (the gain map's, when there is one); exiftool reads it (MakerNotes:Samsung) but cannot delete
    it, and its -trailer:all= cuts the gain map too. Layout: the data blocks, then b"SEFH", a version, a count,
    `count` entries (2-byte pad, 2-byte type, the block's offset back from SEFH, its length), the directory's
    length (uint32 LE) and b"SEFT". The trailer begins at its earliest block. A malformed one raises
    ValueError: the photo fails rather than keep half a trailer."""
    if data[-4:] != b"SEFT":
        return None
    n = int.from_bytes(data[-8:-4], "little")
    h = len(data) - 8 - n
    if n < 12 or h < 0 or data[h:h + 4] != b"SEFH":
        raise ValueError("SEFT without its SEFH directory")
    count = int.from_bytes(data[h + 8:h + 12], "little")
    if n != 12 + 12 * count:
        raise ValueError(f"SEFH directory of {n} bytes for {count} entries")
    starts = []
    for i in range(count):
        _, _, off, size = struct.unpack_from("<HHII", data, h + 12 + 12 * i)
        if not 0 < off <= h or size > off:
            raise ValueError(f"SEF entry {i} outside its trailer")
        starts.append(h - off)
    return min(starts, default=h)


def manifest_key(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def new_row(path: Path) -> dict:
    """A photo's row before anything is known: its manifest fields empty, no problems, no notes."""
    return {"path": manifest_key(path), "sha256_before": "", "sha256_after": "", "pixel_sha256": "",
            "removed": "", "problems": [], "notes": []}


def failing_row(path: Path, problem: str, row: dict | None = None) -> dict:
    """`row` (by default a new one for `path`) with `problem` as its only problem."""
    row = new_row(path) if row is None else row
    row["problems"] = [problem]
    return row


def strip_one(path: Path, dry_run: bool, row: dict | None = None) -> dict:
    """Strip `path` (or, with dry_run, only check what stripping would do). Returns its manifest row
    plus "problems"; raises nothing for a failed check -- the row says so and the original is kept.
    `row` (new_row's), if given, is filled in place, so a caller that catches an exception keeps what was
    found before it (run_one)."""
    row = new_row(path) if row is None else row
    before = read_tags(path)
    sha_before = sha256_file(path)
    pix_before = pixel_digest(path)
    row.update(sha256_before=sha_before, sha256_after=sha_before, pixel_sha256=pix_before)
    if path.suffix.lower() in HEIF_EXTENSIONS:
        # A HEIF thumbnail is an image item, not a tag: exiftool can neither list nor remove it.
        n = sum(len(img.info.get("thumbnails") or []) for img in pillow_heif.open_heif(str(path)))
        if n:
            row["notes"].append("HEIF thumbnail image item stays (exiftool cannot remove it)")
    if any(accepted(k, path.suffix) for k in before):
        row["notes"].append("RAW preview image stays (owner, 2026-10-05)")
    if not any(is_denied(k) and not accepted(k, path.suffix) for k in before):
        return row
    # Beside the photo for a real strip (os.replace is atomic within one filesystem); a dry run writes
    # nothing under the photo's folder.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix,
                                    dir=None if dry_run else path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.unlink()                                 # exiftool -o refuses to overwrite
        proc = exiftool("-q", "-q", *DELETE_ARGS, "-o", str(tmp), str(path))
        if proc.returncode != 0 or not tmp.exists():
            row["problems"].append(f"exiftool failed: {(proc.stderr or proc.stdout).strip()}")
            return row
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            data = tmp.read_bytes()
            try:
                cut = samsung_trailer_start(data)
            except ValueError as e:
                row["problems"].append(f"Samsung trailer: {e}")
                return row
            if cut is not None:
                tmp.write_bytes(data[:cut])
        after = read_tags(tmp)
        removed, problems = compare(before, after, path.suffix)
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


def estimate_decode_bytes(path: Path) -> int:
    """About the memory pixel_digest needs for `path`, from its dimensions in the file's header: nothing is
    decoded. Raises what opening the file raises."""
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            pixels = raw.sizes.width * raw.sizes.height
    elif ext in HEIF_EXTENSIONS:
        pixels = sum(img.size[0] * img.size[1] for img in pillow_heif.open_heif(str(path)))
    else:
        with Image.open(path) as im:
            pixels = im.size[0] * im.size[1]
    return pixels * DECODE_BYTES_PER_PIXEL[ext]


def _meminfo(field: str) -> int | None:
    """A /proc/meminfo field in bytes, or None where there is none."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(f"{field}:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def available_memory() -> int:
    """Bytes of memory available now: /proc/meminfo's MemAvailable, else the free physical pages."""
    available = _meminfo("MemAvailable")
    return available if available is not None else os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def free_swap() -> int:
    """Bytes of swap free now (/proc/meminfo's SwapFree), 0 where unknown."""
    return _meminfo("SwapFree") or 0


def too_big(path: Path, budget: int) -> dict | None:
    """A failing row for `path` if decoding it would need more than `budget` bytes, else None (ML-14 D13): it fails
    rather than run the machine out of memory, and not as a skip, which would let it keep its private tags and the
    run pass. A header that cannot be read is left to run_one, which meets the same error."""
    try:
        need = estimate_decode_bytes(path)
    except Exception:
        return None
    if need <= budget:
        return None
    return failing_row(path, f"too big: about {need / GB:.3g} GB to decode, budget {budget / GB:.3g} GB (--max-memory)")


def run_one(path: Path, dry_run: bool) -> dict:
    """strip_one, with an exception made a failing row for its photo (ML-14 D5): the run goes on, and the
    photo, whose original strip_one replaces only after every check, keeps its bytes. A KeyboardInterrupt is
    not an Exception and still stops the run."""
    row = new_row(path)
    try:
        return strip_one(path, dry_run, row)
    except Exception as e:
        return failing_row(path, f"error: {type(e).__name__}: {e}", row)


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


class ManifestWriter:
    """Appends rows to a manifest one at a time, each flushed as it is written (ML-14 D10, no fsync). A new
    manifest gets its header lines and columns just before its first row, so a run with no row writes no file."""

    def __init__(self, path: Path, header: list[str]):
        self.path, self.header, self._f, self._w = path, header, None, None

    def write(self, row: dict) -> None:
        if self._f is None:
            new = not self.path.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "a", newline="")
            if new:
                self._f.writelines(f"# {line}\n" for line in self.header)
                csv.writer(self._f).writerow(MANIFEST_FIELDS)
            self._w = csv.DictWriter(self._f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()

    def __enter__(self) -> ManifestWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def append_manifest(path: Path, rows: list[dict], header: list[str]) -> None:
    with ManifestWriter(path, header) as w:
        for r in rows:
            w.write(r)


def resume_check(photos: list[Path], done: dict[str, dict]) -> tuple[list[Path], list[str]]:
    """(the photos still to do, the manifest keys whose photo's bytes are not their recorded sha256_after). A
    photo in the manifest with its recorded bytes is done; one with other bytes was changed after its row, and
    the run must not touch anything (ML-14 D6)."""
    todo, changed = [], []
    for p in photos:
        key = manifest_key(p)
        if key not in done:
            todo.append(p)
        elif sha256_file(p) != done[key]["sha256_after"]:
            changed.append(key)
    return todo, changed


@contextmanager
def _sigint_deferred() -> Iterator[None]:
    """A Ctrl-C during the block is handled as it ends, not inside it. A handler, not a blocked signal: with a
    pool's threads running, a SIGINT the main thread blocks is taken by another thread and still raised here."""
    caught = []
    old = signal.signal(signal.SIGINT, lambda *_: caught.append(True))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)
        if caught and callable(old):
            old(signal.SIGINT, None)


class Rows:
    """A run's rows, taken in photo order: each one's progress line, then its manifest row if it is good (ML-14 D3,
    D9). A Ctrl-C lands between two rows, never inside one, so every row taken is in the manifest."""

    def __init__(self, photos: list[Path], writer: ManifestWriter | None):
        self.rows: list[dict] = []
        self._index, self._n, self._writer = {p: i for i, p in enumerate(photos, 1)}, len(photos), writer

    def take(self, path: Path, row: dict, show: bool = True) -> None:
        """The manifest row before the progress line: a photo done keeps its row if the printing fails."""
        with _sigint_deferred():
            self.rows.append(row)
            if self._writer and not row["problems"]:
                self._writer.write(row)
            if show:
                status = "FAIL " + "; ".join(row["problems"]) if row["problems"] else (
                    f"{len(row['removed'].split())} tags" if row["removed"] else "clean")
                print(f"[{self._index[path]}/{self._n}] {row['path']}: {status}", flush=True)


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _result(path: Path, future: Future) -> dict:
    """A worker's row; a pool broken under it (a worker killed, by the kernel's OOM killer say) is a failing row."""
    try:
        return future.result()
    except Exception as e:
        return failing_row(path, f"error: {type(e).__name__}: {e}")


def _stop_at_once(futures: dict[Path, Future]) -> None:
    """Cancels the photos not started and terminates the workers: a photo in progress keeps its original, unless its
    worker had replaced it in the instant before, which leaves it stripped and without its row."""
    for f in futures.values():
        f.cancel()
    for proc in multiprocessing.active_children():
        proc.terminate()
    print("stopped at once: the photos in progress keep their originals (one replaced in the instant before has no "
          "row); their temporary copies (.<name>.*) are beside them, to be deleted", file=sys.stderr, flush=True)


def _take_done(checked: list[tuple[Path, dict | None]], futures: dict[Path, Future], rows: Rows, show: bool) -> None:
    """Takes, in photo order, every row not yet taken that is ready, leaving out the photos never done."""
    for p, row in checked[len(rows.rows):]:
        f = futures.get(p)
        if row is None and f is not None and f.done() and not f.cancelled() and f.exception() is None:
            row = f.result()
        if row is not None:
            rows.take(p, row, show)


def run_pool(checked: list[tuple[Path, dict | None]], workers: int, dry_run: bool, rows: Rows) -> bool:
    """run_one on `workers` spawned processes for every photo in `checked` without a row yet, each row taken in
    photo order as soon as it and every row before it are done (ML-14 D3). Returns True if a Ctrl-C stopped it.

    Workers ignore SIGINT. The first Ctrl-C cancels the photos not yet started, waits for those in progress and takes
    every row done, in photo order, leaving out the photos that never ran (D4); a rerun does them. A second Ctrl-C
    terminates the workers at once (_stop_at_once)."""
    # spawn: rawpy's LibRaw runs OpenMP, which can deadlock in a forked worker (as in coffeecv.find_duplicates).
    # The workers are spawned (by submit) while this process ignores SIGINT, so they ignore it from birth: a Ctrl-C
    # during a worker's imports, before its initializer runs, would kill it and break the pool.
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    futures: dict[Path, Future] = {}
    try:
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_ignore_sigint)
        futures.update((p, pool.submit(run_one, p, dry_run)) for p, row in checked if row is None)
        signal.signal(signal.SIGINT, old_handler)       # inside the try: a Ctrl-C from here on is handled below
        for p, row in checked:
            rows.take(p, row or _result(p, futures[p]))
    except KeyboardInterrupt:
        # Cancelled here, not by shutdown(cancel_futures=True): that cancels in the pool's own thread, later, and
        # concurrent.futures.wait never counts a future cancelled that way as done (CPython 3.12).
        for f in futures.values():
            f.cancel()                                  # the photos not started; those in progress go on
        pool.shutdown(wait=False)
        running = [f for f in futures.values() if not f.done()]
        print(f"\nstopping: waiting for the {len(running)} photos already handed to the workers; Ctrl-C again to "
              f"stop at once", file=sys.stderr, flush=True)
        try:
            for f in running:
                f.exception()                           # waits until it is done
            pool.shutdown()
        except KeyboardInterrupt:
            _stop_at_once(futures)
        _take_done(checked, futures, rows, show=True)
        return True
    except BaseException:
        signal.signal(signal.SIGINT, old_handler)
        # A row that cannot be taken (the manifest's disk full, or stdout closed, say): no worker may strip a photo
        # whose row will not be written, and the photos already done keep their rows if the manifest takes them.
        _stop_at_once(futures)
        try:
            _take_done(checked, futures, rows, show=False)
        except Exception:
            pass                                        # the error raised below may be the manifest's own
        raise
    pool.shutdown()
    return False


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
    ap.add_argument("--jobs", type=int,
                    help="photos at once, each in its own process [min(CPU cores, 10, GB of memory and swap free at the "
                         "start)]")
    ap.add_argument("--max-memory", type=float,
                    help="GB one photo's decode may take; a photo above it fails undecoded [available at the start, "
                         "less 1 GB]")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.manifest:
        ap.error("--manifest is required unless --dry-run")

    photos, other = photos_under([d.resolve() if not d.is_absolute() else d for d in args.dirs])
    started = datetime.now(timezone.utc)
    version = exiftool_version()
    available = available_memory()
    budget = int(args.max_memory * GB) if args.max_memory is not None else available - GB
    # At most 10, about 1 GB a worker of the memory and swap free (ML-14 D15, the owner, 2026-10-30): a worker
    # peaked at 0.6 GB on segmenter_positives (D14), and the swap takes a run of several big photos at once.
    jobs = args.jobs if args.jobs is not None else max(1, min(os.cpu_count() or 1, MAX_DEFAULT_JOBS,
                                                              (available + free_swap()) // GB))
    todo = photos
    if args.manifest and not args.dry_run:
        todo, changed = resume_check(photos, read_manifest(args.manifest))
        for key in changed:
            print(f"  {key}: in the manifest, but its bytes are not the recorded sha256_after", file=sys.stderr)
        if changed:
            return 1
    header = [f"ML-3 D13 strip manifest: coffeecv/strip_metadata.py, exiftool {version}",
              f"started {started.isoformat(timespec='seconds')}", "exiftool " + " ".join(DELETE_ARGS), *args.header]
    writer = ManifestWriter(args.manifest, header) if args.manifest and not args.dry_run else None
    checked = [(p, too_big(p, budget)) for p in todo]
    # One photo for the workers, or --jobs 1: in this process, with no pool (D7).
    workers = min(jobs, sum(1 for _, row in checked if row is None))
    print(f"{len(todo)} photos to do, {max(workers, 1)} at once (--jobs {jobs}); decode budget {budget / GB:.3g} GB "
          f"a photo", file=sys.stderr, flush=True)
    taken = Rows(photos, writer)
    interrupted = False
    try:
        if workers > 1:
            interrupted = run_pool(checked, workers, args.dry_run, taken)
        else:
            for p, row in checked:
                taken.take(p, row or run_one(p, args.dry_run))
    except KeyboardInterrupt:
        # strip_one's finally has removed its temporary file; every row taken is in the manifest already.
        interrupted = True
    finally:
        if writer:
            writer.close()
    rows = taken.rows
    good = [r for r in rows if not r["problems"]]
    out_files = [sys.stdout] + ([open(args.report, "w")] if args.report else [])
    for out in out_files:
        if interrupted:
            print(f"STOPPED by Ctrl-C: {len(todo) - len(rows)} of {len(todo)} photos not done", file=out)
        print(f"{'DRY RUN: ' if args.dry_run else ''}exiftool {version}; {len(rows)} photos checked, "
              f"{sum(1 for r in good if r['removed'])} {'would be ' if args.dry_run else ''}stripped, "
              f"{len(rows) - len(good)} failing", file=out)
        report(rows, other, out)
    for out in out_files[1:]:
        out.close()
    return 130 if interrupted else 1 if len(good) != len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
