"""The page's example photos (ticket OPS-16 D1, D3, D7; ADR 0032; ticket ML-19 D17, D45): each original, found by its
sha256, as its canonical photo (coffeecv.canonical), and given a small WebP thumbnail made from that canonical photo.

    python -I scripts/make_example_photos.py [--dataset DIR] [--out DIR]

An original is looked for at its dataset path and in dataset/example_photos/, where ML-16 D3 moves it. Since ML-19 the
JPEG XL is the canonical photo, so the front passes a picked example on as it is (D17), and pillow-jxl-plugin is no
longer used (D45). Until ML-19 it was a q90 8-bit encode by pillow-jxl-plugin. Each file is checked: canonical, and
decoding to the original's upright size, capped as canonicalization caps.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image, ImageOps  # noqa: E402

from coffeecv import canonical  # noqa: E402
from coffeecv.photo_io import DATASET_MAX_PIXELS  # noqa: E402 -- also registers HEIF with Pillow

THUMB_SIDE = 400   # D7: the picker's thumbnail, its long side
THUMB_QUALITY = 80


@dataclass(frozen=True)
class Example:
    name: str        # webapp/static/examples/<name>.jxl and .webp
    sha256: str      # the original's
    dataset_path: str


EXAMPLES = (
    Example("brazil-grinder", "405fbd0a4aea61b8cb722e67711c6b32a23fd86badd4812ae4dc0deee22d8a4d",
            "random_date_raccoon/class_013__Brazil_SulDeMinas/20261016_131741_Pixel_9_Pro_Brazil.jpg"),
    Example("ethiopia-green", "91077cd6cf3f9f154ca693ee347116485c2b8a475e89f59c4aae6452c4296b81",
            "2026-08-25__iphone/class_001__Ethiopia_Sidamo/IMG_6302D.HEIC"),
    Example("indonesia-roasted", "b3071423896a03d93959b0ca10054608d2408c15a722a559ae73fbc2cb14ce1b",
            "2026-08-30__sony/class_010__Indonesia_Java/PIC_20260830_200224.JPG"),
    Example("cappuccino", "0342176ffd272b836a3cf1989bdcb5a5ebb4d52551333263ecc31bdf1b5abf63",
            "ood_negatives/user_realworld/PXL_20261031_072103744.jpg"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_original(example: Example, dataset: Path) -> Path:
    """The original at its dataset path, else in dataset/example_photos/ (ML-16 D3), by its sha256 either way."""
    candidates = [dataset / example.dataset_path, *sorted((dataset / "example_photos").rglob("*"))]
    for path in candidates:
        if path.is_file() and path.suffix != ".csv" and sha256(path) == example.sha256:
            return path
    raise SystemExit(f"{example.name}: no file with sha256 {example.sha256[:16]} at {example.dataset_path} "
                     f"or in {dataset / 'example_photos'}")


def encode(original: Path, jxl: Path, webp: Path) -> None:
    jxl.write_bytes(canonical.canonicalize_file(original))
    thumb = Image.fromarray(canonical.load_canonical(jxl.read_bytes()))   # 8-bit sRGB, as the models read it
    thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
    thumb.save(webp, "WEBP", quality=THUMB_QUALITY, method=6)


def check(original: Path, jxl: Path, webp: Path) -> list[str]:
    """What is wrong with the two files, if anything."""
    problems = [f"{jxl.name} is not canonical: {p}" for p in canonical.canonical_problems(jxl.read_bytes())]
    with Image.open(original) as im:
        w, h = ImageOps.exif_transpose(im).size if original.suffix.lower() not in (".heic", ".heif") else im.size
    want = canonical.capped_size(w, h)
    got = canonical.load_canonical(jxl.read_bytes()).shape[1::-1] if not problems else None
    if got is not None and tuple(got) != tuple(want):
        problems.append(f"{jxl.name} decodes at {tuple(got)}, not the original's upright {want}")
    with Image.open(webp) as im:
        if im.getexif() or im.info.get("exif") or im.info.get("xmp") or im.info.get("icc_profile"):
            problems.append(f"{webp.name} carries metadata")
        if max(im.size) != THUMB_SIDE:
            problems.append(f"{webp.name} is {im.size}, not {THUMB_SIDE} px on its long side")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=REPO / "dataset", help="the dataset's root (default: the repo's)")
    ap.add_argument("--out", type=Path, default=REPO / "webapp" / "static" / "examples",
                    help="where the files go (default: webapp/static/examples)")
    args = ap.parse_args()
    Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS   # the owner's own photos, as the dataset tools read them
    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    for ex in EXAMPLES:
        original = find_original(ex, args.dataset)
        jxl, webp = args.out / f"{ex.name}.jxl", args.out / f"{ex.name}.webp"
        encode(original, jxl, webp)
        problems = check(original, jxl, webp)
        failed |= bool(problems)
        print(f"{ex.name:18} {original.stat().st_size / 2**20:5.2f} MB -> {jxl.stat().st_size / 2**20:5.2f} MB JXL, "
              f"{webp.stat().st_size / 1024:5.1f} KB WebP  {'; '.join(problems) or 'ok'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
