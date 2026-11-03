"""The page's example photos (ticket OPS-16 D1, D3, D7; ADR 0032): each original, found by its sha256, re-encoded as a
lossy JPEG XL at full resolution and given a small WebP thumbnail made from that JPEG XL.

    python -I scripts/make_example_photos.py [--dataset DIR] [--out DIR]

An original is looked for at its dataset path and in dataset/example_photos/, where ML-16 D3 moves it. The JPEG XL
carries no metadata: no EXIF (pillow-jxl-plugin writes the image's own back unless given exif=b""), no XMP, no ICC
profile, and no orientation tag, since the pixels are turned upright first. Each file is then decoded back through
coffeecv/photo_io.py, the decoder training and serving use, and must come out at the original's size.
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

from coffeecv.photo_io import DATASET_MAX_PIXELS, load_rgb_image  # noqa: E402 -- also registers HEIF and JPEG XL

QUALITY = 90       # D3: "lossy JXL is fine", q90; a photo whose answer changes gets a higher one, said in the plan
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
    Example("vietnam-robusta", "6f9d2dcbd82f898de0e16f5e6aab92f4bd28e029e6dc9b3a4c5f449f813797b1",
            "random_date_raccoon/class_009__Vietnam_Robusta/PXL_20261007_043818022.jpg"),
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
    with Image.open(original) as im:
        # A new image, not the opened one: pillow-jxl-plugin reconstructs an opened JPEG losslessly whatever the
        # quality, and an opened image's EXIF would be written back.
        upright = ImageOps.exif_transpose(im).convert("RGB")
    upright.save(jxl, "JXL", quality=QUALITY, lossless_jpeg=False, exif=b"")
    with Image.open(jxl) as decoded:
        thumb = decoded.convert("RGB")
        thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
    thumb.save(webp, "WEBP", quality=THUMB_QUALITY, method=6)


def check(original: Path, jxl: Path, webp: Path) -> list[str]:
    """What is wrong with the two files, if anything."""
    problems = []
    if load_rgb_image(jxl).shape != load_rgb_image(original).shape:
        problems.append("the JPEG XL decodes at another size or orientation than the original")
    # An ICC profile is not checked in the JPEG XL: libjxl's decoder reports one built from the codestream's colour
    # encoding (504 bytes for every file here, 2026-11-03), and none is written (pillow-jxl-plugin's _save has no
    # icc_profile). The WebP is written from that decode, without it.
    for path in (jxl, webp):
        with Image.open(path) as im:
            if im.getexif() or im.info.get("exif") or im.info.get("xmp") or \
                    (path is webp and im.info.get("icc_profile")):
                problems.append(f"{path.name} carries metadata")
    with Image.open(webp) as im:
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
