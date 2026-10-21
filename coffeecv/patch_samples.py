"""Draw docs/patch_samples.png, the README's sample patches: for each country, one patch of its green beans and one of
its roasted beans, or just the one its training pool has.

The photos are listed in docs/patch_samples.txt (one line per country and roast state). Each patch is cut the way
training cuts it, through `infer.sample_patches` on the photo's segmenter crop and bean mask, with params.yaml's
geometry and a fixed seed, so the picture shows what the model sees and redrawing it gives the same pixels.

    python -m coffeecv.patch_samples                       # docs/patch_samples.txt -> docs/patch_samples.png
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coffeecv.class_list import ClassList, load_classes
from coffeecv.config import REPO_ROOT, RunConfig

LIST_PATH = REPO_ROOT / "docs" / "patch_samples.txt"
OUT_PATH = REPO_ROOT / "docs" / "patch_samples.png"
STATES = ("green", "roasted")
CLASS_DIR_RE = re.compile(r"^class_(\d+)__")

TILE = 200          # px per patch on the sheet (patches are stored at 448)
TILE_GAP = 8        # between a country's two patches
GROUP_GAP = 36      # between countries
ROW_WIDTH = 1300    # a row of countries wraps past this
MARGIN = 24
TITLE_H = 34        # the country's name, above its patches
CAPTION_H = 28      # "green" / "roasted", under each patch
ROW_GAP = 20
BG, INK, MUTED = (255, 255, 255), (33, 28, 21), (108, 98, 82)


@dataclass(frozen=True)
class Sample:
    country: str
    state: str
    crop: Path        # repo-relative


def read_list(path: Path, classes: ClassList) -> list[Sample]:
    """The list's samples, in file order. Refuses an unknown country or state, a repeated (country, state), and a
    crop whose class folder is another country's."""
    samples, seen = [], set()
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path.name}:{n}: expected '<country> <green|roasted> <crop>', got {raw!r}")
        country, state, crop = parts
        if country not in classes.keys:
            raise ValueError(f"{path.name}:{n}: {country!r} is not a country in the class list")
        if state not in STATES:
            raise ValueError(f"{path.name}:{n}: state {state!r} is not one of {STATES}")
        if (country, state) in seen:
            raise ValueError(f"{path.name}:{n}: a second {country} {state} line")
        m = CLASS_DIR_RE.match(Path(crop).parent.name)
        if m is None or classes.key_of(m.group(1)) != country:
            raise ValueError(f"{path.name}:{n}: {crop} is not in a class folder of {country}")
        seen.add((country, state))
        samples.append(Sample(country, state, Path(crop)))
    return samples


def cut_patch(crop: Path, cfg: RunConfig) -> Image.Image:
    """One patch of `crop` (repo-relative), placed against its bean mask exactly as training places them."""
    from coffeecv.dataset import load_bean_mask
    from coffeecv.infer import SegmentedPhoto, sample_patches
    path = REPO_ROOT / crop
    rgb = np.array(Image.open(path).convert("RGB"))
    seg = SegmentedPhoto(photo=rgb, rgb=rgb, mask=load_bean_mask(path, rgb.shape[:2]), photo_mask=None,
                         crop_info=None, seg_diag={}, timing_ms={})
    patches, _ = sample_patches(seg, cfg, n_patches=1, seed_key=[0, 0])
    return patches[0]


def display_name(country: str) -> str:
    """The class key as a reader writes it: CostaRica -> Costa Rica."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", country)


def _font(size: int) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def draw_sheet(groups: list[tuple[str, list[tuple[str, Image.Image]]]]) -> Image.Image:
    """Countries left to right, wrapping into centred rows: each country's name over its one or two patches, each
    patch captioned with its roast state."""
    def width(g):
        return len(g[1]) * TILE + (len(g[1]) - 1) * TILE_GAP

    rows: list[list] = [[]]
    for g in groups:
        row = rows[-1]
        if row and sum(map(width, row)) + len(row) * GROUP_GAP + width(g) > ROW_WIDTH:
            rows.append(row := [])
        row.append(g)
    row_w = [sum(map(width, r)) + (len(r) - 1) * GROUP_GAP for r in rows]
    row_h = TITLE_H + TILE + CAPTION_H
    sheet = Image.new("RGB", (max(row_w) + 2 * MARGIN, len(rows) * row_h + (len(rows) - 1) * ROW_GAP + 2 * MARGIN),
                      BG)
    draw, title_font, caption_font = ImageDraw.Draw(sheet), _font(22), _font(16)
    for r, (row, w) in enumerate(zip(rows, row_w)):
        x, y = (sheet.width - w) // 2, MARGIN + r * (row_h + ROW_GAP)
        for country, tiles in row:
            gw = width((country, tiles))
            draw.text((x + gw / 2, y + TITLE_H / 2), display_name(country), fill=INK, font=title_font, anchor="mm")
            for i, (state, patch) in enumerate(tiles):
                tx = x + i * (TILE + TILE_GAP)
                sheet.paste(patch.resize((TILE, TILE), Image.LANCZOS), (tx, y + TITLE_H))
                draw.text((tx + TILE / 2, y + TITLE_H + TILE + CAPTION_H / 2), state, fill=MUTED, font=caption_font,
                          anchor="mm")
            x += gw + GROUP_GAP
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=LIST_PATH, help="the samples list (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="the sheet to write (default: %(default)s)")
    args = ap.parse_args()

    cfg = RunConfig.from_params_yaml()
    samples = read_list(args.list, load_classes(REPO_ROOT / cfg.classes_file))
    groups: dict[str, list[tuple[str, Image.Image]]] = {}
    for s in sorted(samples, key=lambda s: (display_name(s.country), STATES.index(s.state))):
        groups.setdefault(s.country, []).append((s.state, cut_patch(s.crop, cfg)))
    sheet = draw_sheet(list(groups.items()))
    sheet.save(args.out, optimize=True)
    print(f"{args.out}: {len(groups)} countries, {len(samples)} patches, {sheet.width}x{sheet.height}")


if __name__ == "__main__":
    main()
