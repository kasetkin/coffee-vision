"""What the judge sees for one (photo, mask) item (ticket ML-2 P1, plan §4.3): an overview of the whole photo
plus full-resolution tiles along the mask boundary, so the mask is checked at full resolution and not only
in a thumbnail (R6).

    overview.png   long side <= 1568 px; the mask outline in 2 px magenta (it contrasts with green and
                   roasted beans), the D4 crop box dashed yellow, and the tile frames labelled T1..T6
    T1.png..T6.png 768 px tiles at full resolution with the outline, evenly spaced along the outer contour

Deviation from plan §4.3: the plan put two of the six tiles at the lowest-confidence boundary points
(decoder logit nearest 0). The judge's test masks are stored bits with no logits, and a planted defect has
none, so tile placement would differ between the masks under test and the masks the judge is checked on.
All six tiles are placed from the mask alone. An empty mask gets six tiles on a 3 x 2 grid over the photo,
so a missed pile can still be seen at full resolution (rule 4, negatives).

Every image is a PNG encoded from a pixel array, so no EXIF, GPS or C2PA metadata can reach the judge.
`to_photo_xy` maps a judge point given on the overview or a tile back to fractions of the photo.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coffeecv.segment_beans import d4_box

OVERVIEW_LONG_SIDE = 1568
TILE = 768
N_TILES = 6
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 200, 255)


@dataclass
class OverlayMeta:
    photo_hw: tuple[int, int]
    overview_scale: float                      # overview px = photo px * scale
    tiles: list[tuple[int, int, int, int]]     # (x0, y0, w, h) in photo pixels, T1 first
    empty_mask: bool


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def outline(rgb: np.ndarray, mask: np.ndarray, px: int, window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """`rgb` (the photo, or the `window` (x0, y0, w, h) of it) with the mask's boundary drawn about 2 * `px`
    pixels wide, in magenta. Outside the photo counts as outside the mask, so where the mask reaches the frame
    edge the edge is outlined too: a frame-filling mask shows a magenta border, an empty mask shows none.
    `mask` is always the whole photo's, so a window's own edges are not mistaken for mask boundary."""
    x0, y0, w, h = window or (0, 0, mask.shape[1], mask.shape[0])
    padded = np.pad(mask[max(y0 - px, 0):y0 + h + px, max(x0 - px, 0):x0 + w + px].astype(np.uint8),
                    ((max(px - y0, 0), max(y0 + h + px - mask.shape[0], 0)),
                     (max(px - x0, 0), max(x0 + w + px - mask.shape[1], 0))))
    edge = cv2.morphologyEx(padded, cv2.MORPH_GRADIENT, np.ones((2 * px + 1,) * 2, np.uint8))[px:px + h, px:px + w] > 0
    out = (rgb if window is None else rgb[y0:y0 + h, x0:x0 + w]).copy()
    out[edge] = MAGENTA
    return out


def tile_boxes(mask: np.ndarray, n: int = N_TILES, tile: int = TILE) -> list[tuple[int, int, int, int]]:
    """`n` tiles centred on points evenly spaced by arc length along the outer contours (longest first);
    on an empty mask, a 3 x 2 grid over the photo. Tiles are clipped to the photo."""
    h, w = mask.shape
    tw, th = min(tile, w), min(tile, h)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        pts = np.concatenate([c[:, 0, :] for c in sorted(contours, key=len, reverse=True)])
        centres = [tuple(pts[i]) for i in np.linspace(0, len(pts), n, endpoint=False).astype(int)]
    else:
        cols, rows = 3, max(1, n // 3)
        centres = [((c + 0.5) * w / cols, (r + 0.5) * h / rows) for r in range(rows) for c in range(cols)][:n]
    return [(int(np.clip(x - tw // 2, 0, w - tw)), int(np.clip(y - th // 2, 0, h - th)), tw, th) for x, y in centres]


def _dashed_rect(d: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, dash: int = 12) -> None:
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) / dash))
        at = lambda t: (a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n)  # noqa: E731
        for k in range(0, n, 2):
            d.line([at(k), at(min(k + 1, n))], fill=YELLOW, width=2)


def render(rgb: np.ndarray, mask: np.ndarray) -> tuple[dict[str, Image.Image], OverlayMeta]:
    """{"overview": ..., "T1": ..., ...} and the geometry needed to map points back to the photo."""
    h, w = mask.shape
    if rgb.shape[:2] != (h, w):
        raise ValueError(f"mask {mask.shape} does not match photo {rgb.shape[:2]}")
    mask = mask.astype(bool, copy=False)
    s = min(1.0, OVERVIEW_LONG_SIDE / max(h, w))
    size = (round(w * s), round(h * s))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    ov = Image.fromarray(outline(small, small_mask, 1))
    d = ImageDraw.Draw(ov)
    empty = not mask.any()
    if not empty:
        x0, y0, bw, bh = d4_box(mask)
        _dashed_rect(d, x0 * s, y0 * s, (x0 + bw) * s, (y0 + bh) * s)
    tiles = tile_boxes(mask)
    font = _font(max(16, round(max(size) / 50)))
    for k, (tx, ty, tw, th) in enumerate(tiles, 1):
        d.rectangle([tx * s, ty * s, (tx + tw) * s, (ty + th) * s], outline=CYAN, width=2)
        d.text((tx * s + 5, ty * s + 3), f"T{k}", fill=CYAN, font=font, stroke_width=2, stroke_fill="black")
    images = {"overview": ov}
    tile_font = _font(28)
    for k, (tx, ty, tw, th) in enumerate(tiles, 1):
        t = Image.fromarray(outline(rgb, mask, 1, (tx, ty, tw, th)))
        ImageDraw.Draw(t).text((8, 6), f"T{k}", fill=CYAN, font=tile_font, stroke_width=2, stroke_fill="black")
        images[f"T{k}"] = t
    return images, OverlayMeta((h, w), s, tiles, empty)


def write(rgb: np.ndarray, mask: np.ndarray, out_dir: Path) -> tuple[list[Path], OverlayMeta, str]:
    """Write the overlays as PNG plus meta.json. Returns (image paths, overview first; meta; overlay sha256
    over the PNG bytes in that order)."""
    images, meta = render(rgb, mask)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, h = [], hashlib.sha256()
    for name, im in images.items():
        p = out_dir / f"{name}.png"
        im.save(p, format="PNG", optimize=False)
        h.update(p.read_bytes())
        paths.append(p)
    (out_dir / "meta.json").write_text(json.dumps(asdict(meta)) + "\n")
    return paths, meta, h.hexdigest()


def load_meta(out_dir: Path) -> OverlayMeta:
    d = json.loads((out_dir / "meta.json").read_text())
    return OverlayMeta(tuple(d["photo_hw"]), d["overview_scale"], [tuple(t) for t in d["tiles"]], d["empty_mask"])


def to_photo_xy(where: str, x: float, y: float, meta: OverlayMeta) -> tuple[float, float]:
    """A point as fractions (x, y) of the overview or of tile Tk -> fractions of the photo's width/height
    (the convention of BeanSegmenter.predict_with_points)."""
    h, w = meta.photo_hw
    if where == "overview":
        px, py = x * (w - 1), y * (h - 1)
    else:
        k = int(where[1:])
        if not 1 <= k <= len(meta.tiles):
            raise ValueError(f"no tile {where}; this item has T1..T{len(meta.tiles)}")
        tx, ty, tw, th = meta.tiles[k - 1]
        px, py = tx + x * (tw - 1), ty + y * (th - 1)
    return px / (w - 1), py / (h - 1)
