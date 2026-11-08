"""A segmenter label's geometry carried from the stored frame into its canonical photo's (ticket ML-19 D9, D19, D28):
the same turn canonicalization applies (coffeecv.canonical.orient), and for a photo over 60 MP the same cap.

Points, fences and stroke paths are [x, y] fractions of the photo, mapped to pixels as f * (side - 1)
(coffeecv.seg_review.decode_points, fence_mask, paint_mask); under that convention a turn maps fractions exactly: a
flip is 1 - f, a transpose swaps x and y. A stroke's radius is in photo pixels: it is kept under a turn and scaled with
the photo under the cap. A mask (H x W bool, the stored frame) is capped by area averaging, thresholded at one half,
then turned as a view of it, the photo's own order (coffeecv.canonical.decode).
"""
from __future__ import annotations

import numpy as np

from coffeecv import canonical as cn

# EXIF orientation -> the orientation that undoes it: 6 and 8 undo each other, the rest undo themselves
INVERSE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 6}


def turn_point(x: float, y: float, orientation: int) -> tuple[float, float]:
    """A stored-frame point, [x, y] in fractions, in the upright frame of `orientation` (as coffeecv.canonical.orient
    turns the pixels)."""
    return {1: (x, y), 2: (1 - x, y), 3: (1 - x, 1 - y), 4: (x, 1 - y), 5: (y, x), 6: (1 - y, x),
            7: (1 - y, 1 - x), 8: (y, 1 - x)}[orientation]


def turn_points(points: list[list[float]], orientation: int) -> list[list[float]]:
    return [list(turn_point(x, y, orientation)) for x, y in points]


def turn_fence(fence: list[float], orientation: int) -> list[float]:
    """[x0, y0, x1, y1] turned, its corners sorted again."""
    (a, b), (c, d) = turn_point(fence[0], fence[1], orientation), turn_point(fence[2], fence[3], orientation)
    return [min(a, c), min(b, d), max(a, c), max(b, d)]


def turn_strokes(strokes: list[dict], orientation: int, scale: float = 1.0) -> list[dict]:
    """Each stroke's path turned, its radius (photo pixels) times `scale`, its other fields kept."""
    return [{**s, "path": turn_points(s["path"], orientation), "r": s["r"] * scale} for s in strokes]


def cap_mask(mask: np.ndarray) -> np.ndarray:
    """A stored-frame mask at its photo's canonical size (D27): area-averaged and thresholded at one half over
    60 MP, as it is otherwise."""
    h, w = mask.shape
    nw, nh = cn.capped_size(w, h)
    if (nw, nh) == (w, h):
        return mask
    import cv2
    return cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) >= 0.5


def canonical_mask(mask: np.ndarray, orientation: int) -> np.ndarray:
    """A stored-frame mask in its canonical photo's frame: capped, then turned."""
    return np.ascontiguousarray(cn.orient(cap_mask(mask), orientation))


def stored_mask(mask: np.ndarray, orientation: int) -> np.ndarray:
    """A canonical-frame mask turned back into the stored frame (not uncapped): pass14's check against pass13."""
    return np.ascontiguousarray(cn.orient(mask, INVERSE[orientation]))


def round4(v: float) -> float:
    """Fractions as seg_review writes them, four decimals."""
    return round(v, 4)
