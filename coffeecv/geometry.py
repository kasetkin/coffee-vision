"""Geometry for sampling leakage-free patches from the circular macro-lens photos.

Each source photo is square with a lens circle exactly inscribed in the frame
(radius = half the image width) and alpha==0 outside that circle. We compute a
safe square inscribed in that circle, then split the safe square into three
disjoint regions (train/val/test) so that patches cropped from different
regions can never share a pixel, even though they come from the same photo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def compute_valid_region(img_h: int, img_w: int, safety_margin: float = 0.97) -> Region:
    """Largest square inscribed in the lens circle, shrunk by `safety_margin`."""
    r = min(img_h, img_w) / 2
    cy, cx = img_h / 2, img_w / 2
    half_square = (r / math.sqrt(2)) * safety_margin
    # Round inward so the region never extends past the safe bound.
    return Region(
        y0=math.ceil(cy - half_square),
        y1=math.floor(cy + half_square),
        x0=math.ceil(cx - half_square),
        x1=math.floor(cx + half_square),
    )


def split_regions(valid_region: Region) -> dict[str, Region]:
    """Split the safe square into 3 disjoint regions: train (top half),
    val (bottom-left quarter), test (bottom-right quarter)."""
    cy = (valid_region.y0 + valid_region.y1) // 2
    cx = (valid_region.x0 + valid_region.x1) // 2
    return {
        "train": Region(y0=valid_region.y0, y1=cy, x0=valid_region.x0, x1=valid_region.x1),
        "val": Region(y0=cy, y1=valid_region.y1, x0=valid_region.x0, x1=cx),
        "test": Region(y0=cy, y1=valid_region.y1, x0=cx, x1=valid_region.x1),
    }


def sample_patch_boxes(rng: np.random.Generator, region: Region, n: int, crop_size: int) -> list[Region]:
    """Sample n crop_size x crop_size boxes with top-left uniformly random
    such that the whole box stays inside `region`."""
    max_y0 = region.y1 - crop_size
    max_x0 = region.x1 - crop_size
    if max_y0 < region.y0 or max_x0 < region.x0:
        raise ValueError(
            f"crop_size={crop_size} does not fit inside region "
            f"({region.height}x{region.width})"
        )
    ys = rng.integers(region.y0, max_y0 + 1, size=n)
    xs = rng.integers(region.x0, max_x0 + 1, size=n)
    return [Region(y0=int(y), y1=int(y) + crop_size, x0=int(x), x1=int(x) + crop_size) for y, x in zip(ys, xs)]


def assert_region_fully_opaque(alpha: np.ndarray, region: Region) -> None:
    """Defensive check: every pixel in `region` must be inside the lens circle
    (alpha != 0). Fails loudly if a future capture session has different lens
    geometry, instead of silently sampling patches with black/transparent content."""
    sub = alpha[region.y0:region.y1, region.x0:region.x1]
    n_bad = int(np.sum(sub == 0))
    if n_bad:
        raise AssertionError(
            f"Region {region} contains {n_bad} pixels outside the lens circle "
            "(alpha==0) — lens geometry assumption no longer holds."
        )
