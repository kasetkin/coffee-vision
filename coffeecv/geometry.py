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


def rotated_box_side(crop_size: int, angle_deg: float) -> int:
    """Side of the axis-aligned bounding box of a crop_size square rotated by
    `angle_deg`."""
    r = math.radians(abs(angle_deg))
    return math.ceil(crop_size * (math.cos(r) + math.sin(r)))


def sample_rotated_patch_boxes(
    rng: np.random.Generator, region: Region, n: int, crop_size: int, max_jitter_deg: float
) -> list[tuple[Region, float]]:
    """Sample n (bounding_box, angle) pairs for small-angle-jittered patches.

    Each angle is drawn uniformly from [-max_jitter_deg, +max_jitter_deg], and
    the returned box is the axis-aligned bounding box of the rotated crop_size
    square, placed so it lies entirely inside `region`. Cropping that box,
    rotating it by the angle and centre-cropping back to crop_size then yields a
    patch whose every pixel is real source content — no fill, and no change to
    the patch's scale, because the extra pixels the rotation needs are taken from
    the source photo rather than invented.

    Costs placement room: the bounding box is bigger than crop_size, so less of
    the valid region is left to translate within. On this dataset (tightest valid
    region 1016px wide, crop_size 900) that is 116px at 0 degrees, 40px at 5 and
    negative beyond 7 — see exp 32, where ~17px of room was enough to measurably
    hurt. `assert_jitter_fits` checks this up front rather than at sample time.
    """
    angles = rng.uniform(-max_jitter_deg, max_jitter_deg, size=n)
    boxes = []
    for angle in angles:
        side = rotated_box_side(crop_size, angle)
        max_y0, max_x0 = region.y1 - side, region.x1 - side
        if max_y0 < region.y0 or max_x0 < region.x0:
            raise ValueError(
                f"rotated box (side {side} for crop_size={crop_size} at {angle:.2f} deg) "
                f"does not fit inside region ({region.height}x{region.width})"
            )
        y = int(rng.integers(region.y0, max_y0 + 1))
        x = int(rng.integers(region.x0, max_x0 + 1))
        boxes.append((Region(y0=y, y1=y + side, x0=x, x1=x + side), float(angle)))
    return boxes


def assert_jitter_fits(
    region: Region, crop_size: int, max_jitter_deg: float, min_room: int = 25
) -> None:
    """Fail loudly at dataset construction if the jittered bounding box leaves so
    little placement room that patches from this photo would be near-identical
    every time — the failure mode exp 32 found at patch_crop_size=1000."""
    side = rotated_box_side(crop_size, max_jitter_deg)
    room = min(region.width, region.height) - side
    if room < 0:
        raise ValueError(
            f"rotation_jitter_degrees={max_jitter_deg} needs a {side}px box but the valid "
            f"region is only {region.width}x{region.height} — reduce the jitter or crop_size"
        )
    if room < min_room:
        raise ValueError(
            f"rotation_jitter_degrees={max_jitter_deg} leaves only {room}px of placement room "
            f"(need >={min_room}); patches from this photo would barely vary — see exp 32"
        )


def compute_valid_region_rect(img_h: int, img_w: int, safety_margin: float = 0.97) -> Region:
    """Whole-image region shrunk by `safety_margin`, for already-cropped
    rectangular photos with no lens circle to inscribe within (see
    coffeecv.dataset.MultiPhotoPatchDataset, used for the box-rig dataset)."""
    my = img_h * (1 - safety_margin) / 2
    mx = img_w * (1 - safety_margin) / 2
    return Region(
        y0=math.ceil(my), y1=math.floor(img_h - my),
        x0=math.ceil(mx), x1=math.floor(img_w - mx),
    )


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
