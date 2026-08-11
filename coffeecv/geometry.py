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


def scale_range_px(region: Region, frac_min: float, frac_max: float) -> tuple[int, int]:
    """Patch side range in pixels, as a fraction of the region's short side.

    Sizing the patch relative to the frame rather than in absolute pixels is what
    makes one setting mean the same thing on every rig. A fixed 900px patch spans
    ~8.8 beans on the 2026-08-07 rig but only ~2.8 on sony_cam, because the rigs
    differ 3.2x in magnification -- so the model is shown a pile on one rig and a
    single bean on another and asked to call them the same class. These three
    rigs happen to frame a similar bean *count* (~215-282 per photo) despite
    those very different pixel scales, so a fraction of the frame lands within
    ~2-9 beans across on all three without needing any per-rig constant.

    Deliberately not a per-rig calibration: the point is that the model must be
    scale-invariant, so the range is sampled wide and identically everywhere
    rather than normalised away per folder.
    """
    short_side = min(region.width, region.height)
    lo = max(1, int(round(short_side * frac_min)))
    hi = max(lo, int(round(short_side * frac_max)))
    return lo, hi


def sample_scaled_patch_boxes(
    rng: np.random.Generator,
    region: Region,
    n: int,
    frac_min: float,
    frac_max: float,
    max_jitter_deg: float = 0.0,
) -> list[tuple[Region, float, int]]:
    """Sample n (bounding_box, angle, patch_side) triples with per-patch scale.

    `patch_side` is the square the caller should end up with after any rotation;
    the returned box is what to crop (larger than patch_side when the patch is
    rotated, so the rotation lands on real pixels -- same no-fill guarantee as
    `sample_rotated_patch_boxes`).

    Sides are drawn log-uniformly, not uniformly: scale is a ratio quantity, and
    a uniform draw over 0.15-0.60 of the frame would spend most of its mass on
    the coarse half of the range and undersample the close-up end, which is
    exactly where the held-out rig sits.
    """
    lo, hi = scale_range_px(region, frac_min, frac_max)
    sides = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n)).astype(int)
    angles = (
        rng.uniform(-max_jitter_deg, max_jitter_deg, size=n)
        if max_jitter_deg > 0 else np.zeros(n)
    )
    out = []
    for side, angle in zip(sides, angles):
        side = int(side)
        box_side = rotated_box_side(side, angle) if angle else side
        max_y0, max_x0 = region.y1 - box_side, region.x1 - box_side
        if max_y0 < region.y0 or max_x0 < region.x0:
            # Shrink to whatever does fit rather than failing: the small end of
            # the range is always placeable, and refusing here would make the
            # widest scale range unusable on the smallest rig.
            box_side = min(region.width, region.height)
            side = int(box_side / (rotated_box_side(1000, angle) / 1000)) if angle else box_side
            max_y0, max_x0 = region.y1 - box_side, region.x1 - box_side
        y = int(rng.integers(region.y0, max_y0 + 1))
        x = int(rng.integers(region.x0, max_x0 + 1))
        out.append((Region(y0=y, y1=y + box_side, x0=x, x1=x + box_side), float(angle), side))
    return out


def sample_bean_unit_patch_boxes(
    rng: np.random.Generator,
    region: Region,
    n: int,
    pitch_px: float,
    beans_min: float,
    beans_max: float,
    max_jitter_deg: float = 0.0,
) -> tuple[list[tuple[Region, float, int]], int]:
    """Sample n patches sized in *bean units* rather than pixels or frame fractions.

    A patch quoted as B beans across has side B * pitch and contains about B^2
    beans, since B is defined as side/pitch = sqrt(count) over the square. Sizing
    this way makes bean coverage the quantity the code controls directly, instead
    of something that falls out of how the shot was framed -- which is what lets
    the same setting mean the same thing on a rig the model has never seen.

    B is drawn log-uniformly: scale is a ratio quantity, so a uniform draw would
    concentrate mass at the coarse end of the range.

    Returns (boxes, n_clamped). A patch is clamped when the requested bean count
    needs more pixels than the photo has room for; the caller should watch that
    count, because a rig that clamps often is framed too tightly for the
    configured range and its patches will be less varied than intended.
    """
    room = min(region.width, region.height)
    beans = np.exp(rng.uniform(np.log(beans_min), np.log(beans_max), size=n))
    angles = (
        rng.uniform(-max_jitter_deg, max_jitter_deg, size=n)
        if max_jitter_deg > 0 else np.zeros(n)
    )
    out, clamped = [], 0
    for b, angle in zip(beans, angles):
        side = int(round(b * pitch_px))
        box_side = rotated_box_side(side, angle) if angle else side
        if box_side > room:
            # Take the largest patch that fits rather than failing: the
            # alternative is dropping the sample, which would silently bias the
            # scale distribution toward the small end.
            box_side = room
            side = int(box_side / (rotated_box_side(1000, angle) / 1000)) if angle else box_side
            clamped += 1
        max_y0, max_x0 = region.y1 - box_side, region.x1 - box_side
        y = int(rng.integers(region.y0, max_y0 + 1))
        x = int(rng.integers(region.x0, max_x0 + 1))
        out.append((Region(y0=y, y1=y + box_side, x0=x, x1=x + box_side), float(angle), side))
    return out, clamped


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
