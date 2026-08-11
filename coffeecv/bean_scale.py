"""Estimate bean pitch (centre-to-centre spacing) in pixels from a photo.

This is what makes the pipeline usable at inference. Patch size used to be a
fraction of the frame, which only produced sensible bean coverage because every
rig happened to frame a similar bean count -- an assumption nothing checked, and
one a user pointing a phone cannot be expected to satisfy. Measuring the pitch
instead makes "how many beans in a patch" a quantity the code controls directly.

**The same estimator must run at training and at inference.** A more accurate
method offline plus a cheap one live would inflate every reported metric: the
model would be trained and scored on well-sized patches, then meet worse-sized
ones in the field. With one shared estimator the absolute bias cancels -- a
constant k merely redefines what "patch scale" means, identically on both sides
-- so what matters is only consistency across rigs and per-image variance.

For the same reason the estimate is **per photo, not per session**. A session
median is roughly 3x more accurate (6.3% vs 19.2% MAPE), but at inference there
is only one photo, so training on session medians would reintroduce exactly the
mismatch this module exists to avoid. The estimator's ~24% per-image scatter is
instead absorbed as scale augmentation, because it is present during training.

Method and calibration come from analysis/bean_scale/, benchmarked against 30
hand-counted crops across all three rigs. The FFT radial profile won not on raw
accuracy but on bias *consistency* (spread 1.04 across rigs, versus 1.32 for the
distance-transform and granulometry methods that looked more promising). See
that directory's README for the full comparison.
"""
from __future__ import annotations

import cv2
import numpy as np

# Every candidate estimator reads something smaller than centre-to-centre
# spacing -- inscribed radius, or the short axis of an elongated bean -- so all
# of them need a multiplicative correction. This one is calibrated against the
# manual counts, not guessed: mean(estimate/ground_truth) over 30 crops was
# 0.847, so k = 1/0.847.
CALIBRATION_K = 1.18

# Search band for the dominant period, in cycles across the analysis window.
# 4..80 cycles over 1024px corresponds to a pitch of 12.8..256px in the resized
# image, which comfortably brackets every rig once the window is normalised.
_K_LO, _K_HI = 4, 80
_WINDOW = 1024


def _radial_mean(power: np.ndarray) -> np.ndarray:
    n = power.shape[0]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).astype(int)
    return np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


# Fraction of the short side analysed. The estimator is measured, and calibrated,
# on a window this size -- feeding it a whole frame instead overestimates pitch by
# 33% and inflates cross-rig spread from 1.04 to 1.27, because downscaling a 5056px
# photo into the analysis window leaves beans only ~65px across and lets vignetting
# and lighting gradients dominate the low-frequency end. Calibration and use must
# see the same field of view, so the crop happens here rather than at the call site.
ANALYSIS_FRAC = 0.40


def estimate_bean_pitch(img: np.ndarray, max_side: int = 1024) -> float:
    """Centre-to-centre bean spacing in pixels of `img`.

    `img` is BGR or grayscale, full resolution. Runs in ~150ms, which is what
    makes it affordable at inference as well as on every photo of every epoch's
    dataset build.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    side = int(min(gray.shape[:2]) * ANALYSIS_FRAC)
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    gray = gray[cy - side // 2:cy + side // 2, cx - side // 2:cx + side // 2]
    h, w = gray.shape
    scale = max(h, w) / max_side
    if scale > 1:
        gray = cv2.resize(gray, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    n = min(_WINDOW, gray.shape[0], gray.shape[1])
    n -= n % 2
    if n < 64:
        raise ValueError(f"image too small to estimate bean pitch ({gray.shape})")
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    g = gray[cy - n // 2:cy + n // 2, cx - n // 2:cx + n // 2].astype(np.float32)
    g -= g.mean()
    g *= np.outer(np.hanning(n), np.hanning(n))

    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    prof = _radial_mean(power)
    hi = min(_K_HI, len(prof) - 1)
    # Natural image spectra fall off roughly as 1/f, which would put the argmax
    # at the lowest frequency in the band regardless of content; the k^1.5 term
    # flattens that so the bean period itself is what stands out.
    band = prof[_K_LO:hi] * np.arange(_K_LO, hi) ** 1.5
    k = _K_LO + int(np.argmax(band))
    return float(n / k * scale * CALIBRATION_K)


def beans_across(img: np.ndarray, side_px: float | None = None) -> float:
    """How many beans span `side_px` (default: the image's short side).

    Defined as side / pitch, which equals sqrt(bean count) over a square of that
    side -- an area-derived average rather than a tally along one edge, because
    an edge tally depends on which line you pick and is far noisier.
    """
    pitch = estimate_bean_pitch(img)
    if side_px is None:
        side_px = min(img.shape[:2])
    return float(side_px / pitch)
