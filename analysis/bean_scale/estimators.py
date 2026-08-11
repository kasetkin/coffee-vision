"""Four candidate bean-scale estimators, plus the FFT method already on record.

All return an estimated centre-to-centre spacing in pixels of the *input crop*,
directly comparable to the manual ground truth
    spacing = crop_side / sqrt(n_beans)

Each is deliberately given the same input: the identical centre crop that was
rendered for manual counting, so no method gets a different field of view.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------- helpers
def _gray(img_bgr, max_side=1200):
    """Grayscale, capped resolution. Returns (gray, scale) where
    spacing_in_source = spacing_measured * scale."""
    h, w = img_bgr.shape[:2]
    s = max(h, w) / max_side
    if s > 1:
        img_bgr = cv2.resize(img_bgr, (int(w / s), int(h / s)), interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), s


def _radial_profile(power):
    n = power.shape[0]
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    tot = np.bincount(r.ravel(), power.ravel())
    cnt = np.maximum(np.bincount(r.ravel()), 1)
    return tot / cnt


# ---------------------------------------------------------------- M1
def m1_distance_transform(img_bgr):
    """Threshold beans against the darker inter-bean gaps, distance-transform,
    and read the median inscribed radius off the local maxima.

    For touching convex objects the distance-transform peak inside each object is
    its inscribed radius, so centre-to-centre spacing is about twice that. This is
    the standard recipe for sizing touching particles and is what makes it robust
    where simple connected components would merge every bean into one blob.
    """
    gray, scale = _gray(img_bgr)
    gray = cv2.GaussianBlur(gray, (0, 0), 2)
    # Beans are brighter than the shadowed gaps between them.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close pinholes (creases and silverskin read as dark inside a bean).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() < 2:
        return None
    # Local maxima on a neighbourhood scaled to the rough object size, so two
    # peaks inside one bean don't both count.
    approx_r = max(3, int(dist.max() * 0.6))
    mx = ndimage.maximum_filter(dist, size=approx_r)
    peaks = dist[(dist == mx) & (dist > 0.35 * dist.max())]
    if peaks.size < 4:
        return None
    return float(2.0 * np.median(peaks) * scale)


# ---------------------------------------------------------------- M2
def m2_granulometry(img_bgr, max_r=60):
    """Morphological pattern spectrum: open with growing discs and find the radius
    at which the image loses the most intensity. That radius is the characteristic
    object size, by construction rather than by inference."""
    gray, scale = _gray(img_bgr, max_side=600)  # opening is O(r^2); keep it small
    # On raw grayscale the spectrum is swamped by bean-surface texture and peaks
    # at r=3. Blurring hard and binarising first makes the opening act on bean
    # *shapes* rather than on their mottling.
    gray = cv2.GaussianBlur(gray, (0, 0), 5)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vols, radii = [], list(range(1, max_r + 1, 2))
    for r in radii:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        vols.append(float(cv2.morphologyEx(gray, cv2.MORPH_OPEN, se).sum()))
    vols = np.array(vols)
    spectrum = -np.diff(vols)          # intensity lost at each step
    if spectrum.size == 0 or spectrum.max() <= 0:
        return None
    r_star = radii[int(np.argmax(spectrum)) + 1]
    return float(2.0 * r_star * scale)


# ---------------------------------------------------------------- M3
def m3_autocorrelation(img_bgr):
    """2D autocorrelation via FFT; spacing is the first off-centre ring in the
    radial profile. Unlike a spectral argmax this reads a lag directly, so it
    cannot lock onto a harmonic of the true period."""
    gray, scale = _gray(img_bgr, max_side=1024)
    n = min(gray.shape[0], gray.shape[1], 1024)
    n -= n % 2
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    g = gray[cy - n // 2:cy + n // 2, cx - n // 2:cx + n // 2].astype(np.float32)
    g -= g.mean()
    g *= np.outer(np.hanning(n), np.hanning(n))
    F = np.fft.fft2(g)
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ac = np.fft.fftshift(ac)
    prof = _radial_profile(ac)
    prof = prof / prof[0]
    prof = np.convolve(prof, np.ones(9) / 9, mode="same")
    # Read the FIRST ZERO CROSSING, not the first local maximum. For a random
    # packing the autocorrelation decays to zero at about the object radius, so
    # twice that lag is the centre-to-centre spacing. The local-maximum reading
    # tried first latched onto noise ripples and came out ~2x low.
    below = np.where(prof[: len(prof) // 2] < 0)[0]
    if below.size == 0:
        return None
    return float(2.0 * below[0] * scale)


# ---------------------------------------------------------------- M4
_SAM = None


def m4_mobile_sam(img_bgr, weights="/workspace/mobile_sam.pt"):
    """MobileSAM 'segment everything', then the median equivalent diameter of the
    plausible masks. The learned baseline: no thresholding assumptions, but the
    heaviest by far and dependent on SAM finding beans interesting."""
    global _SAM
    if _SAM is None:
        from ultralytics import SAM
        _SAM = SAM(weights)
    gray_side = 512   # 1024 costs ~92 s/image on CPU; 512 is ~4x cheaper
    h, w = img_bgr.shape[:2]
    s = max(h, w) / gray_side
    small = cv2.resize(img_bgr, (int(w / s), int(h / s)), interpolation=cv2.INTER_AREA) if s > 1 else img_bgr
    if s <= 1:
        s = 1.0
    res = _SAM(small, verbose=False)
    if not res or res[0].masks is None:
        return None
    areas = res[0].masks.data.sum(dim=(1, 2)).cpu().numpy().astype(float)
    frame = small.shape[0] * small.shape[1]
    # Drop masks that cannot be a single bean: background sheets and specks.
    areas = areas[(areas > frame * 0.002) & (areas < frame * 0.12)]
    if areas.size < 4:
        return None
    return float(2.0 * np.sqrt(np.median(areas) / np.pi) * s)


# ---------------------------------------------------------------- reference
def m0_fft_radial(img_bgr):
    """The estimator already used to characterise the rigs, kept as the incumbent."""
    gray, scale = _gray(img_bgr, max_side=1024)
    n = 1024
    h, w = gray.shape
    if min(h, w) < n:
        n = min(h, w) - (min(h, w) % 2)
    cy, cx = h // 2, w // 2
    g = gray[cy - n // 2:cy + n // 2, cx - n // 2:cx + n // 2].astype(np.float32)
    g -= g.mean()
    g *= np.outer(np.hanning(n), np.hanning(n))
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    prof = _radial_profile(F)
    lo, hi = 4, min(80, len(prof) - 1)
    band = prof[lo:hi] * np.arange(lo, hi) ** 1.5
    k = lo + int(np.argmax(band))
    return float(n / k * scale)


METHODS = {
    "M1 dist-transform": m1_distance_transform,
    "M2 granulometry": m2_granulometry,
    "M3 autocorrelation": m3_autocorrelation,
    "M4 MobileSAM": m4_mobile_sam,
    "M0 FFT (incumbent)": m0_fft_radial,
}


def timed(fn, img):
    t = time.perf_counter()
    try:
        v = fn(img)
    except Exception as e:
        return None, time.perf_counter() - t, str(e)[:60]
    return v, time.perf_counter() - t, None
