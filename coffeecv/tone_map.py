"""Tone mapping: an HDR main image's light into the canonical photo's standard range (ticket ML-29, P4; D2, D11, D21).
A prototype for the owner's pick: nothing imports it, no DVC stage depends on it, and P5 wires the operator chosen.

    pq_to_relative(signal, white)          -> linear light, 1.0 = reference white
    hlg_to_relative(signal, ootf, ...)     -> the same for HLG
    clip_at_white / eetf_bt2408 / scale_by_peak / reinhard_max_rgb (x, peak) -> linear light in [0, 1]

The curves are written from ITU-R BT.2100-3 (02/2025): PQ from Table 4, HLG from Table 5 and its notes 5f and 5i; no
library. Every operator is a pure function on linear Rec.2020 light, shape (..., 3), 1.0 the reference white
(BT.2408's 203 cd/m2, HLG's 75%, a gain map's standard-range white), `peak` the brightest light it must bring
back, relative to that white (at least 1). Each returns light in [0, 1], which the canonical encode then puts in
Rec.2020 with the BT.709 curve. The candidates (analysis/ml29/research_tone_mapping.md):

- clip_at_white:    white stays white, everything above is clipped per channel (libavif's rendering of PQ and HLG
                    into a standard-range output);
- eetf_bt2408:      BT.2408-8 Annex 5's EETF (the BT.2390 roll-off), source [0, peak], target [0, white], applied to
                    max(R, G, B) so that hue and saturation are kept (Annex 5, option 5);
- scale_by_peak:    the peak becomes white, everything darkened alike (libjxl's linear reading, where 1.0 is the
                    file's intensity target);
- reinhard_max_rgb: libultrahdr's global tone map, extended Reinhard on max(R, G, B) with white point at the peak.
"""
from __future__ import annotations

import numpy as np

REFERENCE_WHITE = 203.0                 # cd/m2: BT.2408-8, 2.1 and Table 1 (58% PQ, 75% HLG)
PQ_PEAK = 10000.0                       # cd/m2: BT.2100 Table 4, signal 1.0
HLG_NOMINAL_PEAK = 1000.0               # cd/m2: BT.2100 Table 5, the display at which gamma is 1.2
HLG_REFERENCE_SIGNAL = 0.75             # BT.2408-8 Table 1
BT2100_LUMINANCE = (0.2627, 0.6780, 0.0593)   # BT.2100 Table 5, the OOTF's Y_S

# BT.2100 Table 4
M1 = 2610 / 16384
M2 = 2523 / 4096 * 128
C1 = 3424 / 4096
C2 = 2413 / 4096 * 32
C3 = 2392 / 4096 * 32
# BT.2100 Table 5 and note 5c
A = 0.17883277
B = 1 - 4 * A
C = 0.5 - A * np.log(4 * A)


# ----- PQ -----

def pq_eotf(e: np.ndarray) -> np.ndarray:
    """PQ signal [0, 1] to display light in cd/m2 (BT.2100 Table 4, EOTF). Negative signal gives 0."""
    p = np.power(np.clip(e, 0, 1), 1 / M2)
    return PQ_PEAK * np.power(np.maximum(p - C1, 0) / (C2 - C3 * p), 1 / M1)


def pq_inverse_eotf(fd: np.ndarray) -> np.ndarray:
    """Display light in cd/m2 to PQ signal (BT.2100 Table 4, EOTF^-1); light above 10,000 cd/m2 is clipped."""
    y = np.power(np.clip(np.asarray(fd, np.float64) / PQ_PEAK, 0, 1), M1)
    return np.power((C1 + C2 * y) / (1 + C3 * y), M2)


def pq_to_relative(e: np.ndarray, white: float = REFERENCE_WHITE) -> np.ndarray:
    """PQ signal to linear light with `white` cd/m2 as 1.0. The default is BT.2408's reference white; libjxl's own
    linear reading takes the file's intensity target instead, which is scale_by_peak in this module's terms."""
    return pq_eotf(e) / white


# ----- HLG -----

def hlg_oetf(e: np.ndarray) -> np.ndarray:
    """Scene light [0, 1] to HLG signal (BT.2100 Table 5, OETF)."""
    e = np.clip(np.asarray(e, np.float64), 0, 1)
    return np.where(e <= 1 / 12, np.sqrt(3 * e), A * np.log(np.maximum(12 * e - B, 1e-12)) + C)


def hlg_inverse_oetf(v: np.ndarray) -> np.ndarray:
    """HLG signal to scene light [0, 1] (BT.2100 Table 5, OETF^-1)."""
    v = np.clip(np.asarray(v, np.float64), 0, 1)
    return np.where(v <= 0.5, v * v / 3, (np.exp((v - C) / A) + B) / 12)


def hlg_system_gamma(lw: float = HLG_NOMINAL_PEAK) -> float:
    """BT.2100 note 5f: 1.2 + 0.42 log10(Lw / 1000) from 400 to 2000 cd/m2, and the extended formula
    1.2 * 1.111 ** log2(Lw / 1000) outside that range."""
    if 400 <= lw <= 2000:
        return 1.2 + 0.42 * np.log10(lw / 1000)
    return 1.2 * 1.111 ** np.log2(lw / 1000)


def _luminance(rgb: np.ndarray, weights) -> np.ndarray:
    return np.asarray(rgb) @ np.asarray(weights, np.float64)


def hlg_ootf(e: np.ndarray, lw: float = HLG_NOMINAL_PEAK, weights=BT2100_LUMINANCE) -> np.ndarray:
    """Scene light RGB (..., 3) in [0, 1] to display light in cd/m2: F_D = alpha * Y_S ** (gamma - 1) * E, alpha = lw
    (BT.2100 Table 5, OOTF; black 0). `weights` are Y_S's, BT.2020's by the Recommendation."""
    e = np.maximum(np.asarray(e, np.float64), 0)
    ys = np.maximum(_luminance(e, weights), 1e-12)[..., None]
    return lw * np.power(ys, hlg_system_gamma(lw) - 1) * e


def hlg_inverse_ootf(fd: np.ndarray, lw: float = HLG_NOMINAL_PEAK, weights=BT2100_LUMINANCE) -> np.ndarray:
    """Display light in cd/m2 (..., 3) to scene light (BT.2100 note 5i)."""
    fd = np.maximum(np.asarray(fd, np.float64), 0)
    gamma = hlg_system_gamma(lw)
    yd = np.maximum(_luminance(fd, weights), 1e-12)[..., None]
    return fd / lw * np.power(yd / lw, (1 - gamma) / gamma)


def hlg_eotf(v: np.ndarray, lw: float = HLG_NOMINAL_PEAK, lb: float = 0.0, weights=BT2100_LUMINANCE) -> np.ndarray:
    """HLG signal (..., 3) to display light in cd/m2: OOTF[OETF^-1[max(0, (1 - beta) E' + beta)]],
    beta = sqrt(3 (lb / lw) ** (1 / gamma)) (BT.2100 Table 5, EOTF)."""
    beta = np.sqrt(3 * (lb / lw) ** (1 / hlg_system_gamma(lw)))
    return hlg_ootf(hlg_inverse_oetf(np.maximum(0, (1 - beta) * np.asarray(v, np.float64) + beta)), lw, weights)


def hlg_to_relative(v: np.ndarray, ootf: bool = True, lw: float = HLG_NOMINAL_PEAK, white: float = REFERENCE_WHITE,
                    weights=BT2100_LUMINANCE) -> np.ndarray:
    """HLG signal to linear light with reference white as 1.0. With the OOTF (BT.2100's display light, a `lw` display,
    black 0) white is `white` cd/m2, 75% HLG on a 1000 cd/m2 display; without it, scene light scaled so that 75%
    HLG (BT.2408's reference level) is 1.0."""
    if ootf:
        return hlg_ootf(hlg_inverse_oetf(v), lw, weights) / white
    return hlg_inverse_oetf(v) / float(hlg_inverse_oetf(HLG_REFERENCE_SIGNAL))


def hlg_peak(ootf: bool = True, lw: float = HLG_NOMINAL_PEAK, white: float = REFERENCE_WHITE) -> float:
    """HLG's signal 1.0 as relative light: 1000 / 203 = 4.93 with the OOTF, 1 / 0.265 = 3.77 without."""
    return float(hlg_to_relative(np.ones(3), ootf, lw, white)[0])


# ----- reference white for PQ -----

def measured_pq_white(luminance_nits: np.ndarray, white: float = REFERENCE_WHITE, factor: float = 2.0) -> float:
    """A measured rule, a candidate only: BT.2408's white, unless the picture's median luminance lies above `factor`
    times it, which BT.2408-graded light does not (its 18% grey card is 26 cd/m2, Table 1); then the file is taken to
    hold standard-range light with its white at PQ's top, 10,000 cd/m2, as darktable up to 5.6.2 writes it."""
    return PQ_PEAK if float(np.median(luminance_nits)) > factor * white else white


# ----- the operators: linear Rec.2020, 1.0 = reference white, to [0, 1] -----

def _by_max_rgb(x: np.ndarray, curve) -> np.ndarray:
    """Apply a curve on max(R, G, B) and scale R, G and B alike, so hue and saturation are kept."""
    x = np.maximum(np.asarray(x, np.float64), 0)
    m = x.max(axis=-1, keepdims=True)
    safe = np.maximum(m, 1e-12)
    return np.clip(x * (curve(safe) / safe), 0, 1)


def clip_at_white(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """Reference white to white; light above it clipped per channel. `peak` is not used."""
    return np.clip(np.asarray(x, np.float64), 0, 1)


def eetf_knee(peak: float, white: float = REFERENCE_WHITE) -> float:
    """Where eetf_bt2408 starts to roll off, as relative light: BT.2408-8 Annex 5 step 2, KS = 1.5 maxLum - 0.5,
    taken back through PQ. 1.0 (no roll-off) when the peak is at or below white."""
    if peak <= 1:
        return 1.0
    lo, hi = pq_inverse_eotf(0.0), pq_inverse_eotf(peak * white)
    ks = 1.5 * (pq_inverse_eotf(white) - lo) / (hi - lo) - 0.5
    return float(pq_eotf(ks * (hi - lo) + lo) / white)


def eetf_bt2408(x: np.ndarray, peak: float, white: float = REFERENCE_WHITE) -> np.ndarray:
    """BT.2408-8 Annex 5 (BT.2390-11, 5.4, points there): mastering black 0 and peak `peak` x `white` cd/m2, target
    black 0 and white `white`; steps 1-5 in the PQ domain, a 1:1 mapping below the knee (eetf_knee) and a Hermite
    spline above it that ends at the target white. Applied to max(R, G, B) (option 5: no colour leaves the gamut).
    With the peak at or below white it is clip_at_white."""
    if peak <= 1:
        return clip_at_white(x)
    lb = pq_inverse_eotf(0.0)
    span = pq_inverse_eotf(peak * white) - lb
    max_lum = (pq_inverse_eotf(white) - lb) / span
    min_lum = 0.0                                               # target black 0: (1 - E2) ** 4 adds nothing
    ks = 1.5 * max_lum - 0.5

    def curve(m: np.ndarray) -> np.ndarray:
        e1 = np.clip((pq_inverse_eotf(m * white) - lb) / span, 0, 1)                    # step 1
        t = (e1 - ks) / (1 - ks)
        p = ((2 * t**3 - 3 * t**2 + 1) * ks + (t**3 - 2 * t**2 + t) * (1 - ks)
             + (-2 * t**3 + 3 * t**2) * max_lum)                                       # step 4
        e2 = np.where(e1 < ks, e1, p)                                                  # step 3
        e3 = e2 + min_lum * (1 - e2) ** 4
        return pq_eotf(e3 * span + lb) / white                                         # step 5

    return _by_max_rgb(x, curve)


def scale_by_peak(x: np.ndarray, peak: float) -> np.ndarray:
    """The peak to white: every light divided by `peak` (at least 1), then clipped at 0 and 1."""
    return np.clip(np.asarray(x, np.float64) / max(peak, 1.0), 0, 1)


def reinhard_max_rgb(x: np.ndarray, peak: float) -> np.ndarray:
    """libultrahdr's globalTonemap (lib/src/jpegr.cpp, ReinhardMap): on m = max(R, G, B),
    m' = m (1 + m / peak**2) / (1 + m), which takes the peak to 1 and reference white to (1 + 1 / peak**2) / 2,
    about half; R, G and B scaled alike. libultrahdr's peak is the format's: 10000 / 203 for PQ, 1000 / 203 for HLG."""
    h2 = max(peak, 1.0) ** 2
    return _by_max_rgb(x, lambda m: m * (1 + m / h2) / (1 + m))


OPERATORS = {"clip": clip_at_white, "eetf": eetf_bt2408, "scale": scale_by_peak, "reinhard": reinhard_max_rgb}

__all__ = ["BT2100_LUMINANCE", "HLG_NOMINAL_PEAK", "HLG_REFERENCE_SIGNAL", "OPERATORS", "PQ_PEAK", "REFERENCE_WHITE",
           "clip_at_white", "eetf_bt2408", "eetf_knee", "hlg_eotf", "hlg_inverse_oetf", "hlg_inverse_ootf",
           "hlg_oetf", "hlg_ootf", "hlg_peak", "hlg_system_gamma", "hlg_to_relative", "measured_pq_white",
           "pq_eotf", "pq_inverse_eotf", "pq_to_relative", "reinhard_max_rgb", "scale_by_peak"]
