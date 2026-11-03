# ML-19: OpenCV or Pillow for the over-60 MP to 50 MP downscale

Researched 2026-11-03 for [ML-19](../../docs/ticket_canonicalization.html) (D27: a photo over 60 MP is downscaled,
aspect kept, to at most 50 MP; in the dataset 16320x12240 to 8164x6123, factor 1.999, and 10584x7936 to 8165x6122,
factor 1.296). Versions: Pillow 12.3.0, opencv-python-headless 5.0.0.93 (OpenCV 5.0.0, Intel IPP ICV 2026.0.0,
pthreads), NumPy 2.5.3, Python 3.12.16, torch 2.14.0, torchvision 0.29.1. The measurements ran on an AMD Ryzen 5
5600U (6 cores, 12 threads, AVX2, no AVX-512). Scripts lived in `/tmp` and are reproduced below. "Inferred" marks a
claim read from source but not tested; "unverified" marks one not checked at all.

## Answer

**Use Pillow `Image.resize(size, Image.Resampling.LANCZOS)`. OpenCV is a reasonable choice only with
`cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)`. No other OpenCV flag is.**

- **Pillow LANCZOS is the only one of these filters that is both antialiased and sharp.** Its kernel widens with the
  downscale factor (support 3 x factor), so it low-passes to the new Nyquist. At the dataset's factor 1.998 it lets
  through -20.2 dB of the content above the new Nyquist, against -13.6 dB for `INTER_AREA`. It keeps more detail
  just below that limit: -1.7 dB at 0.8 x Nyquist, against -3.1 dB. On a natural-like 1/f texture its PSNR against
  an ideal low-pass is 37.9 dB, against 33.7 dB. It is also the filter the earlier 7-photo model check used
  (believed), so that evidence still holds.
- **`INTER_AREA` is OpenCV's only downscale filter that antialiases.** It is an exact pixel-area (box) average,
  fractional edge pixels included. It leaks more than Lanczos and is softer near Nyquist, but it is 5 to 9 times
  faster (multi-threaded) and needs only the output's memory. It takes `uint8`, `uint16` and `float32` RGB directly,
  and the canonical source is 16-bit (D36). Its result did not change with thread count or with IPP switched on
  or off.
- **`INTER_LANCZOS4`, `INTER_CUBIC` and `INTER_LINEAR` must not be used to shrink.** Their kernels are fixed at 8, 4
  and 2 source pixels, so at factor 2 they alias almost everything above the new Nyquist: -1.1 to -4.7 dB leak.
  `INTER_LINEAR` turns into `INTER_AREA` only at a factor of exactly 2.000, and 16320 to 8164 is not exactly 2.
- **The two choices differ less in practice than these numbers suggest.** The 7 photos had 0.005-0.49% of their
  (mean-removed) power above the new Nyquist (README.md here). Rough estimate: the aliased power is at most about
  0.005% of the image's AC power with Pillow LANCZOS (-43 dB) and about 0.02-0.05% with `INTER_AREA` (-37 to
  -33 dB). With `INTER_LANCZOS4` it would be about 0.38% (-24 dB). If the owner picks OpenCV, re-run the 7-photo
  check (`run_infer.py`) on `INTER_AREA` output before relying on it.
- **Pillow's costs:**
  - It is single-threaded: 2.3 s for 200 MP RGB, against 0.26 s for `INTER_AREA` on 12 threads or 1.2 s on 1.
  - It holds a full intermediate image: +572 MiB at 200 MP, against +146 MiB for OpenCV.
  - It cannot hold 16-bit RGB. 16-bit data must be resized one band at a time (mode `I;16` or `F`): 5.3 s and
    +872 MiB at 200 MP.

## 1. Pillow `Image.resize` / `Image.reduce`

- **Filters.**
  - Support, in source pixels: BOX 0.5, BILINEAR 1, HAMMING 1, BICUBIC 2 (a = -0.5), LANCZOS 3 (sinc windowed by
    sinc(x/3)) [P1 L13-87].
  - The docs rate downscaling quality BOX 1 star, BILINEAR 1, HAMMING 2, BICUBIC 3, LANCZOS 4 [P6 L221-240].
  - `resize` defaults to BICUBIC [P2 L2366-2367].
- **Antialiasing.**
  - `precompute_coeffs` sets `filterscale = max(1, in/out)` and `support = filter.support * filterscale`, then
    samples the kernel at `(x - center + 0.5) / filterscale` [P1 L183-262].
  - So the kernel stretches with the downscale factor. At factor 2, LANCZOS takes about 13 taps per axis, and the
    taps are normalized to sum to 1. This is a true low-pass resample.
  - The pass is separable: horizontal first into a full-size intermediate image (`ImagingNewDirty(mode, xsize,
    rows)`), then vertical [P1 L806-860].
  - Measured: the intermediate is the extra memory. At 200 MP, 8164 x 12240 x 4 B + the output = +572 MiB.
- **Pixel-centre convention.** `center = in0 + (xx + 0.5) * scale` [P1 L233], the same as OpenCV's
  `(dx + 0.5) * scale - 0.5` [C1 L4110].
- **Precision.**
  - 8-bit modes: coefficients in double, converted to fixed point with `PRECISION_BITS = 22`, accumulated in `int`,
    rounded and clipped per pass [P1 L93, L270-285, L288-390].
  - The intermediate image is 8-bit, so the result is rounded twice (inferred from the same lines).
  - Pillow stores RGB at 4 bytes per pixel: a 200 MP `Image` is 800 MB, against 600 MB for the NumPy array
    (measured: the +572 MiB above matches 4 B/px).
  - `I;16`: accumulated in double, `CLIP16(ROUND_UP(ss))` [P1 L492-585].
  - `I` and `F`: accumulated in double [P1 L587-700].
- **16-bit RGB.**
  - The docs: "Pillow doesn't yet support multichannel images with a depth of more than 8 bits per channel"; `I;16`
    is single-channel [P6 L58-61, L69-70].
  - Measured: `Image.fromarray` of an HxWx3 `uint16` or `float32` array raises `TypeError: Cannot handle this data
    type`.
  - `I;16`, `I` and `F` single bands resize with LANCZOS (measured). The route for 16-bit RGB is three band resizes
    and a stack. clean-fid does the same with `F` bands [W3 L50-58].
- **`reducing_gap`.**
  - First runs `reduce()` by `int(in/out/reducing_gap)` when that is at least 2, then resizes the remainder; the
    default is `None`, off [P2 L2352-2363, L2414-2430].
  - `reducing_gap` must be at least 1.0 [P2 L2393-2395]. For factors of 2 or less (1.999 and 1.296 here),
    `int(factor / reducing_gap)` is at most 1, so it never applies. It is a speed option for thumbnails
    (`thumbnail()` defaults to 2.0 [P2 L2833]).
- **`Image.reduce(factor)`.**
  - An integer box average; the output size rounds up [P2 L2441-2455].
  - It refuses `I;16` (`ImagingError_ModeError` [P3 L1459-1460]; measured: `ValueError: image has wrong mode`).
- **Threads.**
  - No threading in `Resample.c`. Each pass releases the GIL (`ImagingSectionEnter` -> `PyEval_SaveThread`
    [P1 L303, L407; P4 L208-215]).
  - So one resize uses one core. Python threads can run separate images, or bands, in parallel (inferred).

## 2. OpenCV `cv2.resize`

- **Filters.**
  - `INTER_LINEAR`: 2 taps. `INTER_CUBIC`: 4 taps, A = -0.75 [C1 L965-973].
  - `INTER_LANCZOS4`: "Lanczos interpolation over 8x8 neighborhood" [C2 L260-261], always 8 taps
    (`ksize = 8` [C1 L4085-4090]).
  - The sampling position is `(dx + 0.5) * scale - 0.5` and the kernel is evaluated on the fractional offset only
    [C1 L4110-4141]. The kernel is never widened with the scale, so these three do not antialias when shrinking.
    clean-fid's abstract calls this "a fixed-width prefilter, resulting in aliasing artifacts" [W1].
- **`INTER_AREA`.**
  - When both scales are at least 1 [C1 L4023-4024]:
    - Integer factor (`is_area_fast`): `resizeAreaFast_`, a plain k x k mean.
    - Otherwise: `resizeArea_` with tables from `computeResizeAreaTab`, where each output pixel is the
      overlap-weighted mean of the source pixels its footprint covers, partial edge pixels weighted by their
      fraction [C1 L3343-3383, L4025-4074].
  - That is an exact box prefilter of width = factor. When upscaling it falls back to a bilinear variant (comment at
    L4023-4024).
  - The docs call it "a preferred method for image decimation, as it gives moire'-free results" [C2 L256-259] and
    say "To shrink an image, it will generally look best with INTER_AREA" [C2 L2078-2080]. Measured: it reduces
    aliasing but does not remove it (-10 to -14 dB).
- **`INTER_LINEAR` at exactly 2x.** Remapped to `INTER_AREA` [C1 L4018-4021]. At 1.998 it stays bilinear, and the
  measured leak goes from -10.0 dB to -4.7 dB.
- **`INTER_LINEAR_EXACT`.**
  - Fixed-point bilinear (`resize_bitExact`) for 8U, 8S, 16U, 16S and 32S [C1 L910-950, L3962-3999].
  - At exactly 2x (and cn != 2) it becomes `INTER_AREA` [C1 L3985-3990].
  - For 32F/64F it falls back to `INTER_LINEAR` [C1 L4232-4233].
  - It has bit-exact tests (`test_resize_bitexact.cpp` [C6]). It does not antialias.
- **Precision.**
  - 8U `INTER_LINEAR`/`CUBIC`/`LANCZOS4`: 11-bit fixed-point coefficients (`INTER_RESIZE_COEF_BITS = 11`
    [C1 L962]).
  - 16U, 16S and 32F: float.
  - `INTER_AREA` non-integer: float accumulation for `uchar`, `ushort` and `float` (`area_tab` [C1 L3955-3960]).
- **Dtypes.** Measured, HxWx3: `uint8`, `uint16` and `float32` all work with AREA, LANCZOS4, CUBIC, LINEAR and
  LINEAR_EXACT. The output keeps the input dtype.
- **IPP.**
  - `hal::resize` first tries `cv_hal_resize`. The wheel's custom HAL is IPP, but its HAL has no resize [C3]. It
    then tries `ipp_resize` [C1 L3849-3856].
  - With `useIPP_NotExact() == False` (the default, measured), `ipp_resize` refuses three cases: `ippSuper`
    (= `INTER_AREA`), `ippNearest`, and 8U `ippLinear` [C1 L3750-3755; mapping C4 L96-105].
  - OpenCV asks IPP for 4 Lanczos lobes (`IwiResizeParams(0, 0, 0.75, 4)`, i.e. antialiasing 0, B 0, C 0.75,
    lobes 4 [C1 L3654; C5]), while IW documents only 2 or 3 [C5b L160]. So IPP likely fails and OpenCV's own
    code runs (inferred).
  - Measured, IPP on vs off, max |diff|:

    | dtype | AREA | LANCZOS4 | CUBIC | LINEAR |
    |---|---|---|---|---|
    | `uint8` | 0 | 0 | 1 | 0 |
    | `uint16` | 0 | 0 | 7 | 6 |
    | `float32` | 0 | 0 | 1e-4 | 9e-5 |

  - So IPP serves CUBIC (all dtypes) and LINEAR (16U/32F). It never serves AREA or LANCZOS4.
- **Threads.** `parallel_for_` over output rows (pthreads backend in this wheel); `cv2.setNumThreads(n)` controls it.
  Measured: 12 threads vs 1 gave identical output for every flag.
- **Same output on every CPU?**
  - `resize.cpp` is not one of imgproc's CPU-dispatched files [C7 L3-15]. Its universal-intrinsic code is built
    for the wheel's baseline, SSE3 (`cv2.getBuildInformation()`).
  - Runtime dispatch in resize covers only nearest-neighbour (AVX2/SSE4.1) and the 16U Lanczos vertical pass
    (SSE4.1) [C1 L1139-1160, L1520-1530].
  - So `INTER_AREA` runs the same instructions on any x86-64 CPU for a given wheel (inferred; not tested on a
    second CPU).
  - IPP-served paths (CUBIC, 16U/32F LINEAR) dispatch inside IPP by CPU. This machine reports
    `ippIP AVX2 (l9)`; an AVX-512 machine would run another code path, possibly with other last bits (unverified).
  - On aarch64 builds, the KleidiCV HAL is on by default (`WITH_KLEIDICV` [C8 L176]) and may take over some
    resizes (unverified).
  - Only `INTER_LINEAR_EXACT` and `INTER_NEAREST_EXACT` are documented as bit-exact [C2 L262-266].

## 3. What the CV literature and libraries say

- **Parmar, Zhang, Zhu, "On Aliased Resizing and Surprising Subtleties in GAN Evaluation"** (CVPR 2022; arXiv
  2104.11222, v1 titled "On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation") [W1].
  - "Signal processing principles mandate adjusting prefilter width depending on the downsampling factor ...
    commonly-used implementations use a fixed-width prefilter, resulting in aliasing artifacts. Such aliasing leads
    to corruptions in the feature extraction downstream."
  - Their README: images resized with bicubic from OpenCV, PyTorch and TensorFlow have FID of 6 or more against
    PIL bicubic. PIL's Lanczos, bilinear and box score 0.75 or less [W2 L64].
  - clean-fid's reference resizer is PIL bicubic on float `F` bands [W3 L11-21, L41-58].
- **torchvision.**
  - `antialias=True` became the default in v0.17, "for the PIL and Tensor backends to be consistent". PIL images
    "are always antialiased on bilinear or bicubic" [W4; torchvision 0.29.1 `Resize` docstring].
  - `torch.nn.functional.interpolate(antialias=True, align_corners=False)` "would match Pillow result for
    downsampling" [W5; torch 2.14.0 docstring].
- **scikit-image.** `resize` applies a Gaussian prefilter by default when shrinking, sigma = (s - 1)/2, because
  "It is crucial to filter when downsampling the image to avoid aliasing artifacts" [W6 L123-135, L163-204].
  - Measured, the same prefilter in OpenCV before `INTER_AREA`: leak -17.2 dB at 1.998, but softer than Pillow
    LANCZOS (1/f PSNR 32.9 vs 37.9 dB). At 1.32 sigma is 0.16, so it changes nothing.
- **Consensus across these sources:** before a model, downscale with a filter whose width scales with the factor:
  PIL Lanczos or bicubic, torch/torchvision with `antialias=True`, or OpenCV `INTER_AREA`. Never use a fixed-tap
  interpolator to shrink.

## 4. Experiment

**Quality.**

- Synthetic 2048x2048 `uint8` grey images, downscaled to 1024 (factor 2.000), 1025 (1.998, the dataset's
  16320 to 8164) and 1552 (1.32). Ideal reference: FFT-domain brick-wall low-pass at the new Nyquist,
  pixel-centre aligned, on an even-symmetric extension; 16 px border excluded.
- "Leak" is the output RMS divided by the input RMS for content that the ideal removes entirely: noise with power
  only between 1.05 x and 2 x the new Nyquist (at 1.32, up to 1.32 x), or the zone-plate region at those
  frequencies. Lower is better; 0 dB means all of it folds back.
- The PSNRs compare against the ideal output:
  - band-limited noise below 0.9 x new Nyquist, a passband test (sharpness, ringing);
  - 1/f-amplitude noise, natural-image-like;
  - a full-band zone plate.

Factor 1.998 (the 200 MP photos):

| method | leak, noise (dB) | leak, zone plate (dB) | PSNR passband noise (dB) | PSNR 1/f texture (dB) | PSNR zone plate (dB) |
|---|---|---|---|---|---|
| PIL LANCZOS | **-20.2** | **-18.9** | 35.5 | **37.9** | **24.8** |
| PIL BICUBIC | -18.1 | -17.2 | 31.1 | 35.6 | 22.9 |
| PIL HAMMING | -12.3 | -12.3 | 30.1 | 33.6 | 20.3 |
| PIL BOX | -9.9 | -9.8 | 27.8 | 31.0 | 17.1 |
| cv2 INTER_AREA | -13.6 | -14.4 | 29.7 | 33.7 | 21.0 |
| cv2 INTER_LANCZOS4 | -1.1 | -2.0 | 55.3 | 27.3 | 12.0 |
| cv2 INTER_CUBIC | -2.0 | -3.0 | 48.5 | 28.0 | 12.9 |
| cv2 INTER_LINEAR (= LINEAR_EXACT) | -4.7 | -5.3 | 36.4 | 30.4 | 15.1 |
| cv2 GaussianBlur((s-1)/2) + INTER_AREA | -17.2 | -17.6 | 27.2 | 32.9 | 20.7 |

Factor 2.000 exactly. Same as 1.998 except:

| method | leak, noise (dB) | leak, zone plate (dB) | PSNR passband noise (dB) | PSNR 1/f texture (dB) | PSNR zone plate (dB) |
|---|---|---|---|---|---|
| PIL BOX = cv2 INTER_AREA = cv2 INTER_LINEAR = LINEAR_EXACT | -10.0 | -9.8 | 33.4 | 32.9 | 19.0 |
| cv2 INTER_LANCZOS4 | -2.4 | -2.4 | 53.4 | 27.6 | 12.3 |
| PIL LANCZOS | -20.2 | -18.7 | 35.6 | 37.4 | 24.8 |

Factor 1.32 (the 84 MP photo is 1.296):

| method | leak, noise (dB) | leak, zone plate (dB) | PSNR passband noise (dB) | PSNR 1/f texture (dB) | PSNR zone plate (dB) |
|---|---|---|---|---|---|
| PIL LANCZOS | **-13.7** | **-13.0** | 35.6 | **37.7** | **21.3** |
| PIL BICUBIC | -11.6 | -11.2 | 31.1 | 35.5 | 19.4 |
| PIL HAMMING | -6.4 | -6.3 | 29.5 | 33.0 | 16.9 |
| PIL BOX | -2.3 | -2.2 | 27.8 | 30.1 | 13.8 |
| cv2 INTER_AREA | -9.8 | -9.5 | 28.2 | 33.2 | 17.7 |
| cv2 INTER_LANCZOS4 | -2.3 | -2.2 | 55.1 | 32.4 | 14.9 |
| cv2 INTER_CUBIC | -3.2 | -3.1 | 40.9 | 33.1 | 15.8 |
| cv2 INTER_LINEAR (= LINEAR_EXACT) | -6.0 | -5.9 | 29.8 | 33.0 | 16.8 |

**Response to a sinusoid** (output/input contrast, dB) at 0.5, 0.8 and 0.95 x new Nyquist, `mtf.py`:

| method | factor 1.998 | factor 1.32 |
|---|---|---|
| PIL LANCZOS | +0.10, -1.68, -4.57 | +0.09, -1.69, -4.58 |
| PIL BICUBIC | -0.54, -2.89, -5.18 | -0.55, -2.90, -5.18 |
| cv2 INTER_AREA | -1.16, -3.07, -4.43 | -1.43, -3.78, -5.44 |
| cv2 INTER_LANCZOS4 | -0.03, -0.03, +0.01 | -0.04, +0.08, -0.23 |
| cv2 INTER_LINEAR | -0.43, -1.12, -1.58 | -1.04, -2.71, -3.88 |

How to read the tables:

- The cv2 LANCZOS4 and CUBIC rows have the best passband PSNR only because they filter nothing. The same lack of
  filtering gives them the worst leak.
- The PSNR against the ideal on 1/f texture combines both effects. On it, Pillow LANCZOS beats `INTER_AREA` by
  4.2-4.5 dB.

**Time and peak extra memory.**

- One downscale of a random RGB image, best of 2 runs, each in a fresh process.
- "Extra" is the rise of peak RSS over the RSS with the input already in memory; the input itself is 600 MB (NumPy
  `uint8`) or 800 MB (Pillow RGB).
- Pillow is single-threaded; cv2 ran on 12 threads unless noted.
- `Image.fromarray` costs another 0.42 s (200 MP) or 0.10 s (84 MP).

| method | 16320x12240 to 8164x6123 | 10584x7936 to 8165x6122 |
|---|---|---|
| PIL LANCZOS, `uint8` | 2.34 s, +572 MiB | 1.30 s, +438 MiB |
| PIL BICUBIC, `uint8` | 1.79 s, +572 MiB | 0.99 s, +437 MiB |
| PIL BOX, `uint8` | 0.80 s, +571 MiB | 0.47 s, +437 MiB |
| cv2 INTER_AREA, `uint8` | 0.26 s, +146 MiB | 0.10 s, +145 MiB |
| cv2 INTER_AREA, `uint8`, 1 thread | 1.18 s, +144 MiB | 0.55 s, +143 MiB |
| cv2 INTER_LANCZOS4, `uint8` | 0.46 s, +153 MiB | 0.23 s, +153 MiB |
| cv2 INTER_LANCZOS4, `uint8`, 1 thread | 2.19 s, +145 MiB | 1.28 s, +145 MiB |
| cv2 INTER_CUBIC, `uint8` | 0.18 s, +149 MiB | 0.04 s, +149 MiB |
| cv2 INTER_LINEAR, `uint8` | 0.20 s, +146 MiB | 0.04 s, +145 MiB |
| cv2 INTER_AREA, `uint16` | 0.17 s, +289 MiB | 0.10 s, +289 MiB |
| cv2 INTER_AREA, `uint16`, 1 thread | 0.81 s, +286 MiB | 0.51 s, +287 MiB |
| cv2 INTER_AREA, `float32` | 0.24 s, +574 MiB | 0.14 s, +575 MiB |
| PIL LANCZOS, `uint16`, 3 `I;16` bands | 5.29 s, +872 MiB | 3.04 s, +572 MiB |
| PIL LANCZOS, `uint16`, 3 `F` bands | 8.63 s, +1909 MiB | 3.99 s, +1334 MiB |

**Scripts as run** (Python with `-I`; `dtypes.py`, the dtype / IPP / thread checks quoted in section 2, is not
reproduced).

`quality.py`:

```python
import numpy as np, cv2
from PIL import Image
N = 2048
rng = np.random.default_rng(0)
fy = np.fft.fftfreq(N)[:, None]; fx = np.fft.fftfreq(N)[None, :]
fmax = np.maximum(abs(fx), abs(fy))                       # separable band edge, cycles/px

def from_spectrum(mask_amp, std=30.0):
    z = np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * mask_amp).real
    return np.clip(128 + z * std / z.std(), 0, 255).round().astype(np.uint8)

def zone_plate():
    y, x = np.mgrid[0:N, 0:N] - (N - 1) / 2
    k = np.pi * 0.5 / (N / 2)                             # local freq reaches 0.5 cyc/px at r = N/2
    return np.clip(127.5 + 127.5 * np.cos(k * (x**2 + y**2)), 0, 255).round().astype(np.uint8), k

def ideal(img, n):                                        # FFT brick-wall resample, pixel-centre aligned
    a = img.astype(np.float64)
    a = np.concatenate([a, a[::-1]], 0); a = np.concatenate([a, a[:, ::-1]], 1)
    M, m = 2 * N, 2 * n; d = 0.5 * M / m - 0.5
    k = np.fft.fftfreq(M) * M; ph = np.exp(2j * np.pi * k * d / M)
    F = np.fft.fft2(a) * ph[:, None] * ph[None, :]
    idx = np.where(np.abs(k) < m / 2)[0]; kk = k[idx].astype(int) % m
    G = np.zeros((m, m), complex); G[np.ix_(kk, kk)] = F[np.ix_(idx, idx)]
    return (np.fft.ifft2(G).real * (m * m) / (M * M))[:n, :n]

P = Image.Resampling
pil = lambda f: lambda a, n: np.asarray(Image.fromarray(a).resize((n, n), f))
cv = lambda f: lambda a, n: cv2.resize(a, (n, n), interpolation=f)
METHODS = {"PIL LANCZOS": pil(P.LANCZOS), "PIL BICUBIC": pil(P.BICUBIC), "PIL HAMMING": pil(P.HAMMING),
           "PIL BOX": pil(P.BOX), "cv2 INTER_AREA": cv(cv2.INTER_AREA), "cv2 INTER_LANCZOS4": cv(cv2.INTER_LANCZOS4),
           "cv2 INTER_CUBIC": cv(cv2.INTER_CUBIC), "cv2 INTER_LINEAR": cv(cv2.INTER_LINEAR),
           "cv2 INTER_LINEAR_EXACT": cv(cv2.INTER_LINEAR_EXACT),
           "cv2 GaussianBlur((s-1)/2) + INTER_AREA": lambda a, n: cv2.resize(
               cv2.GaussianBlur(a, (0, 0), (N / n - 1) / 2), (n, n), interpolation=cv2.INTER_AREA)}
psnr = lambda a, b: 10 * np.log10(255**2 / np.mean((a.astype(np.float64) - b) ** 2))

zp, zk = zone_plate()
for n in (1024, 1025, 1552):
    nq = 0.5 * n / N; sl = (slice(16, n - 16),) * 2
    stop = from_spectrum(((fmax > 1.05 * nq) & (fmax < 0.5)).astype(float))
    passb = from_spectrum((fmax < 0.9 * nq).astype(float))
    f = np.hypot(fx, fy); f[0, 0] = 1
    pink = from_spectrum(1 / f, std=40)
    refs = {"pass": ideal(passb, n), "pink": ideal(pink, n), "zone": ideal(zp, n)}
    c = (np.arange(n) + 0.5) * N / n - 0.5 - (N - 1) / 2
    lf = np.maximum(abs(c)[:, None], abs(c)[None, :]) * zk / np.pi
    zstop = (lf > 1.05 * nq) & (lf < 0.5)                 # zone-plate pixels whose content the ideal removes
    for name, fn in METHODS.items():
        o = {k: fn(v, n) for k, v in (("stop", stop), ("pass", passb), ("pink", pink), ("zone", zp))}
        leak = 20 * np.log10(o["stop"][sl].astype(float).std() / stop.astype(float).std())
        zleak = 20 * np.log10((o["zone"] - refs["zone"])[zstop].std() / (127.5 / np.sqrt(2)))
        print(name, n, round(leak, 1), round(zleak, 1), *(round(psnr(o[k][sl], refs[k][sl]), 1)
                                                           for k in ("pass", "pink", "zone")))
```

`mtf.py` (core):

```python
for frac in (0.5, 0.8, 0.95):
    f = frac * 0.5 / s                                    # cycles per source pixel, s = 2048 / n
    a = np.tile(np.round(128 + 100 * np.cos(2 * np.pi * f * np.arange(2048))).astype(np.uint8), (64, 1))
    o = resize(a, (n, 64))[32].astype(float)[32:-32]      # each method as above
    xo = ((np.arange(n) + 0.5) * s - 0.5)[32:-32]
    A = np.c_[np.cos(2 * np.pi * f * xo), np.sin(2 * np.pi * f * xo), np.ones_like(xo)]
    c = np.linalg.lstsq(A, o, rcond=None)[0]; print(20 * np.log10(np.hypot(c[0], c[1]) / 100))
```

`perf.py METHOD W H w h DTYPE THREADS`, one process per run:

```python
cv2.setNumThreads(thr)
rss = lambda k: int(next(l for l in open("/proc/self/status") if l.startswith(k)).split()[1]) // 1024
a = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)   # then .astype(uint16)*257 or float32/255
src = Image.fromarray(a) if (PIL and uint8) else a
open("/proc/self/clear_refs", "w").write("5")                               # reset VmHWM
base = rss("VmRSS"); t = time.perf_counter()
out = src.resize((w, h), filt)                                              # PIL uint8
# PIL 16-bit: np.dstack([np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c])).resize((w, h), filt)) for c in range(3)])
#   ('F' variant: the band cast to float32 first)
# OpenCV:     cv2.resize(a, (w, h), interpolation=flag)
print(time.perf_counter() - t, rss("VmHWM") - base)
```

## 5. Bottom line for this pipeline

- **Default: Pillow `resize((w, h), Image.Resampling.LANCZOS)` on the 8-bit RGB image.** If the downscale runs on
  the 16-bit source (D36), resize each band as `I;16` and stack the bands.
  - It is the best of these filters on both aliasing and detail.
  - It keeps the earlier 7-photo evidence valid.
  - Its cost is 2.3 s of one core and about 0.6 GB beyond the input for a 200 MP photo. That is irrelevant to the
    dataset run (7 photos). On the 500 MB front, D38 refuses such photos above a measured ceiling in any case.
- **If the owner prefers OpenCV: `cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)`, with `(w, h)` computed
  from the 50 MP rule and passed explicitly.**
  - It is a principled antialiasing (area) filter and needs only the output's memory.
  - It takes 3-channel `uint16` and `float32` natively and is fast on any core count.
  - It gives the same output for any thread count and does not go through IPP.
  - The price is about 6 dB more leak and 1.4-2 dB less contrast at 0.8 x Nyquist than Pillow LANCZOS. Re-run the
    7-photo model check on its output.
  - Never use `INTER_LANCZOS4`, `INTER_CUBIC`, `INTER_LINEAR` or `INTER_LINEAR_EXACT` for this step.
- **Not in scope, worth knowing:** neither library linearizes. Both filter the values they are given, whether
  gamma-encoded or linear (no transfer function anywhere in [P1] or [C1]). Downscaling before or after the colour
  conversion changes the result slightly; this was not measured.
- **Cross-machine identity:** D25 asks that the encoder give the same output on the front, the inference host and
  the development machine. The downscale before the encoder needs the same check.
  - For `INTER_AREA` on x86-64 this is expected from source but untested on a second CPU.
  - Pillow's 8-bit path is integer arithmetic after double-precision coefficients (expected identical; untested).

## Sources

Pillow 12.3.0, tag `12.3.0`:

- [P1] `src/libImaging/Resample.c`: https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
  (lines as cited, e.g. #L83-L87, #L183-L262, #L806-L860)
- [P2] `src/PIL/Image.py`: https://github.com/python-pillow/Pillow/blob/12.3.0/src/PIL/Image.py#L2329-L2455
- [P3] `src/libImaging/Reduce.c`: https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Reduce.c#L1451-L1460
- [P4] `src/_imaging.c`: https://github.com/python-pillow/Pillow/blob/12.3.0/src/_imaging.c#L208-L215
- [P6] `docs/handbook/concepts.rst` (modes, filters, comparison table):
  https://github.com/python-pillow/Pillow/blob/12.3.0/docs/handbook/concepts.rst#L24-L240, rendered at
  https://pillow.readthedocs.io/en/stable/handbook/concepts.html

OpenCV 5.0.0, tag `5.0.0`:

- [C1] `modules/imgproc/src/resize.cpp`: https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/resize.cpp
  (e.g. #L962-L1000, #L3343-L3383, #L3639-L3830, #L3835-L4254)
- [C2] `modules/imgproc/include/opencv2/imgproc.hpp`:
  https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/include/opencv2/imgproc.hpp#L250-L268 and #L2060-L2106.
  These are the sources of the `resize` and `InterpolationFlags` pages on docs.opencv.org, which refused automated
  requests (HTTP 403) and were not read.
- [C3] `hal/ipp/include/ipp_hal_imgproc.hpp` (no resize entry):
  https://github.com/opencv/opencv/blob/5.0.0/hal/ipp/include/ipp_hal_imgproc.hpp
- [C4] `modules/imgproc/src/precomp.hpp`, `ippiGetInterpolation`:
  https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/precomp.hpp#L96-L105
- [C5] IPP IW, `iw++/iw_image_transform.hpp` L166 (constructor order: antialiasing, cubicB, cubicC, lanczosLobes), and
  [C5b] `iw/iw_image_transform.h` L160-161 ("Possible values are 2 or 3"). Both are in
  `ippicv_2026.0.0_lnx_intel64_20260327_general.tgz`, the package pinned (md5 `9a3ee0c5c3c02102faa422d60bfd1f4a`,
  checked) by https://github.com/opencv/opencv/blob/5.0.0/3rdparty/ippicv/ippicv.cmake and fetched from
  https://github.com/opencv/opencv_3rdparty/tree/406d398c436d0465c8e53dd432d9ecd9301d5f4a/ippicv
- [C6] https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/test/test_resize_bitexact.cpp
- [C7] https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/CMakeLists.txt#L3-L15
- [C8] https://github.com/opencv/opencv/blob/5.0.0/CMakeLists.txt#L176

Other:

- [W1] Parmar, Zhang, Zhu, arXiv 2104.11222: https://arxiv.org/abs/2104.11222 (v1 title:
  https://arxiv.org/abs/2104.11222v1); project page https://www.cs.cmu.edu/~clean-fid/
- [W2] clean-fid README: https://github.com/GaParmar/clean-fid/blob/e88c4d6269a4bbf04c04deeb578475b57719acee/README.md#L56-L66
- [W3] clean-fid resizer: https://github.com/GaParmar/clean-fid/blob/e88c4d6269a4bbf04c04deeb578475b57719acee/cleanfid/resize.py#L11-L95
- [W4] torchvision `Resize`, antialias:
  https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html (read from the installed
  0.29.1 docstring)
- [W5] `torch.nn.functional.interpolate`, antialias:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html (read from the installed
  2.14.0 docstring)
- [W6] scikit-image `resize`: https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/transform/_warps.py#L123-L204
- Photo power above the new Nyquist and the 7-photo model check: [README.md](README.md),
  [downscale_answers.json](downscale_answers.json) in this directory.
