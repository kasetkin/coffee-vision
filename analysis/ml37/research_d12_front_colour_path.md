# ML-37 D12: the front's colour path to a linear Rec.2020 half-float canonical photo

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D12 (and R3), 2026-11-23, on the local
checkout at `01e6ac1` (branch `ml37-linear-canonical`). A recommendation for the owner, not a decision. The prototype
is `d12_proto.py` (research code: nothing in `coffeecv/` imports it, nothing there changed); the numbers come from
`d12_front_cost.py` (`d12_front_cost.jsonl`), `d12_accuracy.py` (`d12_accuracy.txt`) and `d12_encode_error.py`
(`d12_encode_error.txt`), all in this directory. The timings were taken on a shared machine under load (load average
5.4 to 15.4 during the main run, 3.5 to 3.9 during the re-check, §2); CPU seconds are given beside wall seconds, and
the re-check at lower load agrees with ML-29 R6's idle numbers for today's path.

## Summary

- **Every candidate fits the front's budget with room to spare, and every one beats today's path on gain-map photos.**
  Today's path (integer, or float under a gain map, then the peak pass, the EETF and the BT.709 curve) takes 11.7 to
  15.5 s and 402 to 498 MB on the largest gain-map photos at load 3.7; the store's float path written as linear
  float16 through a chunked encoder takes 4.4 to 7.1 s and 312 to 396 MB on the same photos. Against 40 s (ML-19 D22)
  and 600M (ML-19 D38), (a)'s worst on any photo is 10.6 s (loaded) and 433 MB (the DNG).
- **A chunked float16 encoder is needed for memory.** Handing libjxl the whole float16 picture
  (`imagecodecs.jpegxl_encode`) peaks at 597 MB on the 12.5 MP DNG and 456 to 479 MB on the 15 MP plain photos,
  against 285 to 433 MB chunked. `JxlEncoderAddChunkedFrame` with `JXL_TYPE_FLOAT16` writes the same bytes as the
  whole-picture encode on all seven photos.
- **Faster paths.** LittleCMS with float output is the float path (same bytes, same time): lcms2 never optimizes a
  transform with a float side. LittleCMS's optimized 8-bit matrix-shaper into uint16 is 8 to 11x faster but quantizes
  linear light to 1/16384 and clips at 1.0 (up to 0.8 levels before the encode). **Per-channel tables (the LUT path)
  are 7 to 10x faster than the float path, within 0.14 sRGB levels of it before the encode, mean under 0.0001**, and
  byte-identical on the DNG and the ceiling photo.
- **The file grows only where light above white is now kept**: +4.5 to +9.0% on gain-map photos (0.3 to 0.7 s more
  on the 2.1 Mbit/s link), unchanged on the plain JPEG and the DNG.
- **Recommended: (a), the store's float path for every upload, with a chunked float16 encoder and the gain-map light
  held in float16.** It fits the budget, the front's colour samples become the store's (narrowing ADR 0036's recorded
  departure to the encode settings and the cap), and the integer path goes. Its one cost: +2.4 s on a 15 MP plain
  JPEG against today (5.7 s against 3.3 s, loaded). If the owner wants that back, the LUT path is a measured (b)
  within a stated tolerance (§6).

## 1. What the front does today and what has to change

The front canonicalizes with `FRONT_SETTINGS` (`coffeecv/canonical.py:119`: colour "integer", effort 3, quality 90,
over 18 MP area-averaged to at most 15 MP). For an integer source with no built profile and no alpha, `to_rec2020`
takes `_to_rec2020_integer` (`canonical.py:536-537`, `:601`), LittleCMS's optimized transform from the source's own
8/16-bit samples to uint16 Rec.2020 with the BT.709 curve; any other source, and every gain-map photo (ADR 0041),
takes the float path. Before the encode, the tone map's peak pass reads the whole picture (ML-29). The chunked
encoder `coffeecv/jxl_encode.encode_rec2020` hard-codes `JXL_TYPE_UINT16` (`jxl_encode.py:175`) and the BT.709
transfer (`:242-245`).

ML-37 D1 makes the canonical photo linear Rec.2020 half-float, 1.0 = white, light above white kept; D6 moves the tone
map to the read-back. So the front needs (i) a colour path that ends in linear float, (ii) an encoder that takes
float16, and (iii) no peak pass.

## 2. Time and peak memory (`d12_front_cost.py`)

Each (photo, variant) in a fresh process under `taskset -c 0,1` (the front's 2 vCPU, as `analysis/ml29/p11_front_cost.py`
does), `FRONT_SETTINGS`' cap, quality and effort; peak memory is `VmHWM` (MiB, as the check service logs it). The
photos are ML-29 R6's four gain-map photos and its 60 MP ceiling file, plus the dataset's largest plain JPEG and the
largest DNG under the front's 16 MP RAW ceiling. Variants:

- **today**: `canonical.canonicalize_with(FRONT_SETTINGS)` steps (decode, peak pass, encode).
- **(a) whole / (a) chunked**: the store's float path (`canonical._light`, then LittleCMS on float32 into linear
  Rec.2020), clipped below 0, light above 1 kept, float16; encoded whole by `imagecodecs.jpegxl_encode`, or by the
  prototype's chunked float16 encoder (`d12_proto.encode_chunked`, a copy of `encode_rec2020` with
  `JXL_TYPE_FLOAT16`, 5 exponent bits, linear transfer).
- **(a) chunked32**: (a) chunked with the gain-map light held in float32, as `canonical._with_gain_map` holds it.
- **lcms16 / lcmsf / lut**: the candidate faster colour paths of §3, all chunked, gain-map light in float16.

None of the new variants runs a peak pass. The intensity target is left at libjxl's default (D8 is open).

Main run, 2026-11-23 13:24 to 13:38, best wall of 2 repeats (CPU seconds in brackets), largest peak of the two:

| Photo (decoded size) | today s / MB | (a) whole MB | (a) chunked s / MB | (a) chunked32 MB | lcms16 s / MB | lut s / MB | load |
|---|---|---|---|---|---|---|---|
| Ultra HDR JPEG, Pixel, 12 MP (3072 x 4080) | 24.6 [25.0] / 403 | 400 | 9.1 [10.2] / 309 | 383 | 4.3 [6.6] / 312 | 4.7 [6.7] / 314 | 12.5-15.4 |
| Ultra HDR JPEG, Pixel 9 Pro, 50 MP (3360 x 4463) | 26.1 [27.4] / 432 | 437 | 9.3 [12.3] / 343 | 430 | 6.0 [8.9] / 345 | 6.0 [8.8] / 347 | 12.1-13.4 |
| ISO 21496-1 JPEG, Galaxy, 50 MP (3354 x 4472) | 25.5 [28.2] / 454 | 438 | 8.7 [11.6] / 345 | 429 | 5.8 [8.8] / 343 | 5.3 [8.3] / 345 | 11.1-11.9 |
| ISO 21496-1 HEIC, iPhone, 48 MP (3354 x 4472) | 22.6 [26.5] / 495 | 479 | 10.6 [14.7] / 394 | 484 | 6.2 [10.2] / 394 | 6.5 [10.5] / 403 | 8.8-10.5 |
| Ultra HDR JPEG at the ceiling, 60 MP (4000 x 3750; map over the ceiling, unapplied) | 3.5 [6.1] / 285 | 456 | 5.9 [8.4] / 285 | - | 3.4 [6.0] / 286 | 3.4 [5.9] / 285 | 7.5-9.3 |
| Plain JPEG, 48 MP (4472 x 3354) | 3.3 [5.9] / 297 | 479 | 5.7 [8.4] / 308 | - | 3.6 [6.3] / 307 | 3.6 [6.3] / 302 | 6.0-7.5 |
| DNG, Pixel 9 Pro, 12.5 MP (3072 x 4080) | 5.7 [8.3] / 435 | 597 | 5.6 [8.2] / 433 | - | 5.3 [8.0] / 435 | 3.9 [6.5] / 454 | 5.4-8.7 |

lcmsf (LittleCMS straight to float32) is not in the table: it took (a) chunked's time within the noise on every photo
(5.6 to 9.2 s) and wrote (a)'s bytes (§3). Today's peak pass alone took 3.8 to 5.1 s on the four gain-map photos.

Re-check at lower load, 2026-11-23 17:15, one run each (`tmp/ml37/d12/front_cost_lowload.jsonl`, not committed):

| Photo | today s (peak pass) / MB | (a) chunked s / MB | lut s / MB | lcms16 s / MB | load |
|---|---|---|---|---|---|
| Ultra HDR JPEG, Pixel, 12 MP | 11.7 (2.1) / 402 | 4.4 / 312 | 2.8 / 313 | 2.8 / 311 | 3.5-3.7 |
| ISO 21496-1 HEIC, iPhone, 48 MP | 15.5 (2.5) / 498 | 7.1 / 396 | 5.1 / 397 | 4.8 / 396 | 3.7-3.9 |

Today's numbers at load 3.7 (11.7 and 15.5 s) match ML-29 R6's idle ones (11.61 and 14.78 s, 392 and 487 MB,
`analysis/ml29/README.md` R6), so the main run's walls are inflated by the load, roughly 2x on today's path, less on
the new ones; the ordering holds in both runs and in CPU seconds.

What the numbers say:

- **Gain-map photos get faster and smaller under every candidate.** Today's decode builds float32 light, the peak pass
  reads it, and the encode runs the EETF and the BT.709 curve per strip (7.5 to 8.7 s of "encode" at load 3.7). The
  new paths write the light once into float16 and encode slices of it (2.0 to 2.6 s). Float16 light instead of
  float32 saves 74 to 90 MB (chunked32 against chunked) and changes no byte (§4).
- **Photos without a gain map pay for the float colour step under (a)**: +2.4 s on the 15 MP plain JPEG and the
  ceiling photo against today's integer path; the LUT path and lcms16 take today's time. The DNG already took the
  float path's cost today (its 16-bit input gets no LittleCMS matrix-shaper optimization, §3), and the LUT path is
  the only one faster there.
- **Memory**: (a) chunked holds each photo at or under today's peak (the LUT path is 19 MB over it on the DNG); the
  whole-picture encode adds about 170 MB on the 15 MP photos without a map, and on the DNG reaches 597 MiB, 3 MiB under `MemoryMax=600M` (systemd's M is MiB),
  which also counts page cache and kernel memory that `VmHWM` does not. **A chunked float16 encoder is needed.**

## 3. Candidate faster paths, and what LittleCMS does

`d12_accuracy.py` runs every path on the same 256-row strips of each decoded photo (gain map applied the same way),
reads each strip's float16 samples back as ML-37 D2 does with D6 (b) (linear Rec.2020, 3x3 matrix to linear sRGB,
clip to [0, 1], sRGB curve, float 8-bit levels; `d12_proto.readback_levels`), and compares them with the float path's
(`d12_accuracy.txt`, 2026-11-23 17:12, load 1.9, CPUs 0-1). Colour step alone, one thread:

| Photo (profile) | float (a) s/MP | lcms16 s/MP, max / mean levels | lcmsf s/MP, max | lut s/MP, max / mean levels |
|---|---|---|---|---|
| Ultra HDR 12 MP (Display P3) | 0.166 | 0.016, 0.44 / 0.013 | 0.173, 0 | 0.023, 0.11 / < 0.0001 |
| Ultra HDR 50 MP (Display P3) | 0.188 | 0.017, 0.80 / 0.012 | 0.176, 0 | 0.022, 0.14 / < 0.0001 |
| ISO JPEG 50 MP (DCI-P3 D65, sRGB transfer) | 0.170 | 0.022, 0.39 / 0.018 | 0.176, 0 | 0.021, 0.13 / < 0.0001 |
| ISO HEIC 48 MP (Display P3) | 0.169 | 0.014, 0.45 / 0.016 | 0.167, 0 | 0.017, 0.14 / < 0.0001 |
| Ceiling 60 MP (Display P3, sRGB transfer) | 0.161 | 0.014, 0.09 / 0.007 | 0.160, 0 | 0.016, 0 / 0 |
| Plain JPEG 48 MP (none: sRGB) | 0.150 | 0.013, 0.26 / 0.024 | 0.151, 0 | 0.016, 0.12 / < 0.0001 |
| DNG 12.5 MP (built-in RGB, 16-bit) | 0.125 | 0.123, 0 / 0 | 0.123, 0 | 0.018, 0 / 0 |

No path put any sample more than 0.5 level from the float path.

**lcmsf: LittleCMS with float output is the float path.** `AllocEmptyTransform` gives any transform with a float input
or output format `FloatXFORM` (lcms2 2.19 `src/cmsxform.c:917-938`), and each lossy optimization refuses float formats
(`src/cmsopt.c:663` `OptimizeByResampling`, `:1062` `OptimizeByComputingLinearization`, "This is a lossy optimization!
does not apply in floating-point cases", `:1410` `OptimizeByJoiningCurves`). Measured: same bytes as (a), same speed.

**lcms16: LittleCMS's optimized transform into linear uint16.** For 8-bit RGB input, `OptimizeMatrixShaper` (only on
8-bit input, `cmsopt.c:1681-1694`) replaces the pipeline with 256-entry input shapers into 1.14 fixed point, a fixed-
point matrix, and a 16,385-entry output shaper, clamping the matrix result to [0, 16384] (`cmsopt.c:73`, `:79-90`,
`:1557-1569`). So linear output is quantized to 1/16384 (about 0.2 sRGB levels per step near black) and **cannot
carry anything above 1.0**: harmless for P3 or sRGB sources (inside Rec.2020), but a source wider than Rec.2020 (a
ProPhoto export, say) would lose light the float path keeps, contrary to D1. On 16-bit input it gets no matrix-shaper
optimization (the DNG: same bytes and speed as the float path; optimized and `NOOPTIMIZE` outputs identical). Against
`NOOPTIMIZE`, its uint16 output differed by at most 5 to 7 of 65,535 on the 8-bit photos.

**lut: per-channel tables, the float path's own numbers.** For a matrix-shaper profile the transform is
out = M (f_r(r), f_g(g), f_b(b)), which is a sum of per-channel terms: out = T(0) + L_r[r] + L_g[g] + L_b[b], each
table built once per (profile, bit depth) by running the float path on a ramp of every code (`d12_proto.lut_for`;
3 x 3 x 256 floats for 8-bit, 3 x 3 x 65,536 for 16-bit, 2.4 MB). It is checked when built on 4,096 random colours
against the float path; the largest error was 6.0e-8 in linear light (0 on the DNG), against a tolerance of 1e-5, and
a profile that fails the check (a LUT-based ICC profile, where the transform is not a sum) takes the float path.
Float sources (PQ, HLG, linear, float JPEG XL) take the float path as today (ADR 0036 amendment). The remaining
difference (at most 0.14 level, mean under 0.0001) is float32 summation order and float16 rounding flips.

## 4. After the encode, and the encode's own error for scale

Every new canonical photo decodes to `RGB D65 BT2100 Linear Relative, 16-bit float, lossy (XYB)`. Byte identity
(`d12_accuracy.txt`): (a) whole = (a) chunked on all seven photos; (a) chunked32 = (a) chunked on all four gain-map
photos; lcmsf = (a) on all seven; lut = (a) on the ceiling photo and the DNG.

Read back against (a) chunked after the encode, a tiny input difference moves libjxl's decisions locally:

| Photo | lcms16 max / mean / > 1 level | lut max / mean / > 1 level |
|---|---|---|
| Ultra HDR 12 MP | 19.6 / 0.027 / 0.66% | 1.1 / < 0.0001 / 0.0001% |
| Ultra HDR 50 MP | 12.2 / 0.034 / 0.94% | 2.6 / < 0.0001 / 0.0013% |
| ISO JPEG 50 MP | 24.8 / 0.043 / 1.06% | 3.3 / < 0.0001 / 0.0006% |
| ISO HEIC 48 MP | 43.3 / 0.050 / 1.59% | 2.0 / < 0.0001 / 0.0010% |
| Ceiling 60 MP | 0.8 / 0.001 / 0% | identical bytes |
| Plain JPEG 48 MP | 41.4 / 0.065 / 1.77% | 2.5 / < 0.0001 / 0.0005% |
| DNG 12.5 MP | identical bytes | identical bytes |

For scale, the lossy encode itself (quality 90, effort 3) against its own float16 input (`d12_encode_error.py`,
2026-11-23 17:17, load 2.9): mean 0.87 to 2.02 levels, largest 30 to 191, 32 to 65% of samples over 1 level
(12 MP Ultra HDR 0.87 / 30; HEIC 1.58 / 116; ISO JPEG 1.35 / 64; plain JPEG 1.93 / 80; DNG 2.02 / 191). The LUT
path's difference is four orders of magnitude under the encode's mean; lcms16's mean is about 1 to 1.5% of it. Whether the
encode's own error on linear input is larger than on today's BT.709 integers is ML-37 R2's question, not this one.

## 5. File size on the 2.1 Mbit/s link

| Photo | today | linear float16 ((a) and lut) | change | at 2.1 Mbit/s |
|---|---|---|---|---|
| Ultra HDR 12 MP | 1,247,489 B | 1,327,850 B | +6.4% | +0.31 s (5.06 s) |
| Ultra HDR 50 MP | 1,763,337 B | 1,842,549 B | +4.5% | +0.30 s (7.02 s) |
| ISO JPEG 50 MP | 1,037,439 B | 1,130,398 B | +9.0% | +0.35 s (4.31 s) |
| ISO HEIC 48 MP | 2,120,184 B | 2,297,106 B | +8.3% | +0.67 s (8.75 s) |
| Plain JPEG 48 MP | 1,785,765 B | 1,785,533 B | 0.0% | 0 (6.80 s) |
| DNG 12.5 MP | 1,305,647 B | 1,305,694 B | 0.0% | 0 (4.97 s) |

(The ceiling photo is a synthetic gradient of 10.7 KB; its size says nothing.) The growth is the light above white
that today's EETF compresses and D1 keeps (5 to 23% of these photos' pixels are above white, `d12_accuracy.txt`).
The intensity target was left at libjxl's default; D8 may move these sizes.

## 6. Recommendation

**(a): the store's float path for every upload, written through a chunked float16 encoder, gain-map light in float16,
no peak pass.** Reasons:

1. It fits: at most 10.6 s (loaded) and 433 MB, against 40 s and 600M; at load 3.7 the largest gain-map photo takes
   7.1 s against today's 15.5 s.
2. The front's samples before the encode become the store's: the departure ADR 0036 records narrows to the encode's
   quality and effort and the cap, and the front loses its second colour path (`_to_rec2020_integer`), which cannot
   write linear float as it is.
3. The faster paths buy 1.6 to 2.4 s per photo at moderate load (up to 4 s in the loaded run), a fraction of the
   4 to 9 s the canonical photo then spends on the link.

What it needs: `encode_rec2020` (or a sibling) taking `JXL_TYPE_FLOAT16` with 5 exponent bits and the linear transfer,
which the prototype shows writes the whole-picture encode's bytes (§4); libjxl recommends float input for HDR and wide
gamut (`lib/include/jxl/types.h:41-44`) and documents `JxlEncoderAddChunkedFrame` for images that may not fit in
memory (`lib/include/jxl/encode.h:960-983`); imagecodecs' whole-picture float16 encode (`_jpegxl.pyx:368-371`) is
the 597 MB path to avoid.

**If the owner prefers the time, (b) is the LUT path, with this tolerance**: per-channel tables built through the float
path, held only when the additivity check on 4,096 random colours is within 1e-5 in linear light, else the float
path; measured within 0.5 read-back sRGB level of (a) on every sample before the encode (largest 0.14, mean under
0.0001 on the seven photos). It brings a plain 15 MP JPEG back to today's 3.6 s and saves 1.6 to 2.4 s elsewhere,
for about 40 lines and a table cache. **lcms16 is not recommended**: its 1/16384 quantization puts samples up to 0.8 level
from (a), about 6x the LUT's largest difference, and its clamp at 1.0 breaks D1 for sources wider than Rec.2020. **lcmsf buys nothing.**

## 7. What stays unverified

- Timings on the front's own 2 vCPU host: these are this machine's, held to 2 CPUs, under load (main run load
  5.4 to 15.4; re-check 3.5 to 3.9; only two photos re-checked). Peak memory is load-independent.
- Sources wider than Rec.2020 and LUT-based ICC profiles: none among the measured photos (all matrix-shaper P3, sRGB
  or the DNG's built-in RGB); the LUT's fallback and lcms16's clamp are read from the code, not measured.
- Uploads with alpha (the prototype takes RGB only) and float JPEG XL uploads (float path either way).
- The cgroup's view of memory (`MemoryMax` counts more than `VmHWM`); the margins of §2 are on `VmHWM`.
- The intensity target (D8) and its effect on size and the encode's error (R2).

## Files

- `d12_proto.py`: the four colour paths, the float16 gain-map decode, the chunked float16 encoder, the read-back.
- `d12_front_cost.py` -> `d12_front_cost.jsonl`: time and peak memory, 2 CPUs, fresh process per run.
- `d12_accuracy.py` -> `d12_accuracy.txt`: path differences before and after the encode.
- `d12_encode_error.py` -> `d12_encode_error.txt`: the encode's own error, for scale.
- Scratch, not committed: `tmp/ml37/d12/out/` (canonical photos), `tmp/ml37/d12/ceiling_60mp.jpg` (ML-29 p11's
  `make_ceiling`), `tmp/ml37/d12/front_cost_lowload.jsonl`.

## Sources

- LittleCMS 2.19 source (release tarball, `tmp/ml37/d12/downloads/lcms2-2.19/`): `src/cmsxform.c:855`, `:912-938`
  (float transforms are `FloatXFORM`); `src/cmsopt.c:73`, `:79-90`, `:1540-1569` (matrix-shaper 1.14 fixed point,
  16,385-entry output shaper, clamp to [0, 16384]), `:663`, `:1062`, `:1072`, `:1410` (lossy optimizations refuse
  float formats; 16-bit pre-linearization only on request), `:1681-1694` (matrix-shaper only on 8-bit RGB input),
  `:1961` (`cmsFLAGS_NOOPTIMIZE`).
- libjxl 0.12.0 source (`tmp/ml37/d12/downloads/libjxl-0.12.0/`): `lib/include/jxl/types.h:41-44`, `:57`
  (`JXL_TYPE_FLOAT16 = 5`); `lib/include/jxl/encode.h:854` (`JxlChunkedFrameInputSource`), `:960-983`
  (`JxlEncoderAddChunkedFrame`).
- imagecodecs 2026.10.10 source (`tmp/ml37/d12/downloads/imagecodecs-2026.10.10/`): `imagecodecs/_jpegxl.pyx:368-371`
  (float16 arrays encoded as `JXL_TYPE_FLOAT16`, 5 exponent bits), `:556` (whole frame via `JxlEncoderAddImageFrame`).
- This repo at `01e6ac1`: `coffeecv/canonical.py:119`, `:522-537`, `:601`; `coffeecv/jxl_encode.py:175`, `:242-245`;
  `docs/adr/0036-the-front-makes-its-own-canonical-photo.md`; `analysis/ml29/README.md` (R6);
  `analysis/ml29/p11_front_cost.py`; ML-19 D22 (40 s), D38 (600M); ML-15 (the 2.1 Mbit/s link).
- Measurements: this directory's `d12_*.jsonl` / `.txt`, 2026-11-23.
