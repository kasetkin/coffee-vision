# ML-37: the canonical photo as linear uint16 (I) or linear half-float (H), against today's

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) (D1, D8, D12, R2), 2026-11-25, on the
local checkout at `3df80ac` (branch `ml37-linear-canonical`). A recommendation for the owner, not a decision. The
owner's question: now that the models take 8-bit sRGB rounded once at the read-back, should the canonical photo be
**(I)** lossy JPEG XL, Rec.2020 primaries, D65, linear transfer, **uint16**, with light above white tone-mapped at
canonicalization as today (ML-29 D33's EETF, ADR 0041), values in [0, 1]; or **(H)** the same with **float16** samples
(D1: 1.0 = reference white, light above white kept)? Baseline: **today's** uint16 Rec.2020 with the BT.709 curve.
Every number comes from the four `canon_int_*.py` scripts beside this note and their printed outputs
(`canon_int_*.txt`); the libjxl source was read at 0.12.0 in `tmp/ml37/d8/downloads/` (D8's download, not committed).
Nothing under `coffeecv/`, `docs/`, `data/` or `models/` was changed.

## Summary

- **Yes: linear uint16 compresses exactly as well as half-float, in size and in error, shadows included.** On D8's 20
  dataset photos at the same content (the tone-mapped picture) and the same intensity target, (I)'s file is 1.000×
  (H)'s per photo (range 0.999–1.001) at the store's q95/e7 and at the front's q90/e3. Its read-back, rounded once to
  uint8, has the same pixel RMS (1.854 levels in the store, 2.770 on the front, at 203; ratio 1.000), the same
  32×32 block-mean error (0.074 vs 0.073 and 0.128 vs 0.128) and the same share of 8-bit values that differ from the
  uncompressed picture's (67.7% and 77.5%, identical to three digits) (§3).
- **(I) is not worse in the shadows.** In dark pixels (reference sRGB below 0.1) the error is 1.836 vs 1.836 levels
  RMS (store, 203) and 2.490 vs 2.488 (front); in the darkest tenth of blocks 0.076 vs 0.076 and 0.134 vs 0.135. Per
  photo, (I)/(H) is 1.000 (0.997–1.007) in dark pixels and 0.99–1.00 (0.94–1.03) in the darkest blocks (§3). Linear
  uint16 does put fewer codes near black, and that is visible **before** coding: uint16's own rounding gives 0.156
  levels RMS in dark pixels against half-float's 0.107. But that is a twelfth of the coding error there (1.84), and
  libjxl turns both inputs into the same float picture before XYB (§1), so it is gone after the encode.
- **After rounding to 8 bits once, neither is distinguishable from today's encode at the same intensity target.**
  (I)/today: bytes 1.000, pixel RMS 1.000, block mean 1.00 (0.98–1.02), dark pixels 1.000, at both settings and both
  targets (§3). What does separate them from today's file is D8's intensity target, 203 against today's 255, which
  acts as a quality offset on every form alike: 4.5% / 6.1% smaller files and 6% / 9% more block-mean error
  (store / front; 0.069 → 0.074 and 0.118 → 0.128 levels, still far inside D27's mean ≤ 1.0).
- **Keeping light above white (H as D1 has it) costs bytes, not error.** Against (H) of the tone-mapped picture,
  D1's (H) is 1.022× the bytes in the store and 1.027× on the front over all photos, 1.044× / 1.055× on the ten
  gain-map photos (up to 1.14× / 1.15×), with the same read-back error (block mean 0.074 vs 0.073 at 203). Photos
  without light above white give identical files.
- **The pinned stack writes (I) two ways, with the same bytes.** imagecodecs 2026.10.10 `jpegxl_encode(uint16,
  primaries="BT2100", transfer="LINEAR")` and the repo's ctypes strip encoder with its transfer set to linear write
  byte-identical files (and imagecodecs' default transfer for uint16 with BT2100 primaries is already linear). The
  header reads 16-bit integer, BT2100, Linear, D65, XYB. imagecodecs cannot set the intensity target (it declares
  255); the ctypes encoder sets 203 with one field (§2). For (I), `coffeecv/jxl_encode.py` needs only its transfer
  enumeration and that field, not a float16 pixel type.
- **Time and memory are the same.** At full scale (4 photos, 12–17 MP, 2 threads) the encodes take 2.7–4.1 s in the
  store and 1.2–1.9 s on the front for every form, within 8% of each other; decodes 0.08–0.16 s; the decoded array
  is 6 bytes per pixel for uint16 and float16 alike (§4).
- **DINOv3's `cls_mean` feature does not tell them apart** (§5, 144 test patches of 24 photos, store setting): against
  the uncompressed picture rounded the same way, 1 − cosine has a median of 1.63e-5 for (I) and 1.78e-5 for (H) at
  203 (1.55e-5 and 1.60e-5 at 255; today's file 1.53e-5). (I) against (H) directly: median 8.1e-6, below the
  coding's own change and 70× below a one-pixel shift of the patch (5.8e-4, `research_bits_classifier.md` §3).

## 1. Why the forms code alike: what libjxl does with each input

With `uses_original_profile = 0` (both settings) libjxl codes XYB, built from linear sRGB, whatever the input:

- **Integer samples become float first.** `ConvertFromExternal` loads every uint8/uint16 row as float, scaled by
  `1 / (2^bits − 1)` (`lib/jxl/enc_external_image.cc` L75, L89). A uint16 file's samples are 0…1 floats from then
  on, and float16 samples are loaded as floats too.
- **Then one colour conversion to linear sRGB, then XYB.** `ToXYB` (`lib/jxl/enc_xyb.cc` L233–288) calls
  `ApplyColorTransform` from the declared encoding (BT2100 primaries with the linear or the BT.709 transfer) to
  linear sRGB, then `LinearSRGBToXYB`, scaled by `intensity_target / 255` (D8 §3). The quantizer, the adaptive
  quantization and the AC strategy all see that float XYB picture.

So (I), (H) of the same content and today's file present libjxl with the same float picture up to their own input
rounding: uint16 linear steps of 1/65,535, float16's 11-bit mantissa, and today's BT.709 uint16 steps. All three are
far below the XYB quantizer's steps at distance 0.55 (q95) and 1.0 (q90) (`JxlEncoderDistanceFromQuality`, D8 §3).
The size and error ratios of 1.000 in §3 are what that predicts.

**On the decode side the two differ in one way:** libjxl clamps integer output to [0, max] and rounds it
(`lib/jxl/render_pipeline/stage_write.cc` L273–285), while float output is not clamped (D7 §1). An (I) file decodes
with no negatives and nothing above 1.0; an (H) file decodes with the coder's small negatives and overshoots, which
the read-back's clip removes anyway (D2, D7). The measured errors include this.

## 2. Can the pinned stack write (I)? (`canon_int_probe.py`)

On a synthetic ramp (768×512, with noise and a dark corner) and on the first photo of D8's sample at 1/4 scale, at
q95/e7 and q90/e3 (`canon_int_probe.txt`):

| writer | bytes (photo, q95/e7) | header (`jxl_header.read_header`) | declared intensity target |
|---|---|---|---|
| imagecodecs `jpegxl_encode(uint16, primaries="BT2100", transfer="LINEAR")` | 151,763 | 16-bit int, RGB, D65, BT2100, Linear, Relative, XYB | 255 |
| imagecodecs, the same with `transfer` unset | identical bytes | the same | 255 |
| ctypes strip encoder, `JXL_TYPE_UINT16`, linear, intensity target 0 or 255 | identical bytes | the same | 255 |
| ctypes strip encoder, intensity target 203 | 137,238 | the same | 203 |

- **imagecodecs writes (I)** and decodes it back to uint16. Unset, the transfer is linear for anything but uint8 when
  primaries are given (`imagecodecs/_jpegxl.pyx` L427–446, imagecodecs 2026.10.10), so `transfer="LINEAR"` is
  explicit but not needed. It has no intensity-target parameter and never sets the field (D8 §2), so its file
  declares libjxl's default, 255.
- **The repo's encoder writes (I)** with the same libjxl calls as `encode_rec2020`, the transfer
  `JXL_TRANSFER_FUNCTION_LINEAR` (8) in place of `_709` (1) and `BasicInfo.intensity_target` set. Here
  `canon_int_measure.encode_linear_u16` does it by reusing D8's `d8_intensity_target.encode`. Its bytes equal
  imagecodecs' at the default target, as `encode_rec2020`'s do today (ML-19 D52).
- **The intensity target means the same for both forms.** For an integer file, libjxl's documentation calls it the
  brightness of the largest representable value (D8 §1), and 1.0 is (I)'s largest value. (I) tone-maps to [0, 1]
  with 1.0 = reference white, so 203 is right for (I) under both of D8's readings. For (H) only the code's reading
  holds (D8 §1).
- `jxlinfo` is not installed on this machine; the repo's header reader and libjxl's `JxlDecoderGetBasicInfo` stand in.

## 3. Size and the models' input error, 20 photos (`canon_int_measure.py`)

**Method.** D8's sample and harness, so the numbers line up with D8's note: 20 dataset originals (seed 0, stratified
by camera and gain map; 17 named camera models and one photo without a make, ten with a gain map applied, peaks 1.48–3.18 × white), each decoded as the
store does into D1's light and area-averaged 2× per side (1500–2752 px wide). (D8's note says eight gain-map photos;
the same sample's rows, in D8's own CSV too, flag ten.) From the light and its peak: `sdr` = clip(EETF at the peak, 0,
1). Forms: **today** = `sdr` through LittleCMS to the BT.709 curve, uint16 (D8's `today_u16`); **I** = rint(`sdr` ×
65535), uint16 linear; **Hs** = `sdr` as float16 linear (the same content as I); **Hh** = the light as float16
linear, above white kept (D1). Each encoded at intensity target 255 and 203, decoded with imagecodecs (uint16 or
float16, as stored) and read back as D2 says (matrix, clip, sRGB curve; the EETF at the source's peak for Hh; today's
file through LittleCMS to linear first), then **rounded once to uint8**. The reference is the uncompressed light
through the same read-back, rounded to uint8, one reference for all forms. Measures in 8-bit levels: pixel RMS; D27's
32×32 block means (mean, largest block); dark pixels (reference luma < 25.5, i.e. sRGB < 0.1; 0.2–33% of a photo's
pixels, median 6.5%): RMS; the darkest tenth of blocks by reference luma: block mean; the share of uint8 values that differ from
the reference's. Sizes are totals at half scale; bits per pixel per photo, averaged.

**All 20 photos** (means over photos; "block max" is the largest block over all photos):

| setting | form | IT | MB | bpp | size vs today (255) | pixel RMS | block mean | block max | darkest 10% blocks | dark px RMS | values differing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| store q95/e7 | today | 255 | 18.05 | 2.325 | 1.000 | 1.780 | 0.069 | 3.02 | 0.069 | 1.688 | 66.5% |
| | I | 255 | 18.05 | 2.325 | 1.000 | 1.780 | 0.070 | 2.76 | 0.069 | 1.689 | 66.5% |
| | Hs | 255 | 18.05 | 2.325 | 1.000 | 1.780 | 0.070 | 3.16 | 0.070 | 1.688 | 66.5% |
| | Hh | 255 | 18.34 | 2.363 | 1.022 | 1.777 | 0.070 | 3.16 | 0.070 | 1.690 | 66.1% |
| | today | 203 | 17.28 | 2.226 | 0.955 | 1.854 | 0.073 | 3.60 | 0.075 | 1.835 | 67.7% |
| | **I** | 203 | 17.28 | 2.225 | 0.955 | 1.854 | 0.074 | 3.60 | 0.076 | 1.836 | 67.7% |
| | **Hs** | 203 | 17.28 | 2.226 | 0.955 | 1.854 | 0.073 | 3.52 | 0.076 | 1.836 | 67.8% |
| | **Hh** (D1) | 203 | 17.56 | 2.262 | 0.976 | 1.849 | 0.074 | 3.52 | 0.076 | 1.835 | 67.4% |
| front q90/e3 | today | 255 | 11.44 | 1.476 | 1.000 | 2.633 | 0.118 | 6.11 | 0.118 | 2.282 | 76.7% |
| | I | 255 | 11.44 | 1.476 | 1.000 | 2.632 | 0.118 | 6.09 | 0.118 | 2.282 | 76.7% |
| | Hs | 255 | 11.44 | 1.476 | 1.000 | 2.632 | 0.118 | 6.08 | 0.118 | 2.280 | 76.7% |
| | Hh | 255 | 11.66 | 1.505 | 1.027 | 2.617 | 0.119 | 6.08 | 0.118 | 2.281 | 76.4% |
| | today | 203 | 10.78 | 1.391 | 0.939 | 2.770 | 0.128 | 5.95 | 0.134 | 2.491 | 77.5% |
| | **I** | 203 | 10.78 | 1.391 | 0.939 | 2.770 | 0.128 | 5.92 | 0.134 | 2.490 | 77.5% |
| | **Hs** | 203 | 10.78 | 1.391 | 0.939 | 2.770 | 0.128 | 6.08 | 0.135 | 2.488 | 77.5% |
| | **Hh** (D1) | 203 | 10.99 | 1.419 | 0.965 | 2.753 | 0.129 | 6.08 | 0.135 | 2.488 | 77.2% |
| uncoded (own rounding only) | today | – | | | | 0.054 | 0.003 | 0.30 | 0.008 | 0.115 | 0.4% |
| | I | – | | | | 0.070 | 0.004 | 0.29 | 0.013 | 0.156 | 0.6% |
| | Hs, Hh | – | | | | 0.138 | 0.014 | 0.31–0.44 | 0.009 | 0.107 | 2.1% |

**Per photo, (I) against the others at the same setting and target** (mean ratio, range; `canon_int_measure.txt`):

| (I) / … | bytes | pixel RMS | block mean | dark px RMS | darkest 10% blocks |
|---|---|---|---|---|---|
| Hs, store 203 | 1.000 (0.999–1.001) | 1.000 (0.999–1.001) | 0.999 (0.984–1.023) | 1.000 (0.997–1.003) | 0.996 (0.941–1.034) |
| Hs, front 203 | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.001 (0.997–1.011) | 1.001 (0.999–1.007) | 0.998 (0.975–1.018) |
| today, store 203 | 1.000 (1.000–1.000) | 1.000 | 1.000 (0.979–1.019) | 1.001 (0.999–1.005) | 1.005 (0.936–1.030) |
| today, front 203 | 1.000 (1.000–1.000) | 1.000 | 1.001 (0.990–1.014) | 1.000 (0.998–1.000) | 1.001 (0.990–1.019) |
| Hh, store 203 | 0.979 (0.881–1.003) | 1.004 (0.997–1.037) | 1.000 (0.952–1.077) | 1.001 (0.998–1.007) | 0.995 (0.938–1.037) |
| Hh, front 203 | 0.975 (0.870–1.000) | 1.007 (1.000–1.043) | 0.996 (0.889–1.051) | 1.001 (0.998–1.007) | 0.999 (0.975–1.018) |

At 255 the ratios are the same to within 0.01 (all rows in `canon_int_measure.txt`).

What the tables show:

- **(I) = (H) = today, at the same target.** The three forms of the same content give files equal to 0.1% and errors
  equal to the third digit, overall, in dark pixels and in the darkest blocks. The per-block and per-pixel ranges
  (±2–6% on single photos) go both ways and are the coder's own noise: two encodes of nearly the same float
  picture make different but equally large errors. The largest block differs between forms for the same reason
  (store 203: 3.60, 3.52, 3.60).
- **The shadows.** Uncoded, (I)'s own rounding is coarser near black than half-float's (0.156 vs 0.107 levels RMS in
  dark pixels; 2.9% vs 1.6% of dark values change at the uint8 step) and finer elsewhere (0.070 vs 0.138 overall:
  float16's steps near white are 1/2048 against uint16's 1/65,535). Coded, both are at 1.84 (store) and 2.49 (front)
  levels in dark pixels. The input rounding is 12–16× below the coding error in the shadows and adds to it in
  quadrature, under 0.5%.
- **Today's 255 against D8's 203.** (I) at 203 against today's file as written now (255), per photo: 0.955× the bytes
  and 1.060× the block mean (1.02–1.11) in the store; 0.939× and 1.093× (1.04–1.20) on the front. That is D8's
  finding, unchanged by the form. The worst photo's block mean is 0.129 (store) and 0.186 (front) at 203, against
  0.120 and 0.173 at 255, for every form. The largest block on the front is 5.92–6.11 in every row; it is D8's one
  front block, present in today's encode too.
- **Light above white.** Hh against Hs: identical on the ten photos without a gain map; 1.044× (store) and 1.055×
  (front) on the gain-map photos (range 1.00–1.14, 1.00–1.15), with the read-back error unchanged (§ the Hh rows).
  The bytes buy the light above white in the file, which the read-back then tone-maps away.
- **D8's numbers reproduce.** Hh's sizes (18.34 / 11.66 MB at 255, 17.56 / 10.99 at 203) and float block means
  before rounding (0.055 / 0.110 at 255, 0.060 / 0.121 at 203; column `blkF` in the output) are D8's §4 numbers to
  the last digit. Rounding once to uint8 adds about 0.014 to the store's block mean (0.055 → 0.069) and 0.009 to the
  front's (0.110 → 0.119).

## 4. Time and memory at full scale (`canon_int_cost.py`)

Four photos of the same sample at full size: the two gain-map photos with the highest peaks (iPhone 15 Pro, 3.2×;
Galaxy Z Fold6, 3.0×), the 17 MP Sony and an iPhone 13 without a map. Linear forms at intensity target 203, today's at
255; 2 threads per call (the front's), one call at a time; load average 3.3–3.7 on 12 CPUs (the 20-photo run had
just ended). Memory is the rise in the process's peak RSS over its RSS just before the call (VmHWM reset through
`/proc/self/clear_refs`), so the input picture is not counted.

| photo | MP | setting | today: bytes, enc s, dec s | I | Hs | Hh |
|---|---|---|---|---|---|---|
| iPhone 15 Pro, map 3.2× | 12.2 | store | 1,555,596; 2.70; 0.12 | 1,480,786; 2.80; 0.08 | 1,481,026; 2.76; 0.08 | 1,550,190; 2.75; 0.08 |
| | | front | 984,880; 1.24; 0.11 | 921,618; 1.24; 0.09 | 921,581; 1.30; 0.09 | 980,646; 1.27; 0.09 |
| Galaxy Z Fold6, map 3.0× | 12.0 | store | 2,551,072; 2.92; 0.12 | 2,471,187; 2.90; 0.10 | 2,470,377; 2.93; 0.10 | 2,636,742; 2.98; 0.10 |
| | | front | 1,518,189; 1.27; 0.12 | 1,388,363; 1.29; 0.11 | 1,388,085; 1.29; 0.10 | 1,509,054; 1.30; 0.10 |
| Sony G8441 | 17.0 | store | 2,966,583; 3.86; 0.15 | 2,845,802; 4.00; 0.13 | 2,845,144; 4.04; 0.13 | 2,845,144; 4.09; 0.14 |
| | | front | 1,671,125; 1.75; 0.16 | 1,569,935; 1.80; 0.14 | 1,570,010; 1.87; 0.14 | 1,570,010; 1.82; 0.13 |
| iPhone 13, no map | 12.2 | store | 1,542,548; 2.77; 0.10 | 1,444,280; 2.90; 0.08 | 1,444,533; 2.89; 0.08 | 1,444,533; 2.91; 0.08 |
| | | front | 907,664; 1.15; 0.11 | 827,223; 1.24; 0.09 | 827,311; 1.26; 0.09 | 827,311; 1.27; 0.09 |

- **Encode time** is the same for (I) and (H) (within 4%), and −1% to +8% against today's at 255. **Decode** is 0.08–0.16 s for all; the integer and float16 outputs take as long.
- **Memory.** The encoder's rise is 95–233 MB per call, with no consistent order by form across the four photos. The decode's rise is the output array, 70–99 MB, 6 bytes per
  pixel for uint16 and float16 alike. Since neither rise depends on the form, D12's float16 encoder numbers (R3) should carry over to (I); not
  re-measured with the front's path.
- **Bytes at full scale** follow §3: (I) = (Hs) to 0.03%; (I) at 203 is 0.94–0.97× today's file at 255 in the store
  and 0.91–0.94× on the front; Hh is +4.7–6.7% on the two gain-map photos.

## 5. DINOv3's feature (`canon_int_dino.py`)

**Method.** `bits_cls_noise.py`'s photo pick from exp290's test split (one per session in turn, crop at most 16 MP),
24 photos, exp290's 144 test boxes. The content is `pre`, the uint16 BT.709 picture today's store encode hands
libjxl (so already tone-mapped: (H) here is Hs, the same content as (I)), converted to linear Rec.2020. Each form is
encoded at q95/e7, decoded, read back the one way (matrix, clip, sRGB curve), the crop cut and its bean mask filled,
and each box made into the model's input with the 8-bit step once (`bits_cls_common.model_input`, "u8": rounded once
at the crop, resizes in float). DINOv3 ViT-B/16 `cls_mean`, float32 on CPU, 8 threads; 16 min, peak RSS 2.8 GB.

| 1 − cosine, 144 boxes | median | p95 | max |
|---|---|---|---|
| uncoded: u8 vs f32 (the 8-bit step alone) | 6.0e-7 | 7.2e-6 | 1.1e-4 |
| today (255) vs uncoded u8 | 1.53e-5 | 1.64e-4 | 5.1e-4 |
| today, IT 203 vs uncoded u8 | 1.75e-5 | 2.15e-4 | 8.3e-4 |
| **(I), 203** vs uncoded u8 | 1.63e-5 | 2.12e-4 | 7.2e-4 |
| **(H), 203** vs uncoded u8 | 1.78e-5 | 2.18e-4 | 8.0e-4 |
| (I), 255 vs uncoded u8 | 1.55e-5 | 1.80e-4 | 4.6e-4 |
| (H), 255 vs uncoded u8 | 1.60e-5 | 1.57e-4 | 4.9e-4 |
| **(I) vs (H)**, 203 | 8.1e-6 | 5.0e-5 | 7.8e-5 |
| (I) vs (H), 255 | 7.9e-6 | 4.8e-5 | 1.2e-4 |
| (I) vs today, 203 | 8.1e-6 | 5.3e-5 | 7.8e-5 |
| (I), 255 vs today | 7.4e-6 | 4.6e-5 | 1.2e-4 |
| today 203 vs today 255 | 1.66e-5 | 1.88e-4 | 3.5e-4 |

- (I) and (H) move the feature by the same amount from the uncompressed picture (medians within 10%, both at either
  target), and that amount is the store's coding error, 1.1e-5 in `research_bits_classifier.md` §3.
- (I) against (H) is half the coding's own change and the same as (I) against today's file: two encodes of the same
  float picture. Changing the intensity target (today 203 vs 255) moves the feature twice as much as changing the
  form. All of it is 70× below a one-pixel shift of the patch (median 5.8e-4, the same note §3).
- Not measured: the live head's answers on these features.

## 6. The answer

- **Does linear uint16 compress as well as half-float?** Yes, exactly: same bytes to 0.1%, same 8-bit error to the
  third digit, at both settings and both intensity targets (§3). libjxl turns both into the same float XYB picture
  (§1).
- **Is it worse in the shadows?** No, after coding. Its own rounding is coarser near black than float16's (0.156 vs
  0.107 levels RMS in dark pixels, before any coding), but the lossy coding error there is 1.84–2.49 levels and the
  two forms' dark errors agree to 0.1% (§3).
- **Is either distinguishable from today's after rounding to 8 bits once?** Not at the same intensity target: (I)
  and (H) match today's file in size, pixel error, block error and dark error (ratios 1.000). The only difference a
  move to either brings is D8's intensity target (203 vs 255), a quality offset of the same size for both (−4.5% /
  −6.1% bytes, +6% / +9% block error), and, for (H) as D1 has it, +4–6% bytes on gain-map photos for the light above
  white.
- **So the choice between (I) and (H) is not about compression or the models' input.** It is about what the file
  keeps: (I) gives up the light above white at canonicalization, as today (ADR 0041), and keeps the canonicalization
  EETF and its peak pass; (H) keeps that light (D1), costs 4–6% more bytes on gain-map photos and moves the tone map
  to the read-back (D6). (I) needs the smaller change to `coffeecv/jxl_encode.py` (the transfer and the intensity
  target; uint16 stays) and decodes clamped to [0, 1]: uint16 holds no negatives, so D7's option (b), keeping them, does not exist under (I).
  (H) is what D12's research prototyped. Those trade-offs are the owner's, and nothing measured here favours one form
  on quality.

## 7. What this does not measure

- **Full-resolution error.** §3 is at half scale, like D8. Full-scale bytes (§4) follow the same ratios on four photos;
  the error ratios at full scale were not measured.
- **The front's own colour path.** All forms here start from the store's float decode (D8's `light_of`). The front's
  integer path, and whether it can make (I)'s linear uint16 as cheaply as today's BT.709 uint16 (it quantizes linear
  light, D12 §3), is D12's question and was not re-measured.
- **The models beyond the classifier's feature.** No head was refitted and the segmenter was not run. §5 measures
  DINOv3's `cls_mean` feature only, on the store's setting, on SDR content (no Hh), with no head's answers.
- **Efforts 8 and above**, other qualities, other libjxl versions than 0.12.0, and `jxlinfo` itself (not installed).
- **What a viewer shows.** Both forms declare a linear transfer; how browsers and HDR viewers render them is D8 §6's
  and ML-34's ground, not re-checked here.
- **Statistics.** 20 photos; the ratios are steady per photo (§3), but no confidence intervals were computed.

## Files

Scripts and their printed outputs, in `analysis/ml37/` (run with `cd /workspace && PYTHONPATH=/workspace uv run
python analysis/ml37/<script>`):

- `canon_int_probe.py` → `canon_int_probe.txt`: can imagecodecs and the ctypes encoder write (I); headers, targets.
- `canon_int_measure.py` → `canon_int_measure.txt`: §3, 20 photos × 4 forms × 2 settings × 2 targets (3 workers × 2
  threads, 4 min). Per-row data in `tmp/ml37/canon_int/canon_int_rows.csv` (gitignored). The summary in the `.txt`
  was re-printed with `--summarise` after a grouping fix for the "uncoded" rows; the measured rows were not re-run.
- `canon_int_cost.py` → `canon_int_cost.txt`: §4, full scale, 4 photos.
- `canon_int_dino.py` → `canon_int_dino.txt`: §5.
- Reused, unchanged: `d8_intensity_target.py` (sample, light, encoder, read-back), `bits_cls_common.py`,
  `bits_cls_noise.py`, `bits_cls_check_boxes.py`, `bits_cls_embed.py`.

## Sources

- **The repo at `3df80ac`:** `coffeecv/jxl_encode.py`, `coffeecv/jxl_header.py`, `coffeecv/tone_map.py`,
  `coffeecv/canonical.py`; the ML-37 ticket; `research_d8_intensity_target.md` (§1–§4), `research_d7_out_of_gamut.md`
  (§1), `research_d12_front_colour_path.md`, `research_bits_classifier.md` (§2, §3).
- **libjxl 0.12.0** (tag `v0.12.0`, the library in imagecodecs' wheel), read in `tmp/ml37/d8/downloads/libjxl/`:
  `lib/jxl/enc_external_image.cc` L75–91 (integer input scaled to float), `lib/jxl/enc_xyb.cc` L233–288 (`ToXYB`),
  `lib/jxl/render_pipeline/stage_write.cc` L273–285 (integer output clamped and rounded).
- **imagecodecs 2026.10.10**, the source distribution, read in `tmp/ml37/d7/downloads/imagecodecs/_jpegxl.pyx`:
  L153–176 (`jpegxl_encode`'s parameters), L355–373 (pixel types), L427–446 (colour encoding and the default
  transfer), L827–847 (the decoder's output type from the header's bit depth).
