# ML-37 D7: colours outside Rec.2020 in the linear canonical photo

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D7, 2026-11-23, on the local checkout at
`01e6ac1`. A recommendation for the owner, not a decision: D7 stays open until the owner picks (a) or (b). The
question: when canonicalization converts a photo to linear Rec.2020 (D1: lossy JPEG XL, 16-bit half-float, 1.0 =
reference white, light above white kept), are negative samples (colours outside Rec.2020) (a) clipped at 0, as
`coffeecv/canonical.py` does today, or (b) kept, and clipped only by the read-back (D2: linear Rec.2020 → 3×3
matrix → tone map or clip to [0, 1] → sRGB curve)? The ticket currently recommends (a). This note recommends (b), and
gives the evidence for and against.

The numbers come from four scripts in this folder, each run on the pinned imagecodecs 2026.10.10 (libjxl 0.12.0,
LittleCMS 2.19.0): `d7_jxl_probe.py` (libjxl's behaviour, output `d7_jxl_probe.txt`), `d7_readback.py` (matrices and
read-back, output `d7_readback.txt`), `d7_dataset_gamut.py` (the dataset: `profiles`, `photos`, `raw` and `raw_jxl`
modes; outputs `d7_dataset_gamut_profiles.txt` and, local only, `tmp/ml37/d7/photos.txt`, `raw.txt`, `raw_jxl.txt`,
`float_jxl.txt`), and `d7_decoded_negatives.py` (negatives made by the lossy encode itself, output
`d7_decoded_negatives.txt`). All read-back levels below are 8-bit sRGB levels (0–255).

## Summary

- **libjxl keeps negative float samples through a lossy XYB encode and decode, except where its opsin step clamps.**
  The encoder converts the input to linear sRGB with its CMS, mixes it with the opsin absorbance matrix plus a bias
  of 0.00379, and clamps each mixed value at 0 ("mixed* should be non-negative even for wide-gamut", `enc_xyb.cc`
  92–95). The decoder does not clamp ("Avoids clamping to [0, 1]", `dec_xyb-inl.h` 33–34); float output is not
  clamped, integer output is (`stage_write.cc` 282). Measured: negatives down to −0.3 come back within ±0.0006 of what
  went in; only colours whose mixed value falls below 0 (near black with a negative channel, or far outside the gamut)
  come back wrong, by up to 0.19 (§1).
- **The dataset's JPEG, HEIC, PNG and JPEG XL photos hardly go below 0, and clipping changes nothing in what the models
  read.** Only the Display P3 family (1,304 of 2,851 files) can go below 0, because P3 red lies just outside Rec.2020:
  at worst −0.0012, exhaustively over every 8-bit triple. In 40 sampled photos, the lowest value is −0.00007, and the
  read-back is identical under (a) and (b) in every pixel (0.00 levels). sRGB, Adobe RGB and the other ICC sources
  never go below 0 (§2).
- **RAW photos are the one real source of negatives, but today's developers clip them before canonicalization sees
  them.** LibRaw's camera matrix makes channels below 0 in up to 6.1% of a photo's pixels, down to −0.069 of white
  (−0.81 after LibRaw's auto-bright), and LibRaw clips them to 0 itself (`convert_to_rgb_loop`). Kept, they would
  change the read-back by 0.5 level or more in 0–1.2% of pixels, by up to 126 levels in single noisy pixels. A
  darktable float JPEG XL (not in the dataset) goes down to −0.0093, with 0.1% of pixels changed by up to 4.8
  levels (§2).
- **Clipping at 0 before the encode does not keep the stored numbers inside Rec.2020.** The lossy encode makes its own
  negatives: 0.2–5.8% of the pixels of photos clipped at 0 decode below 0, down to −0.017, which is more than any
  non-RAW source makes. The read-back must therefore handle negatives under either option. That undercuts the ticket's
  reason for (a), "the store's numbers stay inside the gamut it names" (§3).
- **The two options give identical results for every colour that is inside Rec.2020.** For colours outside it, (b)
  reads back as a direct conversion to sRGB followed by a clip, and (a) shifts the hue (§4).
- **The tone map and gain maps do not interact with D7.** The EETF scales R, G and B by the same positive factor, set
  by max(R, G, B), so it commutes with a clip at 0. The peak (max(R, G, B)) is unchanged by negatives. Gain maps are
  applied in linear Rec.2020 before any clip and make negatives no larger than −0.00007 (§5).
- **Recommended: (b) keep them.** For the dataset as developed today the two options read the same, so this is a
  choice on principle. (a)'s guarantee does not survive the lossy encode. (b) is one fewer step, keeps the
  wide-gamut colour that ADR 0033 stores Rec.2020 for, and gives the right sRGB colour for any future source that has
  real out-of-gamut colour. The opsin clamp is the cost of (b): it affects 0–0.10% of pixels even in the worst RAW
  tiles, and only if a developer keeps RAW negatives (§6).

## 1. What libjxl 0.12 does with negative float samples

**Source** (libjxl `lib/jxl/`, read in `tmp/ml37/d7/downloads/libjxl/`, the release imagecodecs reports as
`libjxl-0.12.0`):

- `enc_xyb.cc` 233–286, `ToXYB`: an input that is neither linear sRGB nor sRGB goes through `ApplyColorTransform`
  (the CMS) to linear sRGB (281–283), then `LinearSRGBToXYB`.
- `enc_xyb.cc` 50–68, `OpsinAbsorbance`: mixed = kOpsinAbsorbanceMatrix · rgb · (intensity_target / 255)
  (`ComputePremulAbsorb`, 214–226) + kOpsinAbsorbanceBias. The bias is 0.0037930732552754493 for each channel
  (`cms/opsin_params.h` 36–38).
- `enc_xyb.cc` 92–95: "mixed* should be non-negative even for wide-gamut, so clamp to zero." and `ZeroIfNegative` on
  all three values, before the cube root (99–101). A linear-sRGB colour whose mixed values are all ≥ 0 survives with
  its negative channels intact. Rec.2020's primaries do (the matrix rows are positive), so the clamp bites only where
  a mixed value is below 0. That happens when the weighted sum of the linear-sRGB channels is below −0.0038 · 255 /
  intensity target.
- `dec_xyb-inl.h` 32–34, `XybToRgb`: "Avoids clamping to [0, 1] - out of (sRGB) gamut values may be in-gamut after
  transforming to a wider space."
- `render_pipeline/stage_write.cc` 264–286 (`MakeUnsigned`, `Clamp(Zero, v, mul)` at 282) and `dec_external_image.cc`
  64–65 clamp only on conversion to integer samples. A float or half-float request keeps negatives.
- imagecodecs `_jpegxl.pyx` 435–442: a float array is encoded with the linear transfer unless told otherwise, so
  `jpegxl_encode(float16, primaries="BT2100")` is D1's form.

**Probe** (`d7_jxl_probe.txt`: float16, BT2100 primaries, linear, level 95, effort 7):

| linear Rec.2020 in | opsin mixed | decoded (inner mean) | error |
|---|---|---|---|
| (0.50, 0.40, −0.10) | ≥ 0 | (0.4996, 0.3999, −0.0997) | ≤ 0.0004 |
| (−0.10, 0.30, 0.30) | ≥ 0 | (−0.0998, 0.2997, 0.3002) | ≤ 0.0003 |
| (1.50, 1.20, −0.05), above white and outside | ≥ 0 | (1.4998, 1.2001, −0.0495) | ≤ 0.0005 |
| ProPhoto blue, (−0.143, −0.011, 1.035) | ≥ 0 | (−0.1430, −0.0106, 1.0343) | ≤ 0.0010 |
| (0, 0, −0.01) | clamped | (−0.0001, −0.0003, −0.0062) | 0.0038 |
| ProPhoto green, (−0.058, 1.081, −0.041) | clamped | (−0.0576, 1.0801, −0.0312) | 0.0096 |
| (−0.30, 0.80, −0.05) | clamped | (−0.3080, 0.7842, 0.1398) | 0.19 |

The table shows 7 of 24 rows: 12 chosen colours and 12 ProPhoto primaries and secondaries. Every row whose mixed
value is ≥ 0 decodes within 0.0010.

The probe also put a natural photo's light (one dataset JPEG, 768×768) beside an out-of-gamut square
(−0.10, 0.30, 0.30). Keeping the square's negatives instead of clipping them moves the in-gamut pixels in a 16-px
ring around it by a mean of 0.0014 (max 0.025). The coding error in that ring is the same either way: mean 0.0035
kept, 0.0036 clipped. Keeping negatives does not spread error into neighbouring pixels.

## 2. Where the dataset goes below 0 in linear Rec.2020

The dataset's colour sources come from the ML-29 scan (`tmp/ml29_scan/files.csv`, 2,851 files):

- JPEG, Display P3 ICC: 1,081
- JPEG, untagged sRGB: 841
- JPEG, sRGB IEC61966-2.1: 547
- HEIC, Display P3: 164
- DNG: 72
- JPEG, DCI-P3 D65 with sRGB transfer: 50
- PNG: 20
- JPEG XL codestream, 12-bit sRGB: 19
- CR3: 19
- JPEG, c2 ICC: 13
- JPEG, Display P3 with sRGB transfer: 9
- JPEG, Adobe RGB: 7
- JPEG, other sRGB-family ICC: 8
- GIF: 1

**Every 8-bit triple, through each source's own profile** (`d7_dataset_gamut.py profiles`: LittleCMS float, relative
colorimetric, unclipped, the profile and curve that `canonical.decode` takes from one photo of each source):

| source | triples < 0 | < −1e-3 | lowest |
|---|---|---|---|
| Display P3 ICC | 0.094% | 0.0018% | −0.00119 |
| DCI-P3 D65 / Display P3 with sRGB transfer | 0.096% | 0.0021% | −0.00121 |
| sRGB (untagged, ICC, JPEG XL enumeration), Adobe RGB, c2, Canon sRGB, Sony Camera RGB | 0 | 0 | 0 |
| ProPhoto (reference, not in the dataset) | 37.7% | 36.8% | −0.2006 |

The P3 figure follows from the primaries (`d7_readback.txt` §1): Rec.2020's blue from linear P3's red is −0.00121.
P3 red, (0.680, 0.320), lies just above Rec.2020's red–green edge, which passes y = 0.3183 at x = 0.680. sRGB and
Adobe RGB matrices into Rec.2020 have no negative coefficient.

**40 sampled photos, decoded as the store does** (`photos.txt`: one strip at a time, gain maps applied):

- **Every P3-family photo, with or without a gain map:** 0.003–1.37% of pixels below 0, the lowest −0.00007, none
  below −1e-3.
- **sRGB, Adobe RGB and the other sources:** 0%.
- **Read-back:** for all 40 photos, no pixel changes by 0.5 level or more between (a) and (b); the largest difference
  is 0.00 levels.

**RAW** (`raw.txt`, 9 photos at half size). LibRaw clips its matrix output to 0–65535 (LibRaw 0.22.1
`src/postprocessing/postprocessing_utils.cpp` 105–112, `CLIP` in `convert_to_rgb_loop`), so what LibRaw would have
made negative was measured on its camera-RGB output times its own matrix. That matrix is `out_rgb` × `rgb_cam`
(`postprocessing_utils_dcrdefs.cpp` 101; `rec2020_rgb`, `colorconst.cpp` 54). `rgb_cam` comes from the file, or from
LibRaw's camera table when the file has none (`identify.cpp` 1193–1194). The read-back was taken after the auto-bright
that canonicalization's `rawpy` call keeps on, with white at the brightest channel's 99th percentile.

| RAW | developer | px < 0 | < −1e-3 | lowest (1.0 = sensor white) | read-back ≥ 0.5 level apart | largest |
|---|---|---|---|---|---|---|
| Google DNG | rawpy | 0.63% | 0.35% | −0.0202 | 0.21% | 3.9 |
| Google DNG | rawpy | 6.11% | 0.95% | −0.0060 | 0.21% | 2.5 |
| Google DNG | rawpy | 0.05% | 0.01% | −0.0047 | 0.001% | 1.1 |
| JPEG XL LinearRaw DNG ×2 | `linear_dng` | 0.004–0.05% | 0 | −0.0001 | 0 | 0.0 |
| Canon CR3 | rawpy | 4.42% | 2.58% | −0.0403 | 1.18% | 49 |
| Canon CR3 | rawpy | 2.16% | 0.04% | −0.0687 | 0.16% | 126 |
| Apple DNG, OnePlus DNG | rawpy | ≤ 0.001% | 0 | −0.0068 | 0 | 0.2 |

The negatives are dark, noisy pixels. Matrixing camera noise near black makes some channels negative, and auto-bright
scales them up by ×1.4–12. These negatives never reach the store today. LibRaw clips them in rawpy, and
`coffeecv/linear_dng.py` clips at 77 and 124. Under (b) they would reach the store only if RAW development changed to
keep them (camera RGB plus our own matrix in float), which is outside D7.

**A float JPEG XL** (`float_jxl.txt`: the owner's darktable 32-bit float sRGB export from ML-29, not a dataset photo;
the dataset's 19 JPEG XL files are 12-bit integer). The source samples run from −0.295 to 1.074, sRGB-coded. In
linear Rec.2020: 0.60% of pixels below 0, 0.21% below −1e-3, the lowest −0.0093. Read-back: 0.099% of pixels differ
by 0.5 level or more between (a) and (b), at most 4.8 levels.

## 3. The lossy encode makes its own negatives

`d7_decoded_negatives.txt` covers six sampled photos, one per camera and colour source. Each was taken as the centre
1024×1024 in linear Rec.2020, clipped at 0 as (a) would do, encoded as D1's canonical photo and decoded.

| photo | px < 0 before | decoded px < 0 | decoded < −1e-3 | decoded lowest | read-back change from clipping those |
|---|---|---|---|---|---|
| Google, P3, gain map | 0.19% | 0.43% | 0.09% | −0.0080 | 1.7 levels |
| OnePlus 7 Pro, sRGB, gain map | 0 | 0.26% | 0.02% | −0.0033 | 1.4 |
| Sony, untagged sRGB | 0 | 0.22% | 0.005% | −0.0019 | 0.6 |
| Apple HEIC, P3, gain map | 0.06% | 0.29% | 0.13% | −0.0168 | 2.3 |
| untagged sRGB (web) | 0 | 5.82% | 1.89% | −0.0110 | 3.6 |
| Samsung, ISO 21496-1 gain map | 0 | 0.68% | 0.10% | −0.0035 | 1.4 |

A store clipped at 0 still decodes with negatives, larger than any the non-RAW sources make (§2). The read-back
therefore needs a defined way to handle negatives under both options. D2's order (matrix first, then clip) is the one
that matches (b); clipping in Rec.2020 before the matrix would change these pixels by up to 3.6 levels.

Behind RAW's largest differences, the same effect dominates. `raw_jxl.txt` encoded the darkest 768×768 tile of five
RAW photos, the tile with the most negatives, both ways:

- **Before the encode:** (a) and (b) read back 0.00–0.06 levels apart on 32×32 block means. Per pixel, 0–0.76% of
  pixels are 2 levels or more apart.
- **After the encode:** the coding error is 0.14–0.91 levels on block means, and 3.8–13.6 levels per pixel.
- **libjxl's clamp:** it touched 0–0.10% of the pixels in these tiles under (b). The share is 0.07% at intensity
  target 203 (D8's recommendation), against 0.10% at 255.
- **Determinism:** the encoder is deterministic (re-encoded bit-identical). Its error differs between (a) and (b)
  because the input differs.

In the dark tiles, the D7 choice is far below the coding error.

## 4. What keeping negatives changes at the read-back

From `d7_readback.txt` §2:

- **Inside Rec.2020** (no channel below 0), (a) and (b) are the same input, so the same output. This was checked on
  1,000,000 random colours in [0, 2).
- **Outside Rec.2020**, (b) gives the matrix's true linear-sRGB value, then the per-channel clip. That is exactly what
  converting the source straight to sRGB and clipping gives. (a) first replaces the negative channel by 0, which adds
  light of that Rec.2020 primary, and the matrix spreads it:

| colour | (a) read back | (b) read back | (b) − (a) |
|---|---|---|---|
| P3 red, (0.754, 0.046, −0.0012) | (255, 0, 0) | (255, 0, 0) | 0 |
| (0.5, 0.4, −0.00004), the dataset's size of negative | (202.7, 167.9, 0) | same | 0 |
| (−0.10, 0.30, 0.30) | (0, 157.0, 150.1) | (0, 159.6, 150.5) | G +2.6 |
| (0.02, −0.05, 0.40) | (13.1, 0, 178.3) | (51.3, 0, 179.2) | R +38.2 |
| ProPhoto (0, 0.5, 0.5) | (0, 158.8, 146.3) | (0, 160.3, 146.5) | G +1.5 |
| ProPhoto green, blue, red at 1.0 | saturated | saturated | 0 |

The sign of the change follows the matrix. A negative Rec.2020 green raises sRGB red (coefficient −0.588) and sRGB
blue (−0.101), so (a) under-reads them. Colours in the dataset's range (≥ −0.0012) never move a level.

## 5. The tone map's peak, the EETF and gain maps

- **Peak.** D33's peak is the 99.99th percentile of max(R, G, B) (`coffeecv/tone_map.py` 207–212). For any pixel
  with a negative channel, max(R, G, B) clipped at 0 is the same whether the channels are clipped first or not;
  checked on 1,000,000 random colours (`d7_readback.txt` §2). If D6 measures the peak after the matrix, in linear
  sRGB, (b) can raise an out-of-gamut pixel's sRGB max through the matrix's negative coefficients. With the dataset's
  negatives (≥ −0.00007) that is below 0.0001, nothing against a peak that is at least 1.0.
- **EETF.** `eetf_in_place` multiplies R, G and B by curve(m) / m, where m = max(R, G, B) (`tone_map.py` 215–228;
  its docstring: "Negative light is scaled with its pixel and left … to the clip after the conversion"). That is a
  positive factor that depends only on m, so clip-at-0 then EETF equals EETF then clip-at-0. In `d7_readback.txt`,
  the "EETF at peak 2" columns differ between (a) and (b) only where the plain read-back does: ProPhoto cyan 239.8
  against 240.2. Above-white light outside the gamut, such as (1.5, 1.2, −0.05), reads the same both ways, with or
  without the EETF.
- **Gain maps.** `canonical.py` 393–394 applies the map in linear Rec.2020 after LittleCMS, before any clip:
  (base + base_offset) · 2^(log2 gain · weight) − alternate_offset (`gain_map.py` 138), or Apple's
  1 + (headroom − 1) · map, a positive multiplier.
  - The offsets can make a near-black base slightly negative when the gain leads down, and a positive gain scales a
    base negative.
  - Measured: the gain-map photos go no lower than −0.00007, the ISO 21496-1 Samsung photo has the largest share
    (1.37%), and the read-back difference is 0.
  - Under (a) the clip must come after the map, as it does today in `to_rec2020` (`canonical.py` 555); under (b)
    there is no clip to place.

## 6. Recommendation

**(b): keep negatives in the canonical photo, and clip only in the read-back, after the matrix.**

- **For the dataset as developed today, the two options are the same.** Every non-RAW photo reads back identically
  (§2). RAW negatives are clipped by LibRaw and `linear_dng` before canonicalization sees them, under either option.
  The choice costs nothing in accuracy and needs no re-measurement.
- **The ticket's reason for (a) does not hold after the encode.** The decoded store goes below 0 under (a) by more
  than the sources do (§3), so it never stays "inside the gamut it names". Every reader (the models' read-back, the
  preview, any JPEG XL viewer, whose integer output libjxl clamps) must clip anyway. Clipping once, in the read-back,
  is one rule instead of two.
- **(b) is the faithful record.** ADR 0033 stores Rec.2020 "so the stored photos keep their wide-gamut colour". With
  (b), a source with real out-of-gamut colour reads back as its direct conversion to sRGB. That covers a float JPEG
  XL, or a later RAW developer that keeps the camera matrix's negatives. Under (a), the same colours shift hue, by up
  to 4.8 levels in the float export and up to 126 in noisy RAW pixels.
- **The cost of (b) is libjxl's opsin clamp.** Colours whose mixed value goes below 0 decode wrong, by up to 0.19 in
  the extreme probe colour. That happens near black with a strongly negative channel. It reached 0–0.10% of pixels in
  the worst RAW tiles if RAW negatives were kept, and 0 in everything the store sees today. In those tiles the coding
  error on 32×32 block means (0.14–0.91 levels) is several times to tens of times the D7
  difference (0.00–0.06).

What the ticket's D7 row and D2's text would say under (b):

- canonicalization does not clip at 0; it still replaces NaN with 0 (`canonical.py` 553–554) and +inf as the tone
  map does;
- the read-back clips after the matrix (and after the tone map), never before;
- RAW development keeping its negatives is a separate decision, not part of D7.

If the owner prefers (a), it is harmless for today's dataset. Its clip must then sit after the gain map and the
tone-map peak, and the read-back must still clip, because of §3.

**Confidence.**

- **High:** libjxl's behaviour (source and probe agree) and the non-RAW dataset sources (exhaustive per-profile cubes,
  40 photos).
- **Medium:** RAW (9 of 91 RAW files at half size; LibRaw's matrix and auto-bright emulated from its source, not
  through its own pipeline).
- **Judgment:** the recommendation is a choice on principle, because both options read the same on today's data.

**Not verified:**

- file size of (b) against (a);
- the whole dataset photo by photo (a sample of 40 plus the exhaustive profile cubes);
- any model's accuracy;
- an actual encode at intensity target 203 (the clamp share at 203 is computed, not encoded);
- jxl-rs and browser decoders' handling of negative float samples;
- applying gain maps in the base image's own colour space when `use_base_colour_space` is set (the code applies them
  in Rec.2020).

## Files

- **This folder:**
  - `d7_jxl_probe.py` (`.txt`): libjxl probe.
  - `d7_readback.py` (`.txt`): matrices and read-back. Run with a path argument, it gives §2's float JPEG XL; that
    output is in `tmp/ml37/d7/float_jxl.txt`.
  - `d7_dataset_gamut.py`, with modes `profiles` (`d7_dataset_gamut_profiles.txt`), `photos`, `raw` and `raw_jxl`
    (outputs in `tmp/ml37/d7/`, local only, because they name dataset folders).
  - `d7_decoded_negatives.py` (`.txt`).
- **Sources read (local only, in `tmp/ml37/d7/downloads/`):**
  - libjxl `lib/jxl/enc_xyb.cc`, `dec_xyb-inl.h`, `render_pipeline/stage_write.cc`, `dec_external_image.cc`,
    `cms/opsin_params.h` (release 0.12.0, <https://github.com/libjxl/libjxl>);
  - LibRaw 0.22.1 `src/postprocessing/postprocessing_utils.cpp`, `postprocessing_utils_dcrdefs.cpp`,
    `src/metadata/identify.cpp`, `src/tables/colorconst.cpp` (<https://www.libraw.org>);
  - imagecodecs 2026.10.10 `imagecodecs/_jpegxl.pyx` (<https://github.com/cgohlke/imagecodecs>).
- **Repository code cited:**
  - `coffeecv/canonical.py` (393–394, 415, 550–555);
  - `coffeecv/tone_map.py` (207–228);
  - `coffeecv/gain_map.py` (113–138);
  - `coffeecv/linear_dng.py` (73, 77, 124);
  - ADR 0033;
  - `analysis/ml29/README.md` (the darktable exports).
