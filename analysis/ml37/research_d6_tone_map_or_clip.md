# ML-37 D6: light above white at the read-back, tone-map or clip

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D6, 2026-11-23, on the local checkout at
`01e6ac1`. This is a recommendation for the owner, not a decision: D6 stays open until the owner picks an option. Every
number comes from the scripts beside this note and their printed results: `d6_readback_options.py` (rows in
`tmp/ml37/d6/readback_rows.json`, tables in `d6_readback_options.txt` via `d6_summarize.py`) and `d6_peak_cost.py`
(`d6_peak_cost.txt`). The standards and source code were read in `tmp/ml37/d6/downloads/` (not committed).

## Summary

- **Recommended: (a), with one correction to how the ticket words it.** The EETF (ML-29 D33: BT.2408-8 Annex 5 on
  max(R, G, B), the picture's own peak) must take its max(R, G, B) **from the linear Rec.2020 values**, as today. Then
  the read-back matches today's models' input on all 23 gain-map photos measured: median 0.05 of a level on 32×32
  bean blocks, worst block 2.4, every photo within D27 (mean ≤ 1.0, no block over 6) (§2).
- **D2's order can stay.** The EETF on max(R, G, B) scales each pixel's R, G and B by one number, so it commutes with
  the 3×3 matrix. Matrix first, then each pixel scaled by the gain from its Rec.2020 max, gives today's result to
  3.6 × 10⁻⁷ (§3).
- **Taking the max after the matrix, in sRGB, does not read close to today.** Only 4 of 23 photos stay within D27 on
  the beans and 1 of 23 on the whole photo. The median photo moves 0.6 of a level, the worst 5.8 on average, with
  blocks up to 20.7 levels darker. Bean pixels' sRGB max is 1.06–1.38× their Rec.2020 max (per-photo medians). Today
  those pixels are clipped per channel by LittleCMS after the Rec.2020 EETF (up to 48% of bean pixels on one iPhone);
  an sRGB max compresses them instead. The peak's space barely matters: the sRGB peak is a median 1.02× the
  Rec.2020 peak (§3).
- **(b), clip at 1.0**: within D27 on 3 of 23 photos. The median photo's beans move 0.5 of a level brighter, the worst
  photo's 8.0, with blocks up to 75 levels off.
- **(c), a fixed roll-off**: within D27 on 3 of 23 (peak 2× white) and 0 of 23 (4×). It also changes standard-range
  photos, because any roll-off that does not clip starts below white. Its knee sits at 0.70 (2×) or 0.48 (4×), so
  standard-range photos move up to 11.5 (2×) and 22.7 (4×) levels on a block. That breaks the ticket's "photos with
  nothing above white read the same under all three" (§4).
- **The standards do not support (c) for photos.** BT.2446-1's three static methods are for broadcast at a fixed
  1000 cd/m² (or 291) bridge, and the Report itself says dynamic methods may do better. BT.2408-8 §6.4 lists clip,
  static and dynamic (scene maximum) mapping. In Annex 5 the parameters are the mastering peak, or 10,000 cd/m² when
  that is unknown. libultrahdr's fixed Reinhard curve sets its headroom per transfer function (10,000/203 or
  1000/203) and maps white to about 0.5. In ISO 21496-1 gain-map photos, as libultrahdr decodes them, the
  standard-range rendition is the camera's own base image, which is per picture. BT.2446-1 and libultrahdr both
  tone-map in BT.2020/BT.2100 and convert the gamut afterwards (§6).
- **The per-picture peak is cheap at the read-back and survives the lossy coding.** The peak pass takes 0.55–0.62 s
  per 12 MP photo (1.1 s at 24.5 MP) in single-threaded numpy, or 0.03–0.07 s on a 1-in-16 subsample (within 0.4%).
  The lossy half-float JPEG XL moves the peak by at most 0.1%. In the numpy prototype the EETF itself costs more than
  the peak: 1.2–1.9 s against 0.5 s for a clip on 12 MP (§5).
- **Crops must take the whole photo's peak.** A crop that measured its own peak would get 1.95 instead of 2.61 (an
  iPhone 15 Pro Max) and move 52% of its pixels by over half a level. Under D10 (a) or (b) the crop holds read-back
  values, so it inherits the peak. Under D10 (c) the whole photo's peak has to be passed in (§5).

## 1. The question and the options measured

D2 (decided) fixes the models' input as linear Rec.2020 fp16 → 3×3 matrix → tone-map or clip to [0, 1] → sRGB curve,
float32, with no 8-bit step. D6 asks which of three is the "tone-map or clip": (a) D33's EETF at the picture's own peak,
"applied after the matrix", (b) a clip at 1.0, (c) a fixed roll-off for every picture (ticket §6, D6).

The measurement (`d6_readback_options.py`) uses 23 gain-map photos from ML-34's experiment A. All have a bean mask
(`data/seg_masks_ml19/xl0_v7r_s123`) and come from 8 cameras: Pixel 9 Pro ×5, OnePlus 7 Pro ×5, Galaxy Z Fold8
Ultra ×3, iPhone 15 ×3, iPhone 15 Pro Max ×3, iPhone 16 Pro Max ×2, Pixel 7 ×1, iPhone 15 Pro ×1. Per camera they
span the share of bean pixels above the EETF's knee, the camera's highest always included, so **the sample
overweights photos with much light above white**. Their median share of bean pixels above white in sRGB is 2.8%;
ML-34's experiment A gives 0.3% above white (in Rec.2020) for the median gain-map photo (ticket §2). The script also
measured 6 standard-range photos; 4 were measured and 2 failed in the script (an empty region histogram, an
`IndexError` in its percentile lookup) and were not re-run.

Each original is decoded as the store decodes it (`canonical.decode`, gain map applied): linear Rec.2020,
1.0 = reference white, light above white kept, negatives clipped at 0 (D7 (a)). That is D1's canonical photo before
its lossy coding, so only D6's choice differs between options. Options, all float32:

| Name | What |
|---|---|
| `a2020` | EETF on max(R, G, B) **in Rec.2020** at the Rec.2020 peak (today's D33 maths), then the matrix, clip, sRGB curve |
| `aSRGB` | the matrix first, then the EETF on max(R, G, B) **of linear sRGB** at the peak measured there: the ticket's (a) read literally |
| `aSRGBp` | as `aSRGB` but at the Rec.2020 peak (isolates the max from the peak) |
| `b` | matrix, clip at [0, 1], curve |
| `c2`, `c4` | the EETF on the sRGB max at a fixed peak of 2× or 4× white for every picture |

The reference **T** is today's models' input: `canonical.load_canonical` of the photo's canonical photo in
`data/canonical`, which is uint16 Rec.2020 with the BT.709 curve, lossy, already tone-mapped at canonicalization,
then LittleCMS to 8-bit sRGB. The measure is ML-29 D27's: 32×32 block means in levels of 255 (mean |d|, worst block,
signed shift). It is taken over bean blocks (half their pixels bean) and over the whole photo.

## 2. How close each option comes to today's input, gain-map photos (n = 23)

Bean blocks, median photo (worst photo):

| Option | mean \|d\| | worst block | signed shift | photos within D27, beans / whole |
|---|---|---|---|---|
| `a2020` | **0.049** (0.119) | **0.78** (2.43) | −0.002 (−0.073 to +0.006) | **23 / 23** |
| `aSRGB` | 0.606 (5.760) | 10.35 (20.71) | −0.576 (to −5.760) | 4 / 1 |
| `aSRGBp` | 0.533 (5.709) | 9.69 (20.63) | −0.502 (to −5.709) | 4 / 1 |
| `b` | 0.502 (7.974) | 28.39 (74.68) | +0.476 (to +7.963) | 3 / 1 |
| `c2` | 0.372 (4.460) | 12.52 (20.66) | −0.350 (−4.460 to +0.039) | 3 / 0 |
| `c4` | 1.619 (8.787) | 18.78 (24.36) | −1.616 (to −8.787) | 0 / 0 |

Whole photo, median (worst): `a2020` 0.045 (0.084), worst block 0.83 (2.83); `aSRGB` 1.20 (5.35), 13.1 (20.7); `b`
1.97 (7.74), 43.7 (82.2); `c2` 0.97 (3.29), 13.6 (20.7); `c4` 3.01 (7.82), 20.2 (24.4) (`d6_readback_options.txt` §1).

- `a2020` stays this close to T even though T comes from a lossy integer photo rounded to 8 bits. Per bean pixel,
  `a2020` against T differs by a median 0.96 of a level, 99th percentile 5.9. That is today's own coding noise, and
  the block means average it away (`d6_readback_options.txt` §2).
- `b` is close on photos with little light above white (OnePlus 7 Pro at peak 1.35: 0.04) and far on the iPhone 15
  Pro Max, iPhone 16 Pro Max and Fold8 photos, where 23–50% of bean pixels lie above white in sRGB. On those it reaches
  2.8–8.0 on average and blocks of 51–75 levels (per-photo rows in `readback_rows.json`).
- `c2` happens to suit photos whose peak is near 2. It overshoots photos with low peaks (Pixel 9 Pro at 1.55: mean
  0.97, against `a2020`'s 0.04) and undershoots the iPhones at 2.6–3.3.

## 3. The order: EETF before or after the matrix

The EETF on max(R, G, B) (`tone_map.eetf_in_place`) multiplies a pixel's three channels by one number,
curve(m)/m, where m is that pixel's max. A linear 3×3 matrix commutes with a per-pixel scalar: M(g·x) = g·(M x). So
what matters is **which max the gain is computed from**, not where in the chain it is applied. Checked on 200,000
random pixels: the EETF in Rec.2020 then the matrix, against the matrix then the gain from the Rec.2020 max, differ by
at most 3.6 × 10⁻⁷. D2's written order (matrix → tone map) can therefore deliver today's maths exactly, if the gain
comes from the Rec.2020 max.

Taking the max after the matrix changes the result because, on the beans, the sRGB max is larger:

- max(R, G, B) of linear sRGB over that of linear Rec.2020, per bean pixel: per-photo medians 1.06–1.38, 95th
  percentiles up to 1.47. The ratio is highest on the iPhone 15 Pro/Pro Max, Pixel 9 Pro and Fold8 photos
  (`d6_readback_options.txt` §3). Brown, saturated beans sit near the sRGB gamut's edge, where the matrix enlarges
  the leading channel.
- Today's Rec.2020 EETF brings the Rec.2020 max to at most 1, but the sRGB channels can still exceed 1. LittleCMS
  clips them per channel. The share of bean pixels clipped that way ranges from 0.05% to 48.4% (median photo 2.0%;
  iPhone 15 Pro Max 18–48%, iPhone 16 Pro Max 17–37%, OnePlus 7 Pro up to 16%). Those clipped pixels are part of
  today's input, and `a2020` reproduces them.
- With the max in sRGB, these pixels pass a knee at 0.60–0.80 and are compressed instead. The result is darker:
  every photo's signed shift under `aSRGB` is negative, down to −5.8 levels on average over the beans.
- The peak's space is a minor factor. The sRGB peak is a median 1.020× the Rec.2020 peak (1.000–1.162), and
  `aSRGBp` (sRGB max, Rec.2020 peak) is nearly as far from T as `aSRGB` (0.53 against 0.61).

Compressing in sRGB is arguably the better tone map: it avoids clipping in the output gamut, which is what BT.2408's
Table A5.1 credits maxRGB with in the space it runs in. But it is a new look, and the ticket's reason for (a) is to
keep today's input so the retrain "measures the new pipeline rather than a new look" (ticket §6, D6).

## 4. Standard-range photos: what (c) does to them

With nothing above white, D33's peak is 1.0 and the EETF is the identity (`tone_map.eetf_in_place` returns at
`peak <= 1`). So `a2020`, `aSRGB` and `b` are the same function. On the 4 standard-range photos measured they agree
with T to 0.045 (0.058) mean and 1.25 (2.28) worst block, which is coding and rounding only.

A fixed roll-off cannot keep that. A curve that sends everything above white to at most 1.0 without clipping must
start bending below white. The BT.2408 EETF at a fixed peak of 2 has its knee at 0.70 of white, and at 4 at 0.48
(`tone_map.eetf_knee`). On the standard-range photos 0–13.5% of the region lies above the 2× knee and 0–28.3%
above the 4× knee. `c2` moves them by a median 0.25 (worst 0.94) on average, blocks up to 11.5 levels. `c4` moves
them by 1.14 (2.41), blocks up to 22.7 (`d6_readback_options.txt` §1, §4). The standard-range photos are most of the
dataset, so (c) would change the input for most of the dataset, not only the 1,684 gain-map photos.

## 5. The per-picture peak at the read-back: its cost, and crops

`d6_peak_cost.py` covers three gain-map photos from the sample: the Pixel 9 Pro of median peak, the iPhone 15 Pro
Max with the most bean light above white, and the 24.5 MP iPhone 15 Pro. Each is written as D1's canonical photo would
be (`imagecodecs.jpegxl_encode`, float16, BT2100 primaries, distance 0.55, effort 3) and decoded back. All timings
are single-threaded numpy on this shared machine except the codec (2 threads). Peak RSS was 945 MB, including the
original's decode.

| | Pixel 9 Pro 12.5 MP | iPhone 15 Pro Max 12.2 MP | iPhone 15 Pro 24.5 MP |
|---|---|---|---|
| peak, original light / decoded | 1.5528 / 1.5527 | 2.6059 / 2.6055 | 2.7311 / 2.7285 |
| exact peak pass (float32 maxima, 4 B/px) | 0.62 s | 0.55 s | 1.12 s |
| 1-in-16 subsample (every 4th row and column) | 1.5553, 0.04 s | 2.6055, 0.03 s | 2.7189, 0.07 s |
| streaming histogram (4096 log bins, strip by strip) | 1.5555, 0.57 s | 2.6110, 0.56 s | 2.7331, 1.18 s |
| read-back, (b) clip / (a) EETF | 0.50 / 1.21 s | 0.49 / 1.85 s | 0.92 / 3.50 s |
| pixels where (a) and (b) differ by > ½ level | 7.2% | 36.2% | 29.8% |

- On the 23 original photos the peak pass took a median 53 ms per MP (40–75) (`d6_readback_options.txt` §5).
- The lossy coding moves the peak by at most 0.1% and the knee by at most 0.0003. Measuring the peak on the decoded
  photo at the read-back is therefore equivalent to measuring it on the original at canonicalization.
- The EETF's own cost exceeds the peak's in this prototype, because `_eetf_curve` runs PQ in float64 on every pixel
  above the knee and so grows with that share. Neither is optimised. A 1-in-16 subsample gives the peak to 0.4% at
  about 3 ms/MP.
- R4's "the tone map needs the whole picture's peak before any strip is written" holds. The decoder returns the
  whole picture anyway, so it costs one extra pass over memory, plus 4 B/px for the exact percentile (or nothing
  beyond a strip for the histogram).

**Crops.** Today a crop is cut from the read-back of a canonical photo that was tone-mapped with the whole photo's
peak, so it inherits that peak without anyone passing it on. Under (a) the same must hold. Measured on the bean
mask's bounding box, the region the segmenter crops:

- iPhone 15 Pro Max, box 28% of the photo: own peak 1.9531 (knee 0.71) against the photo's 2.6055 (knee 0.61).
  Read at its own peak, **52.5%** of the crop's pixels differ by over half a level.
- iPhone 15 Pro, box 18%: own peak 1.9346 against 2.7285; 2.9% of pixels differ.
- Pixel 9 Pro: the box is the whole photo; no difference.

So a crop must use the whole photo's peak. Under D10 (a) or (b) the crop stores read-back values made from the whole
photo, so it inherits the peak as today. Under D10 (c), crops cut when the head is fitted, the whole photo must be
read back, or its peak passed in. `d10_crop_forms.txt` measures (c) as already decoding the whole photo (0.29 s), so
the peak pass adds about 0.6 s per 12 MP (0.04 s subsampled). Alternatively, the peak is computed once and kept with
the canonical photo. Apple's Core Image does this for a per-image headroom: `CIImage.contentHeadroom` comes from the
file's metadata and is kept by warp kernels and filters such as `CILanczosScaleTransform` and `CIAreaAverage`
(Apple's documentation, §8).

## 6. What the standards and libraries say, to judge (c)

- **ITU-R BT.2408-8 §6.4** (PQ above 1000 cd/m²) names three approaches: "(1) Clip … (2) Static mapping … (e.g.
  using an EETF curve …) (3) Dynamic mapping", where "an adaptive algorithm could vary the EETF … based on statistics
  of the image content (scene maximum for example)". Clipping "is simple to implement"; static mapping "avoids hard
  clipping of detail in the highlights". D33's per-picture peak is approach (3), (b) is (1) and (c) is (2).
- **BT.2408-8 Annex 5** sets the EETF's input range by "the mastering display minimum black and peak white
  luminances" when "known or reasonably can be assumed", and otherwise "a value of 10 000 can be used for LW". The
  standard's fixed fallback is the whole PQ range, not a 2× or 4× roll-off. Table A5.1: maxRGB "does not produce
  colours significantly outside the Target Colour Volume", meaning the volume of the space the EETF runs in.
- **BT.2408-8 §2.2**: static HDR-to-SDR down-mappers "are usually optimised around these reference levels", and
  departures from them "may lead to difficulties such as loss of important detail". **§7.2**: "a dynamic HDR to SDR
  converter may be required to deliver a satisfactory SDR output … accommodating a wider range of exposures than
  might be possible with a fixed (or static) tone mapping curve". **Annex 10**: in informal tests a dynamic conversion
  "was able to preserve more details in the SDR signal than a static conversion".
- **ITU-R BT.2446-1 §1**: its three methods "are static, offering a fixed processing independent of content", and "a
  suitable dynamic method could provide better performance across a wider range of video content". Table 1 fixes
  each method's conversion peak (1000, 291, 1000 cd/m²) for graded or live broadcast. On order: Method B compresses
  highlights first, then brings colours "outside the target BT.709 colour volume" into it by hard clipping (§5.2.4).
  Method C applies the BT.2407 gamut conversion "after HDR to SDR conversion" (§6.1). Both tone-map in BT.2020 before
  the gamut change, as `a2020` does.
- **BT.2390** was not re-read for this note. ML-29's reading found that BT.2390-11 §5.4 now points to BT.2408 for the
  EETF (`analysis/ml29/research_tone_mapping.md` §1).
- **libultrahdr** (`lib/src/jpegr.cpp`, `lib/src/gainmapmath.cpp`). When given only an HDR picture it makes the
  standard-range one with `globalTonemap`, an extended Reinhard on max(R, G, B). The headroom is
  `hdr_white_nits / kSdrWhiteNits`, and `getReferenceDisplayPeakLuminanceInNits` fixes it per transfer: 10,000 for
  PQ and linear, 1000 for HLG, over 203. That is a fixed curve per transfer function, which ML-29 found maps
  reference white to about 0.5. The order is the tone map in linear BT.2100, then `hdrGamutConversionFn` to Display
  P3, then "Hard clip out-of-gamut values". On decode it weights the gain map by
  `(log2(display_boost) − log2(hdr_capacity_min)) / (log2(hdr_capacity_max) − log2(hdr_capacity_min))`, clamped to
  [0, 1]. On a standard display (boost 1) the weight is 0, so the standard-range rendition is the camera's own base
  image. libultrahdr writes and reads ISO 21496-1 metadata (`kIsoNameSpace "urn:iso:std:iso:ts:21496:-1"`).
- **ISO 21496-1** itself is paywalled and was not read. Its public descriptions (ML-29 §1: Android's Ultra HDR v1.1,
  libavif's `avifGainMap`) give the same weighting between a baseline and an alternate headroom, and so the same
  conclusion: each gain-map photo carries its own, per-picture standard-range rendition.
- **Apple Core Image**: `CIImage.contentHeadroom` is per image ("1.0 … if the image is SDR", "greater than 1.0 … if
  the image is HDR"), from the file's metadata or deduced from its format.

**Verdict on (c).** No source here defines a fixed roll-off for still photos. The static curves in BT.2446 and
libultrahdr assume a known, fixed source peak (a 1000 cd/m² bridge, or a transfer function's top). Our photos' peaks
range from 1.2 to 3.4× white. Every source that tone-maps per picture, whether BT.2408's dynamic mapping, gain-map
rendition or Apple's content headroom, keys on the picture's own peak or headroom, which is D33's choice.

## 7. Recommendation

**(a): D33's EETF at the picture's own peak, with the gain taken from max(R, G, B) of the linear Rec.2020 values.**
It can be applied before the matrix, as today, or after it in D2's order as a per-pixel gain; the two are identical.
The peak is D33's (99.99th percentile of the Rec.2020 max, at least 1), measured on the decoded canonical photo at the
read-back or once at canonicalization, and **always the whole photo's**, with crops inheriting it.

- Only this keeps today's input: 23 of 23 gain-map photos within D27, median 0.05 of a level. On standard-range
  photos it is the identity, the same as (b).
- If the ticket's "(a) … applied after the matrix" is meant as the max taken in sRGB, it should be reworded. That
  variant is a new look, 1 of 23 photos within D27 on the whole photo, darker by up to 5.8 levels on average.
- (b) is simpler and costs nothing extra, but it changes the photos with the most light above white the most: up to
  8 levels on average, blocks of 75.
- (c) has no standard behind it for photos, and it changes every standard-range photo.
- Cost: about 0.6 s per 12 MP for the exact peak (0.04 s subsampled) plus 0.7–1.4 s for the EETF over a clip, in an
  unoptimised numpy prototype. This belongs in R3/R4's measurements on ML-29 R6's photos.

**Confidence**: high on the ranking and on the before/after-matrix point. It holds on 23 photos from 8 cameras, the
mechanism is explained (§3), and the commutation is exact. Medium on the timings (shared machine, prototype code).
Low on how representative the averages are: the sample overweights photos with much light above white.

## 8. What stays unverified

- The comparison on every dataset photo (R1's measurement) and on the 1,684 gain-map photos: here 23, chosen to span
  each camera's range.
- The read-back's time and memory on the inference host and the front (R3, R4), and with an optimised EETF, for
  example a lookup table on max(R, G, B).
- Where the peak should live if it is kept rather than recomputed. Not examined: the JPEG XL header's intensity
  target is D8's, so a sidecar or the store's index would be needed.
- ISO 21496-1's own text (paywalled) and BT.2390-11 (relied on via ML-29's reading).
- 2 of the 6 standard-range photos (script failure, not re-run). The 4 measured all agree.
- Whether the darker sRGB-max look would help or hurt the classifier: not asked, not measured.

## Files and sources

- `analysis/ml37/d6_readback_options.py`, `d6_summarize.py` → `d6_readback_options.txt`; per-photo rows in
  `tmp/ml37/d6/readback_rows.json`, run log `tmp/ml37/d6/readback.log` (local, not committed).
- `analysis/ml37/d6_peak_cost.py` → `d6_peak_cost.txt`. Its photo choice, the strip-wise (a)-vs-(b) comparison and
  the crop section were added after the first run was cut off.
- `coffeecv/tone_map.py` (`eetf_in_place`, `eetf_knee`, `picture_peak`), `coffeecv/canonical.py` (`decode`,
  `load_canonical`, `rec2020_to_srgb`).
- ITU-R BT.2408-8 (11/2024), <https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-8-2024-PDF-E.pdf>: §2.2, §6.4,
  §7.2, Annex 5, Annex 10.
- ITU-R BT.2446-1 (03/2021), <https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2446-1-2021-PDF-E.pdf>: §1, Table 1,
  §5.2.4, §6.1.
- libultrahdr, <https://github.com/google/libultrahdr>: `lib/src/jpegr.cpp` (`globalTonemap`, `ReinhardMap`, the
  tone-map loop, the decode's `gainmap_weight`), `lib/src/gainmapmath.cpp`
  (`getReferenceDisplayPeakLuminanceInNits`), `lib/include/ultrahdr/gainmapmath.h` (`kSdrWhiteNits = 203`). Fetched
  on 2026-11-23 from the default branch; commit not recorded.
- Apple, `CIImage.contentHeadroom`,
  <https://developer.apple.com/documentation/coreimage/ciimage/contentheadroom>.
- Earlier research relied on: `analysis/ml29/research_tone_mapping.md` (BT.2390, ISO 21496-1's public descriptions,
  libultrahdr's white), `analysis/ml34/research_hdr_pu21.md`, the ML-37 ticket §2 (experiment A) and §6 (D2, D6, D10).
