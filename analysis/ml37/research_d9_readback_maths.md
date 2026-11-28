# ML-37 D9: who does the read-back's maths, numpy or LittleCMS

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D9, 2026-11-23, on the local checkout at
`01e6ac1` (branch `ml37-linear-canonical`). A recommendation for the owner, not a decision. The read-back is fixed by
the ticket: the canonical photo's linear Rec.2020 half-float → 3×3 matrix → tone map or clip to [0, 1] → sRGB curve,
float32. D9 asks only who computes it: (a) numpy float32, the matrix from the primaries (SMPTE RP 177, BT.2087) and the
IEC 61966-2-1 curve in closed form, tested against LittleCMS; (b) LittleCMS's float transform, as ML-19 D31 has it.
The numbers come from four scripts in this folder, each with its output beside it: `d9_matrix.py` → `d9_matrix.txt`,
`d9_curve.py` → `d9_curve.txt`, `d9_realdata.py` → `d9_realdata.txt`, `d9_speed.py` → `d9_speed.txt`. Library:
`imagecodecs` 2026.10.10 (the repo's pin), bundling LittleCMS 2.19 (`LCMS_VERSION 2190`, `lcms2.h`); numpy 2.5.3
(`uv.lock`).

## Summary

- **The two options compute the same maths; numpy computes it more exactly.** LittleCMS's linear Rec.2020 → sRGB
  transform is the exact matrix up to the s15.16 rounding of the profiles' colorant tags (max element error
  4.2e-5 with `canonical.LINEAR_REC2020`, 4.4e-4 with the committed libjxl Rec.2020 profile), and its sRGB curve is
  IEC's up to the s15.16 rounding of its parameters (max 7.2e-6). numpy float32 is within 4.9e-8 (matrix) and 1.6e-7
  (curve) of the float64 standard (§1, §2).
- **Neither difference matters to a model.** On 20 real canonical photos (837 M samples), LittleCMS against a float64
  reference: max 0.15 of an 8-bit level, mean 0.0015; numpy: max 0.0002, mean 0.00001. At 16 bits (D10's crop store)
  LittleCMS moves 38.8% of samples by a code (max 39 codes), numpy 0.20% (max 0.04 of a code) (§3).
- **numpy is about 2× faster than LittleCMS's single float transform, and 3× faster than the two transforms that D6's
  tone map needs in (b).** 2 threads, 15 MP: numpy 0.89 s, LittleCMS 1.99 s (one transform) and 2.76 s (two), today's
  uint16 → uint8 read-back 1.85 s; 50 MP: 2.54 s, 5.14 s, 8.58 s, 4.81 s. Memory is the float32 output in both
  (181 MB at 15 MP, 602 MB at 50 MP); numpy's strip temporaries add 35–60 MB of peak (§4).
- **LittleCMS's float path has no fast path.** In 2.19 a float transform skips every optimization except joining
  adjacent matrices, runs without a cache and evaluates the sRGB curve with `pow` per sample (§4).
- **(b) cannot hold D6's tone map in one transform**: the tone map sits between the matrix and the curve, so (b) is two
  transforms with numpy between them, and two libraries touch the read-back either way (§4).
- **Recommendation: (a), as the ticket already proposes**, with its tests checking the matrix and curve against
  LittleCMS and against the standards' printed numbers. It contradicts an owner decision (ML-19 D31, (b), "use
  LittleCMS … for both decode and encode") and amends ADR 0033's "one library does all colour"; the owner's stated
  reason, no drift, is met by pinning numpy exactly as `imagecodecs` is (§5, §6). Confidence: high on the numbers;
  medium on the cross-machine reproducibility of numpy's float32 `pow`, not measured here (§6).

## 1. The matrix

**The construction.** A set of primaries and a white give the normalized primary matrix NPM (RGB → XYZ) by SMPTE RP 177's
construction, the one `tests/test_canonical.py` already writes (`rgb_to_xyz`, lines 45–46). The read-back's matrix is
M = NPM(709)⁻¹ · NPM(2020), with BT.709's primaries (BT.709-6 item 1.2), BT.2020's (BT.2020-2 Table 3) and D65
(0.3127, 0.3290). Checks against the standards' printed luminance rows: NPM(709)'s Y row is 0.21264 0.71517 0.07219
(BT.709 item 3.2 prints 0.2126 0.7152 0.0722), NPM(2020)'s 0.26270 0.67800 0.05930 (BT.2020 Table 4: 0.2627 0.6780
0.0593) (`d9_matrix.txt` §1).

```
M (float64) = [[ 1.6604910021 -0.5876411388 -0.0728498633]
               [-0.1245504745  1.1328998971 -0.0083494226]
               [-0.0181507634 -0.1005789800  1.1187296614]]
```

Its rows sum to 1 exactly (white maps to white). As float32 its largest element error is 4.9e-8 (`d9_matrix.txt` §1).

**BT.2087.** BT.2087-0 gives the BT.709 → BT.2020 conversion and says (NOTE 3): "All matrix values above were
calculated with high precision and then rounded to four decimal digits" (`downloads/bt2087.pdf`). The exact 709 → 2020
matrix rounds to BT.2087's 0.6274 0.3293 0.0433 / 0.0691 0.9195 0.0114 / 0.0164 0.0880 0.8956; inverting that
rounded matrix is off by up to 6.9e-5, and rounding the exact inverse to four decimals by up to 5.0e-5, with white
then landing at (1.0001, 1.0000, 0.9999) (`d9_matrix.txt` §3). BT.2087's numbers therefore check M to about 1e-4; they
are not a better M. The same holds for IEC 61966-2-1's own 7-digit XYZ → sRGB matrix (as printed in the ICC's sRGB
note): 3.4e-4 from inv(NPM(709)), giving an M 1.7e-4 off with white at 0.99983 (`d9_matrix.txt` §2).

**What LittleCMS applies.** An ICC matrix/TRC profile stores its colorants already adapted to the D50 PCS
(`cmsD50X/Y/Z` 0.9642 1.0 0.8249, `lcms2.h` 292–294) with Bradford, which LittleCMS computes in `cmswtpnt.c`
(235–263) when it builds an RGB profile, and stores them as s15.16 numbers. Between two RGB profiles with the same
white, the transform is C(sRGB)⁻¹ · C(Rec.2020), where C = Bradford · NPM, and the Bradford factor cancels exactly
(2.4e-16 in float64, `d9_matrix.txt` §4). So the only gap from M is the tags' rounding:

| LittleCMS source profile | tag error vs unrounded B·NPM | effective matrix vs M, max element | white → |
|---|---|---|---|
| `canonical.LINEAR_REC2020` (imagecodecs-built, gamma 1) | 6.5e-6 | 4.2e-5 | (0.99998, 0.99999, 1.000002) |
| the committed libjxl Rec.2020 profile (ML-19 D46) | 3.1e-4 | 4.4e-4 | (0.99994, 0.99998, 1.00045) |

The matrix read back from the float transform equals the one computed from the tags to 4.8e-8, so LittleCMS 2.19
multiplies the two profiles' matrices into one (`_MultiplyMatrix` in `PreOptimize`, `cmsopt.c` 282) and applies it in
float. The float transform does not clip: values at 4.0 pass through the matrix unchanged in ratio, and Rec.2020 green
gives linear sRGB (−0.588, 1.133, −0.101) (`d9_matrix.txt` §4). The committed profile's blue column sums to 0.82521
against D50's 0.8249, and its curve gives blue 1.0004 at 1.0, which is why it is the farther of the two; a (b) read-back
of D1's linear photo would use a linear profile like the first row, not it.

**In sRGB levels**, on 2 M random linear Rec.2020 colours in [0, 1]³, after clip and curve, against M in float64
(`d9_matrix.txt` §5):

| matrix | max, 8-bit levels | max, 16-bit codes | mean, 16-bit codes |
|---|---|---|---|
| M in float32 | 0.0000 | 0.01 | 0.000 |
| LittleCMS, `LINEAR_REC2020` → linear sRGB | 0.16 | 40.5 | 0.40 |
| LittleCMS, committed Rec.2020 → linear sRGB | 0.19 | 49.7 | 2.56 |
| BT.2087, 4 digits inverted | 0.21 | 52.7 | 0.82 |
| IEC 7-digit | 0.44 | 112.5 | 1.86 |

## 2. The sRGB curve

IEC 61966-2-1 encodes linear L as 12.92·L at or below 0.0031308 and 1.055·L^(1/2.4) − 0.055 above it (as quoted in the
ICC's sRGB note, `downloads/icc_srgb_spec.pdf`; `tests/test_canonical.py` line 59 writes the same). LittleCMS builds its
sRGB profile from the exact parameters (`Build_sRGBGamma`, `cmsvirt.c` 639–650: 2.4, 1/1.055, 0.055/1.055, 1/12.92,
0.04045) as a type-4 parametric curve, stores them as s15.16 in the profile's `para` tag, and encodes by inverting the
stored curve analytically (type −4, `cmsgamma.c` 496–525). The stored parameters are 2.399994, 0.947861, 0.052139,
0.077393, 0.040451, so its linear segment's slope is 12.9211 instead of 12.92 and its break moves to 0.0031315
(`d9_curve.txt`). Against IEC in float64 over 2,000,001 points in [0, 1]:

| implementation | max abs diff | where | in 8-bit levels | in 16-bit codes | other 16-bit code | other 8-bit level |
|---|---|---|---|---|---|---|
| LittleCMS, float64 or float32 out | 7.25e-6 | L = 0.003135 (the break) | 0.0019 | 0.48 | 17.7% of points | 1,375–1,388 points |
| numpy float32 closed form | 1.62e-7 | L = 0.97 | 0.00004 | 0.011 | 0.29% | 17 points |

LittleCMS's curve keeps going outside [0, 1] (4.0 → 1.8248, the IEC formula gives 1.8248; −0.01 → −0.129), so (b) needs
an explicit clip just as (a) does. Its uint8 output differs from rint(255·IEC) at 699 of 1,000,001 greys, by one level
(`d9_curve.txt`). IEC's two pieces meet at its encode threshold to 2.9e-8; LittleCMS's 0.04045 decode threshold is the
standard's (`d9_curve.txt`, last lines). The gap is the s15.16 storage, not a different curve.

## 3. On real photos

`d9_realdata.py` takes every 141st photo of `data/canonical` (20 photos, 3.7–50 MP, phones, a Sony, HEIC and Telegram
sources, OOD and segmenter photos). D1's linear half-float input is emulated: today's uint16 BT.709-curve samples are
taken to linear Rec.2020 by LittleCMS's float transform (which undoes the committed profile's curve) and rounded to
float16; the lossy re-encode is the same for every option and is left out. The clip is the tone map (nothing in
today's photos is above white). Five read-backs: A today's `rec2020_to_srgb` (uint8); B LittleCMS REC2020 → SRGB on the
uint16 samples, float32; C option (b), LittleCMS `LINEAR_REC2020` → SRGB on the float16, clipped; D option (a), numpy
float32; R the same as D in float64. All photos together (`d9_realdata.txt`, last block):

| pair | max | mean | samples that round differently |
|---|---|---|---|
| C − R, option (b) vs reference | 38.97 16-bit codes (0.152 8-bit levels) | 0.396 codes (0.0015 levels) | 38.8% at 16 bits |
| D − R, option (a) vs reference | 0.044 16-bit codes (0.0002 levels) | 0.0018 codes | 0.20% at 16 bits |
| C − D, (b) vs (a) | 38.99 16-bit codes (0.152 levels) | 0.395 codes | 38.7% at 16 bits |
| B − D, today's maths in float vs (a) | 1.20 8-bit levels | 0.013 levels | 1.26% at 8 bits |
| A − D, today's uint8 vs (a) | 1.64 8-bit levels | 0.243 levels | 1.26% at 8 bits |
| A − B, today's 8-bit rounding | 0.50 8-bit levels | 0.243 levels | 0.0005% at 8 bits |

The largest C − R values (39 and 31 codes) are in two photos, a Pixel segmenter positive and an iPhone HEIC; in the
other 18 C − R stays under 10 codes (`d9_realdata.txt`, per photo). B − D, the move from today's file to D1's input, is up to
1.2 levels on the same photo and is D1's change, not D9's: it includes the committed profile's 4.4e-4 matrix (§1) and
the float16 rounding. At 8 bits, what the models read today, (a) and (b) are indistinguishable (0.15 of a level at
worst). At 16 bits, where D10 (a) would store crops, (b) rounds 39% of samples to another code than the standard's
maths, (a) 0.2%; that matters to a golden file and to cross-checks, not to a model.

## 4. Speed and memory

`d9_speed.py`: the 50 MP example `webapp/static/examples/brazil-grinder.jxl` (6144 × 8160) and its centre 3360 × 4480
(15.05 MP, the front's cap), each prepared once as uint16 (today's input) and as emulated float16 linear (D1's).
Each method runs in its own process, 2 threads, 128-row strips, output kept whole; best of three; Ryzen 5 5600U with
12 CPUs shared with other jobs, so times carry a few tenths of a second of noise (`d9_speed.txt`):

| read-back | 15 MP time | 15 MP peak RSS | 50 MP time | 50 MP peak RSS |
|---|---|---|---|---|
| today: `rec2020_to_srgb`, uint16 → uint8, NOOPTIMIZE | 1.85 s | 171 MB | 4.81 s | 472 MB |
| (b), one transform `LINEAR_REC2020` → SRGB, float32, then clip | 1.99 s | 300 MB | 5.14 s | 902 MB |
| (b), two transforms: → linear sRGB, clip (D6's tone map), → SRGB | 2.76 s | 310 MB | 8.58 s | 921 MB |
| (a), numpy float32: nine multiply-adds, clip, IEC curve | 0.89 s | 335 MB | 2.54 s | 964 MB |

The input is 90 MB / 301 MB and the float32 output 181 MB / 602 MB in every float option, so peak memory is the same
to within numpy's strip temporaries (+35 MB at 15 MP, +43–62 MB at 50 MP; each strip's float32 intermediates are
128 × width × 3 × 4 bytes per thread). The 50 MP case fitted in one process (under 1 GB), so it is measured, not
extrapolated. `imagecodecs` refuses an in-place transform (`ValueError: cannot transform in-place`), so the two-transform
(b) needs a strip-sized temporary too.

**Why LittleCMS's float path is slow.** In 2.19, `OptimizeByResampling`, `OptimizeByComputingLinearization` and
`OptimizeByJoiningCurves` return early on float formats (`cmsopt.c` 663, 1062, 1410), the matrix-shaper optimization
works "only on 8 bit input" (`cmsopt.c` 1693), and a float transform runs without a cache (`cmsxform.c` 1164–1165). What
is left is `PreOptimize`, which multiplies the two matrices (`cmsopt.c` 282), and a per-pixel float pipeline whose
output curve is the type −4 inverse, `pow` per sample (`cmsgamma.c` 513–525). `imagecodecs` also builds and frees a
transform on every call, under `nogil` (`_cms.pyx` 460–497), once per strip. numpy does the same arithmetic as whole
strips of vectorized ufuncs.

**Why (b) is two transforms.** The ticket puts the tone map between the matrix and the curve, in linear sRGB. A single
LittleCMS transform from linear Rec.2020 to sRGB has nowhere to put it; with the clip alone as the tone map the single
transform works, because the sRGB curve is monotone with 0 → 0 and 1 → 1, so clipping after it equals clipping
before. Any other tone map (D6) makes (b) LittleCMS → numpy → LittleCMS: 3.4× numpy's time at 50 MP, and a second
library in the read-back anyway.

## 5. ADR 0033, ML-19 D31 and the tests

**What (a) contradicts.** ADR 0033 says one library, `imagecodecs` at one exact version, does "both colour conversions,
by the LittleCMS it bundles, at float precision". ML-19 D31 offered "(a) LittleCMS into Rec.2020, own maths back to
sRGB" and the owner chose (b), 2026-11-03: "use LittleCMS (or opencv2 if it's possible, i prefer cv2) for both decode
and encode. And pin library version so it will not drift." D9 (a) is D31's rejected (a), so recommending it reverses an
owner decision, and the ticket's D9 row already says it amends ADR 0033.

**What has changed since D31.** In ML-19 the read-back started from a curve-encoded Rec.2020 file described by an ICC
profile, so it was a profile-to-profile conversion; in ML-37 it starts from linear Rec.2020, which no profile needs to
describe: a constant matrix fixed by two published standards, then a published closed-form curve. LittleCMS still does
the only profile-driven conversion, the original's declared colour into Rec.2020 (ticket §3). The ADR's principle can be
kept in the form "one library does every ICC conversion; the fixed read-back from linear Rec.2020 to sRGB is the
standards' maths in numpy".

**Drift, the owner's reason.** LittleCMS's result here depends on its s15.16 tag and parameter encoding and on its
pipeline, both of which a version change can move (ADR 0033 already treats an `imagecodecs` upgrade as a
re-canonicalization). numpy's result depends on IEEE-754 float32 multiply-add, exact and portable, and on its float32
`pow`, which comes from numpy's own or the platform's maths library and is not guaranteed to round the same on every
CPU and build (not measured here). Today numpy is held at 2.5.3 by the lock but not pinned by an exact specifier, and
the lock test checks only `imagecodecs` (`tests/test_canonical.py` 191–194). Under (a), numpy becomes colour maths:
pin it exactly and extend the lock test to it, which is the same discipline D31 asks of `imagecodecs`.

**Tests.** Today's D8 tests compare LittleCMS's output with "colour maths written here from the primaries'
chromaticities" (`tests/test_canonical.py` lines 3–4, 45–72). Under (a) that independent maths and the production maths
are the same formulas, so the independent check has to come from elsewhere:

- the production matrix against float64 RP 177 within 1e-7, against BT.2087's four-digit matrix inverted within 1e-4,
  and its rows summing to 1;
- the production matrix against LittleCMS's effective matrix (`LINEAR_REC2020` → linear sRGB, float transform on the
  identity) within 1e-4 (measured 4.2e-5);
- the production curve against IEC in float64 within 5e-7 (measured 1.6e-7) and against LittleCMS's `SRGB` within 1e-5
  (measured 7.2e-6), across the break at 0.0031308;
- known points: white → 1, black → 0, Rec.2020's primaries → the clipped values, and D8's golden file regenerated.

Under (b) the existing pattern holds unchanged, with tolerances set from §1 and §2 (LittleCMS is 4e-5 from the
standards' matrix, not 1e-7).

## 6. Recommendation

**(a), numpy float32**, for three measured reasons: it is the more exact (0.2% of 16-bit samples off the standard's
rounding against 39%), the faster (2.0–2.2× LittleCMS's single float transform, 3.1–3.4× the two transforms D6's tone map
requires, at the same memory within 60 MB), and the only one that holds the tone map without splitting the read-back
across libraries. Neither choice changes what a model sees by more than 0.15 of an 8-bit level, so this is a decision
about speed, exactness and testability, not accuracy. It reverses the owner's ML-19 D31 and amends ADR 0033; the
reason D31 gave, no drift, is met by pinning numpy exactly and testing against both the standards and LittleCMS (§5).

If the owner keeps (b): use a linear Rec.2020 profile built from the primaries (`canonical.LINEAR_REC2020`), not the
committed libjxl profile, whose rounded tags put blue 0.04% high; clip explicitly; and expect the two-transform form
once D6's tone map is more than a clip.

**Unverified.** Whether numpy's float32 `pow` gives bit-identical results on the front, the inference host and the
training machine (D30's three machines); a float64 `pow` cast to float32 would remove the question at some cost in time,
not measured. Timings are from one shared machine, not the N100 host (ML-21 measured that host separately). D1's input
is emulated from today's files; the lossy linear re-encode is not in §3.

## Sources

Standards (fetched into `tmp/ml37/d9/downloads/`, not committed):

- Recommendation ITU-R BT.709-6, items 1.2 and 3.2 (primaries, luminance coefficients). `bt709.pdf`.
- Recommendation ITU-R BT.2020-2, Tables 3 and 4 (primaries, luminance coefficients). `bt2020.pdf`.
- Recommendation ITU-R BT.2087-0 (2015), the BT.709 → BT.2020 conversion and its NOTE 3 on four-decimal rounding.
  `bt2087.pdf`.
- IEC 61966-2-1:1999 and Amd. 1, as quoted by the ICC, "How to interpret the sRGB color space (specified in IEC
  61966-2-1) for ICC profiles" (the encode, its 0.0031308 threshold and the 7-digit XYZ → sRGB matrix).
  `icc_srgb_spec.pdf`.
- SMPTE RP 177-1993, the NPM construction, as written in `tests/test_canonical.py` (the standard itself not fetched).
- ISO 15076-1 (ICC.1) D50 PCS white, as LittleCMS defines it (`lcms2.h` 292–294).

Library source (LittleCMS 2.19 and imagecodecs 2026.10.10, fetched into `tmp/ml37/d9/downloads/`):

- `lcms_cmsvirt.c` 639–670 (`Build_sRGBGamma`, the built-in sRGB profile); `lcms_cmsgamma.c` 496–525 (parametric types 4
  and −4); `lcms_cmswtpnt.c` 235–263 (Bradford to D50); `lcms_cmsopt.c` 252–282 (`PreOptimize`, `_MultiplyMatrix`), 663,
  1062, 1410 (float formats skip optimization), 1690–1693 (matrix-shaper: 8-bit only); `lcms_cmsxform.c` 1164–1165
  (float inhibits the cache), 918 (float formatters); `lcms2.h` 97 (`LCMS_VERSION 2190`).
- `_cms.pyx` 460–497 (`cms_transform`: transform created, applied and freed per call, without the GIL).

Repository:

- `docs/ml-0037_ticket_linear_canonical_photo.html` (D9, §3); `docs/ml-0019_ticket_canonicalization.html` (D8, D31,
  D46); `docs/ml-0021_ticket_canonical_decode_time.html` (today's read-back time); `docs/adr/0033-photos-are-canonicalized-once-at-the-front.md`;
  `coffeecv/canonical.py` 67–75, 740–759; `tests/test_canonical.py`; `uv.lock` (numpy 2.5.3, `imagecodecs` ==2026.10.10).

Measurements (this folder): `d9_matrix.py`/`.txt`, `d9_curve.py`/`.txt`, `d9_realdata.py`/`.txt`, `d9_speed.py`/`.txt`.
