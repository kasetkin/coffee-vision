# ML-29 P4: bringing HDR light into the canonical photo's standard range

Research for [ML-29](../../docs/ml-0029_ticket_what_canonicalization_reads.html) P4 (D2, D10, D11, D21, D25, D27, D29,
R3), 2026-11-15, on the local checkout at `be51eb8`. A recommendation for the owner, not a decision: the operator and
its tolerance become the ticket's next D-number when the owner picks them. The prototype is `coffeecv/tone_map.py`
(nothing imports it, no DVC stage depends on it; `tests/test_tone_map.py`); the numbers come from
`analysis/ml29/tone_map_measure.py`. Facts measured before P4 are cited from [README.md](README.md), not re-derived.

## Summary

- **Where standard white sits is the central question, and the standards answer it**: 203 cd/m² for PQ, 75% HLG
  through the OOTF at 1000 cd/m² (BT.2408-8 §2.1, Table 1; BT.2100-3 Table 5). libavif, libultrahdr and both
  gain-map specifications use it; libjxl's linear 1.0 is the file's intensity target, which its encoder sets to 10,000
  for every PQ file. The owner's darktable (≤ 5.6.2) wrote PQ white at 10,000 cd/m² and HLG with a curve of its own;
  darktable fixed both on 2026-10-05 (commit `4b0878d`, issue #20562), not yet released.
- **No reading puts both the owner's exports and BT.2408 files within D27**: their PQ whites are 49× apart. A pixel
  rule rescues the owner's PQ export but not its HLG one, and guesses where ADR 0040 says honour.
- **Proposed: BT.2408-8 Annex 5's EETF on max(R, G, B), peak measured from the picture, BT.2408's white.** Identical to
  a clip when nothing is above white (made files 0.10–0.19; ≤ 1.2), never worse than the clip above it.
- **Proposed tolerance: D27's own** (mean ≤ 1.0, no block over 6) for HDR files holding a standard-range picture; for
  light above white, behavioural tests and a regression pin, since no standard-range twin is a fixed target.
- **The owner's PQ and HLG exports would fall outside**, recorded with the reason; the made files, which are what
  darktable's next release writes, take their place in D11's check. The alternative is in §6.

## 1. What the standards say

**ITU-R BT.2100-3 (02/2025).** Read from the PDF at
<https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-3-202502-I!!PDF-E.pdf>.
- PQ, Table 4: EOTF F_D = 10000 · (max(E′^(1/m2) − c1, 0) / (c2 − c3 E′^(1/m2)))^(1/m1), with m1 = 2610/16384,
  m2 = 2523/4096 · 128, c1 = 3424/4096, c2 = 2413/4096 · 32, c3 = 2392/4096 · 32; F_D in cd/m², absolute. Note 4a:
  the same curve converts between signal and linear light. PQ carries no statement of where white is.
- HLG, Table 5: OETF E′ = √(3E) for E ≤ 1/12, a ln(12E − b) + c above; a = 0.17883277, b = 1 − 4a, c = 0.5 − a ln(4a)
  (note 5c: b = 0.28466892, c = 0.55991073). E is scene light, relative, [0, 1].
- HLG OOTF, Table 5: F_D = α Y_S^(γ−1) E per component, Y_S = 0.2627 R_S + 0.6780 G_S + 0.0593 B_S, α = L_W, the
  display's nominal peak. γ = 1.2 at 1000 cd/m². Note 5f: γ = 1.2 + 0.42 log10(L_W/1000) from 400 to 2000 cd/m², and
  outside that range the extended formula γ = 1.2 · 1.111^log2(L_W/1000). Note 5i gives the inverse OOTF. EOTF:
  F_D = OOTF[OETF⁻¹[max(0, (1 − β)E′ + β)]], β = √(3 (L_B/L_W)^(1/γ)), the black lift. Note 5e: the OOTF applies gamma
  to luminance; legacy displays that apply it per component only approximate it.

**ITU-R BT.2408-8 (11/2024)**, a Report. <https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-8-2024-PDF-E.pdf>.
- §2.1 and Table 1: HDR Reference White, the level of a 100% diffuse white card, graphics white too, is **203 cd/m²**:
  **58% PQ, 75% HLG** on a 1000 cd/m² HLG display. An 18% grey card is 26 cd/m² (38% PQ, 38% HLG). Table 1 note 2:
  "The signal level of 'HDR Reference White' is not directly related to the signal level of SDR 'peak white'."
- §5.1.2 (SDR into PQ, display-referred): E′ = EOTF_PQ⁻¹[scaling · EOTF_BT.1886[V]], scaling 2.03, so 100% SDR lands
  at 203 cd/m², 58% PQ. This is the inverse of candidate (a) below, and the way `make_test_files.py` made its PQ and
  HLG files ([README.md](README.md), "PQ and HLG").
- §5.2 (HDR to SDR down-mapping): display-light down-mapping "is usually preferred"; BT.2446 has three example methods.
  Table 6 and Annex 10: a "hybrid-linear" down-mapper scales display light linearly, with "compressed highlights
  filling the upper SDR signal range" (§5.2); the 2.03× linear scaling round-trips with it.
- §7.6, Table 10: for PQ to SDR the conversions listed are "hard clip" and "down-mapping".
- **Annex 5, "Displaying PQ – calculating the EETF"** (BT.2390-11 §5.4, 03/2023, now only points to BT.2408 and BT.2446,
  <https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2390-11-2023-PDF-E.pdf>). Step 1 normalises PQ by the mastering
  display's black and white L_B, L_W ("a value of 0 can be used for L_B and a value of 10 000 can be used for L_W" when
  unknown); minLum, maxLum are the target display's black and peak in that scale; step 2 knee KS = 1.5 maxLum − 0.5;
  steps 3–4 a 1:1 mapping below KS and a Hermite spline P[B] above it, ending at maxLum; E3 = E2 + b(1 − E2)⁴; step 5
  undoes the normalisation. Five application spaces (ICtCp, Y′C′bC′r, YRGB, R′G′B′, maxRGB); Table A5.1: only R′G′B′
  and maxRGB keep colours inside the target volume, and maxRGB "preserves chromaticity at the expense of lightness".

**ITU-R BT.2446-1 (03/2021)**, a Report. <https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2446-1-2021-PDF-E.pdf>.
Three HDR-to-SDR methods for broadcast (Table 1): A maps a 1000 cd/m² HDR picture to a 100 cd/m² display through a
knee in a perceptual domain (§4.1, Tables 2–3); B converts at 291 cd/m²; C at 1000 cd/m² to 120. All three put the
whole HDR range into the SDR range, so reference white lands below SDR white. Not run here: built for graded video
at a 1000 cd/m² bridge, not for photos whose standard-range rendering already exists.

**Gain maps (ISO 21496-1).** The standard is paywalled; read through its public descriptions.
- Android's Ultra HDR format v1.1, <https://developer.android.com/media/platform/hdr-image-format>: the primary image
  is the standard-range rendition and "legacy readers … display the conventional low dynamic range image"; "Encode",
  `Yhdr` is "defined in the range 0.0 to max content boost", relative to the standard-range image's white; "Display":
  HDR = (SDR + offset_sdr) · exp2(log_boost · weight_factor) − offset_hdr, log_boost = gain_map_min (1 − r) +
  gain_map_max · r, r = (map/255)^(1/gamma), weight_factor = clamp((log2(max_display_boost) − hdr_capacity_min) /
  (hdr_capacity_max − hdr_capacity_min), 0, 1); "ISO 21496-1 Compatibility": prefer the ISO metadata when both are
  present.
- Apple, "Applying Apple HDR effect to your photos",
  <https://developer.apple.com/documentation/appkit/applying-apple-hdr-effect-to-your-photos>: the map is 8-bit,
  "encoded using the Rec.709 transfer function"; headroom from MakerNote tags 33 and 48; hdr = sdr · (1 + (headroom − 1)
  · gainmap), and "the result … is a linear HDR image with a scale where 1.0 is the reference white and the peak value
  is no greater than the headroom". (`gainmap_in_beans.py` linearised Apple's map with the sRGB curve; this page says
  Rec.709. P7 should follow the page.)
- libavif's `include/avif/avif.h` (`avifGainMap`, "standardized in ISO 21496-1"): a base and an alternate image, each
  with a log2 HDR headroom; for a display headroom H, w = clamp((H − base)/(alternate − base), 0, 1), negative when the
  map leads down. A map from an HDR base down to standard range is the same arithmetic with w < 0.
- Adobe's Gain Map specification (<https://helpx.adobe.com/camera-raw/using/gain-map.html>) refused automated fetching
  (HTTP 403); not read. libavif's code implements its arithmetic.

## 2. What the libraries do when asked for a standard-range rendering

Read from source on GitHub, `main` at the time of writing.

| Library | Where 1.0 (standard white) sits | HLG | Above white |
|---|---|---|---|
| libjxl | linear 1.0 = the file's **intensity target** (`lib/jxl/cms/transfer_functions.h`, `TF_PQ_Base::DisplayFromEncoded` × 10000/intensity_target; `jxl_cms.cc` `BeforeTransform`). The encoder's default intensity target is 10,000 for PQ, 1000 for HLG, 255 otherwise (`lib/jxl/luminance.cc`, `SetIntensityTarget`); `codestream_header.h` calls it an "upper bound on the intensity level present in the image", not a white | OOTF γ = 1.2 · 1.111^log2(target/1000), per luminance (`jxl_cms.cc`, `ApplyHlgOotf`) | nothing unless asked: `JxlDecoderSetDesiredIntensityTarget` turns on the `ToneMapping` stage (`render_pipeline/stage_tone_mapping.cc`): for PQ a `Rec2408ToneMapper` from [0, intensity target] to [0, desired] on luminance (YRGB, BT.2408 Annex 5 with L_B = 0), then `GamutMap` (`cms/tone_mapping.h`); "not meant to be considered authoritative" (`decode.h`). Its generated ICC profile for PQ bakes the same EETF from 10,000 to 255 cd/m² into a table (`jxl_cms_internal.h`, `CreateTableCurve`) |
| libavif | `#define SDR_WHITE_NITS 203.0f`: PQ linear = nits/203 (`src/colr.c`, `avifToLinearPQ`) | per-channel E^1.2 × 1000/203 (`avifToLinearHLG`, "Inverse OETF followed by the OOTF", approximating Y_S by each channel) | clipped: `avifImageApplyGainMap` with no map weight converts and clamps (`src/gainmap.c`, `avifNanSafeClamp`); otherwise only a gain map brings it down |
| libultrahdr | `kSdrWhiteNits = 203` (`lib/include/ultrahdr/gainmapmath.h`); PQ normalised to 10,000, HLG to 1000, then × peak/203 | `hlgOotfApprox`, per channel, γ 1.2 (`gainmapmath.cpp`) | when given an HDR picture alone it makes the standard-range one by `globalTonemap` (`lib/src/jpegr.cpp`): extended Reinhard on max(R, G, B), headroom = 10,000/203 (PQ) or 1000/203 (HLG), m′ = m (1 + m/h²)/(1 + m), hue kept, then a hard clip; it maps reference white to about 0.5 |
| darktable ≤ 5.6.2 (the owner's) | PQ 1.0 = 10,000 cd/m² = standard white; HLG: signal 1.0 = standard white, `_HLG_fct` = OETF⁻¹(0.96 E′ + 0.04), no OOTF (`src/common/colorspaces.c`) | | |
| darktable master, since 2026-10-05 | 203 cd/m² for PQ, and the OOTF at 1000 cd/m² for HLG (commit `4b0878d`, "Fix PQ luminance scaling and HLG display conversion", `src/common/hdr-transfer.h`: `DT_HDR_REFERENCE_WHITE_NITS 203.0f`); fixes issue #20562, "PQ and HLG transfer function problems (too dark import, too bright export)": "exports 50x too bright". In no release yet (5.6.2 is 2026-10-04) | | |

So every library and both public specifications put standard white at 203 cd/m² (PQ) and at 75% HLG through the
OOTF at 1000 cd/m², except libjxl, whose linear 1.0 is the file's intensity target, a peak that its own encoder sets to
10,000 for any PQ file. darktable wrote its standard white at PQ's top until a fix of last month, which now agrees with
the rest.

## 3. What darktable did, and how much lies above white

From [README.md](README.md): darktable's PQ Rec2020 JPEG XL decodes to the sRGB export's light within 0.2% above
Y 0.02 when PQ's 10,000 cd/m² is taken as 1; its HLG one, through the inverse OETF alone, is 3–7% darker. The source
(§2) explains both: darktable up to 5.6.2 builds its PQ and HLG output profiles from `_PQ_fct` and `_HLG_fct`, with no
luminance scaling and no OOTF, and its HLG curve lifts black: linear = OETF⁻¹(0.96 E′ + 0.04). Decoded that way the
owner's HLG export agrees with the sRGB export's luminance within 1.4% from Y 0.003 to 0.01 and within 0.3% above Y
0.01 (median ratio per luminance band, quarter size, `tone_map_measure.py bands`; and the "HLG darktable ≤ 5.6.2" row
below, mean 0.09 levels); the PQ export read with 10,000 cd/m² as white, within 2.1% and 0.6%.

Both files therefore hold a standard-range picture placed at the top of the signal, nothing above its white:
`tone_map_measure.py darktable` (the logs' first line per file), quarter size:

| File | Read as | Share of pixels above reference white | Median luminance | 99.9th percentile | Signal at its top (`bands`) |
|---|---|---|---|---|---|
| darktable PQ Rec2020, intensity target 10,000 | BT.2408 (203 cd/m² = white) | 94.4% | 2332 cd/m² | 6796 cd/m² | 0.004% of pixels at 254 or more |
| darktable HLG Rec2020, intensity target 1000 | BT.2408 (OOTF at 1000 cd/m², 75% = white) | 37.5% | 163 cd/m² | 616 cd/m² | 0.003% at 254 or more |
| the files made here, PQ and HLG (`make_test_files.py`) | BT.2408 | 0.0% | 47 cd/m² | 135 cd/m² | |
| `pq-rec2020-highlights-to-812nits-16bit.jxl` | BT.2408 | 1.1% | 47 cd/m² | 290 cd/m² | |

Read by their own (old darktable) convention, neither has anything above white: 0.002% of the sRGB export's pixels
have a channel at 255 (`bands`). Read as BT.2100 and BT.2408 say, which is how libavif, libultrahdr and darktable's own
next release read PQ, every light in the owner's PQ export is 10,000/203 = 49 times too bright, its median at
2332 cd/m² where the made files' is 47 (issue #20562: "exports 50x too bright"). darktable master's `dt_hdr_encode`
uses the same constants as `make_test_files.py` (203 cd/m², 1000 cd/m², γ 1.2), so the made PQ and HLG files are what
the owner's darktable will write once it carries the fix.

## 4. The candidates (`coffeecv/tone_map.py`)

Each is a pure function on linear Rec.2020 light, shape (…, 3), 1.0 the reference white, taking `peak` (the brightest
light it must bring back, relative to white, at least 1) and returning [0, 1]:

- **(a) `clip_at_white`**: white to white, each channel clipped at 1. libavif's rendering; BT.2408 Table 10's "hard
  clip".
- **(b) `eetf_bt2408`**: BT.2408-8 Annex 5, mastering black 0 and white `peak` × 203 cd/m², target black 0 and white
  203 cd/m², on max(R, G, B) (option 5). Knee (`eetf_knee`): 0.52 of white for a peak of 707 cd/m², 0.48 for 812,
  0.43 for 1000, 0.12 for 10,000. Below the knee, identity; above, a spline that reaches white at the peak.
- **(c) `scale_by_peak`**: everything divided by the peak. With the peak declared by a JPEG XL's intensity target this
  is libjxl's own linear reading ("c-declared").
- **(d) `reinhard_max_rgb`**: libultrahdr's `globalTonemap`, with libultrahdr's own peak (10,000/203 PQ, 1000/203 HLG).

The reference-white rules, for PQ: BT.2408's 203 cd/m² (`pq_to_relative`); the file's declared peak as white (= (c)
declared); `measured_pq_white`, 203 unless the picture's median luminance exceeds 406 cd/m², then 10,000. For HLG: the
OOTF at 1000 cd/m² with γ 1.2 and 203 as white (`hlg_to_relative`), or scene light with 75% as white (no OOTF); and,
to explain the owner's file, darktable ≤ 5.6.2's own curve.

"measured" peaks below are the 99.99th percentile of max(R, G, B); "declared" is a JPEG XL's intensity target, an AVIF's
`clli` MaxCLL, else PQ's 10,000 or HLG's 1000 cd/m² (for a gain map, its largest boost).

## 5. Measurements

All three sets: `python analysis/ml29/tone_map_measure.py darktable | libjxl | public | gainmaps`. Cells are D27's
metric, **mean; worst** block, in 8-bit levels, as the models read the result (the script's docstring has the path). Every
candidate here reads PQ at 203 cd/m² and HLG through the OOTF at 1000 cd/m²; the white rules follow each table.
**Speed-up, the same on both sides:** set (i)'s full-size files, the sRGB reference included, are tone-mapped at their
own size and area-averaged in linear light to a quarter (1746×1164, as `make_test_files.py` made its files), then
encoded; so the made files' 0.94 of P2 (README.md), which came from averaging the reference in 8-bit sRGB, falls to
0.10–0.19 here.

### (i) The owner's darktable exports and the files made from them, against the 8-bit sRGB export

| File | (a) clip | (b) EETF, measured peak | (c) scale, measured peak | (c) scale, declared peak | (d) Reinhard, libultrahdr's peak |
|---|---|---|---|---|---|
| darktable PQ Rec2020 (8-bit JXL) | 125.07; 192.0 | 94.23; 166.1 | 6.62; 11.6 | **0.09; 0.6** | 90.78; 160.1 |
| darktable HLG Rec2020 (8-bit JXL) | 74.08; 106.5 | 61.52; 90.8 | 8.77; 18.3 | 18.77; 25.8 | 34.08; 48.6 |
| PQ P3 16-bit JXL (made) | **0.11; 0.9** | **0.11; 0.9** | **0.11; 0.9** | 110.01; 192.3 | 14.81; 45.8 |
| PQ Rec2020 10-bit AVIF (made) | **0.19; 1.2** | **0.19; 1.2** | **0.19; 1.2** | 110.00; 192.2 | 14.85; 45.7 |
| PQ Rec2020 12-bit HEIC (made) | **0.15; 1.2** | **0.15; 1.2** | **0.15; 1.2** | 110.02; 192.2 | 14.90; 45.8 |
| HLG P3 16-bit JXL (made) | **0.11; 0.8** | **0.11; 0.8** | **0.11; 0.8** | 66.87; 116.3 | 14.14; 43.8 |
| HLG Rec2020 10-bit AVIF (made) | **0.11; 0.7** | **0.11; 0.7** | **0.11; 0.7** | 66.88; 116.2 | 14.18; 43.7 |
| HLG Rec2020 12-bit HEIC (made) | **0.10; 0.8** | **0.10; 0.8** | **0.10; 0.8** | 66.90; 116.3 | 14.20; 43.7 |
| PQ Rec2020 with highlights to 812 cd/m² (made) | 7.37; 50.6 | 6.99; 39.8 | 32.58; 48.6 | 108.50; 179.1 | 10.78; 19.6 |

Bold: within D27 (mean ≤ 1.0, no block over 6). Measured peaks: 1.00 for every made file but the highlights one
(2.17); 44.0 and 4.05 for darktable's PQ and HLG. Declared: 49.26 (intensity target or PQ's 10,000) for every PQ file,
4.93 for HLG.

- With nothing above white, (a), (b) and (c) with the picture's own peak are the same function, and every made file,
  in all three containers, lands within D27 with room: mean 0.10–0.19, worst 0.7–1.2.
- (d) halves white by design (§4) and misses by 14–15 levels on every one. libjxl's own rendering misses too:
  asked for 203 cd/m² (`libjxl` run; 255 in brackets), PQ P3 6.24; 31.1 (15.11; 39.8), the highlights file 2.05; 5.1
  (10.66; 16.4), HLG P3 44.55; 89.9 (47.81; 93.6): its EETF from the intensity target, 10,000, puts the knee at 0.12 of
  white.
- The highlights file cannot match its reference by any operator: it was brightened below white too (up to ×1.6 at
  Y 0.5), and only 1.1% of it lies above white. Its numbers compare the candidates' treatment of the boost, not their
  fidelity; of the two that keep white, (b) has the smaller worst block (39.8 against 50.6).

**The white rules, with (a):**

| File | PQ 203 cd/m² (BT.2408) | PQ white at 10,000 | PQ measured (`measured_pq_white`) | HLG OOTF at 1000, 75% white | HLG scene light, 75% white | HLG as darktable ≤ 5.6.2 | libjxl's rendering to 203 cd/m² |
|---|---|---|---|---|---|---|---|
| darktable PQ | 125.07; 192.0 | **0.09; 0.6** | **0.09; 0.6** | | | | 114.25; 173.7 |
| darktable HLG | | | | 74.08; 106.5 | 79.20; 106.8 | **0.09; 1.2** | 1.54; 3.8 |
| made PQ (three files) | **0.11–0.19; ≤ 1.2** | 110.0; 192.3 | **0.11–0.19; ≤ 1.2** | | | | 6.24; 31.1 (JXL) |
| made HLG (three files) | | | | **0.10–0.11; ≤ 0.8** | 14.6–14.7; 22.5 | 46.1; 92.8 | 44.55; 89.9 (JXL) |

- No single standard reading passes both the owner's exports and the made files: they put white 49× apart in PQ.
- The measured PQ rule passes both: the owner's PQ median is 2332 cd/m², the made files' 47, Seine's 136, the colours'
  249. The rule ran with its threshold at 2 × 203 cd/m²; any factor from 1.23 to 11.5 sorts these files alike.
- No rule read from the pixels separates darktable's old HLG from BT.2408 HLG: its median (163 cd/m²) and share above
  white (37.5%) lie among the public samples' (136–250 cd/m², 46–57%). Only darktable's own old curve reads it (0.09).
  libjxl's rendering comes nearest among the others (1.54; 3.8), because its OOTF towards 203 cd/m² has γ < 1.
- Scene light without the OOTF misses the made HLG files by 14.6 levels: the OOTF is part of reading HLG.

### (ii) Public same-scene pairs (libavif's samples)

| File | (a) clip | (b) EETF, measured peak | (c) scale, measured peak | (c) scale, declared peak | (d) Reinhard |
|---|---|---|---|---|---|
| seine_hdr_rec2020 (peak 2.82; MaxCLL 703) | 20.95; 47.1 | 16.32; 30.7 | 37.69; 64.4 | 49.82; 75.8 | 15.71; 27.7 |
| seine_hdr_srgb | 21.01; 47.1 | 16.58; 30.8 | 36.96; 63.7 | 50.05; 76.2 | 15.56; 27.1 |
| seine_hdr_gainmap_srgb (its base) | 21.02; 47.1 | 16.53; 30.6 | 37.63; 64.4 | 50.02; 76.0 | 15.52; 27.1 |
| colors_hdr_rec2020 (peak 2.35; MaxCLL 470) | 35.91; 132.4 | 19.27; 70.7 | 1.70; 4.1 | 1.46; 2.9 | 15.80; 38.1 |
| colors_hdr_p3 | 35.96; 133.4 | 19.27; 70.7 | 1.59; 3.5 | 1.37; 3.0 | 15.75; 37.9 |
| colors_hdr_srgb | 36.21; 134.2 | 19.28; 70.7 | 1.54; 4.1 | 2.02; 5.3 | 15.64; 37.7 |

The two pairs were made in opposite directions, and no operator fits both:
- **Seine** is a BT.2408-style HDR photo (45.6% of it above white, median 136 cd/m²) whose standard-range JPEG Photoshop
  rendered separately. Per luminance band (`tone_map_measure.py bands`): in the midtones (standard-range Y 0.1–0.4)
  the HDR file holds 197–209 cd/m² per unit of standard-range light, BT.2408's white; the standard-range rendering then rolls the
  highlights off (Y 0.6–0.8 holds 290 cd/m², 1.4 × white) and deepens the shadows. A roll-off fits it best: (b) and (d)
  15.7–16.6, (a) 21, a scale by the peak 37.
- **Colours** is a standard-range image brightened into HDR in Camera Raw: 470–485 cd/m² per unit of standard-range
  light at every level (`bands`), white at its peak. Only a scale by the peak fits (mean 1.4–2.0, worst 2.9–5.3 with
  the declared MaxCLL).
- What the producer's own standard-range rendering did is not in the HDR file. Seine's carries a map that leads back to
  it (`seine_hdr_gainmap_srgb.avif`, an ISO 21496-1 `tmap` item); reading it is P7's (D10), not measured here.

### (iii) Dataset gain-map photos, against today's canonical photo

18 photos: 5 per camera, spread over the bean-region boost `gainmap_in_beans.py` measured (its 10th to 90th
percentile per camera); two had no canonical photo or mask in the store at the time of the run (a OnePlus 7 Pro and a
Samsung). Full size, no downscale. The map applied at full weight (Ultra HDR's formula for the JPEGs, Apple's for the
HEICs, its map undone by the Rec.709 curve as Apple's page says), then each candidate with the peak it takes: measured
(1.39–3.06 × white, the 99.99th percentile), or the map's own largest boost for (c) declared and (d). Median photo per
camera; "shift" positive is brighter; worst is the largest block of any photo. Three of the five Google photos are
close-ups whose mask covers the whole frame, so Google's "outside" is the median of two.

| Camera | Candidate | Beans: mean | shift | worst | Outside: mean | shift |
|---|---|---|---|---|---|---|
| Apple | none (the path alone) | 0.06 | +0.00 | 0.8 | 0.03 | +0.00 |
| Apple | (a) clip | 22.54 | +22.54 | 71.1 | 18.76 | +18.76 |
| Apple | (b) EETF | 21.06 | +21.06 | 63.7 | 14.20 | +14.20 |
| Apple | (c) scale, measured | 23.05 | −23.05 | 46.7 | 19.00 | −19.00 |
| Apple | (c) scale, map's top | 27.54 | −27.54 | 83.5 | 22.74 | −22.74 |
| Apple | (d) Reinhard | 6.42 | +5.67 | 34.1 | 4.64 | +1.62 |
| OnePlus 7 Pro (4) | none | 0.04 | +0.00 | 0.5 | 0.03 | +0.00 |
| OnePlus 7 Pro | (a) clip | 16.27 | +16.27 | 52.4 | 13.98 | +13.98 |
| OnePlus 7 Pro | (b) EETF | 15.69 | +15.69 | 49.2 | 13.97 | +13.97 |
| OnePlus 7 Pro | (c) scale, measured | 3.54 | −3.54 | 42.6 | 3.55 | −3.51 |
| OnePlus 7 Pro | (c) scale, map's top | 3.81 | −3.81 | 43.7 | 3.68 | −3.68 |
| OnePlus 7 Pro | (d) Reinhard | 6.56 | +6.56 | 21.8 | 7.03 | +7.03 |
| Google | none | 0.04 | +0.00 | 0.5 | 0.03 | +0.00 |
| Google | (a) clip | 15.41 | +15.41 | 52.5 | 21.51 | +21.51 |
| Google | (b) EETF | 14.87 | +14.87 | 51.6 | 19.29 | +19.29 |
| Google | (c) scale, measured | 11.25 | −11.25 | 40.4 | 16.01 | −15.88 |
| Google | (c) scale, map's top | 12.34 | −12.34 | 41.2 | 16.37 | −16.37 |
| Google | (d) Reinhard | 2.89 | +2.75 | 24.1 | 4.50 | +0.55 |
| Samsung (4) | none | 0.04 | +0.00 | 0.9 | 0.03 | +0.00 |
| Samsung | (a) clip | 21.90 | +21.90 | 82.9 | 33.07 | +33.07 |
| Samsung | (b) EETF | 21.86 | +21.86 | 58.8 | 29.21 | +29.21 |
| Samsung | (c) scale, measured | 14.03 | −14.03 | 34.8 | 16.63 | −16.63 |
| Samsung | (c) scale, map's top | 26.27 | −26.27 | 74.9 | 36.72 | −36.72 |
| Samsung | (d) Reinhard | 13.69 | +13.69 | 24.9 | 14.33 | +12.00 |

- The path itself is faithful: with no map, 0.04–0.06 levels from today's canonical photo, no block over 0.9.
- **In the bean region, (a) and (b) are the same**: the beans' boosted light stays below the EETF's knee (0.56–0.85 of
  white for these peaks, `eetf_knee`), so the EETF only differs from a clip outside it (14.2 against 18.8 on Apple,
  29.2 against 33.1 on Samsung). Either brightens the median photo's beans by 15–23 levels: the map's boost, kept.
- (c) darkens everything instead (4–28 levels); (d) is nearest to today (3–14 levels) because it roughly undoes the
  boost: Reinhard's white sits near 0.5, the beans' boost near 1.3–1.8× ([README.md](README.md)). Neither is "the
  camera's HDR picture brought back"; the main image already is the camera's own standard-range rendering (Ultra HDR
  spec, "Introduction"; Apple's page).
- Whether brighter beans help the models is P8–P9's question (D12), not P4's; P4's operator decides only how the light
  the map adds above white comes back.

## 6. Recommendation

Proposed, for the owner to accept or change (the ticket's next D-number):

**Operator: (b), BT.2408-8 Annex 5's EETF on max(R, G, B)**, `tone_map.eetf_bt2408`: mastering black 0, mastering
peak = the picture's own peak (the 99.99th percentile of max(R, G, B) in relative light, at least white), target black
0 and white = the reference white; linear below the knee, the Hermite roll-off above it, every colour scaled alike.
The same operator for PQ, HLG, a float JPEG XL's values above white (D21) and an applied gain map (D10, D11).
- Where nothing lies above white (every standard-range picture written into PQ or HLG by BT.2408, and every photo P5
  must not change) it is exactly (a), a clip that clips nothing: the made files land at 0.10–0.19; worst 0.7–1.2, in
  JPEG XL, AVIF and HEIC, 10 to 16 bits.
- Where light lies above white it is the method the standards name for PQ on a lower-peak display, and on every set
  its error is at or below the clip's: highlights file 6.99; 39.8 against 7.37; 50.6, Seine 16.3; 30.7 against 21.0;
  47.1, colours 19.3; 70.7 against 35.9; 132, gain maps outside the beans 14.0–29.2 against 14.0–33.1 (median photo per camera).
  It keeps hue and stays in gamut (Annex 5, Table A5.1), which a per-channel clip does not.
- Not (c): it darkens the whole picture as soon as one pixel is above white (Seine −37.7, the highlights file −32.6).
  Not (d): it halves reference white by design, 14–15 levels off on files with nothing above white. Not libjxl's own
  rendering: it starts from the intensity target, which libjxl's encoder sets to 10,000 for any PQ file, and rolls off
  from 0.12 of white (6.24; 31.1 on the made PQ file).
- The peak is measured, not declared: no declaration reliably carries it (a JPEG XL's intensity target defaults to
  10,000; nclx has none; a `clli` box is optional and absent from the files made here). It costs P5 one pass over the
  decoded picture before the encode; P5 measures that on the front (D26).

**Reference white: BT.2408's.** PQ: 203 cd/m² is 1.0 (`pq_to_relative`). HLG: BT.2100's OOTF at a nominal
1000 cd/m², γ 1.2, black 0, luminance weights from the file's own primaries (as libjxl does, and as the made P3 files
were written), 203 cd/m² (75% HLG) is 1.0 (`hlg_to_relative`). This is what libavif, libultrahdr, Apple's and
Android's gain maps and darktable's next release assume, and the scene-light reading without the OOTF misses the made
HLG files by 14.6 levels.

**The owner's two darktable exports then fall outside, and the reason is the files.** darktable up to 5.6.2 wrote PQ
with its standard white at 10,000 cd/m² and HLG with a non-standard curve (§2, §3); darktable calls this a bug (#20562)
and has fixed it for its next release, which will write what `make_test_files.py` wrote. Read as BT.2100 says, the
owner's PQ export is 49 times too bright, as any standards-following viewer shows it. Three ways for the owner:
1. **Recommended: honour the declaration.** The owner's PQ and HLG exports are recorded as out of tolerance, with this
   reason, and D11's test set becomes the made PQ and HLG files (darktable's own corrected encoding, in all three
   containers). ADR 0040's rule, "honoured or refused, never guessed", holds unchanged.
2. Add `measured_pq_white`: when a PQ picture's median luminance is above about 1000 cd/m² (5 × 203; these files
   separate anywhere from 1.23 × to 11.5 × 203), white is read at 10,000 cd/m². The owner's PQ export then reads
   within 0.09; 0.6, and nothing else measured here changes. It is a guess from the pixels, which ADR 0040 rules out for
   declarations, and a real HDR photo that bright would be read 49 times too dark.
3. Nothing reads the owner's HLG export within tolerance but darktable's own old curve (0.09; 1.2), and nothing in the
   file (a bare codestream, [README.md](README.md)) says it came from darktable. Under 1 and 2 alike it stays outside.

**Tolerance for HDR files, proposed.** For a PQ or HLG file that holds a standard-range picture by BT.2408 (white at
203 cd/m² or 75%, nothing above it): **D27's bar unchanged, mean ≤ 1.0 level and no block over 6**, against the
standard-range file of the same picture; measured 0.10–0.19 and 0.7–1.2, so it leaves room for 8-bit HDR files. For
light above white no tolerance against a standard-range twin is meaningful: the two public pairs' own standard-range
renderings pull opposite ways: the operator nearest Seine's (d, 15.6) misses the colours' by 15.8, and the nearest the
colours' (c, 1.4–2.0) misses Seine's by 37–50, because each producer made its rendering its own way. There the check
is the operator's behaviour (`tests/test_tone_map.py`: identity below the knee, the peak to white, monotonic, hue kept)
and, as a regression guard for P5, the highlights file within 0.5 level of P4's 6.99; 39.8.

What this does not settle: gain maps (Part 2). With (b), applying a map brightens the median photo's beans by 15–23
levels and leaves the main image's own rendering behind; P8 and P9 measure whether the models gain from that, and the
owner decides there (D12, D16).

## Files

- `coffeecv/tone_map.py`, `tests/test_tone_map.py` (21 tests: `python -m unittest tests.test_tone_map`).
- `analysis/ml29/tone_map_measure.py`: every number in §3 and §5.
- Local only, never committed (D29): the public samples in `tmp/ml29_samples/`, per-file rows in `tmp/ml29_tm/`, the
  owner's page `tmp/ml29_tonemap_crops/index.html`, the sources read in `tmp/ml29_src/`.

### Public samples (D29: fetched into `tmp/ml29_samples/`, not committed)

From libavif's `tests/data` at commit `37a5461042aa3b1eb105193724ee41bfd2478d0b`,
`https://raw.githubusercontent.com/AOMediaCodec/libavif/37a5461042aa3b1eb105193724ee41bfd2478d0b/tests/data/<file>`.
Licence line for each, from `tests/data/README.md`:
"License: [same as libavif](https://github.com/AOMediaCodec/libavif/blob/main/LICENSE)" (BSD 2-clause).

| File | What it is (README) | Declaration read (`colr`, `clli`) | Size |
|---|---|---|---|
| `seine_hdr_rec2020.avif` | a photo made HDR in Photoshop (Camera Raw 16.0.1), PQ, Rec.2020 | nclx 9/16, 10-bit; MaxCLL 703, MaxFALL 165 cd/m² | 400×300 |
| `seine_hdr_srgb.avif` | the same, sRGB primaries | nclx 1/16, 10-bit; MaxCLL 707, MaxFALL 165 | 400×300 |
| `seine_hdr_gainmap_srgb.avif` | HDR base (PQ) with an ISO 21496-1 map down to standard range (libavif's test generator) | nclx 1/16, 10-bit; MaxCLL 707; a `tmap` item | 400×300 |
| `seine_sdr_gainmap_srgb.jpg` | the same photo saved by Photoshop as a standard-range JPEG with a gain map: **the standard-range reference** | ICC profile (sRGB) | 400×300 |
| `seine_sdr_gainmap_srgb.avif` | that JPEG re-encoded by avifenc, ISO map up | nclx 1/13, 8-bit; a `tmap` item | 400×300 |
| `colors_sdr_srgb.avif` | a 32-bit Photoshop image exported as PNG, then AVIF: **the standard-range reference** | nclx 1/13, 8-bit | 200×200 |
| `colors_hdr_srgb.avif`, `colors_hdr_p3.avif`, `colors_hdr_rec2020.avif` | the same image, brightened in Camera Raw's HDR mode ("drag the histogram towards the right"), PQ | nclx 1/16, 12/16, 9/16, 10-bit; MaxCLL 486, 470, 470 | 200×200 |
| `colors_wcg_hdr_rec2020.avif` | wide-gamut colours, PQ (fetched, not measured: no standard-range twin) | nclx 9/16; MaxCLL 506 | 200×200 |
