# ML-29: measurements behind what canonicalization reads from an original

Measured 2026-11-13 and 2026-11-14 on the local checkout at `2fce5af`, for
[ML-29](../../docs/ml-0029_ticket_what_canonicalization_reads.html). Everything was read from the files' bytes, never
from their names. Per-file rows are local scratch (`tmp/ml29_scan/`, not committed); the counts are here.

## The owner's darktable exports (one Canon EOS 90D photo, 6984×4660)

Eleven JPEG XL, five AVIF and six HEIC exports of one photo, read with `coffeecv.jxl_header.read_header`, an ISO BMFF
box reader (the one in `scan_dataset.py`) and `coffeecv.canonical`. "Error" is the mean absolute difference, in 8-bit
levels over 32×32 block means, between what the models would read from the file and from the 8-bit sRGB JPEG XL.
The one-off scripts that made the two tables were not kept.

JPEG XL: every file a bare lossy codestream, no boxes, the colour as a header enumeration, never an ICC profile.

| darktable profile | Samples | Header (primaries, transfer) | Today |
|---|---|---|---|
| sRGB | 8, 10, 12, 16-bit integer | sRGB, sRGB | read; all four reach the models alike (mean 123.16) |
| sRGB | 32-bit float, values −0.295 to 1.074 | sRGB, sRGB | refused: float |
| linear Rec2020 | 8, 16-bit | BT2100, Linear | refused |
| Display P3 | 8-bit | P3, sRGB | refused |
| Adobe RGB | 8-bit | Custom primaries, explicit gamma 0.454707 | refused |
| HLG Rec2020 | 8-bit | BT2100, HLG | refused |
| PQ Rec2020 | 8-bit | BT2100, PQ | refused |

AVIF and HEIC: accepted every time, and read as sRGB whenever the file has no ICC profile.

| File | Colour declaration in the file | Today | Error |
|---|---|---|---|
| AVIF sRGB, 8, 10, 12-bit | coded tag 1/13 (sRGB) | read as sRGB; 10- and 12-bit decoded at 8 bits | 0.46–0.60 |
| AVIF Adobe RGB, 8-bit | ICC profile, and a coded tag 2/2 (unspecified) | read through the ICC | 0.43 |
| AVIF linear Rec2020, 8-bit | coded tag 9/8 | **read as sRGB** | **64.5** |
| HEIC sRGB, 8, 10, 12-bit | coded tag 1/13 | read as sRGB | 0.32–0.38 |
| HEIC Adobe RGB, 12-bit | ICC profile | read through the ICC | 0.41 |
| HEIC Display P3, 12-bit | coded tag 12/13 | **read as sRGB** | **2.2** (largest block 8.2) |
| HEIC linear Rec2020, 12-bit | coded tag 9/8 | **read as sRGB** | **64.7** |

## The dataset: colour, bit depth, second images (`scan_dataset.py`)

    python analysis/ml29/scan_dataset.py

2,851 photos under `dataset/`: 2,556 JPEG, 164 HEIC, 72 DNG, 20 PNG, 19 JPEG XL, 19 CR3, 1 GIF. About 30 s; nothing
is decoded.

**What each photo's colour is taken from today.** No dataset photo is misread.

- JPEG: 1,715 through an ICC profile (Display P3 1,081, sRGB IEC61966-2.1 547, DCI-P3 D65 with sRGB transfer 50,
  "c2" 13, Display P3 with sRGB transfer 9, Adobe RGB 7, sRGB 5, Canon's sRGB 2, "Camera RGB Profile" 1); 841 with
  none, read as sRGB. Of those 841, none carries the R03 (Adobe RGB) interoperability tag; one says EXIF
  "Uncalibrated" and nothing else.
- HEIC: all 164 carry a Display P3 ICC profile on the main image.
- PNG: all 20 without an ICC profile. 9 use the legacy chunks, all with sRGB's own chromaticities, 7 of them with
  gamma 2.2; 1 has the sRGB chunk; none has the coded tag (cICP).
- JPEG XL: 19, enumerated sRGB, 12-bit. DNG and CR3: developed by rawpy into linear Rec.2020.
- Bit depth: every JPEG, HEIC and PNG is 8-bit. No main image anywhere is PQ or HLG.

**Gain maps: 1,731 photos (61%).** In every one the main image is standard-range.

| Kind | Photos | Cameras |
|---|---|---|
| Ultra HDR JPEG (hdrgm XMP, MPF second image) | 1,570 | Google 984, GM1911 (OnePlus 7 Pro) 536, Samsung 50 (these 50 also carry ISO 21496-1 metadata) |
| Apple HEIC auxiliary image | 161 | Apple; 158 also carry an ISO 21496-1 `tmap` item |

- JPEG maps: one component (brightness only); `GainMapMin` 0 in all; `GainMapMax` 0.16 to 2.58 stops, median 0.74
  (Google 0.74, OnePlus 7 Pro 0.71, Samsung 2.3); sizes about a quarter of the photo's width (768×1020, 750×1000) up
  to a half.
- Apple maps: 8-bit, half the photo's width (2016×1512 of 4032×3024 in 142); headroom 1.79× to 4.55×, median 2.69×
  (158 with the tag).
- No gain map: Sony 455, the other OnePlus 262, every DNG, CR3, JPEG XL and PNG.

**Other second images.** HEIC thumbnail 157; depth or disparity 8; sky matte 6, style delta map 6, linear thumbnail
6, portrait and skin mattes 2. Motion photo (a video after the image) 27. DNG local tone map
(`ProfileGainTableMap`) 71, which rawpy does not apply.

**Capture metadata**, none of it in a canonical photo: subject distance 1,687 photos, 35 mm-equivalent focal length
2,181, exposure time, f-number and ISO about 2,655, brightness value 2,124.

**The Sony photos.** 454 of the 455 come from one phone, model G8441, on Android 10 firmware. Each file holds
exactly one JPEG, nothing after its end marker, no MPF, no `hdrgm`, no "GainMap", no ISO 21496-1 text; `CustomRendered`
is "Normal" in all. The phone's HDR setting merges exposures into the one standard-range JPEG and leaves nothing in
the file. OnePlus's own flag (`IsHDRActive`: true 29, false 220) sits on photos that have no gain map either.

## How much a gain map would brighten the bean region (`gainmap_in_beans.py`)

    python analysis/ml29/gainmap_in_beans.py

The 452 hand-checked full-photo masks of `data/seg_masks_ml19`: 247 belong to a photo with a gain map, 206 were
measured (40 have an empty mask, a photo with no beans; one HEIC has no headroom tag). The boost is what the map asks
for at full HDR, in stops (log2 of HDR light over standard-range light), at the map's own resolution. JPEG by the
Ultra HDR formula; HEIC by Apple's published formula, not the ISO item.

| Camera | Photos | Mean boost in the bean region (median photo) | Spread inside one region, p95 − p5 (median photo; worst) | Photos with a constant boost over the region | Mean boost outside it |
|---|---|---|---|---|---|
| Google | 100 | 0.37 | 0.00; 0.59 | 50 | 0.37 |
| GM1911 (OnePlus 7 Pro) | 70 | 0.43 | 0.00; 0.42 | 37 | 0.43 |
| Samsung | 12 | 0.86 | 0.03; 0.28 | 1 | 0.94 |
| Apple | 24 | 0.44 | 0.44; 0.66 | 0 | 0.42 |
| All | 206 | 0.43 | 0.02; 0.66 | 88 | 0.43 |

- The maps are not near zero on beans, as had been guessed: the median photo's bean region is boosted by 0.43 stops.
- On Google and OnePlus 7 Pro the boost is one value over the whole region in about half the photos (in 39 of each,
  the map's maximum), and varies by under 0.16 stops in three quarters and under 0.28 in nine tenths: close to a
  plain brightness shift. Samsung's is larger and nearly flat.
- Only Apple's maps vary inside the bean region in every photo.
- Outside the region the spread is larger (median 0.51 stops on Google, 0.48 on Apple, 0.38 on Samsung).
- Applying the maps would brighten each brand by its own amount and leave Sony, the other OnePlus and the raws as
  they are.

## Files made here (D25), not by darktable (`make_test_files.py`)

    python analysis/ml29/make_test_files.py

Ten files in `tmp/ml29_made/` (local, not committed), about 30 s. All are the 16-bit linear Rec.2020 JPEG XL export,
area-averaged in linear light to a quarter of each side (1746×1164; the bottom 4 rows of 4660 dropped), re-encoded
lossy at quality 90 with another declaration. Everything in the table was read from the files' bytes:
`coffeecv.jxl_header` and libjxl's decoder for JPEG XL, the primary item's `colr` and `pixi` boxes (the reader in
`scan_dataset.py`) for AVIF and HEIC. "Error" is the made file decoded to linear Rec.2020 by the script's own maths,
from the declaration read back, against what it was meant to hold, in linear light with white 1: the mean per sample,
then the largest 8×8 block mean. Most of it is the lossy coding; decoded with sRGB primaries instead of BT.2020 the
PQ AVIF's largest block would be 0.063, as HLG instead of PQ 0.48. The last column is libjxl's own conversion of the
same file to linear Rec.2020 against the script's maths, at the 99.9th percentile (libjxl's float output keeps the
codec's overshoot past 0 and 1, which the integer samples clip).

| File | How made | Declaration read back | Bits | Error | Against libjxl |
|---|---|---|---|---|---|
| `linear-rec709-16bit.jxl` | `jpegxl_encode(primaries=1, transfer=8)` | sRGB primaries, D65, Linear | 16 | 0.0118; 0.0118 | 7e-06 |
| `rec709-16bit.jxl` | `jpegxl_encode(primaries=1, transfer=1)` | sRGB primaries, D65, BT709 | 16 | 0.0118; 0.0123 | 1e-05 |
| `pq-p3-16bit.jxl` | `jpegxl_encode(primaries=11, transfer=16)` | P3, D65, PQ; intensity target 10000 | 16 | 0.0120; 0.0157 | 4e-05 |
| `hlg-p3-16bit.jxl` | `jpegxl_encode(primaries=11, transfer=18)` | P3, D65, HLG; intensity target 1000 | 16 | 0.0120; 0.0146 | 2e-05 |
| `linear-prophoto-d50-16bit.jxl` | libjxl's `JxlEncoderSetColorEncoding` over ctypes | Custom primaries R 0.7347,0.2653 G 0.1596,0.8404 B 0.0366,0.0001; Custom white 0.3457,0.3585; Linear | 16 | 0.0118; 0.0123 | 1e-05 |
| `pq-rec2020-highlights-to-812nits-16bit.jxl` | `jpegxl_encode(primaries=9, transfer=16)` | BT2100, D65, PQ; intensity target 10000 | 16 | 0.0145; 0.0273 | 8e-05 |
| `pq-rec2020-10bit.avif` | `avif_encode(bitspersample=10, primaries=9, transfer=16, matrix=9)` | coded tag 9/16 (matrix 9, full range) | 10 | 0.0121; 0.0196 | |
| `hlg-rec2020-10bit.avif` | `avif_encode(bitspersample=10, primaries=9, transfer=18, matrix=9)` | coded tag 9/18 (matrix 9, full range) | 10 | 0.0093; 0.0105 | |
| `pq-rec2020-12bit.heic` | pillow-heif `encode(bit_depth=12, color_primaries=9, transfer_characteristics=16, matrix_coefficients=9)` | coded tag 9/16 (matrix 9, full range) | 12 | 0.0094; 0.0101 | |
| `hlg-rec2020-12bit.heic` | the same, `transfer_characteristics=18` | coded tag 9/18 (matrix 9, full range) | 12 | 0.0092; 0.0123 | |

Every JPEG XL is a bare codestream with the colour as a header enumeration, relative intent. Clipped by the change of
primaries: 0.01% of samples into Rec.709, none into P3 or ProPhoto.

**PQ and HLG.** The standard-range picture is put in as display light with its white at BT.2408's reference,
203 cd/m².
- PQ: the BT.2100 (Table 4) inverse EOTF of 203 · L / 10000. White is signal 0.5807.
- HLG: BT.2100 (Table 5) with nominal peak 1000 cd/m², black 0 and system gamma 1.2. The inverse OOTF,
  E = (F_D / 1000) · (Y_D / 1000)^((1 − γ) / γ) with F_D = 203 · L and Y_D its luminance (weights from the file's
  own primaries; for BT.2020 these are BT.2100's 0.2627, 0.6780, 0.0593), then the OETF. White is signal 0.7499,
  BT.2408's 75%.
- Highlights file: before the PQ step each pixel is multiplied by 4^t, t = clip((Y − 0.25) / 0.75, 0, 1), Y its
  Rec.2020 luminance. Below Y 0.25 nothing changes and white goes to 812 cd/m². 1.1% of pixels end above 203 cd/m²,
  the brightest at 805.

**pillow-heif writes a coded tag.** `encode`/`save` take `color_primaries`, `transfer_characteristics`,
`matrix_coefficients` and `full_range_flag` and write them as the primary item's `nclx` box. The docstring's
`transfer_characteristic` (no s) is ignored without a warning, and the file is tagged 9/13 (sRGB transfer). With no
coded values it writes 1/13 (sRGB, matrix 6); with `save_nclx_profile=False`, no `colr` box. It takes 16-bit samples
and keeps the top 10 or 12 bits (`bit_depth`), and decodes them back shifted to the top of 16 bits (white 65472 or
65520). `avif_encode` takes samples at the coded depth (white 1023) and clips larger ones rather than scaling them.

**Read back by `imagecodecs.jpegxl_decode`** (the darktable exports and the files above). It returns the samples in
the file's declared encoding and converts nothing: linear, P3, PQ, HLG and the custom-primaries files come back as
coded, and agree with the sRGB export only after the declared curve and primaries are applied. The dtype follows the
header's depth: `uint8` for 8-bit; `uint16` at the file's own depth for 10, 12 and 16-bit (white 1023, 4095, 65535,
not scaled to 16 bits); `float32` for the 32-bit float file, unclipped (−0.295 to 1.074, sRGB-coded). Nothing says
which declaration the samples are in: that comes from the header alone.

**Seen in darktable's exports along the way.**
- PQ Rec2020: the sRGB export's linear light, decoded through the PQ EOTF with 10000 cd/m² as 1, within 0.2% at every
  brightness above Y 0.02. darktable writes the standard-range white at PQ's 10000 cd/m², not at 203.
- HLG Rec2020: through the inverse OETF alone, 3–7% darker than the sRGB export above Y 0.1, 33% darker near Y 0.01.
  libjxl's own conversion to linear applies the OOTF (intensity target 1000) and gives a median 0.71.
- Adobe RGB JPEG XL: the header holds Adobe RGB's chromaticities and gamma, but its samples, decoded by them (here, and
  by libjxl's own conversion, which agree), are not the other exports' photo: mean linear R, G, B 0.39, 0.17, 0.06
  against 0.32, 0.22, 0.18 from the sRGB export. Good for its declaration, not as a reference for pixels.

## P2: what the models read from each file, against the sRGB export (`exports_tolerance.py`)

    python analysis/ml29/exports_tolerance.py tmp/darktable_exported/*/* tmp/ml29_made/*

2026-11-14, P2's code, the store's settings. Each file canonicalized and read back as 8-bit sRGB; against the
darktable 8-bit sRGB JPEG XL read the same way; 32×32 block means; "mean" over every block and channel, "worst" the
largest in any block and channel, in 8-bit levels. D27's bar: mean at most 1.0, worst at most 6.

| File | Mean | Worst | D27 |
|---|---|---|---|
| JPEG XL sRGB 10, 12, 16-bit, 32-bit float | 0.06–0.07 | 0.6–0.9 | within |
| JPEG XL linear Rec2020 8, 16-bit | 0.16, 0.15 | 2.4, 2.2 | within (refused before) |
| JPEG XL Display P3 | 0.14 | 2.0 | within (refused before) |
| JPEG XL Adobe RGB | 26.4 | 70.0 | outside: its samples are another rendering (above), the declaration read right |
| AVIF sRGB 8, 10, 12-bit | 0.53, 0.22, 0.23 | 3.2, 3.0, 3.4 | within (10 and 12-bit now at their own depth) |
| AVIF Adobe RGB (ICC) | 0.43 | 4.2 | within |
| AVIF linear Rec2020 | 0.59 | 5.6 | within (64.5 before) |
| HEIC sRGB 8, 10, 12-bit | 0.38, 0.32, 0.32 | 3.6, 3.8, 3.4 | within |
| HEIC Adobe RGB (ICC) | 0.41 | 5.4 | within |
| HEIC Display P3 | 0.32 | 4.1 | within (2.2 before) |
| HEIC linear Rec2020, q50 | 0.46 | 9.6 | outside on worst only: 59 of 31,610 blocks over 6 |
| Made here: linear Rec709, Rec709, linear ProPhoto D50 | 0.94 | 5.3–5.5 | within |
| PQ, HLG: darktable's two JPEG XL, the seven made here | | | refused, naming the curve, until P5 |

The linear Rec2020 HEIC's 59 blocks over 6 are all dark (the reference's 8 to 35 levels), each shifted towards red:
the same photo as a 16-bit linear JPEG XL reads within the bar (worst 2.2), and darktable's own q50 sRGB HEIC too
(3.4). It is the file's coding, 4:2:0 chroma at q50 on linear light, which the curve to sRGB magnifies in the
shadows, not its reading. The made files are a quarter of the size; the reference is area-averaged to it in 8-bit
sRGB, while they were averaged in linear light, which accounts for most of their 0.94.

**The front's time and memory (D26)**, one process per file, `taskset` to two CPUs as the front has, the front's
settings and pixel ceiling; each 6984×4660 export area-averaged to 4741×3163 (15 MP):

| File | Colour path | Time | Peak RSS |
|---|---|---|---|
| AVIF sRGB 8-bit | integer | 4.0 s | 325 MB |
| AVIF linear Rec2020 8-bit | float (D26) | 6.5 s | 324 MB |
| HEIC sRGB 12-bit | integer | 5.7 s | 399 MB |
| HEIC linear Rec2020 12-bit | float (D26) | 7.6 s | 393 MB |
| JPEG XL sRGB 8-bit | integer | 3.8 s | 310 MB |
| JPEG XL linear Rec2020 8-bit | float (D26) | 6.4 s | 310 MB |
| JPEG XL sRGB 32-bit float | float (D26) | 6.8 s | 612 MB |

The float path costs about 2.5 s at 15 MP and no memory. A float JPEG XL is decoded whole as float32 before the cap,
12 bytes a pixel: 33 MP peaked at 612 MB, over the front's `MemoryMax` of 600M (swap allowed, ML-19 D38), and a
60 MP one would need about twice that.

## P5: HDR main images through the tone map (`hdr_pin.py`, `exports_tolerance.py`)

D33's operator wired (2026-11-15): BT.2408-8 Annex 5's EETF on max(R, G, B) in linear Rec.2020, the picture's peak
measured once (the 99.99th percentile of max(R, G, B), `tone_map.picture_peak`), BT.2408's reference white.

**Production against P4's research** (`hdr_pin.py`, store settings, 32×32 block means of what the models read): the
canonical photo against the research's own rendering of the chosen operator differs by a mean of 0.04–0.05 and at
most 0.6 levels on every made HDR file: the two code paths' rounding. Against the research's reference (the 8-bit
sRGB export at a quarter):

| File (made here, D25) | Peak | Mean; worst |
|---|---|---|
| PQ P3 16-bit JPEG XL | 1.00 | 0.11; 0.8 |
| PQ Rec2020 10-bit AVIF | 1.00 | 0.19; 1.2 |
| PQ Rec2020 12-bit HEIC | 1.00 | 0.15; 1.1 |
| HLG P3 16-bit JPEG XL | 1.00 | 0.11; 0.9 |
| HLG Rec2020 10-bit AVIF | 1.00 | 0.11; 0.8 |
| HLG Rec2020 12-bit HEIC | 1.00 | 0.10; 1.0 |
| PQ Rec2020 with highlights to 812 cd/m² | 2.17 | 6.99; 39.9 (D35's pin, P4's 6.99; 39.8: held) |

**Against the full-size sRGB export** (`exports_tolerance.py`, the reference area-averaged to each file's size), store
and front settings alike: every made PQ and HLG file within D27 (mean 0.86–0.94, worst 5.3–5.8, as the made
standard-range Rec709 file, 0.93–0.94; 5.5–5.6); the float sRGB JPEG XL 0.06–0.07; 0.9–1.0 and the linear Rec2020
JPEG XL 0.15–0.27; 1.6–2.2, both still within. The owner's darktable PQ and HLG exports are outside, as D34 records:
92.8; 172.8 and 60.7; 91.9 (store), darktable ≤ 5.6.2's own encoding.

**The front's time and memory** (D26's method: one process per file, `taskset` to two CPUs, the front's settings), on
15 MP files made from the darktable linear Rec2020 16-bit JPEG XL (4741×3163, white at 203 cd/m², a 300-row band at
four times white):

| File | Time | Peak RSS |
|---|---|---|
| PQ 16-bit JPEG XL | 13.3 s | 370 MB |
| PQ 12-bit AVIF | 13.3 s | 372 MB |
| PQ 12-bit HEIC | 13.9 s | 414 MB |
| HLG 16-bit JPEG XL | 12.7 s | 370 MB |
| HLG 12-bit AVIF | 12.7 s | 376 MB |
| HLG 12-bit HEIC | 13.6 s | 398 MB |
| linear Rec2020 16-bit JPEG XL (32 MP, capped to 15 MP), for comparison | 6.1 s | 350 MB |

An HDR photo costs about 7 s more than a standard-range one on the float path, within the front's memory and within
ML-19 D22's budget of 40 s for canonicalization on the front. Of the
PQ JPEG XL's 13.4 s (profiled): the PQ curve, run twice in float64 (the peak's pass and the conversion's), 2.8 s;
the peak pass's LittleCMS transform 1.8 s; the main pass's two transforms (to linear Rec.2020, then to the canonical
profile) and the rest 4.5 s; the JPEG XL encode 2.4 s. A lookup table for the curve on integer samples would save
most of the 2.8 s; not done.

After the review (2026-11-15): a float JPEG XL keeps the profile its declaration gives (LittleCMS's own sRGB for
sRGB), and the tone map's two transforms run only when the picture's peak is above white, so a float picture with
nothing above white converts to the bytes P2 gave it; transparent light is weighted by its alpha in the peak; +inf is
read as PQ's top. `hdr_pin.py` and the float export's tolerance unchanged after these (6.99; 39.9, held; 0.07; 0.9).

## P7: the gain-map reader, not wired in (`gain_map_check.py`)

    python analysis/ml29/gain_map_check.py [--per-brand 20] [--sections 123]

`coffeecv/gain_map.py` reads a photo's gain map in D14's order (ISO 21496-1 in a HEIF's or AVIF's `tmap` item or a
JPEG gain map's APP2, then Ultra HDR's `hdrgm` XMP, then Apple's auxiliary image) and applies it at full weight, up
from a standard-range main image or down from an HDR one, by ISO 21496-1's formula as libavif 1.4 applies it (Apple's
own formula for Apple's map). Light is relative to standard-range white, 1.0 at 203 cd/m², as the tone map reads it.
Only `canonical.decode` with `Settings.gain_maps` on calls it (off in the store's and the front's settings, `e79f5d4`). Its checks (2026-11-15/16), with the store re-run using four cores beside them:

**R5, ISO 21496-1 against the vendor's map** on the 208 dataset photos carrying both, the rendition each asks for at
the map's size from the main image shrunk to it, in stops where the main image is above 1% of white:

| Camera | Photos | Mean difference (median photo; worst) | p99 (median; worst) | Boost, ISO; vendor |
|---|---|---|---|---|
| Samsung (Ultra HDR JPEG) | 50 | 0.000; 0.000 | 0.000; 0.000 | 0.89; 0.89 |
| Apple (HEIC) | 158 | 0.017; 0.094 | 0.043; 0.102 | 0.41; 0.42 |

Samsung writes the same numbers twice. Apple's ISO item and its own formula (`HDRGainMapHeadroom`) differ by under a
tenth of a stop in the worst photo: the ISO reader is right on both containers, and D14's order changes little.

**libultrahdr against this module** (imagecodecs' `ultrahdr_decode`, linear, white 1.0), 20 Ultra HDR JPEGs per
brand at full size, where the main image is above 5% of white:

| Camera | Form read | Mean, stops (median; worst) | p99 (median; worst) | libultrahdr's brightest |
|---|---|---|---|---|
| Google | Ultra HDR | 0.0061; 0.0069 | 0.030; 0.040 | 1.85 |
| GM1911 (OnePlus 7 Pro) | Ultra HDR | 0.0075; 0.0077 | 0.042; 0.044 | 1.36 |
| Samsung | ISO 21496-1 | 0.0140; 0.0179 | 0.115; 0.150 | 4.32 |

No bias in any brightness band (Samsung's first photo: mean +0.0005 or less in each), and libultrahdr's brightest
pixel and 99.9th-percentile gain are this module's: libultrahdr gives the unclipped linear result too. The p99 is the
map's upsampling (a quarter-size map, stretched bilinearly here, sampled by libultrahdr its own way). Two Samsung
photos, 16320×12240 (Fold8 Ultra, 200 MP), are beyond this libultrahdr's 8192×8192 and were skipped.

Why this module and not libultrahdr: libultrahdr refuses Adobe's per-channel `hdrgm` form ("could not find attribute
hdrgm:GainMapMax", the public JPEG sample), does not read HEIF or AVIF, stops at 8192×8192, and decodes the main image
again; it stays the check. libavif's gain-map API is not exposed by imagecodecs. The owner accepted this as D37
(2026-11-16), asking that it be noticeable: `coffeecv/gain_map.py`'s docstring opens with where to look first when a
gain-map photo comes out wrong.

**The public samples** (libavif's test files, D29: in `tmp/`, not committed):

| File | Map | Against | Mean, stops | Mean absolute | p99 |
|---|---|---|---|---|---|
| `seine_sdr_gainmap_srgb.avif` | ISO, up, 3 channels | Photoshop's PQ rendition (`seine_hdr_srgb.avif`) | +0.004 | 0.093 | 0.64 |
| `seine_hdr_gainmap_srgb.avif` | ISO, down, 3 channels | the standard-range image | +0.000 | 0.101 | 0.72 |
| `seine_sdr_gainmap_srgb.jpg` | Ultra HDR, per-channel `rdf:Seq` | the AVIF made from it, up | | 0.016 | |

The two AVIFs carry the same map (to one level), but the PQ one is not that map applied: its light reaches 3.73, the
map's headroom is 1.3 stops (2.46). They are renditions made apart, which is what the 0.09 to 0.10 stops are, and
the same up and down; the median ratio up is 1.002. The JPEG, which libultrahdr refuses, reads as its AVIF does.

## P8: the scratch store and the pixel report (`p8_scratch.py`)

    python analysis/ml29/p8_scratch.py canonicalize [--jobs 2] [--limit N]   # resumable
    python analysis/ml29/p8_scratch.py report [--jobs 2]
    python analysis/ml29/p8_scratch.py page [--per-camera 4] [--largest 12]
    python analysis/ml29/p8_scratch.py check [--per-camera 2]

D12's step 1. Every gain-map photo the scan finds under a folder the store canonicalizes, through
`coffeecv.canonical` with the store's settings and `Settings.gain_maps` on, into `outputs/ml29_p8/canonical/`
(gitignored; outside `data/canonical` and DVC). `report` reads each against today's canonical photo as the models
read them (`load_canonical`, 8-bit sRGB) and gives the change in levels over the photo, in the bean region (the test
split's masks, `data/seg_masks_ml19/xl0_v7r_s123`) and outside it, per camera; `page` makes the owner's side-by-side
page (`outputs/ml29_p8/review/index.html`), full-resolution crops around the bean region; `check` runs the plan's two
checks. Tests: `tests/test_ml29_p8.py`.

The photos, from a fresh scan (2026-11-16, `tmp/ml29_p8/scan/files.csv`; the plan's 1,731 was 2026-11-14's): 1,741
with a gain map (ML-31 added 10), 1,687 under the store's folders, 54 elsewhere (`classes_labels_only` 40,
`2026-08-06__box_pictures` 10, `example_photos` 4), which have no canonical photo to compare with. 155 are over
20 MP, 8 over 60 MP (Fold8 Ultra, 84 and 200 MP, capped to 50 MP by the store's settings).

The toy run (2026-11-16, 7 photos, one per map form and with and without a mask, through the commands that will run):
about 22 s a 12 MP photo on one job beside the store re-run, three times today's canonicalization: the libjxl encode,
LittleCMS's float transforms, the tone map's peak and the map applied, all the store's own code, so not made faster
here. 1,687 photos: about 5 hours at 2 jobs, about 2 at 4 with the machine to itself; a 50 MP photo takes about
3 GB, so the run waits for the store re-run's memory. Three iPhone 13 HEICs carry an Apple map in its older form,
whose headroom is in Apple's maker notes: the strip removes maker notes (ML-3 D13), so no map is read and they stay
as today; `check` counts them apart. Since D38 `canonicalize` logs each applied map's channels and `check` fails on a three-channel
map: the setting applies maps in linear Rec.2020, exact for a one-channel map only (every dataset photo's, 1e-7
relative against 11% for a three-channel one). The toy photos changed by +15 to +37 levels on average (brighter), the bean
region by +16 to +25; the crops, opened at full size, show an even brightening without artefacts.

The full run (2026-11-16 to 2026-11-17, 3 jobs, then 1 for the report; the machine's memory ran out once, at 23:30 with
three jobs beside a training run, and every step resumed): all 1,687 canonicalized, none failed, and `check` PASS:
1,687 in the scratch store against the scan's 1,687; the three iPhone 13 HEICs without headroom, as expected; 0
three-channel maps (D38); 27 photos without a map (2 per camera, 15 cameras) the same bytes as today with the setting
on. Map forms: Ultra HDR 1,476 (median photo +20.8 levels), ISO 21496-1 208 (+15.8), the three Apple maps without
headroom 0.

`report` (`outputs/ml29_p8/report.csv`, `report.log`), per camera, the median photo's mean change in 8-bit levels:

| Camera | Photos | Masks | Under a level | Whole photo | Bean region | Outside it |
|---|---|---|---|---|---|---|
| Apple iPhone 13 | 3 | 1 | 3 | +0.00 | +0.00 | +0.00 |
| Apple iPhone 15 | 98 | 9 | 0 | +15.18 | +18.10 | +13.11 |
| Apple iPhone 15 Pro | 4 | 1 | 0 | +11.56 | +11.03 | +18.67 |
| Apple iPhone 15 Pro Max | 37 | 10 | 0 | +10.79 | +13.24 | +11.24 |
| Apple iPhone 16 Pro | 1 | 0 | 0 | +3.60 | - | - |
| Apple iPhone 16 Pro Max | 8 | 4 | 0 | +28.84 | +24.33 | +35.53 |
| OnePlus 7 Pro (GM1911) | 522 | 70 | 1 | +17.32 | +15.46 | +14.05 |
| Google Pixel 7 | 9 | 3 | 0 | +9.86 | +1.17 | +11.62 |
| Google Pixel 9 Pro | 945 | 97 | 6 | +22.18 | +11.71 | +13.87 |
| Samsung Galaxy Z Fold6 | 13 | 1 | 0 | +27.56 | +18.41 | +31.67 |
| Samsung Galaxy Z Fold8 Ultra | 47 | 16 | 0 | +26.95 | +24.75 | +24.91 |

Bean region and outside it are the median photo's signed mean change over the test split's masks; outside it
also counts the 40 photos whose test mask is empty (the whole photo outside), so its photos are not quite the bean
column's. Every photo with a map read is brighter, none darker; 10 change by under a level (the 3 iPhone 13s and 7 more). Over
all 1,687 the median photo moves +20.1 levels, its 99th percentile pixel 43 levels; the largest, +52, is an OOD
negative (a Pixel 9 Pro street scene with sky). On the 198 photos whose test-split mask leaves a region outside the
beans, the bean region moves +15.3 levels at the median and the rest +14.0: the maps lift the beans about as much as
the background, on 76 of the 198 more. The masks are the test split's only (252 photos), so the bean column is a
sample, the Pixel 9 Pro's 97 of 945.

The review page (`page`, `outputs/ml29_p8/review/index.html`): 12 largest changes and 40 sampled over 11 cameras,
today's canonical photo beside the scratch one, whole and a full-resolution crop around the beans. The one pair
looked at here (a Fold8 Ultra Brazil photo, `random_date_raccoon`) shows what the toy run's did: an even brightening,
no banding, halos or clipping. The page is the owner's to look through.

## P10: re-crop, retrain, compare (`p10_pool.py`, `fit_frozen_head --pool`)

    python analysis/ml29/p10_pool.py build
    python -m coffeecv.fit_frozen_head --seeds 42 123 7 --start-exp 286 --allow-dirty
    python -m coffeecv.fit_frozen_head --seeds 42 123 7 --start-exp 289 --allow-dirty \
        --pool outputs/ml29_p10/segcropped/pool --label ml29gm
    python -m coffeecv.compare_experiments 289 290 291 --vs 286 287 288     # -> compare_289_291.txt
    python analysis/ml29/p10_pool.py compare --gain 289 290 291 --base 286 287 288   # -> p10_compare.txt

D12's step 3. The scratch pool (`outputs/ml29_p10/segcropped/pool`, gitignored) is `data/segcropped/pool` with the
1,255 gain-map photos' crops and masks replaced by P9's (the shipped segmenter, `xl0_v7r_s123`, on P8's scratch
photos, as the segcrop stage crops), hard links; every other file a symbolic link to the tracked pool's. Same
names, so the same split (it keys on the folder's name, `pool`); 27 scratch crops are of photos the pool excludes;
no photo becomes or stops being a D18 fallback. `p10_pool_manifest.json` lists every swapped crop by hash, the tree
`dff5c24c`, which the gain-map runs' resume fingerprint holds. Matched by class id and file name, as `merge_rig`
merges: the first build matched folder names and left 141 crops of sessions that name a class otherwise at today's
(fixed, `382e138`, before any run). Tests: `tests/test_ml29_p10.py`, `tests/test_fit_frozen_head_pool.py`.

Two departures from the plan. The baseline is refitted, not the shipped model's runs: exp283-285 trained on the
pool of 80c44ac (3,363 files), the pool now has 3,527 (ML-31), so exp286-288 are the same recipe on today's pool,
and the two arms' configs differ in `train_capture_dirs` alone (checked key by key before the gain-map arm). Both
arms ran with `--allow-dirty`: the uncommitted files were another ticket's docs (ML-32), the board, and from
2026-11-17 the owner's pass17 labels (ML-33), none read by a fit; `dvc status` clean. A run records the commit
checked out when it is archived, not when it started: the baseline launched at `8018b6d`, exp286 and exp287 record
`fb11906`, exp288 `9349517`; the gain-map arm launched at `9349517` and exp289-291 record it. Between `8018b6d` and
`9349517` the code a fit runs changed only by `NAMED_SEGMENTERS` gaining a name (`42bdf63`, which no fit reads);
the rest is `p10_pool.py`, docs and the pass17 labels. One machine, 6 torch threads; 3.3 to 4.8 h a seed (4.8 with
P9's answers beside it).

Before the runs, the changed path ran at toy scale through the real command (2026-11-16, `fit_frozen_head --smoke
--allow-dirty --seeds 42 --pool <a partial scratch pool of links> --label smoke`): 23 crops swapped, 3,481 files
linked, the fit to its end in 100 minutes, peak memory 1.8 GB; not archived (smoke).

The scratch pool is outside git and DVC, as the plan has P8 to P10 beside the store until P11, so a checkout
cannot restore exp289-291's input. It is rebuilt from committed code on this machine (libjxl's output depends on
the CPU, `dvc.yaml`): `p8_scratch.py canonicalize`, `p9_models.py crop`, `p10_pool.py build`; the manifest's
hashes (each crop and mask, and the tree's, `dff5c24c`, which the runs' fingerprint holds) check the result.

Patch macro-F1, ML-27's recipe (600/160/160 patches a class, frozen DINOv3 ViT-B/16 `cls_mean`, C = 0.1, 8
augmented views a training patch):

| seed | baseline | gain maps | val, unaug | val, aug | test, unaug | test, aug |
|---|---|---|---|---|---|---|
| 42 | 286 | 289 | 0.9518 -> 0.9481 | 0.9571 -> 0.9477 | 0.9537 -> 0.9500 | 0.9592 -> 0.9560 |
| 123 | 287 | 290 | 0.9456 -> 0.9468 | 0.9568 -> 0.9574 | 0.9543 -> 0.9473 | 0.9586 -> 0.9559 |
| 7 | 288 | 291 | 0.9471 -> 0.9375 | 0.9514 -> 0.9450 | 0.9298 -> 0.9297 | 0.9401 -> 0.9433 |

- Paired by seed (augmented heads, the shipped form): test -0.0009 on average (-0.0032 to +0.0031), val -0.0051
  (-0.0094 to +0.0006). The baseline's own seed range is 0.9401-0.9592 on test, 0.9514-0.9571 on val. Per class
  on test no mean moves more than 0.006 (Kenya -0.006, Ethiopia +0.005); on val Peru -0.016 and Kenya -0.011 move
  most, Peru lower at every seed (`compare_289_291.txt`).
- D16's bar is not met: the overall test metric does not improve, let alone beyond the seed range. No class drops
  beyond it either.
- The control (`p10_compare.txt`): the photos whose crops the arms share move as much as the swapped ones. Test:
  swapped -0.0014 (-0.0037 to +0.0029), shared -0.0009 (-0.0050 to +0.0024); val: swapped -0.0035, shared -0.0075
  (-0.0091 to -0.0063, lower at every seed). By camera, the cameras without gain maps (Sony G8441, 351 to 430
  patches) val -0.0074 at every seed; the Pixel 9 Pro (604 to 713 patches) test -0.0013, val +0.0003; the OnePlus 7
  Pro (309 to 417) test -0.0032, val -0.0128. The deltas are the head's refit on other features, spread over every
  photo, not an effect on the photos the maps brighten. The small cameras (3 to 30 patches) move by single photos.
  Accuracy moves with macro-F1 (test: swapped -0.0012, shared -0.0000).
- Per class per camera (D13; the groups, and the cameras with at least 100 patches in every seed): on the swapped
  photos no class's F1 moves more than 0.007 on test or 0.012 on val on average; on the shared ones Colombia -0.010
  on test, Guatemala -0.030 on val, Sony G8441 most. The largest single cell, the iPhone 15 on val (115 to 142
  patches), Brazil -0.049 and Ethiopia +0.057, is a few photos either way.
