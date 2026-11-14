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
