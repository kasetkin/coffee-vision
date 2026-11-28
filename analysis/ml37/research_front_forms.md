# ML-37: what the front should send to `yappy`: linear uint16 (A), lossy 8-bit sRGB (B) or lossless 8-bit sRGB (C)

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) (D1–D3, D9, D12), 2026-11-25, on the
local checkout at `e1b03c8` (branch `ml37-linear-canonical`). A recommendation for the owner, not a decision. The
owner's question: the front's check service canonicalizes an upload and sends it over the 2.1 Mbit/s link to the
inference host (ADR 0031, 0033, 0036). Which of three forms should it send?

- **A.** Lossy JPEG XL, Rec.2020 (Rec.2100) primaries, D65, **linear** transfer, uint16, intensity target 203:
  ML-37 D1's canonical photo, light above white tone-mapped at canonicalization (ML-29 D33's EETF), clipped to [0, 1].
- **B.** Lossy JPEG XL of the **8-bit sRGB picture** the models take (A's read-back, D2/D3).
- **C.** Lossless JPEG XL of that 8-bit picture, at effort 1 (**C1**) and effort 3 (**C3**).

Every number comes from the scripts beside this note and their printed outputs (`front_forms_*.txt`). Nothing under
`coffeecv/`, `docs/`, `webapp/`, `data/` or `models/` was changed.

## Summary

40 dataset photos (seeded, stratified by format, camera, gain map and size; §3), the front's settings, 2 threads,
pinned to 2 CPUs, median of 3 runs per photo. Mean ± std over photos; "total" is generate + transfer.

| form | generate s | size MB | bytes/px | transfer s at 2.1 Mbit/s | **total s** | pixel RMS (8-bit levels) | D27 block mean | largest block | host decode s |
|---|---|---|---|---|---|---|---|---|---|
| **A** linear u16, q90 e3, IT 203 | 6.56 ± 3.62 | 1.33 ± 0.81 | 0.114 ± 0.059 | 5.06 ± 3.08 | **11.62 ± 5.62** | 2.31 ± 1.35 | 0.148 ± 0.148 | 8.42 | 0.16 + 0.39 read-back |
| **B** 8-bit sRGB, q90 e3 | 5.29 ± 3.29 | 1.42 ± 0.87 | 0.122 ± 0.063 | 5.43 ± 3.30 | **10.71 ± 5.45** | 2.28 ± 1.31 | 0.145 ± 0.152 | 13.38 | 0.17 |
| **C1** 8-bit sRGB, lossless e1 | 5.02 ± 3.24 | 11.75 ± 6.12 | 0.956 ± 0.365 | 44.75 ± 23.30 | **49.77 ± 24.88** | 0 | 0 | 0 | 0.21 |
| **C3** 8-bit sRGB, lossless e3 | 6.27 ± 3.54 | 10.12 ± 5.71 | 0.822 ± 0.349 | 38.54 ± 21.74 | **44.81 ± 23.44** | 0 | 0 | 0 | 1.00 |

The errors are each decoded picture (A read back and rounded to uint8 once) against the uncompressed 8-bit picture.
The 37 photos the live front accepts (three RAW photos are over its 16 MP RAW ceiling, §3) give the same picture:
totals 11.46 ± 5.77 (A), 10.55 ± 5.57 (B), 46.60 ± 22.67 (C1), 41.63 ± 20.80 s (C3) (`front_forms_measure.txt`).

- **Lossless (C) is out.** It is 9.1× B's bytes per photo (4.7–23.9×; 0.82–0.96 bytes per pixel), and the link
  turns that into 39–45 s per photo on average and up to 113 s, against 5–5.4 s for A or B. Encoding is not the
  problem (C1's encode takes 0.04 s), the link is. If C were wanted anyway, **effort 3** is the reasonable one on
  a 2-vCPU front: it costs 1.3 s more to encode and saves 15% of the bytes (C3/C1 0.85 ± 0.06), about 6 s on the
  link, so its total is 5 s lower. It also takes the host 1.0 s to decode against C1's 0.2 s.
- **B is 0.9 s faster end to end than A** (per photo 0.91 ± 0.38 s, 0.05–1.48 s): 1.28 s faster to make, 7% more bytes
  (B/A 1.071 ± 0.024), which costs 0.37 s on the link. Both are within 10% of each other on every photo.
- **Neither gain comes from the form itself:**
  - **Bytes.** A's 7% fewer bytes are D8's intensity target, not the linear uint16 form. At intensity target 255,
    A's file is the same size as B's to within 0.4% on all seven photos tried (`front_forms_encoders.txt`, `ctA255` against
    `icB`), and its error is lower than B's at 203 (pixel RMS 1.825 vs 1.847 on the Sony, 1.538 vs 1.608 on an iPhone).
  - **Time.** B's 1.3 s comes from libjxl, not from the pixels: A's encode takes 1.9–2.4 s and B's 0.3–0.4 s for the
    same 12–15 MP picture. A uint16 picture in linear sRGB encodes as fast as B (`icLin8`, 0.26–0.43 s) and B's
    picture as uint16 nearly as fast (`icB16`), so the cost is libjxl's colour conversion from the Rec.2100
    primaries before XYB (`ToXYB`'s `ApplyColorTransform`, `lib/jxl/enc_xyb.cc` L233–288;
    `research_canonical_int_vs_half.md` §1). Today's canonical photo has the same primaries and pays the same cost.
- **B is worse at the sRGB gamut's edge.** Overall the errors are equal: pixel RMS B/A 0.99 ± 0.10, block mean
  0.99 ± 0.41, both far inside D27's mean ≤ 1.0. But B codes a picture already clipped to sRGB, and where a channel
  sits at 0 or 255 the coder's error can only push it inwards, which the decoder keeps. A codes the unclipped
  Rec.2020 colour, so its error there falls outside sRGB and the read-back's clip removes it. Over all 471,489
  blocks, **1.27% of B's blocks are off by more than 2 levels against A's 0.18%**. 95% of B's are at the gamut edge
  (a channel at 0 or 255 in ≥ 10% of the block). At the edge the worst channel's block mean is 0.72 for B and 0.32
  for A. Elsewhere B is slightly better (0.194 vs 0.228), from its higher intensity target (`front_forms_detail.txt`).
  It is concentrated in saturated colour. On the two plain Pixel 9 Pro JPEGs (Display P3), 30% and 10% of B's blocks
  are off by more than 2 levels (A: 0% and 0.06%), with block means 0.49 and 0.26 against A's 0.15 and 0.14. B's
  blocks over 7 levels (7.8–13.4, on the two plain Pixel photos and five gain-map photos) are yellows and cyans with
  the blue or red channel clipped at 0 (B's worst, on a Pixel photo: blue 11.2 → 24.6); A's largest on the same
  photos are 1.7–4.9.
- **What B gives up beyond error.** B sends the models' input, not the canonical photo. ADR 0033 keeps one lossy
  generation at the front and stores Rec.2020, and ML-37 D1 makes that file linear Rec.2020. Under B, the host
  receives no canonical photo: no wide-gamut colour, nothing above sRGB to keep or re-read. A proposal for B
  contradicts ADR 0033's choice of Rec.2020 storage made once at the front, and would need that ADR amended (and
  ADR 0036, under which the front writes a canonical photo with its own settings).
- **Recommended: keep A.** B's 0.9 s per photo is 8% of the total, its file is 7% larger at the same quality setting,
  and its error is up to 13 levels in saturated colour where A's is under 5. The time A loses is in libjxl's
  primaries conversion and in the front's colour path (§2), which can be cut without changing the form: D12's LUT
  path would save about as much as B does (1.6 s of float colour, 7–10× faster). If the owner wants the intensity target's bytes back, A at 255 costs +7% bytes
  (+0.37 s) and has less error than B.

## 1. Results by format (`front_forms_measure.txt`)

Means per format. Each cell: generate s / kB / transfer s / pixel RMS / D27 block mean.

| format | n | MP sent | A | B | C1 | C3 |
|---|---|---|---|---|---|---|
| JPEG | 11 | 11.1 | 3.43 / 1210 / 4.61 / 1.91 / 0.138 | 2.32 / 1288 / 4.91 / 1.96 / 0.169 | 2.08 / 8536 / 32.5 | 3.20 / 7323 / 27.9 |
| gain-map JPEG | 12 | 13.6 | 9.78 / 1435 / 5.47 / 1.83 / 0.135 | 8.33 / 1525 / 5.81 / 1.78 / 0.116 | 8.04 / 12415 / 47.3 | 9.46 / 10268 / 39.1 |
| gain-map HEIC | 6 | 14.1 | 10.46 / 1493 / 5.69 / 2.02 / 0.123 | 8.96 / 1602 / 6.10 / 1.97 / 0.113 | 8.66 / 13137 / 50.1 | 10.08 / 11055 / 42.1 |
| HEIC, no map | 1 | 12.2 | 3.90 / 1405 / 5.35 / 2.01 / 0.202 | 2.66 / 1512 / 5.76 / 1.91 / 0.122 | 2.37 / 8096 / 30.8 | 3.58 / 7617 / 29.0 |
| DNG | 4 | 10.6 | 4.06 / 737 / 2.81 / 2.55 / 0.088 | 2.91 / 799 / 3.04 / 2.50 / 0.085 | 2.69 / 11273 / 43.0 | 3.80 / 9666 / 36.8 |
| CR3 | 2 | 15.0 | 6.46 / 2147 / 8.18 / 4.65 / 0.103 | 4.85 / 2361 / 9.00 / 4.55 / 0.099 | 4.47 / 24547 / 93.5 | 6.05 / 23193 / 88.4 |
| PNG | 2 | 1.1 | 0.32 / 119 / 0.45 / 3.50 / 0.517 | 0.21 / 125 / 0.48 / 3.37 / 0.491 | 0.19 / 736 / 2.8 | 0.29 / 576 / 2.2 |
| JPEG XL (12-bit) | 2 | 15.0 | 5.48 / 2378 / 9.06 / 4.43 / 0.116 | 3.90 / 2610 / 9.94 / 4.31 / 0.105 | 3.53 / 22195 / 84.6 | 5.09 / 20359 / 77.6 |

- **Gain-map photos take 8–10 s to make in every form**, against 2–4 s for plain photos of the same size. Decoding
  and applying the map takes 2.8 s, the peak pass 2.2 s and the EETF 1.1 s on average, against 0.5 / 0 / 0.01 s for
  photos without a map.
- **Noisy sources compress worst.** The Canon 90D's CR3s and the JPEG XLs developed from them come to 2.1–2.6 MB
  lossy and 20–25 MB lossless. A plain 12 MP phone JPEG comes to 1.2–1.5 MB.
- **The PNG row is two small, odd files.** The RGB PNG is a 0.9 MP web picture with large flat saturated areas, and
  its largest block (8.4 A, 7.8 B) is the largest A block in the sample. The grey PNG is a segmentation mask, where
  lossless (6 kB) beats both lossy forms (21 kB).

## 2. Where the time goes (`front_forms_measure.txt`, `front_forms_encoders.txt`)

Per step, the mean ± std over photos of each photo's median of 3:

| decode (incl. gain map, cap) | peak pass | colour (float) | tone map + clip | A: rint to uint16 | B/C: read-back to uint8 | encode A | encode B | encode C1 | encode C3 |
|---|---|---|---|---|---|---|---|---|---|
| 1.53 ± 1.35 | 0.98 ± 1.11 | 1.60 ± 0.58 | 0.48 ± 0.58 | 0.08 ± 0.04 | 0.39 ± 0.15 | 1.89 ± 0.68 | 0.30 ± 0.12 | 0.04 ± 0.01 | 1.29 ± 0.47 |

- **The part every form shares, from decode to tone map, is 4.6 ± 3.2 s.** It is most of the generate time for A
  and B. The float colour path (1.6 s) is what D12 measured, and its LUT path does the same job 7–10× faster
  (`research_d12_front_colour_path.md` §3). That speeds up all four forms alike.
- **B's read-back (0.39 s) costs 0.31 s more than A's rounding**, but B's encode saves 1.6 s.
- **Encoder diagnosis** (7 photos, q90 e3, 2 threads, median of 3):

  | input | Sony 15 MP | Pixel 12.5 MP | iPhone gain map 15 MP |
  |---|---|---|---|
  | A: ctypes, uint16 linear Rec.2100, IT 203 | 2.38 s, 1,917,007 B | 1.93 s, 1,374,850 B | 2.43 s, 2,406,782 B |
  | the same at IT 255 | 2.38 s, 2,028,543 B | 1.92 s, 1,448,592 B | 2.44 s, 2,575,351 B |
  | imagecodecs, the same picture and header (IT 255) | 2.41 s, identical bytes | 1.93 s, identical | 2.43 s, identical |
  | B: imagecodecs, uint8 sRGB | 0.38 s, 2,028,619 B | 0.31 s, 1,446,776 B | 0.43 s, 2,578,029 B |
  | B's picture as uint16 sRGB | 0.45 s | 0.34 s | 0.49 s |
  | A's picture as uint16 **linear sRGB** (matrix, clip) | 0.37 s | 0.31 s | 0.43 s |

  Neither the bit depth nor the transfer function makes the encode slow. The Rec.2100 primaries do. The repo's
  ctypes encoder and imagecodecs take the same time and write the same bytes, so the chunked encoder is not the
  cause either.

## 3. Method

- **Sample** (`front_forms_measure.py`, `sample()`; list in `front_forms_sample.txt`). ML-29's scan of the dataset
  (`tmp/ml29_scan/files.csv`), seed 0, a fixed count per stratum:
  - plain JPEG: Sony 17 MP ×3, OnePlus ×2, Pixel 9 Pro ×2, no camera tag ×2, Xiaomi ×1, Samsung ×1;
  - gain-map JPEG: Pixel ≥ 48 MP ×3, Samsung ≥ 48 MP ×2, Samsung 24 MP ×1, Pixel 12 MP ×3, OnePlus-tagged ×2,
    Samsung 12 MP ×1;
  - gain-map HEIC: iPhone ≥ 48 MP ×2, 24 MP ×2, 12 MP ×2; plain HEIC ×1;
  - DNG: Pixel ×2, iPhone ×1, OnePlus ×1; CR3 ×2;
  - PNG: RGB ×1, grey ×1 (alpha excluded: the read-back here has no fill step);
  - JPEG XL ×2 (12-bit).

  Photos over the front's 60 MP ceiling are excluded, because the front refuses them. The first draw had a
  200 MP Samsung, which Pillow refuses as a decompression bomb. RAW photos over the 16 MP RAW ceiling (one Pixel
  DNG, both 32.5 MP CR3s) are kept and measured with no ceiling, flagged `front_accepts = False`. Everything over
  18 MP is area-averaged to 15 MP by `FRONT_SETTINGS`' cap, so the photos as sent are 0.6–17 MP (mean 12.2).
- **Making the pictures.** Each run:
  - Decode: `canonical.decode(..., settings=FRONT_SETTINGS)`. This applies the gain map and the cap.
  - Peak pass: `Source.peak()`.
  - Colour, per 256-row strip: the store's float colour path, `canonical._light`, then LittleCMS on float32 to linear
    Rec.2020. This is not the front's integer path, which D12 §3 shows quantizes linear uint16 output to 1/16384.
  - Tone map: the EETF at the peak (tone-mapped sources only), then clip to [0, 1].
  - **A**: rint(× 65535), uint16, written by the repo's ctypes strip encoder (`canon_int_measure.encode_linear_u16`,
    which is `d8_intensity_target.encode` on `coffeecv/jxl_encode.py`'s libjxl bindings) with
    `JXL_TRANSFER_FUNCTION_LINEAR`, Rec.2100 primaries, intensity target 203, quality 90, effort 3. The ctypes
    encoder is the one that can set the target, `research_canonical_int_vs_half.md` §2.
  - **B/C**: the read-back from the same clipped float light, ticket D9 (a) in numpy float32: the matrix from linear
    Rec.2020 to linear sRGB, clip, the sRGB curve, × 255, rint to uint8. B is written by
    `imagecodecs.jpegxl_encode(uint8, level=90, effort=3)`, which declares sRGB and intensity target 255. C by
    `lossless=True` at effort 1 and 3.
- **B's picture is made directly, not by the front canonicalizing and then reading back.** Going through A would
  add A's encode (1.9 s), a decode (0.2 s) and a lossy generation before B's own. So it would not be a form the
  front would choose. The check that the direct picture is A's read-back: the read-back of A's uncompressed uint16
  picture equals it to within 1 level, with 0.14% of values differing on average (at most 0.69%, from the uint16
  rounding).
- **Timing.**
  - Setup: one process under `taskset -c 0,1` (the front's 2 vCPU, as `d12_front_cost.py`), libjxl with 2 threads,
    `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=2`. Wall clock per step, one untimed warm-up on the first photo, 3 runs
    per photo, the median per step.
  - Generate time = decode + peak + colour + tone map + (A: rounding + encode A; B/C: read-back + their encode), as
    if the front made only that form. Transfer = bytes × 8 / 2.1e6.
  - Run: 15:45–15:51 and 15:51–16:06 UTC, resumed after the 200 MP exclusion. The 14 photos already measured were
    kept, not re-run.
  - Load average: 0.5–0.8 before the run and 1.4–3.0 during it (the run itself keeps 2 CPUs busy), on a 12-thread
    AMD Ryzen 5 5600U. Peak RSS 1.16 GB.
- **Error.** Measured on the first run's files, each decoded by `imagecodecs.jpegxl_decode` (2 threads), A read back
  as above and rounded to uint8 once. The reference is the uncompressed 8-bit picture (B's and C's input).
  - Pixel RMS in 8-bit levels.
  - ML-29 D27's measure: 32 × 32 block means, the mean and the largest absolute difference over blocks and channels.
  - The share of values that differ.
  - `front_forms_detail.py` adds the worst channel per block, the share of blocks over 1 and 2 levels, and a split
    at the gamut edge (a block where some channel of the reference is 0 or 255 in at least 10% of its pixels).
  - C's decode equals the reference exactly on all 40 photos.

## 4. What this does not measure

- **The real front.** The timings come from this machine's two cores (Ryzen 5 5600U) pinned with taskset, not the
  front's 2 vCPU, which are likely slower. The ratios between forms should carry over better than the seconds.
  The front's ordinary CPU contention (the page, other requests) is not here.
- **The front's actual colour path for A.** A uses the store's float path. D12's LUT path, or a faster libjxl
  conversion, would shorten A's (and B's) generate time. Today's front integer path cannot make A without the
  1/16384 quantization.
- **The link itself.** Transfer is size × 8 / 2.1 Mbit/s: no TCP slow start, no multipart overhead, no variation in
  the link. The photo is assumed sent after it is made, not streamed while libjxl writes it.
- **Memory.** Whole-picture arrays here (uint16 + uint8 + float strips). The front's strip-by-strip budget (600M,
  ML-19 D38) is D12's and R3's question.
- **What the models answer.** Only pixel and block errors. No classifier feature or segmenter output was compared
  between A and B, and the gamut-edge blocks were not related to bean pixels.
- **B at intensity target 203, or A at 255 on every photo.** Only the seven photos of `front_forms_encoders.txt`
  were tried. Other qualities and efforts for the lossy forms were not tried.
- **Statistics.** 40 photos, one stratified draw, no confidence intervals. The ± is the spread over photos, not
  an uncertainty of the mean.

## Files

Scripts and their printed outputs in `analysis/ml37/` (run with `cd /workspace && PYTHONPATH=/workspace uv run python
analysis/ml37/<script>`; the timed ones under `taskset -c 0,1` as their docstrings say):

- `front_forms_measure.py` → `front_forms_measure.txt`: the 40-photo run, summary tables, per-format and per-photo rows.
  The per-photo rows are in `tmp/ml37/front_forms/rows.csv`.
- `front_forms_detail.py` → `front_forms_detail.txt`: where A's and B's errors sit (gamut edge).
- `front_forms_encoders.py` → `front_forms_encoders.txt`: encode time and bytes by writer, primaries, transfer and
  bit depth; A at intensity target 255.
- `front_forms_sample.txt`: the sample, by dataset path.
- The generated files are in `tmp/ml37/front_forms/{A,B,C1,C3}/` (gitignored, made from the owner's photos), one
  per photo, named `<index>_<original stem>.jxl`.
- Reused, unchanged: `canon_int_measure.py` (`encode_linear_u16`), `d8_intensity_target.py` (the encoder, matrix,
  sRGB curve, blocks).

## Sources

- **The repo at `e1b03c8`:**
  - `coffeecv/canonical.py`: `FRONT_SETTINGS` L119, `decode`, `_light`, `to_rec2020`, `measure_peak`, `_cap`.
  - `coffeecv/jxl_encode.py`, `coffeecv/tone_map.py`.
  - `webapp/front/check_service.py`: L78–79, the 60 MP and 16 MP RAW ceilings; L380–392, `canonicalize`.
  - ADR 0031 (the 2.1 Mbit/s link), 0033, 0036, 0041.
  - The ML-37 ticket (D1–D3, D8, D9, D12, D27's measure).
  - `research_canonical_int_vs_half.md` (§1, §2), `research_d12_front_colour_path.md` (§1–§3),
    `research_d8_intensity_target.md`.
- **libjxl 0.12.0** (in imagecodecs 2026.10.10's wheel): `lib/jxl/enc_xyb.cc` L233–288 (`ToXYB`, the colour
  conversion before XYB), as read for `research_canonical_int_vs_half.md` §1.
