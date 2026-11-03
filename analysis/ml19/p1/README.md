# ML-19 P1: the encoder and the colour on three machines

Measured 2026-11-03 for [ML-19](../../docs/ticket_canonicalization.html), plan phase P1
([plan](../../docs/plan_canonicalization.html)). The prototype is not P3's module and has no tests. Every remote run
used a throwaway uv environment under `/tmp/ml19p1` (uv's cache and Python included), never a release's `.venv`. Both
remote directories were deleted afterwards. The raw results are in `results/`.

## What was run

- `pipeline.py`: D44's recipe in one script, with its dependencies pinned inline (PEP 723): imagecodecs 2026.10.10
  (LittleCMS 2.19.0, libjxl 0.12.0), numpy 2.5.3, opencv-python-headless 5.0.0.93, pillow 12.3.0, pillow-heif 1.8.0,
  rawpy 0.27.1. The steps:
  1. Decode at the decoder's depth: JPEG through imagecodecs (the same pixels as Pillow's on all five test JPEGs), PNG
     through imagecodecs (16 bits kept), HEIC through pillow-heif (10 bits kept, scaled by its own full value, 65472), RAW through rawpy at 16 bits into
     linear Rec.2020 with `use_camera_wb`.
  2. Turn upright as a view.
  3. Cap: over 60 MP down to 50 MP by `INTER_AREA`.
  4. Convert with LittleCMS on float32, relative colorimetric, into libjxl's own Rec.2020 profile
     (`rec2020_bt709.icc`, 536 bytes, sha256 `877c5b36…`, D46), in strips. Clip to [0, 1] and round to uint16.
     Transparent pixels go over the segmenter's fill.
  5. Encode with `jpegxl_encode(level=95, primaries='BT2100', transfer='BT709', effort=E)`.
  6. Read back: `jpegxl_decode` to uint16, then LittleCMS (NOOPTIMIZE) to 8-bit sRGB.
- `jxl_header.py`: a reader of the codestream's image header (ISO/IEC 18181-1), standard library only, for finding 5.
  `check_header.py` checks it against libjxl's own reading.
- `efforts.sh`: the effort sweep, one process per run.
- **The test set.** All the photos are copied from `dataset/` except the three synthetic ones:
  - `p3rot6.jpg`: DCI-P3 D65 with the sRGB curve, EXIF rotate 90 CW, 12 MP. Galaxy Z Fold8,
    `random_date_raccoon/class_014__Colombia_Excelso/20261103_150637_…`.
  - `srgb.jpg`: sRGB, 12 MP, MPO. `2026-09-24__oneplus/…/PXL_20260924_104154289.jpg`.
  - `heic8.heic`: iPhone, Display P3. `2026-08-25__iphone/…/IMG_6248.HEIC`.
  - `heic10.heic`: **synthetic**, 10-bit with the iPhone's Display P3 profile. Every HEIC in the dataset is 8-bit.
  - `raw.dng`: 12 MP. `ood_negatives/user_realworld/20240328_162102.dng`.
  - `untagged0.jpg`: no profile, EXIF Orientation 0. `random_date_raccoon/…/20261008_092817_KB2005_Guatemala.jpg`.
  - `adobergb.jpg`: Adobe RGB. `confusable_grain_031.jpg`.
  - `rgba.png`: transparent. `confusable_grain_001.png`.
  - `png16.png`: **synthetic**, 16-bit, untagged. The dataset has no 16-bit PNG.
  - `p3pixel50mp.jpg`: Display P3, 50.1 MP. `segmenter_positives/PXL_20261014_104437677.jpg`.
  - `dcip3_60mp.jpg`: **synthetic**, 59,996,352 px, a 200 MP Fold8 photo cut to 60 MP, with its profile kept.
  - `big84mp.jpg`: DCI-P3 D65, 84 MP, rotated, over the cap. `segmenter_positives/20261103_150531_…`. Run here only.
  - `greyalpha.png`: **synthetic**, 8-bit grey with alpha, added after the review. Run here only.

| Machine | CPU | glibc | Python | OpenCV dispatch |
|---|---|---|---|---|
| this one | AMD Ryzen 5 5600U | 2.41 | 3.12.16 | AVX2 |
| the front (`webproxy-deploy`) | AMD EPYC 9374F, 2 vCPU | 2.36 | 3.12.14 | AVX-512 (SKX) |
| the host (`webappremote`, yappy) | Intel N100 | 2.36 | 3.12.14 | AVX2 |

## D30: the same code on three machines

The first nine photos were encoded at effort 7 on every machine (`results/canon_*.jsonl`). Each machine then decoded
this machine's nine files and yappy's `p3rot6` to sRGB (`results/srgb_*.jsonl`, `compare_*.jsonl`).

- **The encode.**
  - The front's nine files are byte-identical to this machine's at effort 7 (`canon_*.jsonl`), and so are the two
    photos swept at efforts 3 to 6 (`srgb`, `p3rot6`; `eff_*.jsonl`).
  - yappy's files are identical for 8 of the 9 at effort 7. `p3rot6` differs, at the same size.
  - Across efforts 3–7, yappy differs for `p3rot6` at efforts 3, 4 and 7, and never for `srgb`.
  - For `p3rot6`, the 16-bit Rec.2020 array fed to the encoder is identical on all three
    (`rec2020_hash_three_machines.txt`). So the decoder, the turn and LittleCMS agree exactly on that photo, and the
    difference is libjxl's on the Intel CPU. The other eight files are identical on yappy, so their arrays must be too. That matches the research
    note's approximate-reciprocal instruction.
- **The decode.**
  - The front decodes every file to the same sRGB bytes as this machine.
  - yappy is within 1 level on every file, with 23 to 526 pixels per photo at 1 level, the rest equal.
- **Encode and decode both on yappy, against both here: up to 2 levels**, in 563 of 12,000,000 pixels of `p3rot6`
  (`compare_p3rot6_yappy_encoded_and_decoded_vs_local.jsonl`). Encoded on yappy and decoded here, it's 2 levels in 66
  pixels (`compare_p3rot6_yappy_encoded_decoded_here_vs_local.jsonl`).
- **The downscale.** `INTER_AREA` on fixed-seed 8-bit and 16-bit images (`pipeline.py resize`) gives the same bytes on
  all three, the front's AVX-512 path included (`resize_local.jsonl`, `resize_front_and_yappy.txt`). The images are
  under 60 MP so the front can hold them, so `cap()`'s own branch was not taken there: it computes the size and makes
  the same call. That branch ran here only, on the 84 MP Fold8 photo (`big84mp.jpg`,
  `segmenter_positives/20261103_150531_…`): turned upright to 7936×10584, it came out at 6122×8165 = 49,986,130 px
  (`canon_local_final.jsonl`).
  - 7000×6000 uint8: `e9c4db71…`
  - 4000×5000 uint16: `30ea0723…`
  - 6145×8165 uint8: `c9d92d0e…`

**What this means for D30.** No path in the design encodes on the Intel machine:

- The store is built here, and visitor photos are encoded on the front, which gives the same bytes.
- The host only decodes, within 1 level.

So every real path meets "within 1 level". The 2-level case would only arise if the front, or the store's machine,
moved to an Intel CPU.

## D25: the effort

A 12 MP photo, 2 threads. Peak memory is the whole process's. The front ran under
`systemd-run --user --scope -p MemoryMax=500M -p MemorySwapMax=0`, and its MemAvailable never went below 568 MB
(`eff_front_memavailable.txt`). The scope also counts `uv run`'s own process and the page cache of the files read, so
the front's peaks here are the Python process's, read from `ru_maxrss`.

| Effort | Bytes (`srgb`) | Front: encode / total s | Front peak MB | yappy encode s | Here: peak MB |
|---|---|---|---|---|---|
| 3 | 2,605,567 | 1.16 / 3.11 | 390 | 2.15 | 381 |
| 4 | 2,676,279 | 1.14 / 3.10 | 396 | 2.22 | 384 |
| 5 | 2,565,055 | 1.86 / 3.82 | 423 | 4.04 | 446 |
| 6 | 2,563,257 | 2.41 / 4.38 | 431 | 5.42 | 446 |
| 7 | 2,565,269 | 2.51 / 4.45 | 432 | 6.98 | 448 |
| 8 | 2,424,409 | killed at 500M: exit 137, `oom-kill` (`front_effort8_oom.txt`) | — | 42.6 | 3,298 |
| 9 | 2,419,988 | failed at 500M; the exit code was lost to a bug in `efforts.sh`, since fixed | — | 59.7 | 3,297 |

At 50 and 60 MP the photos measure about 860 and 990 MB, which the front can't run under 500M, so these were run here.
The front ran the 12 MP photos in 0.77× this machine's time. Efforts 8 and 9 were not run at 50 or 60 MP anywhere: they
take 3.3 GB at 12 MP, and this machine had 6 GB free.

| Effort | 50 MP: bytes / total s / peak MB | 60 MP: bytes / total s / peak MB |
|---|---|---|
| 3 | 4,085,229 / 20.9 / 820 | 7,277,535 / 26.0 / 945 |
| 5 | 4,425,019 / 22.7 / 858 | 7,774,386 / 29.2 / 1,010 |
| 7 | 4,456,517 / 25.9 / 859 | 7,814,449 / 32.8 / 994 |

**Fidelity against the photo taken straight to sRGB, without JPEG XL** (`pipeline.py fidelity`,
`results/fidelity_local.jsonl`):

- Efforts 5 to 7: mean 0.77 levels, 99th percentile 3–4, for both 12 MP photos.
- Effort 3: mean 0.78–0.81, with a slightly higher maximum.
- Efforts 8 and 9: mean 0.82, no better. They spend their time on a different metric, butteraugli.
- The 60 MP photo: mean 0.58–0.61 at every effort.

Higher effort makes no smaller files at q95. Effort 7 runs from 2% smaller to 9% larger than effort 3, and gives the same fidelity as
5 and 6.

**Reading:** effort 7, libjxl's own default, is the highest that runs on the front. Effort 8 needs about 3.3 GB at
12 MP. Effort 7 is also inside D22's 40 s: 4.5 s for 12 MP on the front, and about 25 s estimated for 60 MP (32.8 s
here × 0.77), if memory allowed it.

## The front's memory: what libjxl needs

`pipeline.py floor` loads a Rec.2020 uint16 frame from `.npy` and encodes it at effort 7, 2 threads, here
(`results/floor_local.jsonl`). This is the least any pipeline built on imagecodecs' `jpegxl_encode` needs, since the
call takes the whole frame:

| Size | Frame MB | Peak with the frame loaded MB | Peak while encoding MB |
|---|---|---|---|
| 12 MP | 69 | 130 | 413 |
| 25 MP | 143 | 204 | 591 |
| 40 MP | 229 | 290 | 766 |
| 50 MP | 286 | 347 | 878 |
| 60 MP | 343 | 405 | 993 |

libjxl adds about 280 MB at 12 MP and 590 MB at 60 MP beyond the frame. Under a 500M cap, the whole-frame encode at
effort 7 reaches about 18 MP; efforts 3–4 save about 50 MB. So D38's "60 MP within 400 MB" is out of reach through
this call. The dataset's 50 MP Pixel photos would be refused on the front.

The wheel's bundled libjxl (the same pinned 0.12.0) exports libjxl's chunked input API:
`JxlEncoderAddChunkedFrame`, `JxlEncoderSetOutputProcessor` and the BUFFERING frame option. That API encodes a large
image a region at a time, without holding a full float copy. imagecodecs doesn't expose it, but a small `ctypes`
binding to that same `.so` could reach it. Whether it fits 60 MP in 400 MB, and whether its output differs from the
whole-frame encode, are not measured here.

## Memory in the prototype itself

The first version made avoidable copies:

- **JPEG decode:** Pillow's 4-byte-per-pixel image, then `convert("RGB")`'s second one, then `np.asarray`'s copy.
- **Orientation:** a full copy, from `np.ascontiguousarray`.
- **Grey images:** a 3× copy, from `np.repeat`.
- **Colour strips:** five float32 temporaries per 512-row strip.
- **Hashing:** a full copy of the 16-bit frame, from `rec.tobytes()`.

The fixed version keeps the same arithmetic:

- One array per photo, imagecodecs' JPEG decode, the turn as a view.
- 128-row strips with in-place operations, so a strip's temporaries are two float32 strips.
- Hashing over the buffer itself.

It writes byte-identical files for all eleven photos (`results/canon_local_memfix.jsonl`, against `canon_local.jsonl`
and `eff_local.jsonl`). The peak before the encoder
fell by about 55 MB at 12 MP, 210 MB at 50 MP and 245 MB at 60 MP, to 618 MB at 60 MP. The overall peak didn't move,
since the encoder sets it (above).

`pipeline.py`'s analysis subcommands (`fidelity`, `srgb`, `compare`) are not memory-lean: they hold full frames and
full-frame int16 differences. Only `canon`, `floor` and `resize` were used for memory figures.

**After the review (the final version, `results/canon_local_final.jsonl`):**

- **A 10- or 12-bit HEIC was scaled wrong.** pillow-heif shifts the samples into the top of 16 bits, so 10-bit white is
  65472, not 65535. The final version divides by the decoder's own full scale. `heic10`'s file changed; the other ten
  are byte-identical.
- **The cross-machine results for `heic10` above were made before this fix.** The code it changed is plain NumPy
  arithmetic, which agreed exactly on all three machines for every other file, so this wasn't re-run remotely.
- **Grey with alpha** (a 2-channel PNG) now decodes. A synthetic one ran.
- **CMYK** is refused.

`results/canon_local.jsonl` (the first version) ran all nine photos in one process, so its `peak_mb` is cumulative.
Every other file runs one process per photo. `eff_yappy.jsonl` holds `p3rot6` efforts 3 and 4 twice: once from the
run stopped before effort 8 and 9 on the 50/60 MP files (to keep memory free beside the live service), and once from
the rerun. Both gave the same bytes.

## The file's colour encoding (finding 5)

imagecodecs reports no colour encoding. `jxl_header.py` reads it, plus the size, orientation, bit depth, extra
channels, preview and animation flags, and the container's box types.

- `check_header.py` compared it with libjxl's own reading on 31 files and found no disagreement
  (`results/check_header.txt`). It compares the colour encoding, white point and size. Bit depth, extra channels and
  boxes are printed but have no libjxl oracle here.
- Those 31 files: every primaries × transfer combination imagecodecs writes, 8-bit, 16-bit, float, grey, lossless,
  alpha, the container format, the five example photos and a 60 MP canonical file.
- A canonical file reads as RGB, D65, BT2100 primaries, the BT709 transfer, relative intent, 16 bits, XYB, no extra
  channels, orientation 1 and a bare codestream.
- Not covered: a non-1 orientation and an ICC-only file (`want_icc`). imagecodecs can write neither, so P3's tests
  need crafted headers for those near misses.
