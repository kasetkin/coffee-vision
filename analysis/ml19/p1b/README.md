# ML-19 D52: libjxl's chunked encoder, for the front's memory

Measured 2026-11-03 for [ML-19](../../docs/ticket_canonicalization.html) D52 (plan Q6, option (a)), after P1
([`../p1/README.md`](../p1/README.md)) found that the whole-frame encode needs about 1 GB at 60 MP. Raw results are in
`results/`. The front ran in a throwaway uv environment under `/tmp/ml19p1`, each photo in its own
`systemd-run --user --scope -p MemoryMax=500M -p MemorySwapMax=0`. The scope's own `memory.peak` was read from inside
it (`cgroup_peak_mb`), beside the Python process's `ru_maxrss` (`peak_mb`).

## Why the whole-frame encode was large (libjxl 0.12.0's source)

- `JxlEncoderAddImageFrame`, the call imagecodecs makes, copies the caller's frame (`encode.cc`: "the input callbacks
  are only guaranteed to be available during frame encoding when both the input and the output is streaming. In all
  other cases we need to create an internal copy of the frame data"). So two full 16-bit frames are alive: the
  caller's and libjxl's.
- At effort 7 and distance 0.55, libjxl already encodes one DC group (2048 × 2048) at a time (`CanDoStreamingEncoding`
  in `enc_frame.cc`). At effort 8 and above it doesn't stream at this distance, hence their 3.3 GB at 12 MP.
- `JxlEncoderAddChunkedFrame` with an output processor set streams the input too. libjxl asks for rectangles of at
  most 2048 × 2048 and releases each, so the caller can convert each rectangle from the source only when it's asked
  for.

## The binding

`jxl_chunked.py` is about 200 lines of `ctypes` against the `libjxl` and `libjxl_threads` that ship inside imagecodecs'
wheel, so the pinned library is still the one in use (D44). Nothing is compiled, so nothing has to be built for the
front.

- It configures the encoder as imagecodecs does:
  - uint16 RGB, no container, D7's colour encoding as an enumeration;
  - `JxlEncoderDistanceFromQuality(95)`, effort 7 left as libjxl's default;
  - a frame header with duration 1.
- `stream.py` is P1's pipeline with that encoder: decode once, then each rectangle libjxl asks for goes through P1's
  `to_rec2020`, in 256-row strips.

Why not C, C++ or Rust: the memory is libjxl's, the source image's and the interpreter's. A native wrapper would save
only the interpreter and imports: 38 MB for Python, NumPy and imagecodecs, 61 MB with OpenCV, pillow-heif and rawpy
(`results/local_levers_60mp.txt`). It would also bring a compiled artefact to build per glibc, and the front runs an
older glibc (2.36) than this machine (2.41).

## Results

**The same bytes.**

- The chunked encode of a frame gives the same file as `jpegxl_encode` of that frame, at 12 and 60 MP.
- `stream.py` gives the same files as P1's final pipeline for all ten photos tried here, at 1 and 2 threads
  (`results/local_stream.jsonl`, against `../p1/results/canon_local_final.jsonl`).
- On the front, the files are the same as this machine's, `heic10` with its corrected scale included
  (`results/front_e7_t2.jsonl`).
- So D30 and D51 carry over unchanged.

**Memory and time on the front, effort 7, 2 threads** (`results/front_e7_t2.jsonl`). MemAvailable stayed at or above
577 MB, and the live check service stayed active.

| Photo | Scope `memory.peak` MB | Python peak MB | Total s | Whole-frame, here, peak MB |
|---|---|---|---|---|
| 12 MP JPEG (`srgb`, `p3rot6`) | 285–286 | 319–320 | 4.6 | 407–411 |
| 12 MP HEIC, 8 / 10 bit | 279 / 352 | 310 / 383 | 4.5 / 5.1 | 435 / 470 |
| 12 MP DNG | 366 | 386 | 4.7 | 461 |
| 20 MP JPEG | 324 | 357 | 7.2 | — |
| 30 MP JPEG | 342 | 373 | 10.9 | — |
| 40 MP JPEG | 385 | 416 | 14.4 | — |
| 50 MP JPEG (Pixel) | 396 | 427 | 17.5 | 860 |
| 60 MP JPEG | 470 | 493 | 21.8 | 1,009 |
| 50 MP DNG (Pixel 9 Pro) | killed by the OOM killer at 500M | — | — | — |

The last row is `results/front_raw50_oom.txt`. rawpy's 16-bit development alone holds several full frames.

**One thread uses more memory, not less** (`results/front_e7_t1.jsonl`): 459 MB at 50 MP and 500 MB at 60 MP (at the
cap, but not killed), and 1.5× the time. Two threads it is; the front has 2 vCPU.

**Other parameters, 60 MP here** (`results/local_levers_60mp.txt`):

- BUFFERING 2 or 3: the same bytes and within 12 MB of the default.
- Efforts 6 and 5: the same memory.
- Effort 4: 43 MB less. Effort 3: 53 MB less.

So libjxl's own working set, about 240 MB at effort 7, is the floor. Once Python and the 8-bit source (180 MB at
60 MP) are added, 60 MP needs about 470 MB.

## What it means

- D52 (a) works:
  - every photo up to 60 MP canonicalizes on the front within its 500M;
  - the peak is about halved;
  - every file is byte-identical to the whole-frame encode.
- The owner's fallback (cap everything over 18 MP down to 12 MP) isn't needed.
- For D38 (P8's stop):
  - D38 asked for 60 MP within 400 MB. The front meets that up to 50 MP (396 MB): every Pixel 9 Pro photo, and
    everything under it.
  - 50–60 MP takes up to 470 MB.
  - A ceiling of 50 MP leaves 100 MB under MemoryMax; 60 MP leaves 30.
- For D24 (RAW on the front): a 12 MP DNG fits (366 MB); a 50 MP DNG doesn't.
