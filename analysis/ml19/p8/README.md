# ML-19 P8: the front measured, for D23, D24 and D38

Measured 2026-11-03 on the front (`webproxy-deploy`, AMD EPYC 9374F, 2 vCPU, 958 MB, MemAvailable 577–605 MB before
each run), with `coffeecv/canonical.py` itself (commit `a67a8fe`'s module, and D52's chunked encoder) in a throwaway
uv environment under `/tmp/ml19p1`, deleted after. Each photo ran in its own
`systemd-run --user --scope -p MemoryMax=500M -p MemorySwapMax=0`, through `front_measure.py`, effort 7, 2 threads.
`cgroup_peak_mb` is the scope's own `memory.peak`, the figure the service's MemoryMax would see; `peak_mb` is the
Python process's. The live check service stayed active throughout. Raw lines: `front_canonical.jsonl`.

| Photo | Pixels | Upload | Outcome | Seconds | Scope peak MB |
|---|---|---|---|---|---|
| JPEG, sRGB (MPO) | 12 MP | 3.0 MB | canonical | 4.4 | 282 |
| JPEG, DCI-P3, rotated | 12 MP | 2.8 MB | canonical | 4.5 | 280 |
| HEIC, 8-bit (iPhone) | 12 MP | 1.2 MB | canonical | 4.4 | 270 |
| HEIC, 10-bit (synthetic) | 12 MP | 7.4 MB | canonical | 5.0 | 340 |
| PNG, 16-bit (synthetic) | 12 MP | 58 MB | canonical | 4.7 | 399 |
| DNG (`user_realworld`) | 12 MP | 23 MB | canonical | 4.6 | 373 |
| DNG (Pixel 9 Pro, cropped) | 9.9 MP | 15 MB | canonical | 4.3 | 342 |
| JPEG, DCI-P3 | 20 MP | 3.9 MB | canonical | 7.1 | 320 |
| JPEG, DCI-P3 | 40 MP | 7.3 MB | canonical | 14.3 | 407 |
| JPEG, Display P3 (Pixel 9 Pro) | 50.1 MP | 5.8 MB | canonical | 17.3 | 400 |
| JPEG, DCI-P3 | 60 MP | 11 MB | canonical | 21.4 | 461 |
| JPEG (Fold8), with a 60 MP ceiling | 84 MP | 18 MB | refused from its header (`TooLarge`) | 0.02 | 58 |
| JPEG (Fold8), with a 60 MP ceiling | 200 MP | 31 MB | refused by Pillow's bomb limit (`TooLarge`) | 0.02 | 81 |
| DNG (Pixel 9 Pro) | 50.1 MP | 52 MB | killed by the OOM killer at 500M (`front_dng50_oom.txt`) | — | 500 |

What the measurements say:

- **Time:** every photo up to 60 MP is inside D22's 40 s. The worst is 21.4 s, and it scales at about 0.36 s per MP.
- **Memory follows pixels, not upload bytes.**
  - A 12 MP 8-bit photo takes 270–282 MB.
  - A 16-bit or RAW source adds about 60–120 MB, since it decodes at 16 bits.
  - Peaks: 40 MP 407 MB, 50 MP 400 MB, 60 MP 461 MB.
- **A photo over the ceiling costs nothing to refuse.** It's told from its header before any pixel is decoded
  (`canonicalize(..., max_pixels=...)`, at 58–81 MB, which is Python and its imports).
- **RAW is the exception.**
  - rawpy has to read the whole RAW before it can report its size, and develops at 16 bits through several full
    frames.
  - A 50 MP DNG is killed at 500M, while 10–12 MP DNGs fit (342–373 MB).
  - Under the service's own MemoryMax, an OOM kill takes the service's process, not just the request. So a RAW upload
    must be refused before rawpy develops it, by a lower RAW ceiling.
- **The front's MemAvailable** fell to 207 MB just after that kill, while the killed process's memory was being
  freed. It was back at 580 MB a moment later (`front_memavailable.txt`).

## For the owner (the plan's P8 stop)

- **D38, the pixel ceiling.**
  - D38 asked for 60 MP within 400 MB. Measured:
    - up to 50 MP within 400 MB (every Pixel 9 Pro photo, and everything under it);
    - 60 MP at 461 MB, which leaves 39 MB under MemoryMax 500M.
  - Recommended: a ceiling of 50,200,000 pixels on the front, so the Pixel's 50.14 MP passes. Photos over it would be
    refused with a message saying their size; the store canonicalizes them on this machine.
  - Alternative: 60 MP, with a 39 MB margin.
- **D24, RAW on the front.** Recommended: RAW up to 16 MP, refused above, with a message. That's the dataset's 10 and
  12 MP DNGs, while the 50 MP DNGs are refused. RAW's size is read without developing it, before the 16-bit
  development.
- **D23, the upload cap.**
  - The cap no longer guards memory: the ceilings do.
  - The 200 MP JPEG is 26–33 MB, a 50 MP DNG 52–54 MB, and the 16-bit PNG here 58 MB.
  - Recommended: 40 MB, enough for any photo the ceilings would take, plus a 200 MP JPEG so it can be refused by its
    pixel count instead of its size.
