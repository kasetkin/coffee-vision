# OPS-22 P2: the front's settings on each kind of source (D4)

Measured 2026-11-10 on this machine (Ryzen 5 5600U), each photo in its own process held to 2 CPUs by `taskset -c 0,1`
as the front's 2 vCPU, with `coffeecv/canonical.py`'s `FRONT_SETTINGS` (integer colour path, effort 3, quality 90,
kept up to 18 MP, over that at most 15 MP), by `p2_sources.py`:

    python -I analysis/ops22/p2_sources.py /workspace

"vs float" compares the front's canonical photo, read back as the inference host reads it (`load_canonical`, 8-bit
sRGB), with the same photo made on the float colour path at the same effort, quality and thresholds: the absolute
difference over every sample, in 8-bit levels. Peak MB is the process's `VmHWM`, as the check service logs it; this
machine cannot hold a process to the front's `MemoryMax=600M` (no systemd user bus, a read-only cgroup tree), so the
front's own figure is P4's, from the live log.

| Source | Bits | Upload | Front's photo | Size | Seconds (2 CPUs) | Peak MB | vs float: mean | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| JPEG, 8-bit (Pixel, 12 MP) | 8 | 3.18 MB | 4080 x 3072 | 1.95 MB | 2.12 | 255 | 0.031 | 1 | 15 |
| JPEG, 8-bit (Pixel 9 Pro, 50 MP) | 8 | 7.28 MB | 3360 x 4463 | 1.14 MB | 3.07 | 286 | 0.034 | 1 | 30 |
| HEIC, 8-bit (iPhone) | 8 | 4.94 MB | 3024 x 4032 | 5.05 MB | 2.68 | 301 | 0.046 | 1 | 23 |
| HEIC, 10-bit (from the DNG) | 16 | 8.03 MB | 2732 x 3628 | 2.08 MB | 2.98 | 335 | 0.098 | 2 | 22 |
| DNG (Pixel 9 Pro, 9.9 MP) | 16 | 15.68 MB | 2732 x 3628 | 2.08 MB | 4.15 | 298 | 0.0 | 0 | 10 |
| PNG, 16-bit (from the DNG) | 16 | 52.18 MB | 2732 x 3628 | 2.18 MB | 2.8 | 328 | 0.099 | 2 | 27 |
| PNG, 8-bit with transparency | 8 | 16.90 MB | 4080 x 3072 | 1.26 MB | 4.92 | 275 | 0.0 | 0 | 0 |
| JPEG XL, the grinder example (store's canonical, 50 MP) | 16 | 5.30 MB | 3360 x 4463 | 1.06 MB | 3.79 | 468 | 0.0 | 0 | 4 |

The 10-bit HEIC and the 16-bit PNG were made from the DNG's 16-bit sRGB development (the dataset has neither); the PNG
with transparency is the 12 MP JPEG with a radial alpha. Raw lines: `p2_sources.jsonl` in the work directory, not kept.

What it says:

- **Every source is inside a few seconds**: 2.1-3.1 s for 8-bit photos up to 50 MP (ML-19 P8 measured 4.4 s and 17.3 s
  on the front for the same kinds with the store's settings), 2.8-4.2 s for the 16-bit ones. The PNG with transparency
  keeps the float path (D4 (a)), so it takes the float path's time (4.9 s at 12 MP) and is identical to it.
- **The colour path changes almost nothing once encoded**: a mean of 0.03-0.10 levels, 99% of samples within 1-2. The
  largest differences (10-30 levels) are where q90 at effort 3 quantizes two slightly different inputs differently, at
  edges; before the encode the integer path is within one level of the float path on all five real or derived 8- and
  16-bit sources above (1.0-2.5% of samples one level off, the rest equal; checked with `to_rec2020` on the decoded
  photos, as `tests/test_canonical.py`'s `IntegerColour` does on synthetic ones).
- **The DNG and the grinder example come out as on the float path** (mean 0.0): rawpy's linear Rec.2020 and the
  example's Rec.2020 reach Rec.2020 through LittleCMS's matrix-shaper, which its optimization leaves almost exact.
- **The grinder example peaks at 468 MB** (R3: 467 MB): it is decoded whole at 16 bits before the cap, 300 MB of pixels.
  Under the unit's `MemoryMax=600M` with little room; P4 reads its `peak_mb` on the front.
- **The iPhone's 8-bit HEIC grows** (4.94 to 5.05 MB): its own HEVC is more efficient than JPEG XL at effort 3 on that
  photo. Every other source shrinks, 50 MP ones most.
