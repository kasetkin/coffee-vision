# ML-37 D10: how crops and patches are kept once nothing on the path is 8-bit

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D10, 2026-11-23, on the local checkout
at `01e6ac1` (branch `ml37-linear-canonical`). A recommendation for the owner, not a decision. The ticket's options:
(a) crops as lossless 16-bit integer JPEG XL of the read-back's sRGB values, patches in memory as float16; (b) crops
as lossless half-float JPEG XL; (c) no crop store, crops cut from the read-back when the head is fitted. The ticket
recommends (a); this note recommends (b), and says why below. Every number comes from the scripts listed under
"Files", run on this machine with 2 threads (imagecodecs 2026.10.10, libjxl 0.12.0, numpy 2.5.3).

## Summary

- **All three are exact.** Lossless JPEG XL returns uint16 and float16 crops bit for bit: every one of the 65,536
  uint16 values, every finite float16 pattern plus ±inf, NaN and −0, at efforts 1, 5 and 7 (`d10_exact.txt`), and
  30 of 30 real crops in each form (`d10_crop_forms.txt`). (c) stores nothing, so it is trivially exact.
- **Size, against today's 8-bit crops** (30 crops from 19 sessions, effort 5): 16-bit integer **4.31×**
  (3.03 bytes/px), half-float **2.71×** (1.91 bytes/px), today's 0.70 bytes/px. For the pool's 1,811 crops: today
  13.2 GB, (a) **57.1 GB**, (b) **35.8 GB**, (c) 0. All 1,868 session crops: 13.9, 60.3, 38.0 GB.
- **Half-float is smaller because it stores fewer distinct values where the beans are.** libjxl codes a float16
  sample as its integer bit pattern (sign, exponent, mantissa), and the modular coder then works on those integers
  as it does on uint16 (`enc_modular.cc`, `float_to_int`). Between 0.25 and 1, one float16 step equals 16 to 32
  uint16 steps. A lossless coder has to keep every bit of noise, so the noise costs about 4 to 5 fewer bits per
  sample in half-float (§3).
- **No form's precision shows up in DINOv3's embedding.** On 24 real 448 px patches, 1 − cosine to the float32
  patch's embedding is, at the maximum: uint16 1.2e-7, float16 **3.0e-7**, uint8 2.2e-5. Float16 moves the embedding
  exactly as much as casting the model's input to fp16 does, which the serving graph on `yappy` does to every input
  (ADR 0019). experiments.md's cross-CPU limit is 2e-7. Two patches of the same crop differ by 8.4e-2.
- **Patches in memory, the binding constraint:** float16 and uint16 both take 1.20 MB per 448 px patch: 7.23 GB for
  the training split against 3.61 GB today. The peak while the patches are held would be about 8.1 GB, against
  ML-8 P6's 4.5 GB measured "of about 7 GB available". It does not fit as the fit is written today, whichever form
  the crops take (§5).
- **(c)'s cost:** rebuilding a crop from its canonical photo takes 0.352 s/MP, against 0.142 (a) and 0.156 (b) to
  decode a stored crop. That is about 1.8 h against about 45 to 50 min per pass over the pool at 2 threads. It also
  removes the DVC-pinned copy of what training reads (§6).
- **Disk here:** `/workspace` has 62 GB free (93% used, `.dvc/cache` 84 GB), and the pool is a full copy of the
  session crops (link count 1, ext4). The new crops need cache + sessions + pool, less today's 27 GB of working
  copies: about +150 GB under (a) and +85 GB under (b). Neither fits before a push and a cache cleanup (§4).
- **Recommendation: (b), half-float crops, with patches in memory as float16.** Its crops are 37% smaller than
  (a)'s, and its rounding (at most 2.44e-4, 1/8 of 8-bit's) is the precision the ticket's own (a) already gives the
  patches and the precision DINOv3 runs at when serving. (a)'s extra bits are thrown away one step later. (c) is the
  fallback if disk is the binding constraint. This **contradicts the ticket's recommended (a)**, and the
  contradiction is stated in §7.

## 1. What is kept today, and what D10 changes

- **Crops:** `<stem>__cropped.jxl`, "the crop's 8-bit sRGB pixels as lossless JPEG XL", effort 5
  (`coffeecv/crop_io.py` docstring and `EFFORT = 5`; ML-19 D32: "Lossless JPEG XL in 8-bit sRGB, the array serving
  computes"). Lossless, so "the crop training reads back is the array serving computes, to the pixel"
  (`tests/test_seg_parity.py` docstring). The segcrop stage writes them per session, and `merge_segpool` copies them
  into `data/segcropped/pool` with `shutil.copy2` (`coffeecv/merge_rig.py:113`).
- **Patches:** kept as `np.asarray(patch, dtype=np.uint8)` at `patch_store_size` 448 (`coffeecv/dataset.py:502–506`).
  Each seed's `fit_one_seed` builds each split from the crops, holds the training split while it is embedded with
  its augmented views, and drops it before the head fits (`coffeecv/fit_frozen_head.py:433ff`; ADR 0025).
- **D2/D3 (ticket §1):** "Nothing on the path is 8-bit: not the read-back, the resizes, the stored crops, the patches
  or the augmentation." So both have to change. The read-back's output is float32 sRGB in [0, 1] (tone-mapped or
  clipped), and D10 decides what precision it is kept at.

The stand-in for the read-back (`d10_crop_forms.py` docstring): today's canonical photo decoded, the crop's box read
back by the same LittleCMS transform `load_canonical` uses but with float32 out, clipped to [0, 1] and filled outside
the bean mask. Rounded to 8 bits it is the stored crop: in every crop at least 0.999993 of the samples are equal, and
none differs by more than 1 level. It is today's canonical photo, not D1's half-float one, so the lossy-coding noise in
its low bits is today's. Sizes may shift once R2 measures the new photo.

## 2. Exactness

`d10_exact.txt`, synthetic:
- uint16, all 65,536 values: bit-exact at efforts 1, 5 and 7, tagged sRGB or not, decoded as uint16. The header says
  `16-bit float=False`.
- float16: all 15,361 patterns in [+0, 1] (1,023 subnormals included), every finite pattern, and inf, −inf, NaN and
  −0, all bit-exact at efforts 1, 5 and 7. The header says `16-bit float=True`, and the decoder returns float16.
  imagecodecs writes `bits_per_sample = 16`, `exponent_bits_per_sample = 5` for dtype `'e'`
  (`_jpegxl.pyx`, the `dtype.char == 'e'` branch), the binary16 case of libjxl's `JxlBasicInfo`
  (`codestream_header.h`: "if the original data is half-precision (binary16) floating point, bits_per_sample is 16
  and exponent_bits_per_sample is 5").

`d10_crop_forms.txt`, real crops: "bit-exact: u8 30/30, u16 30/30, f16 30/30", and at efforts 1, 3, 5 and 7 on 5
crops.

What exactness means differs by option. In (a) and (b) the stored crop is exact to its own rounding of the float32
read-back: at most 7.7e-6 for uint16 and 2.44e-4 for float16 (measured, `d10_crop_forms.txt` "rounding"). Today's
pixel-for-pixel parity with serving (`test_seg_parity`) survives only if serving applies the same rounding to its
in-memory crop before cutting patches: `np.rint(x * 65535) / 65535` or `x.astype(np.float16)`, one pass over the crop.
Otherwise parity becomes a tolerance. (c) has parity by construction, since both sides cut from the read-back.

## 3. Size and time

30 crops in the pool, drawn per session (19 sessions, one D18 fallback), 310 MP of crop, 201 MP encoded (crops over
8 MP measured on a centre band). Effort 5, 2 threads (`d10_crop_forms.txt`):

| form | bytes/px | × today | encode s/MP | decode s/MP | per-crop ratio to today (min, median, max) |
|---|---|---|---|---|---|
| u8 (today) | 0.702 | 1.00 | 0.365 | 0.125 | |
| u16 (a) | 3.028 | 4.31 | 0.445 | 0.142 | 3.28, 4.38, 4.84 |
| f16 (b) | 1.906 | 2.71 | 0.496 | 0.156 | 2.03, 2.68, 3.57 |

Effort, on 5 crops (22 MP): f16 is 2.475 bytes/px at effort 1 (encode 0.011 s/MP), 2.017 at 5 and 1.966 at 7
(1.738 s/MP). u16 is 3.549, 3.255 and 3.200. Effort 5 stays the right point. Effort 1 encodes 45 to 55× faster for
23 to 9% more bytes, which is worth knowing for the front, but crops are written by the store.

**Extrapolated** (each session's stored 8-bit bytes × its sample's ratio; `d10_crop_forms.txt`):

| | today | (a) u16 | (b) f16 | (c) |
|---|---|---|---|---|
| pool, 1,811 crops | 13.19 GB | 57.13 GB | 35.84 GB | 0 |
| per crop, mean | 7.3 MB | 31.5 MB | 19.8 MB | 0 |
| all sessions, 1,868 crops (what DVC stores, §4) | 13.91 GB | 60.25 GB | 37.96 GB | 0 |
| writing every session's crops (≈19.8 GP, 2 threads) | 2.0 h | 2.4 h | 2.7 h | 0 |
| reading the pool once (≈18.8 GP, 2 threads) | 0.6 to 1.2 h¹ | 0.74 h | 0.81 h | 1.8 h |

¹ 0.125 s/MP for a crop re-encoded in memory; 0.239 s/MP timing the decode of today's stored file per photo (2.47 s).
The megapixels are derived from bytes ÷ 0.702 bytes/px, so they are approximate.

**Why half-float is smaller than 16-bit integer.** libjxl's modular encoder turns each float sample into an integer
before prediction and entropy coding. For a 16-bit float with 5 exponent bits, `float_to_int` reassembles sign,
exponent and the top 10 mantissa bits into a 16-bit integer, and refuses a value that would lose precision
(`lib/jxl/enc_modular.cc`, v0.12.0, lines 154–235: "convert binary32 float that corresponds to custom [bits]-bit
float ... to a [bits]-bit integer representation"; `mant_bits = bits - exp_bits - 1`). For positive values that
integer rises with the value and is linear within each power-of-two range. In a range [2^e, 2^(e+1)) one step is
2^(e−10):
- [0.5, 1): 1/2048, that is 32 uint16 steps or 1/8 of an 8-bit level;
- [0.25, 0.5): 1/4096, 16 uint16 steps;
- [2^−5, 2^−4) (about 8 to 16 8-bit levels): 2 uint16 steps;
- below 2^−5: finer than uint16.

So [0, 1] holds 15,361 float16 codes against 65,536 uint16 codes (`d10_exact.txt`). The pixels' own fine-scale
variation is 0.17 to 3.79 8-bit levels (median 0.75; robust sigma of the green channel's second difference on bean
pixels, `d10_crop_forms.txt`). That is noise from the sensor and from the canonical photo's lossy coding, and a
lossless coder has to keep it. At 0.75 levels it spans about 190 uint16 codes but only about 6 to 12 float16 codes in
the mid-tones, so it needs about log2(16 to 32) = 4 to 5 fewer bits per sample. The measured gap is 8.1 against 5.1
bits per sample (3.03 and 1.91 bytes/px). Today's 8-bit is 1.9 bits per sample, because its step is larger than most
of the noise. This is an estimate that is consistent with the measurement, not a model of libjxl's coder.

**Weighed.** The bytes half-float saves are noise bits, below anything §5 shows DINOv3 can resolve. What half-float
gives up:
- **A non-uniform step:** at most 2.44e-4 near white, finer than uint16 in deep shadow.
- **Encode and decode about 10% slower** than uint16.
- **Values outside [0, 1]:** it can also store them, which uint16 cannot. The read-back clips or tone-maps to [0, 1],
  so this does not matter today.

## 4. DVC and disk

- **The remote:** DVC addresses objects by content (md5), and `merge_rig` copies crops byte for byte, so the pool's
  copies cost the remote nothing extra. The new bytes are the session totals: **60.3 GB (a), 38.0 GB (b), 0 (c)**,
  against today's 13.9 GB. Today's crops stay in the remote alongside them (ADR 0028: the cache is never the last
  copy, and history is kept). This repo records no push throughput. At an assumed 10 MB/s a push would take
  1.7 h (a), 1.05 h (b) and 0.4 h today; at 50 MB/s, 20 min, 13 min and 5 min. Those rates are hypothetical.
- **This machine:** 62 GB free on `/workspace` (`df -h`, 93% used), with `.dvc/cache` at 84 GB.
  `.dvc/config` sets no `cache.type`. On ext4 there is no reflink, so DVC copies. A pool crop has link count 1
  (`stat`), so it is a full copy. Each crop set lives three times: in the cache, in the session dirs and in the pool.
  - (a): about +60 GB cache, +117 GB working copies, −27 GB of today's working copies, so **≈ +150 GB**.
  - (b): about +38, +74, −27, so **≈ +85 GB**.
  - (c): no crop bytes at all.

  Neither (a) nor (b) fits without first pushing and then removing the old crops from the local cache, which
  ADR 0028 allows once the remote holds them (see `docs/agents/dvc.md` on `gc --all-commits`). Linked working copies
  (`cache.type` hardlink or symlink) would also fit, but that is a DVC configuration change and is not proposed here.
  The remote's free space was not checked.

## 5. Patches: precision and memory

**What DINOv3 resolves** (`d10_embed_precision.txt`). 24 patches of 448 px (2 from each of 12 crops, ≥ 80% bean),
cut from the float32 read-back. Each was kept in four forms, resized to 224 by PIL "F" bilinear (D11 (a)),
normalised, and embedded by the frozen `dinov3_vitb16` (`cls_mean`) in float32 on CPU, 2 threads. 1 − cosine to the
float32 patch's embedding:

| form | median | max | relative L2, median |
|---|---|---|---|
| uint16 (1/65,535) | 0 | 1.19e-7 | 1.1e-5 |
| float16 | 1.19e-7 | 2.98e-7 | 3.1e-4 |
| uint8 (1/255, today) | 3.4e-6 | 2.2e-5 | 2.6e-3 |
| yardstick: the float32 input cast to fp16 after normalisation | 1.19e-7 | 2.98e-7 | 3.3e-4 |
| yardstick: two patches of the same crop | 8.4e-2 | | |

- **Float16 moves the embedding exactly as much as the fp16 cast** that serving's DINOv3 graph already applies on
  `yappy`. ADR 0019 states that this cast "changes 0.48% of top-1 answers" over 157 photos, and patch sampling changes
  0.85%.
- **Uint16 is at float32's own noise floor.**
- **Uint8 is 20 to 70× further from float32 than float16 is**, though still 4 orders of magnitude below what separates
  two patches. The ticket's case against 8-bit is the owner's decision D2/D3, not this measurement.
- **In pixel terms**, at the brightest range [0.5, 1) float16's rounding RMS is 0.036 8-bit levels. Against the crop
  with the least fine-scale variation (0.17 levels) that adds 4.5% to the variance, and 0.2% at the median.
  Uint8's 0.29 levels adds 15% at the median and 290% at the least.

24 patches is small, but the result is at the arithmetic floor and matches its yardstick exactly, so more patches
would not change the conclusion.

**Memory** (`d10_memory.txt`, arithmetic on `params.yaml` and 10 classes): 6,000 training patches at 448:

| patch dtype | per patch | training split | peak while held (ML-8 P6 + the difference) |
|---|---|---|---|
| uint8 (today) | 0.60 MB | 3.61 GB | 4.5 GB (measured) |
| uint16 or float16 | 1.20 MB | 7.23 GB | ≈ 8.1 GB |
| float32 | 2.41 MB | 14.45 GB | ≈ 15.3 GB |

- **The host:** ML-8 P6 measured "peak RSS 5.5 GB (the augmented fit; 4.5 GB while the patches are held), of about
  7 GB available" (`docs/ml-0008_plan_data_refresh_retrain.html`, P6). Patches at 16 bits do not fit that as written.
  This holds for (a), (b) and (c) alike, because the ticket keeps patches at 16 bits in every option.
- **What is needed at a time:** one float32 view for albumentations is 2.4 MB, and a batch of 32 views 77 MB.
  Albumentations' OpenCV transforms reject float16, so each patch is converted to float32 as it is drawn
  (research_d11_float_resize_and_augment.md, §5).
- **Per crop:** while patches are cut, one 18 MP crop takes 109 MB at 16 bits.

So D10 needs one of the following, which is for the plan to choose, not this note:
1. **Embed in chunks.** Views are already keyed by (run seed, patch, view) (ADR 0025), so a chunked fit draws the
   same views. Hold a chunk's patches, embed them and their views, and drop them.
2. **Keep the training split's patches in a disk-backed `np.memmap`** of 7.2 GB.
3. **Run the fit on a host with more memory.**

The memory peak is arithmetic, not measured.

## 6. The cost of (c)

Per photo, 2 threads, scaled to the whole crop (`d10_crop_forms.txt`):
- **The rebuild:** the canonical decode (0.29 s), the box's float read-back (0.319 s/MP) and the fill (0.009 s/MP),
  in all **3.64 s per photo, 0.352 s/MP**.
- **Against stored crops:** 1.47 s (a) and 1.62 s (b) to decode a stored crop, and 2.47 s to decode today's stored
  file.

Each seed of a fit reads every pool crop once (`fit_one_seed` builds each split from the crops). (c) therefore adds
about 1 to 1.1 h per seed at 2 threads over (a) or (b), on top of ML-8's ≈ 2.8 h per seed. More threads shorten
both.

Two caveats on these numbers:
- The read-back here is LittleCMS float. D9 (a)'s numpy matrix may be faster.
- The decode is of today's canonical photo (2.25 MB, 15.2 MP mean). D1's half-float photo is unmeasured (R2).

What (c) gives up beyond time:
- **The pinned training input.** Today a fit's input is the crop files, whose md5s are in `dvc.lock`. Under (c) it is
  the canonical photos plus the read-back code plus the boxes and masks, so a change to the read-back changes
  training without any stage going stale.
- **The crop files themselves**, which `patch_samples.py` and the parity test read.

What (c) gains:
- No crop bytes on disk or the remote.
- Train/serve parity by construction.
- No re-crop when only the read-back changes.

## 7. Recommendation

**(b): crops as lossless half-float JPEG XL (effort 5, tagged sRGB primaries and transfer), patches in memory as
float16.** The reasons:
- **Same exactness as (a).** It is lossless and bit-exact on every pattern (§2).
- **37% smaller than (a).** 35.8 GB against 57.1 GB for the pool, 2.7× today against 4.3×; 38 GB against 60 GB to
  push (§3, §4).
- **Its precision is the one the path already settles on.** The ticket's (a) holds the patches in float16, and
  serving's DINOv3 runs in fp16. Float16 crops lose nothing that the next step keeps, and DINOv3's embedding cannot
  tell them from float32 beyond the fp16-cast floor (§5).
- **One dtype end to end.** The source is a half-float photo and the patches are float16.

Under (b), serving should cast its in-memory crop to float16 before cutting patches, so that `test_seg_parity` stays
pixel-exact (§2).

The ticket's (a) argues "steps of 1/65,535, far finer than anything the models resolve". That is true, and it is the
reason not to pay 1.6× (b)'s bytes for them. (a) is the right choice only if a future reader of the crops needs a
uniform step, for example an integer-only tool. Nothing in the repo is such a reader today: both readers of crops, `dataset.py:446` and
`patch_samples.py:77`, go through `crop_io.load_crop`.

**(c) is the fallback if disk decides it.** It would cost about 1 h more per seed at 2 threads and lose the DVC-pinned
training input.

Whichever option is chosen, the fit has to change how it holds 7.2 GB of patches (§5).

## 8. What stays unverified

- **Sizes on D1's photo.** All are on today's canonical photo (uint16, BT.709 curve, q95). The half-float linear
  photo's coding noise, and so the crops' low bits, may differ (R2).
- **The sample.** 30 crops, those over 8 MP measured on an 8 MP centre band. The extrapolation multiplies each
  session's stored bytes by its sample's ratio, with 1 to 3 crops per session.
- **The memory peak.** 8.1 GB is arithmetic on ML-8 P6's measurement, not a run.
- **The fit host.** Its available memory today is not re-measured.
- **Push throughput and the remote's free space.** Neither was measured.
- **The rebuild time in (c).** It uses LittleCMS's float transform, not D9's numpy read-back.
- **Today's stored-file decode.** Its time (0.239 s/MP) is per stored file and includes reading it. The other decode
  times are from bytes in memory.

## Files

- `analysis/ml37/d10_exact.py` → `d10_exact.txt`: round trips of every uint16 and float16 pattern.
- `analysis/ml37/d10_crop_forms.py` → `d10_crop_forms.txt`: size, time and exactness on 30 crops, the extrapolation,
  and (c)'s rebuild time. Per-crop rows are in `tmp/ml37/d10/rows.jsonl` and the log in `tmp/ml37/d10/crop_forms.log`
  (local only).
- `analysis/ml37/d10_embed_precision.py` → `d10_embed_precision.txt`: DINOv3 on 24 patches
  (`tmp/ml37/d10/patches.npz`, local only).
- `analysis/ml37/d10_memory.py` → `d10_memory.txt`: the patch store's memory.

## Sources

- libjxl v0.12.0, `lib/jxl/enc_modular.cc`, `float_to_int` (lines 154–235) and its call for colour channels (lines
  808–823): https://raw.githubusercontent.com/libjxl/libjxl/v0.12.0/lib/jxl/enc_modular.cc (read in
  `tmp/ml37/d10/downloads/libjxl_mod/`).
- libjxl `include/jxl/codestream_header.h`, `JxlBasicInfo.bits_per_sample` and `exponent_bits_per_sample` (lines
  118–128). imagecodecs `imagecodecs/_jpegxl.pyx`, the encoder's dtype branches (lines 360–373). Both read in
  `tmp/ml37/d10/downloads/`; the installed versions are imagecodecs 2026.10.10 and libjxl 0.12.0.
- The repo at `01e6ac1`:
  - code: `coffeecv/crop_io.py`, `coffeecv/dataset.py:502–506`, `coffeecv/fit_frozen_head.py:433ff, 734ff`,
    `coffeecv/merge_rig.py:113`;
  - tests and configuration: `tests/test_seg_parity.py`, `params.yaml`, `dvc.yaml` (`segcrop`, `merge_segpool`),
    `.dvc/config`;
  - docs: `docs/ml-0037_ticket_linear_canonical_photo.html` (§1, D10), `docs/ml-0019_ticket_canonicalization.html`
    (D32), `docs/ml-0008_plan_data_refresh_retrain.html` (P6), `docs/adr/0019-serving-runs-litert-accelerator-per-host.md`,
    `docs/adr/0025-frozen-head-fits-augment-per-run-in-memory.md`,
    `docs/adr/0028-the-dvc-cache-is-never-the-last-copy.md`, `docs/agents/experiments.md` (cross-CPU cosine),
    `docs/agents/dvc.md`;
  - analysis: `analysis/ml37/research_d11_float_resize_and_augment.md` (§5).
