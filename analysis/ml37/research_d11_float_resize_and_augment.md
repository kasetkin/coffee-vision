# ML-37 D11: the resizer for float input, and every augmentation's float behaviour (R5)

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D11 and R5, 2026-11-23, on the local
checkout at `01e6ac1`. A recommendation for the owner, not a decision: the resizer becomes D11's answer only when the
owner picks it. Nothing was trained and no code under `coffeecv/` was changed. Every number comes from the four
scripts named in [Files](#files), run with Pillow 13.0.0, OpenCV 5.0.0, torch 2.14.0 (CPU), albumentations 2.0.8 and
albucore 0.0.24, the versions installed from `uv.lock`, on 2 threads. The library facts were read from the libraries'
own source at those versions: Pillow's `src/libImaging/Resample.c` and `src/PIL/Image.py` at tag `13.0.0` (the
fetched `Image.py` is byte-identical to the installed one), OpenCV's `modules/imgproc/src/resize.cpp` at tag `5.0.0`,
torch's `nn/functional.py` and albumentations' and albucore's modules as installed.

The canonical photo D1 describes does not exist yet. The "float read-back" used here is today's canonical photo
decoded at 16 bits and put through the same LittleCMS transform as `load_canonical` with float32 output, clipped to
[0, 1]: the same picture as `load_canonical` without its 8-bit step (it differs from it by at most 0.500 level, mean
0.243). That isolates what the resizer and the augmentations do, which is D11's and R5's question. It does not
measure D2's read-back.

## Summary

- **Recommended: (a), PIL's bilinear in "F" mode, channel by channel.** It runs the same filter as today's 8-bit
  path, with the support widened on a downscale. Both paths go through the same `precompute_coeffs`. The 8-bit
  path also quantizes the weights to fixed point and rounds to 8 bits after each of its two passes. Measured on 13
  canonical photos against today's 8-bit output, the float path differs by at most 1.43 levels at the SAM side
  (mean 0.26), 1.45 at the patch store (mean 0.30) and 2.02 over the two classifier resizes (mean 0.31). When
  today's 8-bit picture is fed through "F", the two outputs differ by at most 1.000 level at every single resize:
  that is the rounding, not the filter.
- **(c) torch's antialiased interpolate is the same filter** as (a), to within 0.006 level. Its documentation says
  it was built to match Pillow. It is not an option for serving, though: `model_io` and the inference host run
  without torch (ML-7 D5). That makes it the test oracle for (a), the role torchvision's PIL-based transforms play
  in `tests/test_model_io.py` today.
- **(b) OpenCV is a different filter.** Its `INTER_LINEAR` reads 2 taps whatever the scale, with no antialiasing.
  Its `INTER_AREA` is a box filter, and at exactly 2× (448 → 224) `INTER_LINEAR` turns into it. Against today's
  input it differs by up to 90 levels at the SAM side (47% of values more than 1 level off) for `INTER_LINEAR`, and
  up to 29 levels for `INTER_AREA`. It is 5 to 20 times faster than (a), but the models would see a different
  picture from the one they were trained and pretrained on.
- **Speed of (a)** (2 threads; shared machine, load ≈ 12 on 12 CPUs, so ±2×):
  - 15 MP photo to 1024: 280 ms for the channels in turn, 180–220 ms on a 2-thread pool, against today's 119 ms.
  - 64 patches 448 → 224: 205 ms (157 ms pooled), against 106 ms.
  - 64 crops 800 → 448: 1,005 ms (603 ms pooled), against 468 ms.
  - About 2× today's cost, and small beside the encoders.
- **R5 does not hold as written.** None of the 15 transforms in the three pipelines refuses float32. The strict
  builder catches warnings only while a pipeline is built, and none of these warns, so every pipeline runs on float
  with no error and no warning. Four transforms need care:
  - **`HueSaturationValue`, `ImageCompression` and `Downscale` round the float picture to 8 bits inside.** They are
    wrapped in albucore's `@uint8_io`, so their output is today's 8-bit result divided by 255, exactly. Before:
    0.3–0.7% of the float read-back's values lay on the 8-bit grid. After: 100% (the share rose by 99.6–99.7
    points).
  - **`CoarseDropout`'s `fill: [124, 116, 104]` is taken in the image's own units.** On float32 the holes hold
    124.0. A later clipping colour transform (ColorJitter, PlanckianJitter) turns them pure white, and ColorJitter's
    contrast first reads the image mean those holes inflate. This is the one silently wrong result. The fix is a
    params.yaml value: `[0.486, 0.455, 0.408]` (the same 8-bit values / 255).
- **The other 11 transforms agree with their uint8 results up to rounding.** The flips and SquareSymmetry are
  exact. Affine and the blurs differ by at most 0.5 level, GaussNoise by at most 1. ColorJitter, PlanckianJitter
  and RandomGamma read about 0.96, 0.31 and 0.50 levels brighter on float. On uint8, their lookup tables truncate
  (`astype(np.uint8)`): the float path loses today's small downward bias, not the reverse.
- **The segmenter's fine-tune has no other uint8 assumption in albumentations.** Labels stay uint8 (0, 1, 255) and
  come out identical for uint8 and float photos. No embedding cache is left (ML-6 D12; the validation embeddings
  are float). Its uint8 assumptions are in the repo's code, all listed in §5:
  - `sam_resize` and `_to_chw_normalised`'s `/ 255`.
  - The training set held in memory as uint8: 1,091 training photos, about 2.6 GB today, ×2 in float16, ×4 in
    float32.
  - The silent one, outside D11 (R7): the bean-pitch grey's `astype(np.uint8)` would floor a [0, 1] float picture
    to zeros.

## 1. Pillow: one filter, two arithmetics

**The filter.** `Image.resize` (`src/PIL/Image.py:2332`) passes `Resampling.BILINEAR` to the C `resize`
(`Image.py:2440`). `reducing_gap` defaults to `None`, "no optimization" (`Image.py:2367`), so no `reduce()` step is
taken (`Image.py:2422`). The repo never passes it. In `Resample.c`, `ImagingResample` (lines 778–838) picks the
pass functions by image type:
- `_8bpc` for 8-bit images ("RGB");
- `_32bpc` for `IMAGING_TYPE_FLOAT32` ("F") (lines 795–803).

Both are handed the same `struct filter BILINEAR = {bilinear_filter, 1.0}` (line 126), and both go through the same
`precompute_coeffs` (lines 227–312):
- `filterscale = scale = (in1 - in0) / outSize`, raised to 1.0 if smaller (lines 242–245);
- `support = filterp->support * filterscale` (line 248);
- each weight is `filter((x + xmin - center + 0.5) * inv_filterscale)`, normalised to sum 1 (lines 290–301).

On a downscale by s the triangle is s times wider. On an upscale it is the plain 2-tap triangle. Mode "F" and mode
"RGB" get the same weights.

**Where the arithmetic differs.**

| | 8-bit (`_ImagingResampleHorizontal_8bpc`, `_Vertical_8bpc`) | "F" (`_32bpc`, `IMAGING_TYPE_FLOAT32`) |
|---|---|---|
| weights | quantized to `PRECISION_BITS` = 22 fractional bits (`normalize_coeffs_8bpc`, lines 314–329; line 137) | double, as computed |
| sum | integer, started at `1 << (PRECISION_BITS - 1)` (line 362): rounding half up | double (`double ss`, lines 682–700, 741–757) |
| output | `clip8`: shifted down and clipped to 0–255 (lines 219–224) | `lineOut[xx] = ss`, cast to float32, no clip |
| between passes | the intermediate image is `ImagingNewDirty(imIn->mode, …)` (`PASS`, lines 901–903): **8-bit** | float32 |

The float path therefore rounds nowhere but float32 storage. The bilinear weights are non-negative and sum to 1, so
its output stays inside the input's range and needs no clip.

**Mode "F" is single-channel.** `Image.fromarray` of an H×W×3 float32 array raises `TypeError: Cannot handle this
data type: (1, 1, 3), <f4` (checked). So (a) means three single-plane resizes. `ImagingSectionEnter` in both pass
functions releases the GIL, so the planes can run on threads.

**The measured equivalence** (`d11_resize_parity.txt`):
- An impulse test, 64 → 16 px (4× down): "F" reads 8 input pixels per output (support 4 on each side), exactly the
  inputs torch's antialias reads.
- Today's own 8-bit picture divided by 255 and resized in "F" differs from the 8-bit resize by at most **1.000
  level** at each single resize (SAM side, patch store, 448 → 224). Over the two-step chain the bound is 1.87.
  Nothing above one rounding step is a filter difference.

**A bias today's path carries.** At exactly 2× (448 → 224) the weights are 1/8, 3/8, 3/8, 1/8, exact in fixed point.
An eighth of the sums end in exactly .5, and the 8-bit path rounds them up, once per pass. Today's 224-px input is
therefore **0.136 level brighter on average** than the exact filter gives (`eval`'s mean d −0.136, the same for
"u8 as float"). At non-dyadic scales the bias vanishes: −0.003 at the patch store. The float path removes it.

## 2. OpenCV and torch

**OpenCV `cv2.resize`** (`resize.cpp`, tag 5.0.0):
- Non-area interpolation uses `ksize = 2` for `INTER_LINEAR` (line 4090), with `fx = (dx + 0.5) * scale_x - 0.5`
  (line 4110). That is 2 taps at any scale, with no widening and no antialiasing. The impulse test reads inputs
  [33, 34] for a 4× downscale.
- "True 'area' interpolation is only implemented for the case (scale_x >= 1 && scale_y >= 1)" (line 4023). It
  weighs each input pixel by its overlap with the output pixel: a box filter of width s, 4 taps at 4×.
- `INTER_LINEAR` at exactly 2× is switched to `INTER_AREA` (line 4020). At 448 → 224 the two give identical numbers.
- On uint8, the linear path uses fixed-point weights of `INTER_RESIZE_COEF_BITS = 11` (line 962;
  `fixpt = depth == CV_8U`, line 4081). On float32 it computes in float.

**torch `F.interpolate(…, mode="bilinear", align_corners=False, antialias=True)`.** Its docstring (installed
`torch/nn/functional.py`, line 5057) says:

> "Using anti-alias option together with `align_corners=False`, interpolation result would match Pillow result for
> downsampling operation."

Measured, it equals "F" to at most 0.006 level at every size, downscale or upscale. torchvision's `resize` defaults
to `antialias=True` on tensors (installed `torchvision/transforms/functional.py:392`), so torchvision's `Resize` on a
float tensor is this same filter.

## 3. Measurements: each float path against today's and against each other

Sample (`d11_resize_parity.py`): 13 canonical photos, one per capture folder, drawn with a fixed seed from 22
folders (OOD negatives excluded). Twelve are 3000×4000 to 5056×3792; one is a 959×635 internet photo. Per photo:
- the SAM-side resize (long side 1024, XL0's `image_size`; `model_io.sam_preprocess_shape`);
- 12 square crops, sides log-uniform from 224 to 1,100 px, resized to 448 (the patch store; both down- and
  upscales);
- then 448 → 224 (`eval_resize`).

The 8-bit reference is today's code: the assertion that `pil_u8` equals `model_io.sam_resize` held on every photo.
Levels are float × 255 minus uint8, unrounded.

| Stage | Path (float read-back unless said) | vs today: max | mean \|d\| | >1 level | vs "F": max | mean \|d\| |
|---|---|---|---|---|---|---|
| read-back alone | float vs `load_canonical` | 0.500 | 0.243 | 0% | | |
| SAM side | "F", today's u8 as float | 1.000 | 0.251 | 0% | | |
| SAM side | **(a) "F"** | **1.430** | **0.263** | 0.04% | — | — |
| SAM side | (c) torch antialias | 1.430 | 0.263 | 0.04% | 0.005 | 0.0000 |
| SAM side | (b) cv2 `INTER_LINEAR` | 89.7 | 2.088 | 47.2% | 89.9 | 2.037 |
| SAM side | (b) cv2 `INTER_AREA` | 29.4 | 0.859 | 24.5% | 29.3 | 0.775 |
| store → 448 | **(a) "F"** | **1.446** | **0.301** | 0.38% | — | — |
| store → 448 | (c) torch | 1.447 | 0.301 | 0.38% | 0.006 | 0.0000 |
| store → 448 | (b) `INTER_LINEAR` / `INTER_AREA` | 63.0 / 53.8 | 0.508 / 0.465 | 9.3% / 8.9% | 63.1 / 52.7 | 0.300 / 0.297 |
| 448 → 224 | **(a) "F"** | **2.015** | **0.313** | 0.61% | — | — |
| 448 → 224 | (c) torch | 2.015 | 0.313 | 0.61% | 0.000 | 0.0000 |
| 448 → 224 | (b) either (identical at 2×) | 32.3 | 0.877 | 23.9% | 32.4 | 0.750 |
| both steps | **(a) "F"** | **2.015** | **0.313** | 0.61% | — | — |
| both steps | (b) `INTER_LINEAR` / `INTER_AREA` | 46.1 / 38.1 | 1.028 / 0.976 | 26.7% / 26.6% | 45.9 / 38.2 | 0.908 / 0.860 |

At the patch store, "F" against today: downscales (side > 448) max 1.44, mean 0.29; upscales max 1.45, mean 0.31.
The large (b) differences sit on edges and texture, where a 2-tap or box filter aliases or blurs differently from
the widened triangle. The mean signed differences stay near zero (±0.003 level, beside the 2× bias of §1).

## 4. Speed

`d11_resize_speed.txt`: one canonical photo's centre (4480×3348, 15.0 MP) to 1024×765, and batches of 64 patches.
Medians of 7 runs after a warm-up, in ms. The machine was shared (load average 10–12.6 on 12 CPUs): two runs of
the same planar timing gave torch 98 and 65 ms. Read the table as a ranking with ±2× on each figure.

| Task | today `pil_u8` | (a) "F", in turn | (a) "F", 2-thread pool | (b) `INTER_LINEAR` | (b) `INTER_AREA` | (c) torch, HWC in | (c) torch, CHW in |
|---|---|---|---|---|---|---|---|
| 15 MP photo → 1024 | 119 | 280 | 195 | 6 | 20 | 121 | 35 |
| 64 patches 448 → 224 | 106 | 205 | 157 | 39 | 71 | 169 | 157 (one batched call) |
| 64 crops 800 → 448 | 468 | 1,005 | 603 | 90 | 197 | 495 | 385 (one batched call) |

From a planar 3×H×W float32 picture (`d11_resize_speed_planar.txt`, synthetic, the same 15 MP), "F" took:
- 295–327 ms for the planes in turn;
- 182–219 ms on a 2-thread pool;
- 99–109 ms per plane.

A planar read-back (D9's numpy maths can write one) saves only the channel copies. (a) costs about 2× today's
8-bit resize. At the inference host that is roughly +0.1 s per photo at the SAM side, or less with the planes on
threads. In the head fit it is about +1.5 ms per view.

## 5. Augmentation on float32: albumentations 2.0.8, transform by transform

`d11_augment_float.py` ran each transform of `seg_ft.augment`, `head_augment.patch` and `head_augment.input` alone,
with groups expanded, at p = 1. OneOf's children ran one by one and OneOf as a whole. Each built by
`coffeecv.augment.build` (strict), at its pipeline's size: the photo at 1024 with a label, the 448 patch, the 224
input. Each ran on 24 seeds, on uint8, on uint8 / 255 and on the float read-back. One photo was used, a 4000×3000
OnePlus photo.

How albumentations treats dtypes (albucore 0.0.24):
- `MAX_VALUES_BY_DTYPE` is 255 for uint8 and 1.0 for float32 (`albucore/utils.py:27–41`).
- `@clipped` clips to that maximum (`utils.py:103–122`).
- `@uint8_io` (`albucore/functions.py:822–848`) converts a float input with `from_float`, which is
  `clip(np.rint(img * 255))` (`functions.py:658–679`), runs the uint8 function, and divides the result by 255.
- On uint8, the lookup tables are built in float and cast with `astype` (`clip`, `utils.py:103–108`;
  `create_lut_array`, `apply_lut`, `multiply_add_lut`, `functions.py:57–99, 446–453`), which truncates.

| Pipeline: transform | Float32 accepted | grid vs u8 max / mean signed (levels) | Inside on float | Source |
|---|---|---|---|---|
| seg: HorizontalFlip, VerticalFlip; head: SquareSymmetry | yes | 0 / 0 | index permutation | `geometric/flip.py:52, 206, 608` |
| seg, head: Affine | yes | 0.5 / 0.000 | `cv2.warpAffine` `INTER_LINEAR` (default) in float; fill 0 and reflect mean the same on float; label unchanged | `geometric/transforms.py:445`, `geometric/functional.py:554–590` |
| shared colour: ColorJitter (hue 0) | yes | 2.85 / **+0.96** | brightness: `multiply`; contrast: `multiply_add` around the grey mean; saturation: `cv2.addWeighted`; all clipped to [0, 1]. On uint8, the brightness and contrast LUTs truncate | `pixel/transforms.py:3989`; `pixel/functional.py:1697–1786`; albucore `functions.py:202, 466` |
| shared colour: PlanckianJitter | yes | 1.0 / **+0.31** | R and B multiplied by a constant, `@clipped`; the uint8 LUT truncates | `pixel/functional.py:2154–2215` |
| seg: **HueSaturationValue** | yes, **via uint8** | 0 / 0; **on-grid +0.996** | `shift_hsv` is `@uint8_io`: rounded to 8 bits, OpenCV's 8-bit HSV, back / 255. The class docstring's "for float32 images, the shift values are applied as percentages of the full range" (`pixel/transforms.py:1822`) describes no separate float path | `pixel/functional.py:55` |
| seg: RandomGamma | yes | 1.0 / **+0.50** | `np.power` on float; on uint8 a LUT cast with `astype(np.uint8)` (truncates) | `pixel/functional.py:1223–1240` |
| shared compression: **ImageCompression** | yes, **via uint8** | 0 / 0; **on-grid +0.996** | `image_compression` is `@uint8_io`: `cv2.imencode` and `imdecode` of the 8-bit picture | `pixel/functional.py:457–503` |
| seg: MotionBlur, Defocus; head: GaussianBlur, MotionBlur, Defocus | yes | 0.5 / ±0.002 | `cv2.filter2D` and `sepFilter2D`, `ddepth=-1`, `@clipped`: float in, float out | `pixel/functional.py:4052`; `blur/functional.py:122` |
| shared noise: GaussNoise | yes | 1.0 / +0.002 | noise scaled by `MAX_VALUES_BY_DTYPE` (std_range is relative), `add_noise` `@clipped` | `pixel/transforms.py:2629`; `pixel/functional.py:2218` |
| head input: **Downscale** (area down, linear up) | yes, **via uint8** | 0 / 0; **on-grid +0.997** | `downscale` is `@uint8_io`: two `cv2.resize` calls on the 8-bit picture | `pixel/functional.py:1568–1600` |
| head patch: **CoarseDropout** fill [124, 116, 104] | yes, **wrong values** | 31,496 / +1,180 | `fill_array = np.array(fill, dtype=img.dtype)`: the holes hold 124.0, 116.0, 104.0 on float32 | `dropout/functional.py:268, 305–322` |

Other observations:
- **Whole pipelines on float** (24 seeds each): dtype float32 out, range [0, 1], no error, no warning. The
  `head_augment.patch` range reads [0, 1] only because ColorJitter (p 0.8) and PlanckianJitter (p 0.3) clip the
  124.0 holes. A direct check found every hole pixel at 1.0 after ColorJitter: white, not the ImageNet-mean grey.
  This is how the CoarseDropout fault would pass unnoticed.
- **No refusal, no warning.** `augment.build` turns warnings into errors only while it builds (`augment.py`,
  `_transform`). No transform here warns or raises on float at call time either. R5's "a pipeline with one fails
  loudly instead of quietly converting" does not hold for this pipeline set. The quiet conversion is the behaviour.
- **Size of the 8-bit steps** (params.yaml). In `seg_ft.augment`, HueSaturationValue (p 0.3) and ImageCompression
  (p 0.3): a training photo goes through at least one with probability 1 − 0.7 × 0.7 = 0.51. In
  `head_augment.input`, ImageCompression (p 0.3) and Downscale (p 0.2): 1 − 0.7 × 0.8 = 0.44 of the head's views.
  `head_augment.patch` has none.
- **What the "on-grid" column measures** (`d11_augment_float.py`, `on_grid`): the share of the float output's
  values within 1e-3 level of a multiple of 1/255, minus the input's own share (0.39% for the photo, 0.73% for the
  patch, 0.32% for the input). +0.996 therefore means every output value is on the grid.

The four transforms that need care, with options (for the plan, not decided here):
- **ImageCompression.** A JPEG round trip is 8-bit by nature (`cv2.imencode`). The options:
  - keep it as the one 8-bit step D3 allows, since it simulates an 8-bit JPEG source;
  - apply only its error, `x + (jpeg(q8) − q8) / 255` with `q8 = rint(255x)`, so the float detail survives;
  - drop it.
  This needs the owner's reading of D3, since the ticket says the augmentation has no 8-bit step.
- **HueSaturationValue (segmenter only) and Downscale (classifier input).** Float equivalents exist:
  - for HSV, `cv2.cvtColor` on float32 gives H in degrees and S, V in [0, 1], so OpenCV's 8-bit units become
    hue ×2°, sat/255, val/255;
  - for Downscale, `cv2.resize` with `INTER_AREA` and `INTER_LINEAR` on float32.
  They are not albumentations 2.0.8 classes. `augment.build` builds only albumentations classes, so either the
  builder learns repo-defined transforms or these two stay 8-bit.
- **CoarseDropout.** In params.yaml, `fill: [0.486, 0.455, 0.408]` is the same grey on [0, 1] (124/255, 116/255,
  104/255). Any test should assert that a float pipeline's output stays in [0, 1] and off the 8-bit grid, the
  measure used here.

## 6. The segmenter's fine-tune: other uint8 assumptions

Read in `coffeecv/seg_finetune.py`:
- **Labels.** `label_at` resizes the bool mask in float32 (`INTER_AREA`, ≥ 0.5) to uint8 0/1. `pad_label` fills
  `IGNORE` = 255. Affine's `fill_mask: 255` acts on the label, which stays uint8 whatever the photo's dtype. In the
  test, the labels came out identical for uint8 and float photos on every seed, with values in {0, 1, 255}.
- **No embedding cache.** The module docstring: "nothing is cached, and nothing augmented is written" (ML-6 D12).
  ML-2's and ML-5's caches went with their stages on 2026-10-21 (`dvc.yaml` comment, lines 8–11). `Selection` holds
  the validation embeddings in memory, already float32 from `model_io.sam_input`.
- **The photos in memory.** `TrainSet` holds every training photo at the encoder's frame as `sam_resize` returns
  it, today uint8. pass17's labels have 711 accepted plus 380 negative training photos, 1,091 at about
  1024×768×3:
  - about 2.6 GB as uint8;
  - 5.1 GB as float16;
  - 10.3 GB as float32.
  The transforms above are float32-only in practice: `cv2.warpAffine` and most of OpenCV reject float16. So a
  float16 store would convert each sample to float32 as it is drawn. This is D10's question for the patches, and
  the same arithmetic applies here.
- **The resize and the normalisation** (`model_io.py`, which the fine-tune calls). `sam_resize` (line 68) calls
  `Image.fromarray(rgb)`, which raises on H×W×3 float, so it fails loudly. `_to_chw_normalised` (lines 36–42,
  called at line 77) divides by 255 on any dtype, so it would silently scale a [0, 1] picture to [0, 0.004] if a
  float picture reached it unchanged.

Elsewhere on the classifier's path, for the plan's caller list (R7):
- **The patch store** (`dataset.py:494–506`): `np.asarray(patch, dtype=np.uint8)` and the PIL "RGB" resize.
- **The unaugmented item** (`dataset.py:529–535`): PIL, then `/ 255`.
- **The head's augmented views** (`fit_frozen_head.HeadAugment.view`, lines 154–166; `embed_views`, line 226):
  `Image.fromarray`, and docstrings saying uint8.
- **`eval_resize` and `eval_patch`**, whose API is a PIL image.
- **Serving's patches** (`infer.py:568–573`).
- **The bean-pitch grey.** `dataset.py:456–457` and `infer.grayscale_like_training` (`infer.py:89–97`) end in
  `astype(np.uint8)`. On a [0, 1] float picture that gives zeros: a silent failure, outside D11.

## 7. Recommendation

**D11 (a): PIL's bilinear on mode "F", one plane at a time**, for `sam_resize`, `eval_resize`, the patch store and
serving's patches. The reasons:
- It is today's filter on more bits: `Resample.c`'s one `precompute_coeffs` serves both modes (§1). The models were
  trained, and EfficientViT-SAM's `SamResize` and DINOv3's eval transform were written, against Pillow's widened
  triangle.
- It is the closest of the three to today's input: ≤ 1.45 levels at any single resize and ≤ 2.02 over the
  classifier's two, mean 0.26–0.31. Compare cv2's 29–90 at the SAM side.
- It keeps `model_io` torch-free.
- It has an independent oracle in torch's antialiased bilinear (≤ 0.006 level). The parity tests can hold the float
  path to that at a tight tolerance (e.g. 0.01 level), as they hold today's to torchvision's.

Run the three planes on a small thread pool at the inference host, where the 2× cost matters most. Keep (b) out:
its speed does not pay for a different picture. Keep (c) for the tests.

**R5:** amend the risk's text. The pipelines do not fail on float. Three transforms round to 8 bits inside and one
fill is in 8-bit units. The plan should then do three things:
- change CoarseDropout's fill;
- decide ImageCompression, HueSaturationValue and Downscale (§5);
- add a test that runs each pipeline on float and checks the range and the on-grid share.

ColorJitter, PlanckianJitter and RandomGamma read 0.3–1 level brighter on float than on uint8, today's truncation
gone. The refit absorbs that like the resize's own sub-level shift. It is not a regression to chase.

**Confidence.** High on the filter identity: it rests on source and an exact parity measurement. High on the
transform-by-transform table: source plus measurement over 24 seeds. Medium on the speed figures: a loaded machine.

## 8. What stays unverified

- Every number here comes from today's canonical photo read back as float, not from D1's linear half-float photo
  through D2's read-back. The resizers are linear, so the conclusions carry over. The magnitudes against today's
  input are R1's to measure.
- What the differences do to the models: no embedding, mask or metric was computed. The claim is only that (a) moves
  the input least.
- Speed on the inference host's own CPU, and the thread-pool gain there.
- Behaviour of these transforms with arguments other than params.yaml's, and of albumentations versions other than
  2.0.8 (AlbumentationsX is not an option: ADR 0018's pin).
- Float16 storage feeding the transforms: assumed to need a float32 conversion per sample, from OpenCV's dtype
  support. Not run.

## Files

- `analysis/ml37/d11_resize_parity.py` → `d11_resize_parity.txt`: the impulse test and §3's table (13 photos,
  156 patches).
- `analysis/ml37/d11_resize_speed.py` → `d11_resize_speed.txt`, and `d11_resize_speed_planar.py` →
  `d11_resize_speed_planar.txt`: §4.
- `analysis/ml37/d11_augment_float.py` → `d11_augment_float.txt`: §5 and §6's label check.
- Sources read, fetched into `tmp/ml37/d11/downloads/` (not committed):
  - Pillow `src/libImaging/Resample.c`, `src/libImaging/Reduce.c`, `src/PIL/Image.py` at
    `https://raw.githubusercontent.com/python-pillow/Pillow/13.0.0/…`;
  - OpenCV `modules/imgproc/src/resize.cpp` at `https://raw.githubusercontent.com/opencv/opencv/5.0.0/…`.
  - albumentations 2.0.8, albucore 0.0.24, torch 2.14.0 and torchvision 0.29.1 were read as installed in the
    project's environment.
