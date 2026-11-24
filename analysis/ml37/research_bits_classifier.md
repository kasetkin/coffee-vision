# ML-37: how many bits the classifier's input needs (8, 10, 12 bits or float16)

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) (D3, D10, D13, R9), 2026-11-23, on the
local checkout at `cc4cfba` (branch `ml37-linear-canonical`). A recommendation for the owner, not a decision. The
owner's question: "How many bits is useful for model input? Compare sRGB 8-bit, 10-bit, 12-bit and 16-bit float. We
need to compare if increasing number of bits gives us any quality boost." This note answers it for the **classifier
only**: the frozen DINOv3 ViT-B/16 (`dinov3_vitb16_pretrain_lvd1689m`, `cls_mean` readout) and its L2
logistic-regression head at C = 0.1 (ADR 0006, ADR 0025). The segmenter is a separate study. Nothing was trained as a
recorded run; nothing under `coffeecv/`, `data/`, `models/`, `experiments/`, `params.yaml` or the DVC files was
changed. Every number comes from the `bits_cls_*` scripts under [Files](#files), run on this machine (12 CPUs, at most
8 threads, no GPU; torch on CPU in float32), with each script's printed output beside it.

## Summary

- **No: going above 8 bits gives the classifier no measurable quality gain**, provided the 8-bit step sits where it
  sits today (the stored crop) and everything after it (resizes, normalisation) is float, as D3 and D11 already
  make it. On the live head's 1,600 test patches (266 photos, 10 classes), the 8-, 10-, 12-bit and float16 arms
  change **no** patch's answer and **no** photo's answer against the float32 picture, and give the same test
  macro-F1 (0.9554) and the same F1 in every class (§4). Refitted on paired seeds 42/123/7 (same split and boxes,
  only the crop's bit depth differs; unaugmented heads), 8-bit against float16 changes test macro-F1 by −0.0007,
  +0.0000 and −0.0006: mean **−0.0004, photo-bootstrap 95% CI [−0.0015, +0.0006]**, inside D13's proposed bar
  (lower bound above −0.010). No class's mean paired F1 change is below −0.0024 (Colombia; bar −0.02). Both arms' test macro-F1 spans
  the same seed range, 0.929–0.951 (f16) and 0.928–0.950 (u8). Per seed, 1 to 3 of 1,600 test patches change answer,
  about ten times fewer than D13's near-null pair (§5).
- **From 10 bits on, the input is as close to float as serving's own fp16 cast; from 12 bits (and float16) it is
  at float32's arithmetic floor.** DINOv3's `cls_mean` feature moves by a median 1 − cosine of 2.2e-8 at 10 bits
  (p95 2.3e-7), 1.5e-9 at 12 bits and 3.8e-9 at float16, against 3.8e-8 (p95 1.4e-7) for the fp16 cast an fp16
  serving graph applies to every input (ADR 0019). 8 bits is measurable (median 3.5e-7, max 3.2e-4) but stays
  below every change that does not matter (§3).
- **What 8 bits costs, next to what is already there** (1 − cosine of the feature, median, 62 patches of 12 photos):
  8-bit 1.6e-7; a second q95 JPEG XL generation of the canonical photo 4.7e-6; removing the store's own lossy coding
  1.1e-5; the front's q90 e3 canonical photo of the same original 3.6e-5; moving the patch by one pixel 5.8e-4; another
  patch of the same photo 4.9e-2 (median over 1,600 patches, §3).
- **At the pixel, the 8-bit step is below the coding error but not below the fine-scale noise.** On 12 photos (12
  sessions) the store's q95 e7 coding error is 1.03 levels RMS (0.90–2.09), the front's q90 e3 1.50 (1.30–3.18);
  8-bit rounding is 0.287 (adding about 8% to the store's error variance), 10-bit 0.072, 12-bit 0.018, float16 0.028.
  But phone photos are smooth at the pixel: in the flattest tenth of bean blocks Immerkær's noise σ is 0.15 levels
  (median; 0.05–6.3), below 8-bit's step noise in 10 of 12 photos, so the noise does not dither 8-bit rounding there
  (§2). The downscale to 224 px is what removes it: at the model input 8-bit's error is 0.033 levels RMS (median),
  the store's lossy coding 0.20, the front's 0.35 (§2, §3).
- **The place of the 8-bit step matters more than the number of bits.** Today's path rounds three times (the crop, the
  448 px patch, the 224 px input). It moves the feature 130× more than one 8-bit rounding at the crop (median 4.5e-5
  against 3.5e-7), and is the only arm besides the one-pixel shift that changes answers: 4 of 1,600 patches and 2 of
  266 photos under the live head, test macro-F1 0.9559 against 0.9554 (§3, §4).
- **The effect grows as the patch's box shrinks**, since a box under 448 px is upsampled into the 448 px store and its
  rounding is not averaged away: 8-bit's median 1 − cosine is 1.9e-7 on boxes over 1,000 px and 5.5e-5 on the 7 boxes
  under 300 px (mostly a low-resolution phone photo and a messenger photo), whose largest top-1 probability change
  is 0.027 (§3).
- **Recommendation.** For the classifier, D3's "no 8-bit step" brings no measurable accuracy; it is a correctness and
  parity choice, not a quality one. Float16 patches and crops (D10 (b)) are more than enough: anything from 10 bits
  up is equivalent. If R8 (disk) or R9 (memory) binds, a 10- or 12-bit integer store would lose nothing measurable,
  and even 8-bit crops with float resizes lose nothing measurable for the classifier; but that would reverse the
  owner's D3, which this note does not propose. Whatever is chosen, the resizes after the stored crop should stay
  float: today's later roundings are the largest bit-depth effect measured (§6).

## 1. The arms, and where the quantization sits

**The float picture** is D10's stand-in for ML-37's read-back (`research_d10_crop_and_patch_storage.md` §1): today's
canonical photo decoded at 16 bits (`canonical.decode_canonical`, uint16 Rec.2020, BT.709 curve), the crop's box from
its `segcrop_report.json` cut out, read back to sRGB by the transform `load_canonical` uses (LittleCMS, relative,
`NOOPTIMIZE`) with float32 output, clipped to [0, 1], and filled outside its `__beanmask.png` with `fill_rgb` / 255
(`bits_cls_common.float_crop`). Rounded to 8 bits it is the stored crop in at least 0.999993 of the samples
(`bits_cls_check_boxes.txt`). It is today's canonical photo, not D1's half-float one.

**The arms** (`bits_cls_common.py`), each applied to that float crop:

| arm | the crop's samples | then |
|---|---|---|
| f32 | float32 as read back (the reference) | float |
| f16 | rounded to float16: D10 (b)'s half-float crop, the owner's "16-bit float" | float |
| u12 | round(x · 4095) / 4095 | float |
| u10 | round(x · 1023) / 1023 | float |
| u8 | round(x · 255) / 255 | float |
| today | uint8, then PIL's uint8 bilinear to 448 (rounded), then `model_io.eval_patch` (uint8 bilinear to 224, rounded, / 255) | today's code |

"Then float" is: the box cut, PIL's bilinear in "F" mode channel by channel to `patch_store_size` 448 and then to
`patch_resize` 224 (D11 (a)), and the ImageNet normalisation, all float32. So only the bit depth of the crop differs
between f32, f16, u12, u10 and u8.

**Why quantize at the crop.** That is where today's 8-bit step is first taken and kept: `load_canonical` returns uint8
and the crops are stored as lossless 8-bit JPEG XL (ML-19 D32, `crop_io.write_crop`), and it is what D10 decides for
the new path. Under D3 the resizes, the patches and the augmentation are float, so after the crop nothing else rounds.
Quantization is pointwise, so it is applied to each box's pixels, which equals quantizing the crop and then cutting.
The `today` arm is there to show the other placement: today rounds again after each of the two resizes.

**The boxes are the fit's.** `Geometry.boxes` draws a photo's boxes with `_extract_photo`'s rule (bean-unit sizing,
the D17 mask rule, the RNG key `[seed, class, photo, split]`), measuring the pitch on round(x · 255). On 8 photos of
exp290's test split the pitch and the boxes equal those `MultiPhotoPatchDataset._extract_photo` draws from the stored
crop, 8 of 8 (`bits_cls_check_boxes.txt`). The `today` arm then reproduces exp290's own recorded test predictions:
1,600 of 1,600 answers equal, largest probability difference 1.2e-3 (`bits_cls_predict.txt`, "check"). The residual is
the few crop samples where the float read-back rounds differently from the stored crop (the today arm's input differs
from the stored crop's own patches by at most 1.00 level, `bits_cls_check_boxes.txt`), and the thread count.

## 2. The noise already in the picture (level 1)

`bits_cls_noise.py` on 12 photos of exp290's test split, one per session from 12 sessions (crops at most 16 MP, so
the Pixel, OnePlus, iPhone, box-picture and new-green sessions; no Sony or Canon session). All figures are 8-bit levels (1/255 of [0, 1]
sRGB) on the crop's bean pixels, three channels pooled; medians over the photos (min–max).

**The lossy coding.** The original is decoded and converted exactly as the store's `canonical.encode` does
(`decode`, `Source.peak`, `to_rec2020` at `STORE_SETTINGS`), giving the uint16 picture libjxl was handed. Encoded again
at q95 e7, it gives the stored canonical photo's bytes exactly, on 3 of 3 photos checked ("bytes==stored"), so the
difference between the two read-backs is the store's coding error itself.

| | RMS | note |
|---|---|---|
| store's coding error, q95 e7 (ADR 0036's store setting) | 1.030 (0.897–2.093) | mean −0.005 to +0.009; p99 2.6–6.3; max 11–40 |
| front's canonical photo, q90 e3, integer colour path (ADR 0036), against the same original | 1.497 (1.295–3.177) | what a served upload is made of |
| a second q95 e7 generation of the stored photo | 0.284 (0.211–0.692) | |
| **8-bit rounding (u8)** | **0.287** (0.281–0.288) | uniform: 1/√12 = 0.289 |
| 10-bit rounding (u10) | 0.072 | 1/(4√12) |
| 12-bit rounding (u12) | 0.018 | |
| float16 rounding (f16) | 0.028 (0.026–0.034) | binary16 has 11 significant bits: a step of 2⁻¹¹ in [0.5, 1), like 11-bit integers there, finer below |

Rounding errors add in variance: 8-bit adds 0.287² / 1.030² ≈ 8% to the store's coding error variance and 3.7% to the
front's. 10 bits adds 0.5%.

**The fine-scale noise.** Immerkær's estimator (a 3×3 Laplacian-difference mask, σ = √(π/2) · Σ|I ∗ N| / (6 (W−2)(H−2)),
Immerkær 1996) per 16×16 block lying wholly on beans, per channel:

- flattest tenth of the blocks: **0.15 levels** (0.05–6.27);
- median block: 0.32 (0.12–8.78);
- D10's robust second-difference σ (green): 0.90 (0.46–8.56).

So the decoded phone photos are smoother at the pixel than 8-bit's step noise wherever the beans are flat: the 10th
percentile block is under 0.287 in 10 of the 12 photos, and the median block in 5 of 12. The coding error (1.03 RMS)
is far larger than that fine-scale σ, so it is mostly coarser than a pixel; that is an inference from the two numbers,
not a measurement of its spectrum. The phones' own denoising and q95's removal of fine detail are the plausible
causes; neither was measured. The answer to "is the 8-bit step far below the existing noise?" is therefore: **below the
coding error (3.6× in RMS), not below the pixel noise in flat areas.** In such areas 8-bit rounding is not dithered,
so it can band.

**At the model input** it is averaged away. 1,561 of the 1,600 boxes of §3 are at least 448 px, so the 224 px input
averages several crop pixels into each of its own, and independent per-pixel errors shrink with the box.
`bits_cls_embed.py` and `bits_cls_noise.py` measure the input's RMS difference from the f32 arm's
(normalised difference × std × 255; median over photos):

| change | input RMS (8-bit levels) |
|---|---|
| u12 | 0.002 |
| f16 | 0.003 |
| u10 | 0.009 |
| fp16 cast of the normalised input | 0.011 |
| **u8** | **0.033** |
| a second q95 generation (62 patches) | 0.119 |
| without the store's lossy coding (62 patches) | 0.202 |
| the front's q90 e3 photo (62 patches) | 0.352 |
| today's three roundings | 0.368 |
| the box moved by one crop pixel | 4.40 |

For scale, the training recipe already teaches the head invariance to far larger perturbations: `head_augment` draws
`GaussNoise` with σ 0.01–0.04 of the range (albumentations 2.0.8: "a fraction of the maximum value", 2.6–10.2 levels),
`ImageCompression` at quality 60–95 and `Downscale` (`params.yaml`, ADR 0025, ADR 0027). This is context, not a
measurement.

## 3. What DINOv3's feature sees (level 2)

`bits_cls_embed.py`: exp290's test split, the live head's (`models/dino3b16_segxl0v7r_country_aug8_ml29gm_s123.pt`,
seed 123, `webapp/app.py`'s default checkpoint): 266 photos the head never trained on, 160 patches a class, 1,600 in
all, from at least 18 sessions and six camera groups (Pixel 607 patches, Sony 356, OnePlus 327, iPhone 179, Canon 10,
other 121). Each box embedded in every arm and yardstick, `cls_mean` (1,536-d), float32.

1 − cosine and relative L2 against the f32 arm's feature (`bits_cls_predict.txt`):

| form | 1 − cos median | p95 | max | rel. L2 median | max |
|---|---|---|---|---|---|
| u12 | 1.5e-9 | 1.6e-8 | 1.9e-6 | 5.5e-5 | 2.0e-3 |
| f16 | 3.8e-9 | 3.3e-8 | 7.9e-7 | 8.7e-5 | 1.3e-3 |
| u10 | 2.2e-8 | 2.3e-7 | 1.4e-5 | 2.1e-4 | 5.4e-3 |
| *yardstick:* fp16 cast of the input (ADR 0019's serving graph) | 3.8e-8 | 1.4e-7 | 8.9e-7 | 2.7e-4 | 1.3e-3 |
| **u8** | **3.5e-7** | 3.4e-6 | 3.2e-4 | 8.3e-4 | 2.5e-2 |
| today | 4.5e-5 | 1.8e-4 | 2.4e-3 | 9.5e-3 | 6.9e-2 |
| *yardstick:* the box moved one crop pixel | 6.6e-4 | 2.7e-3 | 1.0e-2 | 3.6e-2 | 1.4e-1 |
| *yardstick:* the next patch of the same photo | 4.9e-2 | | min 6.0e-3 | | |

On `bits_cls_noise.py`'s 12 photos (62 patches), against the stored photo's f32 arm, median / max:

| form | 1 − cos median | max |
|---|---|---|
| f16 | 1.8e-9 | 4.9e-7 |
| u8 | 1.6e-7 | 5.0e-6 |
| a second q95 e7 generation | 4.7e-6 | 8.4e-5 |
| the original without the store's lossy coding | 1.1e-5 | 2.0e-4 |
| today | 2.4e-5 | 3.8e-4 |
| the front's q90 e3 canonical photo | 3.6e-5 | 1.0e-3 |
| the box moved one pixel | 5.8e-4 | 3.5e-3 |

- **u12 and f16 are at the arithmetic floor.** Their medians are of order 1e-9, a hundredth of experiments.md's
  cross-CPU limit (1 − cos 2e-7), which is the size of float32 rounding differences between machines.
- **u10 is level with the fp16 serving cast:** below it at the median (2.2e-8 against 3.8e-8), above it at p95
  (2.3e-7 against 1.4e-7) and at the maximum (1.4e-5 against 8.9e-7, a box under 300 px). The served model already
  takes a perturbation of that size on every input.
- **u8 is about 95× further from float than f16** (median of the per-patch ratio) and 9× the fp16 cast, but, on the
  62 patches of §2's photos, 30× below a second generation of the canonical photo's coding and 70× below the lossy
  coding the store already applies; on all 1,600, 1,900× below a one-pixel shift.
- **Per class** the u8 median is 1.0e-7 (Indonesia) to 9.9e-7 (Brazil); **per camera** 5.9e-8 (Canon) to 9.2e-7
  (other). No class or camera stands out.
- **By box size** (crop pixels; u8 median / max, u10 and f16 alike):

| box side | patches | u8 | u10 | f16 |
|---|---|---|---|---|
| < 300 | 7 | 5.5e-5 / 3.2e-4 | 1.7e-6 / 1.4e-5 | 9.1e-8 / 7.9e-7 |
| 300–448 | 32 | 4.3e-6 / 2.6e-5 | 2.6e-7 / 1.4e-6 | 4.0e-8 / 1.8e-7 |
| 448–700 | 224 | 2.0e-6 / 1.1e-5 | 1.2e-7 / 7.2e-7 | 1.5e-8 / 1.3e-7 |
| 700–1,000 | 309 | 7.9e-7 / 4.6e-6 | 4.9e-8 / 2.6e-6 | 8.7e-9 / 8.6e-8 |
| ≥ 1,000 | 1,028 | 1.9e-7 / 1.8e-6 | 1.1e-8 / 3.2e-7 | 2.1e-9 / 4.9e-7 |

  The three largest u8 changes are the three patches of one low-resolution phone photo
  (`20261105_135107_G8441_Brazil_decaff`, a 1,138 × 1,158 crop, boxes of 239–298 px), the fourth a messenger photo
  (`20261024_151537_Telegram_photo_9`, 475 × 445, boxes of 106–180 px). A box under 448 px is upsampled into the store, so its rounding reaches the model nearly unaveraged.
  Even there 8-bit's largest change (3.2e-4) is half the median one-pixel shift.
- **This sharpens D10's number.** D10's note measured uint8 at 3.4e-6 median, 2.2e-5 max on 24 patches. It rounded
  the 448 px patch, after the first downscale (`d10_embed_precision.py`). Rounding the crop instead, before it, gives
  3.5e-7 on these 1,600: ten times less, as expected if both resizes average the rounding (other patches, so the
  ratio is indicative).

## 4. The live head's answers (level 3)

The live head applied to each arm's features as `fit_frozen_head` scores a split (`linear_head.predict`). A photo's
answer is the argmax of the mean of its patch probabilities, as serving averages its patches. Against the f32 arm
(`bits_cls_predict.txt`):

| form | patches with another answer (of 1,600) | top-1 probability change: median, p99, max | photos with another answer (of 266) | test macro-F1 |
|---|---|---|---|---|
| f32 | — | — | — | 0.9554 |
| f16 | 0 | 2.4e-6, 4.4e-4, 2.0e-3 | 0 | 0.9554 |
| u12 | 0 | 1.4e-6, 3.5e-4, 4.6e-3 | 0 | 0.9554 |
| u10 | 0 | 5.9e-6, 1.4e-3, 8.4e-3 | 0 | 0.9554 |
| **u8** | **0** | 2.4e-5, 5.5e-3, 2.7e-2 | **0** | **0.9554** |
| fp16 cast | 0 | 7.2e-6, 1.0e-3, 4.5e-3 | 0 | 0.9554 |
| today | 4 | 2.6e-4, 3.9e-2, 2.0e-1 | 2 | 0.9559 (exp290's recorded: 0.9559) |
| one-pixel shift | 14 | 9.3e-4, 1.7e-1, 3.5e-1 | 1 | 0.9534 |

Per-class test F1 is identical in f32, f16, u12, u10 and u8 in all ten classes: Ethiopia 0.8869, Kenya 0.9632,
Colombia 0.9148, Costa Rica 0.9937, Guatemala 0.8609, Brazil 0.9623, Vietnam 1.0000, Indonesia 0.9877, Peru 0.9875,
Rwanda 0.9969. For scale, ADR 0019 records that the fp16 DINOv3 graph serving on `yappy` "changes 0.48% of top-1
answers" over 157 photos, and patch sampling 0.85%. A head fitted on today's 8-bit path gives the same answers on the
float picture as on its own input on all but 4 patches; the bit depth of the input is not what it is sensitive to.

## 5. Refitted heads, three seeds (level 4)

`bits_cls_refit.py` and `bits_cls_refit_eval.py`. Not `fit_frozen_head` and not a recorded run: no `experiments/`
entry, nothing under `data/` or `models/` written. Two arms, **u8 and f16** (the owner's two ends), on seeds 42, 123
and 7. At each seed both arms have the same photo split (`split_photos_by_class` over the 1,805 photos
`pooled_class_photos` lists in `data/segcropped/pool`), the same per-class budgets spread over the photos as
`MultiPhotoPatchDataset` spreads them (600 training and 160 test patches a class), and the same boxes
(`Geometry.boxes`, checked in §1). Only the crop's bit depth differs. Each head is fitted as `fit_frozen_head` fits
its shipping head, `linear_head.fit_head_at` at the fixed C = 0.1 (`fit_frozen_head.DEFAULT_C`), and scored on the
seed's test patches. The arms are compared seed by seed with `research_d13_adoption_bar.md`'s proposed bar, using its
bootstrap (test photos resampled within class, the same draw for both arms, 4,000 draws a seed).

What was reduced, and why:
- **No augmented views.** ADR 0025's recipe adds 8 augmented views a training patch (48,000 more embeddings a seed
  and arm, 288,000 for three seeds and two arms). At 0.16–0.19 s an embedding on 8 CPU threads, that is about 14
  hours.
  The heads here are the recipe's unaugmented arm, which every run fits and archives beside the augmented one
  (`model_unaugmented.pt`, `metrics.json` `arms`). The views' noise and compression are far coarser than any
  rounding here (§2), so they could only bring the arms closer; that is not measured.
- **No val split.** C is fixed, so val chooses nothing.
- **No u10 and u12 refits.** The two arms took about 2.3 hours; a third would not have finished by the stop. At
  levels 2 and 3, u12 is level with f16 and u10 sits between f16 and u8, much closer to f16 (§3, §4).

Results (`bits_cls_refit_eval.txt`). At each seed: 6,000 training patches of 1,265 photos, 1,600 test patches
of 273 photos. Every fit converged.

| seed | test macro-F1, f16 | u8 | paired change | test patches with another answer | photos with another answer (right, f16 → u8) |
|---|---|---|---|---|---|
| 42 | 0.9505 | 0.9498 | −0.0007 | 3 / 1,600 | 1 / 273 (266 → 267) |
| 123 | 0.9459 | 0.9459 | +0.0000 | 2 / 1,600 | 0 / 273 (265 → 265) |
| 7 | 0.9291 | 0.9284 | −0.0006 | 1 / 1,600 | 1 / 273 (262 → 263) |
| range over seeds | 0.9291–0.9505 (0.0214) | 0.9284–0.9498 (0.0214) | | | |

- **D13's overall bar:** mean paired change −0.0004, photo-bootstrap SE 0.0005, 95% CI [−0.0015, +0.0006]. The lower
  bound is far above −0.010: **met**. The CI includes 0 and its upper end is +0.0006, so float16 shows no gain over
  8 bits that this test could resolve. At photo level the u8 heads get one more photo right at two seeds.
- **D13's per-class bar** (mean paired F1 change, flag below −0.02), with the bootstrap 95% CI: Ethiopia −0.0020
  [−0.0052, 0], Kenya +0.0010 [0, +0.0029], Colombia −0.0024 [−0.0061, 0], Costa Rica +0.0001 [−0.0028, +0.0034],
  Guatemala −0.0009 [−0.0046, +0.0029], Brazil, Vietnam, Indonesia, Peru and Rwanda 0.0000. Lowest −0.0024: **met**.
- **Per-class F1, min–max over the seeds**, f16 / u8: Ethiopia 0.876–0.902 / 0.876–0.899, Kenya 0.938–0.966 /
  0.941–0.966, Colombia 0.828–0.891 / 0.828–0.887, Costa Rica 0.933–0.988 / 0.930–0.988, Guatemala 0.863–0.929 /
  0.867–0.927, Brazil 0.938–0.966 (both), Vietnam 1.000 (both), Indonesia 0.962–0.976 (both), Peru 0.906–0.981 (both),
  Rwanda 0.958–0.994 (both). In every class the two arms' seed ranges nearly coincide.
- **For scale:** D13's near-null pair (ML-29 P10, gain maps) disagreed on 1.7–2.2% of test patches with a paired
  mean of −0.0009, CI [−0.0045, +0.0025] (`research_d13_adoption_bar.md` §3). Here the arms disagree on 0.06–0.19%,
  and the CI is a third as wide.
- **The heads themselves** differ by 0.9–1.2% in relative weight norm, more than the features (relative L2 8e-4 at
  the median, §3). That figure includes the lbfgs stopping tolerance (scikit-learn's default `tol`, 1e-4), which was
  not separated out. Per test patch, the largest change of any class probability has a median of 6e-5 to 1e-4 at the
  three seeds, and a maximum of 0.028 to 0.065.

## 6. The answer

- **Does going above 8 bits give the classifier a measurable gain?** No, at every level measured. With the 8-bit step
  at the crop and float after it: the feature moves less than any change that does not matter, the live head's
  answers do not move at all, and the refitted heads meet D13's proposed bar with a mean paired change of −0.0004
  [−0.0015, +0.0006] over three seeds, no class below −0.0024.
- **From which bit depth is it indistinguishable from float?** 10 bits: level with the fp16 serving cast in the
  feature (median below it, p95 above), no answer changed. 12 bits and float16 are at float32's own arithmetic
  floor; float16 is a little coarser than 12 bits in the highlights (rounding 0.028 against 0.018 levels RMS) and finer in the shadows. 8 bits is measurable in
  the feature (about 10× the fp16 cast at the median, up to 3e-4 on boxes under 300 px), and nowhere else.
- **What this means for the ticket.**
  - D3 (no 8-bit step) stays the owner's decision; this note gives it no accuracy argument. Its cost is R9's memory
    and R8's disk, against no measurable gain for the classifier.
  - D10 (b), float16 crops and patches, is more precision than the classifier can use; nothing finer is worth paying
    for. A 10- or 12-bit integer store would also lose nothing measurable, should R8 or R9 force a smaller form;
    the size of such crops was not measured.
  - The bigger effect is the placement: whichever form the crop takes, no rounding should follow the resizes
    (today's later roundings move the feature 130× more than the crop's).
  - D13's bar is not threatened by the bit depth; a retrain under ML-37 that fails it would have another cause.

## 7. What this does not measure

- **D1's photo and D2's read-back.** All arms start from today's canonical photo (uint16, BT.709 curve, q95) read back
  by LittleCMS. The half-float linear photo's coding noise may differ (R2), which shifts §2's noise floor, not the
  arms' own rounding.
- **The augmented head.** Level 4 fits the unaugmented head (no views drawn); §5 says why. ADR 0025's views add
  noise and JPEG compression far above every rounding here (§2), which would make the arms closer still; that is an
  argument, not a measurement.
- **The 10- and 12-bit arms at level 4.** Two arms took 2.3 hours of the budget. Both are bracketed by f16 and u8
  at levels 2 and 3, much closer to f16.
- **The noise sample.** 12 photos from 12 phone sessions, crops at most 16 MP; no Sony or Canon session. Those
  cameras' features were measured at levels 2 and 3.
- **The segmenter**, the OOD probe and the serving graph in fp16 on `yappy` (only its input cast was simulated).
- **Spectrum of the coding error.** "Coarser than a pixel" is inferred from RMS against Immerkær σ.

## Files

Scripts and their printed outputs, in `analysis/ml37/` (run with `PYTHONPATH=/workspace uv run python`):

- `bits_cls_common.py`: the float crop, the arms, the model input, the box draw (shared; not a script).
- `bits_cls_check_boxes.py` → `bits_cls_check_boxes.txt`: the box draw against `_extract_photo`, and timings.
- `bits_cls_noise.py` → `bits_cls_noise.txt`: level 1 on 12 photos, and the coding yardsticks' features.
- `bits_cls_embed.py` (log `tmp/ml37/bits_cls/embed.log`) and `bits_cls_predict.py` → `bits_cls_predict.txt`: levels 2
  and 3 on exp290's 1,600 test patches.
- `bits_cls_refit.py` (logs `tmp/ml37/bits_cls/refit_*.log`) and `bits_cls_refit_eval.py` → `bits_cls_refit_eval.txt`:
  level 4.

Features are cached in `tmp/ml37/bits_cls/` (gitignored, local only): `embed/`, `noise/`, `refit/<arms>/`.
Peak memory: the largest process peak was 3.3 GB (`bits_cls_noise.py`, `ru_maxrss`); `bits_cls_refit.py` 3.1 GB, `bits_cls_embed.py` 2.4 GB, `bits_cls_refit_eval.py` 1.1 GB.

## Sources

- The repo at `cc4cfba`:
  - code: `coffeecv/canonical.py` (`decode_canonical`, `rec2020_to_srgb`, `load_canonical`, `to_rec2020`, `encode`,
    `STORE_SETTINGS`, `FRONT_SETTINGS`, `canonicalize_with`), `coffeecv/jxl_encode.py` (`encode_rec2020`),
    `coffeecv/crop_io.py`, `coffeecv/dataset.py` (`MultiPhotoPatchDataset._extract_photo`, `split_photos_by_class`,
    `pooled_class_photos`), `coffeecv/geometry.py` (`sample_masked_bean_unit_boxes`), `coffeecv/model_io.py`
    (`eval_patch`, `eval_resize`), `coffeecv/infer.py` (`grayscale_like_training`), `coffeecv/backbones.py`,
    `coffeecv/linear_head.py` (`fit_head_at`, `predict`), `coffeecv/fit_frozen_head.py` (`DEFAULT_C`, the recipe),
    `coffeecv/metrics.py`, `webapp/app.py` (the default checkpoint);
  - configuration and record: `params.yaml` (patch geometry, `head_augment`, `augment`),
    `experiments/exp290__dino3b16_frozen_clsmean_segcrop_country_aug8_s123_ml29gm/` (`predictions_test.csv`,
    `metrics.json`);
  - docs: `docs/ml-0037_ticket_linear_canonical_photo.html` (§1, §2, D2, D3, D10, D13, R8, R9),
    `docs/adr/0006-frozen-dinov3-b16-head.md`, `docs/adr/0019-serving-runs-litert-accelerator-per-host.md` (fp16
    DINOv3, 0.48% of answers), `docs/adr/0025-frozen-head-fits-augment-per-run-in-memory.md`,
    `docs/adr/0036-the-front-makes-its-own-canonical-photo.md` (q95 e7 against q90 e3), `docs/agents/experiments.md`
    (cross-CPU limit, reporting rules);
  - analysis: `research_d10_crop_and_patch_storage.md` (§1, §5; `d10_embed_precision.py`),
    `research_d11_float_resize_and_augment.md`, `research_d13_adoption_bar.md` (the bar, the bootstrap of
    `d13_seed_range.py` section C).
- albumentations 2.0.8 as installed, `augmentations/pixel/transforms.py`, `GaussNoise` docstring ("std_range … as a
  fraction of the maximum value (255 for uint8 images or 1.0 for float images)").
- J. Immerkær, "Fast Noise Variance Estimation", *Computer Vision and Image Understanding* 64(2):300–302, 1996 (the
  mask N and the σ formula used in `bits_cls_noise.immerkaer_blocks`).
- IEEE 754-2008 binary16: 1 sign, 5 exponent and 10 stored significand bits (11 significant), as D10's note measured
  (15,361 float16 codes in [0, 1]).
