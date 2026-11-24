# ML-37: how many bits the segmenter's input needs (8, 10, 12 bits or float16)

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) (D2, D3, D11, D13, D14, R9), 2026-11-23/24,
on the local checkout at `cc4cfba` (branch `ml37-linear-canonical`). A recommendation for the owner, not a decision.
The owner's question: "How many bits is useful for model input? Compare sRGB 8-bit, 10-bit, 12-bit and 16-bit float.
We need to compare if increasing number of bits gives us any quality boost." This note answers it for the
**segmenter only**: the live EfficientViT-SAM XL0 encoder with the fine-tuned decoder `xl0_v7r_s123` (`params.yaml`
`seg_*`). The classifier is [research_bits_classifier.md](research_bits_classifier.md); this note mirrors its arms,
levels and style so the two compare directly. Nothing was trained or fine-tuned; nothing under `coffeecv/`, `data/`,
`models/`, `experiments/`, `labels/`, `params.yaml` or the DVC files was changed, and no DVC stage was run. Every
number comes from the `bits_seg_*` scripts under [Files](#files), run on this machine (12 CPUs, 8 threads, no GPU;
torch on CPU in float32), with the printed output beside them.

## Summary

- **No: going above 8 bits gives the segmenter no measurable quality gain**, provided the 8-bit step sits where it
  sits today (the whole photo, `load_canonical`'s output) and the resize to the encoder's 1,024 px after it is float,
  as D3 and D11 make it. Against the owner's pass17 labels (validation, 177 positives and 88 negatives,
  `seg_finetune.evaluate`'s measures), mean IoU is 0.96990 in float32, float16 and 12 bits, 0.96989 at 10 bits and
  0.96992 at 8 bits. The paired change 8-bit − float32 is **+0.000016, photo-bootstrap 95% CI [−0.000007,
  +0.000051]**; every arm's CI contains 0. All 88 negatives stay empty or tiny in every arm (§5).
- **From 10 bits on, the input is as close to float as an fp16 cast of it; from 12 bits (and float16) it is at
  float32's arithmetic floor.** The encoder's embedding (256 × 64 × 64) moves by a median 1 − cosine of 7.9e-7 at 8
  bits, 3.5e-8 at 10, 2.0e-9 at 12 and 3.7e-9 at float16 (625 photos), against 3.2e-8 for an fp16 cast of the input
  (§3). These are the classifier's numbers almost exactly ([research_bits_classifier.md](research_bits_classifier.md)).
- **What 8 bits costs, next to what is already there** (1 − cosine of the embedding, median, 625 photos): 8-bit
  7.9e-7; a second q95 e7 JPEG XL generation of the canonical photo 8.6e-6; moving the photo by one pixel 3.8e-4.
  8 bits is a tenth of a second generation and 1/500 of a one-pixel shift (per-photo medians, §3).
- **The masks barely move, and never change the crop decision.** On 360 test photos, served masks against the
  float32 arm's: median IoU 0.99999 at 8 bits (lowest 0.998), 1.00000 from 10 bits up (lowest 0.9996). No mask became
  empty or crossed the fallback threshold in any arm, on either set. The crop box moved on 9 of 237 test positives at
  8 bits (at most 4 px), on none at 10 or 12 bits and on one at float16 (1 px) (§4). The masks that move are the
  segmenter's least confident ones, and they move more under a second JPEG XL generation.
- **The place of the 8-bit step matters more than the number of bits, as for the classifier.** Today's path rounds
  twice (the photo, then the 1,024 px input). It moves the embedding 30× more than one 8-bit rounding of the photo
  (per-photo median), its masks most (test: lowest served IoU 0.990, boxes moved on 30 of 237 positives; validation:
  one box by 114 px), and still scores the same against the labels (−0.0000045 [−0.00014, +0.00017]) (§3–§5).
- **Recommendation.** For the segmenter, D3's "no 8-bit step" brings no measurable accuracy; it is a correctness and
  parity choice, as for the classifier. Anything from 10 bits up is equivalent to float; float16 is more than enough.
  The resize after the photo should stay float (D11 (a)): today's second rounding is the largest bit-depth effect
  measured. A future fine-tune (ML-33) could hold its training photos in float16 rather than float32 at no measurable
  cost (R9); this is an inference, not a measurement of a fine-tune.
- **Reduced:** 360 of the 472 test photos within the time budget (seeded random order, so an unbiased subset); the
  test split has no labels, so level 3 is on validation only. The live encoder is XL0, not the XL1 the brief named
  (§1).

## 1. The arms, and where the quantization sits

**The segmenter.** The live one, as `params.yaml` names it and `segment_beans.seg_params` builds it: the
EfficientViT-SAM **XL0** image encoder (`seg_weights: efficientvit_sam/efficientvit_sam_xl0.pt`) with the fine-tuned
mask decoder `models/seg/xl0_v7r_s123.pt` (`seg_decoder`), one whole-image box prompt, the `multi3` output, and the
D18 fallback under `seg_min_area_frac` 0.0285. The brief named `efficientvit_sam_xl1`; that file is in
`models_pretrained/` but is not the live encoder: the decoder's card (`models/seg/xl0_v7r_s123.json`) records
`base_weights: efficientvit_sam/efficientvit_sam_xl0.pt`, and `segment_beans.load_decoder` refuses a decoder over
other base weights. Torch on CPU in float32 at 8 threads (serving runs the segmenter in fp32 too, ADR 0019).

**Where today's 8-bit step sits.** `load_canonical` (`coffeecv/canonical.py:762`) decodes the uint16 canonical photo
and converts it to **uint8** sRGB with LittleCMS (`rec2020_to_srgb`, line 740). Every segmenter caller hands that
uint8 photo to `BeanSegmenter.predict_mask`: the crop stage (`segcrop_session.py:53`), `seg_predict.py:115`, serving
(`infer.py:444`) and the fine-tune (`seg_finetune.load_pair` → `load_for_record`). There `model_io.sam_input`
(`model_io.py:71`) resizes the photo to a 1,024 px long side with PIL's bilinear **on uint8**, which rounds a second
time (`sam_resize`, line 62), divides by 255, normalises and pads (`_to_chw_normalised`, line 36). So today rounds
twice: the whole photo, then the 1,024 px encoder input.

**The arms** (`bits_seg_common.py`) quantize the whole float photo at the first of those two places, the
`load_canonical` output, and keep everything after it in float32:

| arm | the photo's samples | then |
|---|---|---|
| f32 | float32 as read back (the reference) | float |
| f16 | rounded to float16, the owner's "16-bit float" | float |
| u12 | round(x · 4095) / 4095 | float |
| u10 | round(x · 1023) / 1023 | float |
| u8 | round(x · 255) / 255 | float |
| today | `load_canonical`'s uint8 (LittleCMS with 8-bit output), then `model_io.sam_input` unchanged (uint8 resize, rounded again) | today's code |

"Then float" is: the resize to the encoder's side by PIL's bilinear in mode "F", channel by channel (ML-37 D11 (a),
the same filter as today's, `research_d11_float_resize_and_augment.md` §1; the code is `d14_mask_sensitivity.py`'s
`float_resize`), the EfficientViT-SAM mean and std (`model_io.SAM_MEAN`, `SAM_STD`) and the zero pad, all float32.

**The float picture** is the classifier study's stand-in for ML-37's read-back, on the whole photo instead of a crop:
today's canonical photo decoded at 16 bits (`canonical.decode_canonical`), converted by `load_canonical`'s own
LittleCMS transform with float32 output and clipped to [0, 1] (`bits_cls_common.readback`, imported unchanged). It is
`load_canonical`'s picture without the 8-bit step: rounded to 8 bits it differs from it in at most 503 pixels a photo
(`d14_mask_sensitivity.txt`). It is today's canonical photo, not D1's half-float one.

**Why quantize there.** It is where D3 removes the step: under ML-37 `load_canonical` returns float (D2, D3) and
`sam_resize` becomes the float resizer (D11). Quantization is pointwise, so rounding the photo is what an n-bit
read-back would hand the segmenter. The `today` arm shows the other placement, the second rounding after the resize.

**Yardsticks**, changes that do not matter, each against f32:
- **fp16in**: the f32 arm's normalised encoder input cast to float16 and back.
- **shift1**: the float photo moved one photo pixel left (last column repeated), same size: a quarter to a third of
  an encoder pixel on a 12 MP photo.
- **gen2**: a second JPEG XL generation, the decoded canonical photo re-encoded at the store's quality and effort
  (q95 e7, `jxl_encode.encode_rec2020` as `bits_cls_noise.jxl` does) and read back as f32.
- **thread count** (test photos only): the stored masks of `seg_ml19_predict@xl0_v7r_s123` (today's path at 4
  threads) against the today arm here (8 threads). Mask bits differ by a few pixels across thread counts
  (`segment_beans.THREADS`' comment; `docs/agents/experiments.md`, "Reproducibility limits").

## 2. The photos, and the segmenter's own quality measure

**There is no labelled test set.** The segmenter dataset (`labels/ml5/seg_dataset.yaml`, `coffeecv/seg_dataset.py`)
splits its photos 60/15/25 into train, validation and test, and the owner's labels (`data/ml5_labels/<pass>/labels.csv`)
cover only train and validation: "a test photo never appears there (D4)" (`seg_finetune.py` docstring;
`seg_finetune.label_rows` raises on one). The test split is judged by the owner's blind paired review
(`seg_review paired`, from the stored masks `seg_predict` writes), not by a number. The repo's one quantitative
measure against labels is the fine-tune's checkpoint selection, `seg_finetune.evaluate`: the mean hard IoU of the
serving output (box prompt, `multi3`) on the validation positives, at the encoder's 1,024 px frame with the label
brought there by `label_in_frame` (INTER_AREA, ≥ 0.5, padding ignored), and the share of validation negatives whose
mask is empty or tiny (under `seg_ft.select_min_area_frac` 0.083); the score is their mean.

So the two sets here are:
- **validation** (levels 1–3): every validation photo of the segmenter dataset the owner labelled in pass17, the
  latest session: 177 accepted positives (129 pool, 47 segmenter positives, 1 internet photo) and 88 negatives, 265
  photos, all of them. `xl0_v7r_s123` never trained on them: its card (`models/seg/xl0_v7r_s123.json`) lists pass16's
  train and neg_train roles; pass16's 120 validation positives chose its checkpoint, which biases its absolute
  val_miou up a little and the paired differences here not at all. Each label is checked to be for the same file
  (sha256). Level 3 reuses `seg_finetune.evaluate`'s measures as written (`bits_seg_common.label_scores`:
  `F.interpolate` bilinear to the frame, `> 0`, IoU on the valid pixels, 1 when both are empty), and adds the
  full-resolution IoU of the serving mask against the full-resolution label.
- **test** (levels 1–2): the segmenter dataset's test split, 472 photos (308 positives, 164 negatives), in a seeded
  random order, as many as time allowed (§4). Its stored masks (`data/seg_masks_ml19/xl0_v7r_s123`, today's path at 4
  threads) give the thread-count yardstick.

The camera groups are read from the path as `bits_cls_predict.camera` reads them (pixel, oneplus, iphone, sony,
canon; "other" is mostly Sony Xperia `G8441` and messenger photos); negatives and the internet positive are their
own groups. The validation photos are 0.27 to 50.1 MP, median 12.5 MP.

## 3. What the encoder's embedding sees (level 1)

`bits_seg_measure.py`: each arm's and yardstick's image embedding (the encoder's output, 256 × 64 × 64, float32)
against the f32 arm's: 1 − cosine and relative L2 of the whole embedding, and the largest 1 − cosine of a single
location's 256-vector. Validation, 265 photos (`bits_seg_report.txt`):

| form | 1 − cos median | p95 | max | rel. L2 median | max | worst location, median photo |
|---|---|---|---|---|---|---|
| u12 | 2.0e-9 | 2.2e-8 | 4.9e-7 | 6.4e-5 | 9.9e-4 | 1.6e-7 |
| f16 | 3.8e-9 | 7.1e-8 | 7.8e-6 | 8.7e-5 | 3.9e-3 | 2.9e-7 |
| *yardstick:* fp16 cast of the input | 3.2e-8 | 3.8e-7 | 7.9e-6 | 2.5e-4 | 4.0e-3 | 2.9e-6 |
| u10 | 3.6e-8 | 5.6e-7 | 4.4e-6 | 2.7e-4 | 3.0e-3 | 2.6e-6 |
| **u8** | **8.4e-7** | 2.6e-5 | 2.8e-4 | 1.3e-3 | 2.4e-2 | 6.9e-5 |
| *yardstick:* a second q95 e7 JPEG XL generation | 8.1e-6 | 3.9e-5 | 1.4e-4 | 4.0e-3 | 1.7e-2 | 4.4e-4 |
| today (two roundings) | 2.4e-5 | 1.2e-4 | 1.2e-3 | 7.0e-3 | 4.8e-2 | 1.3e-3 |
| *yardstick:* the photo moved one pixel | 3.7e-4 | 1.7e-3 | 6.7e-3 | 2.7e-2 | 1.2e-1 | 6.3e-3 |

Test, 360 photos (239 positives, 121 negatives), the same measures (`bits_seg_report.txt`): u12 2.1e-9, f16 3.6e-9,
fp16 cast 3.2e-8, u10 3.5e-8, **u8 7.8e-7**, second generation 8.8e-6, today 2.4e-5, one-pixel shift 3.8e-4 (1 − cos
median). Per-photo ratios as on validation: u8 / f16 194, u10 / fp16 cast 1.04, u8 / second generation 0.089,
u8 / one-pixel shift 0.0020, today / u8 30. Both sets together (625 photos): u8 7.9e-7, u10 3.5e-8, u12 2.0e-9,
f16 3.7e-9, fp16 cast 3.2e-8.

- **u12 and f16 are at float32's arithmetic floor**, a hundredth of `experiments.md`'s cross-CPU limit (1 − cos
  2e-7, "Reproducibility limits"). These are the classifier's numbers almost exactly (u12 1.5e-9, f16 3.8e-9 there).
- **u10 is level with an fp16 cast of the input** (per-photo ratio median 1.1). Note the segmenter does not serve
  in fp16 (ADR 0019); the cast is only a size of perturbation the encoder takes without consequence (§4).
- **u8 is about 200× further from float than f16** (median of the per-photo ratio), 24× the fp16 cast; but a tenth
  of a second JPEG XL generation at the store's own quality (ratio median 0.09) and 1/460 of a one-pixel shift.
- **The place of the 8-bit step matters more than the number of bits, as for the classifier.** Today's path, which
  rounds again after the resize, moves the embedding 29× more than one 8-bit rounding of the photo (per-photo
  median), more than a second JPEG XL generation.
- **Per camera group** u8's median runs from 1.8e-7 (Canon) to 1.2e-6 (Pixel): the larger, cleaner photos are
  averaged more by the downscale to 1,024 px. No group stands out.

## 4. The masks and the crop box (level 2)

Each form's full-resolution mask (`BeanSegmenter.predict_mask`'s: `multi3`, `model_io.upsample_mask`) against the f32
arm's mask: IoU, pixels changed, empty and D18-fallback flips, and the largest move of an edge of the D4 crop box
`segcrop` cuts (`segment_beans.d4_box`, `seg_keep_frac` 0.95). A mask under `seg_min_area_frac` is not used (the D18
fallback crops the whole photo), so those photos are counted apart: only a flip would matter there.

Validation, 265 photos: 176 positives and 1 negative with a served mask, 88 photos (87 negatives, 1 positive) that
fall back (`bits_seg_report.txt`):

| form | served masks: IoU median | min | < 0.999 (ADR 0020's bar) | px changed median | max | box moved (max px) | empty / fallback flips (of 265) |
|---|---|---|---|---|---|---|---|
| f16 | 1.00000 | 0.9988 | 2 of 177 | 4 | 6,679 | 3 of 177 (2) | 0 / 0 |
| u12 | 1.00000 | 0.9963 | 1 | 3 | 1,719 | 1 (1) | 0 / 0 |
| u10 | 1.00000 | 0.9934 | 3 | 15 | 4,234 | 2 (2) | 0 / 0 |
| *fp16 cast* | 1.00000 | 0.9950 | 3 | 15 | 9,230 | 6 (4) | 0 / 0 |
| **u8** | **0.99999** | 0.9775 | 4 | 64 | 31,430 | 7 (2) | 0 / 0 |
| *second generation* | 0.99995 | 0.9176 | 7 | 264 | 295,970 | 14 (10) | 0 / 0 |
| today | 0.99991 | 0.9109 | 8 | 388 | 140,121 | 25 (114) | 0 / 0 |
| *one-pixel shift* | 0.99929 | 0.7820 | 75 | 3,992 | 535,642 | 115 (230) | 0 / 0 |

Test, 360 photos in seeded random order (the run stopped at its time limit after 360 of 472): 237 positives and 6
negatives with a served mask, 117 photos (115 negatives, 2 positives) that fall back (`bits_seg_report.txt`, the
`pos` and `neg` rows):

| form | positives (237): IoU median | min | < 0.999 | px changed median | max | box moved (max px) | negatives (6): IoU median | min | box moved |
|---|---|---|---|---|---|---|---|---|---|
| f16 | 1.00000 | 0.99976 | 0 | 4 | 645 | 1 (1) | 0.99999 | 0.99994 | 0 |
| u12 | 1.00000 | 0.99970 | 0 | 3 | 806 | 0 | 0.99999 | 0.99996 | 0 |
| u10 | 1.00000 | 0.99960 | 0 | 14 | 1,120 | 0 | 0.99997 | 0.99986 | 0 |
| *fp16 cast* | 1.00000 | 0.99938 | 0 | 12 | 2,385 | 2 (1) | 0.99999 | 0.99981 | 0 |
| **u8** | **0.99999** | 0.99807 | 2 | 60 | 5,214 | 9 (4) | 0.99985 | 0.99781 | 0 |
| *second generation* | 0.99994 | 0.98325 | 5 | 253 | 45,353 | 22 (7) | 0.99919 | 0.99447 | 3 (2) |
| today | 0.99991 | 0.99037 | 6 | 410 | 38,419 | 30 (4) | 0.99975 | 0.98922 | 2 (6) |
| *one-pixel shift* | 0.99911 | 0.85109 | 111 | 4,370 | 473,202 | 154 (30) | 0.99624 | 0.97108 | 5 (4) |

No empty or fallback flip in any arm or yardstick on the 360 photos. Per camera group (served masks, median IoU
against f32), u8 is 0.99998 or more in every camera group of positives (served negatives 0.99985) and u10 0.99999
or more in every group. The thread-count
yardstick: the stored masks (`seg_ml19_predict`, 4 threads) against the today arm (8 threads) have median IoU
1.00000, at most 8 pixels changed, no box moved.

- **No mask became empty, stopped being empty, or crossed the fallback threshold** in any arm or yardstick, on
  either set: the crop-or-whole-photo decision never changed.
- **The few masks that move are the segmenter's least confident.** The four served masks where u8 is under 0.999
  against f32 have predicted IoUs of 0.66–0.76 (the median served mask's is 0.98): an internet photo
  (`internet_beans_001`, 0.9775; its f32 mask scores 0.11 against its label), a Pixel
  negative the segmenter masks wrongly at 5% of the frame (0.9922), and the two copies of one Canon photo
  (`20261113_172727_Canon_EOS_90D_Brazil` and its `-jxl` twin, 0.9955 and 0.9968). The same photos move under the
  fp16 cast and a second generation, and far more under a one-pixel shift (0.78, 0.95, 0.93, 0.95). They are
  photos near a decision, not photos sensitive to bits.
- **Pixels changed, as a share of the photo** (median / max): f16 5e-8 / 2e-4, u10 3e-7 / 7e-4, fp16 cast
  3e-7 / 5e-4, u8 1.3e-6 / 2.6e-3, second generation 5e-6 / 9e-3, today 1.3e-5 / 5e-3, one-pixel shift 7e-5 / 2.6e-2.
- **The crop box**: u8 moves an edge of 7 of 177 boxes, by at most 2 px; today's path 25 boxes, one by 114 px (the
  Canon photo above, whose 5%-of-frame mask today's rounding changes most); the one-pixel shift 115 boxes, up to
  230 px.
- **Per camera** (served masks, median IoU against f32): u8 is 0.99997 or more in every camera group of positives
  (the single internet photo 0.9775); u10 1.00000 in each of them. Positives and negatives behave alike: 87 of 88 negatives fall
  back in every arm.

## 5. Quality against the owner's labels (level 3)

`seg_finetune.evaluate`'s measures on the 265 labelled validation photos, per arm (`bits_seg_report.txt`):

| form | val_miou (frame) | IoU, full resolution | negatives empty or tiny | score | positives under 0.9 |
|---|---|---|---|---|---|
| f32 | 0.96990 | 0.96993 | 88 / 88 | 0.98495 | 7 |
| f16 | 0.96990 | 0.96993 | 88 / 88 | 0.98495 | 7 |
| u12 | 0.96990 | 0.96993 | 88 / 88 | 0.98495 | 7 |
| u10 | 0.96989 | 0.96992 | 88 / 88 | 0.98495 | 7 |
| **u8** | **0.96992** | 0.96995 | 88 / 88 | 0.98496 | 7 |
| today | 0.96990 | 0.96993 | 88 / 88 | 0.98495 | 7 |
| *fp16 cast* | 0.96990 | 0.96993 | 88 / 88 | 0.98495 | 7 |
| *second generation* | 0.96988 | 0.96991 | 88 / 88 | 0.98494 | 7 |
| *one-pixel shift* | 0.96981 | 0.96983 | 88 / 88 | 0.98490 | 7 |

Paired differences of val_miou per positive photo, mean with a photo-bootstrap 95% CI (positives resampled, 10,000
draws, the same draw for both arms of a pair):

| pair | val_miou change | 95% CI | photos changed by more than 0.001 (of 177) | largest change |
|---|---|---|---|---|
| f16 − f32 | +0.0000001 | [−0.0000032, +0.0000035] | 0 | 0.0002 |
| u12 − f32 | −0.0000017 | [−0.0000076, +0.0000022] | 0 | 0.0004 |
| u10 − f32 | −0.0000094 | [−0.0000247, +0.0000009] | 0 | 0.0010 |
| **u8 − f32** | **+0.0000156** | **[−0.0000069, +0.0000505]** | 1 | 0.0024 |
| **u8 − f16** | **+0.0000155** | **[−0.0000065, +0.0000518]** | 1 | 0.0026 |
| u10 − f16 | −0.0000095 | [−0.0000225, +0.0000001] | 0 | 0.0008 |
| today − f32 | −0.0000045 | [−0.000140, +0.000166] | 6 | 0.011 |
| *fp16 cast − f32* | +0.0000015 | [−0.0000049, +0.0000090] | 0 | 0.0005 |
| *second generation − f32* | −0.0000191 | [−0.000180, +0.000140] | 4 | 0.0094 |
| *one-pixel shift − f32* | −0.0000960 | [−0.000332, +0.000172] | 9 | 0.017 |

- **No arm differs from float in quality.** Every CI contains 0 and is at most ±5e-5 wide for u8: a quality gain
  from more bits, if any, is below 0.00005 of mean IoU, about 1/170 of the interquartile range of the photos' own
  IoU against their labels (f32: 0.9906 to 0.9989, median 0.9958). The full-resolution IoU gives the same changes to within
  4e-6. 8 bits even scores a hair higher than float (+1.6e-5), inside its CI.
- **The negatives do not move at all:** 88 of 88 empty or tiny in every arm and yardstick, no flip.
- **Per camera group** (u8 − f32, mean [95% CI]): Pixel (75) −0.000004 [−0.000012, +0.000003], OnePlus (34)
  −0.000010 [−0.000039, +0.000009], iPhone (20) −0.000004 [−0.000012, +0.000004], Sony (16) +0.000002 [−0.000001,
  +0.000005], Canon (8) +0.000087 [−0.000087, +0.00036], other (23) +0.000014 [−0.000002, +0.000042]; the one
  internet photo +0.0024. f16 − f32 is within ±3.3e-5 in every group. By source: pool (129) +0.000004 [−0.000008,
  +0.000023], segmenter positives (47) −0.000004 [−0.000029, +0.000017].
- **Today's path against the float path** (the whole of D3 for the segmenter, both roundings removed): −0.0000045
  [−0.00014, +0.00017], 0 of 88 negatives flipped. It changes 6 photos by more than 0.001, in both directions, as a
  second JPEG XL generation changes 4.
- **For scale**, the decoder's own record: between consecutive training epochs, `xl0_v7r_s123`'s val_miou moved by
  as much as 0.066 (card `history`, 27 epochs), and ADR 0019 rejected an fp16 segmenter graph whose least mask IoU was 0.9976
  against ADR 0020's bar of 0.999 for every photo.

**The decoder trained on 8-bit input**, noted and not changed: `seg_finetune.TrainSet` holds each training photo as
`sam_resize`'s uint8 at the frame, the augmentation (`seg_ft.augment`) runs on it in uint8, and `encode` feeds it
through `model_io.sam_input` (`seg_finetune.py:273–302`). Its augmentation is far coarser than any rounding here:
ColorJitter brightness, contrast and saturation ±0.2, PlanckianJitter, HueSaturationValue, RandomGamma 80–120, JPEG
quality 60–95 and Gaussian noise of 1–4% of the range (`params.yaml` `augment`, `seg_ft.augment`). The pretrained
encoder too saw only 8-bit JPEG content (SA-1B), read through a float antialiased resize
(`research_d14_segmenter_and_ml33.md` §7). Nothing in the model has learned to depend on the 8-bit grid, and the
measurements above find no sign of it either way.

## 6. The answer

- **Does going above 8 bits give the segmenter a measurable gain?** No. With the 8-bit step at the photo and the
  resize in float, u8 against float32 changes the validation val_miou by +0.000016 [−0.000007, +0.000051] over 177
  labelled positives, flips none of 88 negatives, never changes the fallback decision, moves the median served mask
  by 64 pixels of a 12 MP photo and the crop box by at most 2 px. f16, u12 and u10 are closer still. Nothing an
  owner-judged label can see.
- **From which bit depth is it indistinguishable from float?** 10 bits: level with an fp16 cast of the input in the
  embedding (median 3.6e-8 against 3.2e-8) and in the masks (15 pixels changed at the median for both). 12 bits and
  float16 are at float32's arithmetic floor (embedding 1 − cos 2e-9 and 4e-9, 3–4 mask pixels). 8 bits is
  measurable in the embedding (8e-7 at the median) and in a few dozen mask pixels, and in nothing downstream.
- **Same answer as the classifier**, by the same measures: the segmenter is no more sensitive to bit depth than
  DINOv3 (u8's median 1 − cos 8.4e-7 here against 3.5e-7 for the classifier's 224 px patches).
- **What this means for the ticket.**
  - D3 (no 8-bit step) gives the segmenter nothing measurable; for the segmenter it is a correctness and parity
    choice, as the classifier note found. D14's finding (median IoU 0.9999 under ML-37's simulated input) holds for
    every bit depth here.
  - D11 (a), the float resize, is what removes today's larger effect: the second rounding after the resize moves the
    embedding 29× more than one 8-bit rounding of the photo. Whatever the photo's bit depth, no rounding should
    follow the resize.
  - R9 (memory): the segmenter's fine-tune could hold its training photos at 8 bits, or in float16 (5.1 GB, the D11
    note), with no quality cost measurable here; float32 (10.3 GB) buys nothing over float16 (u8 − f16 and f16 − f32
    above). This is an inference for a future fine-tune, not a measurement of one.
  - D13's mask floor (median IoU per camera ≥ 0.99 against today's masks) is not threatened by the bit depth:
    today's path against float32 is 0.9999 or more at the median on served masks in every camera group of positives
    (the single internet photo 0.952).

## 7. What this does not measure

- **D1's photo and D2's read-back.** All arms start from today's canonical photo (uint16, BT.709 curve, q95) read
  back by LittleCMS; D14's note measured the simulated D1 photo (median IoU 0.9999 against today's).
- **A decoder fine-tuned on float input.** Nothing was trained; ML-33's segmenter does not exist yet.
- **The test split against labels.** It has none (D4); level 3 is on the validation split, which
  `xl0_v7r_s123`'s checkpoint selection saw in part (pass16's 120 of the 177 positives).
- **Light above white and D6.** Today's canonical photos have none.
- **Serving on LiteRT** (`yappy`). Torch on CPU only; the served graph is fp32 (ADR 0019), held to ADR 0020's bars.
- **112 of the 472 test photos.** The test run stopped at its 02:20 UTC limit after 360 photos, taken in a seeded
  random order (`bits_seg_measure.py`), so the 360 are an unbiased subset; validation was measured in full.

## Files

In `analysis/ml37/`:
- `bits_seg_common.py`: the arms, the yardsticks, the float read-back (`bits_cls_common.readback`) and the float
  resize, the live segmenter's loading, and the label scores (`seg_finetune.evaluate`'s measures).
- `bits_seg_measure.py`: levels 1–3 per photo, `--split validation` or `--split test`, with a stop time; writes one
  JSON row a photo to `tmp/ml37/bits_seg/<split>.jsonl` and a progress log beside it (gitignored, local only).
- `bits_seg_report.py`: the tables, from those rows; its output is `bits_seg_report.txt`, regenerated on all 360
  test rows after the run stopped (an earlier copy on 167 test photos is kept in `tmp/ml37/bits_seg/`).

Run, from `/workspace`: `PYTHONPATH=/workspace uv run python analysis/ml37/bits_seg_measure.py --split validation`
(then `--split test`), then `PYTHONPATH=/workspace uv run python analysis/ml37/bits_seg_report.py`. About 21 s a
photo at 8 threads, peak RSS 3.5 GB.
