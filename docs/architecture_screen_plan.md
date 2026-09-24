# Architecture comparison: U-Net variants, Meta model-zoo candidates, and the current ResNet18

Status: DRAFT, written 2026-09-21 while the Stage 1 plateau sweep (exp210-213) still occupies
the VM. Nothing here has been validated through `run_folds.py`. The one empirical result below
(§5) is a frozen-feature probe run locally, which is a *screen*, not an adoption test.

Scope: whether to change the classifier architecture, and where a U-Net-shaped (encoder-decoder)
model would actually earn its place in this pipeline. Excludes training-schedule work, which is
what Stage 1 is currently measuring (`docs/lr_scheduler_plan.md`).

---

## 1. What the current implementation is

One photo goes through five stages, only the fourth of which is a neural network:

1. **Crop** — `coffeecv/crop_tray.py`. Texture-based rough tray box (Otsu, scale-adaptive
   morphology), then bean-vs-rim separation by saturation, then the largest all-bean
   axis-aligned rectangle plus a safety trim. Pure OpenCV heuristics, no learned model.
   Self-checks contamination and raises a `needs_review` flag rather than guaranteeing anything.
2. **Scale** — `coffeecv/bean_scale.py`. Bean centre-to-centre pitch estimated per photo from
   the dominant period of the FFT radial power profile (`k` searched over 4..80 cycles, `k^1.5`
   weighting to defeat 1/f falloff, calibration constant 1.18). ~150 ms/photo.
3. **Patch sampling** — side = B x pitch, B log-uniform in [4, 7] (i.e. 16-49 beans per patch),
   40 patches per photo at inference, resized to 224.
4. **Classify** — `coffeecv/model.py`. torchvision **ResNet18**, ImageNet init, full fine-tune,
   with **MixStyle p=0.5** installed by forward hook after `layer1` and `layer2`, dropout 0.2,
   10-way linear head.
5. **Pool** — 8x dihedral TTA per patch, mean probability over patches, then the OOD guard
   (linear probe v2) decides classify-vs-refuse.

Where it stands (camera-rig era, exp200-203, seed 42):

| held-out camera | cross-rig macro-F1 |
|---|---|
| cam_pixel | 0.8184 |
| cam_sony | 0.7587 |
| cam_oneplus | 0.8304 |
| cam_iphone | 0.8537 (8 classes, not comparable) |
| **mean over the 3 ten-class folds** | **0.8025** |

In-distribution test macro-F1 for those same four fold runs is 0.9002-0.9164 (mean 0.9070); over
all ten camera-rig runs, exp200-209, it is 0.9002-0.9363. The **~10-point gap** between 0.9070
in-distribution and 0.8025 cross-rig is the problem any architecture change is being asked to
close. (README.md still quotes 0.884-0.933 from the 9-class era and is stale, as is its claim
that the service returns "all 9 classes" — the deployed checkpoint is `allrigs_cam_s123.pt`,
exp205, 10 classes.)

---

## 2. The honest framing on "U-Net with a ResNet backbone"

A U-Net is an encoder-decoder for **dense prediction**. There is no such thing as a U-Net
classifier: if you attach a classification head to a U-Net's bottleneck and train only that,
you have built exactly the current ResNet18 classifier plus a decoder whose output nothing
reads. The decoder only pays for itself if something consumes its pixels.

So the real question is not "U-Net or ResNet" — the U-Net *contains* the ResNet — but
**which of this pipeline's stages should become a learned dense-prediction model**. There are
three candidates, and they are not equally well-motivated.

### Role A — learned crop (replaces stage 1)
A U-Net predicting a bean-region mask on a downscaled photo, replacing the saturation
heuristic. The live-crop QA sweep (`outputs/qa_live_crop/qa_summary_all.json`) says what the
heuristic actually costs today:

| session | passthrough rate | IoU vs fixed trim |
|---|---|---|
| box | 0.000 | 0.892 |
| iphone | 0.033 | 0.724 |
| **oneplus** | **0.156** | 0.724 |
| pixel | 0.906 | n/a (frame-filling, passthrough is correct) |

"Passthrough" means the tray detector found nothing and handed back the uncropped photo. For
oneplus that is **15.6% of photos silently classified with tray rim and table still in frame**,
and it is not flagged (`needs_review_rate` is 0.000 everywhere — the flag never fires). A
learned mask would attack exactly this. Real value, but bounded: it fixes one rig's 16%, not
the 13-point cross-rig gap. Needs masks.

### Role B — learned bean density / instance map (replaces stage 2)  ** best evidenced **
This is where the project already has a measured teacher and measured headroom.
`analysis/bean_scale/README.md` benchmarked five pitch estimators against 30 hand-counted
crops. MobileSAM won:

| method | per-image MAPE | rig-bias spread | ms/image |
|---|---|---|---|
| M0 FFT (**shipped**) | 19.2% | 1.04 | 48 |
| M4 MobileSAM | **11.1%** | **1.02** | **97026** |

MobileSAM is nearly 2x more accurate per photo than the shipped estimator and is the only
candidate that beat it — and it is unusable live at 97 s/image. That is the exact shape of a
**distillation** problem: a slow teacher, a fast student, and an offline budget to generate
labels in. A U-Net-R18 predicting a bean density map (or a bean mask + watershed) trained on
MobileSAM's output would target teacher-quality pitch at ~30 ms.

Why this matters more than it looks: per-photo pitch noise is not a cosmetic number. It is
deliberately *kept* (see `bean_scale.py`'s docstring — the same estimator must run at training
and inference) and absorbed as scale augmentation. Halving it tightens what "4-7 beans across"
actually means on an unseen camera, which is the one quantity the cross-rig metric is most
sensitive to.

### Role C — auxiliary decoder, training only (regularizes stage 4)
Shared ResNet18 encoder; U-Net decoder predicting the bean mask as an **auxiliary loss**;
classification head on the bottleneck as today. At inference the decoder is deleted, so
**live cost is unchanged**. This is the literal reading of the request, and it is cheap
(measured: +42% per training step, +0% inference). The mechanism claimed for it — forcing the
encoder to represent bean geometry rather than whatever rig-specific colour statistic separates
the training cameras — is plausible and is the same family of argument as MixStyle, which is
this project's second-largest confirmed lever.

---

## 3. What the Meta model zoos actually offer here

The MODEL_ZOO in question is `facebookresearch/detectron2/MODEL_ZOO.md`: Faster R-CNN,
RetinaNet, Mask R-CNN, Cascade R-CNN, Keypoint R-CNN and Panoptic FPN over R50/R101/X101/RegNet
backbones, on COCO/LVIS/Cityscapes.

**Detectron2 is the wrong tool for this project, for concrete reasons, not taste:**

- Its quoted inference times (0.043 s/im for Mask R-CNN R50-FPN) are **GPU** numbers. This
  project has no GPU anywhere — the workstation is a Ryzen 5 5600U and the training box is a
  4-core VM. Measured here, plain ResNet50 *classification* at 224px already costs 42.6 ms/img;
  Mask R-CNN at 800x1333 with FPN + RPN + ROI heads is orders of magnitude beyond that.
- Training it needs per-bean instance masks for thousands of beans, and a GPU.
- The thing it would buy — per-bean instances — is obtainable offline from SAM, which the repo
  already vendors (`mobile_sam.pt`) and has already benchmarked.

What *is* worth taking from Meta's releases:

| model | role here | licence | verdict |
|---|---|---|---|
| **SAM / MobileSAM** | offline **teacher** for Roles A/B/C masks | Apache-2.0 | **take** — already in repo, already benchmarked as the best pitch estimator |
| **DINOv2** (ViT-S/14, 21M) | frozen backbone alternative | Apache-2.0 | screen it (see §5) |
| DINOv3 | stronger backbone | custom Meta commercial licence, gated behind identity approval | **avoid** — wrong for a publicly deployed service |
| ConvNeXt | backbone alternative | MIT | torchvision's copy is equivalent; 3.6x train cost |
| Detectron2 detectors | per-bean instances | Apache-2.0 | **skip** — CPU-infeasible, and SAM covers it offline |

---

## 4. Measured cost on this hardware

All measured on the workstation, 6 torch threads, batch 32, 224px input
(`analysis/architecture_screen/bench.py`). Training step = forward + backward + AdamW step.

| architecture | params | train step (B=32) | inference | vs current (train / infer) |
|---|---|---|---|---|
| **resnet18 (current)** | 11.7M | **1.33 s** | **13.9 ms/img** | 1.0x / 1.0x |
| UNet-R18, multi-task | 14.3M | 1.89 s | 13.9 ms/img (decoder dropped) | **1.4x / 1.0x** |
| efficientnet_b0 | 5.3M | 2.23 s | 19.7 ms/img | 1.7x / 1.4x |
| resnet34 | 21.8M | 2.24 s | 20.7 ms/img | 1.7x / 1.5x |
| convnext_tiny | 28.6M | 4.73 s | 40.7 ms/img | 3.6x / 2.9x |
| resnet50 | 25.6M | 4.63 s | 42.6 ms/img | 3.5x / 3.1x |
| dinov2_vits14 (frozen) | 22.1M | n/a (frozen) | 43.2 ms/img | - / 3.1x |

Two consequences — but both apply to **fine-tuned** arms only. §5 shows they invert for frozen
ones, so read this section together with §5's deployment table rather than on its own.

**Deployment (fine-tuned arms).** The webapp classifies at 40 patches x 8 dihedral TTA = **320
forward passes per photo**. At ResNet18 that is ~4.5 s of compute on this workstation and more on
the 4-core VM. A fine-tuned resnet50 or convnext_tiny, which still needs TTA for the same reason
ResNet18 does, puts a single `/classify` call near 14 s locally and worse on the VM. Those are
not droppable into the deployed path without also cutting the patch budget or TTA — and TTA is
worth +0.0235 cross-rig, so cutting it to afford a bigger backbone may hand back exactly what the
backbone won.

*The exception that matters:* a **frozen** backbone accurate enough not to need TTA drops the 8x
multiplier entirely, and then 3x per-image is a net 2.6x saving rather than a cost. That is
exactly the DINOv2 case measured in §5 — which is why this row of the table is misleading read
alone.

**Screening (fine-tuned arms).** A fold is ~5-8 h on the VM at ResNet18, so a 4-fold single-seed
screen is ~1 day. At 3.5x that is ~4-5 days for one seed, against a seed-to-seed sd of 0.029 that
already forces multi-seed evidence. **Fine-tuned heavy backbones are not merely slower to serve —
they are too slow to be validated to this project's evidence bar on this hardware.** Frozen
backbones have no backward pass at all: the entire 4-fold sweep in §5 took 17 minutes.

---

## 5. Empirical screen: do other backbone families transfer across cameras better?

### Method

Every arm sees **byte-identical patches** (the project's own `MultiPhotoPatchDataset`, bean-unit
sizing 4-7, 120 patches/class/rig, seed 42, eval transform), is **frozen**, and is scored by the
same plain logistic regression (standardised features, C=1.0, no per-arm tuning). Leave-one-camera-
out: fit on three rigs, score macro-F1 on the fourth. Script: `analysis/architecture_screen/xrig_probe.py`.

This is a **paired** comparison of *pretrained feature quality under camera shift*. It is not the
pipeline and not an adoption test — nothing here is fine-tuned, nothing uses MixStyle or TTA.

### Result

Mean cross-rig macro-F1, frozen features + linear probe:

| backbone | dim | 4-fold mean | 3 ten-class folds | pixel | sony | oneplus | iphone |
|---|---|---|---|---|---|---|---|
| resnet18_in1k (**current family**) | 512 | 0.4645 | 0.4938 | 0.578 | 0.453 | 0.451 | 0.377 |
| resnet50_in1k | 2048 | 0.5198 | 0.5588 | 0.606 | 0.483 | 0.587 | 0.403 |
| convnext_tiny_in1k | 768 | 0.6691 | 0.7013 | 0.739 | 0.628 | 0.737 | 0.572 |
| **dinov2_vits14 (frozen SSL)** | 384 | **0.8324** | **0.8511** | 0.888 | 0.823 | 0.843 | 0.776 |

**The ordering is perfectly sign-consistent: 4/4 folds, every pairwise comparison.** The spread
between the best and worst arm is +0.368 mean — an order of magnitude beyond this project's
0.029 seed sd and its ±0.048 noise band.

### What this does and does not say

The number that should stop you: **frozen DINOv2 features and a logistic regression score 0.8511
on the same three ten-class leave-one-camera-out folds where the fully fine-tuned
ResNet18 + MixStyle + TTA pipeline scores 0.8025.** No fine-tuning, no MixStyle, no TTA, no
bean-unit augmentation benefit — just better features.

Read that carefully, because it is easy to over-claim:

- **The probe's arms train on more photos.** It uses `split="all"` on the three training rigs
  (every photo), where a real fold trains on the 70% photo split. That inflates *every* arm
  equally, so the backbone ranking is unaffected, but it makes the 0.8511-vs-0.8025 comparison
  optimistic rather than exact. Treat it as "the same ballpark from frozen features", not "DINOv2
  beats the shipped model by 0.049".
- **The iphone fold is not 10-class.** cam_iphone has no class_008, and this probe (unlike
  `compute_split_metrics`) does not exclude an absent class from the macro average, so a phantom
  f1=0 drags that column down. Same treatment in every arm; that is why the 3-fold column exists.
- **Architecture is confounded with pretraining recipe.** ConvNeXt-T and ResNet50-V2 ship modern
  training recipes; torchvision's ResNet18 has only the 2016-era V1 weights. Some of the
  ResNet18 -> ConvNeXt gap is recipe, not architecture. §5b separates them.
- **Frozen ranking need not survive fine-tuning.** Full fine-tuning on 938 photos plus MixStyle
  already recovers a lot; the current pipeline goes from 0.4938 frozen to 0.8025 fine-tuned. The
  right inference is "the encoder is a large untested lever", not "swap it tomorrow".

### The deployment arithmetic — and why it favours DINOv2, not against it

The naive comparison (43.2 vs 13.9 ms/img, so "3.1x too slow") is wrong, because it ignores that
**the current pipeline spends 8x of its budget on TTA and the probe used none.**

| configuration | passes/photo | compute/photo | cross-rig macro-F1 (3 ten-class folds) |
|---|---|---|---|
| deployed today: resnet18 fine-tuned, 40 patches x 8 TTA | 320 | **4.45 s** | 0.8025 |
| dinov2 frozen + linear head, 40 patches, **no TTA** | 40 | **1.73 s** | 0.8511 (probe, see caveats) |
| dinov2 frozen, 40 patches x 8 TTA | 320 | 13.8 s | not measured |

TTA exists to average away orientation disagreement in a model that is only approximately
invariant; it is worth +0.0235 to the fine-tuned ResNet18. The probe scored 0.8511 without it. So
in its *deployed* configuration a frozen DINOv2 is **~2.6x cheaper than what is running in
production right now**, not 3.1x more expensive.

The same inversion hits the screening cost, harder. A frozen backbone has no backward pass: a
full 4-fold leave-one-camera-out evaluation is embedding extraction (~4 min/rig) plus seconds of
logistic regression — **the whole sweep above took 17 minutes on this workstation**, against
~30 h on the VM for one seed of a fine-tuned fold sweep. The claim in §4 that heavy backbones are
"too slow to ever be validated on this hardware" holds for *fine-tuned* arms and is exactly
backwards for *frozen* ones.

Two real costs remain, and neither is a blocker:

- **The OOD guard is tied to the 512-d ResNet18 embedding space.** Changing the backbone means
  rebuilding the reference and refitting probe v2 (`build_ood_reference.py`, `fit_ood_probe.py`).
  Scripted, but it must be redone and re-validated against the same negatives — including the
  green-legume set that defeated probe v1.
- **MixStyle does not apply to a frozen backbone at all**, so the +0.1400 lever simply leaves the
  design rather than being ported. That is fine if the frozen features are already camera-
  invariant — which is precisely what the probe measures — but it means the two designs are not
  incrementally comparable, and the fine-tuned-DINOv2 middle ground is untested.

### 5b. Recipe or architecture?

DINOv2's lead could be any of three things, with very different consequences: a modern
**pretraining recipe** (free to copy), the **ViT architecture** (costs 3x), or **DINO
self-supervision on LVD-142M** (only obtainable by using the weights). A second probe, identical
protocol, separates them. Script: `analysis/architecture_screen/xrig_probe2.py`.

| arm | dim | 4-fold mean | 3 ten-class folds | pixel | sony | oneplus | iphone |
|---|---|---|---|---|---|---|---|
| resnet18 a1_in1k (**modern recipe**) | 512 | 0.4496 | 0.4763 | 0.529 | 0.468 | 0.432 | 0.369 |
| resnet18 tv-V1 (**control**) | 512 | 0.4645 | 0.4938 | 0.578 | 0.453 | 0.451 | 0.377 |
| vit_small supervised in1k (**ViT, no SSL**) | 384 | 0.5534 | 0.5548 | 0.638 | 0.463 | 0.564 | 0.549 |
| dinov2_vits14 (**SSL, anchor**) | 384 | 0.8324 | 0.8511 | 0.888 | 0.823 | 0.843 | 0.776 |

The control arm reproduced probe 1 **exactly** — 0.4645, and identical to four decimals on every
fold — as did the DINOv2 anchor at 0.8324. Patch sampling, embedding and probe are fully
deterministic across independent runs, so the two tables are directly comparable.

**Attribution, and it is unusually clean:**

- **Not the recipe.** `resnet18.a1_in1k` (ResNet-strikes-back recipe, same architecture, same
  11.7M params, same 13.9 ms/img) scores **0.4496 against the 2016-era V1 weights' 0.4645** —
  marginally *worse*. There is no free lunch in the weights. This kills the cheapest hypothesis.
- **Partly the architecture, but only a little.** Supervised ViT-S/16 on ImageNet-1k reaches
  0.5534, about +0.09 over ResNet18. Real, but it explains under a quarter of DINOv2's +0.368.
- **Mostly the self-supervised pretraining.** DINOv2 beats the *same-size, same-family,
  same-dimension* supervised ViT-S by **+0.279**. Architecture held constant, the only variables
  left are the DINO objective and LVD-142M's scale.

The practical consequence is blunt: **this gain cannot be engineered around.** It is not a knob
you can turn on ResNet18, not a better recipe, and not a transformer per se. Either the DINOv2
weights are in the pipeline — directly, or as a distillation teacher — or the gain is not
available. That is also why the licence note in §3 matters: DINOv2 is Apache-2.0 and may be used
and redistributed freely; DINOv3 is not, and this project serves a public endpoint.

---

## 6. The MixStyle trap — read before screening any architecture

`build_model` raises if `mixstyle_p > 0` and `model_name != "resnet18"`: MixStyle is wired by
forward hooks onto `layer1`/`layer2` and nothing else. MixStyle p=0.5 is worth **+0.1400**
cross-rig (9/9 sign-consistent, Phase 16) — the second-largest confirmed lever in the project.

So the naive screen — flip `model_name` to `resnet50` and run folds — compares *a new
architecture without MixStyle* against *ResNet18 with MixStyle*, and will reject a good
architecture for a reason that has nothing to do with it. Any architecture arm must port
MixStyle first:

- ResNet34/50: trivially the same, hook `layer1`/`layer2`.
- ConvNeXt: hook `features[1]`/`features[3]`; the per-channel mean/std statistic is well-defined.
- ViT (DINOv2): MixStyle on token statistics is a different formulation and much less
  established. A ViT arm is therefore confounded by construction unless it runs frozen.

Second trap, from `feedback-cli-defaults-overwrite-config`: `run_folds.set_fold()` currently has
no `model_name` parameter at all. Adding one must **inherit** when the flag is omitted, never
default to a hardcoded architecture, or the next sweep silently retrains the wrong baseline.

---

## 7. Dataset features worth adding

The prompt offered new dataset features. Ranked by impact per hour:

1. **MobileSAM bean masks for every cropped photo.** ~97 s/photo x 938 photos ~= 25 CPU-hours,
   one-time, parallelizable, fully automatic — no hand-labelling. Unlocks Roles A, B and C at
   once, and is a prerequisite for every U-Net idea in this document. QA with the existing
   `build_contact_sheet`, at full resolution (see `feedback-cv-pipeline-qa`).
2. **Extend counted pitch ground truth to the five newer rigs.** `analysis/bean_scale` ground
   truth is 3 rigs from 2026-08-11 and predates iPhone/oneplus entirely; it is already the
   named blocker for the `k_lo` change, and it would be the yardstick for a distilled
   estimator. The *paired* question needs only ~10-20 crops/rig (VLM counting is validated for
   exactly this); absolute per-rig bias would need ~190/rig and is not worth it.
3. **iPhone class_008 photos (+20).** Existing item A1; cam_iphone is the only rig that cannot
   produce a 10-class cross-rig number, which is why exp203's 0.8537 is excluded from the mean.
4. **A fifth camera.** Not an architecture change, but with 4 rigs the headline metric is a mean
   of 3 usable numbers. More cameras raise the ceiling on what any architecture screen can
   even resolve.

---

## 8. Recommendation

The frozen-feature probe (§5) changed this document's ranking while it was being written. The
original draft led with the SAM distillation, on the reasoning that the encoder was expensive to
change. The measurement says the encoder is both the **largest** lever available and, in the
frozen form, a **cheaper** one than what is deployed. So it goes first.

### R1 — Evaluate a frozen DINOv2 arm under the real fold protocol. **Do this first; ~1 hour.**

The probe is suggestive, not admissible: it trains on all photos of each training rig instead of
the 70% split, it does not exclude cam_iphone's absent class from the macro, and it scores
patch-wise without photo pooling. All three are fixable, and none of them require the VM.

1. Re-run with `split="train"` on the training rigs, so the photo budget matches a real fold.
2. Use `compute_split_metrics`' macro handling so the iphone fold is a true 9-class average.
3. Add photo-level pooling (`photo_pooling_eval.py`) so the number is comparable to what the
   webapp actually returns.
4. Sweep seeds — free here, since there is no training. This is the first lever in the project's
   history that can be given a *many*-seed answer in an afternoon rather than a 6-seed answer in
   a week.

If it survives: it is simultaneously more accurate cross-rig and **~2.6x cheaper to serve** than
the deployed model, and it makes the whole experimental loop minutes instead of days. If it does
not survive, you have spent an hour and learned where the probe was lying.

Then, and only then, the two follow-ups: refit the OOD guard in the new embedding space against
the same negatives (including the green legumes that beat probe v1), and test the fine-tuned
middle ground (DINOv2 partially unfrozen), which nothing has measured.

### R2 — Distil MobileSAM into a U-Net-R18 bean-density model (Role B).

Unchanged from the original draft, and unaffected by R1: the pitch estimator feeds patch sizing
for *any* backbone. Its teacher already exists and has already been measured on this project's own
ground truth — MobileSAM beats the shipped FFT estimator (11.1% vs 19.2% per-image MAPE, bias
spread 1.02 vs 1.04) and is unusable only at 97 s/image.

1. Generate MobileSAM masks over all 938 cropped photos (~25 CPU-hours, offline, parallel,
   `dvc add` the result). QA a stratified sample at full resolution, not thumbnails.
2. Train U-Net-R18 to regress bean-centre density (integral = count).
3. Benchmark the student against teacher and incumbent through
   `analysis/bean_scale/benchmark.py`, paired, on the same crops.
4. Only if it holds teacher-level MAPE, validate through `run_folds.py`'s cross-rig metric —
   changing the estimator changes patch sizing on both sides, so it is a config change and is
   never adopted on an offline benchmark alone.

The hard constraint from `bean_scale.py` is what makes this legitimate: train and inference must
run the *same* estimator. A distilled student qualifies because it is fast enough for both sides;
MobileSAM itself never was.

### R3 — U-Net-R18 multi-task classifier (Role C). **The literal U-Net question — now third.**

Shared ResNet18 encoder, U-Net decoder supervised by R2's SAM masks, classification head
unchanged, `loss = CE + lambda * dice`; delete the decoder at inference. Measured cost: **1.4x
training, 1.0x serving.**

It is demoted, not dismissed. The honest reason: the probe says the dominant term is *what the
encoder was pretrained on*, and an auxiliary decoder is a second-order regulariser on top of that.
If R1 succeeds, the ResNet18 trunk this modifies may not be the thing serving. If R1 fails, this
becomes the best remaining idea, because it is the only one that buys a new training signal at
zero serving cost and composes with MixStyle without re-wiring it.

### R4 — Modern-recipe ResNet18 weights. **CLOSED — measured, negative.**

This was in the draft as the cheapest possible win: same architecture, same cost, better weights.
§5b measured it and it does not work — `resnet18.a1_in1k` scores 0.4496 against torchvision V1's
0.4645. No change to make. Recorded here so it is not re-proposed.

### What NOT to do

- **Do not port detectron2.** Its zoo times are GPU times; Mask R-CNN at 800x1333 on this CPU is
  not close. What it would buy (per-bean instances) SAM already gives offline.
- **Do not fine-tune ResNet50 or ConvNeXt-T.** At 3.5x per step, one seed of a 4-fold screen is
  4-5 days against a 0.029 seed sd. Their frozen features are interesting; their fine-tuned arms
  are unvalidatable here. If you want their features, distil them.
- **Do not swap `model_name` and run folds** without porting MixStyle (§6) — and note
  `run_folds.set_fold()` has no `model_name` parameter at all today. Adding one must **inherit**
  when omitted, never default to a hardcoded architecture.
- **`powervpsssh` was off-limits while Stage 1 (exp210-213) ran.** Lifted 2026-09-24: Stage 1
  finished on 2026-09-21 and the scheduler task is closed (`docs/lr_scheduler_plan.md` §6).
  Everything in R1 and R4 runs on the workstation anyway.
- This document and `analysis/architecture_screen/` were committed on 2026-09-24, so they no
  longer trip `run_folds.py`'s clean-tree launch gate.

### The uncomfortable one, revised

The original draft closed by arguing that with 938 photos across 4 cameras, camera diversity —
not architecture — was probably the binding constraint. The probes are evidence against that being
the *whole* story: the same data, the same patches and the same linear classifier span 0.4763 to
0.8511 purely on choice of frozen encoder. A 0.37 spread attributable to the pretrained
representation alone is not what a data-starved regime looks like — and §5b shows it is not
reachable by any change to the architecture or recipe you control.

More cameras are still worth collecting, and the cross-rig mean is still an average of three
numbers. But "we need more data before architecture matters" is now the weaker reading.