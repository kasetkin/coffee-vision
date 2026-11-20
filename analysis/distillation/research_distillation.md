# Distillation in the inference path: methods, student architectures, and what has to be measured

Research note, written 2026-11-15 on the local checkout at `68a86ab`. A reading task: nothing was trained, converted or
timed for it. It proposes changes to ADR 0006 and ADR 0015 and says so where it does (§3.9, §4).

**Closed, 2026-11-20.** The owner closed distillation of the classifier backbone ([ML-32](../../docs/ml-0032_ticket_distil_dinov3_mobilenetv4.html) D9, [`.out-of-scope/distillation.md`](../../.out-of-scope/distillation.md)): no training hardware here and no published student. This note stays as the record of what was read.

How each claim is marked:

- **[src]**: read here in source code or a data file at a named tag or commit (downloaded raw, or the installed
  package).
- **[doc]**: read in an official doc page, a paper's own page (arXiv abstract or arXiv HTML full text) or a first-party
  model card or README. Papers were downloaded as arXiv HTML and searched as text; table columns are as that text
  gave them.
- **[repo]**: a project record; the path is given.
- **[sec]**: reported by a secondary source; not followed to its owner.
- **Unverified**: could not be checked here. All of these are collected in §5.3.

Versions: `facebookresearch/dinov3` at `6876159a`; `huggingface/pytorch-image-models` (timm) at `43b549bf` for the
result CSVs and the distillation tasks; timm 1.0.30 and `litert-torch` 0.9.4 as installed in `convert/.venv`;
`facebookresearch/sam2` at `2b90b9f5`; `mit-han-lab/efficientvit` at `de7d7733`. Licence fields are as the Hugging Face
and GitHub APIs returned them on this date.

**The owner's goal** (2026-11-15): "weight does not matter at all, the main goal is speed boost". Students are ranked
by latency at kept accuracy on the serving hardware. Parameter count and file size are not criteria, and a larger
student that runs faster is welcome.

## 1. Question and answer

**Question.** How can the two models on the inference path (the frozen DINOv3 ViT-B/16 classifier backbone and the
EfficientViT-SAM-XL0 segmenter) be distilled, and which student architectures are candidates when the goal is speed
at kept accuracy?

**Answer.** For the classifier, the method the sources support is **offline embedding distillation on this project's
own patches**: the student is trained to reproduce the frozen B/16's `cls_mean` readout (cosine or MSE through a
linear projector), teacher and student seeing the same augmented view, over a long schedule, on patches cut from
training-split photos plus the guard's training negatives; then the logistic-regression head and the OOD probe are
refitted on the student's embeddings and the student is compared with B/16 by paired seeds (ADR 0002). In-domain
transfer data is what works best for a specialist [S9], task-oriented transfer beats generic distillation for small
students [S11], and a method that preserves the teacher's embedding geometry is what keeps OOD detection [S12]; the
guard reads the embedding, so the negatives must be in the transfer set. The student shortlist, in the order to try:
**(1) DINOv3 ViT-S/16**, already on disk, the same graph ops that already place on `yappy`'s GPU, 2.7× (RTX 4090) to
3.2× (laptop CPU) the throughput of ViT-B/16 for a plain ViT [S6], and the only student with a measured accuracy
here (−0.035 cross-camera patch macro-F1 against B/16, in-distribution test 0.939 against 0.955 [R6]); distillation
on bean patches is the attempt to close that gap. **(2) MobileNetV4-Conv-M or -Conv-L (or Hybrid-M)**, the only
family whose paper targets and measures CPU, mobile-GPU and accelerator latency together [S13]; 5× (GPU) to 10× (CPU)
ViT-B/16's throughput in timm's tables at 224 px [S6]. **(3) DINOv3 ConvNeXt-Tiny**, Meta's own convolutional student
of the same teacher family [S1, S2]; about ViT-S speed on a CPU, slower than ViT-S on a desktop GPU [S6].
**(4) LeViT-256 or TinyViT-21M**, both built to be distilled into [S16, S8]. For the segmenter, the first step is not
a distillation: **(1) EfficientViT-SAM-L0 at 512 px with the decoder fine-tuned as today**, on disk, 2.7× XL0 in the
authors' TensorRT and Jetson figures [S20, S21] and 4.9× in this project's own CPU timing of the encoder [R9];
**(2) L0's encoder distilled from XL0-v7r's image embedding** (L2, as EfficientViT-SAM and MobileSAM were made
[S20, S22]) if plain L0 loses mask quality; **(3) a plain binary segmentation network at 512 px** (a MobileNetV4,
RepViT or EfficientViT-B encoder with a light head) trained on the owner-judged labels plus XL0-v7r's masks on
unlabelled photos, which amends ADR 0015. **No source gives a speed for any of these on `yappy`'s WebGPU delegate.**
Every ratio above is from other hardware; MACs do not predict latency [S13, S14, S15, S6], and the first thing to do
is to convert the untrained candidates and time them on `yappy` and on XNNPACK before training anything (§5.1).

## 2. What the project already knows

### 2.1 Decisions and rules a student touches

- **The classifier** [repo: `docs/adr/0006-frozen-dinov3-b16-head.md`]: "DINOv3 ViT-B/16, frozen (no fine-tuning),
  read out as `cls_mean` (1536-d), at 224 px only and without test-time augmentation"; C fixed at 0.1; "ViT-S/16 is
  only a fallback". The gain over ResNet18 "came from DINO's self-supervised pretraining, not from architecture or
  recipe".
- **The guard** [repo: `docs/adr/0016-ood-guard-probe-at-fixed-half.md`]: a logistic regression on the classifier's
  frozen patch embeddings, "one file beside its checkpoint ... that must match it", refusing at mean P(not beans) >
  0.5; fitted on the segmenter dataset's training and validation negatives, measured on its test split.
- **The segmenter** [repo: `docs/adr/0015-xl0-segmenter-owner-judged-labels.md`]: "XL0 at 1024 px, the owner's
  choice, adopted without a comparison against L0"; "there is no latency limit"; "The owner judges every label ...
  every training and validation label and every test mask"; only the mask decoder is fine-tuned; the redraw loop
  depends on point prompts; the photo is cropped "by the same code in training and serving".
- **Evidence** [repo: `docs/adr/0002-paired-multi-seed-evidence.md`]: seeds 42, 123 and 7 against the baseline at the
  same seeds; no automatic rule; the owner decides.
- **Serving** [repo: `docs/adr/0019-serving-runs-litert-accelerator-per-host.md`]: LiteRT; on `yappy` the WebGPU
  accelerator, segmenter fp32, DINOv3 fp16 at batch 8. "The accelerator, and each graph's precision, are chosen for
  the host serving them, measured there against PyTorch."
- **D7** [repo: `docs/adr/0020-converted-models-pass-d7-before-they-serve.md`]: a converted model "is checked against
  the PyTorch fp32 model it was converted from". The fp16 bar was accepted for DINOv3 on `yappy`'s GPU. Every op must
  be proven on the accelerator by the delegate's own report.
- **Exact forms** [repo: `docs/adr/0022-export-forms-are-exact.md`]: "No clamp, approximation or retraining is used
  to make a runtime fit"; "A new model, a new checkpoint of a different architecture, or a new runtime version reruns
  the range scan and the forms' checks before conversion."
- **Parity** [repo: `CODING_STANDARDS.md` line 7]: "An accurate method offline with a cheap one live is rejected,
  because it inflates every reported metric."
- **Closed ideas** [repo: `.out-of-scope/`]: `resnet18-recipe-levers.md` closes the ResNet18 recipe screens (mixup,
  frozen backbone, schedulers) and says "A lever is worth trying again only on the DINOv3 pipeline, as a paired
  multi-seed screen". `cross-camera-evaluation.md` retires the cross-camera folds; a photo-level metric and a
  seed-noise measurement are "deferred by the owner, not rejected; propose them only when asked". Nothing there
  closes distillation. `GLOSSARY.md` has no distillation term.

### 2.2 Prior thinking on distillation

- **DINOv3 as a teacher for ResNet18** [repo: `docs/dinov3_integration_plan.md` §13]: a fallback for the case where
  DINOv3 could not serve: "Add a loss pulling its 512-d embedding toward a projection of DINOv3's embedding on the
  same patch". Never run. Its licence remark: "No clause prohibits distillation, and there is no restriction on
  outputs. But whether a student trained against the teacher's outputs is a 'derivative work' matters only if the
  student is published."
- **SAM into a U-Net-R18** [repo: `docs/architecture_screen_plan.md` §8 R2]: a bean-density student of MobileSAM for
  the pitch estimator, "a slow teacher, a fast student, and an offline budget to generate labels in". Never run. It
  names the constraint this note keeps: "train and inference must run the same estimator. A distilled student
  qualifies because it is fast enough for both sides".

### 2.3 Numbers this note leans on

| Fact | Value | Source |
|---|---|---|
| A 12 MP photo on `yappy`, about 10 s | DINOv3 5.74 s (40 patches, 5 calls of 8), XL0 encoder 1.78 s, photo decoding 1.4-1.5 s, mask decoder about 0.1 s | `analysis/ml7_gpu/README.md` §2; `analysis/quantization/research_quantization.md` §2.4, §5.4 |
| Pi 5 CPU | DINOv3 9.4-9.8 s, encoder 4.17 s | same note §2.4 (OPS-23) |
| DINOv3 graph ops | 48 FULLY_CONNECTED, 24 BATCH_MATMUL, 12 GELU, 1 CONV_2D | `analysis/ml7_gpu/manifest_tflite.json` |
| Encoder graph ops | 65 CONV_2D, 14 DEPTHWISE_CONV_2D, 12 BATCH_MATMUL, 7 DIV, 30 GELU | same |
| The GPU against the CPU on `yappy`, fp32 | encoder 1.78 s against 6.91 s on XNNPACK (3.9×); DINOv3 at batch 8 11.17 s against 33.62 s (3.0×) | `analysis/ml7_gpu/README.md` §2 |
| B/16 against S/16, cross-camera folds (retired metric) | +0.035 patch macro-F1, 11 of 12 pairs; S+/16 null against S/16 | `EXPERIMENTS_LOG.md`, "DINOv3 Experiment 1" |
| B/16 against S/16, in-distribution test | 0.955 against 0.939 | same |
| Live model, in-distribution | val 0.9470-0.9649, test 0.9500-0.9731 over seeds 42/123/7 | `docs/status.md` |
| L0 against XL0, encoder, devcontainer CPU at 4 threads, one 12 MP photo | 474 ms (512 px) against 2,336 ms (1024 px), 4.9× | `analysis/ml5_xl0_timing/timing_devcontainer_{l0,xl0}_t4.json` |
| On disk | DINOv3 ViT-S/16, S+/16, B/16; EfficientViT-SAM L0, L1, L2, XL0, XL1 | `models_pretrained/` |
| Data | 1,681 labelled photos, 10 classes, 600/160/160 patches per class, 8 augmented views | `docs/status.md` |

One observation from the third row: on `yappy` the convolution-heavy encoder gained more from the GPU (3.9×) than the
transformer did (3.0×). That is one pair of graphs, not a rule; it is the only local evidence on how convolutions and
attention compare on this delegate.

## 3. Findings

### 3.1 How Meta distilled DINOv3

- **The objective is the pretraining objective with a frozen teacher.** "Our distillation approach uses the same
  training objective as in the first training phase ... instead of relying on an exponential moving average (EMA) of
  model weights, we use the 7B model directly as the teacher ... the teacher model is fixed. We do not observe
  patch-level consistency issues and therefore do not apply the Gram anchoring technique." **[doc]** [S1 §5.2] The
  model card lists that objective: DINO loss with multi-crop, iBOT masked-image loss, KoLeo on CLS tokens. **[doc]**
  [S2]
- **Length.** "We train the models for 1M iterations then perform 250k iterations of learning-rate cooldown" and then
  the high-resolution phase. **[doc]** [S1 §5.2]
- **Multi-student.** One teacher forward is shared by several students training in parallel on separate GPU groups.
  **[doc]** [S1 §5.2]
- **The code is released; the recipe is not runnable here.** The repo holds
  `dinov3/configs/train/dinov3_vitl16_lvd1689m_distilled.yaml` (`MultiDistillationMetaArch`, DINO loss weight 1.0,
  KoLeo 0.1, global crops 256, local crops 112, 8 local crops, global batch 1920) and
  `dinov3/configs/train/distillation_convnext/convnext_{tiny,small,base,large}_p16.yaml` (500 epochs, iBOT head
  separate). **[src]** [S3] The README's news line: "Distillation code and configurations for ConvNeXt backbones are
  now released!" (2025-11-20), and it documents only a "test setup" (`multi_distillation_test.yaml`, a ViT-L teacher
  on ImageNet, students ViT-S and ViT-B). **[doc]** [S3] The teacher in the real configs is the ViT-7B and the data
  is LVD-1689M, which is not public. This is a recipe for making a general backbone on a cluster, not for one
  consumer GPU and 1.7k photos. Whether the code runs with a ViT-B teacher on a small dataset was not tried
  (Unverified, §5.3).
- **What the distilled students cost and score** **[doc]** [S1 Fig. 16(a), Tab. 14, Tab. 15; S2]:

| Model | Params | GFLOPs at 256 px | GFLOPs at 512 px | IN-ReaL (linear) | IN-R | ObjectNet | ADE20k mIoU |
|---|---|---|---|---|---|---|---|
| ViT-S/16 | 21M | 12 | 63 | 87.0 | 60.4 | 50.9 | 47.0 |
| ViT-S+/16 | 29M | 16 | 79 | 88.0 | 68.8 | 54.6 | 48.8 |
| ViT-B/16 (served) | 86M | 47 | 216 | 89.3 | 76.7 | 64.1 | 51.8 |
| ConvNeXt-Tiny | 29M | 5 | 20 | 86.6 at 256, 87.7 at 512 | 73.7 at 256 | 52.6 at 256 | 42.7 |
| ConvNeXt-Small | 50M | 11 | 46 | 87.9 at 256, 88.7 at 512 | 73.7 at 256 | 52.6 at 256 | 44.8 |
| ConvNeXt-Base | 89M | 20 | 81 | 88.5 at 256, 89.2 at 512 | 77.2 at 256 | 56.2 at 256 | 46.3 |

  The ViT rows are evaluated at 512 px (Tab. 14's protocol), the ConvNeXt rows at the resolution shown, so the two
  blocks are not a like-for-like comparison. The Tiny and Small ConvNeXt rows carry identical IN-R and ObjectNet
  figures in both the paper's Table 15 and the model card; that looks like a copying error upstream and is repeated
  here as printed.
- **What the paper says about ConvNeXt students' speed**: only that they "are highly efficient in terms of FLOPs and
  are well-suited for deployment on devices optimized for convolutional computations", and that convolutional nets
  quantize better. **[doc]** [S1 §7.2] It reports FLOPs, not latency. A ViT-7B to ConvNeXt distillation "bridges two
  fundamentally different architectures ... ConvNeXt relies on convolutional operations without a CLS token".
  **[doc]** [S1 §7.2] How the ConvNeXt student's global feature is formed in the config was not read (Unverified).
- **Measured throughput of the same architectures in timm** (plain ViT and ConvNeXt, not the DINOv3 variants, which
  add RoPE and 4 register tokens) **[src]** [S6]: at 224 px on an RTX 4090 (AMP, channels-last, batch 512)
  ViT-B/16 2,843 img/s, ViT-S/16 7,696, ConvNeXt-Tiny 4,845; on an i7-12700H CPU (fp32, batch 1) ViT-B/16 8.33,
  ViT-S/16 26.71, ConvNeXt-Tiny 26.62. So ConvNeXt-Tiny has 0.37× ViT-S's FLOPs at equal resolution but is not faster
  than ViT-S on either device. The DINOv3 ViTs themselves at 256 px on an RTX 5090: B/16 1,921 img/s, S+/16 3,986,
  S/16 4,738 (2.47× B/16). **[src]** [S6]

### 3.2 Embedding distillation with a frozen teacher and a small domain dataset

What each method needs and whether it fits about 1.7k photos, unlimited patches from them, and one consumer GPU:

| Method | What is matched | Data and compute in its source | Fit here |
|---|---|---|---|
| Hinton KD [S10] | softened class probabilities | any labelled or unlabelled set | fits: the teacher's "logits" are the fitted head's output on B/16 embeddings. Constrains only the 10-class direction of the embedding, so it says nothing about the guard |
| Function matching, Beyer et al. [S9] | teacher output, same view for both, mixup, very long schedules | Flowers102, Pets, Food101, SUN397, ImageNet; ResNet-50 at 82.8% after 9,600 ImageNet epochs | fits, and it is the recipe written for small datasets. Costs teacher forwards on every augmented view unless views are cached by seed |
| TinyViT's stored logits [S8] | sparsified teacher logits saved to disk with the augmentation's random seed (4 bytes) | ImageNet-21k pretraining | the caching idea fits: store B/16's embedding per (patch, augmentation seed), then train any number of students without the teacher |
| Proteus [S7] | CLS token, patch tokens and feature map, MSE through LayerNorm + linear projectors, no labels | ImageNet-1K, 300 epochs; ViT-S/14 from DINOv2-B reaches 81.8% | the losses fit; its data scale does not. Its point is a general student from a generic proxy set |
| CosPress [S12] | a learned map of the teacher's space into the student's that preserves cosine similarities | ImageNet and small target sets ("CosPress finetuning" of specialists) | fits; the only source read here that measures OOD detection of the student |
| Task-oriented transfer, Vemulapalli et al. [S11] | the adapted teacher's task predictions on an unlabelled transfer set, then fine-tune | FastViT-S12 and MobileViT-V2 students from DINOv2 ViT-L/14 and OpenCLIP; CC3M or a task-related transfer set | fits: the project's "adapted teacher" already exists (B/16 plus the fitted head) |
| DeiT distillation token [S17] | a second token trained on the teacher's label | ImageNet | no: it needs a ViT student with an extra token and a classification teacher; nothing here needs it |
| Relational KD [S18] | distances and angles between examples | small benchmarks | optional extra term; CosPress covers the same intent with OOD evidence |
| DINOv2's and DINOv3's own distillation [S4, S1, S3] | DINO + iBOT + KoLeo against a frozen teacher's heads | 1M iterations on LVD-1689M | no (§3.1) |
| AM-RADIO, RADIOv2.5, SigLino, Theia [S25] | several foundation teachers into one student | web-scale sets, multi-GPU | no: agglomeration answers a different question (one student for many teachers) |
| LightlyTrain distillation [S26] | DINOv2 or DINOv3 teacher (ViT-S to 7B, ConvNeXt) into any student; "v1" a global pseudo-label loss over a queue of teacher embeddings, "v2" MSE on spatial features | its docs: epochs "between 100 and 3000"; for "<100k images ... up to 10000 epochs" | a packaged route; its licence was not read (Unverified) |

Findings that decide the recipe:

- **Same view for teacher and student, and patience.** "teacher and student should process the exact same input image
  views"; distilling against precomputed targets for a fixed view "works much worse"; fixed-teacher settings "achieve
  significantly lower distillation loss" yet "lead to students which do not generalize well". **[doc]** [S9] This
  bears directly on caching: one cached B/16 embedding per patch, with the student seeing augmented views, is the
  setting the paper warns against. The cache has to hold one embedding per (patch, view), as TinyViT does with seeds
  [S8]; the project's 8 stored augmented views per patch are a small version of that.
- **In-domain transfer data is best for a specialist.** "distilling using in-domain data works the best ... even if
  the images are completely unrelated, distillation still works to some extent, though results get worse"; related
  generic data can come close "but extra long optimization schedule may be required". **[doc]** [S9 §3.8]
- **Task-oriented beats task-agnostic for a small student.** Task-oriented transfer "outperforms task-agnostic VFM
  distillation, web-scale CLIP pretraining, supervised ImageNet pretraining, and self-supervised DINO pretraining by
  up to 11.6%, 22.1%, 13.7%, and 29.8%", at "up to 9x" lower pretraining compute than task-agnostic distillation;
  "small models may not be able to inherit this vast knowledge. Hence, transferring only task-oriented knowledge
  works better"; "the dataset used for transferring knowledge has a significant effect". **[doc]** [S11]
- **Generic distillation is possible at ImageNet scale, and diversity matters.** Proteus distils "on ImageNet-1K
  without access to the original training data"; it removes one-hot labels and the classification projection head
  because they "lead to dataset bias"; a more diverse proxy set "brings almost 1% average improvement on fine-grained
  classification"; with a single source image "6 out of 12 datasets suffer from performance drop over 10%". **[doc]**
  [S7] That last figure is the nearest published analogue of distilling on one narrow domain and then asking for
  general features: it degrades.
- **OOD behaviour does not transfer by default.** Feature-distilled students "fall short in matching critical
  properties such as robustness and out-of-distribution (OOD) detection performance"; on the OpenOOD ImageNet-1K
  benchmark "Proteus performs very poorly ... with worse performance observed for larger student models. In contrast,
  CosPress is able to distill models that have strong OOD performance". **[doc]** [S12] The benchmark is
  post-hoc OOD scoring on ImageNet, not a linear probe fitted on negatives as in ADR 0016, so it transfers to this
  project as a warning, not as a number.
- **Robust teachers give robust students when augmentation is part of the distillation.** **[doc]** [S19, abstract
  only]
- **A 2026 instance of the same pattern**: a DINOv3 ViT adapted to the task and used as "a task-specialized teacher
  for compact student backbones" (EdgeCrafter); and a livestock pipeline that distils SAM 3's encoder into a
  TinyViT-21M student while simply adopting the published DINOv3 ViT-S as its embedder instead of distilling one.
  **[doc]** [S27, abstracts only]

### 3.3 Generic student or specialist

- **A specialist is what the evidence supports for this data budget.** [S9] and [S11] both put in-domain or
  task-related transfer data first, and the general recipes ([S1], [S7]) need ImageNet scale or more.
- **The risk of a specialist is the guard.** A student trained only on bean patches is constrained only where beans
  are. The probe separates bean patches from non-bean patches in embedding space (ADR 0016); on non-bean inputs the
  student's embedding is unconstrained unless such inputs are in the transfer set. No source measures this exact
  case. What follows from [S7] (single-image proxy degrades other domains) and [S12] (OOD detection is the property
  feature distillation loses) is a requirement, not a result: **the transfer set must include the negatives** the
  probe is fitted on (the segmenter dataset's training and validation negatives, 584 true negatives in `dataset/`
  [repo: ADR 0022]), and the guard's test-split figures (negatives caught, positives refused) are part of the
  student's acceptance, beside macro-F1.
- **Two things keep a specialist general for free.** (a) Starting from pretrained weights: every shortlisted student
  has ImageNet or DINOv3 weights, so distillation on patches is a fine-tune toward the teacher, not training from
  scratch. (b) Adding a generic unlabelled set to the transfer data: [S11] shows a generic web set already beats
  ImageNet pretraining for a small student, and [S9] that related generic data approaches in-domain data. Whether a
  generic admixture is needed here is an experiment (§5.1 step 5).
- **Leakage.** Patches for distillation come from training-split photos only. A student that has matched the teacher
  on validation or test photos has seen them, and the paired comparison under ADR 0002 would be contaminated. The
  same holds for the guard's test split.

### 3.4 What distillation has to beat

- **The published DINOv3 ViT-S/16, undistilled by this project.** On disk; the same ops as the served graph
  (FULLY_CONNECTED, BATCH_MATMUL, GELU), so the export forms and the fp16 scales carry over in kind, though ADR 0022
  requires the range scan again. 12 against 47 GFLOPs at 256 px [S1]; 2.47× B/16's throughput as DINOv3 models on an
  RTX 5090, 2.7× (GPU) and 3.2× (CPU) as plain ViTs at 224 px [S6]. Accuracy here is already measured: −0.035 on the
  retired cross-camera metric, 0.939 against 0.955 in-distribution test [R6]. ADR 0006 names it the fallback.
- **The published DINOv3 ConvNeXt-Tiny or -Small.** Not on disk; under the DINOv3 License [S28]. 5 and 11 GFLOPs at
  256 px [S1]. No latency advantage over ViT-S in timm's tables (§3.1).
- **Token merging (ToMe) on the existing B/16.** "increase the throughput of existing ViT models without needing to
  train"; "Off-the-shelf, ToMe can 2× the throughput of state-of-the-art ViT-L @ 512 and ViT-H @ 518". **[doc]** [S24]
  Its published figures are for large models at high resolution, not B/16 at 224 px with 196 tokens. Its matching
  step uses `argsort` and `gather` [S24, appendix code], which are not in the GPU delegate's documented op list
  [S29]; whether the WebGPU accelerator takes them is Unverified. It changes the embeddings, so the head and probe
  would be refitted, and it is an approximation of the model made for speed, which ADR 0022 would record as a
  departure.
- **Fewer patches or a lower resolution with the same B/16.** Time is linear in patches (5 calls of 8). ADR 0006
  fixes 224 px; ADR 0019 records that patch sampling alone changes 0.85% of top-1 answers between seeds at 40
  patches, so fewer patches raises that noise. Both are recipe changes under ADR 0002, not distillations, and both
  combine with any student.

### 3.5 Student architectures for the classifier

**MACs do not rank latency, and a GPU delegate makes it worse.**

- "the direct metric, e.g., speed, also depends on the other factors such as memory access cost and platform
  characterics ... this work proposes to evaluate the direct metric on the target platform". **[doc]** [S14]
- "these metrics may not correlate well with latency of the network when deployed on a mobile device". **[doc]** [S15]
- "The common practice of using a model's total MACs to proxy latency is the same as targeting a roofline model with
  a Ridge Point (RP)=0", that is, hardware with infinite memory bandwidth. **[doc]** [S13]
- In MobileNetV4's own table, FastViT-SA12 (1.9 G MACs) takes 52.1 ms on a Samsung S23 GPU and 79.6 ms on a Pixel 7
  GPU, against MNv4-Conv-M (1.0 G) at 4.1 and 18.1 ms; ConvNeXt-S fails to run on the Pixel 4's Hexagon DSP and so do
  FastViT and both MNv4-Hybrid models. "FastViT ... is 2nd on CoreML but has >5x the latency of MNv4 on S23 GPU."
  "Due to PyTorch-to-TFLite export tool limitations, EfficientViTs are not benchmarked on CPUs and EdgeTPU." **[doc]**
  [S13 Tab. 6]
- In timm's tables EfficientFormerV2-S2 (1.27 G MACs) runs 809 img/s on an RTX 4090 where ViT-B/16 (17.58 G) runs
  2,843; on the i7 CPU the order reverses (41.6 against 8.3 img/s). **[src]** [S6]
- The LiteRT GPU delegate page: when an op is unsupported "the framework will only run a part of the graph on the GPU
  and the remaining part on the CPU", which "often results in slower performance than when the whole network is run
  on the CPU alone"; "Reshape operations are particularly expensive to run"; tensors are "sliced into 4-channels".
  Its op list includes CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, ADD, MUL, RELU, RELU6, PRELU, pooling,
  RESIZE_BILINEAR, RESHAPE, STRIDED_SLICE, SOFTMAX; it does not list GELU, HARD_SWISH or BATCH_MATMUL. **[doc]** [S29]
  That page describes the TFLite GPU delegate; `yappy` runs the LiteRT WebGPU accelerator, a prebuilt library that
  demonstrably takes BATCH_MATMUL and GELU [R4]. Its op coverage for hard-swish, squeeze-excite's global pooling and
  sigmoid, LayerNorm over channels, and large-kernel depthwise convolutions is Unverified.

**So the only decisive number is a measurement on `yappy`** (WebGPU, fp32 and fp16) **and on XNNPACK** (the CPU host
and the Pi 5), per candidate, with the delegate's placement report.

Candidates. Top-1 is the best ImageNet-1k figure in timm's `results-imagenet.csv` for that architecture, at the
resolution shown; throughput is timm's benchmark at 224 px unless marked, RTX 4090 (AMP, channels-last, batch 512) and
i7-12700H (fp32, batch 1); "× B" is the ratio to ViT-B/16 on the same device **[src]** [S6]. Mobile latencies are from
the MobileNetV4 paper **[doc]** [S13]. Licences are of the timm-hosted weights **[doc]** [S28].

| Candidate | ImageNet top-1 | GMACs at 224 | RTX 4090 img/s (× B) | i7 CPU img/s (× B) | Other reported latency | Ops; delegate risk | Weights licence | Trained with KD upstream |
|---|---|---|---|---|---|---|---|---|
| ViT-B/16 (plain; reference) | 85.1 | 17.58 | 2,843 (1.0) | 8.33 (1.0) | | as the served graph | | |
| **DINOv3 ViT-S/16** | IN-ReaL 87.0 (linear) [S1] | 4.61 (plain S/16) | 7,696 (2.7) | 26.71 (3.2) | | the served graph's ops; lowest risk | DINOv3 License | yes, from ViT-7B [S1] |
| ViT-Ti/16 | 75.5 | 1.26 | 17,590 (6.2) | 77.57 (9.3) | | as above | Apache-2.0 | |
| **MobileNetV4-Conv-M** | 82.8 at 384 (in12k); 79.9 at 256 [S13] | 0.84 | 14,222 (5.0) | 84.5 (10.1) | Pixel 6 CPU 11.4 ms, S23 GPU 4.1 ms, Pixel 7 GPU 18.1 ms | conv, depthwise, ReLU family; designed for delegates; lowest conv risk | Apache-2.0 | paper has a KD recipe [S13] |
| **MobileNetV4-Conv-L** | 83.0 at 384; 82.9 [S13] | 2.86 at 256 | 6,068 at 256 (about 2.1 at 256) | 26.05 at 256 | Pixel 6 CPU 59.9 ms, S23 GPU 13.2 ms | as Conv-M | Apache-2.0 | 85.9% as a distilled student [S13] |
| MobileNetV4-Hybrid-M | 83.0 at 384; 80.7 [S13] | 0.98 | 10,180 (3.6) | 62.29 (7.5) | Pixel 6 CPU 14.3 ms, S23 GPU 5.9 ms; failed on Hexagon | adds Mobile MQA attention (BATCH_MATMUL class) | Apache-2.0 | as above |
| **DINOv3 ConvNeXt-Tiny** | IN-ReaL 86.6 at 256 [S1] | 4.47 | 4,845 (1.7) | 26.62 (3.2) | | 7×7 depthwise, LayerNorm, GELU; Unverified on the delegate | DINOv3 License | yes, from ViT-7B [S1] |
| ConvNeXt V2-Nano | 83.4 at 384 | 2.46 | 3,061 (1.1) | 33.96 (4.1) | | adds GRN (a norm over space); Unverified | CC-BY-NC-4.0 | |
| **LeViT-256** | 81.5 | 1.13 | 14,077 (5.0) | 77.98 (9.4) | | conv stem + attention with hard-swish; attention bias table | Apache-2.0 | yes, distillation head [S16] |
| LeViT-384 | 82.6 | 2.36 | 8,827 (3.1) | 42.25 (5.1) | | as above | Apache-2.0 | yes |
| **TinyViT-21M** | 85.1 | 4.29 | 4,303 (1.5) | 22.73 (2.7) | | MBConv + window attention; Unverified | Apache-2.0 | yes, stored-logit distillation from IN-21k teachers [S8] |
| TinyViT-11M | 83.2 | 2.04 | 6,678 (2.3) | 36.53 (4.4) | | as above | Apache-2.0 | yes |
| RepViT-M1.5 | 82.5 | 2.31 | 4,237 (1.5) | 33.68 (4.0) | | pure conv after fusing; squeeze-excite; GELU | Apache-2.0 | yes ("dist" weights) |
| RepViT-M2.3 | 83.7 | 4.57 | 2,407 (0.85) | 19.97 (2.4) | | as above | Apache-2.0 | yes |
| EfficientViT-B2 (MIT) | 83.1 at 288 | 1.60 | 4,171 (1.5) | 43.16 (5.2) | B1-r224: S23 GPU 5.0 ms, iPhone CoreML 2.4 ms [S13] | the segmenter's op family (LiteMLA with DIV), already converted here with exact forms | Apache-2.0 | |
| EfficientViT-L1 (MIT) | 84.5 | 5.27 | 3,794 (1.3) | 19.21 (2.3) | | as above | Apache-2.0 | |
| FastViT-SA12 | 81.8 at 256 | 1.96 at 256 | 2,725 at 256 | 33.64 at 256 | S23 GPU 52.1 ms, Pixel 7 GPU 79.6 ms, Pixel 6 CPU 86.5 ms [S13] | poor on Android GPUs in [S13] | "other" (Apple; not read) | yes ("dist" weights) |
| EfficientFormerV2-S2 | 82.2 | 1.27 | 809 (0.28) | 41.63 (5.0) | | slow on a desktop GPU despite low MACs | Apache-2.0 | yes ("dist" weights) |
| RegNetY-032 | 81.9 | 3.20 | 3,891 (1.4) | 36.05 (4.3) | | conv, group conv, squeeze-excite, ReLU | Apache-2.0 | |
| ResNet-50 (a1) | 80.4 | 4.11 | 5,088 (1.8) | 43.63 at 160 | | conv, ReLU; the safest op set | Apache-2.0 | |
| EfficientNetV2-S | 83.3 at 300 | 5.35 at 300 | 2,590 at 300 | 19.41 at 300 | | SiLU, squeeze-excite | Apache-2.0 | |
| Hiera-Tiny | 82.8 | 4.91 | 4,467 (1.6) | 24.89 (3.0) | | attention with pooling | CC-BY-NC-4.0 | |
| SHViT-S4 | 79.4 at 256 | 0.99 at 256 | RTX 5090: 21,175 at 256 | not in the CPU table | | single-head attention; licence of the code repo not an SPDX one | MIT (timm card) | |
| FasterNet-T2 | 78.7 | 1.91 | RTX 5090: 15,371 | not in the CPU table | | partial convolution (slice + concat) | Apache-2.0 | |

How to read it:

- **ImageNet top-1 is a weak proxy for what matters here.** ADR 0006 records that the gain "came from DINO's
  self-supervised pretraining, not from architecture or recipe", and a frozen ResNet18 scored 0.5225 where frozen
  DINOv3 S/16 scored 0.8487 on the retired folds [R6]. Whether a convolutional student can hold DINOv3's embedding
  for bean patches is the open question distillation is meant to answer; no source answers it. Meta's ConvNeXt
  students show the architecture can carry DINOv3 features in general (Tab. 15), after 500 epochs on LVD-1689M.
- **On a desktop GPU the plain ViT-S is hard to beat at its accuracy**: at 224 px only MobileNetV4-Conv-M and
  -Hybrid-M, LeViT and ViT-Ti are faster than ViT-S/16 in the 4090 column. On a CPU many more are. Which of the two
  patterns `yappy`'s iGPU follows is not known; §2.3's 3.9× against 3.0× suggests convolutions do at least as well
  there as attention.
- **Speed and capacity both favour "larger but shallow and convolutional"** under the owner's criterion:
  MobileNetV4-Conv-L (32.6M parameters) and ResNet-50 are larger than ViT-S and belong on the timing list. Size is
  not a reason to drop anything.
- **Not shortlisted**: FastViT and MobileOne (Apple-targeted; slow on Android GPUs in [S13]; "other" licence),
  EfficientFormerV2 (slow on a desktop GPU), ConvNeXt V2 and Hiera (non-commercial weights), EfficientViT-MSRA,
  SHViT, FasterNet, RepNeXt, LSNet, MobileCLIP encoders (lower accuracy tiers, or no evidence on a TFLite GPU path;
  MobileCLIP2's weights are under Apple's `apple-amlr` licence [S28]). Any of them can join the timing run at no
  training cost.

**Levers specific to this pipeline** (all change the recipe on both sides, so all need ADR 0002 evidence):

- **Lower input resolution for a convolutional student.** A patch is `B × pitch` photo pixels resized to 224 px
  [repo: `coffeecv/infer.py`, `sample_patches`]. Convolutional students have no token grid tied to 16 px; cost falls
  with area (MobileNetV4-Conv-M: 0.84 G at 224, 1.10 G at 256 [S6]). The student can be distilled at 160 or 192 px
  against the teacher's 224 px embedding of the same patch (the teacher's input is not the student's, which departs
  from [S9]'s same-view rule in resolution only).
- **A fully convolutional student run once over the crop**, with the feature map pooled per sampled region instead of
  40 separate forward passes. 40 patches at 224 px are 2.0 MP of input; one pass over the crop at the same pixels
  per bean costs the crop's area, and shares work where patches overlap. Whether that is fewer pixels depends on the
  crop's size in patch units, which is in the data and was not computed here. A pooled region sees context outside
  its borders (the receptive field), which an independently resized patch does not, so the embedding is a different
  function; it must be the function used in training too (parity), and the teacher's per-patch embedding stays the
  target.
- **A larger field per call** (fewer, bigger patches) is the same change seen from the other side: it alters
  `patch_beans_min/max`, the trained range, and is a recipe change, not a student property.
- **Batch size**: on `yappy` batch 40 was not faster than batch 8 for B/16 (13.21 against 11.17 s in fp32 [R4]); a
  small convolutional student may behave differently. Measure with the timing run.

### 3.6 How the SAM family was distilled

| Model | Encoder, input | How it was made | Reported speed and hardware | Licence |
|---|---|---|---|---|
| EfficientViT-SAM L0-L2, XL0-XL1 [S20, S21] | EfficientViT-L at 512 px, -XL at 1024 px | Stage 1: "distilling the image embedding of SAM-ViT-H into EfficientViT. We employ the L2 loss". Stage 2: end-to-end on SA-1B, 2 epochs, batch 256, box or point prompts with equal probability, up to 64 masks per image | A100, TensorRT fp16: L0 762, L1 638, L2 538, XL0 278, XL1 182 img/s; Jetson AGX Orin: L0 8.2 ms, XL0 22.5 ms. COCO mAP 45.7 (L0), 47.5 (XL0), SAM-ViT-H 46.5 | Apache-2.0 |
| MobileSAM [S22] | TinyViT, 1024 px | "decoupled distillation": the encoder alone is trained to match SAM's image embedding; the original decoder is kept. "1% of the SA-1B dataset ... for 8 epochs on a single GPU", "less than a day" | in [S20]'s table: 278 img/s on the A100, COCO mAP 38.7 | Apache-2.0 |
| EdgeSAM [S23] | RepViT (pure CNN), 1024 px | encoder-only distillation, then "prompt-in-the-loop" distillation with the prompt encoder and decoder in the loop | "over 30 FPS on an iPhone 14"; encoder 26 ms there; in [S20]: 449 img/s, mAP 42.1 | S-Lab License 1.0: "non-commercial purpose" |
| RepViT-SAM [S30] | RepViT | MobileSAM's recipe with a RepViT encoder | "nearly 10× faster inference" than MobileSAM (abstract) | Apache-2.0 |
| EfficientSAM [S31] | ViT-T/S | SAMI: masked-image pretraining that reconstructs SAM's encoder features, then fine-tuning on SA-1B | in [S20]: 183 img/s, mAP 44.4 | Apache-2.0 |
| TinySAM [S32] | TinyViT | "full-stage knowledge distillation ... with hard prompt sampling and hard mask weighting" | not read beyond the abstract | Apache-2.0 |
| FastSAM [S33] | YOLO-style instance segmenter | not a distillation: trained "using only 1/50 of the SA-1B dataset" | "50 times higher run-time speed" than SAM (abstract) | not read |
| SAM 2 / SAM 2.1 [S34] | Hiera-T/S/B+/L, 1024 px | trained on SA-V and SA-1B; "more accurate and 6x faster" than SAM on images | `sam2.1_hiera_tiny`: 38.9M, 91.2 FPS on an A100 (the video pipeline, compiled) | Apache-2.0 |
| EdgeTAM, EfficientTAM [S35] | compact encoders for SAM 2 | distil or retrain SAM 2 with lighter encoder and memory | 16 FPS (EdgeTAM) and about 10 FPS (EfficientTAM) on an iPhone 15 Pro Max | EdgeTAM Apache-2.0 |
| SAM 3, EfficientSAM3 [S36] | Perception Encoder backbone; students via "Progressive Hierarchical Distillation", stage 1 "prompt-in-the-loop training on SA-1B" | concept prompts (noun phrases, exemplars), detection and tracking | not read beyond the abstracts | SAM License (custom; not read) |

- **L0 dominates the other compact SAMs on the one table that holds them all**: faster than MobileSAM, EdgeSAM and
  EfficientSAM on the A100 and more accurate than each. **[doc]** [S20 Tab. 1] The paper attributes it to
  "hardware-friendly operators" despite more parameters, which is the owner's criterion exactly.
- **MACs are misleading here too**: XL0 has 5.3× L0's MACs (185 G against 35 G) and 2.7× its latency on both of the
  authors' devices; this project measured 4.9× on a CPU [R9] and ADR 0015 says "about 3.4 times L0's compute".
- **SAM 2, SAM 3 and their students add video memory, concept prompts and detection**, none of which a single
  whole-image box needs. Each would need the decoder fine-tune repeated and a new set of export forms. No source
  shows any of them faster than L0 for one image on a TFLite-class GPU path.
- **Stage-1 distillation is cheap and label-free.** Both EfficientViT-SAM and MobileSAM start by matching the
  teacher's image embedding with a plain L2 loss; MobileSAM stopped there, on one GPU in under a day. **[doc]**
  [S20, S22] The same step applies with the fine-tuned XL0-v7r as teacher and this project's photos as data: the
  student encoder learns XL0's embedding, and the fine-tuned decoder is reused or fine-tuned again. The official
  repo publishes the stage-1 "distilled models" and a training script [S21]; the vendored copy has
  `third_party/efficientvit/applications/efficientvit_sam/train_efficientvit_sam_model.py` [repo]. Whether L0's
  512 px embedding has the same shape as XL0's 1024 px embedding, which a direct L2 loss needs, was not checked in
  the code (Unverified).

### 3.7 The task-specific route for the segmenter

Serving uses one fixed whole-image box and wants one binary mask (ADR 0015). The promptable parts do no work at
serving, but they are also not where the time goes: the decoder is about 0.1 s of 1.9 s [R4]. A plain segmentation
student saves time only through a cheaper encoder and a lower resolution.

- **SAM into a small task-specific segmenter with pseudo-labels is an established pattern**, mostly in medical and
  remote-sensing work: a small U-Net++ trained on few labels and then on SAM-derived pseudo-labels for unlabelled
  images [S37]; a compact panoramic segmenter learning from SAM "without requiring any labeled data" through a
  teacher assistant [S38]; SAM masks selected by CAM cues as pseudo-labels for a standard segmenter [S39];
  fine-tuning the foundation model on the task first and then distilling into a small model, "when training data is
  limited" [S40]; encoder and decoder distillation of SAM with MSE and a perceptual loss [S41]. **[doc]**, abstracts
  only: none of their numbers is used here.
- **Published real-time segmenters** report Cityscapes mIoU and frames per second on desktop GPUs, at 19 classes
  and 1024×2048 or 1536×768 inputs: PP-LiteSeg 72.0% at 273.6 FPS and 77.5% at 102.6 FPS (GTX 1080 Ti) [S42];
  PIDNet-S 78.6% at 93.2 FPS [S43]; BiSeNet V2 72.6% at 156 FPS (GTX 1080 Ti) [S44]; STDC 71.9% at 250.4 FPS and
  76.8% at 97.0 FPS (GTX 1080 Ti) [S45]; SegFormer-B0 to B5 [S46]; SegNeXt [S47]; TopFormer and SeaFormer for ARM
  phones [S48]; EfficientViT (MIT) "up to 13.9× and 6.2× GPU latency reduction over SegFormer and SegNeXt" on
  Cityscapes [S49]. **[doc]** None of these was measured on a TFLite GPU path in its source. They show that a
  two-class mask at 512 px is far inside what small networks do; they do not rank the candidates for `yappy`.
- **The sensible encoder for such a student is whatever wins the classifier's timing run**, since the same
  convert-and-time step covers both: a MobileNetV4-Conv or RepViT encoder with an FPN or U-Net style head built from
  convolutions and bilinear resizes, all of which are in the GPU delegate's op list [S29]; or EfficientViT-B with its
  segmentation head, whose op family this project has already converted.
- **Against stepping down to L0**: L0 keeps everything the project has built (the decoder fine-tune on box and point
  prompts, `multi3`, the review tools' point redraw, the exact export forms for LiteMLA) and needs no pseudo-labels.
  A plain segmenter drops the prompt path, so the labelling loop would keep a SAM model offline while serving and the
  training crops use the student, which is two segmenters to keep consistent.
- **Lower input resolution for XL0 itself** is not a published operating point: the XL models are trained with the
  longest side at 1024 and the L models at 512 [S20]. It would need its own fine-tune and evidence, and L0 at 512 is
  the published model for that budget.

### 3.8 Practicalities

- **timm has distillation built in.** `train.py` takes `--kd-model-name`, `--kd-distill-type {logit,feature,token}`,
  `--kd-temperature` (default 4.0), `--distill-loss-weight`, `--task-loss-weight`, and feature-dimension overrides.
  `FeatureDistillationTask` matches "pooled pre-logits" features with an MSE loss and "automatically creates a
  projection layer" (`nn.Linear`) when dimensions differ; the teacher is put in `eval()` with gradients off.
  **[src]** [S5] The installed timm 1.0.30 has `timm/task/distillation.py` too [repo: `convert/.venv`]. Limits for
  this project: the teacher is created by timm model name and read at its pooled pre-logits, not at `cls_mean`
  (CLS concatenated with the mean patch token), and it runs on line, with no cache. A `cls_mean` target needs a small
  custom task or loop; the class is a usable template.
- **torchdistill** is a configuration-driven framework for distillation experiments, MIT-licensed. **[doc]** [S50,
  licence field only; its README was not read]
- **Caching teacher outputs.** The pipeline already embeds every patch once with the frozen backbone and fits heads
  in memory (ADR 0025). For distillation the cache is per (patch, view): TinyViT stores a 4-byte seed per view and
  the teacher's output, and replays the augmentation from the seed when training the student. **[doc]** [S8] Mixup
  across patches, which [S9] recommends, needs the teacher run on the mixed image, so it cannot be cached the same
  way; B/16 on a consumer GPU is cheap enough to run on line if that lever is wanted. `resnet18-recipe-levers.md`
  closed mixup for the fine-tuned ResNet18 only; as part of a distillation recipe on the DINOv3 pipeline it is a new
  paired screen, which that file allows.
- **Conversion.**
  - `litert-torch` 0.9.4 has `to_channel_last_io(module, args, outputs)`, which "wraps the module with channel first
    to channel last layout transformations" at the inputs and outputs. **[src]** [repo: `convert/.venv`,
    `litert_torch/_convert/to_channel_last_io.py` L74-92] It changes the interface layout; what layout the converter
    gives the convolutions inside the graph, and whether transposes remain, shows in the converted graph and was not
    checked here.
  - Reparameterised families (RepViT, MobileOne, FastViT, and LeViT's and EfficientViT's conv-BN pairs) must be
    fused before export: timm's `reparameterize_model` calls each module's `fuse()`. **[src]** [repo: `convert/.venv`,
    `timm/utils/model.py` L233] Fusing is exact in real arithmetic, so it is an exact form in ADR 0022's sense, to
    be checked against the unfused model at float rounding like the other forms.
  - ADR 0022 applies in full to any student: the range scan over every photo for an fp16 form, and the forms' checks.
    ConvNeXt's LayerNorm and any attention block are the places the existing fp16 forms were needed for.
  - The precision per graph is decided again (ADR 0019). The encoder failed the fp16 mask bar on this GPU because it
    "accumulates its convolutions' products in fp16"; a convolutional classifier student may meet the same effect,
    and its bar is a cosine on embeddings, not a mask IoU.
  - MobileNetV4's paper notes that a competitor could not be benchmarked "due to PyTorch-to-TFLite export tool
    limitations" [S13]; export is a real filter, and it is why the timing run converts every candidate first.

### 3.9 What a student changes in the project's rules

- **ADR 0006 is superseded by any classifier student.** It fixes the backbone (B/16), "no fine-tuning", and S/16 as
  "only a fallback". A distilled student is a trained backbone, frozen after distillation; the head, C = 0.1, the
  `cls_mean`-style readout and "no TTA" can stay. Serving the published S/16 unchanged is inside ADR 0006's fallback
  wording but still a model change for the owner.
- **The parity rule is not broken by distillation, on one condition.** "An accurate method offline with a cheap one
  live is rejected" describes reporting B/16's metrics while serving a student. A student passes the rule when it is
  the backbone in training too: the head and the probe are fitted on the student's embeddings, and every reported
  metric is computed with the student. The teacher appears only in the student's training loss.
- **ADR 0016 holds as written**: "every model ships with its own probe"; the probe is refitted on the student's
  embeddings and measured on the segmenter test split.
- **ADR 0020's bars do not judge a student against its teacher.** D7 compares a converted graph with "the PyTorch
  fp32 model it was converted from": for a student, the student in PyTorch. Whether the student is good enough is
  ADR 0002's question. The fp16 bar "was accepted by the owner for fp16 on `yappy`'s GPU only" for DINOv3; a new
  graph in fp16 needs that acceptance again.
- **ADR 0022**: choosing a faster architecture is not "approximation ... to make a runtime fit"; the student is a
  different model with its own exact export. ToMe and int8 would be such approximations; a student is not.
- **ADR 0015 is superseded by any segmenter change**: it names XL0 at 1024 px as the owner's choice. Beyond that:
  - L0 with the decoder fine-tuned on the existing labels keeps "the owner judges every label" intact.
  - Encoder-embedding distillation uses no labels at all, so it does not touch the label rule either.
  - Pseudo-label training (§3.7) trains on masks the owner has not judged. That amends ADR 0015's "every training
    and validation label", unless pseudo-labels are confined to an auxiliary loss and validation and test masks stay
    owner-judged; the owner decides which (§5.2).
  - "By the same code in training and serving": the classifier's training pool is cut by the segmenter
    (`merge_segpool`). A new segmenter serves only after the pool is recut with it and the head refitted, or after
    evidence that its crops are interchangeable; otherwise training crops come from XL0 and serving crops from the
    student.
- **ADR 0019**: no change in kind. A student is one more graph whose accelerator and precision are measured on the
  host.
- **Licence** (not legal advice). The DINOv3 License defines "DINO Materials" as what is "distributed by Meta",
  grants the right to "create derivative works of, and make modifications to" them, and makes "Distribution of DINO
  Materials, and any derivative works thereof ... subject to the terms of this Agreement" (§1a, §1b-i); "you are and
  will be the owner of such derivative works and modifications" (§5a). It does not define "derivative works", and it
  mentions outputs only in the warranty disclaimer (§3) and the litigation clause (§5b), never as something
  restricted. **[repo: `docs/reference_dinov3_LICENSE.md`]** So:
  - A student **initialised from DINOv3 weights** (ViT-S/16, ConvNeXt-Tiny) is a modification of DINO Materials and
    carries the Agreement if distributed, exactly like today's checkpoint.
  - A student **initialised from other weights and trained to match DINOv3's embeddings**: the text does not say
    whether that is a derivative work. Treating it as one costs nothing, because ADR 0014 already keeps every weight
    private and the repo publishes none; the question matters only if a student were ever published.
  - Serving predictions from either is the case Meta's maintainers described as needing no licence copy but
    compliance [repo: `docs/dinov3_integration_plan.md` Appendix A, quoting issue #28].
  - **Student weights' own licences**: MobileNetV4, RepViT, LeViT, TinyViT, EfficientViT (MIT lab), RegNet, ResNet,
    EfficientNetV2 and plain ViT weights on timm's hub are Apache-2.0; ConvNeXt V2 and Hiera are CC-BY-NC-4.0;
    FastViT and MobileOne are "other"; MobileCLIP2 is `apple-amlr`; EdgeSAM's repository is under the S-Lab License
    1.0, non-commercial; SAM 3 is under a custom SAM License; SAM 2, EdgeTAM, MobileSAM, EfficientSAM, TinySAM and
    EfficientViT-SAM are Apache-2.0. **[doc]** [S28] Whether the site counts as commercial use is the owner's
    reading (§5.2).

## 4. Options by model

"Speed" is what a source reports and on what; "here" is always to measure. A0006, A0015, A0020, A0022 are the ADRs.

### 4.1 The classifier backbone (5.74 s of about 10 s on `yappy`)

| Option | Accuracy evidence | Speed evidence | Delegate risk | Licence | Recipe | Touches |
|---|---|---|---|---|---|---|
| Published DINOv3 ViT-S/16, no training | measured here: −0.035 (retired folds), test 0.939 against 0.955 [R6] | 2.5-3.2× B/16 [S6]; 12 against 47 GFLOPs [S1] | lowest: the served graph's ops | DINOv3 License; on disk | refit head and probe | A0006 (its fallback) |
| ViT-S/16 distilled from B/16 on patches | none; the aim is to close the row above's gap | as above | lowest | DINOv3 License (initialised from it) | embedding distillation, §3.2; init from DINOv3 S/16 | A0006 |
| MobileNetV4-Conv-M / -L, Hybrid-M | ImageNet 79.9-83.0 [S13, S6]; none on DINOv3 features | 5× (GPU), 10× (CPU) B/16 for Conv-M at 224 [S6]; S23 GPU 4.1 ms [S13] | low for Conv; Hybrid adds attention | Apache-2.0 | embedding distillation from ImageNet init; lower resolution possible | A0006 |
| DINOv3 ConvNeXt-Tiny, with or without distillation on patches | IN-ReaL 86.6 at 256, close to ViT-S's 87.0 at 512 [S1] | 1.7× (GPU), 3.2× (CPU) B/16 [S6] | 7×7 depthwise, LayerNorm, GELU: Unverified | DINOv3 License; a download | refit only, or distil | A0006 |
| LeViT-256 / TinyViT-21M | ImageNet 81.5 / 85.1 [S6]; built as KD students | 5.0× / 1.5× (GPU), 9.4× / 2.7× (CPU) [S6] | attention variants: Unverified | Apache-2.0 | embedding distillation | A0006 |
| RepViT-M1.5, RegNetY-032, ResNet-50, EfficientViT-B2 | ImageNet 80.4-83.1 [S6] | 1.4-1.8× (GPU), 4-5× (CPU) [S6] | low (conv); EfficientViT's forms exist | Apache-2.0 | embedding distillation | A0006 |
| Fully convolutional pass over the crop | none | none; depends on crop size in patch units | as the student's | as the student's | a new readout on both sides | A0006; parity (must train the same way) |
| ToMe on B/16 | paper: large ViTs only [S24] | "2×" for ViT-L/H [S24] | `argsort`, `gather`: Unverified | none new | no training; refit head and probe | A0022 departure; A0006 |
| Fewer patches / lower resolution, B/16 | none | linear in patches | none | none | recipe change | A0006 (224 px); sampling noise (ADR 0019) |

### 4.2 The segmenter (encoder 1.78 s of about 10 s on `yappy`)

| Option | Accuracy evidence | Speed evidence | Delegate risk | Licence | Recipe | Touches |
|---|---|---|---|---|---|---|
| EfficientViT-SAM-L0 at 512 px, decoder fine-tuned | COCO mAP 45.7 against XL0's 47.5 [S20]; was the ML-2 segmenter | 2.7× XL0 on A100 and Orin [S20, S21]; 4.9× here on a CPU [R9] | lowest: the same op family and forms | Apache-2.0; on disk | the existing fine-tune on owner-judged labels | A0015 (XL0 at 1024); recut the pool |
| L0 (or L1) encoder distilled from XL0-v7r's embedding | none here; the method made MobileSAM and stage 1 of EfficientViT-SAM [S20, S22] | as above | as above | Apache-2.0 | L2 on the image embedding over all photos, no labels; then the decoder fine-tune | A0015 (model); labels untouched |
| Plain binary segmenter at 512 px (MobileNetV4, RepViT or EfficientViT-B encoder, light head) | none; pattern established elsewhere [S37-S41] | none on this class of hardware; desktop-GPU figures only [S42-S49] | low for conv encoders with bilinear heads [S29] | Apache-2.0 encoders | supervised on owner-judged labels + XL0-v7r pseudo-labels | A0015 (model, label rule, prompts) |
| EdgeSAM, MobileSAM, RepViT-SAM, EfficientSAM, TinySAM | all below L0 in mAP and A100 throughput [S20] | slower than L0 on the A100 [S20] | new forms needed | EdgeSAM non-commercial; others Apache-2.0 | decoder fine-tune repeated | A0015; no reason over L0 |
| SAM 2.1 Hiera-T, EdgeTAM, EfficientTAM, SAM 3, EfficientSAM3 | image accuracy above SAM (SAM 2) [S34] | A100 or iPhone figures for video [S34, S35] | new forms; Hiera attention with pooling | Apache-2.0; SAM 3 custom | new fine-tune, new export | A0015; unused machinery |
| XL0 at a lower resolution | none; not a published operating point [S20] | about quadratic in side, to measure | none new | none new | re-fine-tune | A0015 |

## 5. What is unknown

### 5.1 To measure here, in order

Each step is cheaper than the next and can stop the ones after it.

1. **Time untrained candidates on `yappy` and on XNNPACK, before training anything.** Convert each candidate with
   random or ImageNet weights at the serving shapes (classifier: batch 8, 224 px, and 160/192 px for convolutional
   ones; segmenter encoders: 512 px) and record, per graph: conversion success, the delegate's placement report
   (ADR 0020's proof), fp32 and fp16 time, the `MemAvailable` drop. Candidates: DINOv3 ViT-S/16; MobileNetV4-Conv-M,
   -Conv-L, -Hybrid-M; ConvNeXt-Tiny (architecture only, no DINOv3 download needed for timing); LeViT-256;
   TinyViT-21M; RepViT-M1.5; RegNetY-032; ResNet-50; EfficientViT-B2; for the segmenter the L0 and L1 encoders and
   one plain segmenter (MobileNetV4-Conv-M with a light head). Weights do not affect timing, so this needs no
   training and no licence decision. It replaces every "×" in this note with a number for the hardware that serves.
   The same graphs on the Pi 5 if that host is still of interest.
2. **The published DINOv3 ViT-S/16 on today's pipeline, paired.** Embed, fit head and probe, seeds 42/123/7 against
   B/16 (ADR 0002), with the guard's test-split figures. No distillation. It gives the accuracy cost of the cheapest
   student on the current in-distribution metric, which this project has only at one earlier date and on a retired
   protocol. If the owner accepts that cost, steps 3-5 are optional for the classifier.
3. **EfficientViT-SAM-L0 at 512 px with the existing decoder fine-tune recipe.** Mask IoU against the owner-judged
   test masks beside XL0-v7r's, the empty-mask rate on negatives, and full-resolution overlays of the worst cases
   (`CODING_STANDARDS.md`: CV output is checked at full resolution). No distillation either.
4. **An embedding-distillation pilot for one transformer and one convolutional student**, chosen by step 1: B/16's
   `cls_mean` as target, cosine or MSE through a linear projector, patches from training-split photos plus the
   guard's training negatives, one cached teacher embedding per (patch, view). Report: the student's embedding
   agreement with the teacher on validation patches (cosine), then head and probe refitted and the paired
   three-seed table against B/16, per-class F1 and the guard's figures included. Seed here means the distillation
   seed too; one distilled student evaluated at three head seeds is not three seeds.
5. **Recipe levers on the pilot, one at a time, paired**: schedule length (the main lever in [S9]); more views per
   patch; mixup with the teacher on line; a CosPress-style loss if the guard's figures fall [S12]; a generic
   unlabelled admixture if the guard's figures fall and the negatives alone do not fix it; a lower student
   resolution; logit distillation through the fitted head as an added term.
6. **If step 3 loses mask quality: the L0 encoder distilled from XL0-v7r's embedding**, then the decoder fine-tune,
   judged as in step 3.
7. **Only if L0 is still too slow or too weak: a plain 512 px segmenter** on owner-judged labels plus pseudo-labels,
   after the owner's decision on the label rule.
8. **For whatever passes: the full conversion path.** Range scan and export forms (ADR 0022), D7 against the
   student's own PyTorch model on `yappy`'s GPU (ADR 0020), ML-7 P7's answer check, and for a segmenter the recut
   pool and refitted head. Then the photo replay for the end-to-end time.
9. **The fully convolutional single pass**, only if a convolutional student wins: first compute from the data how
   many pixels the crop holds at the patch scale against 40 × 224², since that ratio decides whether it can pay.

### 5.2 To decide (the owner's)

- **The acceptance bar for a student.** ADR 0002 has no numeric gate and the in-distribution metrics are near their
  ceiling, so a student can tie B/16 there while having lost what B/16 gained on harder photos. Candidates for what
  is shown beside macro-F1: the guard's test-split figures; the rate at which the student's top-1 differs from
  B/16's on ML-7 P7's served photos, read against patch sampling's 0.85% and fp16's 0.48% (ADR 0019). How much
  accuracy may be traded for how much time is the owner's call; the measured S/16 gap is the reference point.
- **Whether the fp16 bar's acceptance extends to a new graph**, and whether a student's precision is chosen for speed
  the same way.
- **The segmenter's label rule.** Whether unjudged XL0-v7r masks may train a student (amending ADR 0015), only
  supplement owner-judged labels, or not be used; and whether replacing XL0 at serving while keeping it for
  labelling is acceptable.
- **Whether the training pool is recut with a new segmenter** before the classifier is refitted, or interchangeability
  is shown some other way.
- **Licence stance.** Whether a student trained on DINOv3 embeddings from non-DINOv3 initial weights is treated as a
  derivative (the conservative reading changes nothing while weights stay private); whether non-commercial weights
  (ConvNeXt V2, Hiera, EdgeSAM) are usable for this site; whether to download DINOv3 ConvNeXt weights.
- **Which host the speed is for.** The ranking can differ between `yappy`'s GPU and a CPU host (§3.5); step 1
  measures both, and the student is chosen for one.
- **Whether recipe levers closed for ResNet18 (mixup) may be screened inside a distillation recipe.**

### 5.3 Claims that could not be verified

1. Any latency of any candidate on LiteRT's WebGPU accelerator over Mesa ANV, or on XNNPACK on the N100 or the Pi 5.
   No source measures them; every speed ratio in this note is from an RTX 4090 or 5090, an i7-12700H, an A100, a
   Jetson Orin or phones.
2. Which ops of the convolutional candidates the WebGPU accelerator takes: HARD_SWISH, squeeze-excite's pooling and
   sigmoid, GELU inside convolutional blocks, LayerNorm over channels (ConvNeXt), GRN (ConvNeXt V2), 7×7 depthwise
   convolutions, window attention (TinyViT), LeViT's attention bias gather. The documented op list [S29] is for the
   TFLite GPU delegate, not this accelerator.
3. Whether `litert-torch` 0.9.4 converts each candidate at all, and whether the converted convolutional graphs are
   free of layout transposes.
4. Whether DINOv3's released distillation code runs with a ViT-B/16 teacher on a small custom dataset; only the
   configs and the README's test setup were read, not `MultiDistillationMetaArch`.
5. How the DINOv3 ConvNeXt students form their global feature (no CLS token), and so what a `cls_mean`-like readout
   is for them in timm.
6. Whether a convolutional student can reproduce DINOv3's bean-patch embedding well enough to keep the head's
   accuracy. No source tests a domain like this; ADR 0006 records that pretraining, not architecture, carried the
   gain.
7. Whether a student distilled only on bean patches and the guard's negatives keeps the probe's behaviour on
   unseen negatives. [S12] measures a different OOD protocol on ImageNet.
8. The identical IN-R and ObjectNet figures for DINOv3 ConvNeXt-Tiny and -Small [S1, S2]: printed that way in both
   places; probably an upstream error.
9. Whether EfficientViT-SAM-L0's image embedding at 512 px has the same shape as XL0's at 1024 px (needed for a
   direct L2 distillation), and whether XL0-v7r's fine-tuned decoder can be reused on an L0 encoder.
10. ToMe's accuracy cost on a ViT-B/16 at 224 px, and its figure for the accuracy drop in the abstract (the fetched
    text was cut at the number).
11. RepViT's "1 ms" iPhone latency claim, TinySAM's, FastSAM's and SAM 3's speeds, and the per-model numbers of the
    real-time segmenters beyond their abstracts: not read in the papers' tables.
12. The licences of FastViT's and MobileOne's weights ("other" on the hub), of FastSAM, of LightlyTrain, and the text
    of the SAM License; the SHViT repository's licence is not an SPDX one.
13. The column order of MobileNetV4's Table 6 for its last three latency columns (Pixel 7 GPU, Samsung S23 CPU,
    Samsung S23 GPU), which was reconstructed from the HTML header; the figures quoted for the S23 GPU agree with
    the paper's prose ("MNv4 ... on S23 GPU").
14. Whether the site's use counts as commercial under CC-BY-NC or the S-Lab licence.
15. Sources [S19], [S27], [S30], [S32], [S33], [S35]-[S41] and [S46]-[S48] were read as abstracts only.

## 6. Sources

Repository records:

- [R1] `docs/adr/0002-paired-multi-seed-evidence.md`, `0006-frozen-dinov3-b16-head.md`,
  `0015-xl0-segmenter-owner-judged-labels.md`, `0016-ood-guard-probe-at-fixed-half.md`,
  `0019-serving-runs-litert-accelerator-per-host.md`, `0020-converted-models-pass-d7-before-they-serve.md`,
  `0022-export-forms-are-exact.md`
- [R2] `CODING_STANDARDS.md` line 7; `.out-of-scope/` (all five files); `GLOSSARY.md`
- [R3] `docs/dinov3_integration_plan.md` §13 and Appendix A; `docs/reference_dinov3_LICENSE.md`
- [R4] `analysis/ml7_gpu/README.md` §1-§2; `analysis/ml7_gpu/manifest_tflite.json`;
  `analysis/quantization/research_quantization.md` §2.4, §5.4, §5.5
- [R5] `docs/architecture_screen_plan.md` §8 (R2)
- [R6] `EXPERIMENTS_LOG.md`, "DINOv3 Experiment 1 (exp220-231, exp240-251)"; `docs/status.md`
- [R7] `coffeecv/infer.py` (`sample_patches`, `patches_for_photo`)
- [R8] `models_pretrained/` (checkpoints on disk); `convert/.venv` (timm 1.0.30, litert-torch 0.9.4);
  `third_party/efficientvit/applications/efficientvit_sam/`
- [R9] `analysis/ml5_xl0_timing/timing_devcontainer_l0_t4.json`, `timing_devcontainer_xl0_t4.json`

External, read 2026-11-15:

1. [S1] Siméoni et al., "DINOv3", arXiv 2508.10104: §5.2, §7.1 Tab. 14, §7.2 Tab. 15, Fig. 16(a).
   https://arxiv.org/abs/2508.10104
2. [S2] `facebookresearch/dinov3` at `6876159a`, `MODEL_CARD.md`.
   https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/MODEL_CARD.md
3. [S3] `facebookresearch/dinov3` at `6876159a`: `README.md`;
   `dinov3/configs/train/dinov3_vitl16_lvd1689m_distilled.yaml`;
   `dinov3/configs/train/distillation_convnext/convnext_tiny_p16.yaml`;
   `dinov3/configs/train/multi_distillation_test.yaml`.
   https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751
4. [S4] Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2304.07193 (abstract: a
   ViT-g is trained and distilled "into a series of smaller models"). https://arxiv.org/abs/2304.07193
5. [S5] timm at `43b549bf`: `train.py` L531-550, L1179-1215; `timm/task/distillation.py`
   (`LogitDistillationTask`, `FeatureDistillationTask`); `timm/task/token_distillation.py`.
   https://github.com/huggingface/pytorch-image-models/tree/43b549bfe9d680bc94284cfd0b611152d18e97d7
6. [S6] timm at `43b549bf`, `results/`: `results-imagenet.csv`;
   `benchmark-infer-amp-nhwc-pt240-cu124-rtx4090.csv`; `benchmark-infer-fp32-nchw-pt240-cpu-i7_12700h-dynamo.csv`;
   `benchmark-infer-amp-nchw-pt291-cu130-5090.csv`.
   https://github.com/huggingface/pytorch-image-models/tree/43b549bfe9d680bc94284cfd0b611152d18e97d7/results
7. [S7] Zhang et al., "Accessing Vision Foundation Models via ImageNet-1K" (Proteus), arXiv 2407.10366: §2, §3.5.
   https://arxiv.org/abs/2407.10366
8. [S8] Wu et al., "TinyViT: Fast Pretraining Distillation for Small Vision Transformers", arXiv 2207.10666.
   https://arxiv.org/abs/2207.10666
9. [S9] Beyer et al., "Knowledge distillation: A good teacher is patient and consistent", arXiv 2106.05237: §1, §3.1,
   §3.8. https://arxiv.org/abs/2106.05237
10. [S10] Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network", arXiv 1503.02531.
    https://arxiv.org/abs/1503.02531
11. [S11] Vemulapalli et al., "Knowledge Transfer from Vision Foundation Models for Efficient Training of Small
    Task-specific Models", arXiv 2311.18237. https://arxiv.org/abs/2311.18237
12. [S12] Mannix et al., "Preserving Angles Improves Feature Distillation" (CosPress), arXiv 2411.15239.
    https://arxiv.org/abs/2411.15239
13. [S13] Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem", arXiv 2404.10518: abstract, §3
    (roofline), Tab. 6, §8. https://arxiv.org/abs/2404.10518
14. [S14] Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design", arXiv 1807.11164
    (abstract). https://arxiv.org/abs/1807.11164
15. [S15] Vasu et al., "MobileOne: An Improved One millisecond Mobile Backbone", arXiv 2206.04040 (abstract).
    https://arxiv.org/abs/2206.04040
16. [S16] Graham et al., "LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference", arXiv 2104.01136
    (abstract; weights `fb_dist_in1k` in [S6]). https://arxiv.org/abs/2104.01136
17. [S17] Touvron et al., "Training data-efficient image transformers & distillation through attention" (DeiT),
    arXiv 2012.12877. https://arxiv.org/abs/2012.12877
18. [S18] Park et al., "Relational Knowledge Distillation", arXiv 1904.05068. https://arxiv.org/abs/1904.05068
19. [S19] Zhou et al., "Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models",
    arXiv 2311.01441 (abstract). https://arxiv.org/abs/2311.01441
20. [S20] Zhang et al., "EfficientViT-SAM: Accelerated Segment Anything Model Without Accuracy Loss",
    arXiv 2402.05008: Tab. 1, §3.2. https://arxiv.org/abs/2402.05008
21. [S21] `mit-han-lab/efficientvit` at `de7d7733`, `applications/efficientvit_sam/README.md` (Table 1; "distilled
    models").
    https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_sam/README.md
22. [S22] Zhang et al., "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications" (MobileSAM),
    arXiv 2306.14289. https://arxiv.org/abs/2306.14289
23. [S23] Zhou et al., "EdgeSAM: Prompt-In-the-Loop Distillation for SAM", arXiv 2312.06660; licence file of
    `chongzhou96/EdgeSAM`. https://arxiv.org/abs/2312.06660 ;
    https://github.com/chongzhou96/EdgeSAM/blob/master/LICENSE
24. [S24] Bolya et al., "Token Merging: Your ViT But Faster", arXiv 2210.09461. https://arxiv.org/abs/2210.09461
25. [S25] Ranzinger et al., "AM-RADIO", arXiv 2312.06709; Heinrich et al., "RADIOv2.5", arXiv 2412.07679; "SigLino",
    arXiv 2512.20157; Shang et al., "Theia", arXiv 2407.20179 (abstracts and introductions).
    https://arxiv.org/abs/2312.06709 ; https://arxiv.org/abs/2412.07679 ; https://arxiv.org/abs/2512.20157 ;
    https://arxiv.org/abs/2407.20179
26. [S26] LightlyTrain documentation, "Distillation".
    https://docs.lightly.ai/train/stable/pretrain_distill/methods/distillation.html
27. [S27] "EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation", arXiv 2603.18739;
    "Lightweight Distillation of SAM 3 and DINOv3 for Edge-Deployable Individual-Level Livestock Monitoring",
    arXiv 2604.27128 (abstracts). https://arxiv.org/abs/2603.18739 ; https://arxiv.org/abs/2604.27128
28. [S28] Licence fields from the Hugging Face model API (`https://huggingface.co/api/models/<id>`) for
    `timm/mobilenetv4_conv_medium.e500_r256_in1k`, `timm/mobilenetv4_hybrid_medium.ix_e550_r256_in1k`,
    `timm/repvit_m1_5.dist_450e_in1k`, `timm/levit_256.fb_dist_in1k`, `timm/tiny_vit_21m_224.dist_in22k_ft_in1k`,
    `timm/efficientvit_b2.r224_in1k`, `timm/regnety_032.ra_in1k`, `timm/resnet50.a1_in1k`,
    `timm/tf_efficientnetv2_s.in21k_ft_in1k`, `timm/vit_small_patch16_224.augreg_in21k_ft_in1k` (Apache-2.0);
    `timm/convnext_tiny.dinov3_lvd1689m`, `timm/vit_small_patch16_dinov3.lvd1689m` (`dinov3-license`);
    `timm/convnextv2_nano.fcmae_ft_in22k_in1k`, `timm/hiera_tiny_224.mae_in1k_ft_in1k` (CC-BY-NC-4.0);
    `timm/fastvit_sa12.apple_dist_in1k`, `timm/mobileone_s4.apple_in1k` (other); `timm/shvit_s4.in1k`,
    `timm/efficientvit_m5.r224_in1k` (MIT); `apple/MobileCLIP2-S0` (`apple-amlr`); `facebook/sam2.1-hiera-tiny`
    (Apache-2.0); `facebook/sam3` (other). Repository licences from the GitHub API for `mit-han-lab/efficientvit`,
    `ChaoningZhang/MobileSAM`, `yformer/EfficientSAM`, `xinghaochen/TinySAM`, `facebookresearch/sam2`,
    `facebookresearch/EdgeTAM`, `THU-MIG/RepViT`, `google-ai-edge/litert-torch` (Apache-2.0);
    `yoshitomo-matsubara/torchdistill` (MIT); `facebookresearch/sam3` (SAM License).
    https://huggingface.co/timm ; https://github.com/facebookresearch/sam3/blob/main/LICENSE
29. [S29] Google AI Edge, "GPU delegates for LiteRT" (supported ops; tips).
    https://ai.google.dev/edge/litert/performance/gpu
30. [S30] Wang et al., "RepViT: Revisiting Mobile CNN From ViT Perspective", arXiv 2307.09283; "RepViT-SAM",
    arXiv 2312.05760 (abstract). https://arxiv.org/abs/2307.09283 ; https://arxiv.org/abs/2312.05760
31. [S31] Xiong et al., "EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything",
    arXiv 2312.00863. https://arxiv.org/abs/2312.00863
32. [S32] Shu et al., "TinySAM", arXiv 2312.13789 (abstract). https://arxiv.org/abs/2312.13789
33. [S33] Zhao et al., "Fast Segment Anything", arXiv 2306.12156 (abstract). https://arxiv.org/abs/2306.12156
34. [S34] Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv 2408.00714 (abstract);
    `facebookresearch/sam2` at `2b90b9f5`, `README.md` (model table).
    https://arxiv.org/abs/2408.00714 ; https://github.com/facebookresearch/sam2
35. [S35] "EdgeTAM: On-Device Track Anything Model", arXiv 2501.07256; "Efficient Track Anything", arXiv 2411.18933
    (abstracts). https://arxiv.org/abs/2501.07256 ; https://arxiv.org/abs/2411.18933
36. [S36] "SAM 3: Segment Anything with Concepts", arXiv 2511.16719; "EfficientSAM3", arXiv 2511.15833 (abstracts).
    https://arxiv.org/abs/2511.16719 ; https://arxiv.org/abs/2511.15833
37. [S37] "Semi-Supervised Medical Image Segmentation via Knowledge Mining from Large Models", arXiv 2503.06816
    (abstract). https://arxiv.org/abs/2503.06816
38. [S38] "GoodSAM: Bridging Domain and Capacity Gaps via Segment Anything Model for Distortion-aware Panoramic
    Semantic Segmentation", arXiv 2403.16370 (abstract). https://arxiv.org/abs/2403.16370
39. [S39] "Segment Anything Model (SAM) Enhanced Pseudo Labels for Weakly Supervised Semantic Segmentation",
    arXiv 2305.05803 (abstract). https://arxiv.org/abs/2305.05803
40. [S40] "Task-Specific Knowledge Distillation from the Vision Foundation Model for Enhanced Medical Image
    Segmentation", arXiv 2503.06976 (abstract). https://arxiv.org/abs/2503.06976
41. [S41] "Efficient Knowledge Distillation of SAM for Medical Image Segmentation" (KD SAM), arXiv 2501.16740
    (abstract). https://arxiv.org/abs/2501.16740
42. [S42] "PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model", arXiv 2204.02681 (abstract).
    https://arxiv.org/abs/2204.02681
43. [S43] "PIDNet", arXiv 2206.02066 (abstract). https://arxiv.org/abs/2206.02066
44. [S44] "BiSeNet V2", arXiv 2004.02147 (abstract). https://arxiv.org/abs/2004.02147
45. [S45] "Rethinking BiSeNet For Real-time Semantic Segmentation" (STDC), arXiv 2104.13188 (abstract).
    https://arxiv.org/abs/2104.13188
46. [S46] "SegFormer", arXiv 2105.15203 (abstract). https://arxiv.org/abs/2105.15203
47. [S47] "SegNeXt", arXiv 2209.08575 (abstract). https://arxiv.org/abs/2209.08575
48. [S48] "TopFormer", arXiv 2204.05525; "SeaFormer++", arXiv 2301.13156 (abstracts).
    https://arxiv.org/abs/2204.05525 ; https://arxiv.org/abs/2301.13156
49. [S49] Cai et al., "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction",
    arXiv 2205.14756 (abstract). https://arxiv.org/abs/2205.14756
50. [S50] `yoshitomo-matsubara/torchdistill` (repository; licence field). https://github.com/yoshitomo-matsubara/torchdistill
