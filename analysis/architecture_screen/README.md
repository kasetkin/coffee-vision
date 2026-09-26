# Backbone screen — frozen features under camera shift

Measured 2026-09-21 on the workstation (Ryzen 5 5600U, 6 torch threads, CPU only). Written up in
`docs/architecture_screen_plan.md`; this directory holds the scripts and raw results.

## What this measures

For each candidate backbone: **freeze it, embed identical patches, fit one logistic regression,
score leave-one-camera-out macro-F1.** Every arm sees byte-identical patches (the project's own
`MultiPhotoPatchDataset`, bean-unit sizing 4-7, 120 patches/class/rig, seed 42, eval transform)
and the same classifier (standardised features, C=1.0, no per-arm tuning).

It is a **screen of pretrained feature quality under camera shift**, not an adoption test.
Nothing is fine-tuned; MixStyle and TTA are absent by construction.

Known deviations from a real `run_folds.py` fold, all identical across arms so the *ranking* is
unaffected, but which make absolute values optimistic:

- trains on `split="all"` of the three training rigs (every photo) rather than the 70% photo
  split, so arms see ~43% more training photos than a real fold;
- does not exclude an absent class from the macro average the way `compute_split_metrics` does,
  so cam_iphone (no class_008) carries a phantom f1=0 — hence the separate 3-ten-class-fold mean;
- scores patch-wise, with no photo-level pooling.

## Results

`xrig_probe.py` — backbone families:

| backbone | dim | 4-fold mean | 3 ten-class folds |
|---|---|---|---|
| resnet18_in1k (current family) | 512 | 0.4645 | 0.4938 |
| resnet50_in1k | 2048 | 0.5198 | 0.5588 |
| convnext_tiny_in1k | 768 | 0.6691 | 0.7013 |
| **dinov2_vits14** | 384 | **0.8324** | **0.8511** |

`xrig_probe2.py` — is it the recipe, the architecture, or the self-supervision?

| arm | dim | 4-fold mean | 3 ten-class folds |
|---|---|---|---|
| resnet18 a1_in1k (modern recipe) | 512 | 0.4496 | 0.4763 |
| resnet18 tv-V1 (control) | 512 | 0.4645 | 0.4938 |
| vit_small supervised in1k | 384 | 0.5534 | 0.5548 |
| dinov2_vits14 (SSL anchor) | 384 | 0.8324 | 0.8511 |

Control and anchor reproduce probe 1 to four decimals on every fold, so the two runs are
directly comparable.

**Conclusion:** the gain is not the training recipe (a modern-recipe ResNet18 is slightly
*worse*) and only marginally the transformer (supervised ViT-S gains +0.09). DINOv2 beats a
same-size, same-dimension supervised ViT-S by +0.279, so the credit belongs to the DINO
self-supervised objective and LVD-142M's scale.

`bench.py` — measured CPU cost, batch 32 at 224px, forward+backward+AdamW step:

| architecture | params | train step | inference |
|---|---|---|---|
| resnet18 (current) | 11.7M | 1.33 s | 13.9 ms/img |
| UNet-R18 multi-task | 14.3M | 1.89 s | 13.9 ms/img (decoder dropped) |
| efficientnet_b0 | 5.3M | 2.23 s | 19.7 ms/img |
| resnet34 | 21.8M | 2.24 s | 20.7 ms/img |
| resnet50 | 25.6M | 4.63 s | 42.6 ms/img |
| convnext_tiny | 28.6M | 4.73 s | 40.7 ms/img |
| dinov2_vits14 (frozen) | 22.1M | n/a | 43.2 ms/img |

## Reproducing

    python analysis/architecture_screen/bench.py             # ~6 min
    python analysis/architecture_screen/xrig_probe.py  120   # ~17 min
    python analysis/architecture_screen/xrig_probe2.py 120   # ~15 min, needs `pip install timm`

Both probes are deterministic — patch boxes come from the seeded RNG and every model is frozen —
so a re-run reproduces these tables exactly. Neither touches the VM or the DVC cache.

## Experiment 1 — frozen DINOv3 under the real fold protocol (2026-09-24 → 26)

The follow-up that makes the screen above admissible: `docs/dinov3_integration_plan.md` §5, driver
`coffeecv_dino/screen.py`, raw store `outputs/dino_screen/results.json` (VM-produced, fetched). Every
deviation listed at the top of this file is removed: each (seed, held-out camera) fold is built by
`coffeecv.fold_data.build_fold_datasets`, the call `train_baseline` makes, so at seed 42 the patches
are exactly the ones exp200–203 trained and were scored on; the 70/15/15 photo split applies;
cam_iphone is scored as its honest 8-class macro; and the selected cells are also scored photo by
photo (40 patches, no TTA — the `/classify` path). Head: L2 logistic regression on standardised
features, C from an 11-point grid chosen on **val** macro-F1, exported as an `nn.Linear`. 3 seeds
(42, 123, 7) × 4 folds, all on `powervpsssh` (EPYC Genoa, 8 threads), timm 1.0.29.

Cross-rig macro-F1, mean of the three ten-class folds (patch level; seed sd over the 3 per-seed means):

| backbone | readout | dim | 3-fold mean | seed sd | photo-pooled 3-fold | archived |
|---|---|---|---|---|---|---|
| resnet18 tv-V1, frozen (control) | avgpool | 512 | 0.5225 | 0.005 | | |
| dinov2_vits14 (reference) | cls | 384 | 0.8563 | 0.010 | | |
| dinov2_vits14 (reference) | cls_mean | 768 | 0.8522 | 0.009 | | |
| dinov3_vits16 | cls | 384 | 0.8596 | 0.005 | | |
| **dinov3_vits16 (pre-registered primary)** | **cls_mean** | 768 | **0.8487** | 0.003 | 0.9136 | exp220–231 |
| dinov3_vits16plus | cls_mean | 768 | 0.8512 | 0.009 | | |
| dinov3_vitb16 | cls | 768 | 0.8921 | 0.019 | | |
| **dinov3_vitb16 (selected 2026-09-26)** | **cls_mean** | 1536 | **0.8873** | 0.023 | **0.9489** | exp240–251 |
| fine-tuned resnet18 + MixStyle, seed 42 only (exp200–202) | | 512 | 0.8025 (last-10 0.8092) | | | exp200–203 |

In-distribution control (test macro-F1, 3 ten-class folds × 3 seeds): S/16 0.939, B/16 0.955 against
the fine-tuned ResNet18's 0.900–0.916 — neither DINO arm is buying cross-rig accuracy with an
in-distribution loss.

Paired (fold × seed) deltas, cross-rig macro-F1, 12 pairs each:

| comparison | positive | mean | min |
|---|---|---|---|
| S/16 cls_mean − frozen R18 (G1) | 12/12 | +0.352 | +0.283 |
| S/16 cls_mean − V2 cls_mean (G3) | 4/12 | −0.008 | −0.048 |
| S/16 cls − S/16 cls_mean (readout lever) | 9/12 | +0.008 | −0.006 |
| **B/16 cls_mean − S/16 cls_mean** | **11/12** | **+0.035** | −0.002 |
| B/16 cls_mean − S/16 cls_mean, photo-pooled | 11/12 | +0.034 | −0.008 |
| B/16 cls − B/16 cls_mean (readout lever) | 9/12 | +0.005 | −0.004 |
| S+/16 cls_mean − S/16 cls_mean | 5/12 | −0.000 | |

**Conclusions.** The screen's finding survives the real protocol (G1 12/12; G2 0.849 ≥ 0.75), and
frozen DINOv3 beats the fine-tuned production recipe on the only seed that recipe has (S/16 at seed
42: 0.846, B/16 0.896, vs 0.8025 val-peak / 0.8092 last-10). DINOv2 vs DINOv3 at ViT-S is a tie (G3 not
triggered), CLS vs CLS ⊕ patch-mean is a tie at both sizes (the pre-registered cls_mean stays), and
S+/16 is a null. **Size is the one lever that moved:** ViT-B/16 wins 11/12 at patch and photo level.
The owner selected it (plan §0.4), accepting ~3.2× S/16's inference cost (145 vs 45 ms/img at the
VM's production 4 threads, `coffeecv_dino/bench.py`) and a wider seed spread.

**Follow-up kill-switches for the selected B/16 (plan §6), both passed 2026-09-26:** the deployed OOD
probe method separates dev-split negatives at least as well in the B/16 space as in the deployed
ResNet18 space (same-rig and green-legume AUROC 1.000 in both; internet-matched 0.936 → 1.000,
`coffeecv_dino/ood_feasibility.py`), and B/16 without TTA costs ≈ 5.8 s of model time per photo at 4
threads vs 5.0 s for today's ResNet18 with TTA: parity, not the saving S/16 would have been.

Reproducing (on the VM; each seed ≈ 1.5 h for the core arms, B/16 ≈ 40 min per fold with photo pooling):

    COFFEECV_TORCH_THREADS=all python -m coffeecv_dino.screen --seeds 42 123 7 \
        --backbones resnet18 dinov2_vits14 dinov3_vits16 dinov3_vits16plus dinov3_vitb16 \
        --photo-pool dinov3_vits16:cls_mean dinov3_vitb16:cls_mean
    python -m coffeecv_dino.screen --summary

Resumable, and deterministic at a fixed thread count: the seed-42 B/16 fold re-run of 2026-09-26
reproduced its patch-level scores bit for bit.
