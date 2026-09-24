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
