# Experiment log — 9h autonomous quality-tuning run

Started: 2026-07-30 18:48 UTC. Target budget: ~9h. One hypothesis changed per experiment, isolated from all others.

**Current best going in**: `resnet18-frozen` (model_name=resnet18, freeze_mode=full, all else default) —
val_macro_f1=0.6500, val_mcc=0.6358, test_macro_f1=0.8146, test_mcc=0.8094 (best_epoch=15/20).

**Decision rule**: a change is *adopted* (becomes the new baseline for subsequent experiments) only if it
beats the Phase 1 noise band on both val_macro_f1 and test_macro_f1. Otherwise it's recorded as
negative/neutral, params.yaml is reverted to the last-adopted baseline, and we move to the next experiment.

If this log's "Status" says IN PROGRESS for an experiment with no result recorded, and you're reading this
after a context reset/interruption: check `dvc exp show` / `git log` for whether that experiment's `dvc exp
run` actually completed, then resume from there.

---

## Phase 0 — Setup (code changes, no training)

Added to support later experiments:
- `freeze_mode: full|last_block|none` (replaces old `freeze_backbone` bool) + `backbone_lr` for differentiated
  optimizer param groups when backbone is (partially) unfrozen.
- Real `optimizer: adamw|sgd` branching (was previously declared but not actually wired up).
- `color_jitter_strength` (was hardcoded at 0.2 in transforms.py).
- `efficientnet_b0` as a third model_name option.

Status: IN PROGRESS

## Phase 1 — Noise calibration

| # | seed | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | notes |
|---|------|---|---|---|---|---|
| 0 | 42 (baseline, already known) | 0.6500 | 0.6358 | 0.8146 | 0.8094 | resnet18-frozen |
| 1 | 123 | | | | | |
| 2 | 7 | | | | | |

Noise band (max-min across the 3 seeds): TBD

## Phase 2 — Backbone/feature-quality hypotheses

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | adopted? | notes |
|---|---|---|---|---|---|---|
| 3 | efficientnet_b0, frozen | | | | | |
| 4 | resnet18, freeze_mode=last_block, backbone_lr=1e-5 | | | | | |
| 5 | resnet18, freeze_mode=none (full fine-tune), backbone_lr=1e-5 | | | | | time-boxed 60min |

## Phase 3 — Patch/data hypotheses

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | adopted? | notes |
|---|---|---|---|---|---|---|
| 6 | train_patches_per_class 150->450 | | | | | |
| 7 | patch_crop_size 512->768 | | | | | |
| 8 | patch_crop_size 512->320 | | | | | |

## Phase 4 — Augmentation hypotheses

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | adopted? | notes |
|---|---|---|---|---|---|---|
| 9 | color_jitter_strength 0.2->0.0 | | | | | |
| 10 | color_jitter_strength 0.2->0.4 | | | | | |

## Phase 5 — Optimization hypotheses

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | adopted? | notes |
|---|---|---|---|---|---|---|
| 11 | optimizer sgd (momentum=0.9) | | | | | |
| 12 | batch_size 32->64 | | | | | |
| 13 | epochs 20->40 | | | | | |

## Phase 6 — Combine winners

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | notes |
|---|---|---|---|---|---|
| 14 | combination of all adopted changes | | | | |

## Stretch (only if time remains)
- resnet34 frozen
- repeat final combined config with a 2nd seed
- val/test patches_per_class 40->80 for lower-noise final evaluation

## Final summary
TBD
