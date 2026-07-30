# Experiment log — 9h autonomous quality-tuning run

Started: 2026-07-30 18:48 UTC. Target budget: ~9h. One hypothesis changed per experiment, isolated from all others.

**Current best going in**: `resnet18-frozen` (model_name=resnet18, freeze_mode=full, all else default) —
val_macro_f1=0.6500, val_mcc=0.6358, test_macro_f1=0.8146, test_mcc=0.8094 (best_epoch=15/20).

**Decision rule (revised after Phase 1 — see noise band below)**: val_macro_f1's noise band (~0.03) is much
tighter than test_macro_f1's (~0.11, measured from only 3 seeds so treat as rough) — with 40 samples/class,
test is just noisier because it isn't the checkpoint-selection signal. So: treat val_macro_f1 improvement
>0.03 as the primary adopt/reject signal; treat test_macro_f1/mcc as directional confirmation only —
a large test swing (>0.09) corroborates, a small one is inconclusive either way. All Phase 2-5 experiments
below are single-seed (seed=42) for time budget reasons, so borderline results get labeled "inconclusive,"
not confidently adopted or rejected. Only Phase 6's final combined config gets a multi-seed check.

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
| 1 | 123 | 0.6788 | (n/a) | 0.7724 | 0.7760 | best_epoch=12 |
| 2 | 7 | 0.6687 | (n/a) | 0.7050 | 0.7083 | best_epoch=9 |

**Noise band (max-min across the 3 seeds)**: val_macro_f1 spread=0.0288 (mean 0.666, ~0.03),
test_macro_f1 spread=0.1096 (mean 0.764, ~0.11), test_mcc spread=0.1011 (mean 0.765, ~0.10).
Test is far noisier than val at this sample size (40/class) - see revised decision rule above.
Retrospective note: the earlier lr=3e-4 "regression" (pre-Phase-1, val 0.65->0.658, test 0.81->0.72)
was likely mostly/entirely this same noise, not a real optimization effect - flagging honestly rather
than re-litigating it now.

## Phase 2 — Backbone/feature-quality hypotheses

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | adopted? | notes |
|---|---|---|---|---|---|---|
| 3 | efficientnet_b0, frozen | 0.5479 | 0.5526 | 0.5212 | NO | Clear reject - both drops far exceed noise bands (val -0.10, test -0.26). resnet18's frozen features transfer better to this task; not investigating why (e.g. preprocessing mismatch) given time budget. |
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
