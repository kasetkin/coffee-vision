# Experiment log — 9h autonomous quality-tuning run

Started: 2026-07-30 18:48 UTC. Target budget: ~9h. One hypothesis changed per experiment, isolated from all others.

**Current best**: `crop-700` (resnet18, freeze_mode=none, backbone_lr=1e-5, patch_crop_size=700,
all else default) — val_macro_f1=0.8034, test_macro_f1=0.9296, test_mcc=0.9284 (best_epoch=15/20).
(Superseded `resnet18-finetune-full`: 0.7402/0.9277/0.9235; `resnet18-lastblock`: 0.7057/0.9023/0.9008;
`resnet18-frozen`: 0.6500/0.8146/0.8094.)

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
| 4 | resnet18, freeze_mode=last_block, backbone_lr=1e-5 | 0.7057 | 0.9023 | 0.9008 | **YES** | Strong clean win: val +0.056 (>>0.03 band), test +0.088/+0.091 (close to but combined with clear val signal, credible). 7/9 classes near-perfect. **New best/baseline for all subsequent experiments.** |
| 5 | resnet18, freeze_mode=none (full fine-tune), backbone_lr=1e-5 | 0.7402 | 0.9277 | 0.9235 | YES (borderline) | vs last_block: val +0.0345 (just above 0.03 band), test +0.025/+0.023 (within noise band but positive, not contradicting). Kenya/Ethiopia-Sidamo confusion improved 19->11 misclassified, driving most of the gain. Took ~29min (vs ~15-20min for last_block) - real compute cost for a modest gain. Adopted as new best but flagged as the most borderline "adopt" so far; worth re-checking in Phase 6 multi-seed pass. |

**Budget note (~2h24m in, at Phase 3 start)**: current best (full fine-tune) costs ~29min/run vs ~15-20min
for frozen/last_block configs. Remaining Phase 3-5 experiments at full 20 epochs against this baseline would
risk overshooting the 9h budget (esp. epochs=40 at ~58min and originally-planned 450 patches at ~85-90min).
Trimming: train_patches_per_class experiment reduced to 150->300 (2x, not 3x) to control its cost; will drop
stretch goals and/or shorten Phase 6 if still running long. Tracking actual elapsed time after each experiment.

## Phase 3 — Patch/data hypotheses

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | adopted? | notes |
|---|---|---|---|---|---|---|
| 6 | train_patches_per_class 150->300 (trimmed from 450) | 0.7495 | 0.9255 | 0.9204 | NO | val +0.009 (<<0.03 band), test/mcc essentially flat (-0.002/-0.003). Inconclusive/no real effect - more (still correlated) patches from the same 9 photos don't add real information once the model already sees enough views. Reverted to 150. |
| 7 | patch_crop_size 512->700 (was 768 in plan - val/test regions are only ~757-758px, 768 doesn't fit) | 0.8034 | 0.9296 | 0.9284 | YES | val +0.063 (>>0.03 band), test/mcc essentially flat (already near-ceiling at ~0.93). BUT per-class picture is mixed, not uniform: Brazil-MonteCristo went 92%->100%, while Ethiopia-Sidamo dropped to 15/40 and CostaRica to 11/40 (both previously much stronger). Aggregate macro-F1 clearly passes the decision rule so adopting, but flagging this isn't a clean "better everywhere" result - larger crops seem to trade some classes for others. |
| 8 | patch_crop_size 700->320 | 0.7530 | 0.8541 | 0.8380 | NO | Clear reject: val -0.050 (exceeds band), test -0.076, mcc -0.090. Confirms crop_700's win wasn't a fluke direction - less spatial context per patch clearly hurts, consistent with bigger being better here. Reverted to 700. |

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
