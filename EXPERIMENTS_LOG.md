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
| 9 | color_jitter_strength 0.2->0.0 | 0.7338 | 0.9972 | 0.9969 | NO | Val/test sharply DISAGREE for the first time in this run: val -0.070 (clear reject), test +0.068 to near-perfect (359/360). Checked both confusion matrices: val still shows the same real confusion (Ethiopia-Sidamo, CostaRica) as before, test is essentially solved. This looks like a methodological artifact, not genuine improvement - without jitter the model can lock onto exact color values, and since train/val/test are fixed spatial regions of the same photo (likely with a smooth lighting/color gradient across the frame), test's region apparently happens to align well with train's color statistics while val's region doesn't. Rejecting based on val (the more principled, non-inflated signal here) despite the eye-catching test number. Reverted to 0.2. |
| 10 | color_jitter_strength 0.2->0.4 | 0.8264 | 0.8694 | 0.8788 | NO | val +0.023 (just under the 0.03 threshold, not clearly beating noise), test/mcc both down (-0.060/-0.050). best_epoch=4 (very early, suggests noisy/unstable peak rather than a real improvement). Not a clear win either direction; reverted to 0.2 which remains best. |

## Phase 5 — Optimization hypotheses

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | adopted? | notes |
|---|---|---|---|---|---|---|
| 11 | optimizer sgd (momentum=0.9), same lr/backbone_lr as adamw | 0.7389 | 0.8801 | 0.8888 | NO | Clear reject: val -0.065, test -0.050, mcc -0.040, all beyond noise. SGD as a drop-in swap at AdamW's LR underperforms - not surprising, SGD typically wants its own (often higher) LR to match adaptive-optimizer convergence in the same epoch budget, but tuning that is out of scope for a one-variable test. Reverted to adamw. |
| 12 | batch_size 32->64 | 0.7992 | 0.9403 | 0.9361 | NO | Essentially a wash: val -0.004, test +0.011, mcc +0.008, all well within noise. No meaningful effect. Reverted to 32. |
| 13 | epochs 20->40 | 0.8039 | 0.9634 | 0.9602 | NO | val essentially flat (+0.0005, negligible), test/mcc up modestly (+0.034/+0.032) but well below the ~0.09 confirmatory bar. best_epoch=29 shows some continued slow improvement past epoch 20, but not enough to justify ~2x the compute (~58min vs ~29min) for zero clear val gain. Reverted to 20 epochs. |

## Phase 6 — Combine winners

Since each experiment above was tested sequentially on top of the running best (not independently against
a fixed baseline), the final params.yaml **already is** the combination of every adopted change - no
separate combination step was needed. Final config: resnet18, freeze_mode=none (full fine-tune,
backbone_lr=1e-5), patch_crop_size=700, all else at original defaults.

| # | change | val_macro_f1 | test_macro_f1 | test_mcc | notes |
|---|---|---|---|---|---|
| 14 | multi-seed check of final config: seed=123 (vs seed=42's 0.8034/0.9296/0.9284) | 0.6720 | 0.7446 | 0.7893 | **Important finding**: val -0.131, test -0.185, mcc -0.139 - far beyond the ~0.03/~0.11 noise bands measured in Phase 1 (on the simpler frozen-backbone config). Confusion matrix confirms this is a real, valid trained model, not a bug: it resolves different confusions than seed=42 (Ethiopia-Sidamo collapses into Guatemala this time; Kenya/Ethiopia, weak at seed=42, are now strong). Conclusion: full fine-tuning + large crops increased seed-to-seed variance substantially compared to the simple frozen-linear-probe baseline - the 0.80/0.93 numbers reported for seed=42 throughout this log are closer to a best-case draw than a reliable expectation. Ran out of time budget to properly characterize this larger noise band with more seeds - see Final Summary. |

## Stretch (only if time remains)
- resnet34 frozen
- repeat final combined config with a 2nd seed
- val/test patches_per_class 40->80 for lower-noise final evaluation

## Final summary

**Ran ~8h15m of a 9h budget** (18:48 -> ~03:03 UTC), 14 experiments (2 noise-calibration, 10 hypothesis
tests, 1 code-infra phase, 1 final multi-seed check), one hypothesis changed at a time throughout, every
result committed to git with reasoning in the commit message. Stretch goals (resnet34, extra seeds,
higher val/test patch counts) were not reached - the budget went entirely to the core plan plus the
Phase 6 multi-seed finding below, which was worth the time.

**What actually worked** (in order tested, each on top of the previous):
1. `freeze_mode: full -> last_block` (unfreeze resnet18's last conv block, backbone_lr=1e-5): clear win
2. `freeze_mode: last_block -> none` (full fine-tune): smaller further win
3. `patch_crop_size: 512 -> 700` (more context per crop): clear win on val, though with a per-class
   trade-off (some classes better, some worse) rather than uniform improvement

**What didn't work**: efficientnet_b0 as frozen backbone (clearly worse than resnet18), more train patches
(150->300, no effect), smaller crops (320, clearly worse - confirms crop_700's direction), stronger/weaker
color jitter (0.0 produced a misleading near-perfect test score that didn't hold up on val - flagged as a
likely spatial-region artifact, not real; 0.4 was a wash), SGD optimizer (worse, though not LR-tuned for
SGD specifically), larger batch size (no effect), more epochs (marginal, not worth 2x compute).

**The important caveat this run surfaced**: the final config (full fine-tune + 700px crops) was only ever
evaluated at seed=42 during Phases 2-5 for time-budget reasons. The one additional seed checked at the very
end (123) produced dramatically different numbers (val 0.80->0.67, test 0.93->0.74) - both confusion
matrices are legitimate, valid models, they just resolve different classes' confusion differently. This
means: (a) the *direction* of each adopted change is probably real, since several of them (last_block,
crop_700) showed effects several times larger than the noise band measured in Phase 1, but (b) the exact
final numbers (val=0.80, test=0.93) should be treated as a favorable draw, not a reliable estimate of what
this config will score on a fresh run - the true expected performance is somewhere in a wide band, roughly
val 0.67-0.80, test 0.74-0.93, until more seeds are run. This is a bigger noise band than Phase 1 found,
consistent with full fine-tuning introducing more stochasticity than a frozen linear probe.

**Recommended next steps** (not done here, out of budget):
- Run 3-5 more seeds of the final config to properly characterize its noise band (most important - the
  current 2-seed estimate is too thin to trust).
- Given that noise band, consider whether last_block (cheaper, and was checked at only one seed too but
  had a larger effect size relative to Phase 1's band, so plausibly more robust) is the more defensible
  choice than full fine-tune until fine-tune's variance is better understood.
- More capture sessions (different day/lighting/bean batch per class) remains the highest-value action
  for this project overall - every finding in this log, including the variance issue just found, traces
  back to having only 9 source photos total.

## Phase 7 — New dataset: 180-photo box rig, photo-level splits

New capture session `dataset/2026-08-07__box_pictures_all_classes`: 20 photos/class x 9 classes (180
total) from a fixed-position (tripod) box rig, vs. the single circular-lens photo/class used in every
experiment above. This is the "more capture sessions" recommendation from the Final summary, acted on.

Crop pipeline needed rework before this was usable: the adaptive saturation-threshold crop
(`crop_tray.py`'s stage 2) broke on ~a third of the photos, traced to directional lighting drift over
the ~2h shoot fooling per-image Otsu thresholding (verified at full res - the bad crops were 100% clean
bean pixels, just needlessly tiny, down to 178x1065 in the worst case). Replaced with a fixed 10% trim
off the stage-1 rough tray box (that stage is texture-based, not lighting-sensitive, and measured
sub-2% box variance across all 180 photos - the rig is effectively fixed). All 180 crops now land in a
uniform ~1048-1137px band. New `crop_dataset_fixed_trim()` in `coffeecv/crop_tray.py`.

New dataset loader (`MultiPhotoPatchDataset` in `coffeecv/dataset.py`, `compute_valid_region_rect` in
`coffeecv/geometry.py`): unlike `PatchCoffeeDataset`, which had to split train/val/test as *spatial
regions of one photo* (the thing that produced Phase 4 exp 9's val/test color-gradient artifact), splits
are now done at the *photo* level - 14/3/3 photos per class for train/val/test, shuffled per-class with
a seeded RNG (not sliced in filename/timestamp order, to avoid reintroducing a time-correlated split
given the lighting-drift finding above), then patches sampled per-photo as before. Verified zero
train/val/test photo overlap across all 9 classes before training.

Same hyperparameters as the Phase 6 best config (resnet18, freeze_mode=none, patch_crop_size=700,
batch_size=32, epochs=20, adamw, lr=1e-3/backbone_lr=1e-5, seed=42) - only the dataset and split
mechanism changed, so this is a clean one-variable comparison against that config's numbers.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | notes |
|---|---|---|---|---|---|---|
| 15 | switch to 180-photo multi-photo dataset, photo-level splits (Phase 6 config unchanged otherwise) | 0.8998 | 0.8880 | 0.9145 | 0.9040 | best_epoch=18. Compare to Phase 6's seed=42 numbers (val 0.8034/0.9284mcc... test 0.9296/0.9284) - val up ~0.10, test/mcc essentially flat. |

**Read**: raw numbers are close to the old best run, but the old run's val and test were two quarters of
the *same single photo* - val/test agreement there was never a real generalization check, just a
same-photo consistency check, and Phase 6 exp 14 showed the whole config's seed-to-seed band was wide
(val 0.67-0.80, test 0.74-0.93) on that data. Here val/test are disjoint *photos* for the first time, and
they agree closely (macro_f1 0.900 vs 0.915, mcc 0.888 vs 0.904) - that agreement is a much stronger
signal now that it isn't structurally guaranteed by sharing a photo. Per-class f1 is broadly solid
(Ethiopia-Kochere and Vietnam-Robusta both ~1.0 on test) with two plausible real confusions rather than
noise: Brazil-MonteCristo <-> Guatemala-Tata (8/40 test) and Brazil-Cerrado <-> Brazil-MonteCristo (6/40
test) - both same-continent/visually-similar green-bean pairs, not a random scatter. No single-class
collapse like Phase 3 exp 7 saw on the old data.

**Not yet done**: multi-seed check on this new dataset/config (Phase 6 exp 14's variance warning hasn't
been re-tested here - though photo-level splits should structurally reduce that variance vs. the old
single-photo spatial splits, that's an expectation, not yet a measurement). Patch_crop_size sweep above
700 also not tried - see the discussion that motivated keeping 700 as the starting point rather than
assuming bigger is still better on this different geometry.
