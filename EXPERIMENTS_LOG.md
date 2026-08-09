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

Patch_crop_size sweep above 700 not tried yet - see the discussion that motivated keeping 700 as the
starting point rather than assuming bigger is still better on this different geometry.

### Exp 16-17: multi-seed check

Same config as exp 15, only `seed` changed (123, then 7 - the same two extra seeds Phase 1 used, for
direct comparability). This directly answers the open question flagged above and in the original Final
summary ("the current 2-seed estimate is too thin to trust").

| # | seed | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch |
|---|------|---|---|---|---|---|
| 15 | 42 | 0.8998 | 0.8880 | 0.9145 | 0.9040 | 18 |
| 16 | 123 | 0.9195 | 0.9142 | 0.8915 | 0.8796 | 19 |
| 17 | 7 | 0.8979 | 0.8888 | 0.8859 | 0.8755 | 19 |

**Noise band (max-min across the 3 seeds)**: val_macro_f1 spread=0.0216 (mean 0.906), test_macro_f1
spread=0.0286 (mean 0.897), test_mcc spread=0.0285 (mean 0.886).

**This confirms the expectation from exp 15**: photo-level splits collapsed the noise band dramatically.
Compare to Phase 6 exp 14's 2-seed check on this exact hyperparameter config but the *old* single-photo
dataset (val spread 0.13, test spread ~0.19) - here, with 3 seeds instead of 2, test spread is ~7x
smaller (0.029 vs ~0.19), and even beats Phase 1's noise band from the old dataset's much simpler frozen-
backbone config (test spread 0.11). The 20-photo/class dataset isn't just adding raw signal, it's making
the whole evaluation trustworthy in a way patch-level augmentation on 1 photo/class never could be.

Per-class f1 also stayed a consistent *ranking* across seeds rather than reshuffling per seed the way
Phase 6 exp 14 found (there, Ethiopia-Sidamo swung from strong to collapsed and back):

| class | seed 42 | seed 123 | seed 7 |
|---|---|---|---|
| 001 Ethiopia,Sidamo | 0.894 | 0.925 | 0.929 |
| 002 Kenya,AA | 0.892 | 0.827 | 0.911 |
| 003 Colombia,PinkBourbon | 0.937 | 0.895 | 0.925 |
| 004 CostaRica,LaPastora | 0.962 | 0.864 | 0.962 |
| 005 Guatemala,Tata | 0.867 | 0.873 | 0.785 |
| 006 Brazil,Cerrado | 0.889 | 0.851 | 0.851 |
| 007 Brazil,MonteCristo | 0.790 | 0.789 | 0.611 |
| 008 Ethiopia,Kochere | 1.000 | 1.000 | 1.000 |
| 009 Vietnam,Robusta | 1.000 | 1.000 | 1.000 |

008/009 are perfect on every seed (visually distinctive beans). 007 (Brazil-MonteCristo) is the weakest
class on every seed too, and the most seed-sensitive (0.79/0.79/0.61) - a genuine hard class (confusable
with Guatemala-Tata and Brazil-Cerrado per exp 15's confusion matrix), not a fluke of one bad draw.
That's a legitimate target for more data on that class specifically, not a training instability.

**Conclusion**: the exp 15 config (resnet18, full fine-tune, patch_crop_size=700, seed=42) is a
reasonable representative result, expected true performance band is roughly val 0.90-0.92 /
test 0.89-0.91 / test_mcc 0.88-0.90 - a real, narrow band now, not the wide 2-seed guess Phase 6 left
off with.

### Exp 18: epochs 20->50 with early stopping

Motivation: all 3 seeds above hit `best_epoch` at 18 or 19 out of the 20-epoch cap - unlike the old
dataset's Phase 5 exp 13 (`epochs 20->40`, best_epoch=29, "not adopted, marginal at 2x cost"), which
peaked mid-run rather than at the ceiling. That pattern here looked like the run might be getting cut off
early. Added early stopping first (patience=8 on val_macro_f1, see separate infra commit - doesn't change
what `best.pt` selects, only saves compute) so testing a much higher ceiling (50) doesn't cost a full 50
epochs if it plateaus sooner. Same config as exp 15 otherwise (seed=42).

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run | wall time |
|---|---|---|---|---|---|---|---|---|
| 15 | epochs=20 (baseline) | 0.8998 | 0.8880 | 0.9145 | 0.9040 | 18 | 20 | ~33min |
| 18 | epochs=50, early_stop_patience=8 | 0.9086 | 0.8975 | 0.9228 | 0.9129 | 17 | 25 (early-stopped) | ~43min |

**By this run alone, not a clear win**: given a much longer runway, it still peaked at epoch 17 -
essentially the same point exp 15 found at epoch 18 within its tighter 20-epoch cap - then plateaued for
8 more epochs before early-stopping triggered at 25. The 18/19-out-of-20 pattern across the 3 seeds
wasn't the ceiling truncating real improvement; it's just where this config's optimum happens to land.
The deltas that did show up (val +0.0088, test_macro_f1 +0.0083, test_mcc +0.0089) are all smaller than
the noise band exp 16-17 just measured on this exact setup (val spread 0.0216, test spread 0.0286) - not
distinguishable from run-to-run noise off a single data point, at ~30% more wall time (43min vs 33min).

**Decision (user call): adopt epochs=50 as the standing default anyway.** The reasoning above is about
whether *this specific run* proved a win, not about what the right standing ceiling is now that early
stopping exists - those are different questions. With early_stop_patience=8 bounding the typical-case
cost automatically (this run cost 43min, not the full 83min a fixed 50-epoch run would take), the risk of
setting the ceiling higher is small in most cases, while a 20-epoch cap risks silently truncating some
*future* hyperparameter config whose optimum happens to land later than this one's did - and there'd be
no signal that it happened short of noticing best_epoch pinned at the cap again. Set in params.yaml.

### Exp 19: patch_crop_size 700->500

Discussed going bigger (headroom now exists up to ~1000px given the new crop sizes) but the user wanted
to check the other direction first: does a smaller patch still hold up now that per-photo diversity comes
from 20 real photos instead of needing translation variety squeezed out of one? Same config as exp 18
otherwise (seed=42, epochs=50 cap, early_stop_patience=8).

Run via `dvc repro` after editing `patch_crop_size` directly in params.yaml (no `-S` override attempted -
`dvc exp run`'s isolated temp-workspace mode failed to materialize the dataset there, see the separate
DVC workflow note; `dvc repro` runs in the actual workspace and worked fine). Verified `outputs/config.json`
reflects patch_crop_size=500 before trusting the result, since `dvc repro --dry` gave a misleading
"cached, skipping run" message that `dvc status` did not corroborate.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 18 | patch_crop_size=700 (baseline) | 0.9086 | 0.8972 | 0.9228 | 0.9129 | 17 | 25 |
| 19 | patch_crop_size=500 | 0.8867 | 0.8756 | 0.8527 | 0.8387 | 23 | 31 |

**Clear reject.** val -0.022 (right at the noise band edge), but test_macro_f1 -0.070 and test_mcc -0.074
- both more than double the noise band exp 16-17 measured on this dataset (test spread 0.0286), so this
isn't run-to-run noise, it's a real effect. Per-class breakdown shows it's not uniform: Kenya-AA (f1 0.69)
and Colombia-PinkBourbon (f1 0.74) took the biggest hits, while Ethiopia-Kochere/Vietnam-Robusta stayed
at 1.000 regardless (same pattern as the multi-seed check - those two classes appear to be "easy" under
any reasonable patch size). Directionally consistent with Phase 3 exp 8's finding on the old dataset
(700->320 also a clear reject) - bigger patches still win on the new dataset and rig, even though the
mechanism for *why* has shifted (less about needing translation diversity now that real photo diversity
exists, more likely just about how much bean-pile context a smaller crop can hold before the model runs
out of texture to distinguish from). Reverted params.yaml and dvc.lock to the exp 18 (700) state - no
retrain needed to restore it, that state was already correctly captured in the last commit.

## 16h autonomous run (started 2026-08-07 18:30 UTC, budget ends ~2026-08-08 10:30 UTC)

User-directed: patch_crop_size=900 first, then learning rate, then self-directed one-variable-at-a-time
hyperparameter search for the rest of the budget. Same methodology as Phases 1-6: sequential (each test
modifies the current adopted best by one variable), single seed=42 per test unless a result is close to
the noise band and worth confirming, decision rule is "beat the exp 16-17 noise band" (val spread 0.0216,
test_macro_f1 spread 0.0286, test_mcc spread 0.0285). Every experiment - adopted or rejected - gets its
own commit + push, per explicit instruction, using `dvc repro` (not direct python) so dvc.lock stays in
sync automatically (see the DVC workflow memory note for why that matters).

### Exp 20: patch_crop_size 700->900

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 18 | patch_crop_size=700 (baseline) | 0.9086 | 0.8972 | 0.9228 | 0.9129 | 17 | 25 |
| 20 | patch_crop_size=900 | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |

**Clear adopt.** val +0.0493 (>2x the 0.0216 band), test_mcc +0.0312 (exceeds its 0.0285 band),
test_macro_f1 +0.0271 (just under its 0.0286 band alone, but all three metrics moved the same direction
together, not a mixed/ambiguous signal). Notably, Brazil-MonteCristo - the weakest, most seed-sensitive
class in every prior experiment (f1 0.61-0.79 at patch_crop_size=700) - jumped to f1=0.929 here. More
spatial context per patch seems to specifically help resolve exactly the confusion (with Guatemala-Tata
and Brazil-Cerrado) that's been showing up since exp 15. Combined with exp 19's reject at 500, this
brackets a clean picture: bigger patches keep winning up to at least 900, on both sides of 700. Kept
patch_crop_size=900 in params.yaml as the new baseline for everything below.

### Exp 21: lr (head) 0.001->0.002

User's suggested starting point. Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | lr=0.001 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 21 | lr=0.002 | 0.9609 | 0.9565 | 0.9384 | 0.9323 | 23 | 31 |

**No effect - not adopted.** val +0.0030 (well inside the 0.0216 band), test_macro_f1 -0.0115 and
test_mcc -0.0118 (both well inside the ~0.0286 band too) - a small, mixed-direction result fully
consistent with noise. AdamW's already handling per-parameter adaptive scaling and the cosine schedule
anneals either starting point down over the run, so this isn't too surprising - the head lr doesn't seem
to be a sensitive knob at either value tried. Reverted to lr=0.001. (backbone_lr, the other learning rate
in this full-fine-tune config, is a separate untested variable - candidate for later if time allows.)

### Exp 22: backbone_lr 1e-5->3e-5

The other learning rate in this full-fine-tune config (governs how much the pretrained resnet18 backbone
itself adapts, vs. `lr` which is head-only). Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | backbone_lr=1e-5 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 22 | backbone_lr=3e-5 | 0.9860 | 0.9844 | 0.9358 | 0.9331 | 23 | 31 |

**Rejected - val/test disagree, and not in the reassuring direction.** val jumped +0.0281 (beyond the
0.0216 band, looks like a clear win by itself), but test_macro_f1 moved -0.0141 and test_mcc -0.0110
(both within their own noise bands, but the wrong direction to corroborate val's jump). Per-class test
breakdown shows why this looks like overfitting to the validation set rather than genuine improvement:
Vietnam-Robusta - perfect (f1=1.000) in literally every prior experiment, including every seed of the
multi-seed check - dropped to f1=0.952, and Kenya-AA fell to f1=0.750 (from 0.868 at baseline). train_loss
also collapsed to ~0.005-0.02 by epoch 20+, far lower than exp 20 ever reached. The mechanism fits: a 3x
higher backbone_lr gives the pretrained backbone much more freedom to specialize, and best.pt is selected
by peak val_macro_f1 off a val set of only 3 photos/class - exactly the setup where a more flexible
backbone can start fitting idiosyncrasies of those specific 3 photos rather than the class in general.
Unlike Phase 4 exp 9's old-dataset val/test disagreement (a structural artifact from spatial splits
sharing lighting), val/test are genuinely disjoint photos now, so this reflects a real generalization gap,
not a leakage artifact - the higher backbone_lr itself is the problem. Reverted to backbone_lr=1e-5.

### Exp 23: weight_decay 1e-4->1e-3

Motivated directly by exp 22: if higher backbone flexibility overfits the tiny val set, more L2
regularization seemed like a plausible way to buy back some of that generalization margin. Same config
as exp 20 otherwise (backbone_lr back at 1e-5).

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | weight_decay=1e-4 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 23 | weight_decay=1e-3 | 0.9693 | 0.9658 | 0.9497 | 0.9443 | 29 | 37 |

**No effect - not adopted.** val +0.0114 (about half the 0.0216 band, not a clear exceedance), test
essentially dead flat (macro_f1 -0.0002, mcc +0.0002). Per-class breakdown is nearly identical to the
baseline's, confirming this isn't just aggregate coincidence. Worse, it took substantially longer to get
there - best_epoch 29 vs. 14, 37 epochs run vs. 22 (~68% more compute) for zero test improvement, the
same "not worth it" shape as Phase 5 exp 13's old-dataset epochs finding. 10x more weight decay just
slows convergence without changing where it ends up. Reverted to weight_decay=1e-4.

### Exp 24: color_jitter_strength 0.2->0.0 (retest)

Phase 4 exp 9 rejected this exact change on the old dataset, but for a specific structural reason that no
longer applies: without jitter, the model could lock onto exact color values, and since train/val/test
were spatial regions of the *same* photo back then, one region's color statistics happening to align with
train's inflated an old-dataset-only artifact. Photo-level splits removed that mechanism entirely, so this
was worth a clean retest rather than assuming the old verdict still holds. Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | color_jitter_strength=0.2 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 24 | color_jitter_strength=0.0 | 0.9471 | 0.9408 | 0.9609 | 0.9568 | 28 | 36 |

**No effect - not adopted.** val -0.0108 and test +0.0110/+0.0127 - opposite directions again, but
unlike exp 22/23, *both* deltas sit comfortably inside their respective noise bands (0.0216 val, ~0.0286
test) this time, so this reads as genuinely flat rather than a real but small effect. Per-class breakdown
is clean either way (no class regressed, Kenya-AA actually ticked up to its best test f1 yet at 0.880).
Useful confirmation though: the old rejection really was about the spatial-split artifact specifically,
not jitter itself being necessary - removing it here doesn't reproduce anything like Phase 4 exp 9's
sharp val/test disagreement, it's just noise. Reverted to color_jitter_strength=0.2 (no clear reason to
drop it, and it's free regularization).

### Exp 25: dropout 0.2->0.4

Untested regularization knob. Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | dropout=0.2 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 25 | dropout=0.4 | 0.9637 | 0.9596 | 0.9581 | 0.9538 | 37 | 45 |

**Marginal, not adopted - not worth 2x cost.** All four metrics moved positively this time (unlike exp
23/24's mixed directions), but every delta (val +0.0058, test +0.0082/+0.0097) sits well inside the noise
bands - directionally encouraging but not distinguishable from noise on a single seed. Per-class: Kenya-AA
ticked up again to 0.889 (best yet, matching exp 24's pattern), no regressions. But it took roughly double
the compute to get there (best_epoch 37 vs 14, 45 vs 22 epochs run) - same shape as exp 23 and Phase 5
exp 13 before it. Reverted to dropout=0.2. Worth flagging: three regularization/augmentation knobs in a
row (weight_decay, jitter, dropout) have now each nudged in a mildly positive-but-noisy direction on
their own - if there's a real small effect being masked by single-seed noise here, it might only show up
combined, but combining untested single-variable nudges breaks the one-variable-at-a-time discipline, so
leaving this as a note rather than acting on it now.

### Exp 26: label_smoothing 0.0->0.1

Untested regularization knob. Same config as exp 20 otherwise. (Loss values are on a different scale than
usual, ~0.6-0.77 vs the typical ~0.01-0.3 - expected, not a bug: label smoothing raises cross-entropy's
achievable floor since targets are no longer one-hot.)

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | label_smoothing=0.0 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 26 | label_smoothing=0.1 | 0.9468 | 0.9410 | 0.9202 | 0.9097 | 16 | 24 |

**Reject.** val -0.0111 (inside its 0.0216 band), but test_macro_f1 -0.0297 and test_mcc -0.0344 both
clearly exceed their ~0.0286/0.0285 bands - a real effect, and val understates it. Per-class shows diffuse
damage rather than one collapse: Kenya-AA down to 0.800, Brazil-MonteCristo back down to 0.847 (undoing
exp 20's gain on exactly the class patch_crop_size=900 helped most). Label smoothing softens the target
distribution, which trades away some of the model's ability to be confidently correct on the harder,
more visually-similar classes - not a good trade here given how much of the remaining error is
concentrated in genuine class confusion (MonteCristo/Cerrado/Tata) rather than overconfidence. Reverted
to label_smoothing=0.0.

### Exp 27: train_patches_per_class 150->300 (retest)

The old dataset found this exact change had no effect (Phase 3 exp 6), but that was sampling more patches
from a single fixed photo per class - the reasoning for why it might matter now is different (14 real
train photos/class instead of 1, so more patches means covering more of that real variety, not just more
correlated crops of the same image). Worth a clean retest rather than assuming the old verdict transfers.
Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | train_patches_per_class=150 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 27 | train_patches_per_class=300 | 0.9583 | 0.9535 | 0.9524 | 0.9478 | 12 | 20 |

**No effect - not adopted.** Every delta is negligible (val +0.0004, test +0.0025/+0.0037), nowhere near
either noise band. It did converge in fewer epochs (best_epoch 12 vs 14, 20 vs 22 total) but each epoch
has 2x the batches (2700 vs 1350 patches), so total compute is still ~1.8x higher for a flat result. Same
verdict as the old dataset, and for basically the same reason once translated to the new setup: 150
patches/class already covers what's learnable from 14 photos, doubling the patch count doesn't add real
information, it just resamples the same underlying photos more densely. Reverted to
train_patches_per_class=150.

### Exp 28: freeze_mode none->last_block (retest)

Sanity check on a core architectural decision (Phase 2 found full fine-tune beat last_block on the old
dataset) now that patch_crop_size=900 and the new dataset have changed the picture substantially. Same
config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | freeze_mode=none (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 28 | freeze_mode=last_block | 0.9187 | 0.9097 | 0.8891 | 0.8795 | 25 | 33 |

**Clear reject, confirms the standing decision.** All four metrics well beyond their noise bands in the
worse direction (val -0.0392, test_macro_f1 -0.0608, test_mcc -0.0646). Full fine-tune's advantage over
last_block isn't just holding up at the new patch size, it's if anything more decisive than the original
Phase 2 finding. Reverted to freeze_mode=none. Useful negative result - confirms this isn't a decision
worth revisiting again without a real reason to.

### Exp 29: model_name resnet18->efficientnet_b0 (full fine-tune)

Old dataset only tested efficientnet_b0 frozen ("clearly worse than resnet18") - worth retesting under
full fine-tune since that's a very different regime (frozen features never adapted to this domain at all,
full fine-tune lets them). Same config as exp 20 otherwise (freeze_mode=none in both).

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | model_name=resnet18 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 29 | model_name=efficientnet_b0 | 0.9354 | 0.9284 | 0.9125 | 0.9038 | 17 | 25 |

**Clear reject.** val -0.0225 (just exceeds the 0.0216 band), test_macro_f1 -0.0374 and test_mcc -0.0403
(both clearly exceed their ~0.0286/0.0285 bands). Also ~1.5x slower per batch (~4s vs ~2.6s), so this is
worse *and* more expensive - unambiguous. Per-class picture is mixed rather than uniform though: Kenya-AA
actually improved to f1=0.930 (best result for that class in the whole log), but Brazil-Cerrado (0.821)
and Brazil-MonteCristo (0.763) both regressed notably, and those two dominate the aggregate. resnet18
remains the better backbone for this problem even with the field leveled by full fine-tuning. Reverted to
model_name=resnet18.

### Exp 30: combined dropout=0.4 + weight_decay=1e-3

Direct follow-up on the note left in exp 25: weight_decay=1e-3 (exp 23) and dropout=0.4 (exp 25) each
nudged all four metrics mildly positive on their own, both within noise individually. Testing whether
they compound when combined. Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | baseline (dropout=0.2, wd=1e-4) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 30 | dropout=0.4 + weight_decay=1e-3 | 0.9490 | 0.9441 | 0.9440 | 0.9382 | 9 | 17 |

**No effect - not adopted, and answers the open question from exp 25.** All four deltas are small and
slightly negative this time (val -0.0089, test -0.0059/-0.0059), well inside both noise bands. The two
individually-mild-positive nudges don't compound - combined, they're flat-to-marginally-worse, not
better. Interesting side note: it converged much faster this way (best_epoch 9, 17 total vs 14/22
baseline) for essentially the same quality, so if training speed mattered more than squeezing out the
last bit of accuracy, this combination would be a reasonable efficiency trade - it isn't given the
current priority is quality. Reverted both to their exp 20 values (dropout=0.2, weight_decay=1e-4).

### Exp 31: model_name resnet18->mobilenet_v3_small

Completes the architecture comparison (the third option implemented in `coffeecv/model.py`, never tested
in this log before). Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | model_name=resnet18 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 31 | model_name=mobilenet_v3_small | 0.8490 | 0.8321 | 0.8559 | 0.8380 | 12 | 20 |

**Clear reject, not close.** val -0.1089, test_macro_f1 -0.0940, test_mcc -0.1061 - several times each
noise band, the largest single-experiment gap in the whole log. Per-class damage is broad, not
concentrated in the usual hard classes alone (even Vietnam-Robusta, perfect in every other experiment,
dropped to 0.962). Makes sense: mobilenet_v3_small is designed for mobile/edge efficiency at the cost of
capacity, and this is a fine-grained visual task (subtle bean surface/color/texture differences) that
benefits from more model capacity, not less. resnet18 and efficientnet_b0 (exp 29) both clearly beat it.
Reverted to model_name=resnet18. Architecture question now closed for this project's three implemented
options - resnet18 wins outright.

### Exp 32: patch_crop_size 900->1000

Following the strongest signal in the whole log (bigger patches have won every time so far) toward the
practical ceiling - the tightest of the 180 crops is ~1048px, minus the 3% safety margin leaves ~1017px
of valid region, so 1000 leaves only ~17px of random-placement room on the tightest photos specifically.
Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | patch_crop_size=900 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 32 | patch_crop_size=1000 | 0.9426 | 0.9381 | 0.8968 | 0.8931 | 23 | 31 |

**Reject - the trend reverses.** val -0.0153 (inside its band), but test_macro_f1 -0.0531 and test_mcc
-0.0510 both clearly exceed their bands - a real regression, not noise. This brackets the optimum for the
first time: 700 (exp 20 baseline before) < 900 (best) > 1000 (worse) - a genuine sweet spot, not
monotonically-bigger-is-better after all. Per-class points at the likely mechanism: Kenya-AA collapsed to
f1=0.667 (its worst result anywhere in this log). At 1000px, the tightest photos in the dataset have
almost no room left for randomized patch placement, so patches sampled from those specific photos become
nearly identical to each other every time - losing translation-augmentation diversity precisely for
whichever photos happen to be near the small end of the size distribution, rather than a general "too
much context" problem. Reverted to patch_crop_size=900 - confirmed as the adopted optimum, not just the
best value tried so far.

### Exp 33: batch_size 32->64 (retest)

Quick close-out retest of Phase 5 exp 12's old-dataset "no effect" finding, on the now-adopted config.
Same config as exp 20 otherwise.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | batch_size=32 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 33 | batch_size=64 | 0.9547 | 0.9503 | 0.9442 | 0.9381 | 24 | 32 |

**No effect - not adopted, confirms the old finding transfers.** All four deltas small and negative
(val -0.0032, test -0.0057/-0.0060), well inside noise. Took ~1.45x more compute to get there
(best_epoch 24 vs 14). Same verdict as Phase 5 exp 12 on the old dataset. Reverted to batch_size=32.

### Run summary (stopped at 14h4m of the 16h budget, 2026-08-07 18:30 -> 2026-08-08 08:34 UTC)

14 experiments (exp 20-33), one variable at a time, every result (adopted or rejected) committed
individually with `dvc repro`/`dvc checkout` keeping `dvc.lock` in sync. Stopped short of the full 16h
once the sweep had covered every category with a reasonable candidate (patch geometry, both learning
rates, four regularization knobs alone plus one combined, data density, all three implemented
architectures, and the two structural sanity-checks) rather than filling time with lower-value reruns.

**Adopted (1)**: `patch_crop_size` 700->900 (exp 20) - the only clear, robust win. Also the only variable
that showed a real *bracketed optimum* rather than a flat monotonic trend or pure noise: exp 32 pushed to
1000 and the trend reversed (test dropped well beyond its noise band), pinned down by a well-supported
mechanism (the tightest photos in the dataset losing randomized-placement room near the crop-size
ceiling), not just "went too far." 700 < 900 (best) > 1000.

**Rejected, real effects found (4)**: `backbone_lr` 3x up (exp 22, val/test disagree - looks like
overfitting a 3-photo/class val set), `label_smoothing` 0.1 (exp 26, diffuse per-class damage), and both
alternative architectures - `efficientnet_b0` (exp 29) and `mobilenet_v3_small` (exp 31, the largest gap
in the whole log) - plus the `freeze_mode=last_block` sanity-check (exp 28), which confirmed the standing
full-fine-tune decision holds up (more decisively, even) at the new patch size.

**No measurable effect (8)**: head `lr` 2x (exp 21), `weight_decay` 10x (exp 23) and `dropout` 2x
(exp 25) each nudged all-metrics mildly positive alone but the combination (exp 30) was flat -
individually-noisy nudges don't compound. `color_jitter_strength`=0 (exp 24, a clean retest now that the
old dataset's spatial-split confound is gone) and `train_patches_per_class` 2x (exp 27, retested given
richer per-photo data) both transferred their old-dataset "no effect" verdict cleanly. `batch_size` 2x
(exp 33) also reconfirmed its old verdict. Several of the "no effect" tests cost meaningfully more compute
for the same or worse quality (weight_decay, dropout, batch_size all took ~1.5-2x longer to plateau) -
worth remembering as a reason *not* to adopt a change even when it isn't harmful.

**Standing config after this run**: identical to the Phase 6/exp 18 best except `patch_crop_size=900`
(was 700). Current numbers (exp 20, the last time this exact config was retested from scratch):
val_macro_f1=0.9579, test_macro_f1=0.9499, test_mcc=0.9441, best_epoch=14/22 with early stopping.

**Open threads for later**:
- Brazil-MonteCristo remains the hardest class across every experiment in this run (f1 in the 0.76-0.94
  range depending on config) - a data problem (more/different capture angles for that class specifically)
  more than a hyperparameter one at this point, given how consistently hyperparameter changes moved it
  the same direction as everything else rather than fixing it specifically.
- No multi-seed check has been run on the patch_crop_size=900 config yet - everything in this run is
  single-seed (seed=42), same discipline as Phases 1-5 originally used, but worth eventually confirming
  the noise band still looks like exp 16-17's characterization at the new patch size before fully trusting
  it, especially since exp 32's finding shows patch_crop_size interacts with per-photo geometry in a way
  that could plausibly also affect variance, not just the mean.
- `cfg.scheduler` in params.yaml is dead config - `train_baseline.py` hardcodes `CosineAnnealingLR`
  regardless of its value. Not tested here since trying an alternative would need new code, not just a
  value change; noted as a gap rather than fixed.
- `git push` has not worked all run (no SSH key/agent in this container) - everything above is committed
  locally only, pending credentials to actually reach `origin`.

## Multi-seed check on patch_crop_size=900 (closes Phase 7)

The open thread flagged above: exp 32 showed patch_crop_size interacts with per-photo geometry (tighter
photos lose randomized-placement room near the ceiling), which could plausibly affect variance and not
just the mean - worth confirming exp 20's numbers are representative before trusting them the way exp
16-17 did for the pre-900 config. Same two extra seeds as every other multi-seed check in this log
(123, 7), same config as exp 20 otherwise.

### Exp 34: seed=123

| # | seed | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch |
|---|---|---|---|---|---|---|
| 20 | 42 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 |
| 34 | 123 | 0.9604 | 0.9569 | 0.9020 | 0.8931 | 23 |

val essentially matches (+0.0025), but test_macro_f1 -0.0479 and test_mcc -0.0510 - already larger than
the *entire* noise band exp 16-17 measured for the pre-900 config (test spread 0.0286). Per-class shows
this is a genuine reshuffle, not one collapse: Colombia-PinkBourbon (0.822) and CostaRica-LaPastora
(0.806) - both near-perfect at seed 42 - are the weak points here, while Brazil-Cerrado (0.988) and
Brazil-MonteCristo (0.886) are unusually strong for once. Consistent with the original multi-seed check's
finding that different seeds resolve different confusions, but the magnitude here is already a warning
sign that patch_crop_size=900's noise band may be wider than 700's was - one more seed needed before
concluding that, not enough on its own.

### Exp 35: seed=7

| # | seed | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch |
|---|---|---|---|---|---|---|
| 20 | 42 (baseline) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 |
| 34 | 123 | 0.9604 | 0.9569 | 0.9020 | 0.8931 | 23 |
| 35 | 7 | 0.9723 | 0.9689 | 0.9183 | 0.9102 | 30 |

Third and last per the same 3-seed convention as every prior check. Per-class is a third distinct
reshuffle: Ethiopia-Sidamo/Kenya-AA/Colombia-PinkBourbon/CostaRica-LaPastora all strong (0.96-0.98) here,
Guatemala-Tata (0.767) and Brazil-MonteCristo (0.779) the weak points this time. Three seeds, three
different sets of "hard" classes - the aggregate score is stable-ish but *which* classes it's weak on
each run is not.

### Noise band on patch_crop_size=900 (3 seeds) - confirms the suspicion, closes Phase 7

| metric | min | max | spread | mean |
|---|---|---|---|---|
| val_macro_f1 | 0.9579 | 0.9723 | 0.0144 | 0.9635 |
| val_mcc | 0.9533 | 0.9689 | 0.0156 | 0.9597 |
| test_macro_f1 | 0.9020 | 0.9499 | **0.0479** | 0.9234 |
| test_mcc | 0.8931 | 0.9441 | **0.0510** | 0.9158 |

**val actually got *tighter*** than the pre-900 config (0.0144 vs exp 16-17's 0.0216), but **test got
~1.7-1.8x *wider*** (0.0479/0.0510 vs 0.0286/0.0285). This confirms exp 32's warning: patch_crop_size=900
is still the best mean performer of anything tested in this project (test_macro_f1 mean 0.9234 across 3
seeds, vs. patch_crop_size=700's exp 16-17 mean of 0.8973), but it's a *less consistent* one than 700 was
- exp 20's single-seed 0.9499 was a favorable draw within a real 0.90-0.95 band, not a tight point
estimate. Best guess at the mechanism (not confirmed further, same caveat as exp 32): best.pt selection
is peak-val-driven, val itself stayed just as consistent as before, but *which specific epoch* gets picked
and how well that epoch's weights transfer to a different set of 3 test photos/class seems to vary more
at 900px - plausibly connected to the same tight-photo/placement-room issue exp 32 found, since which
photos land in val vs. test differs by seed (photo-to-split assignment is itself seeded per class).

**Practical takeaway**: patch_crop_size=900 remains adopted - it's unambiguously better on average than
700 (0.9234 vs 0.8973 mean test_macro_f1, no overlap between the two 3-seed ranges) - but report the range
(test macro-F1 ~0.90-0.95), not a single run's number, when describing this model's expected performance.
This closes the open thread from the 16h run and, with it, Phase 7 - dataset switch, hyperparameter sweep,
and now a properly characterized noise band on the final adopted config, mirroring how Phase 6 closed on
the old dataset. Phase 8 not yet started/scoped (candidates discussed with the user: chasing
Brazil-MonteCristo's persistent weakness with more capture data, or validating against real unlabeled
photos via the existing `coffeecv/infer.py` - to be decided in a future session).

# Phase 8 (plan) - Dataset augmentation

Written 2026-08-08 for a future session to pick up cold. Focus: augmentation, not more hyperparameter
tuning of the existing knobs (Phase 7 exhausted the reasonably-testable ones - see its closing summary).
User's brief: concentrate on augmentation, at minimum test rotations and zoom, plus other hypotheses.

**Starting point**: the Phase 7-adopted config as of exp 20/35 (resnet18, freeze_mode=none,
patch_crop_size=900, epochs=50/early_stop_patience=8, lr=1e-3, backbone_lr=1e-5, weight_decay=1e-4,
dropout=0.2, label_smoothing=0.0, color_jitter_strength=0.2, batch_size=32, seed=42). All augmentation
changes below are train-transform-only (`coffeecv/transforms.py:build_train_transform`) - `build_eval_transform`
stays deterministic/unaugmented, same as every phase so far.

**Decision rule stays the same discipline as Phases 1-7, but use the wider, patch_crop_size=900-specific
noise band** characterized in the section just above (val spread 0.0144, test_macro_f1 spread 0.0479,
test_mcc spread 0.0510) as the significance bar - not the tighter pre-900 band from exp 16-17, which no
longer applies to this config. Single seed=42 first pass per hypothesis, matching every phase so far;
anything that looks like a real win should get a 3-seed check before being called adopted, given how much
noise-band width itself has already moved once in this project (exp 32/34/35).

## Hypothesis queue

1. **Arbitrary-angle rotation (required minimum).** Current augmentation only does 0/90/180/270°
   (`RandomRightAngleRotation` in transforms.py) - exact and lossless, deliberately chosen to avoid
   interpolation/border artifacts. The domain rationale for rotation at all ("top-down photos of a bean
   pile, no canonical up" - transforms.py's own docstring) applies just as well to arbitrary angles, not
   just multiples of 90. Open implementation question to resolve at the top of the session: arbitrary
   rotation of an already-square crop_size x crop_size patch leaves corners with no valid content, so
   either (a) oversample a larger source crop at the dataset.py level (e.g. crop_size * 1.4) with margin
   to rotate-then-center-crop back down with zero artifacts, or (b) rotate at the transform level with a
   fill/border strategy and accept minor edge artifacts for a much simpler change confined to
   transforms.py. Start with (b) for a fast first test (e.g. angle range +/-25 degrees) since it's a
   same-file change; only invest in (a) if (b) shows promise but the border artifacts look like they're
   capping the effect.

2. **Zoom / scale augmentation (required minimum).** Directly motivated by Phase 7's single biggest
   finding - patch_crop_size massively affects results, with a real (if noisy) optimum at 900. Cheapest
   version: add a `RandomResizedCrop`-style step in `build_train_transform` that zooms into a random
   sub-region of the already-extracted crop_size x crop_size patch (e.g. scale=(0.7, 1.0) of area) before
   the final resize to 224 - pure transforms.py change, no dataset.py/geometry.py touch needed. This is
   "digital zoom" within a fixed physical patch, not literally varying the real-world field of view
   captured. A heavier follow-up, only worth it if the light version underdelivers: draw crop_size itself
   from a range per patch at the dataset.py sampling stage (touches `MultiPhotoPatchDataset`/
   `sample_patch_boxes`, which currently precompute one fixed-size box per sample at construction time) -
   this would give genuine multi-scale training and might explain/absorb some of the per-photo variance
   exp 32/34/35 found sensitive to the exact crop_size choice, rather than just resampling within one
   fixed-size crop.

3. **Random erasing / Cutout.** Randomly mask small rectangular regions of the patch during training.
   Forces the model to rely on distributed texture cues rather than a few salient beans - worth trying
   specifically because Brazil-MonteCristo (and its confusions with Guatemala-Tata/Brazil-Cerrado) have
   been the one persistent weakness that no hyperparameter change in Phase 7 fixed. If this helps anything,
   it's the best candidate to check per-class, not just the aggregate.

4. **Mild perspective/affine jitter.** The capture rig is fixed/tripod-mounted (confirmed in Phase 7's
   crop-pipeline work - rough tray bounding box varies <2% across all 180 photos), so there's zero
   real camera-angle diversity in the training data as-is. A small synthetic perspective warp could help
   generalize to a *future* capture session that isn't perfectly perpendicular, at the cost of being
   pure synthetic diversity with no matching real examples to validate against yet.

5. **Illumination/vignette augmentation.** Grounded in something concretely observed in this exact
   project, not generic advice: the original adaptive crop heuristic broke specifically because of real
   directional lighting drift over the ~2h 2026-08-07 capture session (see Phase 7's crop-pipeline
   writeup). A synthetic lighting-gradient/vignette augmentation during training could build robustness to
   whatever lighting a *future* session has, rather than just the (now-corrected-for) lighting this one
   session happened to have.

6. **Mixup / CutMix (lower priority, exploratory).** Well-established regularizers, but the physical
   interpretation is less clean here than for natural image classes - blending two different origins'
   bean piles doesn't obviously correspond to anything real. Worth a quick single test if budget remains
   after the above, not a priority.

**Not a fresh hypothesis, skip unless something above motivates revisiting it**: color jitter *intensity*
was already swept (0.0 vs 0.2) at this exact patch_crop_size in Phase 7 exp 24, no effect found. Only
worth another look if, e.g., the zoom or rotation work changes what the model is sensitive to enough to
plausibly change that verdict.

# Phase 8 (running) - 12h autonomous augmentation run

Started 2026-08-08 18:45 UTC, budget ends ~2026-08-09 06:45 UTC. User's brief: execute the Phase 8 plan
above, ~12h first pass, review together afterwards and continue in later sessions. Four scoping decisions
taken with the user at the top of the session:

1. **Breadth, then confirm**: single-seed (42) first pass over all six queued hypotheses, then a 3-seed
   check on the best candidate, then a combined run if budget remains. Same discipline as Phases 1-7.
2. **Hypotheses 4-5 (perspective, illumination) judged on a "costs nothing" bar**: they target robustness
   to a *future* capture session, and the current test set is same-rig/same-session, so it structurally
   cannot validate them. Adopt if in-distribution metrics stay flat (free robustness is worth having),
   and record explicitly that the robustness win itself stays unvalidated until a second session exists.
3. **Rotation uses the fill-based variant** - see the finding below, which resolved the plan's open
   implementation question in a way the plan didn't anticipate.
4. **Every experiment's metrics get kept in version control**, not just summarized in this file.

## Phase 8 setup (code changes, no training)

Six knobs added, each defaulting to a no-op: `rotation_degrees`, `zoom_scale_min`, `random_erasing_p`,
`perspective_distortion`, `illum_gradient_strength` (all in `transforms.py:build_train_transform`) and
`mixup_alpha` (in `train_baseline.py:train_one_epoch` - mixup is a loss-level change, not a transform).
`build_eval_transform` untouched, as the plan requires. Rotation later moved out of `transforms.py` entirely -
 see the section below.

**The no-op default is load-bearing and was verified, not assumed.** Phase 8 compares against exp 20's
recorded numbers instead of re-deriving them, which is only legitimate if the augmentation-off pipeline is
*identical* to Phase 7's - a stray extra RNG draw would silently shift every result and quietly invalidate
every verdict. `coffeecv/check_augmentation.py` asserts this bit-exactly: it reads the pre-Phase-8
`transforms.py` straight out of git (ref a87c260), runs both pipelines over the same patches under the
same seeds, and requires `torch.equal`. It also asserts the converse - that each knob *does* change the
output when enabled - so a silently-dead param can't masquerade as a "no effect" result. All checks pass.
(Exp 37 below re-verifies this end-to-end through a full training run, not just at the transform level.)

### Finding: the plan's preferred rotation implementation is impossible on this dataset

The plan left open whether to (a) oversample a ~1.4x larger source crop at the `dataset.py` level and
rotate-then-center-crop back artifact-free, or (b) rotate at the transform level with a fill strategy and
accept border artifacts. **(a) cannot be done here at all.** The cropped photos are only ~1050x1520px, so
after `safety_margin=0.97` the valid region is ~1018px wide - and `patch_crop_size=900` already consumes
all but ~118px of that horizontally. There is no headroom to oversample into: a 900px patch needs a
1278px source to survive a +/-25 degree rotation, and even +/-15 degrees would need 1102px. Neither fits
in 1018px. This is the same geometric ceiling exp 32 ran into from the other direction (patch_crop_size
1000 left only ~17px of placement room and the trend reversed) - worth recording as one shared constraint
rather than two coincidences: **this dataset's photo width is the binding limit on the whole patch
pipeline**, and any future capture session that framed the tray slightly wider would relax both at once.

The first attempt at (b) filled the corner wedges with the patch's own mean RGB, on the reasoning that it
blends better than black. Measured cost (white-square test, share of the 900px frame that is fill): 4.4%
mean at +/-10 deg, 6.1% at +/-15, **8.9% at the plan's suggested +/-25**, 12.2% at +/-45. That run was
launched and then **aborted by the user, who rejected fill-based rotation outright** - correctly. A bean
pile is *nothing but* texture, so ~9% of every patch becoming flat, information-free filler is not a minor
border artifact here, it is deleting a tenth of the signal and asking the model to ignore the hole.

### The design that actually works: small-angle jitter sampled from the source photo

**User's proposal, and it dissolves the problem the plan and both of my alternatives were stuck on**: keep
the exact 0/90/180/270 rotations, and jitter each by a *small* angle - small enough that the rotated patch
still fits inside the source photo. The insight is that the plan's option (a) was never impossible in
general, only impossible *at +/-25 degrees*. Shrink the angle and the headroom appears:

| jitter | bounding box needed | placement room left on the tightest photo |
|---|---|---|
| +/-0 deg (Phase 7) | 900px | 116px |
| +/-3 deg | 946px | 70px |
| +/-5 deg | 976px | **40px** |
| +/-6 deg | 990px | 26px |
| +/-7 deg | 1003px | 13px - exp 32's danger zone |
| +/-8 deg | 1017px | does not fit at all |

So the implementation moved out of `transforms.py` and into the *sampling* stage
(`geometry.sample_rotated_patch_boxes`): draw an angle, take the axis-aligned **bounding box** of the
rotated 900px square from the source photo, rotate that, centre-crop back to 900. Every pixel of the
result is real source content, the patch stays exactly 900px, and there is no fill anywhere - the extra
content the rotation needs is *borrowed from the surrounding photo* instead of invented. This also avoids
the scale confound that the inscribed-crop alternative would have introduced (it would have shrunk the
effective patch to 677px, colliding head-on with patch_crop_size, the strongest known variable in the
project).

Verified directly rather than argued: `check_augmentation.py` runs the real crop-rotate-centre-crop path
over an all-white source at +/-1/3/5/6 degrees and asserts **zero** fill pixels reach the output. (A
black-pixel test on real photos would not have worked - the unrotated patches already contain 1000-8000
pure-black pixels each, in the shadows between beans. Worth remembering as a trap for any future
fill-detection check.)

`assert_jitter_fits` fails loudly at dataset construction if the chosen jitter leaves under 25px of
placement room, so exp 32's failure mode (patches from the tightest photos becoming near-identical every
epoch) can't be reintroduced silently by a future angle change.

**+/-5 degrees chosen for the first test**: the largest angle still clear of exp 32's danger zone, so the
hypothesis gets its best chance to show an effect. Known confound to weigh against a *negative* result:
+/-5 also cuts placement room 116px -> 40px, so "rotation doesn't help" and "the lost translation
diversity hurt" are not separable from this run alone. If it comes back negative, the clean control is a
no-rotation run sampled from a region shrunk to leave the same 40px of room.

## Experiment record now kept in version control

Phase 7's metrics survived only as the hand-written tables in this file: `outputs/` is gitignored and
overwritten by every run, and DVC only ever holds the *latest* run's outputs. Fixed for Phase 8 - split by
what each tool is good at, **git for the small stuff, DVC for the bulk**:

- `experiments/exp<N>__<slug>/` - metrics/config/history/predictions per run (~16KB), committed to git.
  `index.csv` is rebuilt by scanning the archived runs, so it cannot drift out of sync.
- 44MB checkpoints stay in DVC via `dvc.lock`, which is what DVC is for.
- `outputs/metrics.json` un-ignored: `dvc.yaml` already declared it `cache: false` (i.e. "versioned in
  git"), while `/outputs/` in `.gitignore` was quietly overriding that, leaving it tracked by neither.
  **Correction (found at end of run, 06:05)**: the first attempt at this fix did not work and was reported
  as done when it wasn't. `.gitignore` had `/outputs/` followed by `!/outputs/metrics.json`, and git
  **cannot re-include a file whose parent directory is excluded** - a negation is silently dead against a
  directory pattern. So metrics.json stayed untracked for the whole run. Fixed properly by changing the
  pattern to `/outputs/*` (excluding the directory's *contents*, not the directory), after which
  `git check-ignore` confirms metrics.json is trackable and history.json / config.json / predictions /
  checkpoints all remain ignored. No results are affected - the `experiments/` archive was the primary
  mechanism and worked correctly throughout, holding all 10 runs. Lesson worth keeping: verify a
  `.gitignore` negation with `git check-ignore -v`, never by reading the file.
- `experiments/pre_phase8_from_log.csv` backfills exp 18-35 by hand from this file. Aggregate metrics only
  - those runs' artifacts are gone, so no per-class data or curves could be recovered.

**Knock-on effect on hypothesis 4 (perspective jitter)**: it had the same mean-fill problem, so its
implementation was removed along with rotation's rather than left in as dead config. A perspective warp
needs source margin too, and the same trick applies - warp within a slightly larger sampled box and crop
back - but a mild `distortion_scale=0.2` needs up to ~90px of margin, which would leave ~26px of placement
room, right at exp 32's edge. Hypothesis 4 therefore needs either a smaller distortion or an explicit
decision to accept that trade; it is not testable as originally written. Deferred, not silently dropped.

### Exp 36: rotation_jitter_degrees 0 -> 5

First test of the redesigned rotation (small-angle jitter sampled from the source photo, zero invented
pixels). Same config as exp 20 otherwise. Ran 19:24-20:27 UTC, 63 min.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | baseline (no jitter) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 36 | rotation_jitter_degrees=5.0 | 0.9638 | 0.9595 | 0.9384 | 0.9323 | 19 | 27 |

**No measurable effect - not adopted.** val +0.0059/+0.0062, test -0.0115/-0.0118. Every delta is well
inside the patch_crop_size=900 noise band (val 0.0144, test 0.0479/0.0510) - the largest is under a
quarter of its band - and the two splits point in opposite directions, the shape of noise rather than a
small real effect. It also cost ~23% more compute to get there (best_epoch 19 vs 14, 27 epochs vs 22),
the same "not worth it even if it were free" pattern as exp 23/25/27/33.

Per-class test f1: CostaRica-LaPastora, Ethiopia-Kochere and Vietnam-Robusta all 1.000; Guatemala-Tata
0.964, Colombia-PinkBourbon 0.961; the weak end is Kenya-AA 0.865, Brazil-Cerrado 0.873, Brazil-MonteCristo
0.874. MonteCristo is down from the 0.929 exp 20 recorded, but that class swung 0.76-0.94 across Phase 7
depending on nothing in particular, so this is not evidence of anything on its own.

**Caveat that survives this result**: +/-5 degrees also cut placement room 116px -> 40px, so in principle a
real rotation gain and a real translation-diversity loss could be cancelling. Two reasons not to spend a
run chasing that now: the result is *flat*, not negative, so any cancelling pair would both have to be
small; and val moved up while test moved down, which is the signature of noise rather than of two opposed
mechanisms. Recorded as the natural follow-up if rotation ever looks worth revisiting: +/-3 degrees keeps
70px of room, so it separates the two at a milder rotation.

### Exp 37: baseline reproduction (all augmentation off)

Not a hypothesis test. Two jobs: prove the Phase 8 refactor really is the no-op the transform-level check
claimed, through a full training run rather than at the transform boundary only; and archive a *per-class*
baseline, since exp 20 survived only as four aggregate numbers in a markdown table and Phase 7's most
useful findings were all per-class. Ran 20:29-21:15 UTC, 46 min.

| # | run | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 20 | original (Phase 7) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 37 | reproduction (Phase 8 code) | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |

**Bit-for-bit identical** - all four metrics to 4 decimals, both epoch counts, on code that has since
gained six augmentation knobs, a rewritten `transforms.py`, a new sampling path in `dataset.py` and two
new `geometry.py` functions. Everything Phase 8 compares against exp 20 is therefore comparing against a
number this environment still produces, not a historical artifact. Also confirms nothing drifted
environmentally since 2026-08-07 (same torch 2.13.0+cpu / torchvision 0.28.0+cpu).

Worth the ~46 min: every later experiment now gets a real per-class delta instead of an aggregate one, and
`compare_experiments.py` has something to diff against.

**Exp 36 re-read against the proper baseline** (now possible per-class): the aggregate verdict is
unchanged - all four deltas flat - but the per-class picture is a clean illustration of why aggregate
noise is not "nothing happening". Test moved almost entirely through the two Brazils (Cerrado -0.060,
MonteCristo -0.055) while *val* moved MonteCristo the other way (+0.026) and Guatemala-Tata up (+0.029).
The same class going up on val and down on test in one run is the same pattern exp 34/35 found across
seeds: which specific confusions resolve is unstable run to run, even where the aggregate is stable.
Five classes are pinned at or near 1.000 on test in both runs (CostaRica-LaPastora, Ethiopia-Kochere,
Vietnam-Robusta perfect; Colombia-PinkBourbon ~0.95-0.96), so essentially all remaining headroom on this
dataset is the Brazil-Cerrado / Brazil-MonteCristo / Kenya-AA cluster.

### Exp 38: zoom_scale_min 1.0 -> 0.7 (RandomResizedCrop, area 0.7-1.0)

The plan's second required-minimum hypothesis, in its light form: zoom into a random sub-region of the
already-extracted 900px patch before the final resize to 224. `ratio` pinned at 1.0 so this is a pure
zoom and doesn't smuggle in aspect distortion as a second variable. Same config as exp 37 otherwise.
Ran 21:16-22:55 UTC, 99 min.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 37 | baseline | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 38 | zoom_scale_min=0.7 | 0.9693 | 0.9657 | 0.9213 | 0.9169 | 40 | 48 |

**Not adopted.** Every delta is technically inside its noise band, so the honest aggregate verdict is "no
measurable effect" - but this is the least reassuring possible version of that. val moved +0.0114/+0.0124,
about 80% of the way to its band; test moved -0.0286/-0.0272, and the two splits disagree in direction.
That is exp 22's signature (backbone_lr 3x), not exp 24's clean flatness. It also cost **2.2x the compute**
(best_epoch 40 vs 14, 48 epochs vs 22) and nearly hit the 50-epoch cap, so early stopping barely saved it.

**Mechanism, and it follows directly from Phase 7's strongest finding.** `scale` is an area fraction, so
area 0.7-1.0 is *linear* 0.837-1.0 - the model trains on effective fields of view between ~753px and
900px. Phase 7 established that patch scale is the single most consequential variable in this project and
that 900 is a real bracketed optimum: 700 was worse (exp 18 vs 20), 500 clearly worse (exp 19), 1000
clearly worse (exp 32). Zooming in therefore doesn't add neutral diversity - it spends most of training
at scales *known to be worse than the one being evaluated*, and `build_eval_transform` is deterministic at
the full patch, so it also opens a train/eval scale mismatch that didn't exist before.

Per-class test damage lands exactly where that predicts - on the classes that need the most context:
Kenya-AA -0.141 (0.868 -> 0.727, its worst since exp 32's 0.667, which was also a patch-geometry failure)
and **CostaRica-LaPastora -0.082, the first time that class has scored below 1.000 anywhere in this log**.
Meanwhile val shows Guatemala-Tata +0.094 and Kenya-AA *+0.023* - the same class moving opposite ways on
the two splits again.

**Not worth a milder retest** (e.g. 0.9): the mechanism above says a smaller zoom range is just a weaker
dose of the same mismatch, not a different treatment. The plan's heavier follow-up - drawing `crop_size`
itself per patch at the sampling stage - is genuinely different in one respect (it can sample *larger*
than 900, up to the ~1016px valid-region ceiling, instead of only smaller), so it remains open; but it
inherits the same fixed-scale-eval mismatch, so it is not an obvious win either. Recorded, not run.

### Exp 39: random_erasing_p 0.0 -> 0.5

Hypothesis 3: mask small rectangles during training to force reliance on distributed texture rather than
a few salient beans. Erased region is 2-15% of patch area, applied after `Normalize` so the hole is the
ImageNet mean colour - unlike the rejected rotation fill, an information-free region is the *point* here,
not an artifact. Same config as exp 37 otherwise. Ran 22:57-00:12 UTC, 75 min.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 37 | baseline | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 39 | random_erasing_p=0.5 | 0.9635 | 0.9597 | 0.9554 | 0.9509 | 29 | 37 |

**Mildly positive on every metric, but inside noise - the best Phase 8 candidate so far, and the only one
that has earned a multi-seed check.** val +0.0057/+0.0064, test +0.0056/+0.0068. Each delta is ~40% of its
val band and ~12% of its test band, so no single number is remotely significant on its own. What separates
this from exp 36 and 38 is *coherence*: all four metrics moved the same direction, and val and test agree
for the first time in Phase 8. Cost ~1.7x compute (best_epoch 29 vs 14).

Per-class test is where it's most interesting, and this was the hypothesis the plan specifically said to
check per-class rather than in aggregate. Five classes up, and the gains land on the exact cluster exp 37
identified as holding all the remaining headroom: **Kenya-AA +0.036 (0.868 -> 0.904)**, its best result
anywhere in Phase 8 and close to its all-time best (0.930, exp 29); Colombia-PinkBourbon +0.026,
Ethiopia-Sidamo +0.015, Guatemala-Tata +0.012 (0.988). The three perfect classes stay perfect. Against
that, Brazil-Cerrado -0.029 and MonteCristo -0.009 - so it did *not* fix the Brazil confusion the
hypothesis was originally aimed at; it helped Kenya-AA instead, which was the other member of that cluster.

**Explicit caution before reading too much into this.** Phase 7 exp 25 (dropout 0.4) looked almost
identical - all four metrics mildly positive, all inside noise, ~2x compute - and when exp 30 combined it
with the other mild-positive nudge, the combination was flat. Individually-noisy positives have already
failed to compound once in this project. So: 3-seed check queued, and the verdict stays open until it runs.

### Exp 40: illum_gradient_strength 0.0 -> 0.2

Hypothesis 5, judged on the "costs nothing" bar agreed at the top of this run: a random-direction linear
luminance ramp spanning +/-20% corner to corner. Grounded in something real from this project - directional
lighting drift over the 2h 2026-08-07 capture session is what broke the original adaptive crop heuristic -
rather than generic advice. Deliberately *spatial*: uniform brightness is already covered by ColorJitter,
so a flat offset would add nothing new. Same config as exp 37 otherwise. Ran 00:13-00:55 UTC, 42 min.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 37 | baseline | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 40 | illum_gradient_strength=0.2 | 0.9439 | 0.9377 | 0.9468 | 0.9412 | 12 | 20 |

**Does not clear the "costs nothing" bar - not adopted, but not a clear reject either.** test is genuinely
flat (-0.0030/-0.0030, ~6% of its band, the flattest test result in Phase 8). val is not: -0.0140 sits
exactly on its 0.0144 band edge and val_mcc -0.0157 just crosses its 0.0156 band. Under this project's
standing decision rule - val is the primary signal, being both tighter and the checkpoint-selection metric,
with test as directional confirmation - that reads as a small *real* cost rather than noise, which is
precisely what the "free robustness" bar was set up to exclude. It was however the cheapest run in Phase 8
(20 epochs vs 22 baseline), so the cost is quality, not compute.

Per-class is consistent across both splits, which is why this looks real rather than like a draw: the
damage concentrates on Brazil-MonteCristo (val -0.062, test -0.026) and Guatemala-Tata (val -0.022, test
-0.023), while Kenya-AA, Colombia-PinkBourbon and Ethiopia-Sidamo all tick mildly *up* on both. So the
gradient isn't uniformly harmful - it specifically costs the classes distinguished by subtler shading
cues, which is a coherent mechanism: a synthetic luminance ramp is exactly the kind of signal that would
wash out real shading differences between two similar brown bean piles.

**What this does and does not settle.** It does not test the hypothesis's actual claim - robustness to a
*future* session's lighting - which this dataset structurally cannot evaluate, since train/val/test are
all the same 2h shoot. It only measures the in-distribution price, which was the point of the "costs
nothing" bar. The price is small but probably real, and the benefit remains entirely unvalidated. Verdict:
leave it off, and revisit only once a second capture session exists to validate against - at which point
the right test is "does a model trained with this do better on the *new* session", not this table.

### Exp 41: mixup_alpha 0.0 -> 0.2

Hypothesis 6, the plan's own lowest-priority item and the last of the first pass. Beta(0.2, 0.2) batch
mixing with the correspondingly weighted loss against both label sets - a loss-level change in
`train_one_epoch`, not a transform. Same config as exp 37 otherwise. Ran 00:56-01:51 UTC, 55 min.

| # | change | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 37 | baseline | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 41 | mixup_alpha=0.2 | 0.9517 | 0.9472 | 0.9391 | 0.9320 | 18 | 26 |

**No effect - not adopted.** All four metrics mildly negative (val -0.0062/-0.0061, test -0.0108/-0.0121),
every one comfortably inside its band. Uniformly-negative-but-inside-noise is a cleaner null than exp 38's
split disagreement: nothing here suggests a real effect in either direction, just a mild consistent drag.
~1.2x compute.

Per-class test is mildly negative almost everywhere (Kenya-AA -0.042, Ethiopia-Sidamo -0.030, four more
slightly down, three perfect classes unmoved, only Colombia-PinkBourbon +0.012) - diffuse rather than
concentrated, matching the aggregate. Notably it moves Kenya-AA the *opposite* way from exp 39's random
erasing (+0.036), so the two regularizers are not interchangeable despite both being "hide information
from the model" strategies.

Consistent with the plan's own stated scepticism: blending two origins' bean piles doesn't correspond to
anything physical, unlike occluding part of one pile (exp 39), which is just a bean the camera didn't see.
That distinction now has a small piece of evidence behind it rather than being purely a priori.

## Exp 42-43: 3-seed confirmation of random erasing - ADOPTED

Exp 39's single-seed result was mildly positive on all four metrics but far inside every noise band, so the
verdict was held open. Seeds 123 and 7 run with `random_erasing_p=0.5`, everything else at exp 37's config.
Crucially this is a **paired** comparison: Phase 7 already measured the un-augmented baseline at these exact
seeds (exp 34 seed 123, exp 35 seed 7, exp 20/37 seed 42), so each run is compared against its own
same-seed baseline rather than against a wide single-run band. Ran 01:52-04:03 UTC, 54 + 77 min.

| seed | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc |
|---|---|---|---|---|
| 42 (exp 37 -> 39) | +0.0056 | +0.0064 | +0.0055 | +0.0068 |
| 123 (exp 34 -> 42) | +0.0313 | +0.0338 | +0.0078 | +0.0087 |
| 7 (exp 35 -> 43) | +0.0026 | +0.0031 | +0.0370 | +0.0401 |

| | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc |
|---|---|---|---|---|
| baseline mean (exp 20/34/35) | 0.9635 | 0.9597 | 0.9234 | 0.9158 |
| erasing mean (exp 39/42/43) | 0.9767 | 0.9741 | **0.9402** | **0.9343** |
| delta of means | +0.0132 | +0.0144 | +0.0168 | +0.0185 |
| erasing 3-seed spread | 0.0281 | 0.0309 | 0.0456 | 0.0491 |
| baseline 3-seed spread | 0.0144 | 0.0156 | 0.0479 | 0.0510 |

**ADOPT - the first adoption of Phase 8, and the best-evidenced adoption in this project so far.**

The case does *not* rest on effect size: +0.0168 mean test macro-F1 is still only ~35% of the single-run
noise band, and no individual run's delta would have been convincing alone. It rests on **consistency
under pairing** - all 12 deltas across 3 seeds x 4 metrics are positive, and each seed improved against its
own baseline rather than against a pooled average. That is a different and stronger kind of evidence than
Phase 7's adoptions had: exp 20 (patch_crop_size=900) was adopted on a *single* seed, and exp 34/35 later
showed its headline 0.9499 was a favourable draw from a 0.90-0.95 range. Nothing here is a favourable
draw - the worst seed still improved.

Two honest qualifications:
- **The magnitude is small and the per-seed spread is large.** Test deltas were +0.0055, +0.0078, +0.0370:
  consistent in sign, a 7x range in size. Expected test macro-F1 goes from ~0.9234 to ~0.9402; report the
  *range* (~0.91-0.96), not the mean, when describing this model, same discipline as Phase 7's close.
- **Variance is not made worse, but is not clearly improved either.** Erasing's test spread (0.0456) is
  marginally tighter than baseline's (0.0479) - not enough to claim it stabilises anything. Its *val*
  spread nearly doubled (0.0281 vs 0.0144), driven entirely by seed 123 (below).

### Methodology finding: the val set is saturating and losing discriminative power

Seed 123 with erasing hit **val_macro_f1 = 0.9917**, the highest val number anywhere in this project, while
its test only moved +0.0078. That asymmetry is exp 22's signature (val jumps, test doesn't follow), which
in Phase 7 meant overfitting the 3-photo/class val set. It is less alarming here because test still moved
the right way rather than dropping - but a near-perfect val alongside a nearly-static test means the val
set is approaching saturation and is running out of room to discriminate between configs. With 5 of 9
classes already pinned at or near 1.000 on both splits (exp 37), this will only get worse.

**Implication for future sessions, independent of erasing**: `best.pt` is selected by peak val_macro_f1, so
a saturating val set degrades *checkpoint selection*, not just reporting. Once val macro-F1 routinely
exceeds ~0.98, "best epoch" is being chosen among near-ties on a 360-patch/3-photo-per-class set, which is
plausibly a real contributor to the wide test spread Phase 7 attributed to patch geometry alone. Worth
addressing before further tuning: more val photos per class, or selecting on val loss (which keeps
resolving after F1 saturates) rather than on macro-F1.

### Exp 44: random_erasing_p 0.5 -> 0.75 (bracketing the adopted value)

Same move as exp 19/20/32 made for `patch_crop_size`: having adopted a value, push past it to find out
whether it's an optimum or just the first thing tried. Seed 42, compared against exp 39 (p=0.5, same seed)
and exp 37 (no erasing, same seed). Ran 04:05-04:50 UTC, 45 min.

| # | change (seed 42) | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 37 | no erasing | 0.9579 | 0.9533 | 0.9499 | 0.9441 | 14 | 22 |
| 39 | random_erasing_p=0.5 | 0.9635 | 0.9597 | 0.9554 | 0.9509 | 29 | 37 |
| 44 | random_erasing_p=0.75 | 0.9609 | 0.9565 | **0.9694** | **0.9658** | 14 | 22 |

**Promising, not yet adopted - needs the same paired multi-seed treatment p=0.5 got.** vs p=0.5: val
-0.0027 (flat), test **+0.0140/+0.0149**. vs no erasing at all: test +0.0195/+0.0217. test_macro_f1 0.9694
is **the highest single-run test macro-F1 anywhere in this project's log** (previous best 0.9609, exp 24).

Two things make this more interesting than the raw delta:
- **It's cheaper, not more expensive.** best_epoch 14 and 22 epochs run - identical to the un-augmented
  baseline, and a third less compute than p=0.5's 29/37. Every other "mildly positive" result in Phases 7-8
  (exp 23, 25, 27, 30, 33, 39) cost 1.5-2.2x compute for its nudge. This is the first one that doesn't.
- **val and test diverge in the useful direction.** val went slightly *down* while test went up - the
  opposite of the exp 22 / exp 38 / seed-123 overfitting signature, and consistent with the val-saturation
  finding above: with val near its ceiling, val deltas are becoming uninformative and test is the more
  trustworthy signal at this end of the range.

Explicitly not adopted on this evidence. Exp 39 looked good at seed 42 too (+0.0055) and only became
credible once seeds 123 and 7 agreed; a single seed showing +0.0140 inside a 0.0479 band is exactly the
kind of favourable draw exp 34/35 caught out. Exp 45 (p=0.75, seed 123) started to pair against exp 42.

### Exp 45: random_erasing_p=0.75 at seed 123 - rejects p=0.75, p=0.5 stands

Pairs exp 44 (p=0.75, seed 42) the way exp 42/43 paired exp 39. Ran 04:51-05:59 UTC, 68 min.

| paired comparison | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc |
|---|---|---|---|---|
| p=0.75 vs p=0.5, seed 42 (44 vs 39) | -0.0027 | -0.0032 | **+0.0140** | +0.0149 |
| p=0.75 vs p=0.5, seed 123 (45 vs 42) | -0.0028 | -0.0031 | **-0.0122** | -0.0120 |
| p=0.75 vs none, seed 42 (44 vs 37) | +0.0030 | +0.0032 | +0.0195 | +0.0217 |
| p=0.75 vs none, seed 123 (45 vs 34) | +0.0284 | +0.0307 | **-0.0044** | -0.0033 |

**Reject p=0.75; `random_erasing_p=0.5` remains the adopted value.** The two seeds' test deltas against
p=0.5 are near-equal and *opposite* (+0.0140, -0.0122), averaging to ~+0.001 - nothing. Exp 44's 0.9694,
the highest test macro-F1 in this project, was a favourable draw, exactly as flagged when it landed.
Against *no* erasing, p=0.75 even goes slightly negative at seed 123 (-0.0044), where p=0.5 was +0.0078 -
so p=0.75 is not reliably better than the un-augmented baseline, let alone than p=0.5.

Note the val column is the tell that the val metric has stopped being useful here: it is essentially
constant across both seeds (-0.0027, -0.0028) while test swings +0.0140 to -0.0122. val is no longer
tracking the thing that varies, which is the saturation problem described in the exp 42-43 section, now
demonstrated rather than inferred.

**This is the methodological point of Phase 8 in one experiment.** A single seed showing +0.0140 inside a
0.0479 band, on a run that was also *cheaper*, with a plausible mechanism ("more occlusion, more
distributed-texture pressure") and the best headline number in the project's history, was wrong. The only
thing that caught it was refusing to adopt before pairing a second seed. Exp 20 was adopted on exactly that
kind of evidence in Phase 7, and exp 34/35 later showed its headline was also a favourable draw.

## Phase 8 run summary (2026-08-08 18:45 - 2026-08-09 06:00 UTC, 11h15m of a 12h budget)

10 training runs (exp 36-45), one variable at a time, every result committed individually with `dvc repro`
keeping `dvc.lock` in sync. Stopped at 11h15m rather than starting an 11th run that couldn't finish and be
written up inside the budget.

**Adopted (1)**: `random_erasing_p=0.5`. Mean test macro-F1 **0.9234 -> 0.9402**, val 0.9635 -> 0.9767,
across a paired 3-seed check (exp 39/42/43 against exp 20/34/35 at the same seeds). All 12 deltas positive.

**Rejected (5)**: arbitrary rotation jitter +/-5 deg (exp 36, flat), zoom/RandomResizedCrop 0.7 (exp 38,
val up / test down at 2.2x compute), illumination gradient 0.2 (exp 40, fails the "costs nothing" bar),
mixup 0.2 (exp 41, uniformly mildly negative), and `random_erasing_p=0.75` (exp 44/45, seeds disagree in
sign). Hypothesis 4 (perspective) was **not tested** - see below.

**Infrastructure (2 non-hypothesis runs)**: exp 37 reproduced exp 20 bit-for-bit on the refactored code,
validating every Phase 8 comparison and providing the first archived per-class baseline.

### What Phase 8 actually established

1. **Random erasing works, modestly.** ~+0.017 mean test macro-F1. Report the model as **test macro-F1
   ~0.91-0.96** (3-seed range with erasing), not as a point estimate - same discipline Phase 7 closed with.

2. **The evidence standard changed, and it caught a false positive.** Phase 7 adopted on single seeds; exp
   34/35 then showed its flagship adoption's headline number was a favourable draw. Phase 8 adopted only on
   *paired* multi-seed evidence - and the one time a single seed looked spectacular (exp 44: 0.9694, best
   in project history, cheaper to train, plausible mechanism), the paired seed reversed the sign. Pairing
   against a same-seed baseline is the cheapest reliability upgrade available here and should be the
   default from now on.

3. **The val set is saturating, and this is now the top blocker.** Seed 123 reached val_macro_f1 0.9917;
   in exp 44/45 val stayed flat (-0.0027, -0.0028) while test swung +0.0140 to -0.0122. val has stopped
   tracking what varies. This is not merely a reporting problem: `best.pt` is selected on peak
   val_macro_f1, so checkpoint selection is now choosing among near-ties on 3 photos/class - plausibly a
   real contributor to the wide test spread Phase 7 attributed to patch geometry alone.

4. **Five of nine classes are effectively solved** (CostaRica-LaPastora, Ethiopia-Kochere, Vietnam-Robusta
   at ~1.000; Colombia-PinkBourbon and Guatemala-Tata ~0.95-0.99). All remaining headroom is
   **Brazil-Cerrado / Brazil-MonteCristo / Kenya-AA**. Notably, random erasing helped Kenya-AA (+0.036) but
   *not* the Brazil pair it was hypothesised to fix - the rationale failed even though the result held.

5. **Photo width is the binding constraint on the whole patch pipeline.** The valid region is ~1016px
   wide against a 900px patch. That one number blocked the plan's preferred rotation implementation,
   capped rotation jitter at ~6 deg, and is the same ceiling exp 32 hit at patch_crop_size=1000. A future
   capture session framing the tray slightly wider would relax all of these at once.

### Open threads for the next session

- **Fix val before more tuning** (highest value). Either more val photos per class, or select `best.pt` on
  val *loss*, which keeps resolving after macro-F1 saturates. Almost everything else is limited by this.
- **Hypothesis 4 (perspective) is untested.** Its mean-fill implementation was removed with rotation's. A
  real-pixel version needs ~90px of source margin at `distortion_scale=0.2`, leaving ~26px of placement
  room - right at exp 32's edge. Needs either a smaller distortion or an explicit decision to accept that.
- **Rotation jitter at +/-3 deg** (70px room vs 40px at 5) would separate "rotation doesn't help" from
  "lost translation diversity hurt" in exp 36. Low expected value given how flat exp 36 was.
- **Erasing between 0.5 and 0.75, or larger erased areas** (`scale` is hardcoded at 0.02-0.15 and was never
  swept). p=0.5 is adopted but was never bracketed from below - p=0.25 is untested.
- **Combined runs were never reached.** Nothing else was positive enough to combine, and Phase 7 exp 30
  showed individually-noisy nudges don't compound, so this stayed low priority.
- **Brazil-Cerrado / MonteCristo / Kenya-AA** remain the only real headroom, and four phases of
  hyperparameter and augmentation work have not moved them much. This looks like a data problem (more or
  different capture angles for those classes) rather than a training-configuration one.
- `git push` still not possible in this container (no SSH key) - everything is committed locally only.

## Post-run: widening what gets kept from `outputs/` (2026-08-09 08:00 UTC)

Reviewing `outputs/` (90MB) for anything else worth version control, on the user's prompt. Resolved:

| artifact | size | decision |
|---|---|---|
| `plots/confusion_matrix_{val,test}.png` | 116K each | **archived per experiment** |
| `plots/training_curves.png` | 80K | **archived per experiment** |
| `plots/patch_samples.png` | 3.3M | **skipped** - drawn from the *val* set, so nearly identical between runs that don't change patch geometry; would be ~10x the rest of the archive |
| `checkpoints/best.pt` | 43M | **adopted one pinned** to `models/` via `dvc add` + a git-tracked model card |
| `checkpoints/last.pt` | 43M | skipped - the test split is only ever evaluated with `best.pt` |
| `inference/*.json` | 168K | **rescued** to `experiments/inference_runs/` |

**The charts are regenerated, not copied - which is what made backfilling possible.** `outputs/plots/` only
ever holds the most recent run, so copying would have given charts to exp 45 alone and left exp 36-44
permanently chartless. But everything those two plots draw is already archived: the confusion matrix lives
in `metrics.json` and the curves in `history.json`. So `archive_experiment.py` now *rebuilds* them from the
run's own JSON, and `--replot-all` backfilled all 10 experiments retroactively. Archive went 252KB -> 3.5MB.

**What the confusion matrices immediately showed, which the per-class F1 table did not.** On the adopted
config (exp 39), the entire test error is 16 patches out of 360, and 14 of those are just two *directional*
confusions: Kenya-AA predicted as Ethiopia-Sidamo (7/40) and Brazil-Cerrado predicted as Brazil-MonteCristo
(7/40). Both are one-way - Sidamo is never called Kenya, MonteCristo is never called Cerrado. A symmetric
confusion suggests two classes that simply look alike; a one-way one suggests the model has learned a
decision boundary that swallows one class into the other, which is a different (and more fixable) problem.
Worth pointing the next session's data-collection effort at specifically.

**On the checkpoint.** `.dvc/cache` currently holds ~4.7GB across ~90 checkpoints - every run this project
has ever done is still on disk. But they are referenced only by *historical* `dvc.lock` commits, so `dvc gc`
would delete all but the current one, silently and irreversibly (there is still no DVC remote). Pinning the
adopted model as its own `dvc add` artifact puts a `.dvc` file in the working tree, which is what actually
protects it. Deliberately pinned exp 39 and **not** exp 44's 0.9694 - the best confirmed config, not the
best number, since exp 45 showed that number was a favourable draw.

**Provenance gap found while rescuing the inference JSONs**: they record `crop_size`/`patch_resize`/
`n_patches_per_image` but not *which checkpoint produced them* - no model hash, no git commit. They are
Phase 6-era, so two adopted-config generations behind. Kept as a record that the analysis happened, flagged
as not comparable to current numbers. `infer.py` should write the same `env` block `train_baseline.py`
already writes, plus the checkpoint hash, if inference is re-run.

**Housekeeping note on the `dvc commit` that followed.** Editing `archive_experiment.py` changed the hash
of the `coffeecv` directory, which `dvc.yaml` lists as a dep of the `train` stage - so DVC marked the stage
stale and the next `dvc repro` would have retrained for an hour over a file that has nothing to do with
training. Resolved with `dvc commit -f`, which records the existing outputs against the new code hash
without re-running. Stating the small inaccuracy plainly rather than hiding it: those outputs were produced
by the *previous* commit's code, and `dvc.lock` now associates them with this one. That is accurate in
substance (the changed file is not in the training path, and `best.pt`'s md5 is unchanged) but it does mean
`dvc.lock` records a code state that never literally produced these artifacts. The alternative - leaving the
stage permanently stale - would have been worse, and an hour of retraining to restore literal truth is not
a good trade.

## DVC + VS Code extension: what actually shows up (verified 2026-08-09)

Checked against the DVC docs and then verified by running the commands the extension wraps, rather than
assuming. Three findings, one of them a pleasant surprise.

**1. Charts already work across every Phase 8 experiment - including the ones whose metrics were lost.**
`dvc plots diff <commit> <commit> ...` renders all four experiment commits tested (36, 38, 39, 41) with all
three declared sources. The confusion matrices genuinely render (vega `rect` marks, true_label/pred_label
axes) and the curves render as `line`/`circle`/`rule`. This works because `outputs/history.json` and
`outputs/predictions_{val,test}.csv` are **cached DVC outs** - so DVC retrieves each revision's copy from
`.dvc/cache`. That is exactly the property `outputs/metrics.json` lacked (`cache: false` *and* gitignored,
so tracked by neither and unrecoverable for exp 36-44). Same directory, opposite outcome, purely because of
how each file was declared - worth remembering as the concrete cost of `cache: false` without git tracking.

**2. The experiments table was unreadable, and that was fixable.** `dvc exp show` and the extension's
Experiments table flatten every leaf of a *metrics* file into a column. `metrics.json`'s nested per-class
stats expanded to 148 columns, which is why neither view was ever usable here. Moving `metrics.json` from
`metrics:` to `outs:` (still `cache: false`, still git-tracked, still archived per run) and leaving
`summary.json` as the sole declared metric cuts it to **42 columns** with the six headline numbers first,
the rest being params. Note historical commits keep their own `dvc.yaml`, so revisions before this change
still contribute the wide columns when shown in the same table.

**3. Image plots were considered and deliberately not added.** DVC does support PNG/JPG/SVG plots, rendered
side by side across selected experiments. But the confusion matrices and curves *already* render natively
from the CSV/JSON, interactively and for every revision, so declaring `outputs/plots/*.png` as image plots
would duplicate them for no gain - and would require making them cached outs, which pulls the 3.3MB
`patch_samples.png` question back open. The archived PNGs under `experiments/` serve a different purpose:
browsing results in git (or on a forge) without running DVC at all.

**How to use it in VS Code**: open the DVC panel -> Experiments, select up to **7** rows via the circle
beside each (the extension's documented limit), then `DVC: Show Plots`. Rows here are git commits, not
`dvc exp` experiments - `dvc exp run` still fails in this container, but `dvc exp show` reads git history
fine, so the commit-per-experiment discipline this project already follows is what makes the extension work.

## Why `dvc exp run` fails here - the real cause (2026-08-09), and a mistake made finding it

The standing note in this project blamed the machine's no-reflink cache config. **That was a guess and it is
wrong.** Verified by running `dvc exp run --temp -f` and reading the traceback:

```
FileNotFoundError: No cropped photos found in
  /workspace/.dvc/tmp/exps/standalone/tmppdtg03er/dataset/.../class_001__Ethiopia_Sidamo/cropped
```

**The cause is a derived-data gap.** `.dvcignore` excludes `**/cropped/`, so DVC tracks only the 180 *raw*
photos - confirmed against the cached `.dir` listing, where zero of the 180 tracked paths contain "cropped".
But `MultiPhotoPatchDataset` reads *only* `class_*/cropped/*__cropped.jpg`. The crops exist in the working
tree because `crop_tray.py` was run by hand on 2026-08-07, and `crop_tray.py` is not a `dvc.yaml` stage, so
nothing regenerates them. An isolated workspace materialized from DVC-tracked content + git therefore has
the raw photos and no crops.

Two corrections follow, both bigger than the original note:

1. **Plain `dvc exp run` works.** It executes in the real workspace, where the crops are. It ran training
   normally until a 150s timeout killed it. Only `--temp` and `--queue` fail. The blanket claim "dvc exp run
   fails in this container", repeated in several places including this log, was wrong.
2. **The pipeline is not reproducible from a clean checkout.** `git clone` + `dvc pull` + `dvc repro` would
   hit the same error: the 403MB of DVC-tracked raw photos are unusable without a crop step no stage
   performs. The `.dvcignore` comment says the crops are "100% regenerable from the raw photos + code" -
   true in principle, but nothing automates it, and *training reads only the crops*. **The fix is a `crop`
   stage in `dvc.yaml` producing `**/cropped/` as an out - deliberately not done here.** The crop heuristic
   was already broken once by real lighting drift across this capture session, so regenerating crops could
   silently change the dataset every result in this log rests on. That needs a decision and a verification
   pass, not a drive-by commit.

Also fixed while here: `__pycache__/` added to `.dvcignore`. `coffeecv` is a dep of the train stage, so
every run rewrote `.pyc` files, changed the directory hash, and made DVC report the stage as changed when no
source had changed - low-grade noise that made `dvc status` untrustworthy.

### Mistake: `dvc exp remove -A` wiped 24 historical experiment refs

While cleaning up the failed `--temp` run I used `dvc exp remove -A`, which removes *all* experiment refs,
not just the one I meant. It deleted 24 named refs from Phases 1-6 (`crop-700`, `resnet18-finetune-full`,
`optimizer-sgd`, ...). The names lived only in the ref names; `.git/logs/refs/exps` is empty and the
underlying commits carry only `dvc: commit experiment <hash>` messages, so the name->commit mapping is
**not recoverable**.

Actual impact, measured rather than assumed: **small**. The per-experiment record on `main` is untouched -
48 experiment commits, and `dvc plots diff 3b31b65 19f5b27` still renders Phase 5 experiments fine. Those
refs were a parallel record of runs that were also committed to `main`, which is what the extension's rows
and every plot/metric query in this project actually read. What is lost is the ability to see those old runs
under their human-readable names in the experiments table; they now appear as commit rows. No data, no
metrics and no plots were lost.

Worth remembering: `dvc exp remove -A` has no confirmation prompt and no undo.

# Phase 9 (planned) - Make the crop step part of the pipeline

Written 2026-08-09 after the `dvc exp run` investigation above found that training reads only
`**/cropped/`, which no stage produces and DVC does not track. Answering the question directly: **yes, the
full-image -> crop process should be in the pipeline** - and four findings established below make that both
safe and more valuable than a pure plumbing fix.

## What was verified first (so this isn't a risky rewrite)

1. **The crop step is fully deterministic.** Re-ran `crop_tray.py --fixed-trim 0.1` on class_001 into a
   scratch dir: all 20 crop boxes identical to the recorded ones, and all 20 output JPEGs **byte-identical**
   (md5). So turning cropping into a stage regenerates exactly today's crops and cannot silently change the
   dataset that Phases 7-8 rest on. This was the main risk and it is retired.
2. **Provenance already exists and is untracked.** Each `cropped/crop_report.json` records, per photo, the
   final `box`, the `rough_box`, `rough_method`, `trim_frac: 0.1`, and a `needs_review` flag - 180 entries
   across 9 files, all currently inside the dvcignored directory. Small, valuable, and one `rm -rf` from
   being gone.
3. **The exact command is recoverable**: `--fixed-trim 0.1`, i.e. `crop_dataset_fixed_trim`, not the
   adaptive path (which the docstring records as unstable under this session's lighting drift).
4. **`trim_frac` is the lever on the project's binding constraint.** `box = (x + w*t, y + h*t, w - 2wt,
   h - 2ht)`, so `trim_frac=0.1` discards **20% of the tray's width and height**: rough boxes ~1361px wide
   become ~1089px crops. The Phase 8 close identified photo width as the single constraint that blocked the
   preferred rotation implementation, capped rotation jitter at ~6 degrees, and bracketed `patch_crop_size`
   at 900 (1000 failed for want of placement room). At `trim_frac=0.05` crops would be ~1225px wide -
   **+12.5%** - which would relax all three at once. That makes this phase a genuine experiment, not just
   plumbing. Trade-off to respect: less trim means more risk of including the tray rim, which is exactly
   what the 10% trim exists to avoid, so it needs full-resolution visual QA, not a metric alone.

## The structural constraint that shapes the design

`dataset/2026-08-07__box_pictures_all_classes` is tracked by `dvc add` (a `.dvc` file). **DVC will not let a
pipeline stage write outputs inside a `dvc add`-tracked directory.** So the crops cannot become a stage
output where they currently sit. They must move to their own path, which is also the conventional split:
raw captures tracked as data, derived crops produced by a stage.

## Where crop settings live - `params.yaml` would be the wrong place (user's call, 2026-08-09)

The first draft of this plan put `trim_frac` in `params.yaml` as a tracked param. **Rejected, correctly.**
`trim_frac` is not a model hyperparameter; it is a property of *one rig* - fixed tripod, fixed distance,
constant rim width as a fraction of the tray. Future sessions are expected to vary angle and tray shape, and
under that variation a fractional trim off a rough box does not merely need a different *value*, it stops
being the right *operation* (a tilted tray's rim is not a constant fraction of its bounding box; an
irregular shape has no meaningful "trim off each side" at all).

Putting it in `params.yaml` would conflate two things that scale differently:

| | `params.yaml` | per-session crop config |
|---|---|---|
| contents | lr, patch_crop_size, augmentation... | crop method + its settings |
| scope | the whole project | one capture session |
| varies by | experiment | rig geometry |
| sweeping it means | "is this a better model?" | "did I prepare this session's data well?" |

It would also pollute the experiments table (already trimmed from 148 to 42 columns) with a column that is
constant for every run of one session and meaningless across sessions.

**Design instead**: each capture session carries its own git-tracked crop config, e.g.
`dataset/2026-08-07__box_pictures_all_classes.crop.yaml`:

```yaml
method: fixed_trim   # fixed_trim | adaptive | (future: sam, perspective_corrected, manual_boxes)
trim_frac: 0.10
notes: fixed tripod rig; the adaptive path is unstable under this session's lighting drift
```

Declared as a **dep** of the crop stage, not a param - so changing it correctly invalidates the stage and
its hash is recorded in `dvc.lock` for provenance, without ever entering the hyperparameter surface. A future
session with a different geometry gets its own file and may name a different `method` entirely; nothing in
`params.yaml` or the experiments table changes. This is the interface that survives the variation the user
expects, whereas a global `trim_frac` is only correct until the second rig exists.

## Plan

1. **Move crops out of the raw dataset dir** to `data/cropped/2026-08-07__box_pictures_all_classes/class_*/`,
   because DVC will not let a stage write inside a `dvc add`-tracked directory. Update `dataset.py`
   (`find_class_dir`/`list_cropped_photos`) so the crop location is configuration rather than a hardcoded
   `/ "cropped"` suffix. `cropped_dir` (a *path*, like the existing `dataset_dir`) is fine in `params.yaml`;
   the crop *method and its settings* are not.
2. **Add a `crop` stage to `dvc.yaml`**, `foreach` over *sessions* rather than the 9 class dirs - a session
   is the unit that gets added over time, and a per-session stage keeps adding session #2 to a one-line
   change. Deps: the raw session dir, its `.crop.yaml`, and `coffeecv/crop_tray.py`. Outs: the session's
   cropped tree including `crop_report.json` (which already records the per-photo boxes and is currently
   untracked). `train` then depends on the crop output, so the DAG is honest end to end.
3. **Drop `**/cropped/` from `.dvcignore`** (keeping the `__pycache__/` entry added 2026-08-09).
4. **Acceptance test, non-negotiable**: re-run training at seed 42 and require it to reproduce exp 39
   exactly - val_macro_f1 0.9635, test_macro_f1 0.9554, best_epoch 29/37. Same bit-exactness discipline exp
   37 used for the augmentation refactor. If it does not match, the restructure changed something and gets
   reverted rather than explained away.
5. **Then, separately, try a wider crop** (`trim_frac` 0.10 -> 0.05, maybe 0.075) - but logged as a
   *data-preparation* change for this session, not a hyperparameter sweep, and reported as such: its result
   is a statement about this rig, not a project-wide finding, and it will not transfer to a session shot at
   a different angle. Wider crops mean more placement room, so re-test `patch_crop_size` 1000 and rotation
   jitter above 6 degrees on top. Judge on the paired multi-seed standard Phase 8 established, and QA the
   crops at full resolution for rim contamination before trusting any metric.

**Cost**: ~2-3h of code and verification, of which one ~50 min training run is the acceptance test. Cropping
all 180 photos is a few minutes. **Risk**: low for steps 1-4 given the byte-identical reproduction; step 5
is a real experiment with a real trade-off and should be treated as one.

**Deliberately not in scope**: re-cropping with the *adaptive* path, and re-tracking the 2026-08-06 unlabeled
session. Both are separate decisions.

## Phase 9 executed (2026-08-09) - steps 1-6 complete

### Exp 46: acceptance test - PASS

The non-negotiable gate: after moving cropping into the pipeline, seed 42 must reproduce exp 39 exactly.

| metric | exp 39 | exp 46 | |
|---|---|---|---|
| val_macro_f1 | 0.9635 | 0.9635 | match |
| val_mcc | 0.9597 | 0.9597 | match |
| test_macro_f1 | 0.9554 | 0.9554 | match |
| test_mcc | 0.9509 | 0.9509 | match |
| best_epoch | 29 | 29 | match |
| epochs_trained | 37 | 37 | match |

All six exact. Combined with the byte-identical crop check below, the restructure is a verified no-op: every
number in Phases 7-8 still stands.

### Verification performed

1. **All 180 crops regenerate byte-identically.** Moved the existing crops aside, ran `dvc repro crop` from
   the raw photos, compared md5s: 180/180 match, 0 flagged for review. The crop step's determinism was
   spot-checked on 20 photos while planning; this confirms it across the whole session.
2. **Raw dataset `.dvc` hash unchanged** (`cc2f6d9f...`) before and after the move, as predicted - the crops
   were dvcignored, so they were never part of that hash.
3. **The DAG is now honest end to end**: `dataset/<session>.dvc` -> `crop@<session>` -> `train`.
4. **The original failure is fixed.** `dvc exp run --temp -f` no longer dies with `FileNotFoundError: No
   cropped photos found`; the isolated workspace now materializes all 180 crops from the DVC cache plus the
   session's `.crop.yaml` from git, and proceeds into training. (Stopped there deliberately - the failure
   mode was materialization, and an hour of training in a temp dir would add no information.)

### The retest caught the same bug class a second time

First `--temp` attempt after the restructure still failed - but in the *crop* stage, not training, because
`crop_session.py` and `<session>.crop.yaml` were written but **not yet committed**. The temp workspace is
built from git + the DVC cache, so an uncommitted file simply isn't there. Exactly the original disease in
new clothes: the pipeline depending on something that happens to be on disk. Worth keeping as the standing
argument for why `dvc exp run --temp` is the right reproducibility check - the normal workspace cannot
detect this class of problem *by construction*, because the file is sitting right there.

### What is now true that wasn't

- `git clone` + `dvc pull` + `dvc repro` reproduces the pipeline from scratch. Before, the 403MB of tracked
  raw photos were unusable without a crop step nothing performed.
- `crop_report.json` (per-photo boxes for all 180) and the new session-level `crop_manifest.json` are
  tracked outputs instead of untracked files inside a dvcignored directory.
- Crop settings are versioned per session and can vary by rig without touching `params.yaml`, the
  experiments table, or any global config.

### Not done: step 7 (wider crop trial)

`trim_frac` 0.10 -> 0.05 remains untried, deliberately - it is a real experiment with a real trade-off (more
tray-rim contamination risk) and needs full-resolution visual QA plus paired multi-seed evaluation, not a
drive-by run at the end of a refactor. It is the most promising open thread, since wider crops (~1089px ->
~1225px) would relax the photo-width ceiling that capped `patch_crop_size` at 900, limited rotation jitter
to ~6 degrees, and blocked the preferred rotation implementation entirely.

### Incident: `dvc exp run` detached HEAD and stranded five commits (2026-08-09)

Found because the user noticed `data/` was neither git-tracked nor ignored. That symptom turned out to be
the visible edge of a much larger problem.

**What happened.** `dvc exp run` detaches HEAD to run an experiment. The plain (workspace-mode) run during
the "why does `dvc exp run` fail" investigation left HEAD detached and `main` behind at
`2ec2212`. Every commit after that - `08acfef` (the pycache/dvcignore fix) through `ab4e897` (all of Phase
9) - was made on that detached HEAD, **five commits**, without any warning in normal `git log` output, which
happily shows a detached history as if nothing were wrong. A later `dvc exp run --temp` then checked out
`main` on cleanup, which reverted the working tree to `2ec2212`: `crop_session.py` gone, `dvc.yaml`'s crop
stage gone, `dvc.lock`'s crop entry gone, `<session>.crop.yaml` gone.

**Why it wasn't obvious.** `git status` reported *clean*, and `dvc status` reported *up to date* - both
truthfully, since the working tree genuinely matched `main`. Nothing was corrupt; the work was simply on a
branch nobody was standing on. The only visible tell was `data/cropped/` sitting there untracked and
unignored, because the crops are untracked-by-git data and survived the checkout that removed everything
around them.

**Recovery.** `ab4e897` was still in the reflog and descends from `2ec2212`, so `git merge --ff-only
ab4e897` restored all five commits and the working tree in one step. Verified after: git clean, `dvc status`
up to date, `dvc.lock` carrying both stages, all 180 crops intact, `outputs/` still the adopted config.
Nothing lost.

**Lessons, in order of importance:**
1. **`dvc exp run` can leave HEAD detached. Check `git rev-parse --abbrev-ref HEAD` after any `dvc exp`
   command**, and before a run of commits. `git log` alone will not tell you.
2. **`git status` clean is not evidence that your work is safe** - it only says the worktree matches HEAD,
   and says nothing about whether HEAD is a branch anyone will find again.
3. An interrupted `dvc exp run --temp` also deleted DVC's own `data/cropped/.gitignore`, and a blind
   `git add -A` then committed that deletion unnoticed (`20cd1ea` added it, `ab4e897` silently removed it).
   Read what `git add -A` is staging when DVC has been touching the tree.

**Answering the original question - what should `data/` be**: `data/cropped/<session>/` is a `crop` stage
output, so it must be **gitignored** (126MB of JPEGs belong in DVC, not git) while `data/cropped/.gitignore`
itself is **git-tracked** - DVC writes that file so a fresh clone knows to ignore the path. That is now the
state, and an audit confirms every DVC output is correctly handled: all cached outs gitignored,
`outputs/metrics.json` and `outputs/summary.json` git-tracked deliberately (they are `cache: false` metrics).

### Step 7 attempt: `trim_frac` 0.10 -> 0.05 REJECTED at QA, and the premise is largely disproven

Regenerated crops at `trim_frac: 0.05` (crops 1048px -> 1178px min width, valid region 1016 -> 1142px) and
QA'd before training, as the plan required. **Visibly contaminated**: the tray rim is inside the crop on most
classes - dark metal edges along the borders - which is exactly what the 10% trim exists to prevent. No
training run was spent. Reverted via `git checkout HEAD -- dvc.lock *.crop.yaml && dvc checkout`, which
restored the trim-0.10 crops from the DVC cache and is only possible *because* Phase 9 made crops a tracked
stage output.

Then measured rim depth properly rather than guessing an intermediate value: local-texture standard deviation
(illumination-robust, unlike the saturation-Otsu the adaptive method uses), scanning inward from each edge of
the rough tray box until bean texture starts, over **all 180 photos**.

| side | max rim depth | p99 |
|---|---|---|
| left | 0.056 | 0.047 |
| right | **0.063** | 0.062 |
| top | 0.012 | 0.012 |
| bottom | 0.011 | 0.011 |

**Why 0.05 failed**: it sits below the right side's 0.063 worst case. Confirms the visual, and confirms the
detector agrees with the eye.

**The finding that matters, and it kills the original motivation.** The rim is deep (~0.06) on *both*
horizontal sides but shallow (~0.012) top and bottom. So:

- **Width has almost no reclaimable headroom.** The current symmetric 0.10 is close to right for the
  horizontal axis; the rim genuinely occupies ~6% of the rough box on each side. Trimming to a safe
  asymmetric value recovers at most ~7-8% width, not the ~12.5% the plan assumed from a symmetric 0.05.
- **Height wastes ~20%** (trimming 0.10 where ~0.02 would do) - but height was never the constraint. The
  valid region is ~1464px tall against a 900px patch, already 564px of placement room.

Step 7 was motivated by relaxing the **photo-width ceiling** that capped `patch_crop_size` at 900, limited
rotation jitter to ~6 degrees, and blocked the preferred rotation implementation. That ceiling is now shown
to be a property of **how the tray was framed at capture**, not of how conservatively it was cropped. The
tray fills the frame and its rim is genuinely ~6% deep per side; no re-cropping recovers meaningfully more
bean width. **The real unlock is a re-framed capture session** (tray smaller in frame, or higher-resolution
capture), which is a hardware/process change, not a parameter.

Recorded rather than pursued: an asymmetric trim at a *tight* margin (worst case + 0.01, i.e. left 0.066 /
right 0.073 / top 0.022 / bottom 0.021) would keep ~86% of width instead of 80% - about +7.6%, taking the
tightest valid region ~1016 -> ~1137px, which would give `patch_crop_size=1000` ~137px of placement room
instead of the ~17px that made exp 32 fail. That is a real experiment, but it buys a modest gain by spending
most of the measured safety margin, and the texture detector's own error is unquantified. Not taken
unilaterally.

## Analysis: 2026-08-09 new capture format (white cup) - 6 test photos

User is considering changing the capture container from the metal-rimmed box tray to a white measuring cup,
elevated on a jar, shot top-down. Six unlabeled test photos, same phone/sensor (3072x4080). Assessed for
whether the pipeline can use them.

### In favour

- **Bean pixel scale is unchanged.** Dominant texture period (bean-to-bean spacing) 115px on the new format
  vs 110px on the current rig. This matters more than it sounds: the model's learned texture scale would
  transfer, `patch_crop_size` keeps its meaning, and results stay roughly comparable across the change.
- **Focus uniformity is *better*, not worse.** Edge/centre Laplacian-variance ratio 0.62 (new) vs 0.43
  (current tray). The concave bowl surface was the obvious worry and it is not a problem - if anything the
  flat tray's corners were softer.
- **No metal rim.** The entire rim-contamination problem that this session's crop work has been fighting
  simply disappears; a white cup against a white background separates from beans far more cleanly than a
  metal rim does.
- **Circular bean region is already supported.** `geometry.compute_valid_region` (written for the original
  macro-lens rig) computes the largest square inscribed in a circle. That code path exists and is tested.

### Against, as currently shot

- **Usable area is not better, and probably slightly worse.** Largest all-bean square measures **912-968px**
  (conservative, hole-filled max-rectangle); taking the inscribed square of the detected bean circle instead
  gives ~1030-1160px. Current rig after today's asymmetric trim: **1067px**. So at best a wash, at worst a
  ~10% regression - and `patch_crop_size=900` would have only 12-68px of placement room on the conservative
  measure, straight back into exp 32's failure zone (17px). `patch_crop_size=1000` would not fit at all.
  **This format as shot does not relax the width ceiling, which was the reason to want a new format.**
- **The existing crop pipeline fails on 2 of 6 photos.** Hard directional shadow creates a texture edge that
  `locate_tray_rough` follows out into the background; the detected region became the whole frame. Verified
  visually, not just numerically.
- **The jar is in frame and its label is textured**, competing with the beans for texture-based detection.
  All measurements above needed the search restricted to the top 55% of the frame to work at all.
- **Most of the sensor is wasted.** Beans occupy roughly 15-20% of frame area; the cup's outer diameter is
  only ~55% of frame width.

### Verdict and what would make it clearly better

The container is not the problem - **framing and lighting are**. Fixes, in order of value:

1. **Fill the frame.** Move the camera closer (or use a wider cup) so the cup spans ~85% of frame width
   instead of ~55%. That alone would take the usable square from ~940px to ~1500px - a ~40% *improvement*
   over the current rig, which would finally unlock `patch_crop_size` 1200-1400 and make artifact-free
   arbitrary rotation possible (both blocked today by the 1067px ceiling).
2. **Diffuse the light.** The hard shadow breaks detection on a third of the test shots, and directional
   lighting drift is what broke the original adaptive crop heuristic on the 2026-08-07 session too.
3. **Get the jar out of frame** - plain background, cup on a flat surface.

With 1-3 done, this format is strictly better than the tray: no rim, better focus uniformity, easier
segmentation, more usable pixels. As shot today, it is a step sideways or slightly backwards.

**Separately worth knowing**: even unlabeled, a second capture session with a different container and
lighting is exactly the missing piece for Phase 8's hypotheses 4 and 5 (perspective jitter, illumination
gradient). Both were left unvalidated because train/val/test all came from one 2h shoot, so the current test
set structurally cannot measure robustness to a new session. These photos could measure it, once labelled.

### Format b (black box, 1 test photo) - clearly the best option tested

Second candidate format: matte-black rectangular box, beans filling it, even lighting, shot landscape
(4080x3072). One photo only, so treat as indicative.

| | current tray rig | format a (white cup) | **format b (black box)** |
|---|---|---|---|
| usable square | 1067px | 912-968px | **2280px** |
| bean pixel scale (texture period) | 110px | 115px | 120px |
| focus edge/centre | 0.43 | 0.62 | 0.46 |
| detection reliability | good | **fails 2/6** | good (first try) |
| bean fill of frame | high | ~15-20% | 59% |

**Correction to a number quoted earlier in this session**: the first measurement of format b returned a
2904px usable square. That was wrong - the texture-based detector had swallowed the black box walls, since
matte black is as texture-free as the beans are textured, so the "bean" contour engulfed the whole box. Re-measured
with an Otsu threshold on brightness (the natural discriminator for a black box) and verified visually: the
honest figure is **2280px**, and even that rectangle is conservative - there is bean area outside it.

**Why this is the format to pursue:**

- **2.1x the usable square** of the current rig, at essentially the same bean pixel scale (120 vs 110px,
  confirmed at 1:1 side by side). So the data stays comparable in kind while carrying ~4.6x more bean area
  per photo - far more patch diversity per photo, and fewer photos needed per class for the same volume.
- **It unblocks every geometry constraint this project has hit.** Valid region ~2212px after safety margin:
  `patch_crop_size=900` gets 1312px of placement room (vs 167px today), and 1500px patches would still have
  712px. Exp 32 rejected 1000px purely for want of placement room; that whole line reopens. Arbitrary-angle
  rotation also becomes possible artifact-free - the approach ruled out in Phase 8 needs a source of
  patch x 1.42 (1278px at patch 900), which now fits comfortably.
- **Detection is easier, not harder.** The black box separates from beans by plain brightness (Otsu on V,
  worked first try). No metal rim, no rim-depth measurement, no asymmetric trim.
- Lighting is even - no hard shadows, which is what broke detection on 2 of 6 format-a photos and what broke
  the original adaptive crop heuristic on the 2026-08-07 session.

**Caveats before committing to it:**

- **One photo.** Consistency across a real session (box position, fill level, lighting drift over a shoot)
  is unverified, and that is exactly where the previous two rigs sprang surprises.
- Bean scale is ~9% larger than current data, so this is a **retrain, not a transfer** - existing checkpoints
  would not apply directly, though `patch_crop_size` semantics carry over well enough to reuse the tuning.
- The current `locate_tray_rough` is texture-based and, as shown above, is the wrong detector here. A new
  `method` is needed - but that is precisely what the per-session `<session>.crop.yaml` from Phase 9 is for:
  add `method: dark_box` for the new session and nothing global changes. The architecture the user insisted
  on (crop settings per session, not in `params.yaml`) pays off exactly here.

### Format c (frame-filling, no container) - best measured, with one structural caveat

Third candidate: beans filling the entire 4080x3072 frame, no container in shot at all. One photo.

Measured with identical probes across all formats (sharpness at 35% of half-width from centre, brightness
over a 4x4 grid, both restricted to each format's own bean region so the black box does not skew format b):

| | current tray | format a (cup) | format b (black box) | **format c (frame-filling)** |
|---|---|---|---|---|
| usable square | 1067px | 912-968px | 2280px | **2979px** |
| bean area vs current | 1.0x | ~0.8x | 4.6x | **7.8x** |
| bean pixel scale | 110px | 115px | 120px | 120px |
| focus edge/centre | 0.39 | - | 0.43 | **0.53** |
| brightness spread | 13% | - | 10% | 15% |
| non-bean frame content | rim | ~80% | 41% | **0.0%** |

**A correction to an earlier reading in this analysis**: a first pass called format c's focus the *worst*
(0.29-0.36). That was measured at the frame corners, which on a frame-filling shot sit much further from
the optical centre than the probes used for the other formats - not a like-for-like comparison. Re-probed at
the same *relative* radius for every format, c is the best of the three (0.53), not the worst.

**c is the strongest format on every axis measured**: 2.8x the current usable square, the most uniform
focus, the same bean scale as b, and literally zero non-bean pixels (texture map minimum 0.21 across the
whole frame - there is no background anywhere in shot).

**The one structural caveat, and it is not small**: with no container there is no *positive boundary check*.
Format b's black box proves the frame contains only beans; format c relies on the pile actually covering the
frame every time. A slightly shifted camera or a thin patch of beans would silently admit background into
training data with nothing to catch it. That is exactly the class of silent-contamination failure this
project has already been bitten by twice (the adaptive crop collapse, and the rim leak that the whole-box
contamination metric averaged away).

Cheap mitigation, if c is chosen: gate every photo on the texture map at ingest - this photo reads minimum
0.21 / p1 0.26 across the frame, whereas any background or container region reads near zero, so a simple
"p1 texture > 0.15" check per photo would catch a bad frame before it ever reaches training. That belongs in
the crop stage as `method: full_frame` plus a QA assertion, which the per-session config from Phase 9
accommodates without touching anything global.

**Recommendation**: b and c differ mainly in whether a boundary marker exists; both give the same bean
scale and both massively exceed the current rig. The lowest-risk option that captures most of c's benefit is
**format b framed tighter** - keep the box (positive boundary, trivial brightness-based detection) but fill
more of the frame with it. Failing that, c with the ingest QA gate above.

## Cross-rig generalization test: the model does NOT transfer (2026-08-09)

First real test of the adopted model on a different capture rig - the format-c photo (frame-filling, no
container). User confirmed afterwards that **the bean is one of the 9 trained classes**, and that it is
neither of the model's top two predictions. So this is not an open-set problem; the correct answer was
available and the model missed it completely.

**Result: the prediction carries essentially no information.**

| evidence | value |
|---|---|
| top-1 as shot | Vietnam-Robusta, p=0.755 (40 patches; 0.972 on the same beans in format b) |
| probability given to the true class | **<= 0.0098** |
| embedding distance to nearest class centre | **1.92x** vs 0.97 mean / 1.24 max over 27 held-out training photos |
| embedding distance to *all* nine centres | 1.91-2.86x - roughly equidistant, top three within 0.08 |
| effect of a 15% brightness change | **flips top-1 between Vietnam and Kenya** |

The last row is the decisive one. A classifier whose answer flips between its top two classes under a 15%
global brightness change is not discriminating beans - it is responding to global appearance. Both of the
classes it oscillates between are wrong.

### What was diagnosed

- **White balance is not the problem.** Image c's R/G/B ratio is 1.170/1.021/0.810 against training's
  1.179/1.022/0.799 - inside the training spread.
- **Absolute brightness is.** Mean RGB [147,128,102] vs training [102,88,69]; the new photo is ~46% brighter
  in grey mean (126 vs 86). Training augmented brightness by `color_jitter_strength=0.2`, i.e. a factor of
  0.8-1.2, so the new rig sits well outside the range the model ever saw.
- **Scale is a lesser factor**: the prediction was stable across patch sizes 700-1400, so the ~9% bean-scale
  difference is not what broke it.

### The one thing that *did* work: an out-of-distribution guard

Penultimate-layer (512-d) embedding distance to the nearest training-class centre, normalized by that
class's own spread, separates cleanly: 27 held-out training photos score mean 0.97 / p95 1.19 / max 1.24,
while the new photo scores 1.92-1.94 across three seeds. A threshold around 1.4 would have refused this
prediction rather than emitting a confident wrong answer, where **softmax confidence was useless** (0.97 on
format b, indistinguishable from its in-distribution 0.997).

Note what the guard does *not* do: the embedding is roughly equidistant from all nine classes, so it flags
"do not trust this" without offering a usable alternative. Explicitly not guessing a third class here - the
brightness experiment shows the outputs are arbitrary on this input, and a further guess would be false
precision.

### Implications

1. **Cross-rig generalization is now measured, and it is nil.** Every accuracy number in this log
   (test macro-F1 ~0.91-0.96) describes performance *on the 2026-08-07 rig only*. That was always the
   caveat; it is now a measurement rather than a caveat.
2. **Phase 8's hypothesis 5 (illumination) deserves revisiting on this evidence.** It was rejected in exp 40
   for a small in-distribution cost, with the explicit note that its actual claim could not be validated
   against a single-session test set. This is the missing evidence - though note exp 40 tested a *spatial
   gradient*, whereas the failure here is a *global* brightness offset, which `color_jitter` nominally
   covers but only to +/-20%.
3. **Concrete candidate fix**: per-photo brightness normalization as a preprocessing step (not augmentation)
   at both train and inference, normalizing grey mean while preserving R/G/B ratios - which keeps the bean
   colour information that ratio carries. Must be tested, not assumed: absolute brightness may itself carry
   some class signal, so this could cost in-distribution accuracy.
4. **Ship the OOD guard regardless.** Refusing to predict is strictly better than a confident wrong answer,
   and the threshold is empirically supported.

# Phase 10 (planned) - Refuse to predict out of distribution

Cheap, no retraining, and strictly better than the current behaviour: today the model answers every photo
with high confidence regardless of whether it has any business doing so.

**Method** (validated in the 2026-08-09 cross-rig test above): take the penultimate 512-d embedding, measure
distance to the nearest training-class centre, normalize by that class's own mean spread. Held-out training
photos score mean 0.97 / p95 1.19 / max 1.24; the out-of-rig photo scores 1.92-1.94 across three seeds.

**Steps**

1. Compute per-class centroids and spreads from the training split and save them **beside the checkpoint**,
   so the guard travels with the model (extend `models/<name>.json`, which already carries the model card).
2. Add the guard to `infer.py`: report the distance alongside the prediction, and emit
   `unknown / out-of-distribution` above a threshold.
3. Calibrate the threshold at ~1.4 (above the 1.24 in-distribution max, below the 1.92 observed failure) and
   record the basis, not just the number.
4. **Acceptance**: must pass every held-out training photo, and must refuse the format-c photo. Report the
   margin, since one positive example is thin evidence for a threshold.

**Deliberately limited**: the guard says "do not trust this", it does not identify the class. On the format-c
photo the embedding is roughly equidistant from all nine centres (1.91-2.86, top three within 0.08), so
there is no usable alternative prediction hiding in it. Do not oversell it as a fallback classifier.

# Phase 11 (planned) - Brightness normalization and augmentation for generalization

User's direction: brightness handling belongs in the augmentation pipeline alongside rotation, mirroring and
zoom, and generalization across rigs is a first-class goal rather than a footnote.

## The prerequisite, and it is not optional

**A labelled second-rig validation set.** Phase 8 tested five augmentation hypotheses and rejected four of
them - and every one of those verdicts was measured on a test set drawn from the *same 2-hour shoot* as
training. That metric structurally cannot see a generalization benefit. Rotation jitter "no effect", zoom
"not adopted", illumination "fails the costs-nothing bar", perspective "deferred": all of those were judged
on the one axis where such augmentations are *expected* to look neutral or slightly negative, while their
actual payoff is on an axis that did not exist yet. **Phase 11 should not start until some format-b/c photos
are labelled**, or it will repeat exactly that mistake with more compute.

Suggested minimum: 3-5 photos per class on the new rig - enough for a cross-rig macro-F1 with a usable
noise band, far short of a full training session.

## Normalization and augmentation are different tools; use both

- **Normalization** is deterministic preprocessing applied at train *and* inference: scale each photo so its
  grey mean matches a fixed reference, preserving R/G/B ratios so bean colour - which is real class signal -
  survives. This removes a *rig-level offset*. The 2026-08-09 photo sits at grey mean 126 against training's
  86, roughly +46%, far outside anything training ever saw.
- **Augmentation** is random variation during training, teaching invariance to *residual* variation the
  normalizer cannot remove (per-photo drift, gradients across a single frame).

They are complementary, and the honest expectation is that normalization does most of the work here.

## Steps

1. **Per-photo brightness normalization** in the crop or dataset stage, applied identically at inference.
   Test in-distribution first: absolute brightness may itself carry class signal, so this could cost
   accuracy on the current rig. That cost is acceptable if cross-rig improves, but it must be *measured*,
   not assumed.
2. **Widen the brightness augmentation range** well beyond `color_jitter_strength=0.2`'s 0.8-1.2 factor,
   which is narrower than the observed rig-to-rig shift.
3. **Re-test the Phase 8 augmentations under the new two-axis metric** - rotation jitter, zoom, illumination
   gradient, and perspective (never implemented on real pixels). Their rejections are not wrong, but they
   answered a different question than the one that now matters.

## Decision rule for Phase 11 (this is the part that changes)

Report **two** numbers per experiment: in-distribution test macro-F1 (the existing metric and noise band)
and **cross-rig macro-F1** on the new validation set. Adopt on a cross-rig gain, provided the in-distribution
cost stays within its noise band. This inverts Phase 8's bar, where in-distribution was primary - which was
correct while cross-rig could not be measured, and is wrong now that it can.

Keep the paired multi-seed standard from Phase 8: sign consistency across seeds, not effect size on one run.

### Exp 47: asymmetric crop trim, patch_crop_size unchanged at 900

Isolates the crop change from any patch-size change. Wider crops (valid region 1016 -> 1067px) mean more
placement room at patch 900: 116px -> 167px. Same params as exp 39 in every respect; only the session's
crop config differs. Ran 12:04-13:28 UTC, 84 min.

| # | crop | val_macro_f1 | val_mcc | test_macro_f1 | test_mcc | best_epoch | epochs run |
|---|---|---|---|---|---|---|---|
| 39 | symmetric 0.10 | 0.9635 | 0.9597 | 0.9554 | 0.9509 | 29 | 37 |
| 47 | asymmetric per-side | 0.9581 | 0.9537 | **0.9554** | 0.9503 | 31 | 39 |

**No effect - and that is the useful outcome here.** test_macro_f1 is identical to four decimals
(-0.0000) and test_mcc -0.0006; val is down 0.0054/0.0061, comfortably inside its band. So the extra
placement room bought nothing at patch 900, which is not surprising in hindsight: 116px was evidently
already enough, and exp 32's failure was at 17px, an order of magnitude tighter.

The point of the wider crop was never a direct win at patch 900 - it was to make `patch_crop_size=1000`
testable at all (placement room 17px -> 67px). Exp 47's value is that it establishes the crop change is
**neutral**, so any effect exp 48 shows can be attributed to patch size rather than to the crop. Not
adopted or rejected on its own merits; kept, because it is the enabler and costs nothing.

**Tooling gap this exposed, now fixed**: `compare_experiments 47 --vs 39` reported "changed: (nothing -
identical config)". That is literally true - `params.yaml` is byte-identical, because crop settings live per
session in `<session>.crop.yaml` by design. Correct architecture, incomplete record: a run's `config.json`
had no trace of the data it was trained on. `archive_experiment.py` now copies the session crop configs into
each archived run, so the record states the data as well as the hyperparameters.
