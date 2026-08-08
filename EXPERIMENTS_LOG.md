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
