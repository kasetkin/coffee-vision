# ML-37 D13: what the retrain has to show, fixed before the runs

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D13, 2026-11-23, on the local checkout at
`01e6ac1`. A recommendation for the owner, not a decision: the bar becomes D13's answer only when the owner adopts
it. Nothing was trained. Every number about this repo comes from the archived runs in `experiments/` (read only),
recomputed by `analysis/ml37/d13_seed_range.py` into `analysis/ml37/d13_seed_range.txt`. Sections A to F named
below are that file's. Facts from earlier tickets are cited to them, not re-derived. The statistical sources were
read from the documents themselves; §4 quotes them.

## Summary

- **Recommended: (a), no regression, stated as a non-inferiority bar with a margin fixed now.** Option (b),
  improving by more than the seed range, would need a paired gain of about +0.019 in test macro-F1 (the P10
  baseline's range). No change in the country era has come near that. The largest paired gain the record has is
  the augmented head's, +0.003 to +0.009 on average (§3).
- **"The seed range" mostly measures which photos land in test, not noise in the fit.** In this repo the seed
  draws each class's photo split and the patch boxes. The head is convex, so refitting the same input at the same
  seed reproduces the run (`fit_frozen_head.py`: "Seeds only vary the patch draw (the head is convex)";
  `dataset.split_photos_by_class`). Across the ten country-era triples, the range of test macro-F1 over seeds
  42/123/7 is 0.0046 to 0.0230, median 0.0136. For the live model's family it is 0.0191 (exp286-288) and 0.0127
  (exp289-291). A single run's photo-bootstrap SE is 0.007 to 0.009 (section C), so the range is about two to
  three test-sampling SEs.
- **Pairing takes the split out, and what remains is about ten times smaller.** In ML-29 P10 the arms disagree on
  1.7% to 2.2% of test patches. The paired mean difference is −0.0009, with a photo-level paired bootstrap SE of
  0.0018, 95% CI [−0.0045, +0.0025]. The three deltas' own SE is 0.0020 (section C). Per class, the SE of the mean
  paired difference is 0.002 to 0.008.
- **The literal per-class rule misbehaves.** Take "no class's F1 drops beyond the seed range", reading the range
  off the baseline's three seeds. The class ranges run from 0.000 (Vietnam, 1.000 at every seed) to 0.082
  (Colombia, 0.840 to 0.922). That makes the rule hair-trigger on easy classes and blind on hard ones: three
  Colombia test photos lost (mean −0.054) are caught 9% of the time (section E). The sign rule, "lower at all three
  seeds", fires on some class in 63% of null draws (section D).
- **Proposed instead.** The overall statistic is the mean paired delta in test macro-F1, non-inferior if its 95%
  photo-bootstrap lower bound is above **−0.010**. Per class, a class regresses if its mean paired F1 delta is below
  **−0.02**. Under P10's level of disagreement, the overall bar fails a change that does nothing well under 0.1% of
  the time. The per-class flag fires on some class 0.3% to 0.7% of the time, and catches one lost Kenya photo 88%
  of the time and two lost Colombia photos 83% of the time (sections D, E, F). P10 itself would have met it: test
  −0.0009 [−0.0045, +0.0025], no class below −0.006.
- **The OOD probe and the segmenter get floors, not adoption bars**: the probe refitted keeps catching every test
  negative and refuses at most 4 of the 295 bean photos (today 164/164 and 2/295). Before the retrain, the masks
  on the new input are compared with today's (ML-29 P9's IoU), with the owner opening the outliers. The exact
  wording is in §9.

## 1. What a seed is in this recipe, and what "the seed range" measures

- The classifier is a frozen DINOv3 ViT-B/16, `cls_mean`, with an L2 logistic-regression head at fixed C = 0.1 and 8
  augmented views a training patch (ADR 0006, ADR 0025). The head is convex. The seed enters in three places:
  1. Each class's photo split: `split_photos_by_class`, `rng = np.random.default_rng([seed, class_idx, 9999])`.
  2. Each photo's patch boxes, keyed `[seed, class_idx, photo_idx, split]` (ML-8 D2).
  3. The augmented views, `SeedSequence([seed, index, k])`.
- At a fixed seed, machine and thread count, a refit is bit-identical (`docs/agents/experiments.md`, "Reproducibility
  limits"). Across thread counts, float reduction order changes. So a "seed range" over 42/123/7 is mostly the
  spread between three different test sets of about 266 photos, plus three different training draws. Bouthillier
  et al. found data sampling the largest of these variance sources (§4.4).
- The recorded metric is **patch** macro-F1: 160 test patches a class, drawn from that class's test photos
  (`metrics.json` `support` 160 a class). Patches of one photo are correlated, so the unit that varies is the
  photo. Every interval here resamples photos, not patches. Bouthillier et al. note the same: correlated errors
  give fewer degrees of freedom and a wider distribution (§4.4).
- Consequence for D13: in a paired comparison, the arms see the same test photos and the same boxes at each seed.
  That holds when the pool's photo names are unchanged, as ML-37's re-canonicalization keeps them. What differs is
  only how the head, refitted on slightly different features, falls on them. ML-29 P10 showed this directly: the
  photos the arms shared "move as much as the swapped ones" (`analysis/ml29/README.md`, P10; test: swapped
  −0.0014, shared −0.0009). That is the noise a no-regression bar must tolerate. It is a property of the refit, not
  of the change, and it is much smaller than the seed range.

## 2. The seed range in the record (section A)

Test and val patch macro-F1 over seeds 42/123/7. Per class: the range of test F1 over the seeds, median and largest
over the ten classes.

| Triple | What changed | Test range | Val range | Per-class test range, median / max |
|---|---|---|---|---|
| exp261-263 | ML-3 country classes | 0.9668-0.9808 (0.0140) | 0.0089 | 0.020 / 0.040 Peru |
| exp264-266 | ML-5 XL0-v2 crops | 0.9631-0.9764 (0.0133) | 0.0209 | 0.021 / 0.061 Guatemala |
| exp267-269 | ML-6 XL0-v3 crops | 0.9632-0.9748 (0.0116) | 0.0114 | 0.016 / 0.053 Guatemala |
| exp270-272 | ML-8 one pool, aug8 | 0.9767-0.9813 (0.0046) | 0.0094 | 0.008 / 0.039 Peru |
| exp273-275 | ML-9 XL0-v4 crops | 0.9649-0.9807 (0.0157) | 0.0031 | 0.013 / 0.068 Guatemala |
| exp276-278 | XL0-v5 crops | 0.9694-0.9813 (0.0119) | 0.0104 | 0.022 / 0.057 Guatemala |
| exp280-282 | ML-16 xl0_v7r crops | 0.9556-0.9762 (0.0206) | 0.0176 | 0.033 / 0.062 Guatemala |
| exp283-285 | ML-27 new photos | 0.9500-0.9731 (0.0230) | 0.0180 | 0.027 / 0.076 Colombia |
| exp286-288 | ML-29 P10 baseline | 0.9401-0.9592 (0.0191) | 0.0058 | 0.034 / 0.082 Colombia |
| exp289-291 | ML-29 P10 gain maps, the live family (exp290 shipped) | 0.9433-0.9560 (0.0127) | 0.0124 | 0.032 / 0.074 Guatemala |

- Median test range over the ten triples: 0.0136 (min 0.0046, max 0.0230). The ranges have grown since ML-16, as
  the pool grew to its harder photos (Ethiopia 26 to 46 test photos, Colombia 15 to 33, Brazil 29 to 58).
- The per-class seed range of the baseline a D13 bar would read (exp286-288): Vietnam 0.000, Brazil 0.009,
  Indonesia 0.018, Kenya 0.019, Rwanda 0.029, Costa Rica 0.039, Ethiopia 0.048, Peru 0.052, Guatemala 0.062,
  Colombia 0.082.
- Test photos a class at each seed of exp286-288 (the same counts at all three): Ethiopia 46, Kenya 14,
  Colombia 33, Costa Rica 17, Guatemala 18, Brazil 58, Vietnam 14, Indonesia 23, Peru 25, Rwanda 18. So one
  photo carries 160/14 ≈ 11 of Kenya's or Vietnam's test patches, but only about 3 of Brazil's.
- The pool has grown since exp286-291. ML-35 added 45 accepted photos (classes 013 and 001) and ML-36 3 (016, 017),
  with crops re-run and the owner's review done (both tickets' status rows). Those classes' splits are therefore
  redrawn, and **no archived run is a paired baseline for ML-37**. The baseline has to be refitted on today's pool,
  as P10 refitted exp286-288 rather than pairing with the shipped exp283-285 (`analysis/ml29/README.md`, P10, "Two
  departures").

## 3. Paired differences between arms that should be equivalent, and real small effects (sections B, C, F)

**P10, the one clean paired near-null in the record.** exp289-291 (gain maps applied to 1,255 photos' crops)
against exp286-288 (today's crops), same photos, split and boxes:

| Seed | Test base → new | Δ | Paired bootstrap SE | Val Δ | Photos right → (of 266) | Right→wrong / wrong→right |
|---|---|---|---|---|---|---|
| 42 | 0.9592 → 0.9560 | −0.0032 | 0.0039 | −0.0094 | 264 → 261 | 3 / 0 |
| 123 | 0.9586 → 0.9559 | −0.0027 | 0.0026 | +0.0006 | 262 → 259 | 3 / 0 |
| 7 | 0.9401 → 0.9433 | +0.0031 | 0.0025 | −0.0064 | 257 → 258 | 1 / 2 |

- Mean test Δ −0.0009. Bootstrap (test photos resampled within class, the same draw for both arms) SE 0.0018, 95%
  CI [−0.0045, +0.0025]. The three deltas' own SD gives SE 0.0020. A t-interval on three values,
  [−0.0097, +0.0078], is too wide to judge anything: with 2 degrees of freedom t = 4.30.
- Val: −0.0051, CI [−0.0095, −0.0007], statistically lower and small.
- Per class on test, the mean paired Δ and its bootstrap SE: Kenya −0.006 (0.004), Guatemala −0.004 (0.006),
  Brazil −0.003 (0.004), Indonesia −0.003 (0.003), Peru −0.002 (0.005), Vietnam 0.000 (0.000), Colombia +0.001
  (0.008), Rwanda +0.001 (0.002), Costa Rica +0.002 (0.004), Ethiopia +0.005 (0.007).
- Photo-level answers (mean of a photo's patch probabilities, as serving averages its 40) moved by up to 3 photos
  of 266 at a seed, so a "no answer changes" rule would fire even on this near-null. ML-29 P9 saw the same with
  the unchanged live model: 6 of 1,684 answers changed (`analysis/ml29/README.md`, P9).

**A real but small effect, for scale.** Every aug8 run fits an unaugmented head on the same patches (ADR 0025).
Augmented minus unaugmented test macro-F1, mean over seeds: +0.0038 (exp270-272), +0.0032, +0.0075, +0.0067,
+0.0075, +0.0067, +0.0094 (exp289-291). Per class, |Δ| median 0.006, 90th percentile 0.022, max 0.040. This is the
lever the owner adopted (ML-8). A margin wider than its gain could hide its loss.

**Mixed comparisons (split redrawn), for the owner's history.**
- ML-16: exp280-282 against exp279/277/278 scored −0.0080 (−0.0138 to −0.0013), lower at every seed, with Ethiopia
  −0.024 per class (`analysis/ml16/compare_280_282.txt`). The owner shipped it (ML-16 D11).
- ML-9: exp276-278 against exp273-275 moved test −0.0113 to +0.0113 (EXPERIMENTS_LOG.md).

These pairs mix the change with the split, so they bound nothing. They do show that the owner has accepted losses
of about 0.008 when a change was wanted for other reasons.

## 4. What the methodological sources say

### 4.1 ICH E9, *Statistical Principles for Clinical Trials* (Step 4, 5 February 1998)
- §3.3.2: "An equivalence margin should be specified in the protocol; this margin is the largest difference that
  can be judged as being clinically acceptable and should be smaller than differences observed in superiority
  trials of the active comparator. … only the lower margin is needed for the active control non-inferiority
  trial. The choice of equivalence margins should be justified clinically."
- §3.3.2: "for non-inferiority trials a one-sided interval should be used." The same section: "Concluding
  equivalence or non-inferiority based on observing a non-significant test result of the null hypothesis that
  there is no difference between the investigational product and the active comparator is inappropriate."
  - This is the flaw in reading "nothing beyond the seed range" as "no regression". A noisy measure makes that
    reading easier to pass, not harder.
- §3.3.2: "The equivalence (or non-inferiority) trial is not conservative in nature, so that many flaws in the
  design or conduct of the trial will tend to bias the results towards a conclusion of equivalence."
  - Here, an unpaired baseline or a split that differs between arms adds noise and biases toward "no
    regression".
- §3.5: the sample size of a non-inferiority trial "should normally be based on the objective of obtaining a
  confidence interval for the treatment difference that shows that the treatments differ at most by a clinically
  acceptable difference."
  - Here the test set is fixed, so the margin has to be chosen with the interval's width in view (§7).
- §5.5: "setting type I errors for one-sided tests at half the conventional type I error used in two-sided tests
  is preferable in regulatory settings", so a one-sided 2.5% test is the lower end of a two-sided 95% interval.
- §5.6: "When multiplicity is present … adjustment should always be considered and the details of any adjustment
  procedure or an explanation of why adjustment is not thought to be necessary should be set out in the analysis
  plan."
  - Here there are ten per-class checks (§6).

### 4.2 FDA, *Non-Inferiority Clinical Trials to Establish Effectiveness*, Guidance for Industry, November 2016
- §III.C (p. 8): the NI study "seeks to show that the amount by which the test drug (T) is inferior to the active
  control (C), C-T, is less than some prespecified NI margin (M)". Success is "the lower bound of the two-sided 95%
  confidence interval (equivalent to the lower bound of a one-sided 97.5% confidence interval)" (§III.A, p. 4, for the
  effect), and for NI, that the upper bound of the 95% CI for C−T is below the margin.
- §III.C (pp. 9-10): "M1 = the entire effect of the active control assumed to be present in the NI study; M2 = the
  largest clinically acceptable difference (degree of inferiority) of the test drug compared to the active
  control". "The choice of M2 is a matter of clinical judgment, but M2 can never be greater than M1."
  - Here M1, the classifier's whole effect over guessing, is about 0.85 in macro-F1 and does not bind. M2 is the
    owner's judgement.
- §III.C (p. 10): deficiencies in conduct "could reduce a difference between the active control and the new drug
  … in a non-inferiority trial, it is a 'bias toward the alternative'".
- §IV.G (p. 31): a trial planned to show NI "may also be used to test for superiority without concern about
  inflating the Type I error rate". But "An unplanned determination of non-inferiority following failure to show
  superiority, when the margin was not determined until results of the trial were known, would not be sufficient."
  - That is ML-29's situation: D16's superiority bar was not met and adoption was argued afterwards. It is why D13
    should fix the margin now.

### 4.3 Schuirmann (1987), the two one-sided tests
"A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average
bioavailability", *J. Pharmacokinet. Biopharm.* 15(6):657-680, doi:10.1007/BF01068419 (abstract, PubMed 3450848).
- The Power Approach tests "the hypothesis of no difference at level 0.05 and requir[es] an estimated power of
  0.80". The Two One-Sided Tests Procedure "leads to the same conclusion as the approach proposed by Westlake based
  on the usual (shortest) 1-2 alpha confidence interval". "With appropriate choice of the nominal level of
  significance of the one-sided tests, the two one-sided tests procedure always has uniformly superior properties
  to the power approach."
- Here D13 needs only the lower side: one one-sided test. With α = 2.5%, that test is the lower end of a 95%
  interval, consistent with E9 §5.5.

### 4.4 Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021, arXiv:2103.03098
The PDF's header reads "Proceedings of the 3rd MLSys Conference … 2020".
- §2: "Bootstrapping data stands out as the most important source of variance. In contrast, model initialization
  generally is less than 50% of the variance of bootstrap."
- §2: test-set noise follows a binomial, and "If errors are correlated, not i.i.d., the degrees of freedom are
  smaller and the distribution is wider". The variance due to data sampling "is well explained by the limited
  statistical power in the test set".
- §5: "Improvements smaller than this variance observed on a given test set will not generalize." Also:
  "randomize as many sources of variation as possible", "prefer multiple random splits to fixed test sets", and
  "account for the resulting variance when concluding".
- Their decision rule, P(A > B) ≥ 0.75 with a bootstrap CI (§4), is a superiority rule that needs many runs a
  side. Three seeds cannot estimate it. What carries over: this repo's seeds already randomize the split, as they
  recommend, so the unpaired seed range is mostly data-sampling variance. Pairing on the same photos removes that
  term from the comparison.

### 4.5 Dodge et al., "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping", arXiv:2002.06305 (2020)
- Abstract: "even with the same hyperparameter values, distinct random seeds can lead to substantially different
  results". Weight initialization and data order "contribute comparably to the variance of out-of-sample
  performance".
- Their seed variance comes from non-convex fine-tuning. Here the backbone is frozen and the head convex, so
  neither initialization nor data order exists (§1), and that source of variance does not apply. Their other
  finding does: the best-of-N seeds' validation score grows with N (§3.1, "expected validation performance"). So
  the bar compares the arms' means over the three seeds, not the shipped seed. The val pick happens after the
  bar, as today.

## 5. Small classes: what one photo does to a class's F1 (section E)

A photo wholly lost (every patch turned to the class's main confusion in the baseline) moves that class's mean test
F1 by about:
- Kenya (14 test photos): −0.046 for one photo, −0.089 for two
- Colombia (33): −0.017 for one, −0.035 for two
- Brazil (58): −0.013 for one, −0.023 for two

Vietnam's F1 is 1.000 at all three seeds, so its seed range is 0, and a rule reading that range fires on a single
Vietnam patch. Colombia's range, 0.082, lets about four Colombia photos go unnoticed. A per-class margin is
therefore better fixed in F1 units, the same for every class. At 0.02 it equals about one photo of a small class
(Kenya, Vietnam, Costa Rica, Guatemala, Rwanda), one to two of a mid-size class (Colombia, Peru, Indonesia), and two
of Brazil or Ethiopia.

## 6. How candidate rules behave (sections D, E, F)

**False alarms.** "Nothing changed" is simulated by swapping P10's two arms photo by photo at random, 4,000 draws:
the arms are exchangeable if nothing changed. A rule "fires" if any of the ten classes fails it.

| Rule (per class, over seeds 42/123/7) | Fires under no change (P10's disagreement) |
|---|---|
| mean Δ < −(baseline's per-class seed range), the literal D13 (a) | 0.6% |
| any seed's Δ < −(that range) | 33.9% |
| new arm's mean F1 below the baseline's lowest seed | 11.2% |
| Δ < 0 at all three seeds (ADR 0002's sign agreement) | 63.2% |
| mean Δ < −0.01 | 21.7% |
| **mean Δ < −0.02** | **0.3%** |
| mean Δ < −0.03 | 0.0% |
| Δ < 0 at all three seeds and mean Δ < −0.02 | 0.3% |

P10's actual deltas fire "any seed beyond the range" and "all three negative", and none of the others.

**Power.** A drop is planted (k photos lost, as in §5) and the photos resampled. Each cell is the share of draws in
which the rule flags that class:

| Planted drop | Literal seed-range rule | All three < 0 | **Mean Δ < −0.02** | Mean Δ < −0.03 |
|---|---|---|---|---|
| Kenya, 1 photo (−0.046) | 90% | 64% | **88%** | 78% |
| Kenya, 2 photos (−0.089) | 99% | 88% | **99%** | 97% |
| Colombia, 1 photo (−0.017) | 0% | 37% | **41%** | 16% |
| Colombia, 2 photos (−0.035) | 1% | 64% | **83%** | 61% |
| Colombia, 3 photos (−0.054) | 9% | 82% | **98%** | 91% |
| Brazil, 2 photos (−0.023) | 94% | 79% | **63%** | 20% |
| Brazil, 3 photos (−0.033) | 99% | 92% | **90%** | 60% |

**Sensitivity to a larger refit jitter.** ML-37 changes every photo a little, where P10 changed a third of them a
lot, and the disagreement may differ. In a normal approximation with the per-class null SDs scaled, the per-class
flag at −0.02 fires on some class:
- 0.7% at P10's SD (×1)
- 7.9% at ×1.5
- 22% at ×2

At −0.03 the same three figures are 0.0%, 0.6% and 4.2%. The new comparison's own SEs and its patch disagreement,
against P10's 1.7% to 2.2%, show which row applies, so the printout should carry them.

Reading: the per-class rule at −0.02 is the best of those measured. It holds false alarms under 1% at P10's
noise, and it catches a lost photo in a small class and two in a mid-size one most of the time. ADR 0002's sign
agreement, applied per class, cannot be a stop rule with ten classes. It stays what the ADR has it be, the reading
of the overall delta.

## 7. The overall statistic and its margin

- **Statistic.** The mean over the three seeds of the paired difference in test patch macro-F1 (new − baseline).
  The interval comes from a photo-level paired bootstrap: each seed's test photos are resampled within class, the
  same draw serves both arms, and the seeds' draws are independent. The statistic is the mean of the three
  resampled deltas, and the interval its 2.5th and 97.5th percentiles. This is E9 §5.5's one-sided 2.5% test, and
  Schuirmann's lower one-sided test.
- **Margin.** The table gives the chance that a change that does nothing still fails at margin M: the normal
  approximation P(mean − 1.96·SE < −M) at true Δ = 0. P10's SE is 0.0018.

| SE of the mean paired Δ | M = 0.005 | M = 0.0075 | **M = 0.01** | M = 0.015 |
|---|---|---|---|---|
| 0.0018 (P10) | 21% | 1.4% | **<0.1%** | 0% |
| 0.0027 (×1.5) | 54% | 21% | **4%** | 0% |
| 0.0036 (×2) | 72% | 45% | **21%** | 1.4% |

A true loss equal to M passes 2.5% of the time at any SE, by construction.

- **Why 0.01.** E9 asks the margin to be "smaller than differences observed in superiority trials of the active
  comparator". FDA calls M2 a judgement. Three things set it:
  1. The record's precision: at P10's SE, 0.005 would fail one change in five that does nothing.
  2. The record's typical sizes: 0.01 is about half the baseline's own seed range (0.0191) and below the median
     range (0.0136).
  3. The owner's past tolerance: the ML-16 shipment at −0.0080.
- **What 0.01 costs.** It is wider than the augmented head's gain (+0.003 to +0.009). A pipeline that silently lost
  the whole augmentation benefit could pass. That cost is why the per-class flag and the printed val, per-camera
  and photo-level numbers stay beside it. If the owner wants E9's "smaller than the comparator's effect" strictly,
  0.0075 is the tightest margin that P10's precision supports (a 1.4% false-fail rate). Below that, the test set
  is too small to show anything.
- **P10 under this bar.** Test lower bound −0.0045 > −0.01 and val −0.0095 > −0.01: both met. No class's mean is
  below −0.02, so the per-class flag stays quiet. Gain maps would have met (a), consistent with the owner's reason
  for adopting them: "they do not break anything on our dataset" (ML-29 D16).

## 8. The OOD probe and the segmenter

**The OOD probe.** It is refitted on every retrain (ML-16 D13, ML-27 D4, ML-29 P11) and measured on the segmenter
dataset's test split (ADR 0016). Its record, negatives caught and bean photos refused:

| Model | Negatives caught | Bean photos refused | Source |
|---|---|---|---|
| ML-8 | 160/160 | 0/73 | `analysis/ml8/p8_ood_eval_s123.txt` |
| ML-9 | 160/160 | 0/76 | `analysis/ml9/p6_ood_eval.txt` |
| XL0-v5 | 160/160 | 0/95 | `analysis/xl0v5/ood_eval.txt` |
| XL0-v6 | 160/160 | 0/135 | `analysis/xl0v6/ood_eval.txt` |
| ML-16 | 160/160 | 2/200 | `analysis/ml16/ood_eval.txt` |
| P11, live | 164/164 | 2/295 | `analysis/ml29/p11_ood_eval.txt`, the same two photos, both unseen and without a mask |

- 164/164 has a 95% lower bound of 97.8% (Clopper-Pearson). 2/295 is 0.7%, with a 95% upper bound of 2.4%.
- Under a binomial at 2/295, 5 or more refusals occur 5.2% of the time. So "every negative caught, at most 4 bean
  photos refused" stops roughly one unchanged probe in twenty, and stops a doubled refusal rate 37% of the time.
- The probe is fitted on the classifier's embeddings, which every photo's new read-back moves. Its numbers are
  computed anyway, so a floor costs nothing.
- It is a floor, not part of the adoption bar: the probe is fixed by its refit, not by the pipeline choice. ML-29
  P9 saw no verdict flip when only the input changed.

**The segmenter.** D14 (a) keeps `xl0_v7r_s123` and runs it on the new input, so only its input changes: the float
read-back, the float resize (D11) and the lossy coding.
- Its effect reaches the classifier through the crops, so the paired retrain measures it end to end.
- What the retrain does not measure is the masks on photos outside the pool, and the D18 fallbacks. ML-29 P9's
  measure covers both: the median IoU against today's masks per camera, photos under 0.9, and fallbacks gained or
  lost. With gain maps applied, P9 found the median at least 0.999 for every camera and 11 photos under 0.9
  (`analysis/ml29/README.md`, P9).
- The experiments runbook already has a re-crop opened at full size, with every fallback included and photos new
  to the crops judged by the owner ("Changing the segmenter", step 1).
- Proposed: P9's comparison on every dataset photo, run with R1 before any retrain, with a per-camera median IoU
  of at least 0.99. Every photo under 0.9, and every fallback gained or lost, is opened by the owner. As with the
  probe, a floor beside the bar, not a statistical bar of its own.

## 9. Recommendation: the bar's wording

Proposed for D13, option (a):

> **(a), stated as non-inferiority, fixed before the runs.** The baseline is today's recipe refitted on today's pool
> at seeds 42, 123 and 7, on the same machine and thread count as the new arm. The two configs differ only in the
> input (the runbook's key-by-key diff), and at each seed the two arms' test photos are the same (the prediction
> files' photo keys equal). The statistic is the mean over the three seeds of the paired difference in test patch
> macro-F1 (new − baseline). Its 95% interval comes from a photo-level paired bootstrap: test photos resampled
> within class, the same draw for both arms, each seed drawn independently. The new pipeline meets the bar if:
>
> 1. the interval's lower end is above **−0.010**; and
> 2. no class's test F1 is lower by more than **0.02** on average over the three seeds.
>
> Printed beside the bar, not deciding it: val (the same statistic), per camera, the photo-level answers (right→wrong,
> wrong→right), the share of test patches whose prediction changes (P10: 1.7–2.2%), and each class's bootstrap
> interval. Floors, checked beside it: the refitted OOD probe catches every negative of the segmenter test split and
> refuses at most 4 of its bean photos (today 164/164, 2/295). Before the retrain, the masks on the new input are
> compared with today's on every photo: median IoU per camera at least 0.99, and every photo under 0.9 and every
> D18 fallback gained or lost opened by the owner. Met, or not, the owner decides (ADR 0002).

- **Departure from ADR 0002.** The ADR has no numeric gate: "There is no automatic pass/fail rule … the owner
  decides". As with ML-29 D16, this bar is a reading the owner commits to before the runs, and the owner still
  decides either way. It adds the paired interval and the margin to the ADR's printout.
- **Departure from the ADR's sign reading.** Per class, sign agreement fires on 63% of no-change draws (§6). The
  bar uses a margin there instead. The ADR's sign reading of the overall deltas stays in the printout.
- **The bootstrap is new code.** Neither `compare_experiments` nor `p10_pool.py` computes it, and
  `analysis/ml37/d13_seed_range.py` has it, reading the archived prediction files. The plan should either put it in
  `compare_experiments` with a test or name this script as the reader.

## 10. What stays unverified

- **The size of ML-37's refit jitter.** The false-alarm and margin figures assume a disagreement like P10's,
  1.7–2.2% of test patches. ML-37 moves every photo, by lossy coding, the lost rounding and D6's tone map, where
  P10 moved a third of them by about 20 levels. If the jitter is twice P10's, the overall bar fails a change that
  does nothing about 21% of the time, and the per-class flag fires about 22% of the time (§6, §7). The comparison's
  own SEs show which applies. Nothing short of the runs measures it.
- **One near-null, not a distribution of them.** The record has one clean paired near-null (P10); the permutation
  null is built from it. Two refits of the same input would give no null at all, being identical (§1).
- **The bootstrap treats the three seeds' test sets as independent.** They overlap, being drawn from one pool. The
  bootstrap SE (0.0018) agrees with the three deltas' own SE (0.0020), which suggests the overlap costs little,
  but three values cannot show it.
- **The power figures plant a whole-photo loss** into the class's main confusion. A real regression may spread
  over many photos' patches. Then the patch-level F1 moves with less photo-to-photo variance, so these figures
  are, if anything, conservative.
- **The probe and segmenter floors (4/295, IoU 0.99) are judgements** from the record's spread, not derived from a
  stated loss.
- **Classes 013, 001, 016 and 017 were redrawn by ML-35 and ML-36.** Their test photo counts in the refitted
  baseline will differ a little from exp286-288's.

## Files

- `analysis/ml37/d13_seed_range.py`: every repo number above. It reads `experiments/` only, trains nothing, and runs
  with 2 threads:
  `cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d13_seed_range.py`. Its output, sections A to
  F, is `analysis/ml37/d13_seed_range.txt`.
- Sources read, downloaded into `tmp/ml37/d13/downloads/` (not committed) and read as text:
  - ICH E9, <https://database.ich.org/sites/default/files/E9_Guideline.pdf>: §3.3.2, §3.5, §5.5, §5.6.
  - FDA, *Non-Inferiority Clinical Trials to Establish Effectiveness* (November 2016),
    <https://www.fda.gov/media/78504/download>: §III.A, §III.C, §IV.G.
  - Schuirmann 1987: its abstract, PubMed 3450848, doi:10.1007/BF01068419. The full text is paywalled and was not
    read.
  - Bouthillier et al., <https://arxiv.org/pdf/2103.03098>: §2, §4, §5.
  - Dodge et al., <https://arxiv.org/pdf/2002.06305>: the abstract and §3.
- The repo's record:
  - ADR 0002, 0006, 0016, 0025
  - `docs/agents/experiments.md`, `EXPERIMENTS_LOG.md`, `experiments/index.csv` and `experiments/exp261`-`exp291`
  - ML-29's ticket (D16, the Activity) and `analysis/ml29/README.md` (P9, P10, P11), `p10_compare.txt`,
    `compare_289_291.txt`, `p11_ood_eval.txt`
  - ML-16's D11 and `analysis/ml16/compare_280_282.txt`, `common_photos.txt`
  - `analysis/ml27/common_photos.txt`
  - ML-3's D10 and D11
  - ML-35's and ML-36's status rows
  - `coffeecv/fit_frozen_head.py`, `coffeecv/dataset.py`, `coffeecv/compare_experiments.py`
