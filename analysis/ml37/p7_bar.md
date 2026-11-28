# ML-37 P7: D13's bar, fixed before the runs

Written 2026-11-27, before any of P7's head fits (exp292-294) had run. The comparison script that applies it is
`analysis/ml37/p7_compare.py`; its run on exp289-291 against themselves (the script's check, before the baseline
of ML-37 D29 existed) is `analysis/ml37/p7_compare_baseline_self.txt`. Nothing below may be changed once a
candidate's numbers exist. The baseline was changed before any run, by the owner (ML-37 D29, below).

## The bar, as the ticket words it

[ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D13, "What does the retrain have to show, fixed
before the runs?", answered (a), 2026-11-26:

> (a), 2026-11-26, the owner ("D9 answer (a), D12 answer (a), D13 answer (a), D14 answer (a): first implement this
> ticket, all fine-tunning after that"), with the research's wording below. The aim is a correct pipeline with
> nothing lost; §2's research expects no gain. Research, 2026-11-23 (research_d13_adoption_bar.md), rewords (a): the
> mean paired change in test macro-F1 over the three seeds, with its 95% photo-bootstrap lower bound above −0.010; no
> class's mean paired F1 change below −0.02; the OOD probe still catching every test negative and refusing at most 4
> of the 295 bean photos; before the retrain, a median mask IoU of at least 0.99 per camera against today's. (b) needs
> about +0.019, more than any change in the record. The bootstrap is new code.

The [plan](../../docs/ml-0037_plan_linear_canonical_photo.html)'s P7:

> The comparison, paired by seed against exp289–291 (ADR 0002): mean test macro-F1 change with its 95%
> photo-bootstrap lower bound above −0.010; no class's mean change below −0.02; the OOD probe refitted on the
> seed-123 candidate, catching every test negative and refusing at most 4 of the 295 bean photos.

The mask-IoU floor was for before the retrain; P5 judged it (`analysis/ml37/p5_report.txt`). P7 applies the other
three parts:

| # | Part | Met when |
|---|---|---|
| 1 | Overall | The mean over seeds 42, 123 and 7 of the paired change in test patch macro-F1 (candidate − baseline) has a 95% photo-bootstrap lower bound **above −0.010** (strictly). |
| 2 | Per class | No class's mean paired test F1 change over the three seeds is **below −0.02**. |
| 3 | OOD probe | The probe refitted on the seed-123 candidate (exp293) catches **every** negative of the segmenter dataset's test split and refuses **at most 4 of the 295** bean photos, at the fixed threshold 0.5 (ADR 0016). |

Pairs (ML-37 D29): exp292/295 (seed 42), exp293/296 (seed 123), exp294/297 (seed 7). The baseline is exp295-297,
today's live pipeline refitted on today's pool: `main`'s code at `617c944` (the 8-bit resizes, the canonical photo
before ML-37 D1) and its pool, `data/segcropped/pool` at `95d5dbea06dbf41362ab7043913f6317.dir`, the pool both
`main` and the store before P5 (`f0d36ab`) name. The candidate, exp292-294, is this branch's code on the pool P6
re-cropped. Both arms have the same pool's photos under the same names, so the same split and the same test photos
at each seed: the comparison is fully paired. The plan paired against exp289-291 (the live family, exp290 shipped);
D29 replaced them, since ML-35 and ML-36 had grown the pool and redrawn three classes' splits since. The bar is met
when all three parts are met. Met or not, the owner decides (ADR 0002); the research's §9 notes that this numeric
reading is a departure from ADR 0002's "no automatic pass/fail rule" that the owner committed to before the runs.

## How the script reads it

- **The statistic.** Each seed's test patch macro-F1 (the mean of the 10 classes' F1, as `metrics.json` records it)
  recomputed from `predictions_test.csv`, candidate minus baseline, averaged over the three seeds.
- **The bootstrap.** The unit is the photo, not the patch: a photo's patches move together. A photo is its
  `(true_label, folder_id, photo)` triple in `predictions_test.csv`, as `analysis/ml37/d13_seed_range.py` keys
  it. Test photos are resampled with replacement within each class, the same draw for both arms on photos both arms
  have, each seed drawn independently; a drawn photo's patches enter with its multiplicity as their weight. 2000
  resamples, RNG seed fixed in the script. The lower bound is the 2.5th percentile of the resampled mean change,
  the lower end of the two-sided 95% percentile interval the research's §9 words ("Its 95% interval ... the
  interval's lower end").
- **Photos one arm has and the other does not.** Under D29 there should be none: the script prints the shared
  counts, and a photo on one side only is a fault to explain before the bar is read. The script still handles it:
  shared photos get one draw for both arms, each arm's own photos their own, each within its class; with every
  photo shared this is the research's paired bootstrap. (Against exp289-291, which the plan named, about 200 of 266
  test photos per seed were shared, ML-35 and ML-36 having redrawn Ethiopia, Brazil and Peru: why D29 refits.)
- **Per class.** The mean over the three seeds of each class's paired test F1 change. The worst class decides.
  Each class's bootstrap interval is printed beside it but does not decide part 2.
- **The OOD probe.** `ood_eval --out`'s JSON: negatives caught is `negatives.all.caught` of `negatives.all.n`;
  bean photos refused is `positives.all.trained.refused + positives.all.not_trained.refused` of their `n` (an
  unmeasurable photo counts as refused, as `ood_eval` counts it). If the total is not 295, the report says so.
- **Printed beside the bar, not deciding it** (research §9): ADR 0002's per-seed table (val and test, both arms,
  the deltas, their mean and range, per-class F1), val's same statistic, each class's interval, the photo-level
  answers on shared photos (right→wrong, wrong→right), and the share of patches whose prediction changed.

## On a miss: D21

ML-37 D21, "What happens if the refit misses D13's bar?", answered (a), 2026-11-26:

> (a), 2026-11-26, the owner ("Q2–Q5 as recommended"). The refit changes the transfer, the intensity target, the
> resizes (D11 (b)) and the lossy coding at once; D11 (b) is the one that can be taken out. On a miss, the head is
> refitted with the 8-bit resizes (D11 (a)) and judged by the same bar; passing adopts that, with D11 recorded as
> reversed. A second miss closes the ticket unadopted: the branch is never merged (nothing reaches main before P8),
> and the reasons go into `.out-of-scope/`.

The plan's P7 says the same, adding that "a second miss goes to P8's second branch", and that P7 stops for the
owner either way: adopt (P8's release) or not (P8's second branch). The second run uses the same seeds, the same
baseline (exp295-297) and this same bar and script.
