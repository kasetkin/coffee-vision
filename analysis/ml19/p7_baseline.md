# ML-19 P7: the baseline (D14, D48)

The live models (`models/dino3b16_segxl0v6_country_aug8_s42.pt`, its OOD probe, segmenter XL0-v6 s42) on the
canonical photos' crops, beside their own numbers on today's crops of the originals. This is ML-16's paired baseline
(ADR 0002): ML-16's retrain is compared with these numbers, not with exp279's.

Written by `analysis/ml19/p7/baseline.py` (`baseline.txt`, `baseline.json`, `predictions_test.csv`),
`analysis/ml19/p7/paired.py` (`paired.txt`) and `python -m coffeecv.ood_eval --checkpoint
models/dino3b16_segxl0v6_country_aug8_s42.pt --no-tta --out analysis/ml19/p7/ood_eval.json` (`ood_eval.txt`), on
2026-11-04, over `data/segcropped/pool` as P6 left it (`dvc status` clean).

## Read this first: the test split is not exp279's

P6's re-crop changed the pool: `random_date_raccoon`'s crops were stale (133 of its photos renamed since, 50 added) and
three photos moved to `example_photos/`, so the pool holds 1,429 photos against 1,379. The split is drawn per class
from the pool, so it changed too: of exp279's 209 test photos, 103 are in this test split; 114 of its 217 are new to
it. The whole-split numbers below mix the two changes. The paired section separates them.

## The whole test split (1,600 patches, 160 a class)

| Class | F1, exp279 (originals) | F1, canonical | Recall, exp279 → canonical |
|---|---|---|---|
| Ethiopia | 0.9814 | 0.9191 | 0.9875 → 0.8875 |
| Kenya | 0.9444 | 0.9623 | 0.9563 → 0.9563 |
| Colombia | 0.9563 | 0.9156 | 0.9563 → 0.8812 |
| CostaRica | 0.9938 | 0.9969 | 0.9938 → 0.9938 |
| Guatemala | 0.9560 | 0.9202 | 0.9500 → 0.9375 |
| Brazil | 0.9782 | 0.9666 | 0.9812 → 0.9938 |
| Vietnam | 0.9969 | 1.0000 | 1.0000 → 1.0000 |
| Indonesia | 0.9938 | 0.9969 | 0.9938 → 1.0000 |
| Peru | 0.9778 | 0.9366 | 0.9625 → 0.9688 |
| Rwanda | 0.9969 | 0.9969 | 0.9938 → 0.9938 |
| **Macro F1** | **0.9775** | **0.9611** | accuracy 0.9775 → 0.9613, MCC 0.9750 → 0.9571 |

No D18 fallbacks in the test split, and no patches below the bean-share rule.

## Paired: the photos both splits hold

| On the 103 shared photos | Originals (exp279) | Canonical |
|---|---|---|
| Patches | 908 | 900 |
| Patch accuracy | 0.9791 | 0.9811 |
| Macro F1 (patches) | 0.9478 | 0.9464 |
| Photo accuracy (mean probability's top class) | 1.0000 | 1.0000 |

Not one photo's vote changed. On the same photos, the canonical input costs nothing measurable: patch accuracy moves
by +0.002 and macro F1 by −0.001. That is within what another decode and another mask do to the patches. The patches
themselves differ (8 fewer, since the masks moved slightly), so this is a per-photo comparison, not a per-patch one.

## Where the drop comes from: the 114 photos new to the split

| Photos only in the new split | Photos | Patch accuracy | Wrong patches | Photo accuracy |
|---|---|---|---|---|
| Dated names (`random_date_raccoon`: renamed since exp279, or new to the pool) | 25 | 0.7622 | 34 of 143 | 0.8400 |
| The older names (moved into test from exp279's train or val) | 89 | 0.9803 | 11 of 557 | 0.9888 |

The photos with the older names score as the shared ones do. The 25 dated photos hold 34 of the split's 45 new
errors. They come from the session's mix of phones (KB2005, G8441, OnePlus, Galaxy Z Fold8 Ultra, Pixel 7, Pixel 9 Pro), with
two Pixel 9 Pro photos of Ethiopia wrong on every patch (`20261103_110934`, `20261103_110947`). The Ethiopia and
Colombia recall drops are mostly those photos. Of the 89, 20 were in exp279's val split and the other 69 in neither its test nor its val, so in its train split.
They are easier than truly held-out photos. That makes the drop, if anything, an underestimate of what
the new photos cost.

**Reading:** the input's change is neutral. The new test split is harder because it holds `random_date_raccoon`'s new
and renamed photos, the photos ML-16 trains on. ML-16's retrain is compared with **0.9611 macro F1 / 0.9613
accuracy on this split**, not with exp279's 0.9775.

## The OOD guard (probe threshold 0.5, ADR 0016; no TTA)

| | Today (`analysis/xl0v6/ood_eval.txt`, originals) | Canonical |
|---|---|---|
| Test photos scored | 295 | 318 |
| Negatives caught | 160 / 160 | 160 / 160 |
| Positives refused, trained on | 0 / 64 | 0 / 66 |
| Positives refused, not trained on | 0 / 71 | 0 / 92 |
| Segmenter on the negatives: empty mask | 154 / 160 | 152 / 160 |
| Segmenter on the negatives: empty or tiny | 156 / 160 | 156 / 160 |

The guard's refusals are unchanged: every negative caught, no positive refused. Two `user_realworld` negatives that the
segmenter left empty now get a tiny mask. The empty-or-tiny count is the same, and the probe catches both.

## For the owner

- Read the baseline: the paired photos show no cost from the canonical input, and the whole split's drop is the new
  photos. ML-16's retrain is unblocked once this is read.
- Before ML-16 trains, look at the 25 dated test photos (with the 50 new photos and P6's 6 fallbacks): the two Pixel 9
  Pro Ethiopia photos wrong on every patch are worth a look for a label or a mask problem.
