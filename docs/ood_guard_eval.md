# OOD guard: what it actually does, measured

**Status: findings, not a decision.** Nothing here has been shipped. `OOD_THRESHOLD` is
untouched, `classify_one`'s default metric is untouched, and the holdout split has not been
spent. This document records what the guard was measured to do once it was put in front of
data it had never been tested against, because the answer changed twice under measurement and
the reasoning is worth keeping.

Harness: `coffeecv/ood_eval.py`. Data: `dataset/ood_negatives/`, `dataset/ood_positives/`,
`dataset/ood_positives_internet/` — photo directories are DVC-tracked, while each one's
`<name>.manifest.csv` and README stay in git beside it, the same split the `<session>.crop.yaml`
files already use. Splits, provenance and per-photo baseline verdicts live in those manifests.

Counts below are on the de-duplicated negative set (93). The two narrative findings were
measured just before a byte-identical duplicate was found and removed, so they quote 60 dev
negatives rather than 59; the difference moves AUROC in the third decimal and nothing else.

---

## Why this was opened

The shipped guard refuses when a photo's penultimate-embedding distance to the nearest class
centroid, normalised by that class's spread and pooled to a per-photo median, exceeds
`OOD_THRESHOLD = 1.4`. Two things about that were weaker than the rest of this repo's
standards:

1. **The threshold came from one photo.** Phase 9 measured 1.24 as the in-distribution maximum
   and 1.92 on a single out-of-rig photo, and 1.4 was placed between them. Everywhere else this
   project refuses to adopt on a single example.
2. **It had never been shown a photo without beans in it.** Every negative it had ever been
   evaluated against was still a tray of beans, just from an unfamiliar camera. The one thing
   the guard exists to do was the one thing never measured.

## What was built to answer it

- **Negatives** (`dataset/ood_negatives/`): 93 vetted Wikimedia photos across six scenario
  tags (empty surfaces, ground coffee, confusable grains, nuts/seeds, non-food objects, dried
  coffee cherries), plus **8 real captures from the user's own phone** — outdoor/travel photos
  of rock, scree and wildlife. Those eight are the most valuable rows in the whole set: three
  of them were already being **accepted and assigned a bean origin** by the shipped guard.
- **Positives** (`dataset/ood_positives/`, `dataset/ood_positives_internet/`): genuine bean
  photos the model has never seen — 14 of the user's own (hash-checked against all 938 training
  photos, byte-wise and by capture timestamp) and 12 from Wikimedia. These turned out to be
  the load-bearing part of the whole exercise.

## Finding 1 — against the checkpoint's own held-out split, the metric looks excellent

| population | n | median | max |
|---|---|---|---|
| held-out test photos (familiar sessions) | 144 | 1.022 | 1.384 |
| negatives (dev) | 60 | 1.460 | 2.720 |

AUROC **0.998**. And the production threshold sits *above the entire in-distribution range*
(1.4 > 1.384), so it never falsely refuses — while missing about a third of the negatives.
On this evidence the obvious move is to lower the threshold to ~1.25, which would catch 98% of
negatives at the cost of falsely refusing 1 photo in 144.

**That conclusion was wrong**, and the next section is why.

## Finding 2 — the held-out split is not a stand-in for photos the model has not seen

The checkpoint's own held-out photos come from sessions it trained on: same beans, same room,
same lighting, same day. Genuine bean photos from *other* days score far higher:

| population | n | min | median | max |
|---|---|---|---|---|
| held-out test (familiar sessions) | 144 | 0.844 | 1.022 | 1.384 |
| unseen genuine bean photos | 14 | 1.034 | **1.214** | **1.572** |
| — of those, independent-day only | 7 | 1.200 | 1.266 | 1.572 |
| negatives (dev) | 60 | 1.242 | 1.460 | 2.720 |

```
AUROC vs familiar held-out photos : 0.998
AUROC vs genuine unseen photos    : 0.896
AUROC vs independent-day only     : 0.860
```

Two consequences:

1. **Production already misfires.** Two of the 14 genuine bean photos (1.572, 1.437) are
   refused *today* at 1.4. That is a live false-refusal rate on new photos, not a hypothetical.
2. **No threshold is good.** 1.25 would falsely refuse 36% of genuine unseen photos; 1.40
   refuses 14% of them and still misses 32% of negatives; 1.60 stops the false refusals and
   catches only 22% of negatives.

So the metric, not the threshold, is the binding constraint — the opposite of Finding 1's
reading, and only visible once genuinely unseen positives existed.

A tempting explanation is framing: the high-scoring genuine photos are extreme close-ups, and
score correlates with `beans_across` at r = −0.46. It does **not** hold up as a mechanism —
photos at an identical `beans_across` of 8.22 span scores from 1.03 to 1.57. Framing
contributes; it does not explain.

## Method comparison

Eight metrics, dev split, scored through the identical `classify_one` patch path. The
benchmark is separation against *unseen* positives — deliberately not the 0.998 the easy split
reports.

**Source confounding matters here and is easy to miss.** The negatives are ~92%
internet-sourced while the positives are mixed, so any metric that can detect "internet photo"
collects unearned credit. So AUROC is reported three ways: pooled, and within each source
separately.

| method | pooled | user-matched | internet-matched | mean of matched |
|---|---|---|---|---|
| **linear_probe** | 0.980 | 1.000 | 0.950 | **0.975** |
| ensemble | 0.925 | 0.911 | 0.944 | 0.928 |
| mahalanobis_shared | 0.913 | 0.911 | 0.910 | 0.911 |
| energy | 0.805 | 1.000 | 0.759 | 0.880 |
| knn | 0.874 | 0.822 | 0.910 | 0.866 |
| coverage | 0.855 | 0.800 | 0.894 | 0.847 |
| **centroid (shipped)** | 0.863 | **0.778** | 0.884 | **0.831** |
| patch_vote_agreement | 0.729 | 0.922 | 0.704 | 0.813 |

The shipped metric is the **worst** performer on the source-matched comparison that most
resembles deployment (the user's own phone on both sides). `energy` and
`patch_vote_agreement` are the clearest examples of why the three columns are needed: both
look excellent on one source and poor on the other, which is what small-sample overfitting
looks like.

At an operating point that refuses 5% of unseen genuine photos:

| method | genuine photos wrongly refused | real-world negatives caught | all negatives caught |
|---|---|---|---|
| **linear_probe** | **0/9** | **5/5** | **55/59** |
| ensemble | 1/9 | 4/5 | 47/59 |
| mahalanobis_shared | 1/9 | 2/5 | 34/59 |
| centroid (shipped) | 1/9 | 1/5 | 29/59 |

### The probe's advantage survives a change of negative source

The obvious objection to `linear_probe` is that it is fitted, so it may simply be memorising
this negative set. Tested directly by leave-one-source-out — fit on the internet negatives
only, then asked about the user's real-world captures, a negative source it has never seen:

```
pos_user      median 0.076   (genuine beans, correctly near 0)
neg_user      median 0.9999  (real-world negatives, correctly near 1)
AUROC on the unseen negative source: 1.000
at 5% false-refusal: 5/5 user negatives caught, 0/9 genuine photos refused
```

The reverse direction (fit on just five user negatives, generalise to internet negatives)
gives 0.875 — weaker, as expected from five training photos, but still working. So the probe
is learning something about beans-versus-not that transfers across sources, not the sources
themselves.

Caveat worth keeping: internet positives are the hardest positives for it (median 0.56 under
the internet-negatives-only fit, against 0.08 for the user's own), so the margin is thinner
than the headline AUROC suggests.

## Setting a threshold responsibly

Split-conformal turns a score into a calibrated guarantee, and the sample size sets a hard
floor on what can be claimed: with *n* calibration photos the tightest certifiable
false-refusal rate is 1/(n+1).

```
n= 16 (what we have today) -> tightest certifiable alpha = 5.9%
n= 19                      -> 5.0%
n= 99                      -> 1.0%
```

At a certified ≤10% false-refusal rate, `linear_probe` catches **44/59** negatives against
`centroid`'s **16/59**.

This is also a concrete collection target: **19 unseen genuine photos to certify 5%, 99 to
certify 1%.** More positives are worth more here than more negatives.

## If a change is ever made

Not now — the holdout split is deliberately unspent, and that is the instrument for confirming
whichever metric is chosen. Shipping `linear_probe` would additionally need the fitted
coefficients stored beside the checkpoint (as the reference and embeddings sidecars already
are), and `classify_one` given a non-default `ood_method`. Shipping `mahalanobis_shared`
instead would need none of that — it is a drop-in on the same embeddings with no negatives
required at fit time, which is why it is worth keeping in view despite being 0.07 AUROC behind.

## Things worth not forgetting

- **Green/unroasted beans are positives, not negatives.** They were briefly collected as a
  `wrong_bean_type` negative and removed: the training set contains such photos as legitimate
  examples, so scoring them as negatives would have manufactured false failures.
- **Same-day is not independent.** Seven of the user's positives were shot 35 minutes to 5.5
  hours before a training session on the same day. Byte-distinct, hash-clean, and still not
  independent — they score a median 1.12 against the independent-day 1.27.
- **The harness must check the checkpoint/reference pairing.** A stale reference in `outputs/`
  was paired with a different checkpoint; distances computed against it would have looked
  entirely plausible and meant nothing. `infer.py` already refused this; `ood_eval.py` now does
  too.
- **`build_ood_reference` could not run on this machine** until it was chunked: it stacked
  every training patch into one tensor (~3.4 GB) before the forward pass. The forward pass was
  never the problem.
