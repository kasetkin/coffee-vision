# OOD guard: what it actually does, measured

**Status: `linear_probe` confirmed on the holdout and prepared for deployment (2026-09-11).**
The holdout split has now been spent, once, against a probe frozen beforehand -- see
"Holdout confirmation" below. `OOD_THRESHOLD` and `ood_scores` are still untouched: the
centroid metric remains exactly what runs when no probe file sits beside a checkpoint.

This document records what the guard was measured to do once it was put in front of data it
had never been tested against, because the answer changed twice under measurement and the
reasoning is worth keeping.

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

## Holdout confirmation (2026-09-11) — spent once

The comparison above is cross-validated within dev, which is how to *choose* a method and not
how to *ship* one: it yields five probes and no coefficients. `coffeecv/fit_ood_probe.py` fits
one probe on dev alone, calibrates a threshold on dev positives, writes
`<checkpoint>.ood_probe.json`, and only then scores the holdout — frozen first, measured
second, never adjusted after.

| group | n | probe refuses | centroid refuses |
|---|---|---|---|
| **user positives** (the deployment distribution) | 5 | **0/5** | 0/5 |
| internet positives | 5 | 2/5 | 2/5 |
| **user real-world negatives** | 3 | **3/3** | 2/3 |
| internet negatives | 37 | **37/37** | 24/37 |

```
AUROC on holdout: probe 0.9700   centroid 0.8375
user holdout positives, probe: 0.020 0.032 0.040 0.046 0.078  (threshold 0.4952)
lowest holdout negative:       0.5051
```

The probe is **better or equal on every group**: it refuses no genuine photo the centroid
metric accepted, and it catches 40/40 negatives against 26/40. That is the whole case for
deploying it, and it is the only holdout reading this data can support — re-running `--verify`
and then changing anything would turn the holdout into a second dev set.

### An unplanned independent check: 60 photos of a bean type the model has never seen

On the same day the probe was frozen, 60 photos of a new class (`011 Peru,Minka`, three rigs)
were added to the dataset for a future training run. The deployed 10-class checkpoint has never
seen this origin, and these photos existed in neither dev nor holdout, so scoring them against
the already-frozen threshold is free evidence on exactly the population that matters — the
user's own rigs, an unfamiliar day, real bean trays.

```
probe    refuses 0/60   median 0.0110   p95 0.0759   max 0.1649   (threshold 0.4952)
centroid refuses 1/60   median 1.1981   p95 1.3216   max 1.4051   (threshold 1.4)
```

Zero false refusals across all three rigs, with the worst photo at a third of the threshold.
The centroid metric refuses one of them — a genuine tray of beans, at 1.4051 against its 1.4
threshold — which is the live false-refusal behaviour of Finding 2 reproducing itself on new
data, unprompted.

This was **not** used to recalibrate: the shipped threshold is the one the holdout verified, and
re-fitting against these would have invalidated that verification. They are recorded as
observation. They are, however, the obvious calibration set for the *next* deployment — 60
independent-day photos would move the certificate from α ≤ 20% to roughly α ≤ 1.6%.

## The same-rig batch (2026-09-11) — and what it exposed

Everything above was measured against negatives that were ~92% internet-sourced. On the
afternoon of 2026-09-11 the user shot **41 negatives and 30 positives on the three real rigs**:
buckwheat, mung beans, split peas, lentils and chickpeas, on the same trays, backgrounds and
cameras as the bean photos, so the *only* thing varying between a positive and a negative is
what is in the bowl. This is the confound-free test the internet batch could never be.

**The deployed v1 probe scored worse than its holdout promised, and the gap is the confound:**

```
                     v1 probe    centroid
positives refused      2/30         6/30
negatives missed      11/41        23/41
AUROC                 0.849        0.712     (v1 holdout AUROC had been 0.970)
```

Still a large improvement over centroid on both axes — the deploy was right — but 0.849 is the
honest number. Part of that 0.970 was the probe detecting *internet photo*, not *not-coffee*.

### The failure mode was exactly one thing

All **11 missed negatives were mung beans**, and the set of green-hued photos and the set of
missed photos were *identical* — no exceptions in either direction. The cause is structural:
**green unroasted coffee is a legitimate training positive**, so the probe had learned
green + matte + bean-shaped ⇒ coffee and had never seen a green legume that was not coffee.

The centroid metric catches all 11 (1.71–1.99). The two metrics fail on disjoint sets, which is
tempting — but refusing when *either* fires reaches 41/41 negatives at the cost of 6/30 false
refusals, worse than v1 on the axis that matters most. Rejected.

### v2: refitted with the same-rig negatives

`green_legume` was added as its own tag (in `CLEAN_NEGATIVE_TAGS`, so every patch is a usable
negative label) rather than folded into `confusable_grain`, specifically so a later run can
report whether *this* hole stayed closed. Fitted on dev only; scored once on the fresh holdout,
which is in neither probe's fit set:

| on the 2026-09-11 holdout (12 positives, 16 negatives) | false refusals | negatives caught | green legume | AUROC |
|---|---|---|---|---|
| **v2 refitted** | **0/12** | **16/16** | **4/4** | **1.000** |
| v1 deployed | 1/12 | 12/16 | 0/4 | 0.875 |
| centroid | 2/12 | 8/16 | 4/4 | 0.755 |

v2 is better on every axis, and the certificate improves from α ≤ 20% to **α ≤ 4.3%** because
independent-day calibration photos went from 4 to 22. **Deployed 2026-09-11.**

Two things to keep in view rather than forget:

- **Teaching it green legumes made genuine beans harder.** Calibration positives' median rose
  0.135 → 0.334. The boundary moved toward the positives; that is the price of the fix.
- **The threshold is pinned by one photo** (`PIC_20260911_131316.JPG`, 0.9681, a 0.277 gap to
  the next positive) — a perfectly ordinary dense tray of roasted beans on white, 90% of whose
  patches the probe calls not-beans. Framing does **not** explain it: across the 30 new
  positives `corr(beans_across, score)` is only +0.195. It is simply the hardest positive, and
  a distribution-free threshold is the right tool for exactly that. The cost is a thin margin —
  0.9681 against a lowest holdout negative of 0.9817.

### Why the threshold is calibrated on four photos

Conformal validity needs the calibration photos to be exchangeable with what the service
actually meets, and that ruled out the larger, more tempting calibration sets:

| calibrate on | n | threshold | certificate |
|---|---|---|---|
| all dev positives | 16 | 0.9916 | α ≤ 5.9% |
| user photos | 9 | 0.4952 | α ≤ 10% |
| **user, independent-day only** | **4** | **0.4952** | **α ≤ 20%** |

All 16 positives put the threshold at 0.9916 — set entirely by internet photos, against
negatives whose median is 0.9995. That is a certificate over a population the service will
never see, with almost no margin left. The user's same-day photos are easier than
independent-day ones (0.001–0.144 against 0.072–0.495), so including them flatters the
certificate the same way Finding 1's held-out split did.

The narrowest set gives the **same threshold** with an honest certificate, so the only thing
given up is the size of the promise, not the behaviour. And 20% is a *bound limited by n=4*,
not an estimate: the observed false-refusal rate on the user's photos is **0/9** across dev and
holdout combined.

**This is the collection target, and it is sharper than "more positives":** the binding
resource is *independent-day* photos of the user's own trays. 19 of them would certify 5%; 99
would certify 1%. Same-day photos and internet photos do not move this number.

### What was deployed

- `<checkpoint>.ood_probe.json` (~41 KB) beside the weights. **Its presence is the deployment
  switch** — `load_ood_probe` picks it up, and deleting it reverts to the centroid metric on
  the next restart, with no code change. It carries the checkpoint SHA and is refused against
  other weights, exactly as the reference is.
- No embeddings sidecar is needed: the probe is a dot product over patch embeddings the
  forward pass already computes. `models/*.ood_embeddings.npz` is required only by `knn` and
  `mahalanobis_shared`, neither of which ships.
- `classify_one` records **both** scores on every photo and logs both (`ood_probe`,
  `ood_median`, `ood_method`), so the production series stays continuous across the switch and
  the next recalibration has traffic to work from.
- `--ood-method centroid` reproduces any pre-probe verdict.

`mahalanobis_shared` remains the fallback if the probe ever needs to be withdrawn: a drop-in on
the same embeddings needing no negatives at fit time, 0.064 AUROC behind on dev.

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
