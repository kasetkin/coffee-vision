# The OOD guard is the linear probe at a fixed 0.5, after every classification

The guard stays [ADR 0005](0005-linear-probe-ood-guard.md)'s linear probe: a logistic regression on the classifier's frozen patch embeddings, one file beside its checkpoint (`<checkpoint>.ood_probe.json`) that must match it. What changes is where it runs, its threshold, and what it is fitted and measured on.

- **After every classification, at a fixed 0.5** (ML-5 D9): on the crop, or on the whole frame when the segmenter fails (an empty or tiny mask) or the user picks the full photo. It refuses when the mean P(not beans) over the patches exceeds 0.5, its own decision boundary. No threshold is calibrated: the conformal threshold of ADR 0005 was overridden to 0.5 in every release (0.5957 for the shipped model), and the photos it was calibrated on now serve the segmenter instead (D1, D2). Chosen over running the probe only when the segmenter fails, and over the segmenter's empty mask alone; the segmenter's empty-mask rate on test negatives is reported beside the probe, not used as a guard.
- **Fitted on the segmenter dataset's training and validation negatives, measured on its test split** (D16): negatives caught and positives refused, the positives apart by whether the classifier trained on the photo. The probe's bean photos stay the classifier's own held-out photos, minus any photo in the segmenter test split. Nothing is chosen from the test numbers, so measuring there again spends nothing; fitting on a test photo is what is forbidden.

## Consequences

- The guard no longer carries a false-refusal promise (ADR 0005's certified alpha). Its refusal rate on genuine photos is a measured number on the test split.
- The serving warn band came from the conformal fit (the 20% threshold). A probe fitted under this ADR has none, so the band ends with the first such release.

Source: [ML-5](../ticket_segmenter_dataset.html) D2, D9, D16, D17; supersedes [ADR 0005](0005-linear-probe-ood-guard.md).
