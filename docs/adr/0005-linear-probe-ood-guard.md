# The OOD guard is a linear probe, calibrated on independent-day photos

The out-of-distribution guard is a logistic regression on the classifier's frozen embeddings (bean patches against non-bean patches), chosen over seven alternatives, the earlier centroid distance included. Its threshold is calibrated only on the owner's own photos from days not used in training, because conformal calibration needs photos exchangeable with what the service meets; internet or same-day photos would make it look tighter than it is. The probe is a file beside its checkpoint (`<checkpoint>.ood_probe.json`) that must match it, so every model ships with its own probe, and a holdout used once to confirm a probe is spent.

Origin: owner decision, 2026-09-11; `docs/ood_guard_eval.md`.
