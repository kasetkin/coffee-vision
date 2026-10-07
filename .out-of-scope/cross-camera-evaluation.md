# Cross-camera evaluation

This project does not evaluate by holding cameras out. There are no leave-one-camera-out folds, no cross-camera test split, and no cross-camera metric that a change has to win on.

## Why this is out of scope

All cameras are equal sources of photos ([ADR 0010](../docs/adr/0010-all-cameras-equal.md)): photos are pooled across captures and split within each class, and changes are judged on the in-distribution val/test metrics, paired across seeds ([ADR 0002](../docs/adr/0002-paired-multi-seed-evidence.md)). The fold machinery (`run_folds`, `xrig_eval`, the fold summaries) was deleted with ticket ML-1. The owner accepted the cost: the in-distribution metrics sit near their ceiling and move little.

Two related measurements were deferred by the owner, not rejected; propose them only when asked:

- A photo-level test metric (40 patches per photo, the `/classify` path). `run_photowise` can be restored from `0fa699c^` (ML-1 D2).
- A measurement of seed or bootstrap noise ("later, another way").

## Prior requests

- [ML-1](../docs/ticket_retire_cross_rig.html), owner decision 2026-09-28, and its D1, D2 and D7: the protocol and the screens built on it were retired.
