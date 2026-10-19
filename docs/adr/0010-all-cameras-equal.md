---
status: superseded by ADR-0024
---

# All cameras are equal sources of photos

No camera or capture is held out, trained on selectively, reported separately or used as a label; photos are pooled across captures and split 70/15/15 within each class. The leave-one-camera-out folds and the cross-camera metric are gone, at the accepted cost that in-distribution val/test sit near their ceiling.

Source: [ML-1](../ticket_retire_cross_rig.html), owner decision 2026-09-28 and D1.

Superseded by [ADR 0024](0024-no-camera-or-rig-in-the-pipeline.md) (ML-8 D2), 2026-10-19: no camera or rig anywhere in the data or training pipeline, one pool.
