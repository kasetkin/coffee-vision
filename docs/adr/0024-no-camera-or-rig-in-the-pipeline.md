---
status: accepted
---

# No camera or rig in the data or training pipeline

Training reads one pool, `data/segcropped/pool`, merged by one stage (`merge_segpool`) from every session's crops. No training path, stage, manifest or report reads a camera, a phone model or a rig: the patch budget is per class over the pool (600/160/160), a patch box's seed is (seed, class, photo, split), the manifests have no camera column, and the segmenter dataset reads a file-name time, for a photo without an EXIF time, on one local clock for every photo. The session folders under `dataset/` keep their names, which are history, not pools.

- **Why**: [ADR 0010](0010-all-cameras-equal.md) already pooled the photos for the split, but the camera still shaped training: four camera pools, a budget per pool (a class missing from one pool got three quarters of the others' patches), a pool's position in the patch seed, and a question each time a session mixed phones (6 of the 20 photos of 2026-10-17/18 were not the Pixel their session merged into). Dropping the camera removes those questions instead of answering them.
- **Cost**: the camera column told the grouping that a Pixel names its files in UTC; on one local clock a Pixel photo without an EXIF time no longer chains with a burst mate that has one, and two phones' photos at one local minute may chain. On 2026-10-19 this moved no group: 21 of 2,046 photos took the file-name path.
- **Not changed**: no camera is held out or reported apart (ADR 0010's rule stands); historical records are not rewritten to drop their cameras: one the current pipeline still reads keeps them as written, and one it no longer needs (ML-2's lists, retired `analysis/` scripts) leaves the tree and stays in git history.

Source: [ML-8](../ticket_data_refresh_retrain.html) D2, D8; supersedes [ADR 0010](0010-all-cameras-equal.md). Amended 2026-10-21: records the pipeline no longer needs leave the tree (the owner).
