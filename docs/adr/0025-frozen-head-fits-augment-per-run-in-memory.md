---
status: accepted
---

# A frozen head fit augments its training patches per run, in memory, and stores nothing augmented

The frozen DINOv3 head (ADR 0006) is fitted on every training patch and `views` augmented views of it (params.yaml `head_augment`, 8 since ML-8), drawn in each run from the patches already held in memory, seeded by (run seed, patch, view), embedded in batches and dropped: only the embeddings are kept, and nothing augmented is written. Every pipeline is built by `coffeecv/augment.py`'s strict builder from albumentations 2.0.8, pinned as in [ADR 0018](0018-segmenter-fine-tunes-augment-on-the-fly.md), which fails on an argument the version does not know. Validation, test, the OOD reference and the probe's photos are never augmented.

- **Why per run, not a cache**: the head is one convex fit on fixed embeddings, so a run needs each view once; drawing them per run lets the transform set change without rebuilding anything and keeps disk free (about 330 MB of embeddings per run, in RAM). As in ADR 0018, augmented data is made where it is used and never stored.
- **Why two pipelines**: geometry, occlusion and colour act on the stored 448 px patch; the camera pipeline (compression, blur, noise, downscale) acts after the resize to the model's 224 px input, where its strengths are in the pixels the model sees.
- **Equal total weight**: each of a patch's 1 + views rows weighs 1/(1 + views), so C means what it meant unaugmented. Each run also fits the unaugmented head beside it at the same C and weight, archived with the run: ADR 0002's paired print, without a second run.
- **Cost**: each view is a backbone pass, about 2.3 h of embedding per seed at 8 views on the CPU host. Adopted by the owner without waiting on a screen (ML-8 D4); the split rewards memorisation (by single photo, near-duplicates across splits), so the paired print informs, it does not gate.

Source: [ML-8](../ticket_data_refresh_retrain.html) D4, D12.

Amended 2026-10-20 by [ADR 0027](0027-both-fits-share-augmentation-groups.md): the colour, compression and noise transforms are `params.yaml` `augment`'s shared groups, taken by reference; expanded, the pipeline is the one exp270-272 were fitted with.

Amended 2026-10-27 by [ML-13](../ticket_segmenter_xl0v6.html) D8: so a fit can resume after a host reboot, the embeddings it computes are written to disk as it goes (each split's when it is embedded, the training views' in chunks of whole batches, about 330 MB per seed under `outputs/frozen_head/features/`) and a rerun of the same command loads those whose fingerprint matches and embeds the rest. Views are still made from the patches in memory and dropped; no augmented image is ever written. The embeddings go once the seed's run directory is written, and a run deletes any other run's before it embeds anything, so nothing stays on disk from one experiment into the next.
