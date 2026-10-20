# Segmenter fine-tunes augment each sample as they train, through albumentations, and store nothing augmented

A segmenter fine-tune ([ADR 0015](0015-xl0-segmenter-owner-judged-labels.md)) augments each training sample when it is drawn, through the `albumentations` library, and writes no augmented photo, label or embedding anywhere. The pipeline is `params.yaml` `seg_ft.augment`, a list of transforms built by name and seeded from the run's seed, so one seed repeats its stream. A spatial transform moves the label with the photo, and a warp fills the label's exposed border with the ignore value. Validation photos are not augmented, so checkpoint selection stays deterministic. The model card records the pipeline and the library's version.

- **Why not a cache of views**: ML-5's fine-tune precomputed 4 dihedral views of each photo through the frozen encoder into an embedding cache (3.5–3.6 GB a session), so each photo was only ever seen in those 4 forms. On the fly, each epoch sees new augmentations, nothing needs disk (a pass-3 cache would have taken about 7.6 GB with 11 GB free), and the transform set can change without rebuilding anything.
- **The cost**: the frozen XL0 encoder now runs on every sample in every epoch, about 1.7 s a photo at 8 threads on the CPU host, so a 40-epoch fine-tune takes about 7–10 h rather than about 1.5 h of caching and 50 min of training.
- **Which library**: `albumentations` 2.0.8, MIT, pinned. Upstream archived it on 2025-07-10. Its successor AlbumentationsX is AGPL-3.0 or commercial and sends telemetry by default, so it is not used. 2.0.8 checks PyPI for a newer version on import unless `NO_ALBUMENTATIONS_UPDATE=1`, which the fine-tune sets before importing it.

Source: [ML-6](../ticket_segmenter_retrain.html) D12, D16, D19.

Amended 2026-10-20 by [ADR 0027](0027-both-fits-share-augmentation-groups.md): the pipeline takes the shared groups of `params.yaml` `augment` (colour, compression, noise) and adds its own transforms on top; XL0-v3 trained with the pipeline before it, recorded on its card.
