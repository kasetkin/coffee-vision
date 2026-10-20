---
status: accepted
---

# The segmenter's fine-tune and the classifier's head fit share their augmentation groups

`params.yaml` `augment` holds groups of transforms that both fits take: the segmenter's fine-tune (`seg_ft.augment`, [ADR 0018](0018-segmenter-fine-tunes-augment-on-the-fly.md)) and the classifier's head fit (`head_augment`, [ADR 0025](0025-frozen-head-fits-augment-per-run-in-memory.md)). A pipeline names a group as `{use: <group>}`; the config loader expands it in place (`coffeecv/augment.py` `expand`), so a run's config and a model's card record the transforms that ran, never a name. The groups are `colour` (ColorJitter at 0.2 with hue 0, PlanckianJitter), `compression` (JPEG quality 60-95) and `noise` (Gaussian, 0.01-0.04 of the range): each is the classifier's pipeline of exp270-272, so its recipe is unchanged.

- **What is shared**: only what means the same at the segmenter's 1024 px photo and the classifier's 224 px patch. Geometry is not (the segmenter's moves its label with the photo), nor is blur, whose radius is in pixels.
- **Stronger for the segmenter**: a fit adds its own transforms on top of the groups and never weakens one. The segmenter keeps its hue shift, saturation and value, and gamma; colour carries origin and roast, so the classifier's hue stays 0.
- **What changes**: the segmenter's brightness and contrast (RandomBrightnessContrast 0.2, 0.2 at p 0.5, as XL0-v3 trained) give way to the colour group, which adds saturation and white balance at p 0.8. That takes effect at its next fine-tune; XL0-v3's stage is frozen with the pipeline it trained with.
- **DVC**: a stage that trains with a pipeline lists `augment` in its params beside its own block, or a changed group does not invalidate it.

Source: the owner, 2026-10-20, after [ML-8](../ticket_data_refresh_retrain.html); amends ADR 0018 and ADR 0025.
