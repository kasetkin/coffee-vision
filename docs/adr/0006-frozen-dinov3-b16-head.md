# The classifier is a frozen DINOv3 ViT-B/16 with a logistic-regression head

The backbone is DINOv3 ViT-B/16, frozen (no fine-tuning), read out as `cls_mean` (1536-d), at 224 px only and without test-time augmentation (B/16 costs about 145 ms per image on the VM); the head's C is fixed at 0.1. The gain over the fine-tuned ResNet18 came from DINO's self-supervised pretraining, not from architecture or recipe. ViT-S/16 is only a fallback.

Sources: owner decisions of 2026-09-24 and 2026-09-26, `docs/dinov3_integration_plan.md` §0.4; C: [ML-1 D3](../ticket_retire_cross_rig.html).
