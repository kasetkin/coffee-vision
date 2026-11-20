# Distilling DINOv3 into a smaller student

Training a student backbone (MobileNetV4 or any other) to reproduce the frozen DINOv3 ViT-B/16's embedding, to speed up classification. Closed by the owner on 2026-11-20 ([ML-32](../docs/ml-0032_ticket_distil_dinov3_mobilenetv4.html), D9): "Distillation is impossible on my hardware and there is no pre-trained student. So we will use different approaches to speed up things, not a distillation."

## Why this is out of scope

- **No hardware for it.** This machine has no GPU and torch is the CPU build. Measured 2026-11-16 (ML-32 §2): the B/16 teacher takes 0.225 s a view and a Conv-L training step 0.167 s, so one student at the schedule the sources recommend is about 25 days, and the plan held about 11 of them. Even the reduced plan (a stored teacher, 30 passes, no lever screening) is about two weeks of the machine doing nothing else, and it gives up the quality the plan was meant to protect.
- **No ready-made student.** No MobileNetV4 distilled from DINOv3 is published; timm's MobileNetV4 weights are ImageNet-trained (searched 2026-11-16).

The speed-up is sought by other approaches. This entry does not close swapping in one of Meta's published DINOv3 backbones (ViT-S/16, ConvNeXt-Tiny) with a refitted head and probe: that needs no distillation (ML-32 D8, not decided). Distillation is worth raising again only with training hardware the owner names, or with a published student of DINOv3 in the wanted architecture.

## Prior requests

- [ML-32](../docs/ml-0032_ticket_distil_dinov3_mobilenetv4.html) and [its plan](../docs/ml-0032_plan_distil_dinov3_mobilenetv4.html), 2026-11-16 to 2026-11-20.
- [`analysis/distillation/research_distillation.md`](../analysis/distillation/research_distillation.md), the research note of 2026-11-15.
- `docs/dinov3_integration_plan.md` §13: DINOv3 as a teacher for ResNet18, never run.
