# A learned segmentation stage finds the bean region, the same in training and serving

EfficientViT-SAM-L0, fine-tuned on this project's photos, masks the photo's bean region, and the photo is cropped to it before classification; it replaced the per-session tray heuristic at both ends. Its training masks have no hand annotation: Claude (Opus, through headless `claude -p` on the owner's plan) judges each proposed mask and gives corrective points, and the owner audits the verdicts.

Source: [ML-2](../ticket_segmentation_mask.html) D1, D6, D15.
