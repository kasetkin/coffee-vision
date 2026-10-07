# ResNet18 recipe levers

The fine-tuned ResNet18 is the legacy pipeline: production has run a frozen DINOv3 head since 2026-09-27 ([ADR 0006](../docs/adr/0006-frozen-dinov3-b16-head.md)), and the last ResNet release left `models/` on 2026-10-06. Its recipe screens are closed, and none of these is re-run on it:

- Brightness augmentation: a null result (`EXPERIMENTS_LOG.md`, Phase 13).
- Mixup, a frozen backbone, last-block-only fine-tuning: none adopted; full freezing cost 0.097 macro-F1 (`EXPERIMENTS_LOG.md`, exp106-114).
- AdaBN: not deployable, since serving sees one photo at a time.
- Photometric and multi-scale test-time augmentation: declined. Plain TTA was adopted.
- LR scheduler changes (ReduceLROnPlateau and the plan's later stages): closed by the owner on 2026-09-24, a null result because the scheduler only moves the head's 0.05% of parameters (`docs/lr_scheduler_plan.md`).

## Why this is out of scope

These took weeks of sweeps, and most were scored on the cross-camera metric that no longer exists ([cross-camera-evaluation.md](cross-camera-evaluation.md)). What was adopted (MixStyle 0.5, plain TTA, 100 epochs with patience 20, `eta_min` 1e-5) is in `params.yaml` and `coffeecv/run_all_rigs.py`'s `ADOPTED`. A lever is worth trying again only on the DINOv3 pipeline, as a paired multi-seed screen ([ADR 0002](../docs/adr/0002-paired-multi-seed-evidence.md)).

## Prior requests

- `EXPERIMENTS_LOG.md` Phases 13 and 14 and exp106-114; `docs/lr_scheduler_plan.md`.
