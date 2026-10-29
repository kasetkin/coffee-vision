# Status

What is live, what its numbers mean, and where each run is written up. For what the project is, see the top-level
[README.md](../README.md).

## The live model

`dino3b16_segxl0v6_country_aug8_s42` (exp279, ticket [ML-13](ticket_segmenter_xl0v6.html)), live since 2026-10-29:

- **Pipeline:** the XL0-v6 segmenter (`xl0_v6_s42`) marks the bean region and crops to it (a mask under 2.85% of the photo counts as none); patches of the crop go through a frozen DINOv3 ViT-B/16 (`cls_mean` readout); a logistic-regression head at C = 0.1 (fixed, ML-1 D3, ADR 0006) scores the 10 countries; patch scores are pooled per photo. No TTA. The OOD probe refuses photos that do not look like the training data (threshold 0.5, ADR 0016).
- **Training data:** one pool, `data/segcropped/pool`, every session's crops cut by the segmenter (the `merge_segpool` stage; no camera or rig, ADR 0024), 1,376 photos, split 70/15/15 within each class, 600/160/160 patches per class. The head is fitted on each training patch and 8 augmented views of it (ADR 0025). The label is the country (ADR 0012).
- **Scores:** in-distribution val 0.9680, test 0.9775 patch macro-F1, one seed (42). The model it replaced, on the same photos and seed but XL0-v5's crops (exp276): val 0.9718, test 0.9813, inside that model's own seed spread (exp276-278: val 0.9614-0.9718, test 0.9694-0.9813). On exp279's test patches (XL0-v6's crops, none of the 209 photos trained on by either), both score 0.9775 patch macro-F1 and get every photo right (`analysis/xl0v6/live_baseline.txt`). What changed is the segmenter: the owner preferred its test masks blind, 122 of 135 accepted against XL0-v5's 115 (`analysis/xl0v6/paired_s42_report.txt`).
- **OOD guard:** caught 160/160 test negatives and refused 0/135 bean photos, 71 of them never trained on (`analysis/xl0v6/ood_eval.txt`).
- **Serving:** the model's LiteRT conversion on the web host's GPU, without torch (ticket ML-7, ADR 0019, ADR 0020); about 8.5 s server-side per photo. [inference_path.html](inference_path.html) traces a photo through it step by step; [webapp/README.md](../webapp/README.md) covers running and deploying it.

`models/` also holds the four models before it, `dino3b16_segxl0v5_country_aug8_s42` (exp276, live 2026-10-27 to 2026-10-29), `dino3b16_segxl0v4_country_aug8_s42` (exp273, ticket ML-9, live 2026-10-24 to 2026-10-27), `dino3b16_segxl0v3_country_aug8_s123` (exp271, ticket ML-8, live 2026-10-20 to 2026-10-24) and `allrigs_dino3b16_segxl0v3_country_s42` (exp267, ticket ML-6, live 2026-10-13 to 2026-10-20), not retired.

## Reading the numbers

Quote ranges over seeds, not a single run: seed-to-seed spread is wider than most of the effects being measured, which is why adoptions require a *paired* multi-seed check (ADR 0002). Val patch macro-F1 selects adoptions; test is reported.

**Every val and test score this pipeline now produces is in-distribution, not an estimate of real-world accuracy.** Val and test photos come from the same bean bags and the same cameras as the training photos, so they say nothing about how a model does on a new camera or a new scoop of beans (ML-1, R2). They also sit near ceiling (~0.97) and barely separate choices (R1). Never compare one with a fold-era cross-camera number: 0.887 and 0.975 for the same shipped recipe answer different questions (R3), and `experiments/index.csv` keeps the two kinds of column apart (`xrig_*` is blank for runs since ML-1, meaning "not measured"). Until a generalization test set exists (fresh scoops, a new phone -- its own ticket), camera- or scoop-level drift shows up only in manual spot-checks and the OOD guard, and neither gates a change.

Roast state is uneven across countries (pool photos, `data/segcropped/pool/merge_manifest.json`): Brazil (class 013, 10 photos), Colombia (class 014, 19) and Guatemala (26) have roasted photos beside their green ones, Ethiopia and Vietnam one roasted photo each; Kenya and Costa Rica are green only; Indonesia, Peru and Rwanda are roasted only. For the one-state countries roast state alone narrows the answer, and no number above says how a model does on them in the other state.

Those counts are ML-8's pool. ML-9's added the 2026-10-21 photos, green Indonesia (class 015, a coffee of its own) and green Peru (class 011) among them, and 33 more Brazil photos, less the owner's exclusions ([ML-9](ticket_green_origins_retrain.html)); it was the first model to see Indonesia and Peru green. XL0-v5's added 32 more Brazil photos (2026-10-24 to 2026-10-26) and 18 the owner readmitted from the exclusions on XL0-v4's masks (2026-10-24). The live model's is the same 1,376 photos, re-cropped by XL0-v6; `merge_manifest.json` has the counts per class and session.

## How it got here

- **Fine-tuned ResNet18** (the first model): bean-unit patch sizing at 4-7 beans, MixStyle p=0.5, random erasing p=0.5, 100 epochs / patience 20, TTA. Cross-camera patch macro-F1 0.7908-0.8025 over seeds 42, 123 and 7 (exp200-203, exp232-239). No ResNet18 model ships any more, and its code was removed on 2026-10-20.
- **Frozen DINOv3 ViT-B/16 + linear head** (exp220-231, exp240-254): cross-camera patch macro-F1 **0.8873** (exp240-251, 3 seeds × 4 folds), 0.9489 pooled per photo. The owner selected it on those folds; live since 2026-09-27 (`docs/dinov3_integration_plan.md`).
- **ML-1** (2026-09-29): the leave-one-camera-out folds retired; all cameras equal, none held out (ADR 0010). Cross-camera numbers above come from those folds.
- **ML-2** (live 2026-10-03): a learned bean segmentation stage (EfficientViT-SAM-L0) in front of the classifier (ADR 0011).
- **ML-3** (live 2026-10-06): country classes instead of one class per coffee.
- **ML-5** (live 2026-10-09) and **ML-6** (live 2026-10-13): the XL0-v2, then XL0-v3, segmenter.
- **ML-7** (live 2026-10-17): serving on LiteRT instead of PyTorch.
- **ML-8** (live 2026-10-20): one pool for every session and an augmented head.
- **ML-9** (live 2026-10-24): the XL0-v4 segmenter, and green Indonesia and Peru in training.
- **XL0-v5** (live 2026-10-27): the XL0-v5 segmenter, fine-tuned on the pass7 labels.
- **ML-13** (live 2026-10-29): the XL0-v6 segmenter, fine-tuned on the pass10 labels at seed 42; the live model above.

Every other model is retired; each one's card, sidecars and `.pt.dvc` are in git history and its `.pt` in the DVC cache and remote. `EXPERIMENTS_LOG.md`'s exp270-272 entry lists the last ones retired.

## Where runs are written up

`EXPERIMENTS_LOG.md` holds the experiment history through exp279. exp264-269 (ML-5, ML-6) are summarised there by pointer only; their write-ups are in `docs/plan_segmenter_dataset.html` and `docs/plan_segmenter_retrain.html`. Each ML-n ticket's plan (`docs/plan_*.html`) has the fuller record of its runs; `experiments/index.csv` has every run.

In the log, phases 1-14 (through exp105) were written contemporaneously and are authoritative prose. Phases 15-17 (exp106-175) were **reconstructed on 2026-09-03** from `index.csv` and the archived configs — the numbers are recomputed and exact, but the reasoning-as-it-happened is genuinely lost for those runs, and the section says so. Per-experiment metrics, configs, curves and predictions are archived in `experiments/` — see `experiments/README.md`; `experiments/index.csv` is the one-row-per-run summary and is regenerated from the archive directories by `archive_experiment.rebuild_index()`, never hand-edited.

Ticket ML-1 and its implementation plan (`docs/ticket_retire_cross_rig.html`, `docs/plan_retire_cross_rig.html`) record how and why the fold protocol was retired. `docs/dataset_training_reorg_plan.md` records the dataset and rig reorganisation behind the camera-rig numbers.
