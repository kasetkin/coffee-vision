# Coffee Bean CV Classifier

Computer-vision-first classification of coffee bean origin/quality from photos. Camera-only CV is the active approach; a multi-modal hardware sensor rig (NIR spectroscopy, gas sensing, RF dielectric sensing) is documented as a fallback in `hardware/`, only worth building if plain-image CV underperforms.

Sample train patches, one row per class, cropped from the current dataset (`dataset/2026-08-07__box_pictures_all_classes`):

![Sample patches per class](docs/patch_samples.png)

## Repo layout

- `dataset/` — labeled photo captures. Each capture session is its own dated folder (e.g. `2026-07-24__first_pictures/`), tracked with [DVC](#dataset--dvc) rather than committed directly to git. `classes.txt` maps class id → origin/grade/region and is a plain git-tracked text file.
- `webapp/` — the public web service serving the shipped classifier (see [Web service](#web-service) below).
- `hardware/` — sensor datasheets (NIR: AS7263/AS7265x/AS7343, gas: BME688, LEDs) and a design-research writeup (`Computer vision models for coffee bean origin classification - Claude.pdf`) on which physical/chemical signals actually carry origin information. Reference material for the fallback hardware path — nothing here is built or wired up yet.
- `.devcontainer/` — the dev environment (below).
- `.vscode/c_cpp_properties.json` — C/C++ IntelliSense config anticipating firmware work; unused while CV-only is the active path.

## Dev environment

Open in VS Code with the Dev Containers extension ("Reopen in Container"). It builds `Dockerfile.cpu`: Python 3.12, PyTorch (CPU wheels — no NVIDIA GPU on this machine), OpenCV, scikit-learn, DVC, etc. `--device=/dev/dri` passes through this machine's AMD iGPU for OpenCV's OpenCL path; as configured the devcontainer won't start on a host without that device (cloud VM, macOS, NVIDIA-only box) — there's no separate GPU/cloud variant, training happens on this same workstation.

Persisted across rebuilds via named Docker volumes (not part of the repo — a `docker volume prune` or Docker reset would lose them): bash history, Claude Code's config/auth/chat history, and IPython/Jupyter history.

## Dataset & DVC

New capture sessions get tracked with:

```bash
dvc add dataset/<session-name>
git add dataset/<session-name>.dvc dataset/.gitignore
```

This keeps the actual images out of git (only a small `.dvc` pointer + hash gets committed) while still versioning them alongside code.

**No DVC remote is configured yet** — tracked data only lives in the local `.dvc/cache`, so none of it is backed up anywhere yet. Run `dvc remote add -d <name> <url>` (S3/GCS/local NAS/etc.) and `dvc push` before relying on this for anything you can't afford to lose.

## Web service

`webapp/` serves the currently-shipped checkpoint (`allrigs_mixstyle05_e100p20_s17.pt`) behind a
single `POST /classify` endpoint plus a one-page frontend: upload a photo, get either a
classification with scores for all 9 classes or an actionable refusal ("move the camera back",
"this doesn't look like the training data"). It wraps `coffeecv.infer` rather than reimplementing
any of its scale/OOD logic, so the CLI and the web service can never silently disagree. Deployment
(nginx + gunicorn on a dedicated VM) is fully scripted and git-tracked — see `webapp/README.md` for
architecture, redeploy procedure, and the couple of things (domain, TLS cert) that live outside git
by design.

## Status

Devcontainer, dataset pipeline, and a patch-based training/eval pipeline (`coffeecv/`) are all in place. Current adopted config (resnet18, full fine-tune, **bean-unit patch sizing** at 4-7 beans, MixStyle p=0.5 agnostic, random erasing p=0.5, epochs=100/patience=20, eta_min=1e-5, TTA at inference), over the 20 runs at that config: **in-distribution test macro-F1 0.884-0.933**, and **cross-rig (leave-one-rig-out) macro-F1 0.674-0.851** over 14 folds. Quote the range, not a single run — seed-to-seed spread is wider than most of the effects being measured, which is why adoptions since Phase 8 require a *paired* multi-seed check rather than a single seed.

The ~20-point gap between those two ranges is the project's central open problem: the model is much worse on a camera it has not seen than on one it has. Do not quote the in-distribution figure on its own.

Full experiment history is in `EXPERIMENTS_LOG.md`. Phases 1-14 (through exp105) were written contemporaneously and are authoritative prose. Phases 15-17 (exp106-175) were **reconstructed on 2026-09-03** from `index.csv` and the archived configs — the numbers are recomputed and exact, but the reasoning-as-it-happened is genuinely lost for those runs, and the section says so. Per-experiment metrics, configs, curves and predictions are archived in `experiments/` — see `experiments/README.md`; `experiments/index.csv` is the one-row-per-run summary and is regenerated from the archive directories by `archive_experiment.rebuild_index()`, never hand-edited.

Current state, open problems and the ranked plan live in `docs/dataset_training_reorg_plan.md`; that document supersedes this section wherever they disagree.

Known limitation worth reading before further tuning: the validation set is saturating (5 of 9 classes sit at or near f1=1.000, and one run hit val macro-F1 0.9917). Since `best.pt` is selected on peak val macro-F1, this degrades *checkpoint selection*, not just reporting — see the Phase 8 summary.
