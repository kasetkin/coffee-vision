# Experiments: provenance, comparison and reporting

Read before launching a recorded run or a screen, comparing runs, or reporting results. What gets adopted, and who decides, is [ADR 0002](../adr/0002-paired-multi-seed-evidence.md).

## Launching

- DINOv3 heads: `python -m coffeecv.fit_frozen_head --seeds 42 123 7 --start-exp <N>`. The legacy ResNet18 driver is `coffeecv.run_all_rigs` (it runs `dvc repro train`); a recorded run never comes from calling a training module directly, or `dvc.lock` falls out of step with the commit.
- **Commit the code before the run.** A run's commit must hold everything it depended on: code, `params.yaml`, the data version in `dvc.lock`, and its archived outputs. The drivers refuse a dirty tree, stale stages and a detached HEAD; `--allow-dirty` overrides that and should be rare. A sweep once ran 30 hours on uncommitted code.
- **Run the changed path at toy scale first** (`CODING_STANDARDS.md`, "Proving a change").
- **Before a paired run, diff its config against the baseline's**, key by key; only the variable under test may differ:

  ```python
  import yaml, json, glob
  live = yaml.safe_load(open('params.yaml'))
  base = json.load(open(glob.glob('experiments/exp<N>__*')[0] + '/config.json'))
  for k in sorted(set(base) & set(live)):
      if live[k] != base[k]: print(f'DIFFERS  {k}: running={live[k]}  baseline={base[k]}')
  ```

## Changing the segmenter

A new segmenter decoder or tiny-mask threshold in `params.yaml` (`seg_decoder`, `seg_min_area_frac`) changes every crop the classifier sees. The follow-through, in order:

1. Re-crop: `dvc repro -s` each `segcrop` and `merge_segcam_*` stage, never a plain `dvc repro`. Open a few crops at full size, every D18 fallback included.
2. Regenerate the DINOv3 fixture: `python scripts/make_dinov3_fixture.py`. Its patches come from `data/segcropped/cam_iphone`, so after a re-crop `test_dino_backbone.test_matches_meta_reference` goes red until it is rerun. The script clones and runs Meta's dinov3 at a pinned commit: the owner runs it, or says Claude may. Commit the new `tests/fixtures/dinov3_vits16_reference.npz` beside the re-crop. Precedent: ML-3 P2b, ML-5 P11 (`f585d14`), ML-6 P8.
3. Retrain the classifier as a recorded run (seeds 42/123/7) and compare against the previous segmenter's runs; refit the OOD probe on the shipped candidate.
4. Convert the shipped candidate for serving (below).

## Converting a candidate for serving

The site serves LiteRT conversions, not the `.pt` ([ADR 0019](../adr/0019-serving-runs-litert-accelerator-per-host.md)). Between shipping a candidate (`models/<name>.pt`) and deploying it, add it to the `litert_export` stage's `foreach` in `dvc.yaml` and run `dvc repro -s litert_export@<name>`: it converts the encoder, the decoder and DINOv3 in `convert/.venv` (create it with `uv sync --project convert` if missing), then checks them against PyTorch at D7's fp32 bars on the deploy fixtures ([ADR 0020](../adr/0020-converted-models-pass-d7-before-they-serve.md)). A model that fails loses its manifest and cannot deploy. Push `models/<name>.litert` to the DVC remote and verify it, as for the checkpoint. A different architecture, or a new `litert-torch` or `ai-edge-litert`, reruns the forms' checks first ([ADR 0022](../adr/0022-export-forms-are-exact.md)), and on the serving host P7's answer check (`analysis/ml7_gpu/p7_answers.py`).

## Reading the record

- `experiments/expNNN__<slug>/` (config, metrics, predictions, charts) and `experiments/index.csv` are the record; compare runs with `python -m coffeecv.compare_experiments <id> --vs <id>`.
- A run's `config.json` `git_commit` is the commit it trained on, the previous run's archive commit. For exp116-129 and exp168-175 it no longer resolves, by decision ([ADR 0004](../adr/0004-deleted-provenance-commits-stay-deleted.md)).
- `config.json` does not record the class list: check `metrics.json`'s `class_ids` before comparing runs (exp174 trained 9 classes under the same config as exp175's 10, and is marked VOID).
- Fold-era cross-camera numbers in `index.csv` are not comparable with anything since ML-1, and nothing since ML-3 (country classes) is comparable with what came before it.

## Reproducibility limits

- Results are bit-identical only on the same machine at the same thread count. A different thread count changes float reduction order, which over 70 epochs moved early stopping by 4 epochs; across CPUs, outputs differ in float rounding (cosine >= 0.9999998) and a few mask pixels. That is why masks are computed once and stored.

## Reporting

- Report a result as a range over seeds, never a single number, and give per-class F1 next to the overall figure: the findings here have usually been per class.
- Name a "best epoch" only from the archived record (`history.json`, `metrics.json`), never from a live-log tail, which once missed an earlier peak.
