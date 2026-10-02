# ML-2 P4: classifier refit on the segmenter's pools

Plan §7 of `docs/plan_segmentation_mask.html`; summary in the ticket's Activity and in `EXPERIMENTS_LOG.md`
(exp255-260).

- `paired_258-260_vs_255-257.txt`: `python -m coffeecv.compare_experiments 258 259 260 --vs 255 256 257`, the
  printout D24 asks for (per seed, both arms' val/test macro-F1 and the paired Δ, their mean and range, and the
  per-class F1).
- `ood_probe_exp260.log`: `build_ood_reference --store-knn-embeddings` and `fit_ood_probe --verify` on exp260
  (seed 7), run on the VM; progress lines dropped. The holdout section is the one read of the holdout.
- `exp260.ood_probe.json`: the probe that run wrote beside `outputs/ml2_p4/segcropped/s7/model.pt` on the VM.
  The model is not shipped; P6 ships the chosen one.
