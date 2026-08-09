# Experiment record

Per-experiment metrics, kept in git so the history survives independently of
`outputs/` (gitignored, overwritten by every run) and of DVC (which only ever
holds the *latest* run's outputs). Before Phase 8 the only surviving trace of a
finished experiment was the hand-written table in `EXPERIMENTS_LOG.md`.

Split of responsibilities: **git holds the small stuff** (metrics, config,
history, predictions, charts — ~330KB per run), **DVC holds the bulk** (44MB
checkpoints, via `dvc.lock` and `models/*.dvc`).

Budget note, since charts are 95% of an archived run's size: the archive grows
~330KB per experiment (3.5MB for the first 10). If that ever becomes unwelcome,
the charts are the thing to drop — they are fully derived from the JSON beside
them and `--replot-all` recreates them on demand.

## The adopted model

`models/phase8_best_random_erasing_0.5.pt` is DVC-tracked (git holds the `.dvc`
pointer) with a git-tracked `.json` model card next to it recording provenance,
expected performance as a *range*, and known failure modes.

It is pinned deliberately rather than relied on via `dvc.lock` history: the DVC
cache does still hold every run's checkpoint (~4.7GB, 90 of them), but those are
referenced only by *historical* `dvc.lock` commits, so a `dvc gc` would delete
them. A `.dvc` file in the working tree is what makes this one survive that.

Note it is **not** the highest test score ever recorded — exp 44 scored 0.9694 —
because exp 44's setting was rejected when a paired second seed reversed its
sign. This is the best *confirmed* config, which is the one worth keeping.

## Contents

- `exp<N>__<slug>/` — one directory per Phase 8 run: `metrics.json` (aggregate +
  per-class), `config.json` (full resolved config + torch/torchvision/git env),
  `history.json` (per-epoch curves), `predictions_{val,test}.csv`, `meta.json`,
  and three charts: `confusion_matrix_{val,test}.png` and `training_curves.png`.

  The charts are **regenerated** from that run's own `metrics.json`/`history.json`,
  not copied from `outputs/plots/`. Same picture either way, but regenerating
  means a run archived after `outputs/` was overwritten still gets its charts —
  which is how exp 36-45 got theirs backfilled after the fact. Rebuild them all
  any time with `python -m coffeecv.archive_experiment --replot-all`.

  Deliberately **not** archived: `outputs/plots/patch_samples.png`. It is drawn
  from the *val* dataset, so it barely changes between runs that don't touch
  patch geometry, and at 3.3MB it would be ~10x the rest of the archive combined.

- `inference_runs/` — rescued output of `coffeecv/infer.py` on the unlabeled
  2026-08-06 session. See its README: these lack model provenance and are two
  config generations old.
- `index.csv` — one row per archived run. **Generated**, never hand-edited:
  `python -m coffeecv.archive_experiment` rebuilds it by scanning every
  `exp*/` directory, so a re-archived or deleted run can't leave a stale row.
- `baseline_config.json` — the Phase 7-adopted config (exp 20). `index.csv`'s
  `changed_vs_baseline` column is a diff against this, so each row states what
  it actually varied.
- `pre_phase8_from_log.csv` — exp 18-35 (Phase 7), **transcribed by hand from
  `EXPERIMENTS_LOG.md`**, not generated. Those runs predate this archive and
  their artifacts were overwritten, so aggregate metrics are all that could be
  recovered — no per-class breakdowns, no curves, and exp 34/35's total epoch
  counts weren't recorded in the log tables. Treat it as a convenience index
  into the log's prose, which stays authoritative. Experiments 1-17 ran against
  the old single-photo-per-class dataset or a since-changed split scheme and
  aren't comparable to current numbers, so they're deliberately not transcribed.

## Adding a run

```bash
dvc repro                       # trains; updates dvc.lock and outputs/
python -m coffeecv.archive_experiment --id 36 --slug rotation_25deg \
    --note "arbitrary rotation +/-25deg, mean-colour fill"
```

## Reading it

```bash
column -s, -t experiments/index.csv                    # whole sweep at a glance
python -m coffeecv.compare_experiments 36 --vs 20      # per-class delta vs baseline
```
