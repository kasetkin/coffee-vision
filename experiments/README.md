# Experiment record

Per-experiment metrics, kept in git so the history survives independently of
`outputs/` (gitignored, overwritten by every run) and of DVC (which only ever
holds the *latest* run's outputs). Before Phase 8 the only surviving trace of a
finished experiment was the hand-written table in `EXPERIMENTS_LOG.md`.

Split of responsibilities: **git holds the small stuff** (metrics, config,
history, predictions — ~16KB per run, diffable and greppable), **DVC holds the
bulk** (44MB checkpoints per run, via `dvc.lock`).

## Contents

- `exp<N>__<slug>/` — one directory per Phase 8 run: `metrics.json` (aggregate +
  per-class), `config.json` (full resolved config + torch/torchvision/git env),
  `history.json` (per-epoch curves), `predictions_{val,test}.csv`, `meta.json`.
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
