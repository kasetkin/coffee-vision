"""Archive one finished training run into the git-tracked `experiments/` record.

`outputs/` is gitignored and overwritten by every run, and DVC only ever holds
the *latest* run's outputs — so before Phase 8 the only surviving trace of an
experiment was the hand-written table in EXPERIMENTS_LOG.md. This copies the
small artifacts (metrics/config/history/predictions — ~16KB per run) into
`experiments/<id>__<slug>/`, which is committed to git, and rebuilds
`experiments/index.csv` from every archived run so the whole history stays
queryable without re-running anything. The 44MB checkpoints stay in DVC, which
is what DVC is for.

Usage:
    python -m coffeecv.archive_experiment --id 36 --slug rotation_25deg \
        --note "arbitrary rotation +/-25deg, mean-colour fill"
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

from coffeecv.config import OUTPUTS_DIR, REPO_ROOT

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
BASELINE_FILE = EXPERIMENTS_DIR / "baseline_config.json"
INDEX_FILE = EXPERIMENTS_DIR / "index.csv"

ARCHIVED_FILES = [
    "metrics.json",
    "config.json",
    "history.json",
    "predictions_val.csv",
    "predictions_test.csv",
]

# Crop settings live per session (dataset/<session>.crop.yaml) rather than in
# params.yaml, which is right -- they describe a rig, not the model -- but it
# means a run's config.json does not record them. Exp 47 exposed this: it changed
# the crop and compare_experiments still reported "nothing changed". Copied in so
# each archived run states the data it was actually trained on.
CROP_CONFIGS = sorted((REPO_ROOT / "dataset").glob("*.crop.yaml"))

# Charts are *regenerated* from the archived metrics.json/history.json rather than
# copied from outputs/plots/. Same output either way, but regenerating means a run
# archived after outputs/ has been overwritten still gets its charts -- which is how
# exp 36-45 got theirs backfilled. `patch_samples.png` is deliberately not archived:
# it is drawn from the *val* dataset, so it is near-identical across runs that don't
# change patch geometry, and at 3.3MB it would dominate the archive.
PLOT_FILES = ["confusion_matrix_val.png", "confusion_matrix_test.png", "training_curves.png"]

INDEX_COLUMNS = [
    "exp", "slug", "seed", "val_macro_f1", "val_mcc", "test_macro_f1", "test_mcc",
    "best_epoch", "epochs_trained", "changed_vs_baseline", "note", "created_at", "git_commit",
]


def _changed_vs_baseline(config: dict) -> str:
    """Params that differ from the Phase 7-adopted baseline, as `k=v` pairs."""
    if not BASELINE_FILE.exists():
        return ""
    baseline = json.loads(BASELINE_FILE.read_text())
    diffs = [
        f"{k}={config[k]}"
        for k in sorted(config)
        if k != "env" and k in baseline and config[k] != baseline[k]
    ]
    return "; ".join(diffs)


def _row_for(exp_dir: Path) -> dict | None:
    metrics_file, config_file = exp_dir / "metrics.json", exp_dir / "config.json"
    if not (metrics_file.exists() and config_file.exists()):
        return None
    metrics = json.loads(metrics_file.read_text())
    config = json.loads(config_file.read_text())
    meta = json.loads((exp_dir / "meta.json").read_text()) if (exp_dir / "meta.json").exists() else {}
    val, test = metrics["splits"]["val"], metrics["splits"]["test"]
    return {
        "exp": meta.get("exp", exp_dir.name.split("__")[0]),
        "slug": meta.get("slug", exp_dir.name.split("__", 1)[-1]),
        "seed": config.get("seed"),
        "val_macro_f1": f"{val['macro_f1']:.4f}",
        "val_mcc": f"{val['mcc']:.4f}",
        "test_macro_f1": f"{test['macro_f1']:.4f}",
        "test_mcc": f"{test['mcc']:.4f}",
        "best_epoch": metrics.get("best_epoch"),
        "epochs_trained": metrics.get("epochs_trained"),
        "changed_vs_baseline": _changed_vs_baseline(config),
        "note": meta.get("note", ""),
        "created_at": metrics.get("created_at", ""),
        "git_commit": (config.get("env") or {}).get("git_commit", ""),
    }


def regenerate_plots(exp_dir: Path) -> list[str]:
    """Rebuild this experiment's charts from its own archived JSON.

    Deliberately regenerated rather than copied: `outputs/plots/` holds only the
    most recent run, so copying would have left every earlier experiment
    chartless and unrecoverable. Everything these plots draw is already in
    metrics.json (confusion matrices) and history.json (curves).
    """
    from coffeecv.plotting import plot_confusion_matrix, plot_training_curves

    metrics = json.loads((exp_dir / "metrics.json").read_text())
    class_ids = metrics["class_ids"]
    class_labels = metrics["class_labels"]
    written = []

    for split in ("val", "test"):
        cm = metrics["splits"][split].get("confusion_matrix")
        if not cm:
            continue
        title = f"exp{exp_dir.name.split('__')[0][3:]} {split} confusion matrix"
        if split == "val":
            title += f" (epoch {metrics.get('best_epoch')})"
        out = exp_dir / f"confusion_matrix_{split}.png"
        plot_confusion_matrix(cm, class_ids, class_labels, out, title)
        written.append(out.name)

    history_file = exp_dir / "history.json"
    if history_file.exists():
        out = exp_dir / "training_curves.png"
        plot_training_curves(json.loads(history_file.read_text()), out)
        written.append(out.name)
    return written


def rebuild_index() -> int:
    """Regenerate index.csv from every archived run (rather than appending), so
    a re-archived or deleted experiment can never leave a stale row behind."""
    rows = []
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir() if EXPERIMENTS_DIR.exists() else []):
        if not exp_dir.is_dir():
            continue
        row = _row_for(exp_dir)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (len(str(r["exp"])), str(r["exp"])))
    INDEX_FILE.parent.mkdir(exist_ok=True)
    with open(INDEX_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def archive(exp_id: str, slug: str, note: str) -> Path:
    metrics_file = OUTPUTS_DIR / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"{metrics_file} missing — has the run finished?")

    config = json.loads((OUTPUTS_DIR / "config.json").read_text())
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    run_commit = (config.get("env") or {}).get("git_commit")
    if run_commit and head and run_commit != head:
        # Not fatal: the run legitimately predates the commit that records it.
        # Worth printing, since a *large* gap usually means outputs/ is stale.
        print(f"NOTE: run recorded git_commit={run_commit[:8]}, HEAD is now {head[:8]}")

    exp_dir = EXPERIMENTS_DIR / f"exp{exp_id}__{slug}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVED_FILES:
        src = OUTPUTS_DIR / name
        if src.exists():
            shutil.copy2(src, exp_dir / name)
        else:
            print(f"WARNING: {src} missing, not archived")
    for cfg_path in CROP_CONFIGS:
        shutil.copy2(cfg_path, exp_dir / cfg_path.name)

    (exp_dir / "meta.json").write_text(
        json.dumps({"exp": exp_id, "slug": slug, "note": note}, indent=2) + "\n"
    )

    plots = regenerate_plots(exp_dir)
    n = rebuild_index()
    print(f"Archived to {exp_dir.relative_to(REPO_ROOT)} (+{len(plots)} charts); "
          f"index.csv now has {n} runs")
    return exp_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", help="experiment number, e.g. 36")
    p.add_argument("--slug", help="short kebab/snake name, e.g. rotation_25deg")
    p.add_argument("--note", default="", help="one-line description of the change")
    p.add_argument("--replot-all", action="store_true",
                   help="regenerate charts for every already-archived experiment, then rebuild the index")
    args = p.parse_args()

    if args.replot_all:
        for exp_dir in sorted(d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()):
            written = regenerate_plots(exp_dir)
            print(f"{exp_dir.name}: {len(written)} charts")
        print(f"index.csv now has {rebuild_index()} runs")
        return

    if not (args.id and args.slug):
        p.error("--id and --slug are required unless --replot-all is given")
    archive(args.id, args.slug, args.note)


if __name__ == "__main__":
    main()
