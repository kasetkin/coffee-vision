"""Crop one capture session's raw photos, driven by that session's crop config.

The unit of work is a *session*, not a class directory, because a session is what
gets added to this project over time and what shares a rig geometry. Settings come
from `dataset/<session>.crop.yaml` rather than `params.yaml`: they describe a rig,
not the model, and a future session shot at a different angle may need a different
`method` entirely rather than a different number. See that file's header.

    python -m coffeecv.crop_session --session 2026-08-07__box_pictures_all_classes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from coffeecv.config import REPO_ROOT
from coffeecv.crop_tray import crop_dataset, crop_dataset_fixed_trim

RAW_ROOT = REPO_ROOT / "dataset"
CROPPED_ROOT = REPO_ROOT / "data" / "cropped"


def load_crop_config(session: str) -> dict:
    path = RAW_ROOT / f"{session}.crop.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No crop config at {path}. Every session needs one — it records which crop "
            f"method its rig requires, which is not inferable from the photos."
        )
    cfg = yaml.safe_load(path.read_text()) or {}
    if "method" not in cfg:
        raise ValueError(f"{path} must set `method` (fixed_trim | adaptive)")
    return cfg


def crop_session(session: str) -> dict:
    cfg = load_crop_config(session)
    method = cfg["method"]
    raw_session = RAW_ROOT / session
    out_session = CROPPED_ROOT / session
    if not raw_session.is_dir():
        raise FileNotFoundError(f"No raw session at {raw_session}")

    class_dirs = sorted(p for p in raw_session.iterdir() if p.is_dir() and p.name.startswith("class_"))
    if not class_dirs:
        raise FileNotFoundError(f"No class_* directories under {raw_session}")

    totals = {"classes": 0, "images": 0, "flagged": 0}
    for class_dir in class_dirs:
        out_dir = out_session / class_dir.name
        if method == "fixed_trim":
            # `trim` (per-side mapping) preferred; `trim_frac` scalar still accepted.
            reports = crop_dataset_fixed_trim(class_dir, out_dir, trim=cfg.get("trim", cfg.get("trim_frac")))
        elif method == "adaptive":
            reports = crop_dataset(
                class_dir, out_dir,
                border_max_contamination=float(cfg.get("border_max_contamination", 0.22)),
            )
        else:
            raise ValueError(f"Unknown crop method {method!r} in {session}.crop.yaml")

        flagged = [r for r in reports if r.get("needs_review")]
        totals["classes"] += 1
        totals["images"] += len(reports)
        totals["flagged"] += len(flagged)
        for r in flagged:
            print(f"  REVIEW {class_dir.name}/{r['file']}: {r.get('error') or 'flagged'}")

    # One manifest per session, next to the crops, recording exactly what produced
    # them. crop_tray.py already writes a per-class crop_report.json with the
    # per-photo boxes; this is the session-level summary that ties them together.
    out_session.mkdir(parents=True, exist_ok=True)
    (out_session / "crop_manifest.json").write_text(
        json.dumps({"session": session, "config": cfg, **totals}, indent=2) + "\n"
    )
    print(f"{session}: cropped {totals['images']} images across {totals['classes']} classes "
          f"({totals['flagged']} flagged) using method={method}")
    return totals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True, help="session folder name under dataset/")
    args = p.parse_args()
    crop_session(args.session)


if __name__ == "__main__":
    main()
