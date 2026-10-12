"""Ticket ML-6 P2 (D2 (a)): the shipped XL0-v2 on the new training and validation negatives, before any training.

    python analysis/ml6_p2_baseline/p2_baseline.py            # predict, then report
    python analysis/ml6_p2_baseline/p2_baseline.py --report   # report from the stored masks

The negatives are those of labels/ml5/seg_dataset.yaml's training and validation splits that P1's append added
(the file as of the commit before it, 5a3bfcf^, did not list them). The new test negatives are not looked at:
they wait for P7 (D2). Masks come from seg_predict.predict at the pinned 4 threads, as the seg_ml5_predict stages
make theirs. Rates per split, at serving's tiny-mask threshold (params.yaml seg_min_area_frac) and at the
fine-tune's selection one (seg_ft.select_min_area_frac).

Out: report.json and summary.txt here (git); the masks, their index.csv and a contact sheet of every non-empty
mask under outputs/ml6_p2_baseline/ (not in git: the photos are the owner's own, DVC only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coffeecv.config import RunConfig  # noqa: E402
from coffeecv.dataset import load_rgb_image  # noqa: E402
from coffeecv.repo_files import read_csv  # noqa: E402
from coffeecv.review_masks import view_image  # noqa: E402
from coffeecv.seg_dataset import load_seg_dataset, photo_id  # noqa: E402
from coffeecv.seg_finetune import FtParams  # noqa: E402
from coffeecv.seg_predict import predict  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, named_params  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = REPO_ROOT / "outputs" / "ml6_p2_baseline"
BEFORE = "5a3bfcf^"           # the commit before P1's append
MODEL, THREADS = "xl0_v2", 4
SPLITS = ("train", "validation")


def new_negatives() -> list[dict]:
    """seg_predict's entries for the training and validation negatives P1's append added, in file order."""
    old = yaml.safe_load(subprocess.run(["git", "show", f"{BEFORE}:labels/ml5/seg_dataset.yaml"], cwd=REPO_ROOT,
                                        capture_output=True, text=True, check=True).stdout)["photos"]
    listed = {e["path"] for e in old}
    return [{"id": photo_id(e["path"]), "source": e["source"], "split": e["split"], "batch": e["batch"],
             "tag": e["tag"], "path": e["path"], "sha256": e["sha256"]}
            for e in load_seg_dataset() if e["source"] == "negative" and e["split"] in SPLITS
            and e["path"] not in listed]


def rates(rows: list[dict], thresholds: dict[str, float]) -> dict:
    """Per split and overall: photos, empty masks, and empty or under each threshold."""
    out = {}
    for split in (*SPLITS, "all"):
        sub = [r for r in rows if split == "all" or r["split"] == split]
        area = [float(r["area_frac"]) for r in sub]
        out[split] = {"n": len(sub), "empty": sum(a == 0 for a in area),
                      **{f"empty_or_tiny_{k}": sum(a < t for a in area) for k, t in thresholds.items()}}
    return out


def contact_sheet(rows: list[dict], out: Path, tile: int = 360, cols: int = 6) -> None:
    """Every non-empty mask outlined on its photo, largest first, its area and split under each tile."""
    rows = sorted(rows, key=lambda r: -float(r["area_frac"]))
    sheet = Image.new("RGB", (cols * tile, max(1, -(-len(rows) // cols)) * (tile + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, r in enumerate(rows):
        mask = np.array(Image.open(OUT / "masks" / f"{r['id']}.png")) > 0
        im = Image.fromarray(view_image(load_rgb_image(REPO_ROOT / r["path"]), mask))
        im.thumbnail((tile, tile))
        x, y = (n % cols) * tile, (n // cols) * (tile + 18)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + tile + 2), f"{Path(r['path']).name} {r['split']} {float(r['area_frac']):.4f}",
                  fill="black")
    sheet.save(out, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="report from the stored masks, without predicting")
    args = ap.parse_args()
    cfg = RunConfig.from_params_yaml()
    entries = new_negatives()
    if not args.report:
        torch.set_num_threads(THREADS)
        predict(entries, BeanSegmenter(named_params(MODEL, cfg.seg_mask_select)), OUT / "masks", cfg)
    rows = read_csv(OUT / "masks" / "index.csv")
    if sorted(r["path"] for r in rows) != sorted(e["path"] for e in entries):
        raise SystemExit("the stored masks are not those of the new training and validation negatives")
    thresholds = {"serving": cfg.seg_min_area_frac,
                  "selection": FtParams.from_config(cfg).select_min_area_frac}
    masked = [r for r in rows if float(r["area_frac"]) > 0]
    rep = {"ticket": "ML-6 P2 (D2 (a))", "model": MODEL, "threads": THREADS, "thresholds": thresholds,
           "negatives": {"batches": dict(Counter(f"{r['batch']} / {r['tag']}" for r in rows))},
           "rates": rates(rows, thresholds),
           "masked": [{"path": r["path"], "split": r["split"], "area_frac": float(r["area_frac"])}
                      for r in sorted(masked, key=lambda r: -float(r["area_frac"]))]}
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    lines = [f"XL0-v2 on the {len(rows)} new training and validation negatives (ML-6 P2, D2 (a)); "
             f"test negatives not looked at",
             f"thresholds: serving {thresholds['serving']}, selection {thresholds['selection']}"]
    for split, c in rep["rates"].items():
        lines.append(f"  {split:10s} n={c['n']:3d}  empty {c['empty']:3d}  empty or tiny: serving "
                     f"{c['empty_or_tiny_serving']:3d}, selection {c['empty_or_tiny_selection']:3d}")
    lines.append(f"{len(masked)} photos with a non-empty mask; contact sheet: "
                 f"{(OUT / 'masked_sheet.jpg').relative_to(REPO_ROOT)} (not in git)")
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if masked:
        contact_sheet(masked, OUT / "masked_sheet.jpg")


if __name__ == "__main__":
    main()
