"""Ticket ML-6 P6: XL0-v3's validation result, from its card, with the validation negatives per batch.

    python analysis/ml6_p6_validation/validation_by_batch.py

The card (models/seg/xl0_v3.json) records the chosen epoch, the validation mean IoU and the overall share of
validation negatives empty or tiny. The share per batch is recomputed here with the fine-tune's own selection code
(seg_finetune.Selection, decode, the box prompt, seg_ft.select_min_area_frac) and the chosen decoder; its overall
share must equal the card's. No test photo is read. Out: report.json and summary.txt here.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coffeecv import seg_finetune as ft  # noqa: E402
from coffeecv.config import RunConfig  # noqa: E402
from coffeecv.seg_dataset import load_seg_dataset  # noqa: E402
from coffeecv.segment_beans import load_decoder  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME, THREADS = "xl0_v3", 8


def main() -> None:
    torch.set_num_threads(THREADS)
    card = json.loads((REPO_ROOT / "models" / "seg" / f"{NAME}.json").read_text())
    p = ft.FtParams.from_config(RunConfig.from_params_yaml())
    rows = [r for r in ft.label_rows(card["labels"]) if r["role"] == "neg_select"]
    batch = {e["path"]: f"{e['batch']} / {e['tag']}" for e in load_seg_dataset() if e["source"] == "negative"}
    pred, wsha = ft.build_sam(card["base_weights"])
    load_decoder(pred.model, f"models/seg/{NAME}.pt", wsha)
    pred.model.eval()
    sel = ft.Selection(rows, pred.model)
    k = ft.output_index(card["seg_mask_select"])
    per = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for r, (emb, lab, box) in zip(rows, sel.neg):
            logits, _ = ft.decode(pred.model, emb, box, k)
            ok = float((logits[lab != ft.IGNORE] > 0).float().mean()) < p.select_min_area_frac
            for key in (batch[r["path"]], "all"):
                per[key][0] += ok
                per[key][1] += 1
    chosen = card["chosen"]
    rep = {"model": NAME, "labels": card["labels"], "chosen_epoch": chosen["epoch"], "val_miou": chosen["val_miou"],
           "neg_select_empty_card": chosen["neg_select_empty"], "select_min_area_frac": p.select_min_area_frac,
           "neg_select_by_batch": {k: {"empty_or_tiny": a, "n": n} for k, (a, n) in sorted(per.items())},
           "epochs_run": len(card["history"]) - 1, "threads": card["threads"]}
    a, n = per["all"]
    if abs(a / n - chosen["neg_select_empty"]) > 1e-9:
        raise SystemExit(f"recomputed {a}/{n} differs from the card's {chosen['neg_select_empty']}")
    (HERE / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    lines = [f"XL0-v3 (ML-6 P6), validation only: epoch {chosen['epoch']} of {rep['epochs_run']} chosen, score "
             f"{chosen['score']:.4f}", f"  validation positives: mean IoU {chosen['val_miou']:.4f}",
             f"  validation negatives empty or tiny (< {p.select_min_area_frac}):"]
    lines += [f"    {k:50s} {a}/{n}" for k, (a, n) in sorted(per.items(), key=lambda x: x[0] == "all")]
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
