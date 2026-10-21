"""ML-7 P8: the serving path's answers on the deploy fixtures, three ways, against PyTorch.

    torch       coffeecv.infer.classify_one with the PyTorch models (TorchClassifier, the PyTorch segmenter), here
    litert_cpu  the same with ML-7 P7's served files on LiteRT's CPU, all fp32 (tests.test_litert.build_cpu_dir), here
    litert_gpu  webapp.app on yappy's GPU (p8_yappy.py's --out), the served cell: segmenter fp32, DINOv3 fp16

The seed is the web app's ([42, 0]). Per photo: verdict, top-1, the largest class-probability change, the guard's
score, the mask's area and the crop box; the bars are ML-7 D7's (ADR 0020): no verdict change, and for fp16 DINOv3
top-1 changes on at most 1% of served photos and a p95 probability change of at most 0.05.

    python analysis/ml7_gpu/p8_compare.py --gpu analysis/ml7_gpu/p8_yappy.json --out analysis/ml7_gpu/p8_compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from coffeecv import infer, litert  # noqa: E402
from coffeecv.class_list import load_classes  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, seg_params  # noqa: E402
from tests import test_litert  # noqa: E402

CHECKPOINT = test_litert.CHECKPOINT


def summary(e: dict) -> dict:
    """The fields compared, from a classify_one entry or the web response built from one."""
    ranked = e["ranked"]
    probs = {(r["id"] if isinstance(r, dict) else r[0]): (r["score"] if isinstance(r, dict) else r[2]) for r in ranked}
    ood = e.get("ood") or {}
    return {"verdict": e.get("verdict"), "top1": max(probs, key=probs.get), "probs": probs,
            "guard": ood.get("score"), "mask_area_frac": e.get("mask_area_frac"), "box": (e.get("crop") or {}).get("box")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=Path, help="p8_yappy.py's output")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    photos = [REPO / p for _, _, p in test_litert.FIXTURES]
    cfg, _ = infer.config_for_checkpoint(CHECKPOINT, None)
    classes = load_classes(REPO / cfg.classes_file)
    ids, labels = list(classes.keys), dict(classes.labels)
    tc = infer.TorchClassifier(*infer.load_model(CHECKPOINT, cfg.model_name, len(ids)))
    ref = infer.load_ood_reference(CHECKPOINT, tc.embedding_dim)
    probe = infer.load_ood_probe(CHECKPOINT, embedding_dim=tc.embedding_dim)
    with tempfile.TemporaryDirectory() as tmp:
        test_litert.build_cpu_dir(Path(tmp))
        models = litert.load_models(Path(tmp))
        backends = {"torch": (tc, infer.segmenter_for(cfg)),
                    "litert_cpu": (models.classifier, BeanSegmenter(seg_params(cfg), engine=models.sam))}
        answers = {b: {} for b in backends}
        for photo in photos:
            for b, (clf, seg) in backends.items():
                e = infer.classify_one(photo, cfg, ids, labels, clf, ref, tta=False, probe=probe, segmenter=seg)
                answers[b][photo.name] = summary(e)
            print(photo.name, {b: (a[photo.name]["verdict"], a[photo.name]["top1"]) for b, a in answers.items()},
                  flush=True)
    if args.gpu:
        g = json.loads(args.gpu.read_text())
        answers["litert_gpu"] = {r["photo"]: summary(r["classify"]) for r in g["rows"] if r["pass"] == 0}
    out = {"answers": answers, "against_torch": {}}
    for b in answers:
        if b == "torch":
            continue
        rows = []
        for name, t in answers["torch"].items():
            a = answers[b].get(name)
            if a is None:
                continue
            rows.append({"photo": name, "verdict_same": a["verdict"] == t["verdict"], "top1_same": a["top1"] == t["top1"],
                         "max_prob_change": max(abs(a["probs"][c] - t["probs"][c]) for c in t["probs"]),
                         "guard": [a["guard"], t["guard"]], "mask_area_frac": [a["mask_area_frac"], t["mask_area_frac"]],
                         # A web response carries no crop: GPU boxes, masks and guard scores are in its request log
                         "box_same": None if a["mask_area_frac"] is None else a["box"] == t["box"]})
        changes = [r["max_prob_change"] for r in rows]
        out["against_torch"][b] = {"photos": len(rows), "verdict_changes": sum(not r["verdict_same"] for r in rows),
                                   "top1_changes": sum(not r["top1_same"] for r in rows),
                                   "box_changes": sum(r["box_same"] is False for r in rows),
                                   "max_prob_change_p95": float(np.percentile(changes, 95)),
                                   "max_prob_change_max": max(changes), "rows": rows}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({b: {k: v for k, v in s.items() if k != "rows"} for b, s in out["against_torch"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
