"""ML-13 P7: exp279, the classifier on xl0_v6_s42's crops, beside the live model on the same test patches.

    python analysis/xl0v6/live_baseline.py predict [--runs 279]     # about 5 min a seed
    python analysis/xl0v6/live_baseline.py report  [--runs 279] [--out analysis/xl0v6/live_baseline]

Adapted from analysis/xl0v5/live_baseline.py (ML-12), itself from analysis/ml9/p6_live_baseline.py. The live model is
exp276 (dino3b16_segxl0v5_country_aug8_s42, XL0-v5's crops). exp279 trains on the same pool's photos at the same seed
(the re-crop added none), so its split is exp276's and the live model's training photos are exp279's; the check that
rebuilds the live split proves it. `predict` rebuilds each run's test patches (the same seed, the same patch draw),
embeds them once, and scores them with two heads: the run's own (`outputs/frozen_head/s<seed>/model.pt`), which
must reproduce the run's archived predictions before anything else is trusted, and the live model's. Its
predictions go to outputs/xl0v6_live_baseline/ (gitignored); `report` reads them with the archived runs.

The report: val and test macro-F1 of the candidate, per-class F1 of both models, on every test photo and on the
photos the live model did not train on. The live model is scored on xl0_v6_s42's crops, the ones the candidate
trained on, not on XL0-v5's it serves with: this compares the classifiers, not the pipelines.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coffeecv.class_list import load_classes  # noqa: E402
from coffeecv.dataset import CapturePhoto, split_photos_by_class  # noqa: E402

EXPERIMENTS = REPO_ROOT / "experiments"
OUT = REPO_ROOT / "outputs" / "xl0v6_live_baseline"
LIVE = "dino3b16_segxl0v5_country_aug8_s42"
LIVE_EXP = 276
LIVE_COMMIT = "e6ea780"  # exp276's commit: the pool and class list the live model trained on
SPLITS = ("val", "test")


def run_dir(exp: int) -> Path:
    [d] = list(EXPERIMENTS.glob(f"exp{exp}__*"))
    return d


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def predict(runs: list[int]) -> None:
    from coffeecv.backbones import build_backbone
    from coffeecv.fit_frozen_head import embed, frozen_config
    from coffeecv.fold_data import build_fold_datasets
    from coffeecv.linear_head import predict as head_predict
    from coffeecv.metrics import write_predictions_csv
    from coffeecv.transforms import build_eval_transform

    OUT.mkdir(parents=True, exist_ok=True)
    first = json.loads((run_dir(runs[0]) / "config.json").read_text())
    bb = build_backbone(first["model_name"])
    readout = first["dino"]["readout"]
    for exp in runs:
        d = run_dir(exp)
        config = json.loads((d / "config.json").read_text())
        seed = config["seed"]
        own = REPO_ROOT / "outputs" / "frozen_head" / f"s{seed}" / "model.pt"
        if json.loads((own.parent / "config.json").read_text())["seed"] != seed or \
                (own.parent / "predictions_test.csv").read_bytes() != (d / "predictions_test.csv").read_bytes():
            raise SystemExit(f"{own} is not exp{exp}'s head: its predictions differ from the archive")
        cfg = frozen_config(config["model_name"], seed, smoke=False)
        ds = build_fold_datasets(cfg, None, build_eval_transform(cfg.patch_resize), only=("test",)).test
        with torch.no_grad():
            X, y, _ = embed(ds, bb, readout)
            heads = {name: torch.load(path, map_location="cpu", weights_only=True)
                     for name, path in (("own", own), ("live", REPO_ROOT / "models" / f"{LIVE}.pt"))}
            for name, ck in heads.items():
                if ck["class_ids"] != config["class_ids"]:
                    raise SystemExit(f"{name}'s classes {ck['class_ids']} are not exp{exp}'s")
                w = ck["head"]["weight"]
                linear = torch.nn.Linear(w.shape[1], w.shape[0])
                linear.load_state_dict(ck["head"])
                pred, probs, _ = head_predict(linear, X)
                write_predictions_csv(OUT / f"{name}_exp{exp}_test.csv", y, pred, config["class_ids"],
                                      list(ds.meta), probs)
        check_own(exp)
        print(f"exp{exp} (seed {seed}): {len(y)} test patches scored by its own head and the live one", flush=True)


def check_own(exp: int) -> None:
    """The run's own head on the re-embedded patches gives the archived predictions: same patches, same answers."""
    archived = read_rows(run_dir(exp) / "predictions_test.csv")
    again = read_rows(OUT / f"own_exp{exp}_test.csv")
    keys = [k for k in archived[0] if k.startswith("p_")]
    worst = max(abs(float(a[k]) - float(b[k])) for a, b in zip(archived, again) for k in keys)
    same = [(a["photo"], a["folder_id"], a["pred_label"]) for a in archived] == \
           [(b["photo"], b["folder_id"], b["pred_label"]) for b in again]
    if len(archived) != len(again) or not same or worst > 1e-3:
        raise SystemExit(f"exp{exp}: the re-embedded patches do not reproduce the archive (max |dp| {worst:.2g})")


def live_train() -> set[tuple[str, str]]:
    """(folder_id, photo) of every photo in the live model's training split: its seed's split of the pool at LIVE_COMMIT."""
    lock = yaml.safe_load(subprocess.run(["git", "show", f"{LIVE_COMMIT}:dvc.lock"], cwd=REPO_ROOT, check=True,
                                         capture_output=True, text=True).stdout)
    [out] = [o for o in lock["stages"]["merge_segpool"]["outs"] if o["path"] == "data/segcropped/pool"]
    md5 = out["md5"]
    listing = json.loads((REPO_ROOT / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]).read_text())
    by_folder: dict[str, list[str]] = defaultdict(list)
    for e in listing:
        folder, name = e["relpath"].split("/") if "/" in e["relpath"] else (None, None)
        if folder and name.endswith("__cropped.jpg"):
            by_folder[folder.split("__")[0].removeprefix("class_")].append(name)
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        f.write(subprocess.run(["git", "show", f"{LIVE_COMMIT}:dataset/classes.txt"], cwd=REPO_ROOT, check=True,
                               capture_output=True, text=True).stdout)
        f.flush()
        classes = load_classes(Path(f.name))
    live_cfg = json.loads((run_dir(LIVE_EXP) / "config.json").read_text())
    frac = {s: live_cfg[f"{s}_photo_frac"] for s in ("train", "val", "test")}
    split: dict[str, set] = {s: set() for s in frac}
    for class_idx, key in enumerate(classes.keys):
        pool = [CapturePhoto("pool", name, Path(name), fid) for fid in classes.folders[key] for name in by_folder[fid]]
        for s, chosen in split_photos_by_class(sorted(pool), live_cfg["seed"], class_idx, frac).items():
            split[s] |= {(p.folder_id, p.name) for p in chosen}
    for s in SPLITS:
        archived = {(r["folder_id"], r["photo"]) for r in read_rows(run_dir(LIVE_EXP) / f"predictions_{s}.csv")}
        if archived != split[s]:
            raise SystemExit(f"the rebuilt live split's {s} photos are not exp{LIVE_EXP}'s")
    return split["train"]


def score(rows: list[dict], class_ids: list[str]) -> dict:
    """Patch macro-F1 over the classes present, per-class F1, and photo accuracy (mean patch probability)."""
    if not rows:
        return {"patches": 0}
    present = [c for c in class_ids if any(r["true_label"] == c for r in rows)]
    t = [r["true_label"] for r in rows]
    p = [r["pred_label"] for r in rows]
    per = f1_score(t, p, labels=present, average=None, zero_division=0)
    photos: dict[tuple, list] = defaultdict(list)
    for r in rows:
        photos[(r["folder_id"], r["photo"], r["true_label"])].append([float(r[f"p_{c}"]) for c in class_ids])
    right = sum(class_ids[int(np.mean(v, axis=0).argmax())] == k[2] for k, v in photos.items())
    return {"patches": len(rows), "photos": len(photos), "macro_f1": float(np.mean(per)),
            "per_class_f1": dict(zip(present, map(float, per))), "photo_accuracy": right / len(photos),
            "photos_wrong": sorted(f"{k[1].removesuffix('__cropped.jpg')} ({k[2]} as "
                                   f"{class_ids[int(np.mean(v, axis=0).argmax())]})"
                                   for k, v in photos.items()
                                   if class_ids[int(np.mean(v, axis=0).argmax())] != k[2])}


def span(xs: list[float]) -> str:
    return f"{min(xs):.4f}-{max(xs):.4f} (mean {np.mean(xs):.4f})"


def report(runs: list[int], out: Path) -> None:
    trained = live_train()
    subsets = {
        "all": lambda r: True,
        "unseen by the live model": lambda r: (r["folder_id"], r["photo"]) not in trained,
    }
    result: dict = {"runs": {}, "live": LIVE, "live_trained_photos": len(trained)}
    span_name = f"exp{runs[0]}" if len(runs) == 1 else f"exp{runs[0]}-{runs[-1]}"
    lines = [f"ML-13: {span_name} beside the live model {LIVE} (exp{LIVE_EXP}) on the same test "
             "patches, xl0_v6_s42 crops", ""]
    if len(runs) == 1:  # the plan's R4: a one-seed comparison says so
        lines[-1:-1] = ["One seed: the deltas below are one patch draw, not a mean over seeds (plan R4)."]
    for exp in runs:
        d = run_dir(exp)
        metrics = json.loads((d / "metrics.json").read_text())
        class_ids = metrics["class_ids"]
        seed = json.loads((d / "config.json").read_text())["seed"]
        rec: dict = {"seed": seed, "val_macro_f1": metrics["splits"]["val"]["macro_f1"],
                     "test_macro_f1": metrics["splits"]["test"]["macro_f1"], "subsets": {}}
        rows = {"candidate": read_rows(d / "predictions_test.csv"), "live": read_rows(OUT / f"live_exp{exp}_test.csv")}
        for name, keep in subsets.items():
            rec["subsets"][name] = {m: score([r for r in rs if keep(r)], class_ids) for m, rs in rows.items()}
        result["runs"][exp] = rec
    ids = list(result["runs"])
    lines.append(f"candidate val macro-F1  {span([result['runs'][e]['val_macro_f1'] for e in ids])}")
    lines.append(f"candidate test macro-F1 {span([result['runs'][e]['test_macro_f1'] for e in ids])}")
    lines.append("")
    for name in subsets:
        first = result["runs"][ids[0]]["subsets"][name]["candidate"]
        lines.append(f"test, {name}: {first['photos']} photos, {first['patches']} patches (seed {result['runs'][ids[0]]['seed']})")
        for m in ("candidate", "live"):
            s = [result["runs"][e]["subsets"][name][m] for e in ids]
            lines.append(f"  {m:9s} patch macro-F1 {span([x['macro_f1'] for x in s])}   photo accuracy "
                         f"{span([x['photo_accuracy'] for x in s])}")
        lines.append("  per class, F1 mean over seeds (candidate / live / delta):")
        classes = list(result["runs"][ids[0]]["subsets"][name]["candidate"]["per_class_f1"])
        for c in classes:
            cand = np.mean([result["runs"][e]["subsets"][name]["candidate"]["per_class_f1"].get(c, np.nan) for e in ids])
            live = np.mean([result["runs"][e]["subsets"][name]["live"]["per_class_f1"].get(c, np.nan) for e in ids])
            lines.append(f"    {c:10s} {cand:.4f} / {live:.4f} / {cand - live:+.4f}")
        if name != "all":
            for e in ids:
                for m in ("candidate", "live"):
                    wrong = result["runs"][e]["subsets"][name][m]["photos_wrong"]
                    if wrong:
                        lines.append(f"  exp{e} {m} photos wrong: {', '.join(wrong)}")
        lines.append("")
    out.with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    out.with_suffix(".txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {out.with_suffix('.txt').relative_to(REPO_ROOT)} and .json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("predict", "report"))
    ap.add_argument("--runs", type=int, nargs="+", default=[279])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "analysis" / "xl0v6" / "live_baseline")
    args = ap.parse_args(argv)
    if args.step == "predict":
        predict(args.runs)
    else:
        report(args.runs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
