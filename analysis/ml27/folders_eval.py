"""Ticket ML-27: the release classifier and the candidate, each answering the photos of random_date_raccoon, massimo
and tasty_coffee that neither trained on, one confusion matrix per model (the owner's request, 2026-11-12).

The photos: every pool crop of the three sessions (so the owner's declined crops, labels/ml3/pool_exclude.csv, are
out), less each model's train split. A split is rebuilt as the run drew it: `dataset.split_photos_by_class` over
the pool the run read (exp280: data/segcropped/pool at its commit be3e8c6, from the DVC cache; exp283: the pool
now), at the run's seed and photo fractions. The rebuilt val and test splits are checked against every photo of
the run's own predictions_val.csv and predictions_test.csv; a miss stops the script. Val photos stay in: neither
head fitted on them (C is fixed), though exp283 was picked over its sibling seeds on val (D5).

Each photo is answered as serving does, at the store's canonical photo: segmented once by xl0_v7r_s123 (both
models' segmenter, checked), then `classify_segmented` with 40 patches, seed key [42, 0] and the checkpoint's TTA
setting, per model. The answer is the top-1 country, without the OOD guard: the candidate's probe is being fitted.
A photo `classify_segmented` cannot measure is counted apart.

    python analysis/ml27/folders_eval.py      # writes analysis/ml27/folders_eval.{txt,json} and two PNGs
    python analysis/ml27/folders_eval.py --limit 3 --out /tmp/fe   # a toy run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coffeecv import crop_io  # noqa: E402
from coffeecv.class_list import load_classes  # noqa: E402
from coffeecv.config import REPO_ROOT  # noqa: E402
from coffeecv.dataset import CapturePhoto, split_photos_by_class  # noqa: E402
from coffeecv.infer import (TorchClassifier, classify_segmented, config_for_checkpoint,  # noqa: E402
                            inference_tta_for, load_model, segment_photo, segmenter_for)
from coffeecv.plotting import plot_confusion_matrix  # noqa: E402

SESSIONS = ("random_date_raccoon", "massimo", "tasty_coffee")
MODELS = {  # arm: (checkpoint, experiment, the pool's .dir hash it trained on; None: the pool now)
    "release": ("models/dino3b16_segxl0v7r_country_aug8_s42.pt", 280, "7fcbbbb9ec5b854f6a0eaa78531946ba.dir"),
    "candidate": ("models/dino3b16_segxl0v7r_country_aug8_ml27_s42.pt", 283, None),
}
N_PATCHES, SEED_KEY = 40, [42, 0]
OUT = REPO_ROOT / "analysis" / "ml27"


def folder_id(class_dir_name: str) -> str:
    return class_dir_name.split("__")[0].removeprefix("class_")


def pool_listing(dir_hash: str | None) -> list[tuple[str, str]]:
    """(class folder id, crop name) of every crop in a pool: the .dir listing in the DVC cache, or the pool now."""
    if dir_hash is None:
        root = REPO_ROOT / "data" / "segcropped" / "pool"
        return [(folder_id(p.parent.name), p.name) for p in sorted(root.glob(f"class_*/*{crop_io.CROP_SUFFIX}"))]
    listing = json.loads((REPO_ROOT / ".dvc" / "cache" / "files" / "md5" / dir_hash[:2] / dir_hash[2:]).read_text())
    rels = [e["relpath"] for e in listing if e["relpath"].endswith(crop_io.CROP_SUFFIX)]
    return [(folder_id(r.split("/")[0]), r.split("/")[1]) for r in rels]


def splits(classes, crops: list[tuple[str, str]], cfg) -> dict[str, dict[str, str]]:
    """{split: {crop name: country}}, as the run drew it."""
    frac = {"train": cfg.train_photo_frac, "val": cfg.val_photo_frac, "test": cfg.test_photo_frac}
    out = {"train": {}, "val": {}, "test": {}}
    for idx, key in enumerate(classes.keys):
        pool = [CapturePhoto("pool", name, Path(name), fid) for fid, name in crops if fid in classes.folders[key]]
        for split, photos in split_photos_by_class(pool, cfg.seed, idx, frac).items():
            out[split].update({p.name: key for p in photos})
    return out


def check_against_record(exp: int, split_map: dict[str, dict[str, str]]) -> None:
    (run,) = (REPO_ROOT / "experiments").glob(f"exp{exp}__*")
    for split, csv_name in (("val", "predictions_val.csv"), ("test", "predictions_test.csv")):
        with (run / csv_name).open() as f:
            recorded = {(r["photo"], r["true_label"]) for r in csv.DictReader(f)}
        missing = [(p, t) for p, t in recorded if split_map[split].get(p) != t]
        if missing:
            raise SystemExit(f"exp{exp}: {len(missing)} {split} photos of its record not in the rebuilt split, "
                             f"e.g. {missing[:3]}")
        print(f"exp{exp}: the rebuilt {split} split holds all {len(recorded)} photos of its {csv_name}")


def session_photos() -> dict[str, tuple[str, Path]]:
    """{crop name: (session, canonical photo)} for the three sessions."""
    out = {}
    for session in SESSIONS:
        for crop in sorted((REPO_ROOT / "data" / "segcropped" / session).glob(f"class_*/*{crop_io.CROP_SUFFIX}")):
            stem = crop.name.removesuffix(crop_io.CROP_SUFFIX)
            (canon,) = (REPO_ROOT / "data" / "canonical" / session / crop.parent.name).glob(f"{stem}.*.jxl")
            out[crop.name] = (session, canon)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="answer only the first N photos (a toy run)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    classes = load_classes(REPO_ROOT / "dataset" / "classes.txt")
    keys = list(classes.keys)
    now_pool = dict((name, fid) for fid, name in pool_listing(None))
    photos = session_photos()
    in_pool = {n: v for n, v in photos.items() if n in now_pool}
    print(f"{len(photos)} crops in {', '.join(SESSIONS)}; {len(in_pool)} in the pool "
          f"({len(photos) - len(in_pool)} declined, pool_exclude.csv)")

    arms, trained = {}, set()
    for arm, (ckpt, exp, dir_hash) in MODELS.items():
        cfg, _ = config_for_checkpoint(REPO_ROOT / ckpt, None)
        split_map = splits(classes, pool_listing(dir_hash), cfg)
        check_against_record(exp, split_map)
        trained |= set(split_map["train"])
        model, head = load_model(REPO_ROOT / ckpt, cfg.model_name, len(keys))
        arms[arm] = (cfg, TorchClassifier(model, head), inference_tta_for(REPO_ROOT / ckpt, cfg.model_name), exp)
    seg_keys = ("seg_decoder_sha256", "seg_weights_sha256", "seg_prompt", "seg_mask_select", "seg_min_area_frac",
                "seg_keep_frac")
    cfgs = [a[0] for a in arms.values()]
    if any(getattr(cfgs[0], k) != getattr(c, k) for c in cfgs[1:] for k in seg_keys):
        raise SystemExit("the two models' segmenter settings differ; segment per model")

    held = sorted(n for n in in_pool if n not in trained)[:args.limit]
    print(f"{len(held)} of them in neither model's train split\n")
    segmenter = segmenter_for(cfgs[0])
    rows = []
    for i, name in enumerate(held, 1):
        session, canon = in_pool[name]
        truth = classes.key_of(now_pool[name])
        seg = segment_photo(canon, cfgs[0], segmenter=segmenter)
        row = {"photo": name, "session": session, "true": truth}
        for arm, (cfg, clf, tta, _) in arms.items():
            e = classify_segmented(seg, cfg, keys, dict(classes.labels), clf, None, n_patches=N_PATCHES,
                                   seed_key=SEED_KEY, tta=tta)
            row[arm] = e["ranked"][0][0] if "ranked" in e else None
            row[f"{arm}_p"] = round(e["ranked"][0][2], 4) if "ranked" in e else None
        rows.append(row)
        print(f"  {i}/{len(held)} {name}: {truth} -> release {row['release']}, candidate {row['candidate']}",
              flush=True)

    lines = [f"Photos of {', '.join(SESSIONS)} in the pool and in neither model's train split: {len(rows)}",
             "(by session: " + ", ".join(f"{s} {sum(r['session'] == s for r in rows)}" for s in SESSIONS) + ")",
             f"Serving path at the store's canonical photo, {N_PATCHES} patches, top-1, no OOD guard.\n"]
    for arm, (_, _, _, exp) in arms.items():
        kept = [r for r in rows if r[arm] is not None]
        y, p = [r["true"] for r in kept], [r[arm] for r in kept]
        cm = confusion_matrix(y, p, labels=keys)
        present = [k for k in keys if k in y]
        f1 = f1_score(y, p, labels=present, average=None, zero_division=0)
        acc = float(np.mean([a == b for a, b in zip(y, p)]))
        title = f"{arm} ({Path(MODELS[arm][0]).stem}, exp{exp})"
        lines.append(f"{title}: accuracy {acc:.4f} ({sum(a == b for a, b in zip(y, p))}/{len(kept)}), "
                     f"macro-F1 over the {len(present)} classes present {f1.mean():.4f}; "
                     f"{len(rows) - len(kept)} unmeasurable")
        width = max(len(k) for k in keys) + 2
        lines.append(" " * width + "".join(f"{k[:6]:>7s}" for k in keys) + "   (rows true, columns predicted)")
        for k, r in zip(keys, cm):
            lines.append(f"{k:<{width}s}" + "".join(f"{v:>7d}" for v in r))
        lines.append("per-class F1: " + ", ".join(f"{k} {v:.3f}" for k, v in zip(present, f1)) + "\n")
        plot_confusion_matrix(cm.tolist(), keys, dict(classes.labels), args.out / f"folders_eval_{arm}.png",
                              f"{title}\n{len(kept)} photos of {', '.join(SESSIONS)}, held out from both")
    differ = [r for r in rows if r["release"] != r["candidate"]]
    lines.append(f"Photos the two answer differently: {len(differ)}")
    lines += [f"  {r['photo']} ({r['session']}): true {r['true']}; release {r['release']} ({r['release_p']}), "
              f"candidate {r['candidate']} ({r['candidate_p']})" for r in differ]
    text = "\n".join(lines) + "\n"
    (args.out / "folders_eval.txt").write_text(text)
    (args.out / "folders_eval.json").write_text(json.dumps(rows, indent=1) + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
