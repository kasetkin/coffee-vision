"""ML-29 P10 (D12 step 3): the scratch pool the gain-map arm trains on, and the paired comparison per camera (D13).

    python analysis/ml29/p10_pool.py build                   # outputs/ml29_p10/segcropped/pool
    python -m coffeecv.fit_frozen_head --seeds 42 123 7 --start-exp B                       # baseline: today's pool
    python -m coffeecv.fit_frozen_head --seeds 42 123 7 --start-exp G \\
        --pool outputs/ml29_p10/segcropped/pool --label ml29gm                              # gain maps applied
    python -m coffeecv.compare_experiments G G+1 G+2 --vs B B+1 B+2                         # overall, per class
    python analysis/ml29/p10_pool.py compare --gain G G+1 G+2 --base B B+1 B+2              # per camera

- build: data/segcropped/pool as it is, with each gain-map photo's crop and mask replaced by P9's scratch crop
  (outputs/ml29_p9/crops/scratch: the shipped segmenter on P8's scratch canonical photo, as the segcrop stage crops).
  The swapped crops are hard links to P9's, every other file a symbolic link to the tracked pool's, so the scratch
  pool costs no room of its own and holds the same photos under the same names: fit_frozen_head draws the same
  split, and the two arms differ only in the gain-map photos' pixels and masks. The pool is named `pool`, as the
  split keys on the capture dir's name. Fails if a gain-map photo of the pool's sessions has no scratch crop, or a
  scratch crop matches no crop of the pool and is not on its exclude list. Writes OUT/pool_manifest.json: each
  swapped crop's sha256 and mask's, the D18 fallbacks that change, and the tree's hash as fit_frozen_head versions it.
- compare: per seed and arm, the test (and val) patch accuracy and macro-F1 by camera, from each run's
  predictions files, the camera from the scan (analysis/ml29/scan_dataset.py); the paired delta per seed (gain maps
  minus baseline) and its range. Cameras with no gain map are the arms' control: their photos are the same crops.
  Writes analysis/ml29/p10_compare.txt.

The pool is outside DVC (the plan's P8 to P10 run beside the store until P11); the runs record it by its files'
hash in their resume fingerprint and by its path in config.json, and the manifest, committed, lists every swapped
crop by hash.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "ml29"))

import p8_scratch as p8  # noqa: E402
import p9_models as p9  # noqa: E402

from coffeecv import crop_io  # noqa: E402

TRACKED = REPO / "data" / "segcropped" / "pool"
OUT = REPO / "outputs" / "ml29_p10"
MANIFEST_COPY = REPO / "analysis" / "ml29" / "p10_pool_manifest.json"
COMPARE = REPO / "analysis" / "ml29" / "p10_compare.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_key(class_dir: str) -> str:
    """`class_013` of `class_013__Brazil_Decaff`: merge_rig names the pool's class dir after the first session that
    holds the class, so a session naming it otherwise (class_013__Brazil_SulDeMinas) is matched by its id."""
    return class_dir.split("__")[0]


def scratch_index(scratch_root: Path, sessions: set[str]) -> dict[tuple[str, str], Path]:
    """(class id, crop name) -> the scratch crop, for the photos of the sessions the pool merges."""
    out = {}
    for crop in sorted(scratch_root.glob(f"*/class_*/*{crop_io.CROP_SUFFIX}")):
        session = crop.relative_to(scratch_root).parts[0]
        if session in sessions:
            key = (class_key(crop.parent.name), crop.name)
            if key in out:
                raise ValueError(f"two scratch crops for {key}: {out[key]} and {crop}")
            out[key] = crop
    return out


def build_pool(tracked: Path, scratch: dict[tuple[str, str], Path], pool: Path) -> dict:
    """`pool`: `tracked` with the crops in `scratch` (and their masks) swapped in, every other file a symbolic link.
    Returns the manifest. Crops match by class id and file name, as merge_rig merges them; a scratch crop whose
    photo is not in the tracked pool (excluded from the merge) is not used."""
    if pool.name != tracked.name:
        raise ValueError(f"the scratch pool must be named {tracked.name!r}, as the split keys on it: {pool}")
    shutil.rmtree(pool, ignore_errors=True)
    swapped, linked = [], 0
    for f in sorted(tracked.rglob("*")):
        if f.is_dir():
            continue
        rel = f.relative_to(tracked)
        dst = pool / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.name.endswith(crop_io.MASK_SUFFIX):
            crop_key = (class_key(rel.parent.name), f.name[:-len(crop_io.MASK_SUFFIX)] + crop_io.CROP_SUFFIX)
            if crop_key in scratch:
                continue                      # the scratch crop brings its own mask, or none on a fallback
        key = (class_key(rel.parent.name), f.name)
        if key in scratch:
            src = scratch[key]
            os.link(src, dst)                 # a hard link: the same bytes, no second copy on a nearly full disk
            mask_src, mask_dst = crop_io.mask_path(src), crop_io.mask_path(dst)
            if mask_src.exists():
                os.link(mask_src, mask_dst)
            today_mask = crop_io.mask_path(f)
            swapped.append({"crop": rel.as_posix(), "sha256": sha256(dst), "today_sha256": sha256(f),
                            "mask_sha256": sha256(mask_dst) if mask_src.exists() else None,
                            "fallback_today": not today_mask.exists(), "fallback_scratch": not mask_src.exists()})
        else:
            os.symlink(os.path.relpath(f, dst.parent), dst)
            linked += 1
    in_pool = {(class_key(f.parent.name), f.name) for f in tracked.glob(f"class_*/*{crop_io.CROP_SUFFIX}")}
    unused = sorted(f"{c}/{n}" for c, n in scratch if (c, n) not in in_pool)
    return {"tracked": str(tracked.relative_to(REPO)), "pool": str(pool.relative_to(REPO)),
            "swapped": len(swapped), "linked_files": linked, "unused_scratch_crops": unused,
            "fallbacks_changed": [s["crop"] for s in swapped if s["fallback_today"] != s["fallback_scratch"]],
            "crops": swapped}


def run_build(a: argparse.Namespace) -> None:
    from coffeecv.repo_files import tree_sha256
    merged = json.loads((TRACKED / "merge_manifest.json").read_text())
    sessions = set(merged["sessions"])
    excluded = sorted(f"{class_key(Path(e).parent.name)}/{crop_io.crop_name(Path(e).stem)}"
                      for e in merged["excluded"])
    scratch = scratch_index(a.p9_out / "crops" / "scratch", sessions)
    wanted = [p for p in p9.photos(a.p8_out / "canonical.jsonl")
              if p.relative_to(p8.DATASET).parts[0] in sessions]
    missing = [str(p) for p in wanted if (class_key(p.parent.name), crop_io.crop_name(p.stem)) not in scratch]
    if missing:
        raise SystemExit(f"{len(missing)} gain-map photos of the pool's sessions have no scratch crop yet "
                         f"(p9_models.py crop), e.g. {missing[:3]}")
    pool = a.out / "segcropped" / "pool"
    manifest = build_pool(TRACKED, scratch, pool)
    stray = sorted(set(manifest["unused_scratch_crops"]) - set(excluded))
    if stray:
        raise SystemExit(f"{len(stray)} scratch crops of the pool's sessions match no crop in it and are not on its "
                         f"exclude list, e.g. {stray[:3]}")
    manifest["tree_sha256"] = tree_sha256(pool)
    manifest["tracked_md5_in_dvc_lock"] = _lock_md5()
    text = json.dumps(manifest, indent=1) + "\n"
    (a.out / "pool_manifest.json").write_text(text)
    MANIFEST_COPY.write_text(text)
    print(f"{manifest['pool']}: {manifest['swapped']} crops swapped in, {manifest['linked_files']} files linked; "
          f"{len(manifest["unused_scratch_crops"])} scratch crops on the pool's exclude list; "
          f"{len(manifest['fallbacks_changed'])} D18 fallbacks changed; tree {manifest['tree_sha256'][:16]}")


def _lock_md5() -> str:
    import yaml
    lock = yaml.safe_load((REPO / "dvc.lock").read_text())
    (out,) = lock["stages"]["merge_segpool"]["outs"]
    return out["md5"]


# ----- compare -----

def run_predictions(exp: int, split: str) -> list[dict]:
    (run,) = (REPO / "experiments").glob(f"exp{exp}__*")
    with (run / f"predictions_{split}.csv").open() as f:
        return list(csv.DictReader(f))


def camera_of_crop(scan_rows: list[dict]) -> dict[tuple[str, str], str]:
    """(class folder id, crop name) -> camera, for every photo of the scan in a class folder."""
    out = {}
    for r in scan_rows:
        parts = Path(r["file"]).relative_to(p8.DATASET).parts
        if len(parts) >= 3 and parts[1].startswith("class_"):
            out[(parts[1].split("__")[0].removeprefix("class_"), crop_io.crop_name(Path(r["file"]).stem))] = \
                p8.camera(r)
    return out


def scores(rows: list[dict]) -> dict:
    """Patch accuracy and macro-F1 over the classes present."""
    from sklearn.metrics import f1_score
    y, p = [r["true_label"] for r in rows], [r["pred_label"] for r in rows]
    present = sorted(set(y))
    return {"n": len(rows), "acc": float(np.mean([a == b for a, b in zip(y, p)])),
            "macro_f1": float(f1_score(y, p, labels=present, average="macro", zero_division=0))}


def by_camera(rows: list[dict], cams: dict[tuple[str, str], str], gain_cams: set[str]) -> dict[str, dict]:
    groups = collections.defaultdict(list)
    for r in rows:
        cam = cams.get((r["folder_id"], r["photo"]), "unknown")
        groups[cam].append(r)
        groups["all cameras with gain maps" if cam in gain_cams else "all cameras without"].append(r)
    return {k: scores(v) for k, v in groups.items()}


def run_compare(a: argparse.Namespace) -> None:
    scan = p8.read_scan(a.scan)
    cams = camera_of_crop(scan)
    gain_cams = {p8.camera(r) for r in scan if r["gain_map"]}
    lines = [f"ML-29 P10: gain maps applied (exp {', '.join(map(str, a.gain))}) against the baseline "
             f"(exp {', '.join(map(str, a.base))}), paired by seed; patch accuracy and macro-F1 over the classes "
             "present, per camera. Generated by analysis/ml29/p10_pool.py compare.", ""]
    for split in ("test", "val"):
        table = collections.defaultdict(dict)        # camera -> seed -> (base, gain)
        for g, b in zip(a.gain, a.base):
            gs = by_camera(run_predictions(g, split), cams, gain_cams)
            bs = by_camera(run_predictions(b, split), cams, gain_cams)
            for cam in sorted(set(gs) | set(bs)):
                table[cam][g] = (bs.get(cam), gs.get(cam))
        lines.append(f"{split}: per camera, macro-F1 baseline -> gain maps per seed pair; the delta's mean and range")
        for cam in sorted(table, key=lambda c: (not c.startswith("all"), c)):
            cells, deltas, n = [], [], None
            for g, (bv, gv) in table[cam].items():
                if bv is None or gv is None:
                    cells.append("-")
                    continue
                n = bv["n"]
                deltas.append(gv["macro_f1"] - bv["macro_f1"])
                cells.append(f"{bv['macro_f1']:.4f}->{gv['macro_f1']:.4f}")
            rng = f"{np.mean(deltas):+.4f} [{min(deltas):+.4f}, {max(deltas):+.4f}]" if deltas else "-"
            lines.append(f"  {cam[:30]:30s} n={n or 0:5d}  " + "  ".join(cells) + f"   delta {rng}")
        lines.append("")
    text = "\n".join(lines)
    COMPARE.write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "compare"))
    ap.add_argument("--scan", type=Path, default=p8.SCAN)
    ap.add_argument("--p8-out", type=Path, default=p8.OUT)
    ap.add_argument("--p9-out", type=Path, default=p9.OUT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--gain", type=int, nargs="+", help="compare: the gain-map arm's experiments, in seed order")
    ap.add_argument("--base", type=int, nargs="+", help="compare: the baseline's, in the same seed order")
    a = ap.parse_args()
    a.out = a.out.resolve()
    if a.command == "build":
        run_build(a)
    else:
        if not a.gain or not a.base or len(a.gain) != len(a.base):
            raise SystemExit("compare needs --gain and --base, one experiment per seed each, in the same order")
        run_compare(a)


if __name__ == "__main__":
    main()
