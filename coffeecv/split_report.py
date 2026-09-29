"""Print the pooled photo split: per class, how many photos each split gets and which capture dirs they
came from. Photo counts only -- nothing is decoded, so this runs in seconds.

Since ticket ML-1 (D1(b)) photos are pooled across capture dirs and split 70/15/15 within each class
only. Nothing guarantees a dir a share of val or test any more, so a dir with few photos of a class can
be left out of a split by chance. That is accepted, but it should be seen before a run rather than
noticed after one: this prints every split's per-dir counts and lists each (dir, class, split) that got
zero. `MultiPhotoPatchDataset` warns about the same cases at construction time, and `train_baseline`
records them in metrics.json under `split_starved`.

    python -m coffeecv.split_report                 # params.yaml's dirs, seed and fractions
    python -m coffeecv.split_report --seed 123      # another seed's partition

Informational: exits 0 whether or not anything is starved.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from coffeecv.config import RunConfig
from coffeecv.dataset import load_class_labels, resolve_captures, split_census

SPLITS = ("train", "val", "test")


def starved_cells(census: dict[str, dict]) -> list[tuple[str, str, str]]:
    """(capture, class_id, split) for every dir that has photos of a class but none in that split."""
    return [(capture, cid, split)
            for cid, entry in census.items() for split in SPLITS
            for capture, n in entry.get(split, {}).items() if n == 0]


def print_census(census: dict[str, dict], labels: dict[str, str], captures: list[str]) -> None:
    w = max(7, *(len(c.removeprefix("cam_")) for c in captures))
    dirs = " ".join(f"{c.removeprefix('cam_'):>{w}}" for c in captures)
    head = f"{'class':<26} {'pooled':>6} {'train':>5} {'val':>4} {'test':>4}"
    print(f"\n{head} | val by dir:  {dirs} | test by dir: {dirs}")
    print("-" * (len(head) + 2 * (16 + (w + 1) * len(captures))))
    for cid, entry in census.items():
        name = f"{cid} {labels.get(cid, '?')}"[:25]
        if not entry["pooled"]:
            print(f"{name:<26} {0:>6}  (no photos in any dir)")
            continue
        totals = [sum(entry[s].values()) for s in SPLITS]

        def cells(split: str) -> str:
            out = []
            for c in captures:
                n = entry[split].get(c)
                out.append(f"{'-':>{w}}" if n is None else f"{str(n) + (' !' if n == 0 else ''):>{w}}")
            return " ".join(out)

        print(f"{name:<26} {entry['pooled']:>6} {totals[0]:>5} {totals[1]:>4} {totals[2]:>4}"
              f" | {'':12}{cells('val')} | {'':12} {cells('test')}")
    print("\n'-' = the dir has no photos of that class at all; '0 !' = it has some, but the pooled "
          "split put none in that split")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None, help="default: params.yaml's seed")
    args = ap.parse_args()

    cfg = RunConfig.from_params_yaml()
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    capture_dirs, _heldout, classes_file = cfg.resolve_paths()
    captures = resolve_captures(capture_dirs)
    labels = load_class_labels(classes_file)
    frac = {"train": cfg.train_photo_frac, "val": cfg.val_photo_frac, "test": cfg.test_photo_frac}
    census = split_census(captures, sorted(labels), cfg.seed, frac)

    print(f"pooled split, seed {cfg.seed}, fractions {frac['train']}/{frac['val']}/{frac['test']}, "
          f"over {[c.name for c in captures]}")
    print_census(census, labels, [c.name for c in captures])
    starved = starved_cells(census)
    if starved:
        print(f"\n{len(starved)} dir x class x split cell(s) starved:")
        for capture, cid, split in starved:
            print(f"  {capture:<14} class_{cid}  {split}")
    else:
        print("\nno dir is starved out of any split for any class it has")
    return 0


if __name__ == "__main__":
    sys.exit(main())
