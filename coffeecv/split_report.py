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

Since ticket ML-3 a class can pool several class folders (a country's coffees, D7), so a second table
counts each coffee's photos per split, and the coffees a split left out are listed: the farm-region
diagnostic has nothing for them there (R3).

Starved cells are information. A class with no photos in any dir is an error: exit 1.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from coffeecv.class_list import ClassList, load_classes, read_coffees
from coffeecv.config import RunConfig
from coffeecv.dataset import (Capture, pooled_class_photos, resolve_captures, split_census,
                              split_photos_by_class)

SPLITS = ("train", "val", "test")


def starved_cells(census: dict[str, dict]) -> list[tuple[str, str, str]]:
    """(capture, class_id, split) for every dir that has photos of a class but none in that split."""
    return [(capture, cid, split)
            for cid, entry in census.items() for split in SPLITS
            for capture, n in entry.get(split, {}).items() if n == 0]


def coffee_census(captures: list[Capture], classes: ClassList, seed: int,
                  photo_frac: dict[str, float]) -> dict[str, dict]:
    """`{folder_id: {"class": key, "pooled": n, "train"|"val"|"test": n}}`: each coffee's share of its
    class's pooled split. The split is the class's (split_photos_by_class), so this only counts it."""
    out: dict[str, dict] = {}
    for class_idx, key in enumerate(classes.keys):
        pool, _ = pooled_class_photos(captures, classes.folders[key])
        for f in classes.folders[key]:
            out[f] = {"class": key, "pooled": sum(p.folder_id == f for p in pool)}
        if pool:
            for split, chosen in split_photos_by_class(pool, seed, class_idx, photo_frac).items():
                for f in classes.folders[key]:
                    out[f][split] = sum(p.folder_id == f for p in chosen)
    return out


def print_coffees(coffees: dict[str, dict], labels: dict[str, str]) -> list[tuple[str, str]]:
    """Print the per-coffee table; return (folder_id, split) for each coffee with photos but none in val
    or test."""
    print(f"\n{'coffee':<34} {'class':<10} {'pooled':>6} {'train':>5} {'val':>4} {'test':>4}")
    missing = []
    for f, e in coffees.items():
        name = f"{f} {labels.get(f, '?')}"[:33]
        cells = [e.get(s, 0) for s in SPLITS]
        print(f"{name:<34} {e['class']:<10} {e['pooled']:>6} {cells[0]:>5} {cells[1]:>4} {cells[2]:>4}")
        if e["pooled"]:
            missing += [(f, s) for s in ("val", "test") if not e.get(s)]
    return missing


def print_census(census: dict[str, dict], labels: dict[str, str], captures: list[str]) -> None:
    w = max(7, *(len(c.removeprefix("cam_")) for c in captures))
    dirs = " ".join(f"{c.removeprefix('cam_'):>{w}}" for c in captures)
    head = f"{'class':<26} {'pooled':>6} {'train':>5} {'val':>4} {'test':>4}"
    print(f"\n{head} | val by dir:  {dirs} | test by dir: {dirs}")
    print("-" * (len(head) + 2 * (16 + (w + 1) * len(captures))))
    for cid, entry in census.items():
        label = labels.get(cid, "?")
        name = (cid if label == cid else f"{cid} {label}")[:25]
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
    capture_dirs, classes_file = cfg.resolve_paths()
    captures = resolve_captures(capture_dirs)
    classes = load_classes(classes_file)
    frac = {"train": cfg.train_photo_frac, "val": cfg.val_photo_frac, "test": cfg.test_photo_frac}
    census = split_census(captures, classes, cfg.seed, frac)

    print(f"pooled split, seed {cfg.seed}, fractions {frac['train']}/{frac['val']}/{frac['test']}, "
          f"over {[c.name for c in captures]}")
    print_census(census, dict(classes.labels), [c.name for c in captures])
    starved = starved_cells(census)
    if starved:
        print(f"\n{len(starved)} dir x class x split cell(s) starved:")
        for capture, cid, split in starved:
            print(f"  {capture:<14} class {cid}  {split}")
    else:
        print("\nno dir is starved out of any split for any class it has")

    coffees = coffee_census(captures, classes, cfg.seed, frac)
    missing = print_coffees(coffees, {c.folder_id: c.label for c in read_coffees(classes_file)})
    if missing:
        print(f"\n{len(missing)} coffee x split cell(s) with no photos (D7: by chance, accepted):")
        for f, split in missing:
            print(f"  {f} ({coffees[f]['class']})  {split}")
    else:
        print("\nevery coffee has photos in val and in test")

    empty = [cid for cid, entry in census.items() if not entry["pooled"]]
    if empty:
        print(f"\nFAIL: class(es) {empty} have no photos in any dir; training would refuse to start")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
