"""Ticket ML-5 P5: the segmenter dataset, one committed append-only file (D3, D4, D11-D13, D16).

    python -m coffeecv.seg_dataset build     # add photos new to the data, top up the pool draw, write the file
    python -m coffeecv.seg_dataset counts    # per-source and per-split counts of the committed file

The file, labels/ml5/seg_dataset.yaml, lists every photo the segmenter is fine-tuned, validated and tested
on: all of dataset/segmenter_positives, as many pool photos (dataset/<session>/class_*/), all OOD negatives
and all internet positives. Its schema is `load_seg_dataset`'s, the reader every consumer imports.

  pool draw (D12)   as many pool photos as segmenter positives, skipping any byte-identical to one, over the
                    session folders in proportion to their eligible photos, at random within each folder
  groups (D13)      byte-identical files, photos from one source URL, and shots of one subject chained while
                    each is within GROUP_SECONDS of the last (`groups`, `shot_time`); a group never straddles
                    splits
  split (D13, D16)  params.yaml seg_dataset.split's train/validation/test shares within each source, and
                    within each (batch, tag) for negatives, whole groups at a time
  append-only (D11) a rebuild keeps every listed entry as it is and refuses a listed photo gone or changed;
                    photos new to the data are split on arrival, and the pool draw tops up to the new count
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.repo_files import read_csv, rel, sha256_file

SEG_DATASET_FILE = REPO_ROOT / "labels" / "ml5" / "seg_dataset.yaml"
DATASET = REPO_ROOT / "dataset"
SEGMENTER_POSITIVES = DATASET / "segmenter_positives"
NEGATIVES = DATASET / "ood_negatives"
INTERNET_POSITIVES = DATASET / "ood_positives_internet"
SOURCES = ("segmenter_positive", "pool", "negative", "internet_positive")
SPLITS = ("train", "validation", "test")
GROUP_SECONDS = 120       # D13: shots this close form one group
FIELDS = ("path", "sha256", "source", "batch", "tag", "folder", "split", "group")   # as written, in order
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NAME_TIME = re.compile(r"(?<!\d)(20\d{6})_(\d{6})")


@dataclass(frozen=True)
class Params:
    """params.yaml's seg_dataset block: the seed of the pool draw and of the split order, and the split's
    shares within each source (D13)."""
    seed: int
    split: dict

    @classmethod
    def from_config(cls, block: dict) -> "Params":
        if set(block) != {"seed", "split"}:
            raise ValueError(f"params.yaml seg_dataset: expected keys seed and split, got {sorted(block)}")
        split = {k: float(v) for k, v in block["split"].items()}
        if list(split) != list(SPLITS) or abs(sum(split.values()) - 1) > 1e-9:
            raise ValueError(f"params.yaml seg_dataset.split: {', '.join(SPLITS)} in that order, summing to 1")
        return cls(seed=int(block["seed"]), split=split)

    def as_meta(self) -> dict:
        return {"seed": self.seed, "split": dict(self.split)}


def seeded_rng(seed: int, *stream) -> np.random.Generator:
    """An RNG stream under `seed`, named by `stream` (crc32 of each part's str), so one draw never shifts
    another. The pool draw and the split here; seg_review's paired sides (D22)."""
    return np.random.default_rng([seed, *(zlib.crc32(str(s).encode()) for s in stream)])


def largest_remainder(sizes: dict, total: int) -> dict:
    """`total` split over the keys of `sizes` in proportion to them; ties break on the key."""
    n = sum(sizes.values())
    exact = {k: total * s / n for k, s in sizes.items()} if n else {k: 0 for k in sizes}
    quota = {k: int(v) for k, v in exact.items()}
    for k in sorted(sizes, key=lambda k: (-(exact[k] - quota[k]), str(k)))[:total - sum(quota.values())]:
        quota[k] += 1
    return quota


def draw_pool(eligible: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """n pool photos, allocated over their folders in proportion to each folder's eligible photos (D12),
    then drawn at random within each folder."""
    folders: dict[str, list[dict]] = defaultdict(list)
    for c in sorted(eligible, key=lambda c: c["path"]):
        folders[c["folder"]].append(c)
    if n > len(eligible):
        raise ValueError(f"the pool draw needs {n} photos, only {len(eligible)} are eligible")
    quota = largest_remainder({f: len(v) for f, v in folders.items()}, n)
    drawn = []
    for f in sorted(folders):
        drawn += [folders[f][i] for i in sorted(rng.choice(len(folders[f]), size=quota[f], replace=False))]
    return drawn


IDENTITY = ("sha256", "source", "batch", "tag", "folder")    # what a listed photo may never change


def assign(candidates: list[dict], existing: list[dict], params: Params) -> list[dict]:
    """The dataset's entries, sorted by path. `candidates` is every photo in the data (`scan`'s records);
    `existing` is the committed file's entries, which come back unchanged (D11, append-only): a listed photo
    gone from the data, or with other bytes, source, batch, tag or folder, is refused. Every non-pool
    candidate not yet listed joins, and pool photos are drawn until they match the segmenter positives in
    number (D12). A new photo whose group is already split takes that split; the other new groups are split
    per `split_groups`."""
    by_path = {c["path"]: c for c in candidates}
    for e in existing:
        c = by_path.get(e["path"])
        if c is None:
            raise ValueError(f"{e['path']} is listed but gone from the data; the file is append-only (D11)")
        changed = [k for k in IDENTITY if e.get(k) != c.get(k)]
        if changed:
            raise ValueError(f"{e['path']}: {', '.join(changed)} changed since it was listed; the file is "
                             "append-only (D11)")
    listed = {e["path"] for e in existing}
    entries = [by_path[p] for p in sorted(listed)]
    entries += [c for c in candidates if c["source"] != "pool" and c["path"] not in listed]
    positive_sha = {e["sha256"] for e in entries if e["source"] == "segmenter_positive"}
    eligible = [c for c in candidates if c["source"] == "pool" and c["path"] not in listed
                and c["sha256"] not in positive_sha]
    n_pool = sum(e["source"] == "pool" for e in entries)
    need = sum(e["source"] == "segmenter_positive" for e in entries) - n_pool
    entries += draw_pool(eligible, max(need, 0), seeded_rng(params.seed, "pool", n_pool))
    entries = sorted(entries, key=lambda e: e["path"])

    component = groups(entries)
    old_split: dict[str, str] = {}
    old_name: dict[str, str] = {}
    for e in existing:
        r = component[e["path"]]
        if old_split.setdefault(r, e["split"]) != e["split"]:
            members = sorted(p for p, q in component.items() if q == r and p not in listed)
            raise ValueError(f"new photos {members} join listed groups in two splits; splitting them would move "
                             "a listed photo (D11)")
        old_name[r] = min(old_name.get(r, e["group"]), e["group"])
    split_of = split_groups(entries, component, params, old_split)
    stored = {e["path"]: e for e in existing}
    out = []
    for e in entries:
        if e["path"] in stored:
            out.append(stored[e["path"]])
        else:
            r = component[e["path"]]
            out.append({**e, "split": split_of[e["path"]], "group": old_name.get(r, r)})
    return out


def shot_time(exif: dict, camera: str, name: str) -> tuple[str, datetime] | None:
    """(clock, time) of a shot, for the D13 grouping; only times on one clock are compared.

    1. EXIF DateTimeOriginal with OffsetTimeOriginal: ("utc", that instant). Every pool session's photos
       have both; ML-3's metadata strip keeps them (coffeecv.strip_metadata KEPT_TAGS).
    2. DateTimeOriginal without an offset: ("local <Make> <Model>", the camera's own clock).
    3. No EXIF time (a photo exported without it, or a renamed capture whose manifest keeps the camera's
       file name, passed as `name`): the YYYYMMDD_HHMMSS in the name. A Pixel names its files in UTC
       (0 s from the EXIF instant on all 622 Pixel photos with both, 2026-10-08), so `camera` "pixel" is
       ("utc", ...); any other camera names them in local time ("local <camera>"): OnePlus and Sony did
       (0 s from DateTimeOriginal), the OnePlus with a PXL_ prefix, so the prefix alone says nothing.
    4. Otherwise None: the photo groups by bytes and source URL only."""
    if exif.get("DateTimeOriginal"):
        t = datetime.strptime(exif["DateTimeOriginal"][:19], "%Y:%m:%d %H:%M:%S")
        offset = exif.get("OffsetTimeOriginal")
        if offset:
            sign = 1 if offset[0] == "+" else -1
            hours, minutes = map(int, offset[1:].split(":"))
            return "utc", t - sign * timedelta(hours=hours, minutes=minutes)
        return " ".join(["local", *(str(exif[k]) for k in ("Make", "Model") if exif.get(k))]), t
    m = _NAME_TIME.search(name)
    if m is None:
        return None
    t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return ("utc", t) if camera == "pixel" else (f"local {camera}", t)


def stratum(e: dict) -> tuple:
    """What the split is balanced within (D13): the source, and for negatives the (batch, tag) (D16)."""
    return (e["source"], e["batch"], e["tag"]) if e["source"] == "negative" else (e["source"],)


def groups(entries: list[dict]) -> dict[str, str]:
    """{path: the group's first path}. One group (D13): byte-identical files; photos from one source URL;
    and shots of one subject on one clock, chained while each is within GROUP_SECONDS of the one before.
    `time` is `shot_time`'s (clock, datetime) or None; two clocks are never compared. The subject is the
    beans for every positive source (a segmenter positive is often a pool photo's burst neighbour, on any
    camera of the rig) and each (batch, tag) for negatives: a different negative scenario, or a negative
    shot right after the beans, is a new subject (as ML-2's grouping had it)."""
    parent = {e["path"]: e["path"] for e in entries}

    def find(p: str) -> str:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first: dict[tuple, str] = {}
    clocks: dict[tuple, list[tuple]] = defaultdict(list)
    for e in entries:
        for key in (("sha256", e["sha256"]), ("url", e.get("url") or None)):
            if key[1] is not None:
                union(first.setdefault(key, e["path"]), e["path"])
        if e.get("time") is not None:
            subject = stratum(e) if e["source"] == "negative" else ("beans",)
            clocks[(subject, e["time"][0])].append((e["time"][1], e["path"]))
    for shots in clocks.values():
        shots.sort()
        for (t0, a), (t1, b) in zip(shots, shots[1:]):
            if (t1 - t0).total_seconds() <= GROUP_SECONDS:
                union(a, b)
    return {p: find(p) for p in parent}


def split_groups(entries: list[dict], group_of: dict[str, str], params: Params,
                 group_split: dict[str, str] | None = None) -> dict[str, str]:
    """{path: split}. `group_split` holds the groups already split. Within each stratum, the other groups
    are taken in a seeded random order and each goes whole to the split furthest below its share of the
    stratum (largest remainder of params.split); a group reaching into an earlier stratum is split there."""
    group_split = dict(group_split or {})
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        strata[stratum(e)].append(e)
    for key in sorted(strata):
        members = strata[key]
        target = largest_remainder({s: params.split[s] for s in SPLITS}, len(members)) if members else {}
        count = dict.fromkeys(SPLITS, 0)
        size: dict[str, int] = defaultdict(int)
        for e in members:
            g = group_of[e["path"]]
            if g in group_split:
                count[group_split[g]] += 1
            else:
                size[g] += 1
        new = sorted(size)
        for i in seeded_rng(params.seed, "split", *key).permutation(len(new)):
            g = new[i]
            s = max(SPLITS, key=lambda s: (target[s] - count[s], -SPLITS.index(s)))
            group_split[g] = s
            count[s] += size[g]
    return {e["path"]: group_split[group_of[e["path"]]] for e in entries}


# --- the file -------------------------------------------------------------------------------------

def header(params: Params) -> str:
    """The comment the file opens with, stating the shares and GROUP_SECONDS it was built with."""
    shares = "/".join(f"{round(100 * params.split[s])}" for s in SPLITS)
    return f"""\
# The segmenter dataset (ticket ML-5, D3, D11-D13, D16), written by `python -m coffeecv.seg_dataset build`.
# Append-only (D11): a rebuild adds photos new to the data and never moves or drops a listed one. Read it
# with coffeecv.seg_dataset.load_seg_dataset. source: segmenter_positive | pool | negative | internet_positive;
# split: train | validation | test, {shares} within each source (each batch and tag for negatives); group:
# byte-identical files, one source URL, and shots within {GROUP_SECONDS} s, named by their first path, never in two
# splits. meta records the seed and shares the file was built with.
"""


def load_seg_dataset(path: Path = SEG_DATASET_FILE) -> list[dict]:
    """The `photos` list of the segmenter dataset file, checked. Each entry has `path` (repo-relative,
    unique), `sha256` (64 lower-case hex of the file), `source` (one of SOURCES), `split` (one of SPLITS)
    and `group`; a negative also has `batch` (its folder under dataset/ood_negatives/) and `tag` (that
    batch manifest's scenario_tag), a pool photo its session `folder`. Other keys are kept and ignored.
    Raises ValueError on the first entry that breaks a rule; the photos themselves need not be present
    (they are DVC-tracked)."""
    return read_seg_dataset(path)[1]


def read_seg_dataset(path: Path = SEG_DATASET_FILE) -> tuple[dict, list[dict]]:
    """(meta, photos), photos checked as `load_seg_dataset` says."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    photos = raw.get("photos")
    if not isinstance(photos, list) or not photos:
        raise ValueError(f"{path}: no `photos` list")
    seen: set[str] = set()
    for i, e in enumerate(photos):
        where = f"{path}: photos[{i}] ({e.get('path') if isinstance(e, dict) else e!r})"
        if not isinstance(e, dict):
            raise ValueError(f"{where}: not a mapping")
        missing = [k for k in ("path", "sha256", "source", "split", "group") if k not in e]
        if e.get("source") == "negative":
            missing += [k for k in ("batch", "tag") if k not in e]
        if missing:
            raise ValueError(f"{where}: missing {', '.join(missing)}")
        if e["source"] not in SOURCES:
            raise ValueError(f"{where}: source {e['source']!r} is not one of {', '.join(SOURCES)}")
        if e["split"] not in SPLITS:
            raise ValueError(f"{where}: split {e['split']!r} is not one of {', '.join(SPLITS)}")
        if not _SHA256.fullmatch(str(e["sha256"])):
            raise ValueError(f"{where}: sha256 {e['sha256']!r} is not 64 lower-case hex digits")
        if e["path"] in seen:
            raise ValueError(f"{where}: path listed twice")
        seen.add(e["path"])
    return raw.get("meta") or {}, photos


def write_seg_dataset(params: Params, entries: list[dict], path: Path = SEG_DATASET_FILE) -> None:
    """The file: `header`, then meta (params and GROUP_SECONDS) and the photos' FIELDS."""
    photos = [{k: e[k] for k in FIELDS if k in e} for e in entries]
    meta = {**params.as_meta(), "group_seconds": GROUP_SECONDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header(params) + yaml.safe_dump({"meta": meta, "photos": photos}, sort_keys=False, width=120))


# --- the data -------------------------------------------------------------------------------------

def _require_full_checkout(folder: Path) -> None:
    """Refuse a folder whose files are not all in the working tree: a photo missing from a partial
    checkout would read as a photo gone from the data."""
    dvc = folder.parent / f"{folder.name}.dvc"
    want = yaml.safe_load(dvc.read_text())["outs"][0]["nfiles"]
    have = sum(1 for p in folder.rglob("*") if p.is_file())
    if have != want:
        raise FileNotFoundError(f"{rel(folder)} holds {have} of {want} files: `dvc checkout {rel(dvc)}`")


def exif_dates(paths: list[Path]) -> dict[str, dict]:
    """{repo-relative path: its DateTimeOriginal, OffsetTimeOriginal, Make and Model}, from one exiftool run."""
    if not paths:
        return {}
    out = subprocess.run(["exiftool", "-j", "-q", "-L", "-DateTimeOriginal", "-OffsetTimeOriginal", "-Make",
                          "-Model", "-@", "-"], input="\n".join(str(p) for p in paths), capture_output=True,
                         text=True, check=False)
    if out.returncode not in (0, 1) or not out.stdout.strip():           # 1: some file had none of the tags
        raise RuntimeError(f"exiftool failed: {out.stderr.strip()}")
    return {rel(Path(r["SourceFile"])): r for r in json.loads(out.stdout)}


def scan(only: set[str] | None = None) -> list[dict]:
    """Every photo that may be in the dataset, as `assign`'s candidate records: path, sha256, source,
    batch and tag (negatives), folder (pool), url (internet photos' manifest source_url, which also
    stands in for a shot time) and time (`shot_time`).

    `only` (repo-relative paths) describes just those photos, the same way, without the checks that every
    folder is checked out in full; a path that is none of the sources' photos is refused. fit_ood_probe
    uses it to keep its bean photos out of the groups of the test split (D4)."""
    from coffeecv.strip_metadata import PHOTO_EXTENSIONS

    def _require_checked_out(folder: Path) -> None:
        if only is None:
            _require_full_checkout(folder)

    def photos(folder: Path) -> list[Path]:
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS)

    recs: list[tuple[dict, str, str]] = []          # (record, camera, name for the file-name time)
    _require_checked_out(SEGMENTER_POSITIVES)
    camera = {r["filename"]: r["camera"] for r in read_csv(DATASET / "segmenter_positives.manifest.csv")}
    for p in photos(SEGMENTER_POSITIVES):
        recs.append(({"path": rel(p), "source": "segmenter_positive", "url": ""}, camera[p.name], p.name))
    for manifest in sorted(NEGATIVES.glob("*.manifest.csv")):
        batch = NEGATIVES / manifest.name.removesuffix(".manifest.csv")
        _require_checked_out(batch)
        rows = read_csv(manifest)
        unlisted = sorted(set(map(rel, photos(batch))) - {rel(batch / r["filename"]) for r in rows})
        if unlisted and only is None:
            raise ValueError(f"{rel(manifest)} does not list {unlisted}")
        for r in rows:
            recs.append(({"path": rel(batch / r["filename"]), "source": "negative", "batch": batch.name,
                          "tag": r["scenario_tag"], "url": r["source_url"]}, r["camera"],
                         "" if r["source_url"] else f"{r['source_title']} {Path(r['filename']).name}"))
    _require_checked_out(INTERNET_POSITIVES)
    for r in read_csv(DATASET / "ood_positives_internet.manifest.csv"):
        recs.append(({"path": rel(INTERNET_POSITIVES / r["filename"]), "source": "internet_positive",
                      "url": r["source_url"]}, r["camera"], ""))
    for session in sorted(d for d in DATASET.iterdir() if d.is_dir() and any(d.glob("class_*"))):
        _require_checked_out(session)
        for p in photos(session):
            if p.relative_to(session).parts[0].startswith("class_"):
                recs.append(({"path": rel(p), "source": "pool", "folder": session.name, "url": ""}, "", p.name))
    if only is not None:
        recs = [x for x in recs if x[0]["path"] in only]
        unknown = sorted(set(only) - {r["path"] for r, _, _ in recs})
        if unknown:
            raise ValueError(f"not a photo of the segmenter dataset's sources: {unknown}")
    missing = [r["path"] for r, _, _ in recs if not (REPO_ROOT / r["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"listed in a manifest but not on disk: {missing}")
    exif = exif_dates([REPO_ROOT / r["path"] for r, _, _ in recs if not r["url"]])
    out = []
    for r, cam, name in recs:
        r["sha256"] = sha256_file(REPO_ROOT / r["path"])
        r["time"] = None if r["url"] else shot_time(exif.get(r["path"], {}), cam, name)
        out.append(r)
    return out


def counts_table(entries: list[dict]) -> str:
    """Photos per source (negatives also per batch and tag) and split."""
    rows: dict[str, Counter] = defaultdict(Counter)
    for e in entries:
        rows[e["source"]][e["split"]] += 1
        if e["source"] == "negative":
            rows[f"  {e['batch']} / {e['tag']}"][e["split"]] += 1
        rows["all"][e["split"]] += 1
    order = [s for s in SOURCES if s in rows]
    order[order.index("negative") + 1:order.index("negative") + 1] = sorted(k for k in rows if k.startswith("  "))
    width = max(len(k) for k in rows)
    lines = [f"{'source':<{width}}  " + "".join(f"{s:>11}" for s in (*SPLITS, "total"))]
    for k in [*order, "all"]:
        c = rows[k]
        lines.append(f"{k:<{width}}  " + "".join(f"{c[s]:>11}" for s in SPLITS) + f"{sum(c.values()):>11}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "counts"))
    args = ap.parse_args(argv)
    if args.command == "counts":
        print(counts_table(load_seg_dataset()))
        return 0
    params = Params.from_config(RunConfig.from_params_yaml().seg_dataset)
    meta, existing = read_seg_dataset() if SEG_DATASET_FILE.exists() else ({}, [])
    built_with = {k: meta[k] for k in ("seed", "split") if k in meta}
    if existing and built_with != params.as_meta():
        print(f"refusing: the file was built with {built_with}, params.yaml says {params.as_meta()}; the file "
              "is append-only (D11)", file=sys.stderr)
        return 1
    entries = assign(scan(), existing, params)
    added = len(entries) - len(existing)
    write_seg_dataset(params, entries)
    print(f"{rel(SEG_DATASET_FILE)}: {len(entries)} photos, {added} added\n{counts_table(entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
