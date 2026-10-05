"""The class list: which class each raw class folder trains as, read from one classes.txt (ticket ML-3).

A class folder (`class_NNN__...`) holds the photos of one coffee. classes.txt maps each folder id to what
the model predicts. Two formats exist, and the file's format decides which classes it describes:

    D15 (ML-3)  "id;Country[,Region[,Subregion]];misc"   one class per country (D1, D2): the key is the
                                                         country, keys in order of first appearance by id,
                                                         the label is the key
    old         "id;label"                               one class per folder id, sorted, label as written

So every shipped model's frozen `.classes.txt` (old format) keeps reading as it was fitted, and a model
fitted on a D15 file serves countries. A file is all one format or an error: a reader cannot quietly take
one for the other. Stdlib only, so coverage_report and the release can import it without torch.

    load_classes(path)    the classes a model is fitted on and predicts: country or folder, by format
    folder_classes(path)  one class per folder id, sorted, either format: for consumers that must keep
                          per-folder classes (seg_lists' ML-2 split, the DINOv3 fixture, pick_photos)
    read_coffees(path)    one Coffee per line, for the reports and the farm-region diagnostics
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ID_RE = re.compile(r"^\d+$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class Coffee:
    """One line of classes.txt: a class folder id and what is known about its coffee.

    `country`, `farm_region` and `misc` are set for a D15 line (`farm_region` is "Country,Region[,Sub]",
    or None where only the country is known; `misc` is None when empty) and are all None for an old one,
    whose `label` is the only thing it says. `label` is the line's text after "id;", verbatim."""
    folder_id: str
    label: str
    country: str | None = None
    farm_region: str | None = None
    misc: str | None = None


@dataclass(frozen=True)
class ClassList:
    """What a model predicts. `keys[i]` is output index i; `labels[key]` is its display text; `folders[key]`
    are the class folder ids whose photos train as that key, sorted."""
    keys: tuple[str, ...]
    labels: dict[str, str]
    folders: dict[str, tuple[str, ...]]

    def __post_init__(self):
        if len(set(self.keys)) != len(self.keys):
            raise ValueError(f"duplicate class keys {list(self.keys)}")
        if set(self.labels) != set(self.keys) or set(self.folders) != set(self.keys):
            raise ValueError("labels and folders must name exactly the class keys")
        owners = [f for k in self.keys for f in self.folders[k]]
        if len(set(owners)) != len(owners):
            raise ValueError(f"a class folder belongs to two classes: {owners}")

    def __len__(self) -> int:
        return len(self.keys)

    def index(self, key: str) -> int:
        return self.keys.index(key)

    def key_of(self, folder_id: str) -> str:
        for k in self.keys:
            if folder_id in self.folders[k]:
                return k
        raise KeyError(f"class folder {folder_id} is in no class of this list")

    def select(self, keys) -> ClassList:
        """The same classes, only `keys`, in that order: a caller that sees only some folders on disk
        (one pool, as the DINOv3 fixture does) indexes them as it always has."""
        keys = tuple(keys)
        return ClassList(keys, {k: self.labels[k] for k in keys}, {k: self.folders[k] for k in keys})


def _lines(path: Path) -> list[tuple[int, str, str]]:
    """(line number, id, rest) of every non-blank line. Strict: no line without ';', no repeated id."""
    out, seen = [], set()
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if ";" not in line:
            raise ValueError(f"{path}:{n}: no ';' in {line!r}")
        cid, rest = (s.strip() for s in line.split(";", 1))
        if not _ID_RE.match(cid):
            raise ValueError(f"{path}:{n}: class folder id {cid!r} is not a number")
        if cid in seen:
            raise ValueError(f"{path}:{n}: class folder id {cid} repeats")
        seen.add(cid)
        out.append((n, cid, rest))
    if not out:
        raise ValueError(f"{path} lists no classes")
    return out


def read_coffees(path: Path) -> list[Coffee]:
    """Every line of `path` as a Coffee, in file order. A D15 line has a second ';' (misc may be empty);
    an old line has none. A file mixing the two is refused."""
    lines = _lines(path)
    d15 = [";" in rest for _, _, rest in lines]
    if any(d15) and not all(d15):
        mixed = [f"{n}" for (n, _, _), is_d15 in zip(lines, d15) if not is_d15]
        raise ValueError(f"{path} mixes the D15 format (id;Country[,Region];misc) with the old one "
                         f"(id;label): line(s) {', '.join(mixed)} have one ';'")
    if not all(d15):
        return [Coffee(cid, rest) for _, cid, rest in lines]
    coffees = []
    for n, cid, rest in lines:
        if not rest.isascii():
            raise ValueError(f"{path}:{n}: not ASCII: {rest!r}")
        where, misc = (s.strip() for s in rest.split(";", 1))
        if ";" in misc:
            raise ValueError(f"{path}:{n}: more than two ';'")
        parts = [s.strip() for s in where.split(",")]
        if len(parts) > 3 or not all(_NAME_RE.match(p) for p in parts):
            raise ValueError(f"{path}:{n}: want Country[,Region[,Subregion]], got {where!r}")
        if misc and not _NAME_RE.match(misc):
            raise ValueError(f"{path}:{n}: misc {misc!r} is not one word")
        coffees.append(Coffee(cid, rest, country=parts[0],
                              farm_region=",".join(parts) if len(parts) > 1 else None, misc=misc or None))
    return coffees


def load_classes(path: Path) -> ClassList:
    """The classes a model fitted on `path` predicts: countries for a D15 file, folder ids for an old one."""
    coffees = read_coffees(path)
    if coffees[0].country is None:
        return folder_classes(path)
    keys: list[str] = []
    folders: dict[str, list[str]] = {}
    for c in sorted(coffees, key=lambda c: int(c.folder_id)):
        if c.country not in folders:
            keys.append(c.country)
            folders[c.country] = []
        folders[c.country].append(c.folder_id)
    return ClassList(tuple(keys), {k: k for k in keys}, {k: tuple(sorted(v)) for k, v in folders.items()})


def folder_classes(path: Path) -> ClassList:
    """One class per folder id, sorted, from either format: what every model before ML-3 was fitted on."""
    coffees = sorted(read_coffees(path), key=lambda c: c.folder_id)
    return ClassList(tuple(c.folder_id for c in coffees), {c.folder_id: c.label for c in coffees},
                     {c.folder_id: (c.folder_id,) for c in coffees})
