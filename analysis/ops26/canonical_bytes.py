"""OPS-26 R2/R3: does Pillow 13.0.0 canonicalize the same originals to the same bytes as Pillow 12.3.0?

    <venv>/bin/python analysis/ops26/canonical_bytes.py run OUT.json [--store] PATH ...
    python analysis/ops26/canonical_bytes.py diff OLD.json NEW.json
    python analysis/ops26/canonical_bytes.py pillow-decoded DIR    # the originals under DIR Pillow decodes (R3)
    <venv>/bin/python analysis/ops26/canonical_bytes.py metadata OUT.json DIR   # what Pillow reads from headers

`run` canonicalizes each original with the store's and the front's settings (canonical.STORE_SETTINGS,
FRONT_SETTINGS) under whichever Pillow the interpreter has, and writes each output's sha256 with the Pillow version.
With --store it also reads the original's canonical photo in data/canonical and says whether the store's bytes are
the ones made now. `diff` lists every original whose bytes differ between two runs, with both sizes. Run it with
PYTHONPATH at the repo, from each Pillow version's venv (the web service's lock holds the whole photo stack).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run(paths: list[Path], store: bool) -> dict:
    import PIL

    from coffeecv import canonical
    out = {"pillow": PIL.__version__, "photos": {}}
    for path in paths:
        data = path.read_bytes()
        row = {"format": canonical.recognise(data[:canonical.HEAD_BYTES], path.name)}
        for settings in (canonical.STORE_SETTINGS, canonical.FRONT_SETTINGS):
            try:
                made = canonical.canonicalize_with(data, path.name, settings).canonical
                row[settings.name] = {"sha256": hashlib.sha256(made).hexdigest(), "bytes": len(made)}
            except canonical.Unreadable as exc:
                row[settings.name] = {"refused": str(exc)}
        if store and path.resolve().is_relative_to(canonical.DATASET.resolve()):   # the examples are not the store's
            kept = canonical.canonical_path(path)
            row["store_matches"] = (kept.exists() and
                                    hashlib.sha256(kept.read_bytes()).hexdigest() == row["store"].get("sha256"))
        out["photos"][str(path.resolve().relative_to(REPO))] = row
        print(path.name, row, file=sys.stderr, flush=True)
    return out


def diff(old: dict, new: dict) -> list[str]:
    lines = []
    for path in sorted(set(old["photos"]) | set(new["photos"])):
        a, b = old["photos"].get(path), new["photos"].get(path)
        if a is None or b is None:
            lines.append(f"{path}: only in {'new' if a is None else 'old'}")
            continue
        for key in ("store", "front"):
            if a[key] != b[key]:
                lines.append(f"{path} [{key}]: {a[key]} -> {b[key]}")
        if a.get("store_matches") != b.get("store_matches"):
            lines.append(f"{path} [store_matches]: {a.get('store_matches')} -> {b.get('store_matches')}")
    return lines


def photos(root: Path):
    """(path, its bytes, its accepted format's row) for each photo under `root` in an accepted format."""
    from coffeecv import canonical
    for path in sorted(p for p in root.rglob("*") if p.is_file() and canonical.is_photo(p)):
        data = path.read_bytes()
        fmt = canonical.ACCEPTED_FORMATS.get(canonical.recognise(data[:canonical.HEAD_BYTES], path.name))
        if fmt is not None:
            yield path, data, fmt


def metadata(root: Path) -> dict:
    """What canonicalization reads through Pillow from each original under `root` that Pillow opens (every format but
    HEIC, RAW and JPEG XL): the ICC profile's sha256, the EXIF orientation, mode and size (_pillow_metadata), and the
    colour declaration made from them. Header parsing only, so the whole store in minutes, where re-canonicalizing it
    would take hours."""
    import PIL

    from coffeecv import canonical
    out = {"pillow": PIL.__version__, "photos": {}}
    for path, data, fmt in photos(root):
        try:
            fmt.pillow_plugin             # KeyError for a format Pillow never opens (HEIC, RAW, JPEG XL)
        except KeyError:
            continue
        try:
            icc, orientation, mode, size = canonical._pillow_metadata(data, fmt.pillow_plugin)
            row = {"icc": icc and hashlib.sha256(icc).hexdigest(), "orientation": orientation, "mode": mode,
                   "size": list(size),
                   "declaration": hashlib.sha256(repr(canonical._declare(data, fmt.id, icc)).encode()).hexdigest()}
        except Exception as exc:   # a refusal is a result too: compared across versions like any other
            row = {"error": f"{type(exc).__name__}: {exc}"}
        out["photos"][str(path.resolve().relative_to(REPO))] = row
    return out


def pillow_decoded(root: Path) -> list[Path]:
    """The originals whose pixels Pillow decodes: accepted_formats.csv's pillow.<FORMAT> rows."""
    return [path for path, _, fmt in photos(root) if fmt.decoder.startswith("pillow.")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out", type=Path)
    r.add_argument("--store", action="store_true")
    r.add_argument("paths", nargs="+", type=Path)
    d = sub.add_parser("diff")
    d.add_argument("old", type=Path)
    d.add_argument("new", type=Path)
    p = sub.add_parser("pillow-decoded")
    p.add_argument("root", type=Path)
    m = sub.add_parser("metadata")
    m.add_argument("out", type=Path)
    m.add_argument("root", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        args.out.write_text(json.dumps(run(args.paths, args.store), indent=1) + "\n")
    elif args.cmd == "metadata":
        args.out.write_text(json.dumps(metadata(args.root), indent=1) + "\n")
    elif args.cmd == "diff":
        old, new = (json.loads(p.read_text()) for p in (args.old, args.new))
        lines = diff(old, new) if "store" in next(iter(old["photos"].values()), {}) else \
            [f"{k}: {old['photos'].get(k)} -> {new['photos'].get(k)}"
             for k in sorted(set(old["photos"]) | set(new["photos"])) if old["photos"].get(k) != new["photos"].get(k)]
        print(f"{old['pillow']} -> {new['pillow']}: {len(old['photos'])} photos, {len(lines)} differences")
        print("\n".join(lines))
        return 1 if lines else 0
    else:
        print("\n".join(str(p) for p in pillow_decoded(args.root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
