"""Run a release's own code on fixed photos -- the deploy's measuring instrument (scripts/deploy_webapp.sh).

Run from inside a release directory (its cwd), with whichever interpreter is under test:

    python -B release_probe.py smoke PHOTO          # /classify, /crop, /preview via Flask's test client
    python -B release_probe.py classify --list fixtures.txt --role compare --dir PHOTOS_DIR
                                                    # classify_one's full entry per photo, for --compare
    python -B release_probe.py classify PHOTO...    # the same, on photos named directly

Each prints one line, `PROBE <json>`. The model comes from COFFEE_CV_CHECKPOINT (release.env) and the
request log goes to COFFEE_CV_LOG_DIR, exactly as in the service. This file is deploy tooling and is
NOT part of the release; it refuses to run unless coffeecv and webapp were imported from the cwd.

Locally, two outputs are compared with:

    python release_probe.py compare-smoke EXPECTED.json SMOKE_OUTPUT.txt
    python release_probe.py compare-classify A.json B.json [--tol 1e-4]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

SCORE_TOL = 1e-4   # float32 results differ slightly between machines (docs/ops1_release_isolation_plan.html §4.8)
# The /crop mask's least IoU with the local one (ticket OPS-6 D10, R2). Masks differ across CPUs in about 1e-6
# of their pixels (ML-2 P0); on the smoke photo's mask a one-pixel shift gives 0.993-0.996, 97 flipped
# boundary pixels 0.9997 (OPS-6 P1).
MASK_IOU_MIN = 0.999


def _jsonable(x):
    import numpy as np

    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON-serialisable: {type(x).__name__}")


def _import_release_app():
    release = Path.cwd().resolve()
    sys.path.insert(0, str(release))
    import coffeecv
    import webapp.app as app
    for mod in (coffeecv, app):
        if not Path(mod.__file__).resolve().is_relative_to(release):
            raise SystemExit(f"release_probe: {mod.__name__} came from {mod.__file__}, not from {release}")
    return app


def _emit(payload: dict) -> None:
    print("PROBE " + json.dumps(payload, default=_jsonable, sort_keys=True), flush=True)


def smoke(photo: Path) -> None:
    app = _import_release_app()
    client = app.app.test_client()
    data = photo.read_bytes()
    out = {"build": app.BUILD, "python": sys.version.split()[0]}
    for endpoint in ("/classify", "/crop", "/preview"):
        r = client.post(endpoint, data={"photo": (io.BytesIO(data), photo.name)},
                        content_type="multipart/form-data")
        key = endpoint.strip("/")
        out[key] = {"status": r.status_code, "content_type": r.content_type}
        if r.is_json:
            out[key]["body"] = r.get_json()
    _emit(out)


def classify(photos: list[Path]) -> None:
    app = _import_release_app()
    entries = {}
    for photo in photos:
        entry = app.classify_one(photo, app.cfg, app.class_ids, app.class_labels, app.model, app.head, app.ref,
                                 n_patches=app.N_PATCHES, tta=app.TTA, probe=app.probe)
        entry.pop("timing_ms", None)   # wall-clock, the one field that legitimately differs
        entries[photo.name] = entry
    import numpy
    import torch
    _emit({"build": app.BUILD, "entries": entries, "python": sys.version.split()[0],
           "numpy": numpy.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads()})


def read_fixtures(list_file: Path, role: str, photo_dir: Path) -> list[Path]:
    """The `role` photos of a fixtures.txt (`<role> <sha256> <repo path>` per line), found by basename in
    photo_dir and checked against their sha256 -- the list travels as a file, never as argv."""
    import hashlib

    photos = []
    for line in list_file.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        r, sha, path = line.split()
        if r != role:
            continue
        photo = photo_dir / Path(path).name
        if hashlib.sha256(photo.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"release_probe: {photo} does not match its sha256 in {list_file}")
        photos.append(photo)
    if not photos:
        raise SystemExit(f"release_probe: no {role!r} photos in {list_file}")
    return photos


def _probe_line(text: str) -> dict:
    lines = [ln[6:] for ln in text.splitlines() if ln.startswith("PROBE ")]
    if len(lines) != 1:
        raise SystemExit(f"expected exactly one PROBE line, found {len(lines)}")
    return json.loads(lines[0])


def _max_abs_diff(a, b, path="") -> tuple[float, list[str]]:
    """Largest numeric difference between two JSON trees, and every non-numeric mismatch."""
    if isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if isinstance(a, dict) and isinstance(b, dict):
            worst, bad = 0.0, []
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b:
                    bad.append(f"{path}/{k}: present on one side only")
                    continue
                w, bd = _max_abs_diff(a[k], b[k], f"{path}/{k}")
                worst, bad = max(worst, w), bad + bd
            return worst, bad
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            worst, bad = 0.0, []
            for i, (x, y) in enumerate(zip(a, b)):
                w, bd = _max_abs_diff(x, y, f"{path}[{i}]")
                worst, bad = max(worst, w), bad + bd
            return worst, bad
        return 0.0, ([] if a == b else [f"{path}: {a!r} != {b!r}"])
    return abs(float(a) - float(b)), []


def _mask_pixels(b64: str):
    """A /crop mask (base64 of a 1-bit palette PNG, ticket OPS-6 D7) as a bool array: bean is palette index 1."""
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64)))) == 1


def _compare_masks(expected, served) -> tuple[str, list[str]]:
    """How the served /crop mask differs from the expected one: (for the summary line, failures)."""
    import numpy as np

    if expected is None and served is None:
        return "mask null", []
    if expected is None or served is None:
        seen = "null" if served is None else "present"
        return f"mask {seen}", [f"/crop mask is {seen}, expected {'present' if served is None else 'null'}"]
    a, b = _mask_pixels(expected), _mask_pixels(served)
    if a.shape != b.shape:
        size = f"{b.shape[1]} x {b.shape[0]}, expected {a.shape[1]} x {a.shape[0]}"
        return f"mask {size}", [f"/crop mask is {size}"]
    union = np.count_nonzero(a | b)
    iou = np.count_nonzero(a & b) / union if union else 1.0
    failures = [f"/crop mask IoU {iou:.6f} < {MASK_IOU_MIN}"] if iou < MASK_IOU_MIN else []
    return f"mask IoU {iou:.6f}, {np.count_nonzero(a ^ b)} pixels differ", failures


def compare_smoke(expected_file: Path, smoke_file: Path) -> None:
    """The VM's smoke answers against the ones computed locally from the same staged tree. /crop's mask is
    compared decoded, by IoU >= MASK_IOU_MIN, since the two machines' masks differ in a few pixels; the rest of
    /crop's answer within 1e-3."""
    exp = _probe_line(expected_file.read_text())
    got = {}
    for line in smoke_file.read_text().splitlines():
        tag, _, rest = line.partition(" ")
        got[tag] = rest
    failures = []

    status, _, body = got.get("SMOKE_CLASSIFY", "").partition(" ")
    if status != "200":
        failures.append(f"/classify answered {status or 'nothing'}")
    else:
        body = json.loads(body)
        e = exp["classify"]["body"]
        if body.get("verdict") != e.get("verdict"):
            failures.append(f"/classify verdict {body.get('verdict')} != expected {e.get('verdict')}")
        er, gr = e.get("ranked") or [], body.get("ranked") or []
        if (er and gr and er[0]["id"] != gr[0]["id"]) or bool(er) != bool(gr):
            failures.append(f"/classify top-1 {gr[:1]} != expected {er[:1]}")
        worst, bad = _max_abs_diff([r["score"] for r in er], [r["score"] for r in gr])
        failures += bad
        if worst > SCORE_TOL:
            failures.append(f"/classify scores differ by up to {worst:.2e} (> {SCORE_TOL})")
        print(f"  /classify  {body.get('verdict')}, top-1 {gr[0]['id'] if gr else '-'}, "
              f"max score diff vs local {worst:.1e}")

    status, _, body = got.get("SMOKE_CROP", "").partition(" ")
    if status != "200":
        failures.append(f"/crop answered {status or 'nothing'}")
    else:
        body, e = json.loads(body), exp["crop"]["body"]
        mask = ""
        if "mask" in e and "mask" in body:   # else neither is from OPS-6 on, or _max_abs_diff names the one
            mask, bad = _compare_masks(e.pop("mask"), body.pop("mask"))
            mask, failures = f", {mask}", failures + bad
        worst, bad = _max_abs_diff(e, body)
        failures += bad
        if worst > 1e-3:
            failures.append(f"/crop box differs by up to {worst:.2e}")
        print(f"  /crop      cropped={body.get('cropped')}, max box diff vs local {worst:.1e}{mask}")

    preview = got.get("SMOKE_PREVIEW", "").split()
    if preview[:2] != ["200", "image/jpeg"]:
        failures.append(f"/preview answered {' '.join(preview) or 'nothing'}")
    else:
        print(f"  /preview   200 image/jpeg, {preview[2]} bytes")

    log = json.loads(got.get("SMOKE_LOG", "{}") or "{}")
    served = (log.get("build") or {}).get("model_sha")
    if served != exp["build"]["model_sha"]:
        failures.append(f"model_sha served {served} != expected {exp['build']['model_sha']}")
    code = ((log.get("build") or {}).get("code") or {}).get("commit")
    if code != exp["build"]["code"]["commit"]:
        failures.append(f"commit served {code} != expected {exp['build']['code']['commit']}")
    print(f"  served     model_sha {served}, commit {str(code)[:12]}")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)


def compare_classify(a_file: Path, b_file: Path, tol: float) -> None:
    a, b = _probe_line(a_file.read_text()), _probe_line(b_file.read_text())
    print(f"  A: python {a['python']}, torch {a['torch']}, numpy {a['numpy']}, {a['torch_threads']} threads")
    print(f"  B: python {b['python']}, torch {b['torch']}, numpy {b['numpy']}, {b['torch_threads']} threads")
    if a["build"]["model_sha"] != b["build"]["model_sha"]:
        raise SystemExit(f"  different models: {a['build']['model_sha']} vs {b['build']['model_sha']}")
    identical = a["entries"] == b["entries"]
    worst, bad = _max_abs_diff(a["entries"], b["entries"])
    for name in sorted(a["entries"]):
        ea, eb = a["entries"][name], b["entries"].get(name, {})
        ta = (ea.get("ranked") or [[None]])[0][0]
        tb = (eb.get("ranked") or [[None]])[0][0]
        w, _ = _max_abs_diff(ea, eb)
        print(f"  {name:32} {ea.get('verdict', '?'):32} top-1 {ta} / {tb}  max diff {w:.1e}")
    print(f"  {'BIT-IDENTICAL' if identical else 'NOT identical'}: {len(a['entries'])} photos, "
          f"max numeric diff {worst:.2e}, {len(bad)} non-numeric mismatches")
    for m in bad[:20]:
        print(f"    {m}")
    if tol == 0.0 and not identical:
        raise SystemExit(1)
    if bad or worst > tol:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke").add_argument("photo", type=Path)
    p = sub.add_parser("classify")
    p.add_argument("photos", type=Path, nargs="*")
    p.add_argument("--list", type=Path, help="a fixtures.txt to take the photos from")
    p.add_argument("--role", default="compare")
    p.add_argument("--dir", type=Path, help="where the --list photos are, by basename")
    p = sub.add_parser("compare-smoke")
    p.add_argument("expected", type=Path)
    p.add_argument("smoke", type=Path)
    p = sub.add_parser("compare-classify")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--tol", type=float, default=0.0, help="0 = must be bit-identical (the default)")
    args = ap.parse_args()
    if args.cmd in ("smoke", "classify"):
        os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        sys.dont_write_bytecode = True
    if args.cmd == "smoke":
        smoke(args.photo)
    elif args.cmd == "classify":
        if bool(args.list) == bool(args.photos):
            ap.error("classify takes either --list (with --dir) or photo paths")
        classify(read_fixtures(args.list, args.role, args.dir) if args.list else args.photos)
    elif args.cmd == "compare-smoke":
        compare_smoke(args.expected, args.smoke)
    else:
        compare_classify(args.a, args.b, args.tol)


if __name__ == "__main__":
    main()
