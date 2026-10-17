"""Run a release's own code on fixed photos -- the deploy's measuring instrument (scripts/deploy_webapp.sh).

Run from inside a release directory (its cwd), with whichever interpreter is under test:

    python -B release_probe.py smoke PHOTO          # /analyze's two parts via Flask's test client (ticket OPS-7)
    python -B release_probe.py classify --list fixtures.txt --role compare --dir PHOTOS_DIR
                                                    # classify_one's full entry per photo, for --compare
    python -B release_probe.py classify PHOTO...    # the same, on photos named directly

Each prints one line, `PROBE <json>`. The model comes from COFFEE_CV_CHECKPOINT (release.env) and the
request log goes to COFFEE_CV_LOG_DIR, exactly as in the service. This file is deploy tooling and is
NOT part of the release; it refuses to run unless coffeecv and webapp were imported from the cwd.

PyTorch's answers, the reference a release is held to (ADR 0020), come from the repo instead (cwd the repo, in the
training environment, which has torch), through the same classify_one:

    python -B release_probe.py classify-torch --checkpoint models/<name>.pt PHOTO...

Locally, two outputs are compared with:

    python release_probe.py compare-smoke EXPECTED.json SMOKE_OUTPUT.txt
    python release_probe.py compare-classify --d7 TORCH.json RELEASE.json   # ML-7 D7's bars (deploy --compare)
    python release_probe.py compare-classify A.json B.json [--tol 1e-4]     # bit for bit, or within --tol
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

# The smoke answers' largest class-probability change against the local ones: ML-7 D7's bar for an fp16 DINOv3 on the
# host's GPU (ADR 0020), the local run being on the CPU at fp32. Was 1e-4 between two CPUs, before ML-7.
SCORE_TOL = 0.05
# --compare against PyTorch (ADR 0020, ML-7 D7): no verdict change; top-1 changes on at most 1% of the photos; the
# p95 of each photo's largest class-probability change at most 0.05.
D7_TOP1_SHARE = 0.01
D7_PROB_P95 = 0.05
# The preview mask's least IoU with the local one (ticket OPS-6 D10, R2). Masks differ across CPUs in about 1e-6
# of their pixels (ML-2 P0); on the smoke photo's mask a one-pixel shift gives 0.993-0.996, 97 flipped
# boundary pixels 0.9997 (OPS-6 P1). Measured on the smoke photo's mask only (768 x 1024, 41% bean): a smaller
# or sparser mask has more boundary per bean pixel, so a one-pixel shift costs it more IoU, and so do the same
# few flipped pixels. A new smoke photo needs these margins measured again on its own mask.
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
    r = app.app.test_client().post("/analyze", data={"photo": (io.BytesIO(photo.read_bytes()), photo.name)},
                                   content_type="multipart/form-data")
    body = r.get_data(as_text=True)
    parts = [json.loads(line) for line in body.splitlines()] if r.mimetype == "application/x-ndjson" else []
    _emit({"build": app.BUILD, "python": sys.version.split()[0],
           "analyze": {"status": r.status_code, "content_type": r.mimetype, "parts": parts}})


def classify(photos: list[Path]) -> None:
    app = _import_release_app()
    from coffeecv.infer import classify_one   # the release's own: _import_release_app checked where coffeecv came from
    entries = {}
    for photo in photos:
        entry = classify_one(photo, app.cfg, app.class_ids, app.class_labels, app.classifier, app.ref,
                             n_patches=app.N_PATCHES, tta=app.TTA, probe=app.probe, segmenter=app.segmenter)
        entry.pop("timing_ms", None)   # wall-clock, the one field that legitimately differs
        entries[photo.name] = entry
    import numpy
    from importlib.metadata import version
    _emit({"build": app.BUILD, "entries": entries, "python": sys.version.split()[0],
           "numpy": numpy.__version__, "ai_edge_litert": version("ai-edge-litert")})


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
    """A preview mask (base64 of a 1-bit palette PNG, ticket OPS-6 D7) as a bool array: bean is palette index 1."""
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64)))) == 1


def _compare_masks(expected, served) -> tuple[str, list[str]]:
    """How the served preview mask differs from the expected one: (for the summary line, failures)."""
    import numpy as np

    if expected is None and served is None:
        return "mask null", []
    if expected is None or served is None:
        seen = "null" if served is None else "present"
        return f"mask {seen}", [f"preview mask is {seen}, expected {'present' if served is None else 'null'}"]
    a, b = _mask_pixels(expected), _mask_pixels(served)
    if a.shape != b.shape:
        size = f"{b.shape[1]} x {b.shape[0]}, expected {a.shape[1]} x {a.shape[0]}"
        return f"mask {size}", [f"preview mask is {size}"]
    union = np.count_nonzero(a | b)
    iou = np.count_nonzero(a & b) / union if union else 1.0
    failures = [f"preview mask IoU {iou:.6f} < {MASK_IOU_MIN}"] if iou < MASK_IOU_MIN else []
    return f"mask IoU {iou:.6f}, {np.count_nonzero(a ^ b)} pixels differ", failures


def _compare_thumbnails(expected: str, served: str) -> tuple[str, list[str]]:
    """The served thumbnail against the local one: a JPEG of the same size, long side <= 1024 (OPS-7 P3). Not its
    pixels: two machines' JPEG encoders, and the decoders before them, may differ in a few levels."""
    from PIL import Image

    a, b = (Image.open(io.BytesIO(base64.b64decode(x))) for x in (expected, served))
    summary = f"thumbnail {b.size[0]} x {b.size[1]} {b.format}, {len(base64.b64decode(served))} bytes"
    if b.format != "JPEG":
        return summary, [f"preview thumbnail is a {b.format}, not a JPEG"]
    if b.size != a.size:
        return summary, [f"preview thumbnail is {b.size[0]} x {b.size[1]}, expected {a.size[0]} x {a.size[1]}"]
    if max(b.size) > 1024:
        return summary, [f"preview thumbnail's long side is {max(b.size)} > 1024"]
    return summary, []


def compare_smoke(expected_file: Path, smoke_file: Path) -> None:
    """The VM's smoke answer against the one computed locally from the same staged tree: /analyze's two parts
    (ticket OPS-7 D8). The classification by verdict, top-1 and scores within SCORE_TOL; the preview's mask
    decoded, by IoU >= MASK_IOU_MIN, since the two machines' masks differ in a few pixels; its thumbnail by format
    and size (`_compare_thumbnails`); the rest of the preview within 1e-3."""
    exp = _probe_line(expected_file.read_text())
    got, parts = {}, []
    for line in smoke_file.read_text().splitlines():
        tag, _, rest = line.partition(" ")
        if tag == "SMOKE_PART":
            parts.append(json.loads(rest))
        else:
            got[tag] = rest
    failures = []

    status, _, rest = got.get("SMOKE_ANALYZE", "").partition(" ")
    seconds, _, content_type = rest.partition(" ")
    names = [p.get("part") for p in parts]
    if (status, content_type) != ("200", "application/x-ndjson"):
        failures.append(f"/analyze answered {status or 'nothing'} {content_type}".rstrip())
    elif names != ["preview", "classification"]:
        error = next((f": {p.get('message')}" for p in parts if p.get("part") == "error"), "")
        failures.append(f"/analyze answered the parts [{', '.join(map(str, names))}], expected "
                        f"[preview, classification]{error}")
    else:
        print(f"  /analyze        200 in {seconds} s, parts preview, classification")
        preview, body = parts
        exp_preview, e = exp["analyze"]["parts"]
        if body.get("verdict") != e.get("verdict"):
            failures.append(f"classification verdict {body.get('verdict')} != expected {e.get('verdict')}")
        er, gr = e.get("ranked") or [], body.get("ranked") or []
        if (er and gr and er[0]["id"] != gr[0]["id"]) or bool(er) != bool(gr):
            failures.append(f"classification top-1 {gr[:1]} != expected {er[:1]}")
        worst, bad = _max_abs_diff([r["score"] for r in er], [r["score"] for r in gr])
        failures += bad
        if worst > SCORE_TOL:
            failures.append(f"classification scores differ by up to {worst:.2e} (> {SCORE_TOL})")
        print(f"  classification  {body.get('verdict')}, top-1 {gr[0]['id'] if gr else '-'}, "
              f"max score diff vs local {worst:.1e}")

        preview, e = dict(preview), dict(exp_preview)
        thumb_summary, bad = _compare_thumbnails(e.pop("thumbnail"), preview.pop("thumbnail"))
        failures += bad
        mask_summary, bad = _compare_masks(e.pop("mask"), preview.pop("mask"))
        failures += bad
        worst, bad = _max_abs_diff(e, preview)
        failures += bad
        if worst > 1e-3:
            failures.append(f"preview box differs by up to {worst:.2e}")
        print(f"  preview         cropped={preview.get('cropped')}, max box diff vs local {worst:.1e}, {mask_summary}, "
              f"{thumb_summary}")

    log = json.loads(got.get("SMOKE_LOG", "{}") or "{}")
    served = (log.get("build") or {}).get("model_sha")
    if served != exp["build"]["model_sha"]:
        failures.append(f"model_sha served {served} != expected {exp['build']['model_sha']}")
    code = ((log.get("build") or {}).get("code") or {}).get("commit")
    if code != exp["build"]["code"]["commit"]:
        failures.append(f"commit served {code} != expected {exp['build']['code']['commit']}")
    print(f"  served     model_sha {served}, commit {str(code)[:12]}")

    boot = json.loads(got.get("SMOKE_BOOT", "") or "{}")   # the worker's boot record (ADR 0021), when logged
    if boot:
        gpu = boot.get("gpu")
        where = "on the CPU" if gpu is None else (
            f"GPU {'proven' if gpu['ok'] else 'NOT proven'} (busy {gpu['busy_frac']}, {', '.join(gpu['libraries'])})")
        print(f"  boot       {boot.get('boot_s')} s (warm-up {boot.get('warm_up_s')} s), peak RSS "
              f"{boot.get('peak_rss_mb')} MB, {where}")
        if gpu is not None and not gpu["ok"]:
            failures.append(f"the GPU is not proven (ADR 0020): {gpu['why_not']}")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)


def classify_torch(photos: list[Path], checkpoint: Path) -> None:
    """PyTorch's answers, the reference ADR 0020 holds a release to: the repo's own coffeecv in the training
    environment (cwd the repo), through the same classify_one at the service's settings."""
    sys.path.insert(0, str(Path.cwd().resolve()))
    import numpy
    import torch

    from coffeecv import infer
    from coffeecv.class_list import load_classes
    from coffeecv.config import REPO_ROOT

    checkpoint = REPO_ROOT / checkpoint
    cfg, _ = infer.config_for_checkpoint(checkpoint, None)
    classes = load_classes(REPO_ROOT / cfg.classes_file)
    ids, labels = list(classes.keys), dict(classes.labels)
    clf = infer.TorchClassifier(*infer.load_model(checkpoint, cfg.model_name, len(ids), cfg.dropout))
    ref = infer.load_ood_reference(checkpoint, clf.embedding_dim)
    probe = infer.load_ood_probe(checkpoint, embedding_dim=clf.embedding_dim)
    seg = infer.segmenter_for(cfg) if cfg.crop_method == "segment" else None
    entries = {}
    for photo in photos:
        entry = infer.classify_one(photo, cfg, ids, labels, clf, ref, n_patches=40, tta=False, probe=probe,
                                   segmenter=seg)
        entry.pop("timing_ms", None)
        entries[photo.name] = entry
    _emit({"build": {"model_sha": infer._sha(checkpoint)}, "entries": entries, "python": sys.version.split()[0],
           "numpy": numpy.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads()})


def compare_d7(reference: dict, release: dict) -> list[str]:
    """ADR 0020's bars, the release's answers against PyTorch's: failures (empty if it passes). Prints a line per
    photo and the summary."""
    import numpy as np

    rows, failures = [], []
    for name in sorted(reference["entries"]):
        t, r = reference["entries"][name], release["entries"].get(name)
        if r is None:
            failures.append(f"{name}: no answer from the release")
            continue
        tp = {c: p for c, _, p in t.get("ranked") or []}
        rp = {c: p for c, _, p in r.get("ranked") or []}
        change = max((abs(rp.get(c, 0.0) - p) for c, p in tp.items()), default=0.0)
        top1 = ((t.get("ranked") or [[None]])[0][0], (r.get("ranked") or [[None]])[0][0])
        rows.append((name, t.get("verdict"), r.get("verdict"), top1, change))
        print(f"  {name:32} {str(r.get('verdict')):32} top-1 {top1[0]} / {top1[1]}  largest change {change:.4f}")
        if t.get("verdict") != r.get("verdict"):
            failures.append(f"{name}: verdict {r.get('verdict')} != PyTorch's {t.get('verdict')}")
        if not all(np.isfinite(list(rp.values()))):
            failures.append(f"{name}: a non-finite class probability")
    n = len(rows)
    flips = sum(a != b for *_, (a, b), _ in rows)
    p95 = float(np.percentile([c for *_, c in rows], 95)) if rows else 0.0
    if flips > int(D7_TOP1_SHARE * n):
        failures.append(f"top-1 changed on {flips} of {n} photos (bar: {D7_TOP1_SHARE:.0%})")
    if p95 > D7_PROB_P95:
        failures.append(f"p95 of the largest probability change {p95:.4f} > {D7_PROB_P95}")
    print(f"  {n} photos: {sum(a != b for _, a, b, _, _ in rows)} verdict changes, {flips} top-1 changes, "
          f"p95 largest change {p95:.4f}")
    return failures


def compare_classify(a_file: Path, b_file: Path, tol: float, d7: bool = False) -> None:
    a, b = _probe_line(a_file.read_text()), _probe_line(b_file.read_text())
    for name, r in (("A", a), ("B", b)):
        runtime = f"ai-edge-litert {r['ai_edge_litert']}" if "ai_edge_litert" in r \
            else f"torch {r.get('torch')}, {r.get('torch_threads')} threads"   # classify-torch, or a release before ML-7 P8
        print(f"  {name}: python {r['python']}, {runtime}, numpy {r['numpy']}")
    if a["build"]["model_sha"] != b["build"]["model_sha"]:
        raise SystemExit(f"  different models: {a['build']['model_sha']} vs {b['build']['model_sha']}")
    if d7:
        failures = compare_d7(a, b)
        for f in failures:
            print(f"  FAIL: {f}")
        if failures:
            raise SystemExit(1)
        return
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
    p = sub.add_parser("classify-torch", help="PyTorch's answers from the repo (cwd), the reference for --d7")
    p.add_argument("photos", type=Path, nargs="+")
    p.add_argument("--checkpoint", type=Path, required=True, help="models/<name>.pt, repo-relative")
    p = sub.add_parser("compare-smoke")
    p.add_argument("expected", type=Path)
    p.add_argument("smoke", type=Path)
    p = sub.add_parser("compare-classify")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--tol", type=float, default=0.0, help="0 = must be bit-identical (the default)")
    p.add_argument("--d7", action="store_true", help="A = PyTorch (classify-torch), B = the release: ML-7 D7's bars")
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
    elif args.cmd == "classify-torch":
        classify_torch(args.photos, args.checkpoint)
    elif args.cmd == "compare-smoke":
        compare_smoke(args.expected, args.smoke)
    else:
        compare_classify(args.a, args.b, args.tol, args.d7)


if __name__ == "__main__":
    main()
