"""The owner's ML-5 review page (ticket ML-5 P6; D14, D19, D22, D23): labelling sessions over the segmenter
dataset with g/r point corrections, and the blind paired mode for P10.

Labelling (P7, P9):

    python -m coffeecv.seg_review label --session pass1 --proposer ft_s123 --redraw pretrained_l0
    python -m coffeecv.seg_review status --session pass1      # labels.csv and the report: by round, drops, per source

then open http://localhost:8765 (VS Code forwards the port; else --host 0.0.0.0 and the container's IP, as
review_masks says). `label` first prepares what the session needs, resumably: the proposer's round-1 mask of
every training and validation positive (whole-image box, params.yaml seg_mask_select), the empty label of
every training and validation negative (never shown, D14), and each positive's encoding by the redraw model,
stored once, so that a redraw reruns only the decoder (D19). The proposer and the redraw model are arguments
(pass 1: ft_s123 and pretrained L0, D19), recorded in the session file; a resume with other models is refused.

On the page: A accepts the mask; D (or 1-5, with a reason) declines it. After a decline, hold g and left-click
for an include point (green), r and left-click for an exclude point (red); U or Backspace undoes the last
point; Enter redraws: the redraw model draws the whole-image box plus every point so far through the multi3
output (D23), and that mask is the next round's, judged again. Until it is judged, more points and another
Enter replace it. A plain click opens the spot at full resolution; Space toggles the outline. A photo still
declined in round 3 is dropped, the reason logged (D14).

Blind paired (P10, D22):

    python -m coffeecv.seg_review paired --session p10_test --masks data/seg_masks/<v1> data/seg_masks/<v2>

Each test positive's two masks side by side, left and right drawn at random per photo from params.yaml
seg_paired_seed (or --seed), the model names hidden until every mask is judged; identical masks are judged
once. A/D judge the left mask, J/L the right one. Then the report (P10, D22):

    python -m coffeecv.seg_review paired-report --session p10_test --masks <v1 dir> <v2 dir> --out <json>

per source (internet positives with the segmenter positives) and overall: both pass, v1 only, v2 only, both
fail, with an exact McNemar p-value on the discordant pairs; and each model's empty and empty-or-tiny masks on
the test negatives (read from the same mask directories, which hold every test photo). Once the owner picks a
model (D20), `--chosen <model>` adds the re-derived tiny-mask threshold (D21).

Storage, as ML-2 D16: verdicts and their points in git, masks DVC-tracked (`dvc add data/ml5_labels/<session>`
when the session is done; .gitignore keeps them out of git from the first click).
    labels/ml5/<session>.session.json        the session's models and outputs
    labels/ml5/<session>.verdicts.jsonl      one row per click: round, mask sha256, the points that drew it
    data/ml5_labels/<session>/r<k>/<id>.png  round k's masks (1-bit) + index.csv with their points and models
    data/ml5_labels/<session>/neg/<id>.png   the empty labels of the negatives
    data/ml5_labels/<session>/labels.csv     one row per photo (`status`): accepted, dropped or pending
    labels/ml5/<session>.paired.jsonl        the paired mode's verdicts, side and model per mask
    outputs/ml5_embeddings/                  the stored encodings (not tracked; recomputable)
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import threading
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
from flask import Flask, abort, jsonify, redirect, request, send_file
from PIL import Image

from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import load_rgb_image
from coffeecv.review_masks import REASONS, RULE_FILE, jpeg, view_image, zoom_tile
from coffeecv.repo_files import read_csv, rel, sha256_file
from coffeecv.seg_dataset import load_seg_dataset, photo_id, seeded_rng
from coffeecv.seg_review_pages import LABEL_PAGE, PAIRED_PAGE
from coffeecv.segment_beans import THREADS, BeanSegmenter, mask_sha256, named_params

LABELS_DIR = REPO_ROOT / "labels" / "ml5"
MASK_ROOT = REPO_ROOT / "data" / "ml5_labels"
EMBED_ROOT = REPO_ROOT / "outputs" / "ml5_embeddings"
ROUNDS = (1, 2, 3)
REDRAW_OUTPUT = "multi3"     # D23: every redraw, in place of seg_labels.CORRECTION_OUTPUT's single
LABEL_SPLITS = ("train", "validation")
INDEX_FIELDS = ["id", "path", "photo_sha256", "mask_sha256", "height", "width", "area_frac", "pred_iou",
                "include", "exclude", "model", "output", "threads", "weights_sha256", "decoder_sha256"]
LABEL_FIELDS = ["id", "path", "photo_sha256", "source", "split", "status", "round", "mask", "mask_sha256",
                "include", "exclude", "reason"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).convert("1").save(path)


class _LRU:
    """The last `n` values loaded, by key: decoded photos and masks (a 50 MP photo takes a second or two) and
    the redraw model's stored encodings."""

    def __init__(self, n: int):
        self.n, self.d = n, OrderedDict()

    def get(self, key, load: Callable[[], object]):
        if key not in self.d:
            self.d[key] = load()
            while len(self.d) > self.n:
                self.d.popitem(last=False)
        self.d.move_to_end(key)
        return self.d[key]


# ---------------------------------------------------------------- the loop (pure)

class State(NamedTuple):
    status: str            # pending | judge | points | accepted | dropped
    round: int             # the round whose mask the page shows (0 = none yet)
    reason: str | None


def progress(masks: dict[int, str], verdicts: dict[int, dict]) -> State:
    """Where a photo is in the loop (D14). `masks` {round: mask sha256}; `verdicts` {round: the last verdict on
    that round's current mask}.

    judge k     round k's mask waits for a verdict
    points k    round k was declined: place points and redraw round k + 1
    accepted k  round k's mask is the label
    dropped 3   declined in round 3
    pending 0   no round-1 mask yet (not prepared)
    """
    if 1 not in masks:
        return State("pending", 0, None)
    for k in ROUNDS:
        if k not in masks:
            prev = verdicts[k - 1]
            return State("points", k - 1, prev.get("reason"))
        vk = verdicts.get(k)
        if vk is None:
            return State("judge", k, None)
        if vk["decision"] == "accept":
            return State("accepted", k, None)
    why = verdicts[ROUNDS[-1]].get("reason")
    return State("dropped", ROUNDS[-1], f"declined in round {ROUNDS[-1]}" + (f": {why}" if why else ""))


# ---------------------------------------------------------------- the models at the session's boundary

class SegDrawer:
    """A segmenter as the session uses it: `propose` (round 1, the serving prompt), and for the redraw model
    `prepare` (encode once and store) and `redraw` (the stored encoding, the box plus points, multi3: D23)."""

    def __init__(self, name: str, embed_root: Path = EMBED_ROOT, mask_select: str | None = None):
        self.name = name
        self.seg = BeanSegmenter(named_params(name, mask_select or RunConfig.from_params_yaml().seg_mask_select))
        self.embed_dir = embed_root / self.seg.weights_sha256[:16]
        self._loaded = _LRU(3)

    @property
    def propose_output(self) -> str:
        return self.seg.p.mask_select

    def info(self) -> dict:
        return {"model": self.name, "weights_sha256": self.seg.weights_sha256,
                "decoder_sha256": self.seg.decoder_sha256 or ""}

    def propose(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        return self.seg.predict_mask(rgb), self.seg.pred_iou

    def _file(self, item: dict) -> Path:
        return self.embed_dir / f"{item['sha256']}.pt"

    def prepare(self, item: dict, rgb: np.ndarray) -> None:
        """Encode the photo once and store it; a write cut short leaves no file under the final name."""
        f = self._file(item)
        if not f.exists():
            self.embed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.seg.encode(rgb), f.with_suffix(".part"))
            f.with_suffix(".part").rename(f)

    def needs(self, item: dict) -> bool:
        return not self._file(item).exists()

    def redraw(self, item: dict, include: list, exclude: list) -> tuple[np.ndarray, float]:
        self.seg.set_encoding(self._loaded.get(item["sha256"], lambda: torch.load(self._file(item), weights_only=True)))
        return self.seg.decode_points(include, exclude, REDRAW_OUTPUT)


# ---------------------------------------------------------------- a labelling session's files

class LabelSession:
    """A labelling session's photos, masks and verdicts. `entries` are the dataset file's photos; the session
    covers its training and validation ones. `models` ({"proposer": name, "redraw": name}) is written to the
    session file on first use and must match it after."""

    def __init__(self, name: str, entries: list[dict], models: dict | None, labels_dir: Path = LABELS_DIR,
                 mask_root: Path = MASK_ROOT):
        """`models` None reads them from the session file (a session already started)."""
        self.name = name
        mine = [{**e, "id": photo_id(e["path"])} for e in entries if e["split"] in LABEL_SPLITS]
        self.items = [e for e in mine if e["source"] != "negative"]
        self.negatives = [e for e in mine if e["source"] == "negative"]
        self.meta_path = labels_dir / f"{name}.session.json"
        self.verdicts_path = labels_dir / f"{name}.verdicts.jsonl"
        self.mask_dir = mask_root / name
        if models is None:
            if not self.meta_path.exists():
                raise ValueError(f"no session {name}: {rel(self.meta_path)} does not exist")
            models = json.loads(self.meta_path.read_text())
        meta = {"session": name, "proposer": models["proposer"], "redraw": models["redraw"],
                "redraw_output": REDRAW_OUTPUT, "splits": list(LABEL_SPLITS)}
        if self.meta_path.exists():
            old = json.loads(self.meta_path.read_text())
            diff = [k for k in meta if old.get(k) != meta[k]]
            if diff:
                raise ValueError(f"session {name} was started with " +
                                 ", ".join(f"{k} {old.get(k)!r}" for k in diff) + f", not {models}")
        else:
            labels_dir.mkdir(parents=True, exist_ok=True)
            self.meta_path.write_text(json.dumps({**meta, "created": _now()}, indent=2) + "\n")
        self._cache: dict[int, tuple] = {}

    # masks per round
    def mask_path(self, item: dict, k: int) -> Path:
        return self.mask_dir / f"r{k}" / f"{item['id']}.png"

    def _idx(self, k: int) -> dict[str, dict]:
        """Round k's index.csv, re-read when the file changes (several servers or sessions may share it)."""
        f = self.mask_dir / f"r{k}" / "index.csv"
        stamp = (f.stat().st_mtime_ns, f.stat().st_size) if f.exists() else None
        if k not in self._cache or self._cache[k][0] != stamp:
            rows = {r["id"]: r for r in read_csv(f)} if stamp else {}
            self._cache[k] = (stamp, rows)
        return self._cache[k][1]

    def _put(self, k: int, item: dict, mask: np.ndarray, iou: float, inc: list, exc: list, info: dict,
             output: str) -> None:
        _save_mask(mask, self.mask_path(item, k))
        index = dict(self._idx(k))
        index[item["id"]] = {
            "id": item["id"], "path": item["path"], "photo_sha256": item["sha256"], "mask_sha256": mask_sha256(mask),
            "height": mask.shape[0], "width": mask.shape[1], "area_frac": f"{mask.mean():.4f}",
            "pred_iou": f"{iou:.4f}", "include": json.dumps(inc), "exclude": json.dumps(exc), "model": info["model"],
            "output": output, "threads": torch.get_num_threads(), "weights_sha256": info["weights_sha256"],
            "decoder_sha256": info["decoder_sha256"]}
        f = self.mask_dir / f"r{k}" / "index.csv"
        with open(f, "w", newline="") as out:
            wr = csv.DictWriter(out, fieldnames=INDEX_FIELDS)
            wr.writeheader()
            wr.writerows(index[i] for i in sorted(index))
        self._cache[k] = ((f.stat().st_mtime_ns, f.stat().st_size), index)

    def row(self, item: dict, k: int) -> dict | None:
        return self._idx(k).get(item["id"])

    def points(self, item: dict, k: int) -> dict:
        r = self.row(item, k)
        return {"include": json.loads(r["include"]), "exclude": json.loads(r["exclude"])} if r else \
            {"include": [], "exclude": []}

    # verdicts
    def verdicts(self) -> dict[tuple[str, int], dict]:
        """{(item, round): the last verdict on that round's current mask}."""
        out = {}
        for r in _read_jsonl(self.verdicts_path):
            cur = self._idx(r["round"]).get(r["item"])
            if cur and cur["mask_sha256"] == r["mask_sha256"]:
                out[(r["item"], r["round"])] = r
        return out

    def state(self, item: dict, verdicts: dict | None = None) -> State:
        verdicts = self.verdicts() if verdicts is None else verdicts
        masks = {k: self._idx(k)[item["id"]]["mask_sha256"] for k in ROUNDS if item["id"] in self._idx(k)}
        return progress(masks, {k: verdicts[(item["id"], k)] for k in ROUNDS if (item["id"], k) in verdicts})

    def decide(self, item: dict, decision: str, reason: str | None, reviewer: str) -> State:
        st = self.state(item)
        if st.status == "pending":
            raise _Refused(409, "no mask to judge yet")
        cur = self.row(item, st.round)
        _append_jsonl(self.verdicts_path, {
            "item": item["id"], "path": item["path"], "photo_sha256": item["sha256"], "source": item["source"],
            "split": item["split"], "round": st.round, "mask_sha256": cur["mask_sha256"], "model": cur["model"],
            "output": cur["output"], **self.points(item, st.round), "decision": decision,
            "reason": None if decision == "accept" else reason, "reviewer": reviewer, "via": "seg_review",
            "ts": _now()})
        return self.state(item)

    def redraw(self, item: dict, include: list, exclude: list, drawer) -> State:
        """The next round's mask from the box plus `include`/`exclude` (every point so far), after a decline;
        or a replacement of that mask while it is not judged."""
        st = self.state(item)
        if st.status == "points":
            k, base = st.round + 1, st.round
        elif st.status == "judge" and st.round > 1:
            k, base = st.round, st.round
        else:
            raise _Refused(409, f"{st.status} in round {st.round}: decline the mask before redrawing")
        if not include and not exclude:
            raise _Refused(400, "no point")
        if {"include": include, "exclude": exclude} == self.points(item, base):
            raise _Refused(400, "no new point since this mask was drawn")
        mask, iou = drawer.redraw(item, include, exclude)
        self._put(k, item, mask, iou, include, exclude, drawer.info(), REDRAW_OUTPUT)
        return self.state(item)

    # before the session
    def prepare(self, proposer, redrawer, log: Callable[[str], None] = lambda s: None) -> None:
        """Round-1 masks by `proposer`, encodings by `redrawer`, the negatives' empty labels; resumable."""
        out_neg = self.mask_dir / "neg"
        todo = [it for it in self.items if it["id"] not in self._idx(1) or redrawer.needs(it)]
        negs = [e for e in self.negatives if not (out_neg / f"{e['id']}.png").exists()]
        t0 = time.perf_counter()
        for n, it in enumerate(todo, 1):
            rgb = self._photo(it)
            if it["id"] not in self._idx(1):
                mask, iou = proposer.propose(rgb)
                self._put(1, it, mask, iou, [], [], proposer.info(), proposer.propose_output)
            redrawer.prepare(it, rgb)
            if n % 10 == 0 or n == len(todo):
                log(f"  prepared {n}/{len(todo)} positives, {time.perf_counter() - t0:.0f} s")
        for e in negs:
            _save_mask(np.zeros(self._photo(e).shape[:2], bool), out_neg / f"{e['id']}.png")
        if negs:
            log(f"  {len(negs)} empty negative labels")

    def _photo(self, e: dict) -> np.ndarray:
        path = REPO_ROOT / e["path"]
        if sha256_file(path) != e["sha256"]:
            raise ValueError(f"{e['path']}: sha256 differs from the dataset file")
        return load_rgb_image(path)

    # after
    def labels(self) -> list[dict]:
        """One row per photo (LABEL_FIELDS): accepted (its round's mask), dropped, pending, or negative."""
        verdicts = self.verdicts()
        out = []
        for it in self.items:
            st = self.state(it, verdicts)
            row = self.row(it, st.round) if st.round else None
            acc = st.status == "accepted"
            out.append({"id": it["id"], "path": it["path"], "photo_sha256": it["sha256"], "source": it["source"],
                        "split": it["split"], "status": st.status if st.status in ("accepted", "dropped") else
                        "pending", "round": st.round if acc or st.status == "dropped" else "",
                        "mask": rel(self.mask_path(it, st.round)) if acc else "",
                        "mask_sha256": row["mask_sha256"] if acc else "",
                        "include": row["include"] if acc else "", "exclude": row["exclude"] if acc else "",
                        "reason": st.reason or ""})
        for e in self.negatives:
            f = self.mask_dir / "neg" / f"{e['id']}.png"
            out.append({"id": e["id"], "path": e["path"], "photo_sha256": e["sha256"], "source": e["source"],
                        "split": e["split"], "status": "negative", "round": "", "mask": rel(f),
                        "mask_sha256": mask_sha256(np.array(Image.open(f)) > 0) if f.exists() else "",
                        "include": "", "exclude": "", "reason": "D14: negative, empty label"})
        return out

    def write_labels(self) -> list[dict]:
        rows = self.labels()
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        with open(self.mask_dir / "labels.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        return rows


class _Refused(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- the pages' shared parts

class _Photos:
    """The last few decoded photos and masks (two masks a photo in the paired mode)."""

    def __init__(self, n: int = 3):
        self.rgb, self.masks = _LRU(n), _LRU(2 * n)

    def photo(self, path: str) -> np.ndarray:
        return self.rgb.get(path, lambda: load_rgb_image(REPO_ROOT / path))

    def mask(self, path: Path, sha: str) -> np.ndarray:
        return self.masks.get((str(path), sha), lambda: np.array(Image.open(path)) > 0)


def _points(body: dict, key: str) -> list[list[float]]:
    pts = body.get(key, [])
    ok = isinstance(pts, list) and all(isinstance(p, list) and len(p) == 2 and
                                       all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in p) for p in pts)
    if not ok:
        raise _Refused(400, f"{key}: a list of [x, y] fractions of the photo")
    return [[float(c) for c in p] for p in pts]


def _decision(body: dict) -> tuple[str, str | None]:
    decision, reason = body.get("decision"), body.get("reason")
    if decision not in ("accept", "decline") or (reason is not None and reason not in REASONS):
        raise _Refused(400, "decision: accept or decline, reason: one of REASONS or none")
    return decision, None if decision == "accept" else reason


def _app() -> Flask:
    app = Flask(__name__)

    @app.errorhandler(_Refused)
    def refused(e: _Refused):
        return jsonify({"error": str(e)}), e.code

    return app


def _image_routes(app: Flask, shown: Callable[..., tuple[np.ndarray, np.ndarray]], view_rule: str,
                  zoom_rule: str) -> None:
    """The photo with its mask outlined (`view_rule`) and a full-resolution tile around ?x=&y= (`zoom_rule`),
    ?outline=0 without the outline; `shown(**the rule's arguments)` gives the (photo, mask)."""
    def outline() -> bool:
        return request.args.get("outline", "1") == "1"

    @app.get(view_rule, endpoint="view")
    def view(**where):
        return send_file(jpeg(view_image(*shown(**where), outline())), mimetype="image/jpeg")

    @app.get(zoom_rule, endpoint="zoom")
    def zoom(**where):
        x, y = float(request.args["x"]), float(request.args["y"])
        return send_file(jpeg(zoom_tile(*shown(**where), x, y, outline())), mimetype="image/jpeg")


# ---------------------------------------------------------------- labelling mode

def create_label_app(session: LabelSession, redrawer, rule: str, reviewer: str = "owner") -> Flask:
    """The labelling page over `session`'s positives; `redrawer` draws every redraw (SegDrawer in use)."""
    app = _app()
    photos = _Photos()
    lock = threading.Lock()
    items = session.items

    def get(i: int) -> dict:
        if not 0 <= i < len(items):
            abort(404)
        return items[i]

    def todo(st: State) -> bool:
        return st.status in ("judge", "points", "pending")

    def next_todo(after: int, verdicts: dict) -> int | None:
        return next((j for j in range(after + 1, len(items)) if todo(session.state(items[j], verdicts))), None)

    @app.get("/")
    def home():
        verdicts = session.verdicts()
        first = next((j for j, it in enumerate(items) if todo(session.state(it, verdicts))), 0)
        return redirect(f"/item/{first}")

    @app.get("/item/<int:i>")
    def page(i: int):
        it = get(i)
        verdicts = session.verdicts()
        st = session.state(it, verdicts)
        counts = Counter(session.state(x, verdicts).status for x in items)
        row = session.row(it, st.round) if st.round else None
        mine = verdicts.get((it["id"], st.round))
        info = {"i": i, "n": len(items), "item": it["id"], "source": it["source"], "split": it["split"],
                "status": st.status, "round": st.round, "rounds": len(ROUNDS), "reason": st.reason,
                "model": row["model"] if row else None, "mask_sha256": row["mask_sha256"] if row else "",
                "points": session.points(it, st.round) if st.round else {"include": [], "exclude": []},
                "decision": mine["decision"] if mine else None, "accepted": counts["accepted"],
                "dropped": counts["dropped"], "todo": counts["judge"] + counts["points"] + counts["pending"],
                "reasons": REASONS, "session": session.name}
        return LABEL_PAGE.replace("__INFO__", json.dumps(info)).replace("__RULE__", html.escape(rule))

    def shown(i: int) -> tuple[np.ndarray, np.ndarray]:
        it = get(i)
        st = session.state(it)
        if not st.round:
            abort(404)
        return photos.photo(it["path"]), photos.mask(session.mask_path(it, st.round),
                                                     session.row(it, st.round)["mask_sha256"])

    _image_routes(app, shown, "/img/<int:i>/view.jpg", "/img/<int:i>/zoom.jpg")

    def reply(i: int, st: State):
        return jsonify({"status": st.status, "round": st.round, "reason": st.reason,
                        "next": next_todo(i, session.verdicts())})

    @app.post("/decide/<int:i>")
    def decide(i: int):
        it = get(i)
        decision, reason = _decision(request.get_json(force=True))
        with lock:
            return reply(i, session.decide(it, decision, reason, reviewer))

    @app.post("/redraw/<int:i>")
    def redraw(i: int):
        it = get(i)
        body = request.get_json(force=True)
        include, exclude = _points(body, "include"), _points(body, "exclude")
        with lock:
            return reply(i, session.redraw(it, include, exclude, redrawer))

    return app


# ---------------------------------------------------------------- blind paired mode (D22)

def paired_items(entries: list[dict], mask_dirs: list[Path]) -> list[dict]:
    """Every test positive with its mask from each of the two `mask_dirs` (seg_predict's layout: index.csv with
    id, path, photo_sha256, mask_sha256, and <id>.png). A model is named by its directory."""
    if len(mask_dirs) != 2:
        raise ValueError("the paired mode compares two mask sets")
    names = [d.name for d in mask_dirs]
    if names[0] == names[1]:
        raise ValueError(f"two mask sets named {names[0]}")
    indexes = [{r["photo_sha256"]: r for r in read_csv(d / "index.csv")} for d in mask_dirs]
    out, missing = [], []
    for e in entries:
        if e["split"] != "test" or e["source"] == "negative":
            continue
        masks = []
        for d, name, index in zip(mask_dirs, names, indexes):
            r = index.get(e["sha256"])
            if r is None:
                missing.append(f"{name}: {e['path']}")
                continue
            masks.append({"mask": d / f"{r['id']}.png", "mask_sha256": r["mask_sha256"]})
        if len(masks) == 2:
            out.append({"item": photo_id(e["path"]), "path": e["path"], "photo_sha256": e["sha256"],
                        "source": e["source"], "names": names, "masks": masks,
                        "identical": masks[0]["mask_sha256"] == masks[1]["mask_sha256"]})
    if missing:
        raise ValueError(f"{len(missing)} test positives have no mask in " + "; ".join(missing[:5]))
    return out


def paired_sides(photo_sha256: str, seed: int) -> tuple[int, int]:
    """(left, right) as indexes into the two mask sets, drawn per photo from `seed` (D22)."""
    flip = seeded_rng(seed, photo_sha256).random() < 0.5
    return (1, 0) if flip else (0, 1)


def shown_sides(it: dict, seed: int) -> dict[str, list[int]]:
    """{side: the mask sets it shows}: left and right, or both when the masks are identical (judged once)."""
    if it["identical"]:
        return {"both": [0, 1]}
    left, right = paired_sides(it["photo_sha256"], seed)
    return {"left": [left], "right": [right]}


def paired_decisions(items: list[dict], decisions_path: Path, seed: int) -> dict[tuple[str, str], dict]:
    """{(item, side): the last verdict on the mask that side shows now}."""
    by_item = {it["item"]: it for it in items}
    out = {}
    for r in _read_jsonl(decisions_path):
        it = by_item.get(r["item"])
        if not it:
            continue
        sides = shown_sides(it, seed)
        if r["side"] in sides and it["masks"][sides[r["side"]][0]]["mask_sha256"] == r["mask_sha256"]:
            out[(r["item"], r["side"])] = r
    return out


def create_paired_app(items: list[dict], decisions_path: Path, seed: int, rule: str,
                      reviewer: str = "owner") -> Flask:
    app = _app()
    photos = _Photos()
    lock = threading.Lock()

    def get(i: int) -> dict:
        if not 0 <= i < len(items):
            abort(404)
        return items[i]

    def sides(it: dict) -> dict[str, list[int]]:
        return shown_sides(it, seed)

    def current() -> dict[tuple[str, str], dict]:
        return paired_decisions(items, decisions_path, seed)

    def judged(dec: dict) -> tuple[int, int]:
        total = sum(len(sides(it)) for it in items)
        return sum(1 for it in items for s in sides(it) if (it["item"], s) in dec), total

    def complete(dec: dict) -> bool:
        done, total = judged(dec)
        return done == total

    def reveal_table() -> dict:
        dec = current()
        return {"models": items[0]["names"] if items else [],
                "sides": {it["item"]: {s: [it["names"][k] for k in ks] for s, ks in sides(it).items()} for it in items},
                "verdicts": {it["item"]: {it["names"][k]: dec[(it["item"], s)]["decision"]
                                          for s, ks in sides(it).items() for k in ks} for it in items}}

    @app.get("/")
    def home():
        dec = current()
        first = next((j for j, it in enumerate(items) if any((it["item"], s) not in dec for s in sides(it))), 0)
        return redirect(f"/item/{first}")

    @app.get("/item/<int:i>")
    def page(i: int):
        it = get(i)
        dec = current()
        done, total = judged(dec)
        info = {"i": i, "n": len(items), "item": it["item"], "source": it["source"], "identical": it["identical"],
                "sides": list(sides(it)), "decisions": {s: (dec[(it["item"], s)]["decision"]
                                                            if (it["item"], s) in dec else None) for s in sides(it)},
                "judged": done, "total": total, "complete": done == total, "reasons": REASONS,
                "revealed": ({s: [it["names"][k] for k in ks] for s, ks in sides(it).items()}
                             if done == total else None)}
        return PAIRED_PAGE.replace("__INFO__", json.dumps(info)).replace("__RULE__", html.escape(rule))

    def shown(i: int, side: str) -> tuple[np.ndarray, np.ndarray]:
        it = get(i)
        if side not in sides(it):
            abort(404)
        m = it["masks"][sides(it)[side][0]]
        return photos.photo(it["path"]), photos.mask(m["mask"], m["mask_sha256"])

    _image_routes(app, shown, "/img/<int:i>/<side>.jpg", "/img/<int:i>/<side>/zoom.jpg")

    @app.post("/decide/<int:i>")
    def decide(i: int):
        it = get(i)
        body = request.get_json(force=True)
        decision, reason = _decision(body)
        side = body.get("side")
        if side not in sides(it):
            raise _Refused(400, f"side: one of {list(sides(it))} for this photo")
        ks = sides(it)[side]
        with lock:
            _append_jsonl(decisions_path, {
                "item": it["item"], "path": it["path"], "photo_sha256": it["photo_sha256"], "source": it["source"],
                "side": side, "models": [it["names"][k] for k in ks], "mask_sha256": it["masks"][ks[0]]["mask_sha256"],
                "decision": decision, "reason": reason, "seed": seed, "reviewer": reviewer, "via": "seg_review",
                "ts": _now()})
            dec = current()
        nxt = next((j for j in range(i + 1, len(items))
                    if any((items[j]["item"], s) not in dec for s in sides(items[j]))), None)
        this_done = all((it["item"], s) in dec for s in sides(it))
        return jsonify({"ok": True, "next": nxt, "photo_done": this_done, "complete": complete(dec)})

    @app.get("/reveal")
    def reveal():
        if not complete(current()):
            raise _Refused(403, "the model names are shown once every mask is judged")
        return jsonify(reveal_table())

    return app


# ---------------------------------------------------------------- after a session

# ---------------------------------------------------------------- P10's paired report (D22)

def paired_verdicts(items: list[dict], decisions_path: Path, seed: int) -> dict[str, dict[str, bool]]:
    """{item: {model: accepted}} once every mask is judged; an identical pair's one verdict counts for both."""
    dec = paired_decisions(items, decisions_path, seed)
    missing = [(it["item"], s) for it in items for s in shown_sides(it, seed) if (it["item"], s) not in dec]
    if missing:
        raise ValueError(f"{len(missing)} masks not judged yet, e.g. {missing[0]}")
    return {it["item"]: {it["names"][k]: dec[(it["item"], s)]["decision"] == "accept"
                         for s, ks in shown_sides(it, seed).items() for k in ks} for it in items}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs: binomial(b + c, 1/2), doubled tail."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


PAIRED_GROUPS = {"pool": "pool", "segmenter_positive": "segmenter_positive",
                 "internet_positive": "segmenter_positive"}    # D22: internet positives count with them


def paired_report(items: list[dict], verdicts: dict[str, dict[str, bool]], neg_masks: dict[str, list[np.ndarray]],
                  min_area_frac: float) -> dict:
    """D22: per source and overall, both pass / first only / second only / both fail, and the exact McNemar
    p-value on the discordant pairs; each model's empty and empty-or-tiny masks on the test negatives (the
    D18 rule: no pixel, or an area under `min_area_frac`). Models in the items' order."""
    names = items[0]["names"]
    counts: dict[str, Counter] = {}
    for it in items:
        a, b = (verdicts[it["item"]][n] for n in names)
        key = "both pass" if a and b else f"{names[0]} only" if a else f"{names[1]} only" if b else "both fail"
        for group in (PAIRED_GROUPS[it["source"]], "overall"):
            counts.setdefault(group, Counter())[key] += 1
    out = {"models": names, "min_area_frac": min_area_frac, "groups": {}, "negatives": {}}
    for group in ("pool", "segmenter_positive", "overall"):
        c = counts.get(group, Counter())
        b, d = c[f"{names[0]} only"], c[f"{names[1]} only"]
        n = sum(c.values())
        out["groups"][group] = {
            "n": n, "both pass": c["both pass"], f"{names[0]} only": b, f"{names[1]} only": d,
            "both fail": c["both fail"], f"{names[0]} pass": c["both pass"] + b, f"{names[1]} pass": c["both pass"] + d,
            "mcnemar_p": mcnemar_exact(b, d)}
    for name in names:
        masks = neg_masks[name]
        out["negatives"][name] = {"n": len(masks), "empty": sum(not m.any() for m in masks),
                                  "empty_or_tiny": sum(bool(not m.any() or m.mean() < min_area_frac) for m in masks)}
    return out


def tiny_threshold(items: list[dict], verdicts: dict[str, dict[str, bool]], model: str, areas: dict[str, float],
                   old: float) -> dict:
    """D21, ML-2 D18's rule on the new test split: half the smallest area (mask pixels / photo pixels) among the
    chosen model's test-positive masks the owner accepted; and which test positives of that model fall under the
    old and the new value (empty masks included). `areas` {item: area} for that model's masks."""
    accepted = [areas[it["item"]] for it in items if verdicts[it["item"]][model]]
    new = round(0.5 * min(accepted), 6) if accepted else None
    under = lambda t: [it["path"] for it in items if t is not None and areas[it["item"]] < t]  # noqa: E731
    return {"model": model, "rule": "half the smallest accepted test-positive mask (ML-2 D18, ML-5 D21)",
            "from_n_accepted": len(accepted), "old": old, "new": new, "under_old": under(old), "under_new": under(new)}


def print_paired_report(rep: dict) -> None:
    a, b = rep["models"]
    print(f"test positives, judged blind and paired (D22): {a} vs {b}")
    for group, c in rep["groups"].items():
        print(f"  {group:20s} n={c['n']:3d}  both pass {c['both pass']:3d}  {a} only {c[f'{a} only']:2d}  "
              f"{b} only {c[f'{b} only']:2d}  both fail {c['both fail']:2d}  |  {a} {c[f'{a} pass']}/{c['n']}  "
              f"{b} {c[f'{b} pass']}/{c['n']}  |  exact McNemar p = {c['mcnemar_p']:.3g}")
    print(f"test negatives, empty / empty or tiny (area < {rep['min_area_frac']:g}):")
    for name, c in rep["negatives"].items():
        print(f"  {name:20s} {c['empty']}/{c['n']} empty, {c['empty_or_tiny']}/{c['n']} empty or tiny")
    if t := rep.get("tiny_threshold"):
        print(f"tiny-mask threshold (D21) from {t['model']}'s {t['from_n_accepted']} accepted test positives: "
              f"new {t['new']}, old {t['old']}; test positives under old {len(t['under_old'])}, "
              f"under new {len(t['under_new'])}")
        for path in sorted(set(t["under_old"]) | set(t["under_new"])):
            which = [name for name in ("old", "new") if path in t[f"under_{name}"]]
            print(f"  under {' and '.join(which)}: {path}")


def report(rows: list[dict]) -> dict:
    """P7's report: accepted by round, dropped and why, per source."""
    pos = [r for r in rows if r["status"] != "negative"]
    by_round = Counter(int(r["round"]) for r in pos if r["status"] == "accepted")
    by_source: dict[str, Counter] = {}
    for r in pos:
        by_source.setdefault(r["source"], Counter())[
            f"accepted_r{r['round']}" if r["status"] == "accepted" else r["status"]] += 1
    return {"photos": len(pos), "negatives": len(rows) - len(pos),
            "status": dict(Counter(r["status"] for r in pos)),
            "accepted_by_round": {f"r{k}": by_round.get(k, 0) for k in ROUNDS},
            "by_source": {s: dict(sorted(c.items())) for s, c in sorted(by_source.items())},
            "dropped": [{"path": r["path"], "source": r["source"], "reason": r["reason"]}
                        for r in pos if r["status"] == "dropped"]}


def print_report(rep: dict) -> None:
    print(f"{rep['photos']} positives + {rep['negatives']} negatives (empty labels)")
    print(f"  status: {rep['status']}")
    print(f"  accepted by round: {rep['accepted_by_round']}")
    for s, c in rep["by_source"].items():
        print(f"    {s:20s} {c}")
    for d in rep["dropped"]:
        print(f"  dropped: {d['path']} ({d['source']}): {d['reason']}")


def accepted_labels(session: str, mask_root: Path = MASK_ROOT) -> list[dict]:
    """The accepted labels of a session's labels.csv (written by `status`): path, round, mask, points."""
    f = mask_root / session / "labels.csv"
    if not f.exists():
        raise ValueError(f"{rel(f)} missing: run `python -m coffeecv.seg_review status --session {session}`")
    return [{**r, "round": int(r["round"]), "include": json.loads(r["include"]), "exclude": json.loads(r["exclude"])}
            for r in read_csv(f) if r["status"] == "accepted"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("label", help="prepare and serve a labelling session")
    lab.add_argument("--proposer", required=True, help="round 1's segmenter (pass 1: ft_s123, D19)")
    lab.add_argument("--redraw", required=True, help="the segmenter that redraws from points (pass 1: pretrained_l0)")
    lab.add_argument("--prepare-only", action="store_true", help="prepare, then exit without serving")
    st = sub.add_parser("status", help="write the session's labels.csv and print the report")
    st.add_argument("--out", type=Path, help="also write the report as JSON")
    par = sub.add_parser("paired", help="the blind paired review of two mask sets on the test positives (D22)")
    par.add_argument("--masks", nargs=2, type=Path, required=True, metavar="DIR",
                     help="two mask directories (index.csv + <id>.png), each named by its model")
    par.add_argument("--seed", type=int, default=None, help="side draw (default: params.yaml seg_paired_seed)")
    prep = sub.add_parser("paired-report", help="P10's paired counts, McNemar test and test-negative rates (D22)")
    prep.add_argument("--masks", nargs=2, type=Path, required=True, metavar="DIR", help="as for paired, same order")
    prep.add_argument("--seed", type=int, default=None, help="as for paired")
    prep.add_argument("--out", type=Path, help="also write the report as JSON")
    prep.add_argument("--chosen", help="the model the owner picks (D20): re-derive the tiny-mask threshold (D21)")
    for p in (lab, st, par, prep):
        p.add_argument("--session", required=True, help="the session's name, e.g. pass1")
    for p in (lab, par):
        p.add_argument("--reviewer", default="owner")
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 if VS Code does not forward localhost")
    args = ap.parse_args(argv)
    entries = load_seg_dataset()
    rule = RULE_FILE.read_text().strip()
    if args.command == "status":
        rep = report(LabelSession(args.session, entries, None).write_labels())
        print_report(rep)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rep, indent=2) + "\n")
        return
    if args.command == "paired-report":
        cfg = RunConfig.from_params_yaml()
        seed = cfg.seg_paired_seed if args.seed is None else args.seed
        dirs = [d if d.is_absolute() else REPO_ROOT / d for d in args.masks]
        items = paired_items(entries, dirs)
        verdicts = paired_verdicts(items, LABELS_DIR / f"{args.session}.paired.jsonl", seed)
        neg_masks = {d.name: [np.array(Image.open(d / f"{r['id']}.png")) > 0 for r in read_csv(d / "index.csv")
                              if r.get("source") == "negative"] for d in dirs}
        rep = paired_report(items, verdicts, neg_masks, cfg.seg_min_area_frac)
        if args.chosen:
            k = items[0]["names"].index(args.chosen) if items and args.chosen in items[0]["names"] else None
            if k is None:
                raise SystemExit(f"--chosen {args.chosen}: not one of {items[0]['names'] if items else []}")
            areas = {it["item"]: float((np.array(Image.open(it["masks"][k]["mask"])) > 0).mean()) for it in items}
            rep["tiny_threshold"] = tiny_threshold(items, verdicts, args.chosen, areas, cfg.seg_min_area_frac)
        print_paired_report(rep)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rep, indent=2) + "\n")
        return
    if args.command == "paired":
        seed = RunConfig.from_params_yaml().seg_paired_seed if args.seed is None else args.seed
        items = paired_items(entries, [d if d.is_absolute() else REPO_ROOT / d for d in args.masks])
        path = LABELS_DIR / f"{args.session}.paired.jsonl"
        print(f"{len(items)} test positives, {sum(it['identical'] for it in items)} with identical masks; "
              f"seed {seed}; verdicts -> {rel(path)}")
        create_paired_app(items, path, seed, rule, args.reviewer).run(host=args.host, port=args.port, threaded=True)
        return
    torch.set_num_threads(THREADS)
    session = LabelSession(args.session, entries, {"proposer": args.proposer, "redraw": args.redraw})
    redrawer = SegDrawer(args.redraw)
    proposer = redrawer if args.proposer == args.redraw else SegDrawer(args.proposer)
    print(f"session {args.session}: {len(session.items)} positives, {len(session.negatives)} negatives; "
          f"proposer {args.proposer}, redraw {args.redraw} ({REDRAW_OUTPUT})", flush=True)
    session.prepare(proposer, redrawer, log=lambda s: print(s, flush=True))
    if args.prepare_only:
        return
    print(f"verdicts -> {rel(session.verdicts_path)}; masks -> {rel(session.mask_dir)}", flush=True)
    create_label_app(session, redrawer, rule, args.reviewer).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
