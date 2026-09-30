"""The owner's mask review page (ticket ML-2 D12, plan §4.2): accept or decline (photo, mask) items at full
resolution, one click each, straight into a decisions file committed to git.

    python -m coffeecv.review_masks --list base_candidates            # the D25 base masks
    python -m coffeecv.review_masks --items items.csv --name p2_fails  # any (photo, mask) list
    python -m coffeecv.review_masks --items audit.csv --name p2_audit --blind --model claude-...

then open http://localhost:8765 (VS Code forwards the port out of the devcontainer). If it doesn't
(owner's setup, 2026-09-30), start it with --host 0.0.0.0 and open http://<container IP>:8765
(`hostname -I`; it was 172.17.0.3). That reaches only the Docker bridge, not the network.

The page shows the photo with a thin magenta outline of the mask (Space toggles it), the D4 crop box
dashed, and the D13 rule from labels/ml2/accept_rule.md. A click on the photo opens that spot at full
resolution. A = accept, D = decline, 1-5 = decline with a reason, arrows = previous/next,
N = next unreviewed. Each click appends {item, path, mask_sha256, decision, reason, reviewer, via, ts}
to labels/ml2/review/<name>.decisions.jsonl; the last row for an item wins, and a row made for a
different mask (another sha) does not count, so that item is unreviewed again. --blind hides the judge's
stored verdict until after the click (the P2 audit).
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, redirect, request, send_file
from PIL import Image, ImageDraw

from coffeecv import seg_lists, seg_overlay
from coffeecv.config import REPO_ROOT
from coffeecv.dataset import load_rgb_image
from coffeecv.seg_base_masks import MASK_DIR as BASE_MASK_DIR
from coffeecv.segment_beans import d4_box

REVIEW_DIR = REPO_ROOT / "labels" / "ml2" / "review"
RULE_FILE = REPO_ROOT / "labels" / "ml2" / "accept_rule.md"
REASONS = ("rim/tray included", "beans missed", "snapped to single beans", "not coffee", "other")
VIEW_LONG_SIDE = 1600
ZOOM = 1024


def items_from_list(name: str) -> list[dict]:
    """A photo list whose masks are the D25 base masks (base_candidates, base_dev, base_heldout)."""
    _, lists = seg_lists.load_lists()
    index = {r["path"]: r for r in csv.DictReader((BASE_MASK_DIR / "index.csv").read_text().splitlines())}
    return [{"item": index[e["path"]]["id"], "path": e["path"], "mask_sha256": index[e["path"]]["mask_sha256"],
             "mask": str((BASE_MASK_DIR / f"{index[e['path']]['id']}.png").relative_to(REPO_ROOT)),
             "photo_sha256": e["sha256"]} for e in lists[name]]


def items_from_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.read_text().splitlines()))


def current_decisions(path: Path, items: list[dict]) -> dict[str, dict]:
    """{item: its last decision row}, counting only rows made for the item's current mask."""
    sha = {it["item"]: it["mask_sha256"] for it in items}
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if sha.get(r["item"]) == r["mask_sha256"]:
                    out[r["item"]] = r
    return out


class _Cache:
    """The last few decoded photos and masks: a 50 MP photo takes a second or two to decode."""

    def __init__(self, n: int = 3):
        self.n, self.d = n, OrderedDict()

    def get(self, it: dict) -> tuple[np.ndarray, np.ndarray]:
        k = it["item"]
        if k not in self.d:
            rgb = load_rgb_image(REPO_ROOT / it["path"])
            mask = np.array(Image.open(REPO_ROOT / it["mask"])) > 0
            self.d[k] = (rgb, mask)
            while len(self.d) > self.n:
                self.d.popitem(last=False)
        self.d.move_to_end(k)
        return self.d[k]


def _jpeg(arr: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf


def create_app(items: list[dict], decisions_path: Path, rule: str, verdicts: dict[str, dict] | None = None,
               reviewer: str = "owner") -> Flask:
    """`verdicts` ({item: stored judge row}) turns on blind mode: a verdict is revealed only in the reply
    to a decision."""
    app = Flask(__name__)
    cache = _Cache()
    by_index = {i: it for i, it in enumerate(items)}

    def get(i: int) -> dict:
        if i not in by_index:
            abort(404)
        return by_index[i]

    def state() -> dict:
        dec = current_decisions(decisions_path, items)
        counts = {"accept": 0, "decline": 0}
        for r in dec.values():
            counts[r["decision"]] += 1
        return {"decisions": dec, "reviewed": len(dec), "total": len(items), **counts}

    @app.get("/")
    def home():
        dec = state()["decisions"]
        first = next((i for i, it in by_index.items() if it["item"] not in dec), 0)
        return redirect(f"/item/{first}")

    @app.get("/item/<int:i>")
    def page(i: int):
        it = get(i)
        s = state()
        mine = s["decisions"].get(it["item"])
        info = {"i": i, "n": len(items), "item": it["item"], "reviewed": s["reviewed"], "accept": s["accept"],
                "decline": s["decline"], "decision": mine["decision"] if mine else None,
                "reason": mine.get("reason") if mine else None, "blind": verdicts is not None,
                "verdict": _verdict(it) if (verdicts is not None and mine) else None, "reasons": REASONS}
        return PAGE.replace("__INFO__", json.dumps(info)).replace("__RULE__", html.escape(rule))

    def _verdict(it: dict) -> dict | None:
        v = verdicts.get(it["item"]) if verdicts is not None else None
        return {k: v.get(k) for k in ("verdict", "failed_rules", "reason", "model")} if v else None

    @app.get("/img/<int:i>/view.jpg")
    def view(i: int):
        rgb, mask = cache.get(get(i))
        h, w = mask.shape
        s = min(1.0, VIEW_LONG_SIDE / max(h, w))
        size = (round(w * s), round(h * s))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if request.args.get("outline", "1") == "1":
            small = seg_overlay.outline(small, cv2.resize(mask.astype(np.uint8), size,
                                                          interpolation=cv2.INTER_NEAREST) > 0, 1)
            if mask.any():
                im = Image.fromarray(small)
                x0, y0, bw, bh = d4_box(mask)
                seg_overlay._dashed_rect(ImageDraw.Draw(im), x0 * s, y0 * s, (x0 + bw) * s, (y0 + bh) * s)
                small = np.asarray(im)
        return send_file(_jpeg(small), mimetype="image/jpeg")

    @app.get("/img/<int:i>/zoom.jpg")
    def zoom(i: int):
        """Full resolution around (x, y), given as fractions of the photo."""
        rgb, mask = cache.get(get(i))
        h, w = mask.shape
        x, y = float(request.args["x"]), float(request.args["y"])
        zw, zh = min(ZOOM, w), min(ZOOM, h)
        x0 = int(np.clip(x * w - zw / 2, 0, w - zw))
        y0 = int(np.clip(y * h - zh / 2, 0, h - zh))
        if request.args.get("outline", "1") == "1":
            tile = seg_overlay.outline(rgb, mask, 1, (x0, y0, zw, zh))
        else:
            tile = rgb[y0:y0 + zh, x0:x0 + zw]
        return send_file(_jpeg(tile), mimetype="image/jpeg")

    @app.post("/decide/<int:i>")
    def decide(i: int):
        it = get(i)
        body = request.get_json(force=True)
        decision, reason = body.get("decision"), body.get("reason")
        if decision not in ("accept", "decline") or (reason is not None and reason not in REASONS):
            abort(400)
        if decision == "accept":
            reason = None
        row = {"item": it["item"], "path": it["path"], "mask_sha256": it["mask_sha256"], "decision": decision,
               "reason": reason, "reviewer": reviewer, "via": "review_masks",
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(decisions_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        s = state()
        nxt = next((j for j in range(i + 1, len(items)) if items[j]["item"] not in s["decisions"]), None)
        return jsonify({"ok": True, "reviewed": s["reviewed"], "accept": s["accept"], "decline": s["decline"],
                        "next": nxt, "verdict": _verdict(it)})

    return app


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ML-2 mask review</title>
<style>
body{font-family:sans-serif;margin:0;background:#222;color:#eee;display:flex;height:100vh}
#left{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden}
#left img{max-width:100%;max-height:100vh;cursor:crosshair}
#right{width:560px;padding:12px;overflow:auto;background:#2b2b2b}
#zoom{width:512px;height:512px;background:#111;display:block;margin:8px 0}
button{font-size:15px;margin:3px;padding:6px 10px}
.acc{background:#2e7d32;color:#fff}.dec{background:#c62828;color:#fff}
.rule{font-size:13px;line-height:1.4;background:#333;padding:8px;border-radius:4px}
.state{font-size:18px;margin:8px 0}.v{background:#444;padding:6px;margin-top:6px}
</style></head><body>
<div id="left"><img id="photo"></div>
<div id="right">
 <div id="head"></div>
 <div class="state" id="state"></div>
 <button class="acc" onclick="decide('accept')">Accept (A)</button>
 <button class="dec" onclick="decide('decline')">Decline (D)</button><br>
 <div id="reasons"></div>
 <div id="verdict"></div>
 <button onclick="go(info.i-1)">&larr; prev</button><button onclick="go(info.i+1)">next &rarr;</button>
 <button onclick="location='/'">next unreviewed (N)</button>
 <div style="font-size:12px;color:#aaa">Space: toggle outline &middot; click the photo: full resolution</div>
 <img id="zoom">
 <div class="rule"><b>Accept rule (D13)</b><br>__RULE__</div>
</div>
<script>
const info = __INFO__;
let outline = 1, last = null;
const photo = document.getElementById('photo'), zoom = document.getElementById('zoom');
function src(){ photo.src = `/img/${info.i}/view.jpg?outline=${outline}`;
  if(last) zoom.src = `/img/${info.i}/zoom.jpg?x=${last[0]}&y=${last[1]}&outline=${outline}`; }
function render(){
  document.getElementById('head').textContent = `${info.i+1} / ${info.n}  ${info.item}`;
  document.getElementById('state').innerHTML =
    `reviewed ${info.reviewed}/${info.n} (accepted ${info.accept}, declined ${info.decline})<br>this item: ` +
    (info.decision ? `<b>${info.decision}</b>${info.reason ? ' ('+info.reason+')' : ''}` : '<i>unreviewed</i>');
  document.getElementById('reasons').innerHTML = info.reasons.map((r,k)=>
    `<button class="dec" onclick="decide('decline', '${r}')">${k+1}: ${r}</button>`).join('');
  const v = info.verdict;
  document.getElementById('verdict').innerHTML = v ? `<div class="v">judge (${v.model}): <b>${v.verdict}</b>
    ${v.failed_rules && v.failed_rules.length ? 'rules '+v.failed_rules.join(',') : ''}<br>${v.reason||''}</div>`
    : (info.blind && info.decision ? '<div class="v">judge: no stored verdict</div>' : '');
}
function go(i){ if(i>=0 && i<info.n) location = `/item/${i}`; }
async function decide(decision, reason=null){
  const r = await (await fetch(`/decide/${info.i}`, {method:'POST', headers:{'Content-Type':'application/json'},
                   body: JSON.stringify({decision, reason})})).json();
  Object.assign(info, {decision, reason, reviewed:r.reviewed, accept:r.accept, decline:r.decline});
  if(info.blind){ info.verdict = r.verdict; render(); return; }   // stay: the verdict is shown after the click
  if(r.next !== null) go(r.next); else render();
}
photo.onclick = e => { const b = photo.getBoundingClientRect();
  last = [(e.clientX-b.left)/b.width, (e.clientY-b.top)/b.height]; src(); };
document.onkeydown = e => {
  if(e.key===' '){ outline = 1-outline; src(); e.preventDefault(); }
  else if(e.key==='a'||e.key==='A') decide('accept');
  else if(e.key==='d'||e.key==='D') decide('decline');
  else if(e.key>='1' && e.key<='5') decide('decline', info.reasons[+e.key-1]);
  else if(e.key==='ArrowLeft') go(info.i-1);
  else if(e.key==='ArrowRight') go(info.i+1);
  else if(e.key==='n'||e.key==='N') location='/';
};
render(); src();
</script></body></html>"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--list", help="a photo list with D25 base masks, e.g. base_candidates")
    src.add_argument("--items", type=Path, help="CSV: item, path, mask, mask_sha256[, photo_sha256]")
    ap.add_argument("--name", help="decisions file name (default: the list's name)")
    ap.add_argument("--blind", action="store_true", help="show the judge's stored verdict only after the click")
    ap.add_argument("--model", help="--blind: the judge model whose verdicts to reveal")
    ap.add_argument("--reviewer", default="owner")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 if VS Code does not forward localhost")
    args = ap.parse_args(argv)
    items = items_from_list(args.list) if args.list else items_from_csv(args.items)
    name = args.name or args.list
    if not name:
        ap.error("--items needs --name")
    verdicts = None
    if args.blind:
        from coffeecv.seg_judge import read_verdicts
        if not args.model:
            ap.error("--blind needs --model")
        keyed = {(r["photo_sha256"], r["mask_sha256"]): r for r in read_verdicts() if r["model"] == args.model}
        verdicts = {it["item"]: keyed[(it["photo_sha256"], it["mask_sha256"])] for it in items
                    if (it.get("photo_sha256"), it["mask_sha256"]) in keyed}
    app = create_app(items, REVIEW_DIR / f"{name}.decisions.jsonl", RULE_FILE.read_text().strip(), verdicts,
                     args.reviewer)
    print(f"{len(items)} items; decisions -> {(REVIEW_DIR / f'{name}.decisions.jsonl').relative_to(REPO_ROOT)}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
