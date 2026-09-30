"""The mask judge (ticket ML-2 P1, D6/D13-D16, plan §4.3-4.4). An annotation step: `dvc repro` never calls it.

One (photo, mask) item per call, so the judge never sees the base mask behind a planted defect. The judge
gets the prompt (labels/ml2/judge_prompt.md with the D13 rule from labels/ml2/accept_rule.md inserted
word for word) and the overlays from seg_overlay (overview + full-resolution boundary tiles, PNG from
pixels, no photo metadata). It answers with a verdict, the failed rules and up to 6 corrective points.

    --backend cli   headless `claude -p --model <id>` on the machine's Claude Code login (the owner's plan):
                    the images inside the one message (stream-json input), all tools off, from an empty
                    temporary directory (no CLAUDE.md, no project settings)
    --backend api   Anthropic Messages API with base64 images (needs ANTHROPIC_API_KEY and the `anthropic`
                    package); --batch sends Message Batches

A reply that fails to parse is retried once, then stored as `unjudged`, which never counts as accept.
Verdicts append to labels/ml2/verdicts.jsonl (git, D16), keyed by (photo_sha256, mask_sha256) plus model,
prompt sha, backend and trial; an existing row is skipped unless --rejudge.

    python -m coffeecv.seg_judge judge --set dev --dry-run       # D15 defaults: Opus 5.5 via claude -p
    python -m coffeecv.seg_judge score --set pilot --model claude-sonnet-5-5   # another model's rows

On this devcontainer the `claude` binary is VS Code's: CLAUDE_BIN=$CLAUDE_CODE_EXECPATH.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv import seg_defects, seg_overlay
from coffeecv.config import REPO_ROOT
from coffeecv.dataset import load_rgb_image
from coffeecv.segment_beans import mask_sha256

ML2 = REPO_ROOT / "labels" / "ml2"
RULE_FILE = ML2 / "accept_rule.md"
PROMPT_FILE = ML2 / "judge_prompt.md"
VERDICTS_FILE = ML2 / "verdicts.jsonl"
OVERLAY_DIR = REPO_ROOT / "outputs" / "ml2_p1" / "overlays"
PLACEHOLDER = "{{ACCEPT_RULE}}"
# D15, owner 2026-09-30 after the pilot: Opus 5.5 through headless `claude -p` on the owner's plan.
JUDGE_MODEL = "claude-opus-5-5"
JUDGE_BACKEND = "cli"
MAX_POINTS = 6
MAX_TOKENS = 1024
RETRY_WAIT_S = 30                                     # before retrying a call that failed (limit, network)
# Plan §4.4: every judged item in ML-2, for projecting the pilot's cost per item to the whole ticket.
FULL_RUN_ITEMS = 4700
_append_lock = threading.Lock()


# ---------------------------------------------------------------- prompt and parsing

def build_prompt() -> str:
    """judge_prompt.md with the D13 rule inserted byte for byte from accept_rule.md."""
    template = PROMPT_FILE.read_text()
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"{PROMPT_FILE.name} must hold {PLACEHOLDER} exactly once")
    return template.replace(PLACEHOLDER, RULE_FILE.read_text().strip())


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def parse_verdict(text: str, n_tiles: int) -> dict:
    """The judge's JSON object -> {verdict, failed_rules, reason, points}; ValueError on anything off-schema.
    Tolerates a ```json fence or prose around the one object."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    d = json.loads(m.group(0))
    verdict = d.get("verdict")
    if verdict not in ("accept", "decline"):
        raise ValueError(f"verdict {verdict!r}")
    rules = d.get("failed_rules", [])
    if not isinstance(rules, list) or not all(isinstance(r, int) and 1 <= r <= 4 for r in rules):
        raise ValueError(f"failed_rules {rules!r}")
    if (verdict == "decline") != bool(rules):
        raise ValueError(f"{verdict} with failed_rules {rules}")
    reason = d.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("no reason")
    points = d.get("points", [])
    if not isinstance(points, list) or len(points) > MAX_POINTS:
        raise ValueError(f"points: not a list of at most {MAX_POINTS}")
    where_ok = {"overview", *(f"T{k}" for k in range(1, n_tiles + 1))}
    for p in points:
        if not (isinstance(p, dict) and p.get("where") in where_ok and p.get("label") in ("include", "exclude")
                and all(isinstance(p.get(c), (int, float)) and 0 <= p[c] <= 1 for c in ("x", "y"))):
            raise ValueError(f"point {p!r}")
    if verdict == "accept" and points:
        raise ValueError("points on an accept")
    return {"verdict": verdict, "failed_rules": sorted(set(rules)), "reason": reason.strip(),
            "points": [{k: p[k] for k in ("where", "x", "y", "label")} for p in points]}


# ---------------------------------------------------------------- items and overlays

def load_items(set_name: str | None, items_csv: Path | None) -> list[dict]:
    """Rows with at least item, path, photo_sha256, mask (repo-relative PNG), mask_sha256; trial defaults to 0."""
    rows = seg_defects.load_manifest(set_name) if set_name else \
        list(csv.DictReader(items_csv.read_text().splitlines()))
    for r in rows:
        r["trial"] = int(r.get("trial") or 0)
    return rows


def prepare(item: dict) -> dict:
    """Render (or reuse) the overlays of one item. The directory is named by photo and mask hash only, so
    nothing in a path tells a planted defect from a clean mask."""
    out = OVERLAY_DIR / f"{item['photo_sha256'][:16]}_{item['mask_sha256'][:16]}"
    mask = np.array(Image.open(REPO_ROOT / item["mask"])) > 0
    if mask_sha256(mask) != item["mask_sha256"]:
        raise ValueError(f"{item['item']}: {item['mask']} is not the mask the manifest pins")
    if not (out / "meta.json").exists():
        photo = REPO_ROOT / item["path"]
        if hashlib.sha256(photo.read_bytes()).hexdigest() != item["photo_sha256"]:
            raise ValueError(f"{item['item']}: {item['path']} changed since it was listed")
        seg_overlay.write(load_rgb_image(photo), mask, out)
    meta = seg_overlay.load_meta(out)
    paths = [out / "overview.png", *(out / f"T{k}.png" for k in range(1, len(meta.tiles) + 1))]
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return {"dir": out, "paths": paths, "meta": meta, "overlay_sha256": h.hexdigest(),
            "area": float(mask.mean())}


def message_text(prep: dict) -> str:
    names = ", ".join(p.stem for p in prep["paths"])
    area = "The mask is empty (it covers 0% of the photo)." if prep["meta"].empty_mask else \
        f"The mask covers {100 * prep['area']:.1f}% of the photo."
    return f"{area} Images: {names}."


# ---------------------------------------------------------------- backends

def _content(prep: dict) -> list[dict]:
    """The user message: the area line, then each image as a base64 PNG block after its name. The same
    blocks go to both backends, so the judge sees the same input either way."""
    content = [{"type": "text", "text": message_text(prep)}]
    for p in prep["paths"]:
        content.append({"type": "text", "text": f"{p.stem}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.b64encode(p.read_bytes()).decode()}})
    return content


def call_cli(prompt: str, prep: dict, model: str, claude_bin: str) -> dict:
    """One `claude -p` call on the machine's Claude Code login. The judge prompt replaces Claude Code's system
    prompt and the images go inside the one user message (stream-json input), with every tool off: one
    round trip, no tool definitions, no file reads. Runs from an empty temporary directory, so no
    CLAUDE.md or project settings are picked up. Returns {text, usage, cost_usd, wall_s}."""
    msg = {"type": "user", "message": {"role": "user", "content": _content(prep)}}
    cmd = [claude_bin, "-p", "--model", model, "--input-format", "stream-json", "--output-format", "stream-json",
           "--verbose", "--system-prompt", prompt, "--tools", "", "--no-session-persistence"]
    with tempfile.TemporaryDirectory(prefix="ml2_judge_") as tmp:
        t0 = time.perf_counter()
        res = subprocess.run(cmd, cwd=tmp, input=json.dumps(msg) + "\n", capture_output=True, text=True,
                             timeout=900)
        wall = time.perf_counter() - t0
    out = next((json.loads(l) for l in reversed(res.stdout.splitlines())
                if l.strip().startswith("{") and json.loads(l).get("type") == "result"), None)
    if res.returncode != 0:
        why = (out or {}).get("result") or res.stderr.strip() or res.stdout.strip()[-500:]
        raise RuntimeError(f"claude -p exited {res.returncode}: {str(why)[:500]}")
    if out is None:
        raise RuntimeError(f"claude -p gave no result line: {res.stdout[-500:]}")
    if out.get("is_error"):
        raise RuntimeError(f"claude -p error: {str(out.get('result'))[:500]}")
    return {"text": out.get("result", ""), "usage": out.get("usage", {}), "cost_usd": out.get("total_cost_usd"),
            "wall_s": round(wall, 2), "num_turns": out.get("num_turns")}


def _api_params(prompt: str, prep: dict, model: str) -> dict:
    return {"model": model, "max_tokens": MAX_TOKENS, "system": prompt,
            "messages": [{"role": "user", "content": _content(prep)}]}


def _api_client():
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit("--backend api needs the `anthropic` package (not in the locked environment yet)") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--backend api needs ANTHROPIC_API_KEY")
    return anthropic.Anthropic()


def _usage(u) -> dict:
    return {k: getattr(u, k, None) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                             "cache_read_input_tokens")}


def call_api(client, prompt: str, prep: dict, model: str) -> dict:
    t0 = time.perf_counter()
    r = client.messages.create(**_api_params(prompt, prep, model))
    text = "".join(b.text for b in r.content if b.type == "text")
    return {"text": text, "usage": _usage(r.usage), "cost_usd": None, "wall_s": round(time.perf_counter() - t0, 2)}


def call_api_batch(client, prompt: str, preps: dict[str, dict], model: str, poll_s: int = 30) -> dict[str, dict]:
    """One Message Batch over {custom_id: prep}; waits for it to end. Returns {custom_id: reply}, with
    {"error": ...} for requests that did not succeed. wall_s is the whole batch's time, per request."""
    t0 = time.perf_counter()
    batch = client.messages.batches.create(requests=[
        {"custom_id": cid, "params": _api_params(prompt, prep, model)} for cid, prep in preps.items()])
    print(f"  batch {batch.id}: {len(preps)} requests")
    while (batch := client.messages.batches.retrieve(batch.id)).processing_status != "ended":
        time.sleep(poll_s)
    wall = round(time.perf_counter() - t0, 2)
    out = {}
    for res in client.messages.batches.results(batch.id):
        if res.result.type == "succeeded":
            m = res.result.message
            out[res.custom_id] = {"text": "".join(b.text for b in m.content if b.type == "text"),
                                  "usage": _usage(m.usage), "cost_usd": None, "wall_s": wall, "batch_id": batch.id}
        else:
            out[res.custom_id] = {"error": res.result.type, "wall_s": wall, "batch_id": batch.id}
    return out


# ---------------------------------------------------------------- verdict rows

def run_key(row: dict) -> tuple:
    return (row["photo_sha256"], row["mask_sha256"], row["model"], row["prompt_sha256"], row["backend"],
            row["trial"])


def read_verdicts(path: Path | None = None) -> list[dict]:
    path = path or VERDICTS_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_verdict(row: dict, path: Path | None = None) -> None:
    with _append_lock, open(path or VERDICTS_FILE, "a") as f:
        f.write(json.dumps(row, sort_keys=False) + "\n")


def make_row(item: dict, prep: dict, model: str, prompt_sha: str, backend: str, replies: list[dict]) -> dict:
    """The stored row for one item from its replies (1 or 2 attempts). The last reply that parses wins;
    none parsing = unjudged."""
    parsed, err = None, None
    for rep in replies:
        if "error" in rep:
            err = rep["error"]
            continue
        try:
            parsed = parse_verdict(rep["text"], len(prep["meta"].tiles))
        except (ValueError, json.JSONDecodeError) as e:
            err = str(e)
    row = {"photo_sha256": item["photo_sha256"], "mask_sha256": item["mask_sha256"], "item": item["item"],
           "trial": item["trial"], "model": model, "prompt_sha256": prompt_sha,
           "overlay_sha256": prep["overlay_sha256"], "backend": backend, "round": 1, "attempts": len(replies)}
    if parsed:
        points_photo = [{**dict(zip(("x", "y"), seg_overlay.to_photo_xy(p["where"], p["x"], p["y"], prep["meta"]))),
                         "label": p["label"]} for p in parsed["points"]]
        row.update(parsed, points_photo=points_photo)
    else:
        row.update(verdict="unjudged", error=err, raw=[r.get("text") for r in replies])
    usage = defaultdict(int)
    for rep in replies:
        for k, v in (rep.get("usage") or {}).items():
            if isinstance(v, int):
                usage[k] += v
    costs = [r["cost_usd"] for r in replies if r.get("cost_usd") is not None]
    row.update(usage=dict(usage), cost_usd=round(sum(costs), 6) if costs else None,
               wall_s=round(sum(r.get("wall_s", 0) for r in replies), 2),
               batch_id=replies[-1].get("batch_id"), ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return row


def require_verdicts(keys: list[tuple[str, str]], model: str, prompt_sha: str,
                     rows: list[dict] | None = None) -> dict[tuple[str, str], dict]:
    """D16: the stored verdict of each (photo_sha256, mask_sha256) for this model and prompt (the latest row
    wins). Raises naming every key with no stored verdict, so a pass rate is never computed on a subset."""
    rows = read_verdicts() if rows is None else rows
    have = {}
    for r in rows:
        if r["model"] == model and r["prompt_sha256"] == prompt_sha:
            have[(r["photo_sha256"], r["mask_sha256"])] = r
    missing = [k for k in keys if k not in have]
    if missing:
        raise LookupError(f"{len(missing)} masks have no verdict for {model} / prompt {prompt_sha[:12]}; "
                          f"judge them first: " + ", ".join(f"{p[:12]}/{m[:12]}" for p, m in missing))
    return {k: have[k] for k in keys}


# ---------------------------------------------------------------- commands

def judge(items: list[dict], model: str, backend: str, batch: bool, rejudge: bool, dry_run: bool,
          workers: int, claude_bin: str, limit: int | None) -> None:
    prompt = build_prompt()
    psha = prompt_sha256(prompt)
    done = {run_key(r) for r in read_verdicts() if not _call_failed(r)}
    todo = [it for it in items
            if rejudge or run_key({**it, "model": model, "prompt_sha256": psha, "backend": backend}) not in done]
    todo = todo[:limit] if limit else todo
    print(f"{len(todo)} of {len(items)} items to judge with {model} via {backend}{' (batch)' if batch else ''}, "
          f"prompt {psha[:12]}")
    preps = {}
    for it in todo:
        preps[it["item"]] = prepare(it)
    if dry_run:
        for it in todo[:3]:
            print(f"  {it['item']}: {message_text(preps[it['item']])}  {preps[it['item']]['dir'].relative_to(REPO_ROOT)}")
        print(f"dry run: overlays ready in {OVERLAY_DIR.relative_to(REPO_ROOT)}; nothing sent")
        return
    if backend == "api" and batch:
        client = _api_client()
        first = call_api_batch(client, prompt, preps, model)
        retry = {cid: preps[cid] for cid, rep in first.items()
                 if "error" in rep or not _parses(rep["text"], preps[cid])}
        second = call_api_batch(client, prompt, retry, model) if retry else {}
        for it in todo:
            cid = it["item"]
            replies = [first[cid]] + ([second[cid]] if cid in second else [])
            append_verdict(make_row(it, preps[cid], model, psha, backend, replies))
        return
    client = _api_client() if backend == "api" else None

    def one(it: dict) -> None:
        prep = preps[it["item"]]
        replies = []
        for _ in range(2):
            try:
                rep = call_cli(prompt, prep, model, claude_bin) if backend == "cli" else call_api(client, prompt, prep, model)
            except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
                rep = {"error": str(e)[:500]}
            replies.append(rep)
            if "error" not in rep and _parses(rep["text"], prep):
                break
            if "error" in rep:
                time.sleep(RETRY_WAIT_S)
        if all("error" in r for r in replies):          # no reply at all (limit, network): not a verdict
            print(f"  {it['item']}: call failed, not stored; rerun to judge it: {replies[-1]['error'][:200]}")
            return
        row = make_row(it, prep, model, psha, backend, replies)
        append_verdict(row)
        print(f"  {it['item']}: {row['verdict']} {row.get('failed_rules', '')}  "
              f"{row['wall_s']:.0f}s  ${row['cost_usd'] if row['cost_usd'] is not None else '?'}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))


def _call_failed(row: dict) -> bool:
    """An unjudged row with no reply text: the call failed (rows stored before calls that fail were
    dropped). Such an item is judged again on the next run; the new row is the latest and wins."""
    return row["verdict"] == "unjudged" and not any(row.get("raw") or [])


def _parses(text: str, prep: dict) -> bool:
    try:
        parse_verdict(text, len(prep["meta"].tiles))
        return True
    except (ValueError, json.JSONDecodeError):
        return False


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - half), min(1.0, c + half))


def score(items: list[dict], model: str, backend: str, prompt_sha: str | None,
          usd_per_mtok: tuple[float, float] | None) -> dict:
    """Catch rate per type on consequential and small defects, false-fail rate on clean items, and cost.
    An unjudged item counts against the judge both ways: not a catch, and a false fail."""
    psha = prompt_sha or prompt_sha256(build_prompt())
    latest = {}
    for r in read_verdicts():
        if r["model"] == model and r["backend"] == backend and r["prompt_sha256"] == psha and not _call_failed(r):
            latest[(r["photo_sha256"], r["mask_sha256"], r["trial"])] = r
    cells = defaultdict(lambda: [0, 0, 0])            # caught / failed, n, unjudged
    missing, rows = [], []
    for it in items:
        r = latest.get((it["photo_sha256"], it["mask_sha256"], it["trial"]))
        if r is None:
            missing.append(it["item"])
            continue
        rows.append(r)
        if it["type"] == "clean":
            cell = ("clean", "false_fail")
        else:
            cell = (it["type"], "consequential" if it["consequential"] == "1" else "small")
        c = cells[cell]
        c[0] += r["verdict"] != "accept"
        c[1] += 1
        c[2] += r["verdict"] == "unjudged"
    table = {f"{t}/{kind}": {"k": k, "n": n, "rate": k / n if n else math.nan, "wilson95": wilson(k, n),
                             "unjudged": u} for (t, kind), (k, n, u) in sorted(cells.items())}
    tok_in = [sum((r["usage"].get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                       "cache_read_input_tokens")) for r in rows]
    tok_out = [r["usage"].get("output_tokens") or 0 for r in rows]
    cost = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    if not cost and usd_per_mtok and rows:                # Message Batches bill at half the list price
        cost = [(0.5 if r.get("batch_id") else 1.0) * (i * usd_per_mtok[0] + o * usd_per_mtok[1]) / 1e6
                for r, i, o in zip(rows, tok_in, tok_out)]
    per_item = float(np.mean(cost)) if cost else None
    return {"model": model, "backend": backend, "prompt_sha256": psha, "judged": len(rows), "missing": missing,
            "table": table, "mean_input_tokens": float(np.mean(tok_in)) if rows else None,
            "mean_output_tokens": float(np.mean(tok_out)) if rows else None,
            "usd_per_item": per_item, "usd_full_run": per_item * FULL_RUN_ITEMS if per_item else None,
            "mean_wall_s": float(np.mean([r["wall_s"] for r in rows])) if rows else None}


def print_score(s: dict) -> None:
    print(f"{s['model']} via {s['backend']}, prompt {s['prompt_sha256'][:12]}: {s['judged']} judged"
          + (f", {len(s['missing'])} missing" if s["missing"] else ""))
    for cell, v in s["table"].items():
        lo, hi = v["wilson95"]
        print(f"  {cell:28s} {v['k']:3d}/{v['n']:<3d} {100 * v['rate']:5.1f}%  [{100 * lo:.0f}, {100 * hi:.0f}]"
              + (f"  unjudged {v['unjudged']}" if v["unjudged"] else ""))
    if s["mean_input_tokens"] is not None:
        print(f"  tokens/item in {s['mean_input_tokens']:.0f} out {s['mean_output_tokens']:.0f};  "
              f"wall {s['mean_wall_s']:.0f} s/item;  "
              + (f"${s['usd_per_item']:.4f}/item, ~${s['usd_full_run']:.0f} for {FULL_RUN_ITEMS} items"
                 if s["usd_per_item"] else "cost: pass --usd-per-mtok IN OUT"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["judge", "score", "prompt"])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--set", choices=list(seg_defects.SETS), help="a planted-defect set")
    src.add_argument("--items", type=Path, help="CSV: item, path, photo_sha256, mask, mask_sha256[, trial]")
    ap.add_argument("--model", default=JUDGE_MODEL, help=f"default: {JUDGE_MODEL} (D15)")
    ap.add_argument("--backend", choices=["cli", "api"], default=JUDGE_BACKEND, help=f"default: {JUDGE_BACKEND} (D15)")
    ap.add_argument("--batch", action="store_true", help="api: Message Batches")
    ap.add_argument("--rejudge", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="render the overlays, send nothing")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    ap.add_argument("--prompt-sha", help="score: a past prompt version (default: the current one)")
    ap.add_argument("--usd-per-mtok", type=float, nargs=2, metavar=("IN", "OUT"),
                    help="score: list prices, for rows without a reported cost (api)")
    ap.add_argument("--out", type=Path, help="score: also write the result as JSON")
    args = ap.parse_args(argv)
    if args.command == "prompt":
        p = build_prompt()
        print(p)
        print(f"\n# sha256 {prompt_sha256(p)}")
        return
    if not (args.set or args.items):
        ap.error("judge/score need --set or --items")
    items = load_items(args.set, args.items)
    if args.command == "judge":
        judge(items, args.model, args.backend, args.batch, args.rejudge, args.dry_run, args.workers,
              args.claude_bin, args.limit)
    else:
        if not args.set:
            ap.error("score needs --set: it reads the defect types from the set's manifest")
        s = score(items, args.model, args.backend, args.prompt_sha,
                  tuple(args.usd_per_mtok) if args.usd_per_mtok else None)
        print_score(s)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(s, indent=2) + "\n")


if __name__ == "__main__":
    main()
