"""Each photo's timeline, and the site's visits, from the logs (ticket OPS-3 D13; GLOSSARY.md "Timeline").

    DOMAIN=<the site> python -m coffeecv.timelines summary [--since 2026-11-10] [--until 2026-11-30]
    DOMAIN=<the site> python -m coffeecv.timelines serve [--port 8766]       # then open http://localhost:8766

Reads both hosts over SSH as their deploy users (--front, --inference: the deploy's aliases, scripts/deploy_webapp.sh)
into memory, joins the lines by request ID and prints the counts and each step's times, or serves them on a page on
127.0.0.1. Nothing is written locally. Four logs are read, each with its rotated and compressed files:

    the front's access log       /var/log/nginx/coffee-cv-front.access    page loads, browsers, referrers, Origins
    the check service's log      /var/log/coffee-cv-front/requests.jsonl  each request's line, and the timelines
    the inference host's access  /var/log/nginx/coffee-cv.access          its nginx's time for each photo
    the app's request log        /var/log/coffee-cv/app.jsonl             decode, segmentation, preview, classification

and never the session log, the one file with the visitor's IP (OPS-15 D20): no source names it, and a source that did
is refused. --from-dir DIR reads the same files from a local directory instead (a copy, or a local rig's logs).

What counts (D17's layer 4, D20): a front access line without the browser, the referrer and the marker is from before
the release, and is skipped; a photo is a check service line, which exists only from the release. A timeline counts
when its request ID is an /analyze request the front logged, and only the first one per ID: another is shown as
suspicious, one whose ID no host knows as unverified, and one with no ID (a photo refused on the page, or abandoned
before its answer began) as unverifiable. The deploy's own requests (X-Coffee-CV-Probe: verify, D14) are left out
everywhere. Every string a visitor sent is printed without control characters, and the page shows it as text,
never as HTML (D17's layer 5).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from coffeecv import timeline_schema

# Each log: the host it is on, its directory there, its file name (rotated: name.1, name.2.gz, ...).
SOURCES = {
    "front_access": ("front", "/var/log/nginx", "coffee-cv-front.access"),
    "service": ("front", "/var/log/coffee-cv-front", "requests.jsonl"),
    "inference_access": ("inference", "/var/log/nginx", "coffee-cv.access"),
    "app": ("inference", "/var/log/coffee-cv", "app.jsonl"),
}
SESSION_LOG = "coffee-cv-front.sessions"   # never read (OPS-15 D20)
PROBE = timeline_schema.PROBE
CONTROLS = timeline_schema.CONTROLS
CANONICAL = ("canonicalized", "as_sent")
CRAWLER = re.compile(r"bot|crawl|spider|slurp|preview|fetch|scan|curl|wget|python|go-http|java/|okhttp|httpx|axios|"
                     r"headless|lighthouse|pagespeed|monitor|^-$|^$", re.I)
SIZE_BUCKETS = ((1, "< 1 MB"), (5, "1-5 MB"), (10, "5-10 MB"), (20, "10-20 MB"), (40, "20-40 MB"),
                (float("inf"), "over 40 MB"))


def clean(text, cap: int = 200) -> str:
    """A visitor's string, safe to print: no control or format characters, at most `cap` characters."""
    text = "".join(ch for ch in str(text) if unicodedata.category(ch)[0] != "C")
    return text if len(text) <= cap else text[:cap - 1] + "…"


# ------------------------------------------------------------------------------------------------- reading

def check_source(name: str) -> tuple[str, str, str]:
    host, directory, filename = SOURCES[name]
    if SESSION_LOG in filename or SESSION_LOG in directory:
        raise SystemExit(f"refusing to read the session log ({name}): it holds the visitor's IP (OPS-15 D20)")
    return host, directory, filename


def remote_command(directory: str, filename: str) -> str:
    """The shell command that prints a log and its rotated files, oldest first, uncompressed."""
    d, f = shlex.quote(directory), shlex.quote(filename)
    return (f"cd {d} && for n in $(ls -1 {f} {f}.[0-9]* 2>/dev/null | sort -t. -k3,3nr); do "
            f"zcat -f \"$n\"; done")


def read_ssh(hosts: dict[str, str], run=subprocess.run) -> dict[str, list[str]]:
    """Each source's lines, read over SSH from its host's alias, into memory."""
    logs = {}
    for name in SOURCES:
        host, directory, filename = check_source(name)
        out = run(["ssh", "-o", "BatchMode=yes", hosts[host], remote_command(directory, filename)],
                  capture_output=True, text=True, errors="replace")
        if out.returncode != 0 and not out.stdout:
            print(f"could not read {name} on {hosts[host]}: {clean(out.stderr.strip())}", file=sys.stderr)
        logs[name] = out.stdout.splitlines()
    return logs


def read_dir(root: Path) -> dict[str, list[str]]:
    """Each source's lines from a local directory holding the same file names (gzipped or not)."""
    logs = {}
    for name in SOURCES:
        _, _, filename = check_source(name)
        lines: list[str] = []
        for path in sorted(root.glob(f"{filename}*"), key=lambda p: p.name, reverse=True):
            if path.name != filename and not re.fullmatch(re.escape(filename) + r"\.\d+(\.gz)?", path.name):
                continue
            data = path.read_bytes()
            lines += (gzip.decompress(data) if path.suffix == ".gz" else data).decode("utf-8", "replace").splitlines()
        logs[name] = lines
    return logs


# ------------------------------------------------------------------------------------------------- parsing

def _day(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(timezone.utc).date().isoformat()


def parse_front_access(lines: list[str]) -> list[dict]:
    """The front's access lines from the release on: those with the browser, the referrer and the marker (D20)."""
    out = []
    for line in lines:
        f = re.findall(r'"[^"]*"|\S+', line)
        if len(f) != 11:
            continue   # before the release: no browser, referrer or marker
        request = f[4].strip('"').split(" ")
        try:
            out.append({"day": _day(f[0]), "ts": f[0], "id": f[1], "upstream_id": f[2], "status": int(f[3]),
                        "method": request[0], "path": request[1] if len(request) > 1 else "",
                        "seconds": float(f[5].rstrip("s")), "origin": f[7].strip('"'), "browser": f[8].strip('"'),
                        "referrer": f[9].strip('"'), "probe": f[10].strip('"') == PROBE})
        except (ValueError, IndexError):
            continue
    return out


def parse_inference_access(lines: list[str]) -> dict[str, float]:
    """The inference host's nginx time for each request ID, in ms: the crossing's arrival and the app's work."""
    out = {}
    for line in lines:
        f = re.findall(r'"[^"]*"|\S+', line)
        if len(f) >= 5 and timeline_schema.REQUEST_ID.fullmatch(f[1]):
            try:
                out[f[1]] = round(float(f[4].rstrip("s")) * 1000)
            except ValueError:
                pass
    return out


def _json_lines(lines: list[str]) -> list[dict]:
    out = []
    for line in lines:
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


# ------------------------------------------------------------------------------------------------- joining

@dataclass
class Photo:
    """One photo: its counted timeline (none for a photo no page reported) and the hosts' lines for its ID."""
    request_id: str
    day: str
    service: dict
    timeline: dict | None = None
    app: dict = field(default_factory=dict)
    inference_ms: int | None = None

    @property
    def control(self) -> str:
        return self.timeline["control"] if self.timeline else "no timeline"

    @property
    def canonical(self) -> str | None:
        return self.service.get("canonical")

    @property
    def outcome(self) -> str:
        return self.timeline["outcome"] if self.timeline else f"({self.service.get('outcome', '?')})"


def _service_preview_at(s: dict) -> int | None:
    """When the preview left the check service, from its start: its read, checks, canonicalization and crossing, then
    its wait for the preview."""
    parts = [s.get("read_ms"), s.get("check_ms"), s.get("canonicalize_ms", 0), s.get("crossing_ms"),
             s.get("to_preview_ms")]
    return None if None in parts else sum(parts)


def _diff(a, b):
    return None if a is None or b is None else a - b


def _yappy_received(p: "Photo") -> int | None:
    """The crossing as yappy saw it: its nginx's time, from the request's first byte to the answer's end, less the
    app's, which starts once that nginx has the whole upload and ends with the answer."""
    return _diff(p.inference_ms, p.app.get("latency_ms"))


def _preview_trip(p: "Photo") -> int | None:
    """The preview's way back to the front, and the upload's first byte's way out: the front's time from its connect to
    the preview going on, less yappy's from that first byte to the preview leaving the app. One round trip, roughly."""
    s = p.service
    front = None if None in (s.get("crossing_ms"), s.get("to_preview_ms")) else s["crossing_ms"] + s["to_preview_ms"]
    yappy = _yappy_received(p)
    return _diff(front, None if yappy is None or p.app.get("preview_ms") is None else yappy + p.app["preview_ms"])


def _last_attempt(t: dict) -> dict:
    return t["attempts"][-1] if t and t.get("attempts") else {}


# The ticket's §2 steps, each from the line that measured it (D6). A network leg (~) is what one side saw minus what
# the other did, each on its own clock: approximate, and kept negative when it comes out so, since dropping those would
# bias its median up. The crossing has two: what the front's socket took (9, until its buffers hold the last byte), and
# what yappy received (9y); the rest of the first is still on the wire (OPS-3's phone check, 2026-11-09: 5.6 s sent,
# 9.3 s received). The page's "upload sent" is the browser's upload load event, which Chromium fires on a tick of about
# 100 ms (OPS-3 P4: every upload_ms in the local check was 57-102 ms), so a short leg can read up to that much low.
STEPS: tuple[tuple[str, str, object], ...] = (
    ("1", "pick: tap to file (page)", lambda p: p.timeline and p.timeline.get("pick_ms")),
    ("2", "example downloaded (page)", lambda p: p.timeline and p.timeline.get("download_ms")),
    ("3", "size check (page)", lambda p: p.timeline and p.timeline.get("size_check_ms")),
    ("4", "connection setup (page)", lambda p: _last_attempt(p.timeline).get("connect_ms")),
    ("5", "upload, browser to proxy (page)", lambda p: _last_attempt(p.timeline).get("upload_ms")),
    ("6", "busy waits (page)", lambda p: sum(a.get("wait_ms", 0) for a in p.timeline["attempts"]) or None
        if p.timeline else None),
    ("7", "front's checks (check service)", lambda p: p.service.get("check_ms")),
    ("8", "canonicalization (check service)", lambda p: p.service.get("canonicalize_ms")),
    ("9", "crossing, sent (check service)", lambda p: p.service.get("crossing_ms")),
    ("9y", "crossing, received at yappy (~)", _yappy_received),
    ("10", "decode (app)", lambda p: p.app.get("decode_ms")),
    ("11", "segmentation (app)", lambda p: p.app.get("crop_detect_ms")),
    ("12", "preview built (app)", lambda p: p.app.get("preview_build_ms")),
    ("13a", "preview, yappy to proxy, a round trip (~)", _preview_trip),
    ("13b", "preview, proxy to browser (~)", lambda p: _diff(p.timeline and p.timeline.get("to_preview_ms"),
                                                            _service_preview_at(p.service))),
    ("14", "preview drawn (page)", lambda p: p.timeline and p.timeline.get("preview_drawn_ms")),
    ("15", "classification (app)", lambda p: p.app.get("inference_ms")),
    ("16", "preview to answer (page)", lambda p: _diff(p.timeline and p.timeline.get("to_answer_ms"),
                                                       p.timeline and p.timeline.get("to_preview_ms"))),
    ("16f", "preview to answer's end (check service)", lambda p: _diff(p.service.get("to_end_ms"),
                                                                       p.service.get("to_preview_ms"))),
    ("16s", "answer shown (page)", lambda p: p.timeline and p.timeline.get("answer_shown_ms")),
    ("17", "total: pick to answer on screen (page)", lambda p: p.timeline and p.timeline.get("total_ms")),
    ("17f", "total at the front (check service)", lambda p: p.service.get("total_ms")),
    ("17n", "total at yappy's nginx", lambda p: p.inference_ms),
    ("17a", "total in the app", lambda p: p.app.get("latency_ms")),
)


@dataclass
class Report:
    days: dict = field(default_factory=dict)
    photos: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    failures: dict = field(default_factory=dict)
    origins: dict = field(default_factory=dict)
    flagged: dict = field(default_factory=dict)
    dropped: int = 0
    probes: int = 0


def _stat(values: list[int]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(ordered), "median": round(statistics.median(ordered)),
            "p90": ordered[min(len(ordered) - 1, max(0, round(0.9 * len(ordered)) - 1))]}


def _size_bucket(size: int) -> str:
    return next(label for top, label in SIZE_BUCKETS if size < top * 2**20)


def build(logs: dict[str, list[str]], domain: str, since: str | None = None, until: str | None = None) -> Report:
    """Joins the four logs (D17's layer 4) and counts them (D13), for the days from `since` to `until`."""
    def in_period(day: str) -> bool:
        return (since is None or day >= since) and (until is None or day <= until)

    own = f"https://{domain}"
    access = [a for a in parse_front_access(logs.get("front_access", [])) if in_period(a["day"])]
    service = [s for s in _json_lines(logs.get("service", [])) if "ts" in s and in_period(_day(s["ts"]))]
    app = {a["request_id"]: a for a in _json_lines(logs.get("app", []))
           if a.get("msg") == "request" and isinstance(a.get("request_id"), str)}
    nginx_ms = parse_inference_access(logs.get("inference_access", []))
    report = Report()

    probe_ids = {a["id"] for a in access if a["probe"]} | {s.get("request_id") for s in service if s.get("probe")}
    analyze_ids = {a["id"] for a in access if a["path"].split("?")[0] == "/analyze" and a["method"] == "POST"}
    requests = {s["request_id"]: s for s in service if s.get("kind") == "request" and s.get("endpoint") == "/analyze"}
    report.dropped = sum(s.get("timelines_dropped", 0) for s in service if isinstance(s.get("timelines_dropped"), int))

    # Timelines: each checked again against the schema (the file is trusted no more than the page), the deploy's
    # left out, then counted, flagged or set aside.
    flagged: dict[str, list] = {"unverified": [], "suspicious": [], "unverifiable": [], "malformed": 0}
    counted: dict[str, dict] = {}
    for s in service:
        if s.get("kind") != "timeline":
            continue
        try:   # any example name: the check service held it to the release's list when it wrote the line
            t = timeline_schema.check(s.get("timeline"), None)
        except timeline_schema.Invalid:
            flagged["malformed"] += 1
            continue
        if s.get("probe") or t.get("probe") or s.get("request_id") in probe_ids or t["request_id"] in probe_ids:
            report.probes += 1
            continue
        entry = {"day": _day(s["ts"]), "control": t["control"], "outcome": t["outcome"],
                 "request_id": t["request_id"], "example": t.get("example")}
        if t["request_id"] is None:
            flagged["unverifiable"].append(entry)
        elif t["request_id"] not in analyze_ids and t["request_id"] not in requests:
            flagged["unverified"].append(entry)
        elif t["request_id"] in counted:
            flagged["suspicious"].append(entry)
        else:
            counted[t["request_id"]] = t
    report.flagged = flagged

    # Photos: every /analyze the check service logged that is not the deploy's, and not an earlier attempt of a
    # photo whose timeline names it (a busy retry's), each with its timeline if one counted.
    earlier = {a.get("request_id") for t in counted.values() for a in t["attempts"][:-1]}
    for rid, s in requests.items():
        if rid in probe_ids or (rid in earlier and rid not in counted):
            if rid in probe_ids:
                report.probes += 1
            continue
        report.photos.append(Photo(rid, _day(s["ts"]), s, counted.get(rid), app.get(rid, {}), nginx_ms.get(rid)))
    # And every photo whose timeline counted but which the front's nginx answered itself, so the check service never
    # logged it: busy at the gate, offline with the check service stopped, over the body cap. The access line stands in.
    by_id = {a["id"]: a for a in access}
    for rid, t in counted.items():
        if rid not in requests and rid in by_id:
            a = by_id[rid]
            stub = {"ts": a["ts"], "status": a["status"], "outcome": "refused at the front's nginx"}
            report.photos.append(Photo(rid, a["day"], stub, t))

    # Per day: page loads by people, external referrers, photos by control, example, format and size.
    days: dict[str, dict] = defaultdict(lambda: {"page_loads": 0, "crawler_loads": 0, "referrers": Counter(),
                                                 "by_control": Counter(), "by_example": Counter(),
                                                 "by_format": Counter(), "by_size": Counter(), "timelines": 0})
    for a in access:
        if a["probe"] or a["method"] != "GET" or a["path"].split("?")[0] not in ("/", "/index.html") \
                or a["status"] not in (200, 304):
            continue
        d = days[a["day"]]
        if CRAWLER.search(a["browser"]):
            d["crawler_loads"] += 1
            continue
        d["page_loads"] += 1
        if a["referrer"] in ("-", "") or a["referrer"].startswith(own):
            continue
        try:
            host = urlsplit(a["referrer"]).netloc
        except ValueError:   # a referrer no URL parser reads: a forged one
            host = "(unreadable)"
        if host:
            d["referrers"][clean(host, 80)] += 1
    for p in report.photos:
        d = days[p.day]
        d["by_control"][p.control] += 1
        d["timelines"] += p.timeline is not None
        if p.timeline and p.timeline.get("example"):
            d["by_example"][p.timeline["example"]] += 1
        d["by_format"][clean(p.service.get("format") or f"({p.service.get('ext') or 'none'})", 20)] += 1
        if isinstance(p.service.get("bytes"), int):
            d["by_size"][_size_bucket(p.service["bytes"])] += 1
    report.days = {day: {k: dict(v.most_common()) if isinstance(v, Counter) else v for k, v in d.items()}
                   for day, d in sorted(days.items())}

    # Each step's median and 90th percentile, all, by control and by canonicalized or not (D13).
    groups = {"all": lambda p: True, **{c: (lambda c: lambda p: p.control == c)(c) for c in CONTROLS},
              **{c: (lambda c: lambda p: p.canonical == c)(c) for c in CANONICAL}}
    for key, label, measure in STEPS:
        row = {"step": key, "label": label}
        for name, member in groups.items():
            values = [v for p in report.photos if member(p) and isinstance(v := measure(p), int)]
            row[name] = _stat(values)
        report.steps.append(row)

    # Failures by kind: the page's outcomes, and the check service's.
    report.failures = {
        "page": dict(Counter(p.outcome for p in report.photos if p.timeline and p.outcome != "answered")
                     + Counter(e["outcome"] for e in flagged["unverifiable"])),
        "front": dict(Counter(s.get("outcome", "?") for s in service if s.get("kind") == "request"
                              and s.get("endpoint") == "/analyze" and s.get("outcome") != "relayed"
                              and not s.get("probe"))),
    }
    report.origins = dict(Counter(clean(a["origin"], 80) for a in access if a["path"].split("?")[0] == "/analyze"
                                  and a["origin"] not in ("-", own) and not a["probe"]).most_common())
    return report


# ------------------------------------------------------------------------------------------------- output

def photo_view(p: Photo) -> dict:
    """One photo for the page: its fields as text and numbers, its steps in order."""
    t, s = p.timeline or {}, p.service
    return {"request_id": p.request_id, "day": p.day, "ts": clean(s.get("ts", ""), 40), "control": p.control,
            "example": t.get("example"), "outcome": p.outcome, "rerun_of": t.get("rerun_of"),
            "filename": clean(s.get("filename", ""), 200), "format": clean(s.get("format") or "", 20),
            "ext": clean(s.get("ext") or "", 16), "bytes": s.get("bytes"), "canonical": s.get("canonical"),
            "status": s.get("status"), "front_outcome": clean(s.get("outcome", ""), 40),
            "attempts": len(t.get("attempts", [])) or None, "verdict": clean(p.app.get("verdict") or "", 40),
            "steps": [{"step": key, "label": label, "ms": v if isinstance(v := measure(p), int) else None}
                      for key, label, measure in STEPS]}


def as_json(report: Report) -> dict:
    return {"days": report.days, "steps": report.steps, "failures": report.failures, "origins": report.origins,
            "flagged": {k: v if isinstance(v, int) else v for k, v in report.flagged.items()},
            "dropped": report.dropped, "probes": report.probes,
            "photos": [photo_view(p) for p in sorted(report.photos, key=lambda p: p.service.get("ts", ""),
                                                      reverse=True)]}


def _cell(stat: dict | None) -> str:
    return f"{stat['median']}/{stat['p90']} ({stat['n']})" if stat else "-"


def summary(report: Report) -> str:
    """The counts and the steps, as text; every string a visitor sent already cleaned."""
    out = []
    w = out.append
    w("Per day (UTC): page loads by people (and by crawlers), photos, timelines")
    for day, d in report.days.items():
        w(f"  {day}  loads {d['page_loads']} (+{d['crawler_loads']} crawlers)  photos "
          f"{sum(d['by_control'].values())}  timelines {d['timelines']}")
        for key, title in (("by_control", "control"), ("by_example", "example"), ("by_format", "format"),
                           ("by_size", "size"), ("referrers", "referrers")):
            if d[key]:
                w(f"      {title:10} " + ", ".join(f"{clean(k, 80)} {n}" for k, n in d[key].items()))
    columns = ["all", *CONTROLS, *CANONICAL]
    w("")
    w("Each step, ms: median/p90 (photos)")
    w(f"  {'step':45}" + "".join(f"{c:>20}" for c in columns))
    for row in report.steps:
        w(f"  {row['step'] + ' ' + row['label']:45}" + "".join(f"{_cell(row[c]):>20}" for c in columns))
    w("")
    w("Failures by kind")
    for side, counts in report.failures.items():
        w(f"  {side:6} " + (", ".join(f"{clean(k, 40)} {n}" for k, n in sorted(counts.items())) or "none"))
    w("")
    w("Other sites' calls to /analyze, by Origin")
    w("  " + (", ".join(f"{k} {n}" for k, n in report.origins.items()) or "none"))
    w("")
    f = report.flagged
    w(f"Timelines not counted: {len(f['unverified'])} unverified (no host knows the ID), {len(f['suspicious'])} "
      f"suspicious (a second for an ID), {len(f['unverifiable'])} unverifiable (no ID: "
      + (", ".join(f"{k} {n}" for k, n in Counter(e['outcome'] for e in f['unverifiable']).items()) or "none")
      + f"), {f['malformed']} malformed; {report.dropped} dropped past the daily cap; {report.probes} of the "
      f"deploy's own left out")
    return "\n".join(out)


# The page: static, no data in it. It fetches /report.json and builds every element with textContent, so nothing a
# visitor sent is ever read as HTML (D17's layer 5).
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Coffee CV timelines</title>
<style>
 body { font: 14px system-ui, sans-serif; margin: 1.5em; color: #222; }
 table { border-collapse: collapse; margin: .5em 0 1.5em; }
 td, th { border: 1px solid #ccc; padding: 2px 6px; text-align: right; }
 td:first-child, th:first-child { text-align: left; }
 tr.photo { cursor: pointer; } tr.photo:hover { background: #f3efe8; }
 .bar { display: inline-block; height: 10px; background: #6f4527; vertical-align: middle; }
 h2 { margin-top: 1.5em; }
</style></head>
<body>
<h1>Coffee CV timelines</h1>
<div id="out">Loading...</div>
<script>
const out = document.getElementById('out');
function el(tag, text, attrs) {
  const e = document.createElement(tag);
  if (text !== undefined && text !== null) e.textContent = String(text);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
}
function table(head, rows) {
  const t = el('table');
  const h = el('tr');
  for (const c of head) h.append(el('th', c));
  t.append(h);
  for (const r of rows) {
    const tr = el('tr');
    for (const c of r) tr.append(c instanceof Node ? (() => { const td = el('td'); td.append(c); return td; })() : el('td', c));
    t.append(tr);
  }
  return t;
}
const cell = s => s ? `${s.median}/${s.p90} (${s.n})` : '-';
const pairs = o => Object.entries(o || {}).map(([k, n]) => `${k} ${n}`).join(', ');
fetch('/report.json').then(r => r.json()).then(rep => {
  out.textContent = '';
  out.append(el('h2', 'Per day (UTC)'));
  out.append(table(['day', 'page loads', 'crawlers', 'photos', 'timelines', 'by control', 'by example', 'by format',
                    'by size', 'referrers'],
    Object.entries(rep.days).map(([day, d]) => [day, d.page_loads, d.crawler_loads,
      Object.values(d.by_control).reduce((a, b) => a + b, 0), d.timelines, pairs(d.by_control), pairs(d.by_example),
      pairs(d.by_format), pairs(d.by_size), pairs(d.referrers)])));
  const cols = ['all', 'choose_photo', 'select_file', 'camera', 'example', 'canonicalized', 'as_sent'];
  out.append(el('h2', 'Each step, ms: median/p90 (photos)'));
  out.append(table(['step', ...cols], rep.steps.map(s => [`${s.step} ${s.label}`, ...cols.map(c => cell(s[c]))])));
  out.append(el('h2', 'Failures by kind'));
  out.append(table(['side', 'kinds'], Object.entries(rep.failures).map(([k, v]) => [k, pairs(v) || 'none'])));
  out.append(el('h2', "Other sites' calls to /analyze, by Origin"));
  out.append(el('p', pairs(rep.origins) || 'none'));
  const f = rep.flagged;
  out.append(el('h2', 'Timelines not counted'));
  out.append(el('p', `${f.unverified.length} unverified, ${f.suspicious.length} suspicious, ` +
    `${f.unverifiable.length} unverifiable, ${f.malformed} malformed; ${rep.dropped} dropped past the daily cap; ` +
    `${rep.probes} of the deploy's own left out`));
  out.append(el('h2', `Photos (${rep.photos.length}), newest first: click one for its steps`));
  const detail = el('div');
  const list = table(['time', 'control', 'example', 'outcome', 'format', 'MB', 'canonical', 'filename', 'ID'], []);
  for (const p of rep.photos) {
    const tr = el('tr', undefined, {class: 'photo'});
    for (const c of [p.ts, p.control, p.example || '', p.outcome, p.format || p.ext, p.bytes == null ? '' :
                     (p.bytes / 1048576).toFixed(1), p.canonical || '', p.filename, p.request_id]) tr.append(el('td', c));
    tr.addEventListener('click', () => showPhoto(p, detail));
    list.append(tr);
  }
  out.append(detail, list);
}).catch(err => { out.textContent = `Could not load the report: ${err}`; });
function showPhoto(p, where) {
  where.textContent = '';
  where.append(el('h3', `${p.request_id}: ${p.control}, ${p.outcome}` + (p.rerun_of ? `, a re-run of ${p.rerun_of}` : '')));
  const max = Math.max(1, ...p.steps.map(s => s.ms || 0));
  where.append(table(['step', 'ms', ''], p.steps.filter(s => s.ms !== null).map(s =>
    [`${s.step} ${s.label}`, s.ms, el('span', undefined, {class: 'bar', style: `width:${Math.round(300 * s.ms / max)}px`})])));
}
</script>
</body></html>
"""


def make_server(report: Report, host: str, port: int) -> ThreadingHTTPServer:
    """The page and its data, on `host`:`port` (127.0.0.1 by default: the data is the visitors')."""
    data = json.dumps(as_json(report)).encode()
    page = PAGE.encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body, kind = {"/": (page, "text/html; charset=utf-8"),
                          "/report.json": (data, "application/json")}.get(self.path, (None, None))
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; "
                                                        "style-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(report: Report, host: str, port: int) -> None:
    server = make_server(report, host, port)
    print(f"serving {len(report.photos)} photos on http://{host}:{port}/ (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, help_text in (("summary", "print the counts and the steps"), ("serve", "show them on a local page")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--front", default=os.environ.get("FRONT_HOST", "webproxy-deploy"),
                       help="the front's SSH alias, as the deploy's user (default: $FRONT_HOST, else webproxy-deploy)")
        p.add_argument("--inference", default=os.environ.get("HOST", "webappremote"),
                       help="the inference host's SSH alias (default: $HOST, else webappremote)")
        p.add_argument("--from-dir", type=Path, help="read the logs from this local directory instead of over SSH")
        p.add_argument("--domain", default=os.environ.get("DOMAIN"), help="the site's host name (default: $DOMAIN)")
        p.add_argument("--since", help="the first day counted, YYYY-MM-DD (default: everything since the release)")
        p.add_argument("--until", help="the last day counted, YYYY-MM-DD")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8766)
    args = ap.parse_args(argv)
    if not args.domain:
        ap.error("--domain or DOMAIN is required: it tells the site's own page loads and Origin from others'")
    logs = read_dir(args.from_dir) if args.from_dir else read_ssh({"front": args.front, "inference": args.inference})
    report = build(logs, args.domain, args.since, args.until)
    if args.command == "summary":
        print(summary(report))
    else:
        serve(report, args.host, args.port)


if __name__ == "__main__":
    main()
