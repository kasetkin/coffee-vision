"""Each photo's timeline, and the site's visits, from the logs (ticket OPS-3 D13; GLOSSARY.md "Timeline").

    DOMAIN=<the site> python -m coffeecv.timelines summary [--since 2026-11-10] [--until 2026-11-30]
    DOMAIN=<the site> python -m coffeecv.timelines serve [--port 8766]       # then open http://localhost:8766

serve reads everything since the release once; its page picks the period, filters the photos, sorts them and draws
each one's waterfall in the browser (P11, D25). Its "Reload logs" button reads the hosts again (D26).

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
import threading
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
    front_ms: int | None = None   # the front's nginx, from the request's first byte to its end (D28)

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


def _selected_to_answer(p: "Photo") -> int | None:
    """The headline time (D27): from the photo chosen to the outcome on screen, the page's total less the pick (the
    tap until the file arrived, the visitor choosing in the picker). An example or a re-run has no pick: its total
    starts at the click. The pick also holds the phone handing the file to the browser; no event parts the two."""
    t = p.timeline
    if not t or not isinstance(t.get("total_ms"), int):
        return None
    return t["total_ms"] - (t.get("pick_ms") or 0)


def _last_attempt(t: dict) -> dict:
    return t["attempts"][-1] if t and t.get("attempts") else {}


def _upload_at_front(p: "Photo") -> int | None:
    """Step 5 (D28): the upload as the front received it, on the front's one clock: its nginx's time, from the
    request's first byte, less the check service's, which starts once that nginx holds the whole upload. The browser's
    own "upload sent" is no measure of it: Chromium fires it when the bytes are in its socket buffer (2.2 MB in 102 ms)
    or only with the answer's status (upload_ms 20836, the status 2 ms later; OPS-3 P11, 2026-11-09)."""
    return _diff(p.front_ms, p.service.get("total_ms"))


def _front_start(p: "Photo") -> int | None:
    """When the check service started, from the last attempt's start on the page's clock (D28): the connection set
    up, then the upload received at the front. Its first byte's way out is not counted, so this is early by about half
    a round trip: the upload's own leg (5) is then what the front saw, and 13b a little long."""
    if not p.timeline:
        return None
    upload = _upload_at_front(p)
    return None if upload is None else (_last_attempt(p.timeline).get("connect_ms") or 0) + upload


def _preview_to_browser(p: "Photo") -> int | None:
    """Step 13b: the preview's way from the front to the page. The page's preview arrival, from the last attempt's
    start (its upload_ms and to_preview_ms both count from the browser's event, so its lateness cancels), less the
    front's, from the same start (D28)."""
    t, start = p.timeline, _front_start(p)
    att = _last_attempt(t)
    if start is None or not isinstance(t.get("to_preview_ms"), int) or not isinstance(att.get("upload_ms"), int):
        return None
    front_preview = _service_preview_at(p.service)
    return None if front_preview is None else att["upload_ms"] + t["to_preview_ms"] - (start + front_preview)


# The ticket's §2 steps, each from the line that measured it (D6). A network leg (~) is what one side saw minus what
# the other did, each on its own clock: approximate, and kept negative when it comes out so, since dropping those would
# bias its median up. The crossing has two: what the front's socket took (9, until its buffers hold the last byte), and
# what yappy received (9y); the rest of the first is still on the wire (OPS-3's phone check, 2026-11-09: 5.6 s sent,
# 9.3 s received). The upload (5) is the front's, not the browser's "upload sent" event, which fires early or late
# (_upload_at_front); the page's times that count from that event are only ever used as differences.
STEPS: tuple[tuple[str, str, object], ...] = (
    ("1", "pick: tap to file (page)", lambda p: p.timeline and p.timeline.get("pick_ms")),
    ("2", "example downloaded (page)", lambda p: p.timeline and p.timeline.get("download_ms")),
    ("3", "size check (page)", lambda p: p.timeline and p.timeline.get("size_check_ms")),
    ("4", "connection setup (page)", lambda p: _last_attempt(p.timeline).get("connect_ms")),
    ("5", "upload, received at the front", _upload_at_front),
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
    ("13b", "preview, proxy to browser (~)", _preview_to_browser),
    ("14", "preview drawn (page)", lambda p: p.timeline and p.timeline.get("preview_drawn_ms")),
    ("15", "classification (app)", lambda p: p.app.get("inference_ms")),
    ("16", "preview to answer (page)", lambda p: _diff(p.timeline and p.timeline.get("to_answer_ms"),
                                                       p.timeline and p.timeline.get("to_preview_ms"))),
    ("16f", "preview to answer's end (check service)", lambda p: _diff(p.service.get("to_end_ms"),
                                                                       p.service.get("to_preview_ms"))),
    ("16s", "answer shown (page)", lambda p: p.timeline and p.timeline.get("answer_shown_ms")),
    ("17s", "total: photo selected to answer on screen (page)", lambda p: _selected_to_answer(p)),
    ("17", "total: pick to answer on screen (page)", lambda p: p.timeline and p.timeline.get("total_ms")),
    ("17f", "total at the front (check service)", lambda p: p.service.get("total_ms")),
    ("17n", "total at yappy's nginx", lambda p: p.inference_ms),
    ("17a", "total in the app", lambda p: p.app.get("latency_ms")),
)


LABELS = {key: label for key, label, _ in STEPS}



def waterfall(p: Photo) -> list[dict]:
    """One photo's measured steps on one axis (P11, D25): each bar's start and length in ms from the tap, and its lane,
    the page, the front, yappy or the network between them. The hosts' clocks are tied to the page's at two points
    (D28): the front's nginx starts when the last attempt's connection is set up (its first byte's way out not
    counted), and yappy's nginx when the front's crossing starts. The page's own times count from the browser's
    "upload sent", which fires early or late (_upload_at_front), so they are placed from the total less the answer's
    wait and drawing when the photo was answered, else from the sum of the page's steps before them, which misses the
    gaps between them; either way they agree with each other. A photo with no timeline starts at the front."""
    bars: list[dict] = []

    def bar(step: str, lane: str, start, ms, label: str | None = None) -> None:
        if isinstance(start, int) and isinstance(ms, int):
            bars.append({"step": step, "label": label or LABELS.get(step, ""), "lane": lane, "start": start, "ms": ms})

    def total(*parts):
        return None if any(not isinstance(x, int) for x in parts) else sum(parts)

    t, s, a = p.timeline or {}, p.service, p.app
    sent = 0   # the check service's start on the page's clock; for a photo with no timeline, the axis's 0
    if p.timeline:
        bar("1", "page", 0, t.get("pick_ms"))
        bar("2", "network", 0, t.get("download_ms"))
        cursor = max(t.get("pick_ms") or 0, t.get("download_ms") or 0)
        bar("3", "page", cursor, t.get("size_check_ms"))
        cursor += t.get("size_check_ms") or 0
        attempts = t.get("attempts") or [{}]
        n = len(attempts)
        for i, att in enumerate(attempts[:-1], 1):
            suffix = f", attempt {i} of {n}"
            bar("4", "network", cursor, att.get("connect_ms"), LABELS["4"] + suffix)
            bar("5", "network", cursor, att.get("upload_ms"), "upload, the browser's event" + suffix)
            cursor += att.get("upload_ms") or 0
            bar("", "network", cursor, att.get("headers_ms"), "upload sent to the busy answer" + suffix)
            cursor += att.get("headers_ms") or 0
            bar("6", "page", cursor, att.get("wait_ms"), LABELS["6"] + suffix)
            cursor += att.get("wait_ms") or 0
        last = attempts[-1]
        whole, to_answer = t.get("total_ms"), t.get("to_answer_ms")
        if p.outcome == "answered" and isinstance(whole, int) and isinstance(to_answer, int):
            sent = whole - to_answer - (t.get("answer_shown_ms") or 0)
        else:
            sent = cursor + (last.get("upload_ms") or 0)
        start = sent - (last.get("upload_ms") or 0)
        suffix = f", attempt {n} of {n}" if n > 1 else ""
        bar("4", "network", start, last.get("connect_ms"), (LABELS["4"] + suffix) if suffix else None)
        page_preview = total(sent, t.get("to_preview_ms"))
        bar("14", "page", page_preview, t.get("preview_drawn_ms"))
        bar("16", "page", page_preview, _diff(t.get("to_answer_ms"), t.get("to_preview_ms")))
        bar("16s", "page", total(sent, t.get("to_answer_ms")), t.get("answer_shown_ms"))
        bar("17", "page", 0, t.get("total_ms"))
        bar("17s", "page", t.get("pick_ms") or 0, _selected_to_answer(p))
        # The front on the page's clock (D28): the upload's first byte at the front's nginx as the connection is up,
        # the check service once that nginx holds the whole upload.
        nginx_start = start + (last.get("connect_ms") or 0)
        bar("5", "network", nginx_start, _upload_at_front(p), (LABELS["5"] + suffix) if suffix else None)
        front_start = _front_start(p)
        sent = start + front_start if front_start is not None else sent   # no front time: the browser's event
    else:
        page_preview = None

    # The front, from its start: its read, checks, canonicalization, then the crossing, then the answer relayed.
    bar("", "front", sent, s.get("read_ms"), "read of the upload (check service)")
    checks = total(sent, s.get("read_ms"))
    bar("7", "front", checks, s.get("check_ms"))
    canonical = total(checks, s.get("check_ms"))
    bar("8", "front", canonical, s.get("canonicalize_ms"))
    crossing = total(canonical, s.get("canonicalize_ms", 0))
    bar("9", "network", crossing, s.get("crossing_ms"))
    front_preview = total(crossing, s.get("crossing_ms"), s.get("to_preview_ms"))
    bar("16f", "front", front_preview, _diff(s.get("to_end_ms"), s.get("to_preview_ms")))
    bar("17f", "front", sent, s.get("total_ms"))

    # Yappy, from the crossing's start: its nginx, then the app once that nginx holds the whole upload.
    bar("17n", "yappy", crossing, p.inference_ms)
    received = _yappy_received(p)
    bar("9y", "network-approx", crossing, received)
    app_start = total(crossing, received)
    bar("17a", "yappy", app_start, a.get("latency_ms"))
    app_preview = total(app_start, a.get("preview_ms"))
    if app_preview is not None:   # decode, segmentation and the preview's build end as the preview goes out
        build_at = total(app_preview, -(a.get("preview_build_ms") or 0))
        segment_at = total(build_at, -(a.get("crop_detect_ms") or 0))
        bar("10", "yappy", total(segment_at, -(a.get("decode_ms") or 0)), a.get("decode_ms"))
        bar("11", "yappy", segment_at, a.get("crop_detect_ms"))
        bar("12", "yappy", build_at, a.get("preview_build_ms"))
        bar("15", "yappy", app_preview, a.get("inference_ms"))
    else:   # no preview went out: what was measured, in turn from the app's start
        bar("10", "yappy", app_start, a.get("decode_ms"))
        bar("11", "yappy", total(app_start, a.get("decode_ms") or 0), a.get("crop_detect_ms"))
    bar("13a", "network-approx", app_preview, _diff(front_preview, app_preview))
    bar("13b", "network-approx", front_preview, _preview_to_browser(p) if p.timeline else None)
    return sorted(bars, key=lambda b: b["start"])


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
    origins_by_day: dict = field(default_factory=dict)          # for the page's period (P11)
    front_failures_by_day: dict = field(default_factory=dict)


def _stat(values: list[int]) -> dict | None:
    """The count, the median and the 90th, 95th and 99th percentiles (nearest rank). The page's stat() is the same."""
    if not values:
        return None
    ordered = sorted(values)
    def rank(q: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]
    return {"n": len(ordered), "median": round(statistics.median(ordered)), "p90": rank(0.9), "p95": rank(0.95),
            "p99": rank(0.99)}


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
    by_id = {a["id"]: a for a in access}
    for rid, s in requests.items():
        if rid in probe_ids or (rid in earlier and rid not in counted):
            if rid in probe_ids:
                report.probes += 1
            continue
        front_ms = round(by_id[rid]["seconds"] * 1000) if rid in by_id else None
        report.photos.append(Photo(rid, _day(s["ts"]), s, counted.get(rid), app.get(rid, {}), nginx_ms.get(rid),
                                   front_ms))
    # And every photo whose timeline counted but which the front's nginx answered itself, so the check service never
    # logged it: busy at the gate, offline with the check service stopped, over the body cap. The access line stands in.
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

    # The same two, by day, for the page's period (P11, D25).
    origins_by_day: dict[str, Counter] = defaultdict(Counter)
    for a in access:
        if a["path"].split("?")[0] == "/analyze" and a["origin"] not in ("-", own) and not a["probe"]:
            origins_by_day[a["day"]][clean(a["origin"], 80)] += 1
    report.origins_by_day = {day: dict(c.most_common()) for day, c in sorted(origins_by_day.items())}
    front_by_day: dict[str, Counter] = defaultdict(Counter)
    for s in service:
        if s.get("kind") == "request" and s.get("endpoint") == "/analyze" and s.get("outcome") != "relayed" \
                and not s.get("probe"):
            front_by_day[_day(s["ts"])][s.get("outcome", "?")] += 1
    report.front_failures_by_day = {day: dict(c) for day, c in sorted(front_by_day.items())}
    return report


# ------------------------------------------------------------------------------------------------- output

def photo_view(p: Photo) -> dict:
    """One photo for the page: its fields as text and numbers, its steps in order."""
    t, s = p.timeline or {}, p.service
    return {"request_id": p.request_id, "day": p.day, "ts": clean(s.get("ts", ""), 40), "control": p.control,
            "example": t.get("example"), "outcome": p.outcome, "rerun_of": t.get("rerun_of"),
            "filename": clean(s.get("filename", ""), 200), "format": clean(s.get("format") or "", 20),
            "ext": clean(s.get("ext") or "", 16), "bytes": s.get("bytes"), "canonical": s.get("canonical"),
            "upload_event_ms": _last_attempt(t).get("upload_ms"),
            "status": s.get("status"), "front_outcome": clean(s.get("outcome", ""), 40),
            "timeline": p.timeline is not None, "attempts": len(t.get("attempts", [])) or None, "verdict": clean(p.app.get("verdict") or "", 40),
            "steps": [{"step": key, "label": label, "ms": v if isinstance(v := measure(p), int) else None}
                      for key, label, measure in STEPS],
            "waterfall": waterfall(p)}


def as_json(report: Report) -> dict:
    """Everything the page shows, each count with its day, so the page can recount for any period and filter (D25)."""
    return {"days": report.days, "steps": report.steps, "failures": report.failures, "origins": report.origins,
            "origins_by_day": report.origins_by_day, "front_failures_by_day": report.front_failures_by_day,
            "flagged": {k: v if isinstance(v, int) else v for k, v in report.flagged.items()},
            "dropped": report.dropped, "probes": report.probes,
            "columns": ["all", *CONTROLS, *CANONICAL],
            "size_buckets": [[None if top == float("inf") else top * 2**20, label] for top, label in SIZE_BUCKETS],
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root { --page: #8a5a36; --front: #2f6f8f; --yappy: #6b8f2f; --ink: #2a2522; --muted: #7a716a; --line: #e6e0d8;
         --bg: #f7f4f0; --card: #fff; }
 * { box-sizing: border-box; }
 body { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; color: var(--ink);
        background: var(--bg); font-variant-numeric: tabular-nums; }
 main { max-width: 1280px; margin: 0 auto; padding: 0 24px 48px; }
 header.top { display: flex; align-items: baseline; gap: 16px; padding: 20px 0 8px; }
 header.top h1 { font-size: 22px; margin: 0; }
 #reload { margin-left: auto; align-self: center; } #reload:disabled { opacity: .6; cursor: wait; }
 .muted { color: var(--muted); }
 #filters { position: sticky; top: 0; z-index: 2; background: var(--bg); padding: 10px 0; border-bottom: 1px solid var(--line); }
 .bar-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }
 .bar-row + .bar-row { margin-top: 8px; }
 .field { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); }
 select, input[type=date], button { font: inherit; font-size: 13px; color: var(--ink); background: #fff;
   border: 1px solid #d6cec4; border-radius: 6px; padding: 4px 8px; }
 button { cursor: pointer; } button:hover { background: #f1ebe4; }
 button.chip { border-radius: 999px; padding: 3px 10px; }
 .toggle { display: inline-flex; } .toggle button { border-radius: 0; margin-left: -1px; }
 .toggle button:first-child { border-radius: 6px 0 0 6px; } .toggle button:last-child { border-radius: 0 6px 6px 0; }
 .toggle button.on { background: var(--ink); color: #fff; border-color: var(--ink); }
 #count { margin-left: auto; font-weight: 600; }
 .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }
 .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
 .kpi .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
 .kpi .value { font-size: 28px; font-weight: 650; margin: 2px 0; }
 .kpi .sub { font-size: 13px; color: var(--muted); }
 section.card { margin: 16px 0; }
 section h2 { font-size: 16px; margin: 0 0 10px; display: flex; align-items: center; gap: 12px; }
 section h2 .field { font-weight: normal; }
 table { border-collapse: collapse; width: 100%; }
 th { font-size: 12px; font-weight: 600; color: var(--muted); text-align: right; padding: 6px 8px;
      border-bottom: 1px solid var(--line); white-space: nowrap; }
 td { padding: 5px 8px; text-align: right; border-bottom: 1px solid #f1ece6; white-space: nowrap; }
 th.l, td.l { text-align: left; }
 td.wide { width: 40%; }
 tr.group td { text-align: left; font-weight: 600; color: var(--muted); font-size: 12px; text-transform: uppercase;
               letter-spacing: .04em; padding-top: 14px; border-bottom: 1px solid var(--line); }
 tr.step { cursor: pointer; } tr.step:hover { background: #faf6f1; }
 .key { display: inline-block; min-width: 2.6em; font-size: 12px; color: var(--muted); }
 .dist { position: relative; height: 12px; border-radius: 3px;
         background: repeating-linear-gradient(to right, #e7e1da 0 1px, transparent 1px 25%), #f7f4f0; }
 .scale { position: relative; height: 14px; font-weight: normal; }
 .scale span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
 .scale span:first-child { transform: none; } .scale span:last-child { transform: translateX(-100%); }
 .dist .p90, .dist .med { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }
 .dist .p90 { opacity: .3; }
 .lane-page { background: var(--page); } .lane-front { background: var(--front); } .lane-yappy { background: var(--yappy); }
 .lane-network { background: #9a948e; }
 .lane-network-approx { background: repeating-linear-gradient(45deg, #9a948e 0 3px, #d9d4ce 3px 6px); }
 .swatch { display: inline-block; width: 12px; height: 10px; border-radius: 2px; margin: 0 4px 0 12px; vertical-align: -1px; }
 .chips span { display: inline-block; background: #f1ebe4; border-radius: 999px; padding: 1px 8px; margin: 1px 2px;
               font-size: 12px; }
 .badge { display: inline-block; border-radius: 999px; padding: 1px 8px; font-size: 12px; font-weight: 600; }
 .badge.ok { background: #e3efd9; color: #3d6420; } .badge.bad { background: #f6dfd9; color: #8a2f1d; }
 .badge.none { background: #eee; color: #666; }
 tr.photo { cursor: pointer; } tr.photo:hover { background: #faf6f1; } tr.photo.open { background: #f1e7db; }
 tr.photo td.file { max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
 tr.detail > td { background: #fbf9f6; padding: 12px 16px 18px; white-space: normal; text-align: left; }
 .facts { display: flex; flex-wrap: wrap; gap: 6px 22px; margin-bottom: 10px; }
 .facts b { font-weight: 600; }
 .wf { display: grid; grid-template-columns: minmax(220px, 300px) 1fr 70px; align-items: center; gap: 0 10px; }
 .wf .lbl { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
 .wf .val { text-align: right; font-size: 13px; }
 .wf .track { position: relative; height: 18px;
              background: repeating-linear-gradient(to right, #e7e1da 0 1px, transparent 1px 25%); }
 .wf .seg { position: absolute; top: 4px; height: 10px; min-width: 2px; border-radius: 2px; }
 .wf .axis { position: relative; height: 16px; font-size: 11px; color: var(--muted); }
 .wf .axis span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
 .legend { white-space: nowrap; }
 .wf .axis span:first-child { transform: none; } .wf .axis span:last-child { transform: translateX(-100%); }
 .three { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
 .three h3 { font-size: 13px; margin: 0 0 6px; color: var(--muted); font-weight: 600; }
</style></head>
<body><main>
<header class="top"><h1>Coffee CV timelines</h1><span class="muted" id="read"></span>
  <button id="reload" title="Read both hosts' logs again over SSH">Reload logs</button>
  <span class="muted" id="reload-note"></span></header>
<div id="filters"></div>
<div id="out"><p class="muted">Loading...</p></div>
</main>
<script>
const out = document.getElementById('out'), bar = document.getElementById('filters');
function el(tag, text, attrs) {
  const e = document.createElement(tag);
  if (text !== undefined && text !== null) e.textContent = String(text);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
}
const put = (parent, ...kids) => { parent.append(...kids); return parent; };
const add = (counts, key, n = 1) => { counts[key] = (counts[key] || 0) + n; return counts; };
const byCount = o => Object.entries(o || {}).sort((a, b) => b[1] - a[1]);
const sumOf = o => Object.values(o || {}).reduce((a, b) => a + b, 0);
function chips(o) {
  const d = el('div', undefined, {class: 'chips'});
  for (const [k, n] of byCount(o)) d.append(el('span', `${k} ${n}`));
  if (!d.childNodes.length) d.append(el('span', 'none'));
  return d;
}
// Python's round(), half to even, so the page's medians and 90th percentiles are summary's (timelines._stat).
function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  return d < 0.5 ? f : d > 0.5 ? f + 1 : (f % 2 === 0 ? f : f + 1);
}
function stat(values) {
  if (!values.length) return null;
  const o = [...values].sort((a, b) => a - b), n = o.length;
  const median = n % 2 ? o[(n - 1) / 2] : (o[n / 2 - 1] + o[n / 2]) / 2;
  const rank = q => o[Math.min(n - 1, Math.max(0, pyRound(q * n) - 1))];
  return {n, median: pyRound(median), p90: rank(0.9), p95: rank(0.95), p99: rank(0.99)};
}
const cell = s => s ? `${s.median}/${s.p90} (${s.n})` : '-';
const fmt = v => v === null || v === undefined ? '' : Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${v} ms`;
const formatOf = p => p.format || `(${p.ext || 'none'})`;
const ms = (p, key) => { const s = p.steps.find(s => s.step === key); return s ? s.ms : null; };
const NAMES = {choose_photo: 'Choose Photo', select_file: 'Select File', camera: 'Camera', example: 'Example',
               'no timeline': 'No timeline', canonicalized: 'Canonicalized', as_sent: 'As sent'};
const name = c => NAMES[c] || c;
const shortLabel = label => label.replace(/ [(](page|app|check service)[)]$/, '');
// Where each step runs, as in the waterfall; and the step table's groups.
const LANE = {};
// Coloured by what the time is spent on, not by who measured it: the wire is grey, solid where one side measured it,
// striped where it is one host's time less another's (~).
for (const k of ['1', '3', '6', '14', '16', '16s', '17s', '17']) LANE[k] = 'page';
for (const k of ['7', '8', '16f', '17f']) LANE[k] = 'front';
for (const k of ['10', '11', '12', '15', '17n', '17a']) LANE[k] = 'yappy';
for (const k of ['2', '4', '5', '9']) LANE[k] = 'network';
for (const k of ['9y', '13a', '13b']) LANE[k] = 'network-approx';
const GROUPS = [['Totals', ['17s', '17', '17f', '17n', '17a']],
                ['On the phone, before the upload', ['1', '2', '3', '4', '5', '6']],
                ['At the front', ['7', '8', '9']],
                ['To yappy, and in the app', ['9y', '10', '11', '12', '15']],
                ['Back to the phone', ['13a', '13b', '14', '16f', '16', '16s']]];

let rep, sizeBuckets, open = null, days = [];
// Clock times in the browser's zone or in UTC; days, the period and the per-day counts stay UTC (the logs' days).
let zone = 'local';
const LOCAL = Intl.DateTimeFormat().resolvedOptions().timeZone || 'local time';
const zoneName = () => zone === 'utc' ? 'UTC' : LOCAL;
function clock(ts, withYear) {
  const d = new Date(ts);
  if (!ts || isNaN(d)) return ts || '';
  const u = zone === 'utc', two = n => String(n).padStart(2, '0');
  const [y, mo, da, h, mi, se] = u
    ? [d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDate(), d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()]
    : [d.getFullYear(), d.getMonth() + 1, d.getDate(), d.getHours(), d.getMinutes(), d.getSeconds()];
  return `${withYear ? y + '-' : ''}${two(mo)}-${two(da)} ${two(h)}:${two(mi)}:${two(se)}`;
}
const f = {};   // the filters' elements
function select(where, key, title, options) {
  const s = el('select', undefined, {id: 'f-' + key});
  for (const [value, text] of options) s.append(el('option', text, {value}));
  where.append(put(el('label', title, {class: 'field'}), s));
  s.addEventListener('change', render);
  f[key] = s;
  return s;
}
function dateInput(where, key, value) {
  const i = el('input', undefined, {type: 'date', id: 'f-' + key, value, min: days[0], max: days[days.length - 1]});
  i.addEventListener('change', render);
  where.append(i);
  f[key] = i;
}
const distinct = values => [...new Set(values)].sort();
// A time axis with round ticks: a step of 1, 2, 2.5 or 5 times a power of ten, about five of them, the ends on a tick
// (0..9 ms: 0, 2 ... 10; 0..79 s: 0, 20 ... 80). `grid` is the gridlines' background, one line per tick.
function niceAxis(lo, hi) {
  const raw = (hi - lo) / 5 || 1, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(v => v >= raw * (1 - 1e-9));
  const a = Math.floor(lo / step + 1e-9) * step, z = Math.max(a + step, Math.ceil(hi / step - 1e-9) * step);
  const ticks = [];
  for (let i = 0; a + i * step <= z + step / 1e6; i++) ticks.push(+(a + i * step).toFixed(6));
  const seconds = Math.max(Math.abs(a), Math.abs(z)) >= 1000;
  const label = v => v === 0 ? '0' : seconds ? `${+(v / 1000).toFixed(3)} s` : `${+v.toFixed(3)} ms`;
  const period = (100 / (ticks.length - 1)).toFixed(4);
  return {lo: a, hi: z, ticks, label,
          grid: `background-image: repeating-linear-gradient(to right, #e7e1da 0 1px, transparent 1px ${period}%)`};
}
function axisLabels(axis, cls) {
  const d = el('div', undefined, {class: cls});
  for (const v of axis.ticks)
    d.append(el('span', axis.label(v), {style: `left:${(100 * (v - axis.lo) / (axis.hi - axis.lo)).toFixed(2)}%`}));
  return d;
}
const legend = () => [['page', 'on the phone'], ['front', 'at the front'], ['yappy', 'at yappy'],
                      ['network', 'network, measured'], ['network-approx', 'network, between two clocks (~)']].map(([l, text]) =>
  put(el('span', undefined, {class: 'legend'}), el('span', undefined, {class: `swatch lane-${l}`}), text));
function setPeriod(first, last) { f.since.value = first; f.until.value = last; render(); }
const dayBefore = (day, n) => new Date(Date.parse(day + 'T00:00:00Z') - n * 86400000).toISOString().slice(0, 10);

// Each select's options follow the data; a choice no longer in it falls back to "any".
function setOptions(s, options) {
  const keep = s.value;
  s.textContent = '';
  for (const [value, text] of options) s.append(el('option', text, {value}));
  s.value = options.some(([v]) => v === keep) ? keep : options[0][0];
}
let lo = '', hi = '';
// The data in: the first load builds the filter bar, a reload keeps every choice, and a period that ended on the
// last day moves on with it.
function setData(data) {
  const oldHi = hi;
  rep = data;
  sizeBuckets = rep.size_buckets;
  days = distinct([...Object.keys(rep.days), ...rep.photos.map(p => p.day), ...Object.keys(rep.origins_by_day),
                   ...Object.keys(rep.front_failures_by_day),
                   ...['unverified', 'suspicious', 'unverifiable'].flatMap(k => rep.flagged[k].map(e => e.day))]);
  lo = days[0] || ''; hi = days[days.length - 1] || '';
  const read = rep.read_at ? `, read ${clock(rep.read_at, true)} ${zoneName()}` : '';
  document.getElementById('read').textContent = (days.length ? `logs from ${lo} to ${hi} (UTC)` : 'no logs yet') + read;
  if (!f.since) buildBar();
  for (const k of ['since', 'until']) { f[k].min = lo; f[k].max = hi; }
  if (!f.since.value || f.since.value < lo) f.since.value = lo;
  if (!f.until.value || f.until.value === oldHi || f.until.value > hi) f.until.value = hi;
  setOptions(f.control, [['', 'any'], ...distinct(rep.photos.map(p => p.control)).map(c => [c, name(c)])]);
  setOptions(f.example, [['', 'any'], ['-upload', 'uploads only'], ['-example', 'examples only'],
    ...distinct(rep.photos.filter(p => p.example).map(p => p.example)).map(e => [e, `example: ${e}`])]);
  setOptions(f.format, [['', 'any'], ...distinct(rep.photos.map(formatOf)).map(x => [x, x])]);
  setOptions(f.outcome, [['', 'any'], ...distinct(rep.photos.map(p => p.outcome)).map(x => [x, x])]);
  setOptions(f.sort, [['ts', 'time'], ...rep.steps.map(s => ['step:' + s.step, `${s.step} ${shortLabel(s.label)}`])]);
  render();
}
function buildBar() {
  const row1 = el('div', undefined, {class: 'bar-row'}), row2 = el('div', undefined, {class: 'bar-row'});
  bar.append(row1, row2);
  const period = el('span', 'Period (UTC days)', {class: 'field'});
  row1.append(period);
  dateInput(period, 'since', lo);
  period.append('to');
  dateInput(period, 'until', hi);
  for (const [text, first] of [['All', () => lo], ['Last 7 days', () => dayBefore(hi, 6)], ['Last day', () => hi]]) {
    const b = el('button', text, {class: 'chip'});
    b.addEventListener('click', () => setPeriod(first() < lo ? lo : first(), hi));
    row1.append(b);
  }
  const toggle = el('span', undefined, {class: 'toggle', id: 'zone'});
  for (const [v, text] of [['local', `Local (${LOCAL})`], ['utc', 'UTC']]) {
    const b = el('button', text, {'data-zone': v, class: v === zone ? 'on' : ''});
    b.addEventListener('click', () => {
      zone = v;
      for (const x of toggle.children) x.className = x.dataset.zone === zone ? 'on' : '';
      setData(rep);
    });
    toggle.append(b);
  }
  row1.append(put(el('span', 'Times', {class: 'field'}), toggle), el('span', '', {id: 'count'}));
  select(row2, 'control', 'Control', [['', 'any']]);
  select(row2, 'example', 'Photo', [['', 'any']]);
  select(row2, 'format', 'Format', [['', 'any']]);
  select(row2, 'outcome', 'Outcome', [['', 'any']]);
  select(row2, 'canonical', 'Canonical', [['', 'any'], ['canonicalized', 'canonicalized'], ['as_sent', 'as sent'],
                                          ['-none', 'not logged']]);
  const reset = el('button', 'Reset');
  reset.addEventListener('click', () => {
    for (const k of ['control', 'example', 'format', 'outcome', 'canonical', 'sort', 'order', 'compare'])
      f[k].selectedIndex = 0;
    setPeriod(lo, hi);
  });
  row2.append(reset);
  // The sort and the comparison live with their tables but are made once, here.
  f.sort = el('select', undefined, {id: 'f-sort'});
  f.order = el('select', undefined, {id: 'f-order'});
  for (const [v, t] of [['desc', 'newest / longest first'], ['asc', 'oldest / shortest first']])
    f.order.append(el('option', t, {value: v}));
  f.compare = el('select', undefined, {id: 'f-compare'});
  for (const [v, t] of [['', 'nothing'], ['control', 'control'], ['canonical', 'canonicalized or not']])
    f.compare.append(el('option', t, {value: v}));
  for (const s of [f.sort, f.order, f.compare]) s.addEventListener('change', render);
}
const load = () => fetch('/report.json', {cache: 'no-store'}).then(r => r.json()).then(setData);
// "Reload logs": the server reads both hosts again (D26), then the page takes the new data.
const reloadButton = document.getElementById('reload'), reloadNote = document.getElementById('reload-note');
reloadButton.addEventListener('click', async () => {
  reloadButton.disabled = true;
  reloadButton.textContent = 'Reading logs...';
  reloadNote.textContent = '';
  try {
    const r = await fetch('/reload', {method: 'POST', headers: {'X-Timelines-Reload': '1'}});
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || `the server answered ${r.status}`);
    const before = rep.photos.length;
    await load();
    const n = rep.photos.length - before;
    reloadNote.textContent = n > 0 ? `${n} new photo${n === 1 ? '' : 's'}` : 'no new photos';
  } catch (err) {
    reloadNote.textContent = `Could not reload: ${err.message || err}. The data shown is from the last read.`;
  } finally {
    reloadButton.disabled = false;
    reloadButton.textContent = 'Reload logs';
  }
});
load().catch(err => { out.textContent = `Could not load the report: ${err}`; });

const inPeriod = day => (!f.since.value || day >= f.since.value) && (!f.until.value || day <= f.until.value);
// A photo passes every filter; a timeline with no photo (unverifiable) passes those it has a field for.
function photoPasses(p) {
  const ex = f.example.value, can = f.canonical.value;
  return inPeriod(p.day) && (!f.control.value || p.control === f.control.value)
    && (!ex || (ex === '-upload' ? !p.example : ex === '-example' ? !!p.example : p.example === ex))
    && (!f.format.value || formatOf(p) === f.format.value) && (!f.outcome.value || p.outcome === f.outcome.value)
    && (!can || (can === '-none' ? !p.canonical : p.canonical === can));
}
function entryPasses(e) {
  const ex = f.example.value;
  return inPeriod(e.day) && !f.format.value && !f.canonical.value
    && (!f.control.value || e.control === f.control.value) && (!f.outcome.value || e.outcome === f.outcome.value)
    && (!ex || (ex === '-upload' ? !e.example : ex === '-example' ? !!e.example : e.example === ex));
}
const sizeBucket = bytes => sizeBuckets.find(([top]) => top === null || bytes < top)[1];
function section(title, ...extra) {
  const s = el('section', undefined, {class: 'card'});
  s.append(put(el('h2', title), ...extra));
  out.append(s);
  return s;
}
function kpi(label, value, sub) {
  return put(el('div', undefined, {class: 'card kpi'}), el('div', label, {class: 'label'}),
             el('div', value, {class: 'value'}), el('div', sub, {class: 'sub'}));
}

function render() {
  const photos = rep.photos.filter(photoPasses);
  document.getElementById('count').textContent = `${photos.length} of ${rep.photos.length} photos`;
  out.textContent = '';

  // Counts not of a photo follow the period only.
  let loads = 0, crawlers = 0;
  const perDay = {};
  const day = d => perDay[d] || (perDay[d] = {loads: 0, crawlers: 0, referrers: {}, photos: 0, timelines: 0,
    control: {}, example: {}, format: {}, size: {}});
  for (const [d, v] of Object.entries(rep.days)) if (inPeriod(d)) {
    const x = day(d);
    x.loads = v.page_loads; x.crawlers = v.crawler_loads; x.referrers = v.referrers;
    loads += v.page_loads; crawlers += v.crawler_loads;
  }
  for (const p of photos) {
    const x = day(p.day);
    x.photos++; x.timelines += p.timeline ? 1 : 0;
    add(x.control, name(p.control));
    if (p.example) add(x.example, p.example);
    add(x.format, formatOf(p));
    if (typeof p.bytes === 'number') add(x.size, sizeBucket(p.bytes));
  }
  const unverifiable = rep.flagged.unverifiable.filter(entryPasses);
  const pageFail = {}, frontFail = {}, origins = {};
  for (const p of photos) if (p.timeline && p.outcome !== 'answered') add(pageFail, p.outcome);
  for (const e of unverifiable) add(pageFail, e.outcome);
  for (const [d, v] of Object.entries(rep.front_failures_by_day)) if (inPeriod(d))
    for (const [k, n] of Object.entries(v)) add(frontFail, k, n);
  for (const [d, v] of Object.entries(rep.origins_by_day)) if (inPeriod(d))
    for (const [k, n] of Object.entries(v)) add(origins, k, n);

  const total = stat(photos.map(p => ms(p, '17s')).filter(v => typeof v === 'number'));   // the headline (D27)
  const answered = photos.filter(p => p.outcome === 'answered').length;
  out.append(put(el('div', undefined, {class: 'cards'}),
    kpi('Photos', photos.length, `${answered} answered, ${photos.filter(p => p.timeline).length} with a timeline`),
    kpi('Page loads', loads, `by people; ${crawlers} by crawlers`),
    kpi('Selected to answer, median', total ? fmt(total.median) : '-', total ? `p90 ${fmt(total.p90)}, p95 ${fmt(total.p95)}, p99 ${fmt(total.p99)}, ` +
        `${total.n} photos` : 'no timelines'),
    kpi('Failures', sumOf(pageFail) + sumOf(frontFail), `${sumOf(pageFail)} on the page, ${sumOf(frontFail)} at the front`)));

  // Where the time goes: each step's median and percentiles, grouped by where it runs.
  const compare = f.compare.value;
  const cmpValues = compare ? distinct(photos.map(p => compare === 'control' ? p.control : p.canonical || '-')) : [];
  const steps = section('Where the time goes', put(el('label', 'compare by', {class: 'field'}), f.compare));
  steps.append(put(el('p', 'Median (bar) and p90 (pale) of each step over the photos shown, all on one ' +
    "scale; click a step to sort the photos by it. ~ marks a network leg, one host's time less another's.",
    {class: 'muted'}), ...legend()));
  // One scale for every row, the longest 90th percentile, so any two bars compare.
  const stepStat = {};
  for (const s of rep.steps) stepStat[s.step] = stat(photos.map(p => ms(p, s.step)).filter(v => typeof v === 'number'));
  const axis = niceAxis(0, Math.max(1, ...Object.values(stepStat).map(s => s ? Math.max(s.p90, s.median) : 0)));
  const top = axis.hi, scale = axisLabels(axis, 'scale');
  const t = el('table');
  t.append(put(el('tr'), el('th', 'step', {class: 'l'}), el('th', 'median'), el('th', 'p90'), el('th', 'p95'), el('th', 'p99'),
    el('th', 'photos'),
    put(el('th', undefined, {class: 'l'}), scale), ...cmpValues.map(v => el('th', `${name(v)} median`))));
  const groups = {all: () => true};
  for (const c of rep.columns.slice(1)) groups[c] = p => p.control === c || p.canonical === c;
  const known = new Set(GROUPS.flatMap(g => g[1]));
  const allGroups = [...GROUPS, ['Other', rep.steps.map(s => s.step).filter(k => !known.has(k))]];
  for (const [title, keys] of allGroups) {
    const rows = rep.steps.filter(s => keys.includes(s.step)).sort((a, b) => keys.indexOf(a.step) - keys.indexOf(b.step));
    if (!rows.length) continue;
    const stats = rows.map(s => stepStat[s.step]);
    t.append(put(el('tr', undefined, {class: 'group'}), el('td', title, {colspan: 7 + cmpValues.length})));
    rows.forEach((s, i) => {
      const st = stats[i], tr = el('tr', undefined, {class: 'step', 'data-step': s.step,
        'data-q': st ? `${st.median}/${st.p90}/${st.p95}/${st.p99}` : '-'});
      for (const c of rep.columns)
        tr.setAttribute('data-' + c, cell(stat(photos.filter(groups[c]).map(p => ms(p, s.step)).filter(v => typeof v === 'number'))));
      const dist = el('div', undefined, {class: 'dist', style: axis.grid});
      if (st) {
        const w = v => `${Math.max(0, Math.min(100, 100 * v / top)).toFixed(1)}%`;
        dist.append(el('span', undefined, {class: `p90 lane-${LANE[s.step] || 'page'}`, style: `width:${w(st.p90)}`}),
                    el('span', undefined, {class: `med lane-${LANE[s.step] || 'page'}`, style: `width:${w(st.median)}`}));
      }
      tr.append(put(el('td', undefined, {class: 'l'}), el('span', s.step, {class: 'key'}), shortLabel(s.label)),
        ...['median', 'p90', 'p95', 'p99'].map(q => el('td', st ? fmt(st[q]) : '-')), el('td', st ? st.n : 0),
        put(el('td', undefined, {class: 'wide'}), dist),
        ...cmpValues.map(v => {
          const c = stat(photos.filter(p => (compare === 'control' ? p.control : p.canonical || '-') === v)
            .map(p => ms(p, s.step)).filter(x => typeof x === 'number'));
          return el('td', c ? `${fmt(c.median)} (${c.n})` : '-');
        }));
      tr.addEventListener('click', () => {
        f.sort.value = 'step:' + s.step; f.order.value = 'desc'; render();
        document.getElementById('photos').scrollIntoView();
      });
      t.append(tr);
    });
  }
  steps.append(t);

  // The photos, sorted; a row opens its waterfall beneath it.
  const sort = f.sort.value, sign = f.order.value === 'asc' ? 1 : -1;
  const stepKey = sort === 'ts' ? null : sort.slice(5);
  const key = stepKey ? (p => ms(p, stepKey)) : (p => p.ts);
  const sorted = photos.map((p, i) => [p, key(p), i]).sort((a, b) => {
    const [x, y] = [a[1], b[1]];
    if (x === null || x === '') return y === null || y === '' ? a[2] - b[2] : 1;   // no value: last
    if (y === null || y === '') return -1;
    return x < y ? -sign : x > y ? sign : a[2] - b[2];
  }).map(r => r[0]);
  const ps = section('Photos', put(el('label', 'sort by', {class: 'field'}), f.sort, f.order));
  ps.id = 'photos';
  ps.append(el('p', 'Click a photo for its waterfall.', {class: 'muted'}));
  const list = el('table');
  const stepLabel = stepKey && rep.steps.find(s => s.step === stepKey);
  list.append(put(el('tr'), ...[`time (${zoneName()})`, 'control', 'format', 'size', 'outcome', 'selected to answer',
    ...(stepKey ? [`${stepKey} ${shortLabel(stepLabel.label)}`] : []), 'file', 'ID']
    .map((h, i) => el('th', h, {class: [0, 1, 4, 7 + (stepKey ? 1 : 0)].includes(i) ? 'l' : ''}))));
  for (const p of sorted) {
    const tr = el('tr', undefined, {class: 'photo' + (p.request_id === open ? ' open' : ''), 'data-id': p.request_id,
      'data-total': ms(p, '17s') ?? '', 'data-value': stepKey ? (ms(p, stepKey) ?? '') : ''});
    const when = p.ts ? clock(p.ts) : p.day;
    const badge = el('span', p.outcome, {class: 'badge ' + (p.outcome === 'answered' ? 'ok' : p.timeline ? 'bad' : 'none')});
    tr.append(el('td', when, {class: 'l'}),
      el('td', p.example ? `Example · ${p.example}` : name(p.control), {class: 'l'}), el('td', formatOf(p)),
      el('td', p.bytes == null ? '' : `${(p.bytes / 1048576).toFixed(1)} MB`), put(el('td', undefined, {class: 'l'}), badge),
      el('td', fmt(ms(p, '17s') ?? ms(p, '17f'))), ...(stepKey ? [el('td', fmt(ms(p, stepKey)))] : []),
      el('td', p.filename, {class: 'l file', title: p.filename}),
      el('td', p.request_id.slice(0, 8), {title: p.request_id, class: 'muted'}));
    tr.addEventListener('click', () => { open = open === p.request_id ? null : p.request_id; render();
      const o = document.querySelector('tr.photo.open'); if (o) o.scrollIntoView({block: 'nearest'}); });
    list.append(tr);
    if (p.request_id === open) {
      const td = el('td', undefined, {colspan: 9});
      showPhoto(p, td);
      list.append(put(el('tr', undefined, {class: 'detail'}), td));
    }
  }
  ps.append(list);

  // What went wrong, and what was not counted.
  const g = rep.flagged, n = k => g[k].filter(e => inPeriod(e.day)).length;
  const health = section('Failures and the uncounted');
  const box = (title, ...kids) => put(el('div'), el('h3', title), ...kids);
  health.append(put(el('div', undefined, {class: 'three'}),
    box('Failed on the page', chips(pageFail)), box("Failed at the front (period only)", chips(frontFail)),
    box("Other sites' calls to /analyze (period only)", chips(origins)),
    box('Timelines not counted', el('div', `${n('unverified')} unverified, ${n('suspicious')} suspicious, ` +
      `${unverifiable.length} unverifiable in the period`),
      el('div', `over everything read: ${g.malformed} malformed, ${rep.dropped} dropped past the daily cap, ` +
        `${rep.probes} of the deploy's own left out`, {class: 'muted'}))));

  // Per day.
  const pd = section('Per day (UTC)');
  const dt = el('table');
  dt.append(put(el('tr'), ...['day', 'page loads', 'crawlers', 'photos', 'timelines', 'by control', 'by format',
    'by size', 'by example', 'referrers'].map((h, i) => el('th', h, {class: i === 0 || i > 4 ? 'l' : ''}))));
  for (const d of Object.keys(perDay).sort().reverse()) {
    const x = perDay[d];
    dt.append(put(el('tr'), el('td', d, {class: 'l'}), el('td', x.loads), el('td', x.crawlers), el('td', x.photos),
      el('td', x.timelines), ...[x.control, x.format, x.size, x.example, x.referrers].map(o =>
        put(el('td', undefined, {class: 'l'}), chips(o)))));
  }
  pd.append(dt);
}

function showPhoto(p, where) {
  const facts = el('div', undefined, {class: 'facts'});
  const fact = (k, v) => v !== null && v !== undefined && v !== '' && facts.append(put(el('span'), el('b', k + ' '), String(v)));
  fact('time', p.ts ? `${clock(p.ts, true)} ${zoneName()}` : p.day); fact('ID', p.request_id); fact('file', p.filename); fact('verdict', p.verdict);
  fact('canonical', p.canonical); fact("browser's upload event", p.upload_event_ms == null ? null : fmt(p.upload_event_ms)); fact('attempts', p.attempts); fact('re-run of', p.rerun_of);
  fact('front', p.front_outcome);
  where.append(facts);
  const bars = p.waterfall;
  if (!bars.length) { where.append(el('p', 'Nothing measured.', {class: 'muted'})); return; }
  const span = niceAxis(Math.min(0, ...bars.map(b => Math.min(b.start, b.start + b.ms))),
                        Math.max(1, ...bars.map(b => Math.max(b.start, b.start + b.ms))));
  const lo = span.lo, hi = span.hi;
  const x = v => `${(100 * (v - lo) / (hi - lo)).toFixed(2)}%`;
  where.append(put(el('p', "From the tap, on the phone's clock. The hosts' clocks are tied to it where the upload's first byte was " +
    'about to go out and where the photo left for yappy, so the striped network legs are approximate.', {class: 'muted'}),
    ...legend()));
  const wf = el('div', undefined, {class: 'wf'});
  const axis = axisLabels(span, 'axis');
  wf.append(el('div'), axis, el('div'));
  for (const b of bars) {
    const a = Math.min(b.start, b.start + b.ms), z = Math.max(b.start, b.start + b.ms);
    const track = el('div', undefined, {class: 'track', style: span.grid});
    track.append(el('span', undefined, {class: `seg lane-${b.lane}`,
      style: `left:${x(a)};width:${(100 * (z - a) / (hi - lo)).toFixed(2)}%`,
      title: `${b.label}: ${b.ms} ms, from ${b.start} ms`}));
    wf.append(put(el('div', undefined, {class: 'lbl', title: b.label}), el('span', b.step, {class: 'key'}),
                  shortLabel(b.label)), track, el('div', fmt(b.ms), {class: 'val'}));
  }
  where.append(wf);
}
</script>
</body></html>
"""


LOOPBACK_NAMES = {"localhost", "127.0.0.1", "[::1]"}
RELOAD_HEADER = "X-Timelines-Reload"   # only the page sends it: another site's page cannot without a CORS preflight


def make_server(report: Report, host: str, port: int, reload=None) -> ThreadingHTTPServer:
    """The page and its data, on `host`:`port` (127.0.0.1 by default: the data is the visitors'). `reload`, if given,
    makes a new Report from the hosts' logs: POST /reload calls it and swaps the data in (P11, D26). Every request must
    name a loopback host (or `host` itself), so a site that points its own name at this machine reads nothing; and
    /reload must carry RELOAD_HEADER, so another site's page in the same browser cannot make it read the hosts."""
    page = PAGE.encode()
    state = {"data": b""}
    lock = threading.Lock()

    def publish(r: Report) -> None:
        read_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state["data"] = json.dumps({**as_json(r), "read_at": read_at}).encode()

    publish(report)
    allowed = LOOPBACK_NAMES | ({host} if host not in ("", "0.0.0.0", "::") else set())

    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; "
                                                        "style-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self) -> bool:
            name = (self.headers.get("Host") or "").lower()
            name = name.rsplit(":", 1)[0] if not name.endswith("]") else name
            if name in allowed:
                return True
            self.send_error(421, "this page answers only on a loopback name")
            return False

        def do_GET(self):  # noqa: N802
            if not self.host_ok():
                return
            if self.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            elif self.path == "/report.json":
                self.send(200, state["data"], "application/json")
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            if not self.host_ok():
                return
            if self.path != "/reload" or reload is None:
                self.send_error(404)
                return
            if self.headers.get(RELOAD_HEADER) != "1":
                self.send_error(403, "not from the page")
                return
            if not lock.acquire(blocking=False):
                self.send(409, json.dumps({"error": "a reload is already running"}).encode(), "application/json")
                return
            try:
                fresh = reload()
                publish(fresh)
            except Exception as exc:   # the old data stays; the page says what went wrong
                self.send(502, json.dumps({"error": clean(f"{type(exc).__name__}: {exc}", 300)}).encode(),
                          "application/json")
                return
            finally:
                lock.release()
            self.send(200, json.dumps({"photos": len(fresh.photos)}).encode(), "application/json")

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(report: Report, host: str, port: int, reload=None) -> None:
    server = make_server(report, host, port, reload)
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
        if name == "summary":   # serve reads everything; its page picks the period (D25)
            p.add_argument("--since", help="the first day counted, YYYY-MM-DD (default: everything since the release)")
            p.add_argument("--until", help="the last day counted, YYYY-MM-DD")
        else:
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8766)
    args = ap.parse_args(argv)
    if not args.domain:
        ap.error("--domain or DOMAIN is required: it tells the site's own page loads and Origin from others'")
    def read() -> dict[str, list[str]]:
        return read_dir(args.from_dir) if args.from_dir else read_ssh({"front": args.front, "inference": args.inference})

    if args.command == "summary":
        print(summary(build(read(), args.domain, args.since, args.until)))
    else:   # the page's "Reload logs" reads them all again (D26)
        serve(build(read(), args.domain), args.host, args.port, reload=lambda: build(read(), args.domain))


if __name__ == "__main__":
    main()
