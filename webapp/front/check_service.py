"""The front's check service (ADR 0031, ticket OPS-15 P1; ADR 0033, ticket ML-19 P9): POST /analyze from the front's
nginx, checked, canonicalized, then forwarded to the inference host, whose answer streams back as it arrives.

    python webapp/front/check_service.py --listen 127.0.0.1:8100 --inference http://<tailnet address>:12346

It refuses an upload over 40 MB (ML-19 D23; OPS-15 D15 and OPS-16 D8 before) and a file in no accepted format
(upload_check; ADR 0034, OPS-18). Then the photo becomes its
canonical photo (coffeecv.canonical, ML-19 D12, D35), made with the front's own settings (canonical.FRONT_SETTINGS,
OPS-22 D1, ADR 0036: kept up to 18 MP, over that area-averaged to at most 15 MP; the integer colour path; quality 90,
effort 3), not the store's. Every upload is canonicalized here, an already canonical one (each of the page's examples)
included: ML-19 D17's pass-through is gone (OPS-22 D3). A photo over the pixel ceiling (D38; a RAW one over its own,
D24) is refused with a 413 from its header before any pixel is decoded, and one that cannot be read is the 400 the
inference host gives. The form goes on
with the canonical photo in place of the visitor's (named <stem>.jxl), every other field and header as sent; the
original is never stored or forwarded (D21). An inference host that cannot be reached, or whose app is down behind its
nginx, is a 503 "the classifier is offline" (D12), apart from its busy 503, which is relayed as it is. Every request
keeps the ID the front's nginx made for it (OPS-3 D8): forwarded to the inference host, and on every answer.

On http.server (D19): the front's nginx gates one request at a time (D11), so a thread per request never means more
than one at work, and canonicalizing (with the store's settings, up to about 21 s and 460 MB at 60 MP,
analysis/ml19/p8/README.md; with the front's, about 3 s for a 50 MP photo, OPS-22 §2) happens inside the same
one-at-a-time gate as the crossing. Since ML-19 the service is no longer standard-library only: it
imports the release's coffeecv.canonical, with training's decoders, colour management and encoder at training's
versions (coffeecv.lock_check).

It logs one JSON line per request to a file of its own (OPS-3 D9, requests.jsonl in systemd's LogsDirectory), with
no client IP: it never sees one; the proxy's stream block logs the visitor's IP per session (D20). The line is what
this service alone knows of the photo's timeline (OPS-3 D6): the visitor's original, its name, extension, detected
format and size (D10); the settings it was canonicalized with (OPS-22: "as_sent" in a line from before that, for a
photo sent canonical), its size, decoded depth, time and the process's peak and swapped memory while it was at work (ML-19 D38: the ceiling and MemoryMax are reviewed
from these lines); the crossing to the inference host, and the answer's headers, preview and end as they passed.
Only its start-up line goes to stderr, for the journal.
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import WatchedFileHandler
from urllib.parse import unquote, urlsplit

from pathlib import Path, PurePath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the release's root: coffeecv/
from upload_check import HEAD_BYTES, refusal  # noqa: E402

from coffeecv import canonical, timeline_schema  # noqa: E402

# ML-19 D23: the photo's own size, as the page measures it. Since the front canonicalizes, the photo that crosses the
# tunnel is the canonical one (at most about 14 MB at 60 MP, analysis/ml19/p4_sizes.md) and memory is held by the
# pixel ceilings below, so this cap only bounds an upload: room for every photo the ceilings take (a 12 MP DNG is
# 23 MB) and for a 200 MP JPEG (26-33 MB), refused by its pixels rather than its bytes. 10 MB before (OPS-16 D8).
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
# The form around the photo (its boundaries, the part headers, skip_crop) is a few hundred bytes; the body may be this
# much over the photo's cap, and the front's nginx allows the same (client_max_body_size 41024k).
FORM_ALLOWANCE = 64 * 1024
# D16, OPS-16 D8: 10 MB crosses the tunnel in about 40 s (2.1 Mbit/s measured into yappy, 2026-11-02) and
# /analyze takes about 9-12 s there. The connect and the whole upload (sendall) get UPLOAD_TIMEOUT_S, then each read
# of the answer ANSWER_TIMEOUT_S: at most about 180 s before an answer starts, which the front's nginx outwaits
# (proxy_read_timeout 190s).
UPLOAD_TIMEOUT_S = 120
# ML-19 D38, decided by the owner 2026-11-04: the largest photo canonicalized here, in pixels, told from its header,
# 60 MP as canonical.KEEP_MAX_PIXELS (ticket ML-19 P8: 50 MP took 400 MB and 17 s on the front, an 8-bit 60 MP 461 MB
# and 21 s, under a 500M scope without swap; a 16-bit one is not measured). The unit's MemoryMax is 600M, swap allowed;
# the log lines are the review. D24: a RAW photo is developed at 16 bits through several full frames, so its own
# ceiling (a 12 MP DNG took 373 MB; a 50 MP one was killed at 500M): 16 MP, decided by the owner 2026-11-04.
MAX_PIXELS = 60_000_000
RAW_MAX_PIXELS = 16_000_000
ANSWER_TIMEOUT_S = 60
# OPS-22 D1: the front's own canonical photo, cheaper than the store's; the inference host reads both (ADR 0036).
SETTINGS = canonical.FRONT_SETTINGS
OFFLINE = {"error": "the classifier is offline"}
BUSY = {"error": "the classifier is busy with another photo"}   # the front's nginx gate says the same (D11)
TOO_LARGE = {"error": f"this photo is over the {MAX_UPLOAD_BYTES // 2**20} MB limit"}   # and so does its 413
# Statuses from the inference host's nginx that mean its app is not answering (stopped for a deploy, ADR 0026, or
# down): the classifier is offline, as when the host cannot be reached.
APP_DOWN = {502, 504}
# Not forwarded to the inference host: the hop-by-hop headers (RFC 9110 §7.6.1) and the two this request sets again
# for its own hop, Host and Content-Length. Everything else the visitor's browser sent goes on (the app logs a coarse
# User-Agent). No visitor IP is among them: the front's nginx knows none and drops a forged one (D20).
# X-Request-Id is set again too: exactly one, in nginx's form (request_id).
NOT_FORWARDED = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade",
                 "proxy-authorization", "proxy-authenticate", "host", "content-length", "x-request-id"}
# Not relayed from the inference host's answer: its hop-by-hop headers, and Server and Date, which send_response
# writes. Its Content-Length is relayed, since the body goes on as it came, and so is its X-Request-Id.
NOT_RELAYED = NOT_FORWARDED - {"host", "content-length", "x-request-id"} | {"server", "date"}
# The request ID (OPS-3 D8): nginx's $request_id, 32 lowercase hex characters, made by the front's nginx for every
# request and carried on every hop, so the front's, this service's and the inference host's lines and the page's
# timeline join on it.
REQUEST_ID = timeline_schema.REQUEST_ID
# The log (OPS-3 D9): one JSON line per request in LOG_FILE, in the directory systemd makes for the unit
# (LogsDirectory=coffee-cv-front) and names in $LOGS_DIRECTORY, rotated at 365 days
# (webapp/deploy/coffee-cv-front.logrotate.template). DEFAULT_LOG_DIR is the same directory, for a run outside systemd.
DEFAULT_LOG_DIR = "/var/log/coffee-cv-front"
LOG_FILE = "requests.jsonl"
FILENAME_CAP = 200   # the visitor's filename, kept (D10), up to this many characters
EXT_CAP = 16
ERROR_CAP = 300
# The deploy's own requests (OPS-3 D14): --verify sends this header and value, the log keeps it, the script leaves
# such requests out of the counts. A visitor who sends it hides only their own requests.
PROBE_HEADER, PROBE = "X-Coffee-CV-Probe", timeline_schema.PROBE
# POST /timeline (OPS-3 D7): the page's timeline of a photo, checked against coffeecv/timeline_schema.py, whose example
# names come from the release's list. D23: past this many bytes of /timeline's lines in a UTC day, the rest of the
# day's are dropped and counted in the next request line: a flood adds at most this much a day (about 2 MB once
# compressed), and a real day is a few hundred KB. The count starts again when the service restarts.
EXAMPLES_LIST = Path(__file__).resolve().parents[1] / "static" / "examples" / "examples.json"
TIMELINE_DAILY_BYTES = 20 * 2**20


def request_id(sent: str | None) -> str:
    """The ID the front's nginx sent, or, when there is none or it is not in nginx's form (a direct call), one made
    here in the same form."""
    return sent if sent is not None and REQUEST_ID.fullmatch(sent) else uuid.uuid4().hex


def photo_part(content_type: str, body: bytes) -> tuple[str, bytes, int] | None:
    """The `photo` form field's filename, its first HEAD_BYTES bytes and its size, or None when the body has none."""
    found = _photo(content_type, body)
    return None if found is None else (found[0], found[1][:HEAD_BYTES], len(found[1]))


def _boundary(content_type: str) -> bytes | None:
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not content_type.lower().startswith("multipart/form-data") or m is None:
        return None
    return b"--" + m[1].encode()


def _photo(content_type: str, body: bytes) -> tuple[str, bytes, int] | None:
    """The `photo` form field's filename, its bytes and the index of its part in the body split at the boundary."""
    boundary = _boundary(content_type)
    if boundary is None:
        return None
    for index, part in enumerate(body.split(boundary)[1:], 1):
        headers, sep, data = part.partition(b"\r\n\r\n")
        disposition = re.search(rb"(?im)^content-disposition:(.*)$", headers)
        if not sep or disposition is None or not re.search(rb';\s*name="photo"', disposition[1]):
            continue
        data = data.removesuffix(b"\r\n")   # the line break before the next boundary is the form's, not the photo's
        plain = re.search(rb';\s*filename="([^"]*)"', disposition[1])
        encoded = re.search(rb";\s*filename\*=[^']*'[^']*'([^;\s]+)", disposition[1])   # RFC 5987, as werkzeug reads
        filename = unquote(encoded[1].decode("ascii", "replace")) if encoded else \
            plain[1].decode("utf-8", "replace") if plain else ""
        return (filename, data, index) if filename else None
    return None


def with_photo(content_type: str, body: bytes, filename: str, data: bytes) -> bytes:
    """The form with its `photo` part replaced by `data` under `filename`, every other part as it was."""
    boundary = _boundary(content_type)
    parts = body.split(boundary)
    _, _, index = _photo(content_type, body)
    parts[index] = (f'\r\nContent-Disposition: form-data; name="photo"; filename="{filename}"\r\n'
                    "Content-Type: image/jxl\r\n\r\n").encode() + data + b"\r\n"
    return boundary.join(parts)


def reset_peak_memory() -> bool:
    """Resets the process's peak resident memory (VmHWM) to its current size (Linux 4.0+): requests are canonicalized
    one at a time, so the peak after one is its own. False where it cannot be reset: the peak is then the process's."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def process_memory_mb() -> tuple[int, int]:
    """(peak resident, swapped out) of this process, in MB, from /proc/self/status; (0, 0) where it cannot be read."""
    try:
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        return tuple(round(int(status[k].split()[0]) / 1024) for k in ("VmHWM", "VmSwap"))
    except (OSError, KeyError, ValueError):
        return 0, 0


class JsonLines:
    """The service's log (OPS-3 D9): one JSON object per line, appended to a file that logrotate moves aside and the
    next line starts again (WatchedFileHandler, as the app's request log). json.dumps escapes every control and
    non-ASCII character, so a line is one line whatever a visitor sent. Opened at once: a log directory that cannot be
    written stops the service at its start, not at its first photo."""

    def __init__(self, path: Path):
        self.handler = WatchedFileHandler(path)
        self.handler.setFormatter(logging.Formatter("%(message)s"))

    def write(self, record: dict) -> None:
        self.write_line(json.dumps(record))

    def write_line(self, line: str) -> None:
        self.handler.handle(logging.makeLogRecord({"msg": line}))


def ms_since(t: float) -> int:
    return round((time.monotonic() - t) * 1000)


def utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class DailyBudget:
    """What /timeline may write in a UTC day (D23), and how many lines it dropped past that, since last asked."""

    def __init__(self):
        self.lock = threading.Lock()
        self.day, self.written, self.dropped = "", 0, 0

    def take(self, size: int) -> bool:
        with self.lock:
            if (day := utc_day()) != self.day:
                self.day, self.written = day, 0
            if self.written + size > TIMELINE_DAILY_BYTES:
                self.dropped += 1
                return False
            self.written += size
            return True

    def take_dropped(self) -> int:
        with self.lock:
            dropped, self.dropped = self.dropped, 0
            return dropped


class CheckHandler(BaseHTTPRequestHandler):
    inference: tuple[str, int]   # set by make_server
    crossing: threading.Lock     # set by make_server: held while a photo crosses and its answer comes back
    log: JsonLines               # set by make_server
    examples: frozenset[str]     # set by make_server: the release's example names
    budget: DailyBudget          # set by make_server
    server_version = "coffee-cv-front"
    sys_version = ""

    def do_POST(self):  # noqa: N802
        """Answers, then writes the request's one line: what this service alone knows of it (OPS-3 P2), and never the
        client's address, which is the front's nginx's 127.0.0.1 anyway (D20)."""
        t0 = time.monotonic()
        self.request_id = request_id(self.headers.get("X-Request-Id"))
        path = urlsplit(self.path).path
        self.record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "kind": "request",
                       "request_id": self.request_id, "method": "POST", "endpoint": path}
        if self.headers.get(PROBE_HEADER) == PROBE:   # the deploy's own check (OPS-3 D14); any other value is not kept
            self.record["probe"] = PROBE
        if path == "/timeline":
            return self.timeline()
        outcome = "error"   # what the line says if this handler itself fails
        try:
            outcome = self.analyze() if path == "/analyze" else self.json(404, {"error": "not found"}, "not_found")
        finally:
            self.record.update(outcome=outcome, total_ms=ms_since(t0))
            if dropped := self.budget.take_dropped():
                self.record["timelines_dropped"] = dropped
            self.log.write(self.record)

    def do_GET(self):  # noqa: N802
        self.request_id = request_id(self.headers.get("X-Request-Id"))
        self.record = {}
        path = urlsplit(self.path).path
        self.json(405 if path in ("/analyze", "/timeline") else 404,
                  {"error": f"use POST {path if path == '/timeline' else '/analyze'}"}, "")

    def timeline(self) -> None:
        """POST /timeline (OPS-3 D7, D16): a photo's timeline from the page, written as a line of kind "timeline" when
        coffeecv/timeline_schema.py accepts it, else refused with a 400 and a request line saying which key, never what
        was sent (D17's layer 2). The front's nginx caps the body, the rate and the Origin first (layer 3); the daily
        cap is here (D23). It never takes the photo's gate: a timeline is a few hundred bytes, written at once."""
        status, outcome, reason = 204, "written", None
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length < 0:
                raise ValueError
        except ValueError:
            status, outcome = 411, "bad_length"
        else:
            if length > timeline_schema.MAX_BODY:
                status, outcome = 413, "too_large"   # unread: the connection is closed after the answer
            else:
                try:
                    sent = timeline_schema.parse(self.rfile.read(length), self.examples)
                except timeline_schema.Invalid as exc:
                    status, outcome, reason = 400, "refused", str(exc)
        if status == 204:
            line = json.dumps({"ts": self.record["ts"], "kind": "timeline", "request_id": self.request_id,
                               **({"probe": PROBE} if "probe" in self.record else {}), "timeline": sent})
        else:
            self.close_connection = True
            line = json.dumps({**self.record, "status": status, "outcome": outcome,
                               **({"reason": reason} if reason else {})})
        if self.budget.take(len(line) + 1):
            self.log.write_line(line)
        elif status == 204:
            status = 503
        if status == 204:
            self.send_response(204)
            self.send_header("X-Request-Id", self.request_id)
            self.end_headers()
        else:
            error = {411: "a Content-Length is required", 413: "this timeline is too large",
                     503: "no more timelines today"}.get(status, f"not a timeline: {reason}")
            self.json(status, {"error": error}, outcome)

    def analyze(self) -> str:
        if "Content-Length" not in self.headers:
            self.close_connection = True
            return self.json(411, {"error": "a Content-Length is required"}, "bad_length")
        try:
            length = int(self.headers["Content-Length"])
            if length < 0:
                raise ValueError
        except ValueError:
            self.close_connection = True
            return self.json(400, {"error": "a bad Content-Length"}, "bad_length")
        if length > MAX_UPLOAD_BYTES + FORM_ALLOWANCE:
            # Read and dropped, not left unread: a sender cut off mid-body sees a reset, not the 413. The front's nginx
            # caps the body at the same size first, so this is a backstop.
            left = length
            while left > 0 and (chunk := self.rfile.read(min(65536, left))):
                left -= len(chunk)
            return self.json(413, TOO_LARGE, "too_large")
        t = time.monotonic()
        body = self.rfile.read(length)
        self.record["read_ms"] = ms_since(t)
        t = time.monotonic()
        part = photo_part(self.headers.get("Content-Type", ""), body)
        if part is None:
            return self.json(400, {"error": "no photo uploaded"}, "no_photo")   # the app's wording (webapp/app.py)
        filename, head, size = part
        # The visitor's original, as sent (OPS-3 D10): its name, kept whole up to a cap, its format by its content.
        self.record.update(filename=filename[:FILENAME_CAP], ext=PurePath(filename).suffix.lower()[:EXT_CAP], bytes=size)
        if (fmt := canonical.recognise(head, filename)) is not None:
            self.record["format"] = fmt
        if size > MAX_UPLOAD_BYTES:
            self.record["check_ms"] = ms_since(t)
            # Rounded up, as the page rounds it: a photo just over the cap never reads as exactly the cap.
            return self.json(413, {"error": f"this photo is {math.ceil(size / 2**20 * 10) / 10:.1f} MB; the limit is "
                                            f"{MAX_UPLOAD_BYTES // 2**20} MB"}, "too_large")
        reason = refusal(filename, head)
        self.record["check_ms"] = ms_since(t)
        if reason is not None:
            return self.json(400, {"error": f"could not read this as an image: {reason}"}, "not_an_image")
        # One photo is canonicalized and crosses at a time (D11). The front's nginx gates first, but it frees its gate
        # when a visitor leaves, and the crossing that visitor started goes on here until the inference host answers.
        if not self.crossing.acquire(blocking=False):
            return self.json(503, BUSY, "busy")
        try:
            content_type = self.headers.get("Content-Type", "")
            _, data, _ = _photo(content_type, body)
            raw = PurePath(filename).suffix.lower() in canonical.RAW_EXTENSIONS
            try:
                canon = self.canonicalize(data, filename, RAW_MAX_PIXELS if raw else MAX_PIXELS)
            except canonical.TooLarge:
                limit = (RAW_MAX_PIXELS if raw else MAX_PIXELS) / 1e6
                return self.json(413, {"error": f"this photo has too many pixels; the limit is {limit:.0f} MP"
                                                + (" for a camera RAW file" if raw else "")}, "too_many_pixels")
            except canonical.Unreadable as exc:
                return self.json(400, {"error": f"could not read this as an image: {exc}"}, "unreadable")
            except Exception as exc:   # noqa: BLE001 -- any other failure is an answer, not a dropped connection
                self.record["error"] = f"{type(exc).__name__}: {exc}"[:ERROR_CAP]
                return self.json(500, {"error": "could not process this photo"}, "failed")
            del data
            self.record["canonical_bytes"] = len(canon)
            # The visitor's name, kept to a safe alphabet: it goes into the forwarded part's header.
            name = f"{re.sub(r'[^A-Za-z0-9._-]', '_', PurePath(filename).stem)[:100] or 'photo'}.jxl"
            return self.forward(with_photo(content_type, body, name, canon))
        finally:
            self.crossing.release()

    def canonicalize(self, data: bytes, filename: str, max_pixels: int) -> bytes:
        """The photo decoded and encoded with the front's settings (OPS-22 D1), an already canonical one too (D3: it is
        re-encoded, not passed on), noting in the log line the settings, what was decoded and the memory it took
        (ML-19 D38)."""
        t0, measured = time.monotonic(), reset_peak_memory()
        source = canonical.decode(data, filename, max_pixels, settings=SETTINGS)
        (h, w), bits = source.pixels.shape[:2], source.pixels.dtype.itemsize * 8
        canon = canonical.encode(source, settings=SETTINGS)
        del source
        peak_mb, swap_mb = process_memory_mb()
        self.record.update(canonical="canonicalized", canonical_settings=SETTINGS.label,
                           canonicalize_ms=ms_since(t0), width=w, height=h, bits=bits, peak_mb=peak_mb,
                           swap_mb=swap_mb)
        if not measured:
            self.record["peak_since_start"] = True
        return canon

    def forward(self, body: bytes) -> str:
        """Sends the photo on and relays the answer. The log line's crossing_ms is the connection and the whole upload;
        its to_*_ms count from the upload's last byte (OPS-3 D6, steps 9-16)."""
        t = time.monotonic()
        conn = http.client.HTTPConnection(*self.inference, timeout=UPLOAD_TIMEOUT_S)
        try:
            conn.putrequest("POST", self.path, skip_accept_encoding=True)
            for k, v in self.headers.items():   # items(), not a dict: a header sent twice goes on twice
                if k.lower() not in NOT_FORWARDED:
                    conn.putheader(k, v)
            conn.putheader("X-Request-Id", self.request_id)
            conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body)
            sent = time.monotonic()
            self.record["crossing_ms"] = ms_since(t)
            conn.sock.settimeout(ANSWER_TIMEOUT_S)   # the photo has crossed: from here, each read of the answer
            resp = conn.getresponse()
            self.record["to_headers_ms"] = ms_since(sent)
        except (OSError, http.client.HTTPException) as exc:   # refused, unreachable, timed out, not HTTP
            conn.close()
            self.record["error"] = type(exc).__name__
            return self.json(503, OFFLINE, "offline")
        try:
            return self.relay(resp, sent)
        finally:
            conn.close()

    def relay(self, resp: http.client.HTTPResponse, sent: float) -> str:
        if resp.status in APP_DOWN:
            self.record["upstream_status"] = resp.status
            return self.json(503, OFFLINE, "offline")
        self.send_response(resp.status)
        self.record["status"] = resp.status
        for k, v in resp.getheaders():
            if k.lower() not in NOT_RELAYED:
                self.send_header(k, v)
        if resp.getheader("X-Request-Id") is None:   # an answer of the inference host's nginx, not of its app
            self.send_header("X-Request-Id", self.request_id)
        self.end_headers()
        # The answer as it arrives (ADR 0023): read1 returns what is there, so the preview goes out before the
        # classification exists. Without a Content-Length, the end of the answer is this connection's close.
        sent_bytes, outcome = 0, "relayed"
        # A streamed answer's first line is its preview (ADR 0023): when its end has gone out, step 13's front half.
        preview_pending = resp.getheader("Content-Type", "").startswith("application/x-ndjson")
        try:
            while chunk := resp.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
                sent_bytes += len(chunk)
                if preview_pending and b"\n" in chunk:
                    self.record["to_preview_ms"], preview_pending = ms_since(sent), False
        except (OSError, http.client.HTTPException) as exc:
            # Either side gone mid-answer (IncompleteRead: the inference host's): the visitor's answer ends where it
            # was cut, as when the app fails mid-answer, and the page says so.
            self.record["error"], outcome = type(exc).__name__, "cut"
        self.record.update(to_end_ms=ms_since(sent), relayed_bytes=sent_bytes)
        return outcome

    def json(self, status: int, payload: dict, outcome: str) -> str:
        """Sends a JSON answer of this service's own; returns `outcome`, the log line's word for it."""
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.record["status"] = status
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Request-Id", self.request_id)
        self.end_headers()
        self.wfile.write(data)
        return outcome

    def log_message(self, *args):
        pass   # do_POST logs its own line, without the client address http.server would print


def make_server(listen: tuple[str, int], inference_url: str, log_dir: Path) -> ThreadingHTTPServer:
    url = urlsplit(inference_url)
    if url.scheme != "http" or not url.hostname or url.path not in ("", "/"):
        raise SystemExit(f"--inference wants http://<host>:<port>, got {inference_url!r}")
    handler = type("Handler", (CheckHandler,), {"inference": (url.hostname, url.port or 80),
                                                "crossing": threading.Lock(), "log": JsonLines(log_dir / LOG_FILE),
                                                "examples": timeline_schema.example_names(EXAMPLES_LIST),
                                                "budget": DailyBudget()})
    server = ThreadingHTTPServer(listen, handler)
    server.daemon_threads = True
    return server


def options(argv: list[str], env: Mapping[str, str]) -> tuple[tuple[str, int], str, Path]:
    """(host, port) to serve on, the inference host's URL and the log's directory: the flags, else front.env's
    variables, which the release's unit loads (webapp/deploy/coffee-cv-front.service.template), and for the log
    systemd's $LOGS_DIRECTORY."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", default=env.get("COFFEE_CV_FRONT_LISTEN"),
                    help="host:port to serve on, loopback behind the front's nginx (default: $COFFEE_CV_FRONT_LISTEN)")
    ap.add_argument("--inference", default=env.get("COFFEE_CV_INFERENCE_URL"),
                    help="the inference host's site, http://<tailnet address>:<port> (default: $COFFEE_CV_INFERENCE_URL)")
    ap.add_argument("--log-dir", default=env.get("LOGS_DIRECTORY"),
                    help=f"where {LOG_FILE} is written (default: $LOGS_DIRECTORY, which systemd sets from the unit's "
                         f"LogsDirectory=, else {DEFAULT_LOG_DIR})")
    args = ap.parse_args(argv)
    if not args.listen or not args.inference:
        ap.error("--listen and --inference, or COFFEE_CV_FRONT_LISTEN and COFFEE_CV_INFERENCE_URL, are required")
    host, _, port = args.listen.rpartition(":")
    return (host.strip("[]"), int(port)), args.inference, Path(args.log_dir or DEFAULT_LOG_DIR)


def main() -> None:
    listen, inference, log_dir = options(sys.argv[1:], os.environ)
    server = make_server(listen, inference, log_dir)
    # The journal's one line: each request's goes to the log file.
    sys.stderr.write(f"check service on {listen[0]}:{listen[1]}, forwarding to {inference}, "
                     f"logging to {log_dir / LOG_FILE}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
