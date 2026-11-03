"""The front's check service (ADR 0031, ticket OPS-15 P1; ADR 0033, ticket ML-19 P9): POST /analyze from the front's
nginx, checked, canonicalized, then forwarded to the inference host, whose answer streams back as it arrives.

    python webapp/front/check_service.py --listen 127.0.0.1:8100 --inference http://<tailnet address>:12346

It refuses what must not cross the tunnel to the inference host (2.1 Mbit/s, OPS-15 §2): an upload over 10 MB (D15,
OPS-16 D8) and a file that is not one of the formats the service reads (upload_check). Then the photo becomes its
canonical photo (coffeecv.canonical, ML-19 D12, D35): one already canonical goes on as it is (D17), any other is
canonicalized here, a photo over the pixel ceiling (D38; a RAW one over its own, D24) is refused with a 413 from its
header before any pixel is decoded, and one that cannot be read is the 400 the inference host gives. The form goes on
with the canonical photo in place of the visitor's (named <stem>.jxl), every other field and header as sent; the
original is never stored or forwarded (D21). An inference host that cannot be reached, or whose app is down behind its
nginx, is a 503 "the classifier is offline" (D12), apart from its busy 503, which is relayed as it is.

On http.server (D19): the front's nginx gates one request at a time (D11), so a thread per request never means more
than one at work, and canonicalizing (up to about 21 s and 460 MB at 60 MP, analysis/ml19/p8/README.md) happens
inside the same one-at-a-time gate as the crossing. Since ML-19 the service is no longer standard-library only: it
imports the release's coffeecv.canonical, with training's decoders, colour management and encoder at training's
versions (coffeecv.lock_check). It logs one line per request to stderr, for the journal, with no
client IP: it never sees one; the proxy's stream block logs the visitor's IP per session (D20).
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import sys
import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from pathlib import Path, PurePath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the release's root: coffeecv/
from upload_check import HEAD_BYTES, RAW_EXTENSIONS, refusal  # noqa: E402

from coffeecv import canonical  # noqa: E402

# D15, until ML-15, raised to 10 MB by ticket OPS-16 D8: the photo's own size, as the page measures it.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# The form around the photo (its boundaries, the part headers, skip_crop) is a few hundred bytes; the body may be this
# much over the photo's cap, and the front's nginx allows the same (client_max_body_size 10304k).
FORM_ALLOWANCE = 64 * 1024
# D16, OPS-16 D8: a 10 MB photo crosses the tunnel in about 40 s (2.1 Mbit/s measured into yappy, 2026-11-02) and
# /analyze takes about 9-12 s there. The connect and the whole upload (sendall) get UPLOAD_TIMEOUT_S, then each read
# of the answer ANSWER_TIMEOUT_S: at most about 180 s before an answer starts, which the front's nginx outwaits
# (proxy_read_timeout 190s).
UPLOAD_TIMEOUT_S = 120
# ML-19 D38: the largest photo canonicalized here, in pixels, told from its header (ticket ML-19 P8: 50 MP took 400 MB
# and 17 s on the front, 60 MP 461 MB and 21 s, under MemoryMax 500M). 50,200,000 lets the Pixel 9 Pro's 50.14 MP
# through. D24: a RAW photo is developed at 16 bits through several full frames, so its own ceiling (a 12 MP DNG took
# 373 MB; a 50 MP one was killed at 500M). Both await the owner's decision at P8's stop: these are P8's recommendations.
MAX_PIXELS = 50_200_000
RAW_MAX_PIXELS = 16_000_000
ANSWER_TIMEOUT_S = 60
OFFLINE = {"error": "the classifier is offline"}
BUSY = {"error": "the classifier is busy with another photo"}   # the front's nginx gate says the same (D11)
TOO_LARGE = {"error": f"this photo is over the {MAX_UPLOAD_BYTES // 2**20} MB limit"}   # and so does its 413
# Statuses from the inference host's nginx that mean its app is not answering (stopped for a deploy, ADR 0026, or
# down): the classifier is offline, as when the host cannot be reached.
APP_DOWN = {502, 504}
# Not forwarded to the inference host: the hop-by-hop headers (RFC 9110 §7.6.1) and the two this request sets again
# for its own hop, Host and Content-Length. Everything else the visitor's browser sent goes on (the app logs a coarse
# User-Agent). No visitor IP is among them: the front's nginx knows none and drops a forged one (D20).
NOT_FORWARDED = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade",
                 "proxy-authorization", "proxy-authenticate", "host", "content-length"}
# Not relayed from the inference host's answer: its hop-by-hop headers, and Server and Date, which send_response
# writes. Its Content-Length is relayed, since the body goes on as it came.
NOT_RELAYED = NOT_FORWARDED - {"host", "content-length"} | {"server", "date"}


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


class CheckHandler(BaseHTTPRequestHandler):
    inference: tuple[str, int]   # set by make_server
    crossing: threading.Lock     # set by make_server: held while a photo crosses and its answer comes back
    server_version = "coffee-cv-front"
    sys_version = ""

    def do_POST(self):  # noqa: N802
        t0 = time.monotonic()
        outcome = self.analyze() if urlsplit(self.path).path == "/analyze" else self.json(404, {"error": "not found"})
        self.log_line(f"{self.command} {urlsplit(self.path).path} {outcome} {round((time.monotonic() - t0) * 1000)}ms")

    def do_GET(self):  # noqa: N802
        self.json(405 if urlsplit(self.path).path == "/analyze" else 404, {"error": "use POST /analyze"})

    def analyze(self) -> str:
        if "Content-Length" not in self.headers:
            self.close_connection = True
            return self.json(411, {"error": "a Content-Length is required"})
        try:
            length = int(self.headers["Content-Length"])
            if length < 0:
                raise ValueError
        except ValueError:
            self.close_connection = True
            return self.json(400, {"error": "a bad Content-Length"})
        if length > MAX_UPLOAD_BYTES + FORM_ALLOWANCE:
            # Read and dropped, not left unread: a sender cut off mid-body sees a reset, not the 413. The front's nginx
            # caps the body at the same size first, so this is a backstop.
            left = length
            while left > 0 and (chunk := self.rfile.read(min(65536, left))):
                left -= len(chunk)
            return self.json(413, TOO_LARGE)
        body = self.rfile.read(length)
        part = photo_part(self.headers.get("Content-Type", ""), body)
        if part is None:
            return self.json(400, {"error": "no photo uploaded"})   # the app's wording (webapp/app.py, analyze)
        filename, head, size = part
        if size > MAX_UPLOAD_BYTES:
            # Rounded up, as the page rounds it: a photo just over the cap never reads as exactly the cap.
            return self.json(413, {"error": f"this photo is {math.ceil(size / 2**20 * 10) / 10:.1f} MB; the limit is "
                                            f"{MAX_UPLOAD_BYTES // 2**20} MB"})
        reason = refusal(filename, head)
        if reason is not None:
            return self.json(400, {"error": f"could not read this as an image: {reason}"})
        # One photo is canonicalized and crosses at a time (D11). The front's nginx gates first, but it frees its gate
        # when a visitor leaves, and the crossing that visitor started goes on here until the inference host answers.
        if not self.crossing.acquire(blocking=False):
            return self.json(503, BUSY, "busy (a photo is still crossing)")
        try:
            content_type = self.headers.get("Content-Type", "")
            _, data, _ = _photo(content_type, body)
            raw = PurePath(filename).suffix.lower() in RAW_EXTENSIONS
            try:
                canon = canonical.canonicalize(data, filename, max_pixels=RAW_MAX_PIXELS if raw else MAX_PIXELS)
            except canonical.TooLarge:
                limit = (RAW_MAX_PIXELS if raw else MAX_PIXELS) / 1e6
                return self.json(413, {"error": f"this photo has too many pixels; the limit is {limit:.0f} MP"
                                                + (" for a camera RAW file" if raw else "")}, "413 (pixels)")
            except canonical.Unreadable as exc:
                return self.json(400, {"error": f"could not read this as an image: {exc}"})
            except Exception as exc:   # noqa: BLE001 -- any other failure is an answer, not a dropped connection
                return self.json(500, {"error": "could not process this photo"},
                                 f"500 canonicalization failed ({type(exc).__name__}: {exc})")
            del data
            name = f"{PurePath(filename).stem or 'photo'}.jxl"
            return self.forward(with_photo(content_type, body, name, canon))
        finally:
            self.crossing.release()

    def forward(self, body: bytes) -> str:
        conn = http.client.HTTPConnection(*self.inference, timeout=UPLOAD_TIMEOUT_S)
        try:
            conn.putrequest("POST", self.path, skip_accept_encoding=True)
            for k, v in self.headers.items():   # items(), not a dict: a header sent twice goes on twice
                if k.lower() not in NOT_FORWARDED:
                    conn.putheader(k, v)
            conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body)
            conn.sock.settimeout(ANSWER_TIMEOUT_S)   # the photo has crossed: from here, each read of the answer
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:   # refused, unreachable, timed out, not HTTP
            conn.close()
            return self.json(503, OFFLINE, f"offline ({type(exc).__name__})")
        try:
            return self.relay(resp)
        finally:
            conn.close()

    def relay(self, resp: http.client.HTTPResponse) -> str:
        if resp.status in APP_DOWN:
            return self.json(503, OFFLINE, f"offline ({resp.status} from the inference host)")
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in NOT_RELAYED:
                self.send_header(k, v)
        self.end_headers()
        # The answer as it arrives (ADR 0023): read1 returns what is there, so the preview goes out before the
        # classification exists. Without a Content-Length, the end of the answer is this connection's close.
        sent = 0
        try:
            while chunk := resp.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
                sent += len(chunk)
        except (OSError, http.client.HTTPException) as exc:
            # Either side gone mid-answer (IncompleteRead: the inference host's): the visitor's answer ends where it
            # was cut, as when the app fails mid-answer, and the page says so.
            return f"{resp.status} cut after {sent} bytes ({type(exc).__name__})"
        return f"{resp.status} relayed {sent} bytes"

    def json(self, status: int, payload: dict, outcome: str | None = None) -> str:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return outcome or str(status)

    def log_message(self, *args):
        pass   # do_POST logs its own line, without the client address http.server would print

    def log_line(self, line: str) -> None:
        sys.stderr.write(line + "\n")


def make_server(listen: tuple[str, int], inference_url: str) -> ThreadingHTTPServer:
    url = urlsplit(inference_url)
    if url.scheme != "http" or not url.hostname or url.path not in ("", "/"):
        raise SystemExit(f"--inference wants http://<host>:<port>, got {inference_url!r}")
    handler = type("Handler", (CheckHandler,), {"inference": (url.hostname, url.port or 80),
                                                "crossing": threading.Lock()})
    server = ThreadingHTTPServer(listen, handler)
    server.daemon_threads = True
    return server


def options(argv: list[str], env: Mapping[str, str]) -> tuple[tuple[str, int], str]:
    """(host, port) to serve on and the inference host's URL: the flags, else front.env's variables, which the
    release's unit loads (webapp/deploy/coffee-cv-front.service.template)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", default=env.get("COFFEE_CV_FRONT_LISTEN"),
                    help="host:port to serve on, loopback behind the front's nginx (default: $COFFEE_CV_FRONT_LISTEN)")
    ap.add_argument("--inference", default=env.get("COFFEE_CV_INFERENCE_URL"),
                    help="the inference host's site, http://<tailnet address>:<port> (default: $COFFEE_CV_INFERENCE_URL)")
    args = ap.parse_args(argv)
    if not args.listen or not args.inference:
        ap.error("--listen and --inference, or COFFEE_CV_FRONT_LISTEN and COFFEE_CV_INFERENCE_URL, are required")
    host, _, port = args.listen.rpartition(":")
    return (host.strip("[]"), int(port)), args.inference


def main() -> None:
    listen, inference = options(sys.argv[1:], os.environ)
    server = make_server(listen, inference)
    sys.stderr.write(f"check service on {listen[0]}:{listen[1]}, forwarding to {inference}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
