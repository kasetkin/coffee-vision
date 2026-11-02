"""The front's check service (ADR 0031, ticket OPS-15 P1): POST /analyze from the front's nginx, checked, then forwarded
unchanged to the inference host, whose answer streams back as it arrives.

    python webapp/front/check_service.py --listen 127.0.0.1:8100 --inference http://<tailnet address>:12346

It refuses what must not cross the tunnel to the inference host (2.1 Mbit/s, OPS-15 §2): an upload over 5 MB (D15)
and a file that is not one of the formats the service reads (upload_check). Everything else goes on as the visitor
sent it, the photo's bytes untouched (D2): decoding and every pixel the model sees stay on the inference host. An
inference host that cannot be reached, or whose app is down behind its nginx, is a 503 "the classifier is offline"
(D12), apart from its busy 503, which is relayed as it is.

Standard library only (D7), on http.server (D19): the front's nginx gates one request at a time (D11), so a thread
per request never means more than one at work. It logs one line per request to stderr, for the journal, with no
client IP: the front's nginx logs that (D18).
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from upload_check import HEAD_BYTES, refusal

MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # D15, until ML-15; the front's nginx caps the body at the same 5m
# D16: a 5 MB photo crosses the tunnel in about 20 s (2.1 Mbit/s measured into yappy, 2026-11-02) and /analyze takes
# about 9-12 s there; 60 s for each read or write on the connection to the inference host leaves room for both.
INFERENCE_TIMEOUT_S = 60
OFFLINE = {"error": "the classifier is offline"}
# Statuses from the inference host's nginx that mean its app is not answering (stopped for a deploy, ADR 0026, or
# down): the classifier is offline, as when the host cannot be reached.
APP_DOWN = {502, 504}
# Not forwarded either way: they describe one connection, not the request (RFC 9110 §7.6.1). Content-Length is set
# again for the body as sent.
HOP_BY_HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-authenticate", "host", "content-length"}
# Not relayed from the inference host's answer: its own hop-by-hop headers, and Server and Date, which send_response
# writes. Its Content-Length is: the relay sends the body as it came.
NOT_RELAYED = HOP_BY_HOP - {"content-length"} | {"server", "date"}


def photo_part(content_type: str, body: bytes) -> tuple[str, bytes] | None:
    """The `photo` form field's filename and its first HEAD_BYTES bytes, or None when the body has none."""
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not content_type.startswith("multipart/form-data") or m is None:
        return None
    for part in body.split(b"--" + m[1].encode())[1:]:
        headers, sep, data = part.partition(b"\r\n\r\n")
        disposition = re.search(rb"(?im)^content-disposition:(.*)$", headers)
        if not sep or disposition is None or not re.search(rb';\s*name="photo"', disposition[1]):
            continue
        filename = re.search(rb';\s*filename="([^"]*)"', disposition[1])
        if filename is None or not filename[1]:
            return None
        return filename[1].decode("utf-8", "replace"), data[:HEAD_BYTES]
    return None


class CheckHandler(BaseHTTPRequestHandler):
    inference: tuple[str, int]   # set by make_server
    server_version = "coffee-cv-front"
    sys_version = ""

    def do_POST(self):  # noqa: N802
        t0 = time.monotonic()
        outcome = self.analyze() if self.path == "/analyze" else self.json(404, {"error": "not found"})
        self.log_line(f"{self.command} {self.path} {outcome} {round((time.monotonic() - t0) * 1000)}ms")

    def do_GET(self):  # noqa: N802
        self.json(405 if self.path == "/analyze" else 404, {"error": "use POST /analyze"})

    def analyze(self) -> str:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            # Read and dropped, not left unread: a sender cut off mid-body sees a reset, not the 413. The front's nginx
            # caps the body at the same 5 MB first, so this is a backstop.
            left = length
            while left > 0 and (chunk := self.rfile.read(min(65536, left))):
                left -= len(chunk)
            return self.json(413, {"error": f"this photo is {length / 2**20:.1f} MB; the limit is "
                                            f"{MAX_UPLOAD_BYTES // 2**20} MB"})
        body = self.rfile.read(length)
        part = photo_part(self.headers.get("Content-Type", ""), body)
        if part is None:
            return self.json(400, {"error": "no photo uploaded"})   # the app's wording (webapp/app.py, analyze)
        reason = refusal(*part)
        if reason is not None:
            return self.json(400, {"error": f"could not read this as an image: {reason}"})
        return self.forward(body)

    def forward(self, body: bytes) -> str:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        conn = http.client.HTTPConnection(*self.inference, timeout=INFERENCE_TIMEOUT_S)
        try:
            conn.request("POST", "/analyze", body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as exc:   # refused, unreachable, timed out: nothing has gone to the client yet
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
        except OSError as exc:   # either side gone mid-answer: the client sees a cut answer, as from the app
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
    handler = type("Handler", (CheckHandler,), {"inference": (url.hostname, url.port or 80)})
    server = ThreadingHTTPServer(listen, handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", required=True, help="host:port to serve on, loopback behind the front's nginx")
    ap.add_argument("--inference", required=True, help="the inference host's site, http://<tailnet address>:<port>")
    args = ap.parse_args(argv)
    host, _, port = args.listen.rpartition(":")
    server = make_server((host, int(port)), args.inference)
    sys.stderr.write(f"check service on {args.listen}, forwarding to {args.inference}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
