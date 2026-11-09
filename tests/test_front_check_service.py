"""The front's check service (ticket OPS-15 P1), between a client and a fake inference host. Plain unittest.

    python -m unittest tests.test_front_check_service

webapp/front/check_service.py takes POST /analyze from the front's nginx, refuses what must not cross the tunnel
(over 40 MB, ML-19 D23; not an image, D2), canonicalizes the photo (ticket ML-19 P9: a canonical one goes on as
it is, one over the pixel ceiling is a 413, one that cannot be read a 400), forwards the form with the canonical photo
in place of the visitor's to the inference host, and streams its answer back (ADR 0023); an inference host it cannot
reach is "the classifier is offline", apart from busy (D12). The service runs in this process on a loopback port, as
under systemd; the inference host is a stdlib server here.
"""
from __future__ import annotations

import http.client
import io
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "front"))
import check_service  # noqa: E402

from coffeecv import canonical  # noqa: E402
from tests._tiers import real_data  # noqa: E402

def _jpeg() -> bytes:
    """A real photo, small: the service decodes and canonicalizes what it forwards. Bytes after its end marker (the
    cap tests pad it) are ignored by the decoder."""
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(buf, "JPEG",
                                                                                                 quality=90)
    return buf.getvalue()


JPEG = _jpeg()
RAW_DNG = "dataset/segmenter_positives/20261027_135613_Pixel_9_Pro_Guatemala.dng"   # 3628 x 2732, 9.9 MP
BOUNDARY = "----coffeeboundary7"
RID = "0123456789abcdef0123456789abcdef"   # a request ID as the front's nginx makes it ($request_id, OPS-3 D8)


def forwarded_photo(body: bytes) -> tuple[str, bytes]:
    """The filename and bytes of the photo part of a form the service forwarded."""
    filename, data, _ = check_service._photo(f"multipart/form-data; boundary={BOUNDARY}", body)
    return filename, data


def form(filename: str | None, data: bytes, skip_crop: bool = False) -> bytes:
    """A multipart/form-data body as a browser sends it: the photo part, and skip_crop when set."""
    parts = []
    if filename is not None:
        parts.append(f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="photo"; filename="{filename}"\r\n'
                     f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    if skip_crop:
        parts.append(f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="skip_crop"\r\n\r\n1\r\n'.encode())
    return b"".join(parts) + f"--{BOUNDARY}--\r\n".encode()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeInferenceHost(BaseHTTPRequestHandler):
    """yappy's side: records each request, then answers as `reply` says."""

    seen: list[dict] = []
    # stream: two parts a second apart; cut: the first part of a chunked answer, then the connection dropped;
    # garbage: not HTTP; or an int, a plain status with a JSON body
    reply = "stream"
    release = threading.Event()   # set: answer at once; cleared: hold the answer until it is set
    extra_headers: dict[str, str] = {}   # sent with every status or stream answer as well (tests/test_front_site.py)
    # The app answers under the ID it was sent (OPS-3 D8); off, an answer with none, as yappy's nginx's own busy 503.
    echo_id = True

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        FakeInferenceHost.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        FakeInferenceHost.release.wait(10)
        if FakeInferenceHost.reply == "garbage":
            self.wfile.write(b"this is not an HTTP answer\r\n\r\n")
            return
        if FakeInferenceHost.reply == "cut":
            self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n"
                             b'14\r\n{"part": "preview"}\n\r\n')
            self.wfile.flush()
            self.close_connection = True
            return
        if FakeInferenceHost.reply != "stream":
            payload = json.dumps({"error": f"inference host says {FakeInferenceHost.reply}"}).encode()
            self.send_response(FakeInferenceHost.reply)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_id()
            self.send_extra_headers()
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("X-Accel-Buffering", "no")
        self.send_id()
        self.send_extra_headers()
        self.end_headers()
        self.wfile.write(b'{"part": "preview"}\n')
        self.wfile.flush()
        time.sleep(1.0)
        self.wfile.write(b'{"part": "classification"}\n')

    def send_id(self):
        if FakeInferenceHost.echo_id and "X-Request-Id" in self.headers:
            self.send_header("X-Request-Id", self.headers["X-Request-Id"])

    def send_extra_headers(self):
        for k, v in FakeInferenceHost.extra_headers.items():
            self.send_header(k, v)

    def log_message(self, *args):
        pass


FakeInferenceHost.release.set()   # answering at once unless a test holds it (tests/test_front_site.py uses it too)


def log_lines(log_dir: Path) -> list[dict]:
    """The check service's log, one JSON object per line (OPS-3 P2)."""
    path = log_dir / check_service.LOG_FILE
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def log_line(log_dir: Path, rid: str) -> dict:
    """The check service's one line for request `rid`, once written: it is written after the answer has gone out."""
    for _ in range(100):
        found = [ln for ln in log_lines(log_dir) if ln.get("request_id") == rid]
        if found:
            [line] = found
            return line
        time.sleep(0.02)
    raise AssertionError(f"no log line for {rid}")


class CheckService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log_dir = Path(tempfile.mkdtemp(prefix="check-service-log-"))
        cls.addClassCleanup(shutil.rmtree, cls.log_dir, ignore_errors=True)
        cls.inference = ThreadingHTTPServer(("127.0.0.1", 0), FakeInferenceHost)
        threading.Thread(target=cls.inference.serve_forever, daemon=True).start()
        cls.service = cls.serve(f"http://127.0.0.1:{cls.inference.server_address[1]}")

    @classmethod
    def serve(cls, inference_url: str):
        server = check_service.make_server(("127.0.0.1", 0), inference_url, cls.log_dir)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        cls.addClassCleanup(server.server_close)
        cls.addClassCleanup(server.shutdown)
        return server

    @classmethod
    def tearDownClass(cls):
        cls.inference.shutdown()
        cls.inference.server_close()

    def setUp(self):
        FakeInferenceHost.seen.clear()
        FakeInferenceHost.reply = "stream"
        FakeInferenceHost.release.set()
        FakeInferenceHost.echo_id = True

    def post(self, body: bytes, server=None, headers: dict | None = None, path: str = "/analyze"):
        """Opens the request, with a fresh request ID unless `headers` names one; the caller reads the response.
        Returns (connection, response)."""
        server = server or self.service
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        conn.request("POST", path, body=body, headers={
            "Content-Type": f"multipart/form-data; boundary={BOUNDARY}", "X-Request-Id": uuid.uuid4().hex,
            **(headers or {})})
        self.addCleanup(conn.close)
        return conn, conn.getresponse()

    @real_data   # about 1.5 s: the fake answer's parts are a second apart
    def test_a_photo_crosses_canonicalized_and_its_answer_streams_back(self):
        body = form("beans.jpg", JPEG, skip_crop=True)
        _, resp = self.post(body, headers={"Accept-Language": "pt-BR", "User-Agent": "Mozilla/5.0 (Android)"})
        first = resp.readline()
        t_first = time.monotonic()
        second = resp.readline()
        waited = time.monotonic() - t_first
        self.assertEqual(resp.status, 200)
        self.assertEqual((first, second), (b'{"part": "preview"}\n', b'{"part": "classification"}\n'))
        self.assertGreater(waited, 0.5, "the preview was held back until the classification: the answer was buffered")
        self.assertEqual(resp.getheader("Content-Type"), "application/x-ndjson")
        self.assertEqual(resp.getheader("X-Accel-Buffering"), "no", "the front's nginx would buffer the answer")
        [seen] = FakeInferenceHost.seen
        self.assertEqual(resp.getheader("X-Request-Id"), seen["headers"]["X-Request-Id"])
        self.assertEqual(seen["path"], "/analyze")
        name, photo = forwarded_photo(seen["body"])
        self.assertEqual(name, "beans.jxl")
        self.assertTrue(canonical.is_canonical(photo))
        self.assertEqual(photo, canonical.canonicalize(JPEG, "beans.jpg"))   # the visitor's photo never crosses
        self.assertIn(b'name="skip_crop"\r\n\r\n1\r\n', seen["body"])     # every other field as sent
        self.assertEqual(seen["headers"]["Content-Type"], f"multipart/form-data; boundary={BOUNDARY}")
        self.assertEqual(seen["headers"]["Accept-Language"], "pt-BR")
        self.assertEqual(seen["headers"]["User-Agent"], "Mozilla/5.0 (Android)")


    def test_the_forwarded_name_cannot_break_the_part_header(self):
        body = form("beans.jpg", JPEG).replace(b'filename="beans.jpg"', b"filename*=utf-8''a%22b%0D%0AX-Evil%3A%201.jpg")
        _, resp = self.post(body)
        resp.read()
        self.assertEqual(resp.status, 200)
        name, photo = forwarded_photo(FakeInferenceHost.seen[0]["body"])
        self.assertEqual(name, "a_b__X-Evil__1.jxl")
        self.assertTrue(canonical.is_canonical(photo))

    def test_a_canonical_photo_crosses_as_it_is(self):   # ML-19 D17
        canon = canonical.canonicalize(JPEG, "beans.jpg")
        _, resp = self.post(form("beans.jxl", canon))
        resp.read()
        self.assertEqual(resp.status, 200)
        self.assertEqual(forwarded_photo(FakeInferenceHost.seen[0]["body"]), ("beans.jxl", canon))

    def test_a_photo_over_the_pixel_ceiling_never_crosses(self):   # ML-19 D38, D24
        for name, attr in (("beans.jpg", "MAX_PIXELS"), ("beans.dng", "RAW_MAX_PIXELS")):
            with self.subTest(name=name), mock.patch.object(check_service, attr, 48 * 64 - 1):
                photo = JPEG if name.endswith(".jpg") else b"II*\x00" + JPEG   # a TIFF signature for the .dng
                status, payload = self.answer(form(name, photo))
                if name.endswith(".dng"):   # not a real DNG: rawpy cannot read it, so the 400 comes first
                    self.assertEqual(status, 400)
                    continue
                self.assertEqual(status, 413)
                self.assertIn("too many pixels", payload["error"])
        self.assertEqual(FakeInferenceHost.seen, [])

    def test_a_failure_while_canonicalizing_is_an_answer_not_a_dropped_connection(self):
        with mock.patch.object(check_service.canonical, "encode", side_effect=MemoryError("no room")):
            self.assertEqual(self.answer(form("beans.jpg", JPEG)), (500, {"error": "could not process this photo"}))
        self.assertEqual(FakeInferenceHost.seen, [])

    def test_a_heic_crosses_canonicalized(self):   # ML-19 P9: the iPhone's format
        import pillow_heif
        heif = pillow_heif.from_bytes(mode="RGB", size=(64, 48), data=np.full((48, 64, 3), 90, np.uint8).tobytes())
        buf = io.BytesIO()
        heif.save(buf, quality=90)
        _, resp = self.post(form("IMG_0001.HEIC", buf.getvalue()))
        resp.read()
        self.assertEqual(resp.status, 200)
        name, photo = forwarded_photo(FakeInferenceHost.seen[0]["body"])
        self.assertEqual(name, "IMG_0001.jxl")
        self.assertTrue(canonical.is_canonical(photo))

    @real_data   # a real DNG from the dataset (dvc pull), developed by rawpy
    @unittest.skipUnless((REPO / RAW_DNG).is_file(), "needs the dataset's 9.9 MP DNG")
    def test_a_raw_photo_over_its_own_ceiling_never_crosses(self):   # ML-19 D24
        data = (REPO / RAW_DNG).read_bytes()
        with mock.patch.object(check_service, "MAX_UPLOAD_BYTES", 32 * 2**20), \
                mock.patch.object(check_service, "RAW_MAX_PIXELS", 9_000_000):
            status, payload = self.answer(form("photo.dng", data))
        self.assertEqual(status, 413)
        self.assertIn("camera RAW", payload["error"])
        self.assertEqual(FakeInferenceHost.seen, [])

    def test_a_photo_that_cannot_be_decoded_never_crosses(self):
        status, payload = self.answer(form("beans.jpg", JPEG[:200]))     # a JPEG's signature, cut short
        self.assertEqual(status, 400)
        self.assertTrue(payload["error"].startswith("could not read this as an image: "), payload)
        self.assertEqual(FakeInferenceHost.seen, [])

    def answer(self, body: bytes, **kw) -> tuple[int, dict]:
        _, resp = self.post(body, **kw)
        return resp.status, json.loads(resp.read())

    def test_a_file_that_is_not_an_image_never_crosses(self):
        status, payload = self.answer(form("beans.jpg", b"just some text, renamed"))
        self.assertEqual(status, 400)
        self.assertTrue(payload["error"].startswith("could not read this as an image: "), payload)
        self.assertEqual(FakeInferenceHost.seen, [])

    def test_a_request_without_a_photo_is_the_apps_400(self):
        self.assertEqual(self.answer(form(None, b"")), (400, {"error": "no photo uploaded"}))
        self.assertEqual(self.answer(form("", b"")), (400, {"error": "no photo uploaded"}))
        self.assertEqual(FakeInferenceHost.seen, [])

    @real_data   # a few seconds: two 40 MB uploads and one answer
    def test_a_photo_over_the_cap_never_crosses(self):   # ML-19 D23
        cap = check_service.MAX_UPLOAD_BYTES
        self.assertEqual(cap, 40 * 2**20)
        body = form("beans.jpg", JPEG + b"\x00" * cap)
        status, payload = self.answer(body)
        self.assertEqual(status, 413)
        self.assertIn("40 MB", payload["error"])
        self.assertEqual(FakeInferenceHost.seen, [])
        ok = form("beans.jpg", JPEG + b"\x00" * (cap - 4096))
        _, resp = self.post(ok)
        resp.read()
        self.assertEqual(resp.status, 200, "a photo just under the cap is refused")

    def test_an_unreachable_inference_host_is_offline(self):
        service = self.serve(f"http://127.0.0.1:{free_port()}")   # nothing listens there
        status, payload = self.answer(form("beans.jpg", JPEG), server=service)
        self.assertEqual((status, payload), (503, {"error": "the classifier is offline"}))

    def test_the_inference_hosts_app_down_is_offline(self):
        for down in (502, 504):
            with self.subTest(status=down):
                FakeInferenceHost.reply = down
                self.assertEqual(self.answer(form("beans.jpg", JPEG)), (503, {"error": "the classifier is offline"}))

    def test_the_inference_hosts_own_answers_are_relayed_as_they_are(self):
        for status in (400, 503, 500):   # its app's refusal, its busy backstop (D11), a failure
            with self.subTest(status=status):
                FakeInferenceHost.reply = status
                _, resp = self.post(form("beans.jpg", JPEG))
                self.assertEqual((resp.status, json.loads(resp.read())),
                                 (status, {"error": f"inference host says {status}"}))
                self.assertEqual(resp.getheader("X-Request-Id"), FakeInferenceHost.seen[-1]["headers"]["X-Request-Id"])
                self.assertEqual(len(resp.msg.get_all("Server")), 1, "the inference host's Server header relayed too")

    def test_only_post_analyze_is_served(self):
        self.assertEqual(self.answer(form("beans.jpg", JPEG), path="/classify")[0], 404)
        conn = http.client.HTTPConnection("127.0.0.1", self.service.server_address[1], timeout=10)
        self.addCleanup(conn.close)
        conn.request("GET", "/analyze")
        self.assertEqual(conn.getresponse().status, 405)
        self.assertEqual(FakeInferenceHost.seen, [])


    def line(self, resp) -> dict:
        """The service's log line for the request `resp` answered."""
        return log_line(self.log_dir, resp.getheader("X-Request-Id"))

    def test_a_cut_answer_ends_the_visitors_answer_where_it_was_cut(self):
        FakeInferenceHost.reply = "cut"
        _, resp = self.post(form("beans.jpg", JPEG))
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.read(), b'{"part": "preview"}\n')
        line = self.line(resp)
        self.assertEqual((line["status"], line["outcome"], line["relayed_bytes"]), (200, "cut", 20))
        self.assertEqual(line["error"], "IncompleteRead")

    @real_data   # about 1 s: the fake answer's parts are a second apart
    def test_the_log_line_of_a_canonicalized_photo(self):   # OPS-3 P2, D6, D10; ML-19 D38 for the ceiling's review
        _, resp = self.post(form("IMG 0001.jpg", JPEG))
        answer = resp.read()
        line = self.line(resp)
        rid = resp.getheader("X-Request-Id")
        timings = {k: line.pop(k) for k in ("read_ms", "check_ms", "canonicalize_ms", "crossing_ms", "to_headers_ms",
                                            "to_preview_ms", "to_end_ms", "total_ms", "peak_mb", "swap_mb")}
        self.assertRegex(line.pop("ts"), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00$")
        self.assertEqual(line, {
            "kind": "request", "request_id": rid, "method": "POST", "endpoint": "/analyze", "status": 200,
            "outcome": "relayed", "filename": "IMG 0001.jpg", "ext": ".jpg", "format": "jpeg", "bytes": len(JPEG),
            "canonical": "canonicalized", "width": 64, "height": 48, "bits": 8,
            "canonical_bytes": len(canonical.canonicalize(JPEG, "beans.jpg")), "relayed_bytes": len(answer)})
        for k, v in timings.items():
            self.assertIsInstance(v, int, k)
            self.assertGreaterEqual(v, 0, k)
        self.assertGreaterEqual(timings["to_end_ms"] - timings["to_preview_ms"], 900, "the parts are a second apart")
        self.assertLessEqual(timings["to_headers_ms"], timings["to_preview_ms"])
        self.assertGreaterEqual(timings["total_ms"], timings["to_end_ms"] + timings["crossing_ms"])

    def test_the_log_line_of_a_photo_sent_canonical(self):   # ML-19 D17: nothing decoded, the original's own fields
        canon = canonical.canonicalize(JPEG, "beans.jpg")
        _, resp = self.post(form("beans.jxl", canon))
        resp.read()
        line = self.line(resp)
        self.assertEqual({k: line.get(k) for k in ("filename", "ext", "format", "bytes", "canonical", "canonical_bytes",
                                                   "canonicalize_ms", "width")},
                         {"filename": "beans.jxl", "ext": ".jxl", "format": "jxl", "bytes": len(canon),
                          "canonical": "as_sent", "canonical_bytes": len(canon), "canonicalize_ms": None,
                          "width": None})

    def test_each_refusal_is_logged_by_kind(self):   # OPS-3 P2, D13's failures by kind
        def cases():
            yield "not_an_image", form("notes.txt", b"some text"), 400
            yield "unreadable", form("beans.jpg", JPEG[:200]), 400
            yield "no_photo", form(None, b""), 400
            yield "too_large", form("beans.jpg", JPEG + b"\x00" * check_service.MAX_UPLOAD_BYTES), 413
            with mock.patch.object(check_service, "MAX_PIXELS", 48 * 64 - 1):
                yield "too_many_pixels", form("beans.jpg", JPEG), 413
            with mock.patch.object(check_service.canonical, "encode", side_effect=MemoryError("no room")):
                yield "failed", form("beans.jpg", JPEG), 500
            FakeInferenceHost.reply = 502
            yield "offline", form("beans.jpg", JPEG), 503
            FakeInferenceHost.reply = 400
            yield "relayed", form("beans.jpg", JPEG), 400
            FakeInferenceHost.reply = "stream"
            gate = self.service.RequestHandlerClass.crossing
            gate.acquire()
            try:
                yield "busy", form("beans.jpg", JPEG), 503
            finally:
                gate.release()
        for outcome, body, status in cases():
            with self.subTest(outcome):
                _, resp = self.post(body)
                resp.read()
                line = self.line(resp)
                self.assertEqual((line["status"], line["outcome"]), (status, outcome))
        self.assertEqual(self.line(resp).get("format"), "jpeg")
        line = log_line(self.log_dir, next(ln["request_id"] for ln in reversed(log_lines(self.log_dir))
                                           if ln["outcome"] == "not_an_image"))
        self.assertEqual((line["filename"], line["ext"], line["bytes"], line.get("format")), ("notes.txt", ".txt", 9, None))
        line = log_line(self.log_dir, next(ln["request_id"] for ln in reversed(log_lines(self.log_dir))
                                           if ln["outcome"] == "offline"))
        self.assertEqual(line["upstream_status"], 502)

    def test_an_unreachable_inference_host_is_logged_with_its_error(self):
        service = self.serve(f"http://127.0.0.1:{free_port()}")   # nothing listens there
        _, resp = self.post(form("beans.jpg", JPEG), server=service)
        resp.read()
        line = self.line(resp)
        self.assertEqual((line["outcome"], line["error"]), ("offline", "ConnectionRefusedError"))

    def test_the_probe_marker_is_logged_and_nothing_else_a_visitor_puts_there(self):   # OPS-3 D14
        for sent, logged in (("verify", "verify"), ("anything", None), (None, None)):
            with self.subTest(sent=sent):
                _, resp = self.post(form("beans.jpg", b"some text"),
                                    headers={} if sent is None else {"X-Coffee-CV-Probe": sent})
                resp.read()
                self.assertEqual(self.line(resp).get("probe"), logged)

    def test_a_filename_is_capped_and_always_one_line(self):   # OPS-3 D10, D17's layer 2
        for encoded, kept in ((b"a%0Ab%0D%0A%22c%00.jpg", 'a\nb\r\n"c\x00.jpg'), (b"x" * 300 + b".jpg", "x" * 200)):
            with self.subTest(kept=kept[:12]):
                body = form("beans.jpg", b"some text").replace(b'filename="beans.jpg"', b"filename*=utf-8''" + encoded)
                _, resp = self.post(body)
                resp.read()
                self.assertEqual(self.line(resp)["filename"], kept)
        for raw in (self.log_dir / check_service.LOG_FILE).read_bytes().splitlines():
            json.loads(raw)

    def test_no_line_holds_an_address_and_nothing_goes_to_stderr(self):   # OPS-3 D9: no IP; the journal keeps start-up
        with mock.patch("sys.stderr") as stderr:
            for body in (form("beans.jpg", JPEG), form("beans.jpg", b"some text")):
                _, resp = self.post(body)
                resp.read()
                self.line(resp)
        stderr.write.assert_not_called()
        self.assertNotIn("127.0.0.", (self.log_dir / check_service.LOG_FILE).read_text())

    def test_a_rotated_log_is_reopened(self):   # logrotate moves the file; the next line starts a new one
        path = self.log_dir / check_service.LOG_FILE
        _, resp = self.post(form("beans.jpg", b"some text"))
        resp.read()
        self.line(resp)
        path.rename(path.with_suffix(".jsonl.1"))
        self.addCleanup(lambda: path.with_suffix(".jsonl.1").unlink(missing_ok=True))
        _, resp = self.post(form("beans.jpg", b"some text"))
        resp.read()
        rid = self.line(resp)["request_id"]
        self.assertEqual([ln["request_id"] for ln in log_lines(self.log_dir)], [rid])

    def test_an_answer_that_is_not_http_is_offline(self):
        FakeInferenceHost.reply = "garbage"
        self.assertEqual(self.answer(form("beans.jpg", JPEG)), (503, {"error": "the classifier is offline"}))

    def test_an_answer_that_never_starts_is_offline_after_the_answers_timeout(self):
        """The photo has crossed; the answer's reads get ANSWER_TIMEOUT_S, not the upload's longer one (OPS-16 D8)."""
        FakeInferenceHost.release.clear()
        self.addCleanup(FakeInferenceHost.release.set)
        with mock.patch.multiple(check_service, UPLOAD_TIMEOUT_S=30, ANSWER_TIMEOUT_S=0.3):
            t0 = time.monotonic()
            self.assertEqual(self.answer(form("beans.jpg", JPEG)), (503, {"error": "the classifier is offline"}))
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual(len(FakeInferenceHost.seen), 1, "the photo never crossed")

    @real_data   # about 3 s: canonicalizing a 3 MP photo of noise
    def test_an_upload_that_stalls_is_offline_after_the_uploads_timeout(self):
        """An inference host that takes the connection and never reads: the whole upload gets UPLOAD_TIMEOUT_S."""
        stalled = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(stalled.close)
        held = []
        threading.Thread(target=lambda: held.append(stalled.accept()[0]), daemon=True).start()
        self.addCleanup(lambda: [s.close() for s in held])
        service = self.serve(f"http://127.0.0.1:{stalled.getsockname()[1]}")
        # More than loopback's buffers hold, as forwarded: a canonical photo of noise (4.6 MB), which goes on as it is.
        noise = np.random.default_rng(1).integers(0, 255, (1500, 2000, 3), dtype=np.uint8)
        photo = canonical.canonicalize(imagecodecs.png_encode(noise))
        with mock.patch.multiple(check_service, UPLOAD_TIMEOUT_S=0.3, ANSWER_TIMEOUT_S=30):
            t0 = time.monotonic()
            self.assertEqual(self.answer(form("beans.jpg", photo), server=service),
                             (503, {"error": "the classifier is offline"}))
        self.assertLess(time.monotonic() - t0, 5)

    def test_a_query_string_still_reaches_the_app(self):
        _, resp = self.post(form("beans.jpg", JPEG), path="/analyze?source=agent")
        resp.read()
        self.assertEqual(resp.status, 200)
        self.assertEqual(FakeInferenceHost.seen[0]["path"], "/analyze?source=agent")

    def test_a_request_without_a_length_is_refused(self):
        for length in (None, "many"):
            with self.subTest(length=length), socket.create_connection(self.service.server_address, timeout=10) as s:
                head = "POST /analyze HTTP/1.1\r\nHost: front\r\n"
                head += f"Content-Length: {length}\r\n" if length else ""
                s.sendall((head + "\r\n").encode())
                reply = b""
                while chunk := s.recv(4096):   # the service closes the connection after its answer
                    reply += chunk
                self.assertRegex(reply, rb"^HTTP/1\.[01] (400|411) ")
                self.assertIn(b'"error"', reply)
        self.assertEqual(FakeInferenceHost.seen, [])

    def test_the_cap_is_the_photos_own_size_not_the_forms(self):
        photo = JPEG + b"\x00" * (check_service.MAX_UPLOAD_BYTES - len(JPEG))   # exactly the cap: its form is larger
        _, resp = self.post(form("beans.jpg", photo, skip_crop=True))
        resp.read()
        self.assertEqual(resp.status, 200)
        status, payload = self.answer(form("beans.jpg", photo + b"\x00"))
        self.assertEqual((status, payload), (413, {"error": "this photo is 40.1 MB; the limit is 40 MB"}))

    def test_one_photo_crosses_at_a_time(self):
        """The front's nginx frees its gate when a visitor leaves; the crossing it started goes on (D11)."""
        FakeInferenceHost.release.clear()
        first = threading.Thread(target=lambda: self.post(form("beans.jpg", JPEG))[1].read())
        first.start()
        for _ in range(100):
            if FakeInferenceHost.seen:
                break
            time.sleep(0.02)
        status, payload = self.answer(form("beans.jpg", JPEG))
        FakeInferenceHost.release.set()
        first.join(10)
        self.assertEqual((status, payload), (503, {"error": "the classifier is busy with another photo"}))
        self.assertEqual(len(FakeInferenceHost.seen), 1)

    def test_the_fronts_request_id_crosses_and_comes_back(self):   # OPS-3 D8
        _, resp = self.post(form("beans.jpg", JPEG), headers={"X-Request-Id": RID})
        resp.read()
        self.assertEqual(resp.getheader("X-Request-Id"), RID)
        [seen] = FakeInferenceHost.seen
        self.assertEqual(seen["headers"]["X-Request-Id"], RID)

    def test_a_missing_or_malformed_request_id_is_replaced_by_one_in_nginxs_form(self):   # OPS-3 P1
        for sent in (None, "", "abcd1234", RID.upper(), RID + "0", "g" * 32, f"{RID}, {RID}"):
            with self.subTest(sent=sent):
                FakeInferenceHost.seen.clear()
                _, resp = self.post(form("beans.jpg", JPEG), headers={} if sent is None else {"X-Request-Id": sent})
                resp.read()
                [seen] = FakeInferenceHost.seen
                made = seen["headers"]["X-Request-Id"]
                self.assertRegex(made, r"^[0-9a-f]{32}$")
                self.assertNotEqual(made, sent)
                self.assertEqual(resp.getheader("X-Request-Id"), made)

    def test_every_answer_of_its_own_carries_the_request_id(self):   # OPS-3 P1: the page needs it for its timeline
        def refusals():
            yield "not an image", form("beans.jpg", b"some text"), 400
            yield "no photo", form(None, b""), 400
            yield "over the cap", form("beans.jpg", JPEG + b"\x00" * check_service.MAX_UPLOAD_BYTES), 413
            with mock.patch.object(check_service.canonical, "encode", side_effect=MemoryError("no room")):
                yield "failed", form("beans.jpg", JPEG), 500
            FakeInferenceHost.reply = 502
            yield "offline", form("beans.jpg", JPEG), 503
            FakeInferenceHost.reply, FakeInferenceHost.echo_id = 503, False
            yield "the inference host's busy, with no id", form("beans.jpg", JPEG), 503
            FakeInferenceHost.reply, FakeInferenceHost.echo_id = "stream", True
            gate = self.service.RequestHandlerClass.crossing
            gate.acquire()
            try:
                yield "busy", form("beans.jpg", JPEG), 503
            finally:
                gate.release()
        for name, body, status in refusals():
            with self.subTest(name):
                _, resp = self.post(body, headers={"X-Request-Id": RID})
                resp.read()
                self.assertEqual((resp.status, resp.getheader("X-Request-Id")), (status, RID))

    def test_a_browsers_form_in_any_case_and_an_encoded_filename(self):
        body = form("beans.jpg", JPEG).replace(b'filename="beans.jpg"', b"filename*=utf-8''caf%C3%A9.jpg")
        _, resp = self.post(body, headers={"Content-Type": f"Multipart/Form-Data; boundary={BOUNDARY}"})
        resp.read()
        self.assertEqual(resp.status, 200)


class Options(unittest.TestCase):
    """Under systemd the unit gives no flags: front.env, in the release, names the address and the inference host."""

    ENV = {"COFFEE_CV_FRONT_LISTEN": "127.0.0.1:8100", "COFFEE_CV_INFERENCE_URL": "http://inference.test:12346"}

    def test_from_the_environment(self):
        self.assertEqual(check_service.options([], self.ENV),
                         (("127.0.0.1", 8100), "http://inference.test:12346", Path("/var/log/coffee-cv-front")))
        self.assertEqual(check_service.options([], {**self.ENV, "LOGS_DIRECTORY": "/tmp/x"})[2], Path("/tmp/x"))

    def test_a_flag_wins(self):
        self.assertEqual(check_service.options(["--listen", "127.0.0.1:9", "--log-dir", "/tmp/y"], self.ENV),
                         (("127.0.0.1", 9), "http://inference.test:12346", Path("/tmp/y")))

    def test_an_ipv6_address(self):
        self.assertEqual(check_service.options(["--listen", "[::1]:8100"], self.ENV)[0], ("::1", 8100))

    def test_neither_is_refused(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            check_service.options([], {})


if __name__ == "__main__":
    unittest.main()
