"""The front's check service (ticket OPS-15 P1), between a client and a fake inference host. Plain unittest.

    python -m unittest tests.test_front_check_service

webapp/front/check_service.py takes POST /analyze from the front's nginx, refuses what must not cross the tunnel
(over 5 MB, D15; not an image, D2), forwards the rest unchanged to the inference host, and streams its answer back
(ADR 0023); an inference host it cannot reach is "the classifier is offline", apart from busy (D12). The service
runs in this process on a loopback port, as under systemd; the inference host is a stdlib server here.
"""
from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "front"))
import check_service  # noqa: E402
from tests._tiers import real_data  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 2000
BOUNDARY = "----coffeeboundary7"


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
    reply = "stream"   # stream: two parts a second apart; or an int, a plain status with a JSON body

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        FakeInferenceHost.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if FakeInferenceHost.reply != "stream":
            payload = json.dumps({"error": f"inference host says {FakeInferenceHost.reply}"}).encode()
            self.send_response(FakeInferenceHost.reply)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Request-Id", "abcd1234")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Request-Id", "abcd1234")
        self.end_headers()
        self.wfile.write(b'{"part": "preview"}\n')
        self.wfile.flush()
        time.sleep(1.0)
        self.wfile.write(b'{"part": "classification"}\n')

    def log_message(self, *args):
        pass


class CheckService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The service's one line per request, to stderr under systemd, kept out of the test output.
        cls.enterClassContext(mock.patch.object(check_service.CheckHandler, "log_line"))
        cls.inference = ThreadingHTTPServer(("127.0.0.1", 0), FakeInferenceHost)
        threading.Thread(target=cls.inference.serve_forever, daemon=True).start()
        cls.service = cls.serve(f"http://127.0.0.1:{cls.inference.server_address[1]}")

    @classmethod
    def serve(cls, inference_url: str):
        server = check_service.make_server(("127.0.0.1", 0), inference_url)
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

    def post(self, body: bytes, server=None, headers: dict | None = None, path: str = "/analyze"):
        """Opens the request; the caller reads the response. Returns (connection, response)."""
        server = server or self.service
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        conn.request("POST", path, body=body, headers={
            "Content-Type": f"multipart/form-data; boundary={BOUNDARY}", **(headers or {})})
        self.addCleanup(conn.close)
        return conn, conn.getresponse()

    @real_data   # about 1.5 s: the fake answer's parts are a second apart
    def test_a_photo_crosses_unchanged_and_its_answer_streams_back(self):
        body = form("beans.jpg", JPEG, skip_crop=True)
        _, resp = self.post(body, headers={"X-Real-IP": "203.0.113.7", "User-Agent": "Mozilla/5.0 (Android)"})
        first = resp.readline()
        t_first = time.monotonic()
        second = resp.readline()
        waited = time.monotonic() - t_first
        self.assertEqual(resp.status, 200)
        self.assertEqual((first, second), (b'{"part": "preview"}\n', b'{"part": "classification"}\n'))
        self.assertGreater(waited, 0.5, "the preview was held back until the classification: the answer was buffered")
        self.assertEqual(resp.getheader("Content-Type"), "application/x-ndjson")
        self.assertEqual(resp.getheader("X-Accel-Buffering"), "no", "the front's nginx would buffer the answer")
        self.assertEqual(resp.getheader("X-Request-Id"), "abcd1234")
        [seen] = FakeInferenceHost.seen
        self.assertEqual(seen["path"], "/analyze")
        self.assertEqual(seen["body"], body)
        self.assertEqual(seen["headers"]["Content-Type"], f"multipart/form-data; boundary={BOUNDARY}")
        self.assertEqual(seen["headers"]["X-Real-IP"], "203.0.113.7")
        self.assertEqual(seen["headers"]["User-Agent"], "Mozilla/5.0 (Android)")


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

    @real_data   # about 1.5 s: two 5 MB uploads and one answer
    def test_a_photo_over_5_mb_never_crosses(self):
        body = form("beans.jpg", JPEG + b"\x00" * (5 * 2**20))
        status, payload = self.answer(body)
        self.assertEqual(status, 413)
        self.assertIn("5 MB", payload["error"])
        self.assertEqual(FakeInferenceHost.seen, [])
        ok = form("beans.jpg", JPEG + b"\x00" * (5 * 2**20 - 4096))
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
                self.assertEqual(resp.getheader("X-Request-Id"), "abcd1234")
                self.assertEqual(len(resp.msg.get_all("Server")), 1, "the inference host's Server header relayed too")

    def test_only_post_analyze_is_served(self):
        self.assertEqual(self.answer(form("beans.jpg", JPEG), path="/classify")[0], 404)
        conn = http.client.HTTPConnection("127.0.0.1", self.service.server_address[1], timeout=10)
        self.addCleanup(conn.close)
        conn.request("GET", "/analyze")
        self.assertEqual(conn.getresponse().status, 405)
        self.assertEqual(FakeInferenceHost.seen, [])


if __name__ == "__main__":
    unittest.main()
