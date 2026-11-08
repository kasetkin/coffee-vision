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
import socket
import sys
import threading
import time
import unittest
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
            self.send_header("X-Request-Id", "abcd1234")
            self.send_extra_headers()
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Request-Id", "abcd1234")
        self.send_extra_headers()
        self.end_headers()
        self.wfile.write(b'{"part": "preview"}\n')
        self.wfile.flush()
        time.sleep(1.0)
        self.wfile.write(b'{"part": "classification"}\n')

    def send_extra_headers(self):
        for k, v in FakeInferenceHost.extra_headers.items():
            self.send_header(k, v)

    def log_message(self, *args):
        pass


FakeInferenceHost.release.set()   # answering at once unless a test holds it (tests/test_front_site.py uses it too)


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
        FakeInferenceHost.release.set()

    def post(self, body: bytes, server=None, headers: dict | None = None, path: str = "/analyze"):
        """Opens the request; the caller reads the response. Returns (connection, response)."""
        server = server or self.service
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        conn.request("POST", path, body=body, headers={
            "Content-Type": f"multipart/form-data; boundary={BOUNDARY}", **(headers or {})})
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
        self.assertEqual(resp.getheader("X-Request-Id"), "abcd1234")
        [seen] = FakeInferenceHost.seen
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
                self.assertEqual(resp.getheader("X-Request-Id"), "abcd1234")
                self.assertEqual(len(resp.msg.get_all("Server")), 1, "the inference host's Server header relayed too")

    def test_only_post_analyze_is_served(self):
        self.assertEqual(self.answer(form("beans.jpg", JPEG), path="/classify")[0], 404)
        conn = http.client.HTTPConnection("127.0.0.1", self.service.server_address[1], timeout=10)
        self.addCleanup(conn.close)
        conn.request("GET", "/analyze")
        self.assertEqual(conn.getresponse().status, 405)
        self.assertEqual(FakeInferenceHost.seen, [])


    def test_a_cut_answer_ends_the_visitors_answer_where_it_was_cut(self):
        FakeInferenceHost.reply = "cut"
        with mock.patch.object(check_service.CheckHandler, "log_line") as log:
            _, resp = self.post(form("beans.jpg", JPEG))
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.read(), b'{"part": "preview"}\n')
            time.sleep(0.2)
        self.assertIn("cut after", log.call_args[0][0])

    def test_the_log_line_says_what_was_canonicalized_and_its_peak_memory(self):   # ML-19 D38: for the ceiling's review
        with mock.patch.object(check_service.CheckHandler, "log_line") as log:
            _, resp = self.post(form("beans.jpg", JPEG))
            resp.read()
            time.sleep(0.2)
        line = log.call_args[0][0]
        self.assertRegex(line, r"^POST /analyze 200 relayed \d+ bytes \d+ms; \.jpg 64x48 8-bit, canonicalized in "
                               r"\d+\.\d s, peak \d+ MB, swap \d+ MB$")
        with mock.patch.object(check_service.CheckHandler, "log_line") as log:
            _, resp = self.post(form("beans.jxl", check_service.canonical.canonicalize(JPEG, "beans.jpg")))
            resp.read()
            time.sleep(0.2)
        self.assertTrue(log.call_args[0][0].endswith("ms; canonical as sent"), log.call_args[0][0])

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

    def test_a_browsers_form_in_any_case_and_an_encoded_filename(self):
        body = form("beans.jpg", JPEG).replace(b'filename="beans.jpg"', b"filename*=utf-8''caf%C3%A9.jpg")
        _, resp = self.post(body, headers={"Content-Type": f"Multipart/Form-Data; boundary={BOUNDARY}"})
        resp.read()
        self.assertEqual(resp.status, 200)


class Options(unittest.TestCase):
    """Under systemd the unit gives no flags: front.env, in the release, names the address and the inference host."""

    ENV = {"COFFEE_CV_FRONT_LISTEN": "127.0.0.1:8100", "COFFEE_CV_INFERENCE_URL": "http://inference.test:12346"}

    def test_from_the_environment(self):
        self.assertEqual(check_service.options([], self.ENV), (("127.0.0.1", 8100), "http://inference.test:12346"))

    def test_a_flag_wins(self):
        self.assertEqual(check_service.options(["--listen", "127.0.0.1:9"], self.ENV),
                         (("127.0.0.1", 9), "http://inference.test:12346"))

    def test_an_ipv6_address(self):
        self.assertEqual(check_service.options(["--listen", "[::1]:8100"], self.ENV)[0], ("::1", 8100))

    def test_neither_is_refused(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            check_service.options([], {})


if __name__ == "__main__":
    unittest.main()
