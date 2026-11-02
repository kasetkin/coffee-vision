"""The front's nginx site (ticket OPS-15 P3), served by a real nginx behind a stream map like the proxy's. Plain unittest.

    python -m unittest tests.test_front_site

webapp/deploy/coffee-cv-front.nginx.conf.template is what the proxy serves for the domain (ADR 0031): TLS on a local
port that the proxy's stream map reaches, the page and the agent files from the release, and /analyze through the gate
(D11), the 5 MB cap (D15) and the check service (P1) to the inference host; its offline 503 kept apart from busy (D12).
The visitor's IP is in the stream block's session log alone (webapp/deploy/coffee-cv-stream.conf.template, D20),
rotated at 30 days. Here the stream server with that snippet, the site, the real check service and a fake inference
host all run on loopback, the /var/log/nginx paths moved to a scratch directory; the visitor connects from 127.0.0.5,
so its address is told apart from the stream hop's 127.0.0.1. Without nginx and its stream module the tests skip.
"""
from __future__ import annotations

import http.client
import json
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "webapp" / "deploy"
SITE_TEMPLATE = DEPLOY / "coffee-cv-front.nginx.conf.template"
STREAM_TEMPLATE = DEPLOY / "coffee-cv-stream.conf.template"
sys.path.insert(0, str(DEPLOY))
sys.path.insert(0, str(REPO / "webapp" / "front"))
import check_service  # noqa: E402
from render_template import render  # noqa: E402

from tests._tiers import real_data  # noqa: E402
from tests.test_front_check_service import BOUNDARY, JPEG, FakeInferenceHost, form, free_port  # noqa: E402

NGINX = shutil.which("nginx") or next((p for p in ("/usr/sbin/nginx",) if Path(p).exists()), None)
STREAM_MODULE = Path("/usr/lib/nginx/modules/ngx_stream_module.so")
DOMAIN = "coffee.test"
VISITOR = "127.0.0.5"   # the visitor's address; the stream hop to the site comes from 127.0.0.1
NGINX_LOG_DIR = "/var/log/nginx/"
STATIC = REPO / "webapp" / "static"


class Visit(http.client.HTTPSConnection):
    """An HTTPS request as a visitor makes it: from VISITOR, to the proxy's stream port, with the domain as SNI."""

    def __init__(self, port: int, ca: Path):
        super().__init__("127.0.0.1", port, timeout=30, context=ssl.create_default_context(cafile=str(ca)))

    def connect(self):
        raw = socket.create_connection((self.host, self.port), self.timeout, source_address=(VISITOR, 0))
        self.sock = self._context.wrap_socket(raw, server_hostname=DOMAIN)


@real_data   # several seconds: nginx, the check service and the fake answers' one-second gaps
class FrontSite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if NGINX is None or not STREAM_MODULE.exists():
            raise unittest.SkipTest("no nginx with its stream module on this machine")
        # The service's one line per request, to stderr under systemd, kept out of the test output.
        cls.enterClassContext(mock.patch.object(check_service.CheckHandler, "log_line"))
        cls.tmp = Path(tempfile.mkdtemp(prefix="front-site-"))
        cls.logs = cls.tmp / "log"
        cls.logs.mkdir()
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj",
                        f"/CN={DOMAIN}", "-addext", f"subjectAltName=DNS:{DOMAIN}", "-keyout", cls.tmp / "key.pem",
                        "-out", cls.tmp / "cert.pem"], check=True, capture_output=True)
        app_root = cls.tmp / "opt"
        app_root.mkdir()
        (app_root / "current").symlink_to(REPO)   # the release's tree: current/webapp/static

        cls.inference = ThreadingHTTPServer(("127.0.0.1", 0), FakeInferenceHost)
        threading.Thread(target=cls.inference.serve_forever, daemon=True).start()
        cls.check_port = free_port()
        cls.start_check_service()

        cls.stream_port, front_port = free_port(), free_port()
        site = render(SITE_TEMPLATE.read_text(), {
            "DOMAIN": DOMAIN, "FRONT_PORT": str(front_port), "CHECK_PORT": str(cls.check_port),
            "APP_ROOT": str(app_root), "TLS_CERT": str(cls.tmp / "cert.pem"), "TLS_KEY": str(cls.tmp / "key.pem")})
        assert NGINX_LOG_DIR in site, f"the site no longer logs under {NGINX_LOG_DIR}"
        (cls.tmp / "site.conf").write_text(site.replace(NGINX_LOG_DIR, f"{cls.logs}/"))
        temp_paths = "".join(f"    {kind}_temp_path {cls.tmp}/{kind};\n"
                             for kind in ("client_body", "proxy", "fastcgi", "uwsgi", "scgi"))
        # The proxy's stream block: the domain's SNI to the site's local port, anything else nowhere, and the session
        # log the owner includes there (D20).
        stream_log = render(STREAM_TEMPLATE.read_text(), {"DOMAIN": DOMAIN})
        assert NGINX_LOG_DIR in stream_log, f"the stream snippet no longer logs under {NGINX_LOG_DIR}"
        (cls.tmp / "stream.conf").write_text(stream_log.replace(NGINX_LOG_DIR, f"{cls.logs}/"))
        (cls.tmp / "nginx.conf").write_text(f"""load_module {STREAM_MODULE};
pid {cls.tmp}/nginx.pid;
error_log {cls.tmp}/error.log;
events {{}}
stream {{
    map $ssl_preread_server_name $target {{
        {DOMAIN} 127.0.0.1:{front_port};
        default 127.0.0.1:9;
    }}
    include {cls.tmp}/stream.conf;
    server {{
        listen 127.0.0.1:{cls.stream_port};
        ssl_preread on;
        proxy_pass $target;
    }}
}}
http {{
{temp_paths}    access_log off;
    include {cls.tmp}/site.conf;
}}
""")
        cmd = [NGINX, "-p", str(cls.tmp), "-c", str(cls.tmp / "nginx.conf")]
        check = subprocess.run([*cmd, "-t"], capture_output=True, text=True)
        if check.returncode:
            cls.tearDownClass()
            raise AssertionError(f"nginx -t rejected the rendered site:\n{check.stderr}")
        cls.nginx = subprocess.Popen([*cmd, "-g", "daemon off;"], stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", cls.stream_port), timeout=0.1).close()
                break
            except OSError:
                time.sleep(0.05)

    @classmethod
    def start_check_service(cls):
        cls.check = check_service.make_server(("127.0.0.1", cls.check_port),
                                              f"http://127.0.0.1:{cls.inference.server_address[1]}")
        threading.Thread(target=cls.check.serve_forever, daemon=True).start()

    @classmethod
    def stop_check_service(cls):
        cls.check.shutdown()
        cls.check.server_close()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "nginx", None):
            cls.nginx.terminate()
            cls.nginx.wait(10)
        cls.stop_check_service()
        cls.inference.shutdown()
        cls.inference.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        FakeInferenceHost.seen.clear()
        FakeInferenceHost.reply = "stream"

    def visit(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        """Opens the request; the caller reads the response."""
        conn = Visit(self.stream_port, self.tmp / "cert.pem")
        self.addCleanup(conn.close)
        headers = {**({"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"} if body is not None else {}),
                   **(headers or {})}
        conn.request(method, path, body=body, headers=headers)
        return conn.getresponse()

    def answer(self, body: bytes) -> tuple[int, dict]:
        resp = self.visit("POST", "/analyze", body)
        return resp.status, json.loads(resp.read())

    def test_the_page_and_the_agent_files_are_the_releases(self):
        resp = self.visit("GET", "/")
        self.assertEqual((resp.status, resp.read()), (200, (STATIC / "index.html").read_bytes()))
        self.assertEqual(resp.getheader("Cache-Control"), "no-cache")
        for path in ("/llms.txt", "/openapi.json", "/.well-known/ard.json"):
            with self.subTest(path=path):
                resp = self.visit("GET", path)
                self.assertEqual((resp.status, resp.read()), (200, (STATIC / path[1:]).read_bytes()))
                self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), "*")
        resp = self.visit("GET", "/no-such-file")
        resp.read()
        self.assertEqual(resp.status, 404)

    def test_a_photo_reaches_the_inference_host_and_streams_back(self):
        resp = self.visit("POST", "/analyze", form("beans.jpg", JPEG), {"X-Real-IP": "203.0.113.9"})
        first = resp.readline()
        t_first = time.monotonic()
        second = resp.readline()
        waited = time.monotonic() - t_first
        self.assertEqual(resp.status, 200)
        self.assertEqual((first, second), (b'{"part": "preview"}\n', b'{"part": "classification"}\n'))
        self.assertGreater(waited, 0.5, "the preview was held back until the classification: the answer was buffered")
        self.assertEqual(resp.getheader("X-Request-Id"), "abcd1234")
        [seen] = FakeInferenceHost.seen
        self.assertNotIn("X-Real-IP", seen["headers"], "a visitor's own X-Real-IP went through (D20)")

    def test_a_photo_over_5_mb_is_refused_at_the_front(self):
        photo = JPEG + b"\x00" * (5 * 2**20 - len(JPEG))
        resp = self.visit("POST", "/analyze", form("beans.jpg", photo))   # exactly 5 MB: its form is larger
        resp.read()
        self.assertEqual(resp.status, 200)
        FakeInferenceHost.seen.clear()
        self.assertEqual(self.answer(form("beans.jpg", photo + b"\x00")),   # the check service's 413
                         (413, {"error": "this photo is 5.1 MB; the limit is 5 MB"}))
        self.assertEqual(self.answer(form("beans.jpg", photo + b"\x00" * 2**20)),   # nginx's, before it
                         (413, check_service.TOO_LARGE))
        self.assertEqual(FakeInferenceHost.seen, [])

    def test_a_file_that_is_not_an_image_is_refused_at_the_front(self):
        status, payload = self.answer(form("beans.jpg", b"some text"))
        self.assertEqual(status, 400)
        self.assertTrue(payload["error"].startswith("could not read this as an image: "), payload)
        self.assertEqual(FakeInferenceHost.seen, [])

    def test_a_second_photo_while_one_is_in_flight_is_busy(self):
        FakeInferenceHost.reply = "stream"
        first = self.visit("POST", "/analyze", form("beans.jpg", JPEG))   # held: its answer takes a second
        status, payload = self.answer(form("beans.jpg", JPEG))
        first.read()
        self.assertEqual((status, payload), (503, check_service.BUSY))
        self.assertEqual(len(FakeInferenceHost.seen), 1)

    def test_the_inference_host_down_is_offline(self):
        FakeInferenceHost.reply = 502
        self.assertEqual(self.answer(form("beans.jpg", JPEG)), (503, check_service.OFFLINE))

    def test_the_check_service_stopped_is_offline_and_the_page_stays_up(self):
        self.stop_check_service()   # a deploy stops it (D14)
        try:
            self.assertEqual(self.answer(form("beans.jpg", JPEG)), (503, check_service.OFFLINE))
            resp = self.visit("GET", "/")
            resp.read()
            self.assertEqual(resp.status, 200)
        finally:
            self.start_check_service()

    def test_the_visitors_ip_is_only_in_the_session_log_rotated_at_30_days(self):
        conn = Visit(self.stream_port, self.tmp / "cert.pem")
        conn.request("POST", "/analyze", body=form("beans.jpg", JPEG),
                     headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
        conn.getresponse().read()
        conn.close()   # the stream block logs a session when it ends
        sessions = self.logs / "coffee-cv-front.sessions"
        for _ in range(50):
            if sessions.exists() and VISITOR in sessions.read_text():
                break
            time.sleep(0.1)
        rotate = logrotate_days()
        with_ip = {p.name for p in self.logs.iterdir() if VISITOR in p.read_text()}
        self.assertEqual(with_ip, {"coffee-cv-front.sessions"})
        self.assertEqual(rotate.get(NGINX_LOG_DIR + "coffee-cv-front.sessions"), 30)
        for p in self.logs.iterdir():
            with self.subTest(log=p.name):
                self.assertIn(NGINX_LOG_DIR + p.name, rotate, "a log nothing rotates")
        self.assertIn("abcd1234", (self.logs / "coffee-cv-front.access").read_text(),
                      "a request line that cannot be joined to the app's log")


def logrotate_days() -> dict[str, int]:
    """Each file the front's logrotate rules name, with the number of daily rotations it is kept."""
    days = {}
    for paths, block in re.findall(r"^([^#\s][^{]*)\{(.*?)^\}", (DEPLOY / "coffee-cv-front.logrotate.template")
                                   .read_text(), flags=re.M | re.S):
        assert re.search(r"^\s*daily$", block, flags=re.M), paths
        for path in paths.split():
            days[path] = int(re.search(r"^\s*rotate (\d+)$", block, flags=re.M)[1])
    return days


if __name__ == "__main__":
    unittest.main()
