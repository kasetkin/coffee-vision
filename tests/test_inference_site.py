"""The inference host's nginx site (ticket OPS-15 P2), run in a real nginx in front of a fake app. Plain unittest.

    python -m unittest tests.test_inference_site

webapp/deploy/coffee-cv.nginx.conf.template is what `yappy` serves once the front holds TLS and the page (ADR 0031):
plain HTTP on its tailnet address only (D5), /analyze only, one request at a time (D11), the whole upload buffered
before the app (D10), the app's 20 MB cap (D15), the answer streamed (ADR 0023), and the client IP the front
forwards logged only in a file rotated at 30 days (D18). The site is rendered as scripts/deploy_webapp.sh renders it,
its /var/log/nginx paths moved to a scratch directory, and served by the nginx on PATH as an unprivileged user;
without nginx the tests skip.
"""
from __future__ import annotations

import http.client
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "webapp" / "deploy"
SITE_TEMPLATE = DEPLOY / "coffee-cv.nginx.conf.template"
sys.path.insert(0, str(DEPLOY))
from render_template import render  # noqa: E402
from tests._tiers import real_data  # noqa: E402

NGINX = shutil.which("nginx") or next((p for p in ("/usr/sbin/nginx",) if Path(p).exists()), None)
ADDR = "127.0.0.1"   # stands in for the tailnet address; 127.0.0.2 is "anywhere else" on this machine
NGINX_LOG_DIR = "/var/log/nginx/"


def free_port() -> int:
    with socket.socket() as s:
        s.bind((ADDR, 0))
        return s.getsockname()[1]


class FakeApp(BaseHTTPRequestHandler):
    """The app's side of /analyze: records what reached it, answers in two lines a second apart."""

    seen: list[dict] = []
    hold = threading.Event()   # cleared: the app holds its answer until it is set

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        FakeApp.seen.append({"path": self.path, "headers": dict(self.headers), "bytes": len(body)})
        FakeApp.hold.wait(10)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Request-Id", f"req{len(FakeApp.seen)}")
        self.end_headers()
        self.wfile.write(b'{"part": "preview"}\n')
        self.wfile.flush()
        time.sleep(1.0)
        self.wfile.write(b'{"part": "classification"}\n')

    def log_message(self, *args):
        pass


@real_data   # about 5 s: the fake app's answers take a second each
class InferenceSite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if NGINX is None:
            raise unittest.SkipTest("no nginx on this machine")
        cls.tmp = Path(tempfile.mkdtemp(prefix="inference-site-"))
        cls.logs = cls.tmp / "log"
        cls.logs.mkdir()
        cls.port, app_port = free_port(), free_port()
        cls.app = ThreadingHTTPServer((ADDR, app_port), FakeApp)
        threading.Thread(target=cls.app.serve_forever, daemon=True).start()
        FakeApp.hold.set()

        site = render(SITE_TEMPLATE.read_text(),
                      {"INFERENCE_ADDR": ADDR, "INFERENCE_PORT": str(cls.port), "PORT": str(app_port)})
        assert NGINX_LOG_DIR in site, f"the site no longer logs under {NGINX_LOG_DIR}"
        (cls.tmp / "site.conf").write_text(site.replace(NGINX_LOG_DIR, f"{cls.logs}/"))
        temp_paths = "".join(f"    {kind}_temp_path {cls.tmp}/{kind};\n"
                             for kind in ("client_body", "proxy", "fastcgi", "uwsgi", "scgi"))
        (cls.tmp / "nginx.conf").write_text(
            f"pid {cls.tmp}/nginx.pid;\nerror_log {cls.tmp}/error.log;\nevents {{}}\n"
            f"http {{\n{temp_paths}    access_log off;\n    include {cls.tmp}/site.conf;\n}}\n")
        cmd = [NGINX, "-p", str(cls.tmp), "-c", str(cls.tmp / "nginx.conf")]
        check = subprocess.run([*cmd, "-t"], capture_output=True, text=True)
        if check.returncode:
            cls.tearDownClass()
            raise AssertionError(f"nginx -t rejected the rendered site:\n{check.stderr}")
        cls.nginx = subprocess.Popen([*cmd, "-g", "daemon off;"], stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection((ADDR, cls.port), timeout=0.1).close()
                break
            except OSError:
                time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "nginx", None):
            cls.nginx.terminate()
            cls.nginx.wait(10)
        cls.app.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        FakeApp.seen.clear()
        FakeApp.hold.set()

    def request(self, method: str, path: str, body: bytes = b"", headers: dict | None = None):
        conn = http.client.HTTPConnection(ADDR, self.port, timeout=20)
        conn.request(method, path, body=body or None, headers=headers or {})
        return conn, conn.getresponse()

    def fetch(self, method: str, path: str, body: bytes = b"", headers: dict | None = None):
        """A whole request: its response, already read, on a closed connection."""
        conn, resp = self.request(method, path, body, headers)
        resp.read()
        conn.close()
        return resp

    def test_serves_no_page_nor_any_path_but_analyze(self):
        for path in ("/", "/index.html", "/llms.txt", "/robots.txt", "/analyze/x"):
            with self.subTest(path=path):
                resp = self.fetch("GET", path)
                self.assertEqual(resp.status, 404)
        self.assertEqual(FakeApp.seen, [])

    def test_analyze_reaches_the_app_with_the_fronts_ip_and_streams_back(self):
        conn, resp = self.request("POST", "/analyze", b"x" * 1000, {"X-Real-IP": "203.0.113.7"})
        first = resp.readline()
        t_first = time.monotonic()
        second = resp.readline()
        waited = time.monotonic() - t_first
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertEqual((first, second), (b'{"part": "preview"}\n', b'{"part": "classification"}\n'))
        self.assertGreater(waited, 0.5, "the preview was held back until the classification: the answer was buffered")
        self.assertEqual(resp.getheader("X-Accel-Buffering"), "no", "the front's nginx would buffer the answer")
        self.assertEqual(resp.getheader("X-Request-Id"), "req1")
        [seen] = FakeApp.seen
        self.assertEqual((seen["path"], seen["bytes"]), ("/analyze", 1000))
        self.assertEqual(seen["headers"]["X-Real-IP"], "203.0.113.7")

    def test_a_second_request_while_one_is_in_flight_is_refused_busy(self):
        FakeApp.hold.clear()

        first = threading.Thread(target=self.fetch, args=("POST", "/analyze", b"x"))
        first.start()
        for _ in range(100):
            if FakeApp.seen:
                break
            time.sleep(0.02)
        resp = self.fetch("POST", "/analyze", b"y")
        FakeApp.hold.set()
        first.join(10)
        self.assertEqual(resp.status, 503)
        self.assertEqual(len(FakeApp.seen), 1)

    def test_the_cap_is_the_apps_and_a_larger_upload_never_reaches_it(self):
        cap = int(re.search(r"MAX_UPLOAD_BYTES = (\d+) \* 1024 \* 1024", (REPO / "webapp" / "app.py").read_text())[1])
        resp = self.fetch("POST", "/analyze", b"x" * (cap * 1024 * 1024 + 1))
        self.assertEqual(resp.status, 413)
        resp = self.fetch("POST", "/analyze", b"x" * (cap * 1024 * 1024 - 4096))
        self.assertEqual(resp.status, 200)
        self.assertEqual([s["bytes"] for s in FakeApp.seen], [cap * 1024 * 1024 - 4096])

    def test_the_app_sees_the_upload_only_once_all_of_it_has_crossed(self):
        body = b"x" * 200_000
        with socket.create_connection((ADDR, self.port), timeout=10) as s:
            s.sendall(b"POST /analyze HTTP/1.1\r\nHost: inference\r\nContent-Length: %d\r\n\r\n" % len(body))
            s.sendall(body[:100_000])
            time.sleep(0.5)
            self.assertEqual(FakeApp.seen, [], "the app was handed the request before its body had arrived")
            s.sendall(body[100_000:])
            self.assertIn(b"200", s.recv(64))
        self.assertEqual([s["bytes"] for s in FakeApp.seen], [len(body)])

    def test_listens_on_the_tailnet_address_only(self):
        for family, addr in ((socket.AF_INET, "127.0.0.2"), (socket.AF_INET6, "::1")):
            with self.subTest(addr=addr), socket.socket(family) as s:
                s.settimeout(2)
                with self.assertRaises(ConnectionRefusedError):
                    s.connect((addr, self.port))

    def test_the_client_ip_is_only_in_files_rotated_at_30_days(self):
        ip = "198.51.100.23"
        resp = self.fetch("POST", "/analyze", b"x", {"X-Real-IP": ip})
        rotate = logrotate_days()
        with_ip = {p.name for p in self.logs.iterdir() if ip in p.read_text()}
        self.assertTrue(with_ip, "the front's X-Real-IP is in no log")
        for name in with_ip:
            self.assertEqual(rotate.get(NGINX_LOG_DIR + name), 30, f"{name} holds an IP; logrotate keeps it longer")
        req_id = resp.getheader("X-Request-Id")
        for p in self.logs.iterdir():
            with self.subTest(log=p.name):
                self.assertIn(NGINX_LOG_DIR + p.name, rotate, "a log nothing rotates")
                self.assertIn(req_id, p.read_text(), "a line that cannot be joined to the app's log")


class DeployRendersTheSite(unittest.TestCase):
    def test_the_deploy_passes_the_variables_the_site_uses(self):
        script = (REPO / "scripts" / "deploy_webapp.sh").read_text()
        call = re.search(r"render \"\$SHA\" webapp/deploy/coffee-cv\.nginx\.conf\.template(.*?)(\)|>)", script, re.M | re.S)
        self.assertIsNotNone(call, "scripts/deploy_webapp.sh no longer renders the site")
        passed = set(re.findall(r"\b([A-Z_]+)=\"", call[1]))
        used = set(re.findall(r"\$\{?(INFERENCE_[A-Z_]*|PORT|DOMAIN|APP_ROOT|PUBLIC_PORT)\b", SITE_TEMPLATE.read_text()))
        self.assertEqual(passed, {"INFERENCE_ADDR", "INFERENCE_PORT", "PORT"})
        self.assertEqual(used, passed)


def logrotate_days() -> dict[str, int]:
    """Each file the rendered logrotate rules name, with the number of daily rotations it is kept."""
    text = render((DEPLOY / "coffee-cv.logrotate.template").read_text(), {"SERVICE_USER": "coffee-cv"})
    days = {}
    for paths, block in re.findall(r"^([^#\s][^{]*)\{(.*?)^\}", text, flags=re.M | re.S):
        assert re.search(r"^\s*daily$", block, flags=re.M), paths
        for path in paths.split():
            days[path] = int(re.search(r"^\s*rotate (\d+)$", block, flags=re.M)[1])
    return days


if __name__ == "__main__":
    unittest.main()
