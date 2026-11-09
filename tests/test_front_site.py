"""The front's nginx site (ticket OPS-15 P3), served by a real nginx behind a stream map like the proxy's. Plain unittest.

    python -m unittest tests.test_front_site

webapp/deploy/coffee-cv-front.nginx.conf.template is what the proxy serves for the domain (ADR 0031): TLS on a local
port that the proxy's stream map reaches, the page and the agent files from the release, and /analyze through the gate
(D11), the 40 MB cap (ML-19 D23; 10 MB by OPS-16 D8 before) and the check service (P1) to the inference host; its offline 503 kept apart from busy (D12).
Every /analyze answer is readable by a cross-origin caller, and its preflight is answered before the gate (OPS-19,
ADR 0035).
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
from tests.test_front_check_service import BOUNDARY, JPEG, RID, FakeInferenceHost, form, free_port  # noqa: E402

NGINX = shutil.which("nginx") or next((p for p in ("/usr/sbin/nginx",) if Path(p).exists()), None)
STREAM_MODULE = Path("/usr/lib/nginx/modules/ngx_stream_module.so")
DOMAIN = "coffee.test"
VISITOR = "127.0.0.5"   # the visitor's address; the stream hop to the site comes from 127.0.0.1
NGINX_LOG_DIR = "/var/log/nginx/"
STATIC = REPO / "webapp" / "static"
ELSEWHERE = "https://other-site.example"   # the origin of a page on another site (OPS-19)


class Visit(http.client.HTTPSConnection):
    """An HTTPS request as a visitor makes it: from VISITOR, to the proxy's stream port, with the domain as SNI."""

    def __init__(self, port: int, ca: Path):
        super().__init__("127.0.0.1", port, timeout=30, context=ssl.create_default_context(cafile=str(ca)))

    def connect(self):
        raw = socket.create_connection((self.host, self.port), self.timeout, source_address=(VISITOR, 0))
        self.sock = self._context.wrap_socket(raw, server_hostname=DOMAIN)


class SiteAgreesWithTheCheckService(unittest.TestCase):
    """The site's limits, read from the template, are the check service's (OPS-15 D15, D16; OPS-16 D8)."""

    SITE = SITE_TEMPLATE.read_text()
    ANALYZE = re.search(r"location = /analyze \{(.*?)\n    \}", SITE, re.S)[1]

    def test_the_body_cap_is_the_photos_cap_and_the_forms_allowance(self):
        [size] = re.findall(r"^\s*client_max_body_size (\d+)k;", self.ANALYZE, re.M)
        self.assertEqual(int(size) * 1024, check_service.MAX_UPLOAD_BYTES + check_service.FORM_ALLOWANCE)

    def test_nginx_outwaits_the_check_services_longest_wait_for_an_answer(self):
        [wait] = re.findall(r"^\s*proxy_read_timeout (\d+)s;", self.ANALYZE, re.M)
        self.assertGreater(int(wait), check_service.UPLOAD_TIMEOUT_S + check_service.ANSWER_TIMEOUT_S)

    def test_nginxs_own_413_says_what_the_check_services_does(self):
        body = re.search(r"location @too_large \{.*?return 413 '([^']*)';", self.SITE, re.S)[1]
        self.assertEqual(json.loads(body), check_service.TOO_LARGE)


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
        FakeInferenceHost.extra_headers = {}
        FakeInferenceHost.echo_id = True

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

    def assert_cross_origin(self, resp):
        """A cross-origin caller can read this answer and its X-Request-Id (OPS-19 D1, D4), each header sent once."""
        self.assertEqual(resp.headers.get_all("Access-Control-Allow-Origin"), ["*"])
        self.assertEqual(resp.headers.get_all("Access-Control-Expose-Headers"), ["X-Request-Id"])

    def cross_origin(self, body: bytes) -> tuple[int, dict]:
        """A photo sent by a script on another site's page: its answer, which that script can read."""
        resp = self.visit("POST", "/analyze", body, {"Origin": ELSEWHERE})
        self.assert_cross_origin(resp)
        return resp.status, json.loads(resp.read())

    def test_a_cross_origin_caller_reads_the_streamed_answer(self):   # OPS-19 D1, D4
        resp = self.visit("POST", "/analyze", form("beans.jpg", JPEG), {"Origin": ELSEWHERE})
        self.assertEqual((resp.status, resp.read()), (200, b'{"part": "preview"}\n{"part": "classification"}\n'))
        self.assertRegex(resp.getheader("X-Request-Id"), r"^[0-9a-f]{32}$")
        self.assert_cross_origin(resp)

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

    def test_the_example_photos_are_served_revalidated_each_visit(self):
        """OPS-16: the files the page's example row fetches, each with its type; no-cache, as the page (D15)."""
        for path, kind in (("/examples/brazil-grinder.jxl", "image/jxl"), ("/examples/brazil-grinder.webp", "image/webp"),
                           ("/examples/examples.json", "application/json")):
            with self.subTest(path=path):
                resp = self.visit("GET", path)
                self.assertEqual((resp.status, resp.read()), (200, (STATIC / path[1:]).read_bytes()))
                self.assertEqual(resp.getheader("Content-Type"), kind)
                self.assertEqual(resp.getheader("Cache-Control"), "no-cache")
        resp = self.visit("GET", "/examples/no-such-example.jxl")
        resp.read()
        self.assertEqual(resp.status, 404)

    def test_only_analyze_and_the_agent_files_are_open_to_other_origins(self):   # OPS-19 scope, ADR 0035
        for path, cors in (("/", None), ("/examples/brazil-grinder.webp", None), ("/no-such-file", None),
                           ("/llms.txt", ["*"]), ("/openapi.json", ["*"]), ("/.well-known/ard.json", ["*"])):
            with self.subTest(path=path):
                resp = self.visit("GET", path, headers={"Origin": ELSEWHERE})
                resp.read()
                self.assertEqual(resp.headers.get_all("Access-Control-Allow-Origin"), cors)
                self.assertIsNone(resp.getheader("Access-Control-Expose-Headers"))

    def test_a_photo_reaches_the_inference_host_and_streams_back(self):
        forged = {"X-Real-IP": "203.0.113.9", "X-Forwarded-For": "203.0.113.9", "Forwarded": "for=203.0.113.9"}
        resp = self.visit("POST", "/analyze", form("beans.jpg", JPEG), forged)
        first = resp.readline()
        t_first = time.monotonic()
        second = resp.readline()
        waited = time.monotonic() - t_first
        self.assertEqual(resp.status, 200)
        self.assertEqual((first, second), (b'{"part": "preview"}\n', b'{"part": "classification"}\n'))
        self.assertGreater(waited, 0.5, "the preview was held back until the classification: the answer was buffered")
        [seen] = FakeInferenceHost.seen
        self.assertEqual(resp.getheader("X-Request-Id"), seen["headers"]["X-Request-Id"])
        self.assertEqual(set(forged) & set(seen["headers"]), set(), "a visitor's own IP header went through (D20)")

    def test_a_photo_over_the_cap_is_refused_at_the_front(self):   # ML-19 D23
        photo = JPEG + b"\x00" * (check_service.MAX_UPLOAD_BYTES - len(JPEG))
        resp = self.visit("POST", "/analyze", form("beans.jpg", photo))   # exactly the cap: its form is larger
        resp.read()
        self.assertEqual(resp.status, 200)
        FakeInferenceHost.seen.clear()
        self.assertEqual(self.answer(form("beans.jpg", photo + b"\x00")),   # the check service's 413
                         (413, {"error": "this photo is 40.1 MB; the limit is 40 MB"}))
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

    def preflight(self):
        resp = self.visit("OPTIONS", "/analyze", headers={
            "Origin": ELSEWHERE, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-trace"})
        resp.read()
        return resp

    def test_a_preflight_is_answered_by_the_front(self):   # OPS-19 D2
        resp = self.preflight()
        self.assertEqual(resp.status, 204)
        self.assertEqual({k: resp.headers.get_all(k) for k in ("Access-Control-Allow-Origin", "Access-Control-Allow-Methods",
                                                                "Access-Control-Allow-Headers", "Access-Control-Max-Age")},
                         {"Access-Control-Allow-Origin": ["*"], "Access-Control-Allow-Methods": ["POST"],
                          "Access-Control-Allow-Headers": ["*"], "Access-Control-Max-Age": ["7200"]})
        self.assertEqual(FakeInferenceHost.seen, [])

    def test_a_preflight_takes_no_gate_and_never_reaches_the_check_service(self):   # OPS-19 D2, R3
        first = self.visit("POST", "/analyze", form("beans.jpg", JPEG))   # held: its answer takes a second
        self.assertEqual(self.preflight().status, 204)
        first.read()
        self.stop_check_service()
        try:
            self.assertEqual(self.preflight().status, 204)
        finally:
            self.start_check_service()

    def test_a_cors_header_from_upstream_is_never_sent_twice(self):   # OPS-19 D1: a browser rejects a doubled one
        FakeInferenceHost.extra_headers = {
            "Access-Control-Allow-Origin": "https://upstream.example", "Access-Control-Expose-Headers": "X-Upstream",
            "Access-Control-Allow-Credentials": "true", "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "X-Upstream", "Access-Control-Max-Age": "1"}
        resp = self.visit("POST", "/analyze", form("beans.jpg", JPEG), {"Origin": ELSEWHERE})
        resp.read()
        self.assertEqual(resp.status, 200)
        self.assert_cross_origin(resp)
        self.assertEqual({k for k in resp.headers if k.lower().startswith("access-control-")},
                         {"Access-Control-Allow-Origin", "Access-Control-Expose-Headers"})

    def test_a_cross_origin_caller_reads_the_check_services_refusals(self):   # OPS-19 D1
        status, payload = self.cross_origin(form("beans.jpg", b"some text"))
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        photo = JPEG + b"\x00" * (check_service.MAX_UPLOAD_BYTES + 1 - len(JPEG))
        self.assertEqual(self.cross_origin(form("beans.jpg", photo)),
                         (413, {"error": "this photo is 40.1 MB; the limit is 40 MB"}))
        FakeInferenceHost.reply = 502
        self.assertEqual(self.cross_origin(form("beans.jpg", JPEG)), (503, check_service.OFFLINE))
        FakeInferenceHost.reply = 503   # the inference host's own busy answer, relayed as it is
        self.assertEqual(self.cross_origin(form("beans.jpg", JPEG)), (503, {"error": "inference host says 503"}))

    def test_a_cross_origin_caller_reads_the_fronts_own_refusals(self):   # OPS-19 D1
        photo = JPEG + b"\x00" * (check_service.MAX_UPLOAD_BYTES + 2**20)
        self.assertEqual(self.cross_origin(form("beans.jpg", photo)), (413, check_service.TOO_LARGE))
        first = self.visit("POST", "/analyze", form("beans.jpg", JPEG))   # held: its answer takes a second
        self.assertEqual(self.cross_origin(form("beans.jpg", JPEG)), (503, check_service.BUSY))
        first.read()
        self.stop_check_service()
        try:
            self.assertEqual(self.cross_origin(form("beans.jpg", JPEG)), (503, check_service.OFFLINE))
        finally:
            self.start_check_service()

    def access_line(self, rid: str) -> list[str]:
        """The access log's line for request `rid`, split at spaces, once nginx has written it."""
        for _ in range(50):   # nginx writes the line once the answer is out
            lines = [ln.split(" ") for ln in (self.logs / "coffee-cv-front.access").read_text().splitlines()]
            found = [ln for ln in lines if ln[1] == rid]
            if found:
                [line] = found
                return line
            time.sleep(0.05)
        self.fail(f"no access line for {rid}")

    def test_one_request_id_on_every_hop(self):   # OPS-3 D8
        resp = self.visit("POST", "/analyze", form("beans.jpg", JPEG), {"X-Request-Id": RID})
        resp.read()
        rid = resp.getheader("X-Request-Id")
        self.assertRegex(rid, r"^[0-9a-f]{32}$")
        self.assertNotEqual(rid, RID, "a visitor's own X-Request-Id went on")
        [seen] = FakeInferenceHost.seen
        self.assertEqual(seen["headers"]["X-Request-Id"], rid)
        self.assertEqual(self.access_line(rid)[1:3], [rid, rid])   # $request_id, then the one the answer carried

    def test_the_fronts_own_refusals_carry_the_request_id(self):   # OPS-3 P1: the page needs it for its timeline
        def refusals():
            yield "over the body cap", JPEG + b"\x00" * (check_service.MAX_UPLOAD_BYTES + 2**20), 413
            first = self.visit("POST", "/analyze", form("beans.jpg", JPEG))   # held: its answer takes a second
            yield "busy", JPEG, 503
            first.read()
            self.stop_check_service()
            try:
                yield "offline", JPEG, 503
            finally:
                self.start_check_service()
        for name, photo, status in refusals():
            with self.subTest(name):
                resp = self.visit("POST", "/analyze", form("beans.jpg", photo), {"Origin": ELSEWHERE})
                resp.read()
                rid = resp.getheader("X-Request-Id")
                self.assertEqual(resp.status, status)
                self.assertEqual(resp.headers.get_all("X-Request-Id"), [rid])
                self.assertEqual(self.access_line(rid)[1:3], [rid, "-"])
                self.assert_cross_origin(resp)

    def test_the_access_log_names_the_calling_pages_origin(self):   # OPS-19 D3
        access = self.logs / "coffee-cv-front.access"
        for headers, last in (({"Origin": ELSEWHERE}, f'"{ELSEWHERE}"'), ({}, '"-"')):
            with self.subTest(origin=last):
                before = len(access.read_text().splitlines())
                resp = self.visit("POST", "/analyze", form("beans.jpg", b"some text"), headers)
                resp.read()
                for _ in range(50):   # nginx writes the line once the answer is out
                    lines = access.read_text().splitlines()
                    if len(lines) > before:
                        break
                    time.sleep(0.05)
                self.assertIn('"POST /analyze HTTP/1.1"', lines[-1])
                self.assertEqual(lines[-1].rsplit(" ", 1)[1], last)

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
        self.assertIn(FakeInferenceHost.seen[-1]["headers"]["X-Request-Id"],
                      (self.logs / "coffee-cv-front.access").read_text(),
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
