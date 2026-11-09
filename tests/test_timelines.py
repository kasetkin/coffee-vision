"""The reading script (ticket OPS-3 P6, D13, D17's layers 4 and 5, D20). Plain unittest.

    python -m unittest tests.test_timelines

coffeecv/timelines.py reads the front's access log, the check service's log, yappy's access log and the app's log,
joins them by request ID and counts. Here on lines written in the hosts' own formats, forged, duplicate and malicious
ones included, and on tests/fixtures/ops3_rig/, the front's two logs as P4's browser check left them (an example, an
own photo and its re-run, a photo over the cap, a busy retry, a photo abandoned by leaving the page, and another
page's photo with no timeline).
"""
from __future__ import annotations

import gzip
import http.client
import json
import shutil
import tempfile
import threading
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from coffeecv import timelines as tl

REPO = Path(__file__).resolve().parents[1]
RIG = REPO / "tests" / "fixtures" / "ops3_rig"
DOMAIN = "coffee.test"
PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"
DAY = "2026-11-10"


def rid() -> str:
    return uuid.uuid4().hex


def access(request_id: str, request: str, status: int = 200, origin: str = "-", browser: str = PHONE,
           referrer: str = "-", probe: str = "-", day: str = DAY, upstream: str | None = None) -> str:
    """A front access line, as its log_format writes one (webapp/deploy/coffee-cv-front.nginx.conf.template)."""
    return (f'{day}T12:00:00+00:00 {request_id} {upstream or request_id} {status} "{request}" 0.500s 1234 "{origin}" '
            f'"{browser}" "{referrer}" "{probe}"')


def service(request_id: str, day: str = DAY, **fields) -> str:
    """A check service request line for a photo answered, canonicalized."""
    line = {"ts": f"{day}T12:00:00.000+00:00", "kind": "request", "request_id": request_id, "method": "POST",
            "endpoint": "/analyze", "status": 200, "outcome": "relayed", "filename": "IMG_0001.HEIC", "ext": ".heic",
            "format": "heif", "bytes": 3 * 2**20, "canonical": "canonicalized", "read_ms": 20, "check_ms": 1,
            "canonicalize_ms": 2000, "crossing_ms": 9000, "to_headers_ms": 2500, "to_preview_ms": 2600,
            "to_end_ms": 8000, "relayed_bytes": 300000, "total_ms": 19000}
    line.update(fields)
    return json.dumps({k: v for k, v in line.items() if v is not None})


def timeline_line(timeline: dict, day: str = DAY, probe: bool = False) -> str:
    return json.dumps({"ts": f"{day}T12:00:30.000+00:00", "kind": "timeline", "request_id": rid(),
                       **({"probe": "verify"} if probe else {}), "timeline": timeline})


def timeline(request_id: str | None, control: str = "choose_photo", outcome: str = "answered", **fields) -> dict:
    t = {"v": 1, "request_id": request_id, "control": control, "outcome": outcome, "pick_ms": 4000,
         "size_check_ms": 0, "to_preview_ms": 15000, "preview_drawn_ms": 30, "to_answer_ms": 21000,
         "answer_shown_ms": 5, "total_ms": 40000,
         "attempts": [{"request_id": request_id, "status": 200, "connect_ms": 120, "upload_ms": 3000,
                       "headers_ms": 14800}] if request_id else []}
    t.update(fields)
    return {k: v for k, v in t.items() if v is not None or k == "request_id"}


def app(request_id: str) -> str:
    return json.dumps({"ts": f"{DAY}T12:00:20.000+00:00", "level": "INFO", "msg": "request", "request_id": request_id,
                       "endpoint": "/analyze", "status": 200, "latency_ms": 5400, "preview_ms": 2400,
                       "preview_build_ms": 60, "decode_ms": 300, "crop_detect_ms": 1900, "inference_ms": 2900,
                       "verdict": "predicted"})


def yappy(request_id: str) -> str:
    return f'{DAY}T12:00:20+00:00 {request_id} 200 "POST /analyze HTTP/1.1" 9.100s 300000'


def logs(**sources: list[str]) -> dict[str, list[str]]:
    return {name: sources.get(name, []) for name in tl.SOURCES}


class OnePhotoJoined(unittest.TestCase):
    """A photo's timeline from the page, the front's two lines, yappy's nginx and the app, joined by its ID (D8)."""

    def setUp(self):
        self.id = rid()
        self.report = tl.build(logs(front_access=[access(self.id, "POST /analyze HTTP/1.1", origin=f"https://{DOMAIN}")],
                                    service=[service(self.id), timeline_line(timeline(self.id))],
                                    app=[app(self.id)], inference_access=[yappy(self.id)]), DOMAIN)

    def step(self, key: str, column: str = "all") -> int | None:
        row = next(r for r in self.report.steps if r["step"] == key)
        return row[column] and row[column]["median"]

    def test_each_step_from_the_line_that_measured_it(self):
        self.assertEqual(len(self.report.photos), 1)
        for key, want in (("1", 4000), ("4", 120), ("5", 3000), ("7", 1), ("8", 2000), ("9", 9000), ("10", 300),
                          ("11", 1900), ("12", 60), ("15", 2900), ("17", 40000), ("17f", 19000), ("17n", 9100),
                          ("17a", 5400)):
            with self.subTest(step=key):
                self.assertEqual(self.step(key), want)

    def test_the_network_legs_are_one_sides_time_less_the_others(self):
        self.assertEqual(self.step("13a"), 2600 - 2400)                        # the front's wait, less the app's
        self.assertEqual(self.step("13b"), 15000 - (20 + 1 + 2000 + 9000 + 2600))   # the page's, less the front's
        self.assertEqual(self.step("16"), 21000 - 15000)
        self.assertEqual(self.step("16f"), 8000 - 2600)                        # the front's own, preview to end
        self.assertEqual(self.step("16s"), 5)

    def test_split_by_control_and_by_canonicalized(self):
        self.assertEqual(self.step("8", "choose_photo"), 2000)
        self.assertIsNone(self.step("8", "camera"))
        self.assertEqual(self.step("8", "canonicalized"), 2000)
        self.assertIsNone(self.step("8", "as_sent"))

    def test_the_days_counts(self):
        day = self.report.days[DAY]
        self.assertEqual((day["by_control"], day["by_format"], day["by_size"], day["timelines"]),
                         ({"choose_photo": 1}, {"heif": 1}, {"1-5 MB": 1}, 1))

    def test_a_photo_view_for_the_page(self):
        view = tl.as_json(self.report)["photos"][0]
        self.assertEqual((view["request_id"], view["filename"], view["control"], view["verdict"]),
                         (self.id, "IMG_0001.HEIC", "choose_photo", "predicted"))
        self.assertEqual([s["step"] for s in view["steps"]], [key for key, _, _ in tl.STEPS])


class WhatCounts(unittest.TestCase):
    """D17's layer 4: a timeline counts when its ID is a real /analyze request, the first per ID; D14, D20."""

    def test_forged_duplicate_and_idless_timelines_are_flagged_not_counted(self):
        real, page_load, invented = rid(), rid(), rid()
        report = tl.build(logs(
            front_access=[access(real, "POST /analyze HTTP/1.1"), access(page_load, "GET / HTTP/1.1")],
            service=[service(real),
                     timeline_line(timeline(real)),
                     timeline_line(timeline(real, control="camera")),            # a second for the same ID
                     timeline_line(timeline(invented)),                          # an ID no host knows
                     timeline_line(timeline(page_load)),                         # a real ID, but a page load's
                     timeline_line(timeline(None, outcome="refused_on_page")),   # no ID
                     json.dumps({"ts": f"{DAY}T12:00:00.000+00:00", "kind": "timeline", "request_id": rid(),
                                 "timeline": {"v": 1, "control": "<script>"}})]), DOMAIN)
        [photo] = report.photos
        self.assertEqual(photo.timeline["control"], "choose_photo", "the first timeline for the ID counts")
        f = report.flagged
        self.assertEqual((len(f["suspicious"]), len(f["unverified"]), len(f["unverifiable"]), f["malformed"]),
                         (1, 2, 1, 1))
        self.assertEqual(report.failures["page"], {"refused_on_page": 1})

    def test_the_deploys_own_requests_are_left_out(self):   # D14
        probe, real = rid(), rid()
        report = tl.build(logs(
            front_access=[access(probe, "POST /analyze HTTP/1.1", probe="verify"), access(rid(), "GET / HTTP/1.1",
                                                                                        probe="verify"),
                          access(real, "POST /analyze HTTP/1.1")],
            service=[service(probe, probe="verify"), timeline_line(timeline(probe), probe=True), service(real)]),
            DOMAIN)
        self.assertEqual([p.request_id for p in report.photos], [real])
        self.assertEqual(report.days[DAY]["page_loads"], 0)
        self.assertGreaterEqual(report.probes, 2)

    def test_a_busy_retrys_earlier_attempts_are_not_photos(self):
        first, last = rid(), rid()
        t = timeline(last, attempts=[{"request_id": first, "status": 503, "wait_ms": 3000},
                                     {"request_id": last, "status": 200, "upload_ms": 2000}])
        report = tl.build(logs(front_access=[access(first, "POST /analyze HTTP/1.1", 503),
                                             access(last, "POST /analyze HTTP/1.1")],
                               service=[service(first, status=503, outcome="busy"), service(last),
                                        timeline_line(t)]), DOMAIN)
        self.assertEqual([p.request_id for p in report.photos], [last])
        self.assertEqual(next(r for r in report.steps if r["step"] == "6")["all"]["median"], 3000)

    def test_a_photo_the_fronts_nginx_refused_is_counted_with_its_timeline(self):   # busy at the gate, offline
        first, last, offline = rid(), rid(), rid()
        busy = timeline(last, outcome="busy", attempts=[{"request_id": first, "status": 503, "wait_ms": 3000},
                                                        {"request_id": last, "status": 503}])
        report = tl.build(logs(front_access=[access(first, "POST /analyze HTTP/1.1", 503, upstream="-"),
                                             access(last, "POST /analyze HTTP/1.1", 503, upstream="-"),
                                             access(offline, "POST /analyze HTTP/1.1", 503, upstream="-")],
                               service=[timeline_line(busy), timeline_line(timeline(offline, outcome="offline"))]),
                          DOMAIN)
        self.assertEqual(sorted(p.outcome for p in report.photos), ["busy", "offline"])
        self.assertEqual(report.failures["page"], {"busy": 1, "offline": 1})
        self.assertEqual(report.days[DAY]["by_control"], {"choose_photo": 2})
        self.assertEqual(sorted(p.service.get("outcome") for p in report.photos), ["refused at the front's nginx"] * 2)

    def test_a_photo_no_page_reported_is_counted_without_a_timeline(self):   # R4: an agent, or a lost beacon
        report = tl.build(logs(service=[service(rid())]), DOMAIN)
        self.assertEqual(report.days[DAY]["by_control"], {"no timeline": 1})
        self.assertEqual(report.days[DAY]["timelines"], 0)

    def test_lines_from_before_the_release_are_skipped(self):   # D20: no browser, no referrer, no marker
        old = f'{DAY}T12:00:00+00:00 {rid()} - 200 "GET / HTTP/1.1" 0.001s 1234 "-"'
        report = tl.build(logs(front_access=[old]), DOMAIN)
        self.assertEqual(report.days, {})

    def test_page_loads_by_people_and_crawlers_and_where_they_came_from(self):
        report = tl.build(logs(front_access=[
            access(rid(), "GET / HTTP/1.1", referrer="https://www.google.com/search?q=coffee+origin"),
            access(rid(), "GET /index.html HTTP/1.1", referrer=f"https://{DOMAIN}/"),
            access(rid(), "GET /?utm=x HTTP/1.1", status=304),
            access(rid(), "GET / HTTP/1.1", browser="Mozilla/5.0 (compatible; Googlebot/2.1)"),
            access(rid(), "GET / HTTP/1.1", browser="-"),
            access(rid(), "GET /llms.txt HTTP/1.1"),
            access(rid(), "GET / HTTP/1.1", status=404)]), DOMAIN)
        day = report.days[DAY]
        self.assertEqual((day["page_loads"], day["crawler_loads"], day["referrers"]),
                         (3, 2, {"www.google.com": 1}))

    def test_other_sites_calls_by_origin(self):
        report = tl.build(logs(front_access=[
            access(rid(), "POST /analyze HTTP/1.1", origin="https://other.example"),
            access(rid(), "POST /analyze HTTP/1.1", origin="https://other.example"),
            access(rid(), "POST /analyze HTTP/1.1", origin=f"https://{DOMAIN}"),
            access(rid(), "POST /analyze HTTP/1.1")]), DOMAIN)
        self.assertEqual(report.origins, {"https://other.example": 2})

    def test_a_period(self):
        report = tl.build(logs(service=[service(rid(), day="2026-11-10"), service(rid(), day="2026-11-12")]), DOMAIN,
                          since="2026-11-11", until="2026-11-30")
        self.assertEqual(list(report.days), ["2026-11-12"])

    def test_dropped_timelines_are_summed(self):
        report = tl.build(logs(service=[service(rid(), timelines_dropped=40), service(rid(), timelines_dropped=2)]),
                          DOMAIN)
        self.assertEqual(report.dropped, 42)


class WhatAVisitorSentIsText(unittest.TestCase):
    """D17's layer 5: printed without control characters, shown as text."""

    EVIL = "<img src=x onerror=alert(1)>\x1b[31m‮\r\nX"

    def setUp(self):
        photo = rid()
        self.report = tl.build(logs(
            front_access=[access(rid(), "GET / HTTP/1.1", referrer=f"https://{self.EVIL}.example/"),
                          access(photo, "POST /analyze HTTP/1.1", origin=f"https://{self.EVIL}")],
            service=[service(photo, filename=self.EVIL, format=self.EVIL)]), DOMAIN)

    def test_the_summary_has_no_control_characters(self):
        text = tl.summary(self.report)
        self.assertFalse(any(ord(ch) < 32 and ch != "\n" or 0x7f <= ord(ch) < 0xa0 or ch == "‮" for ch in text))
        self.assertIn("<img src=x onerror=alert(1)>", text, "shown as it was, as text")
        self.assertIn("(unreadable)", text, "a referrer no URL parser reads")

    def test_the_page_builds_with_text_only(self):
        self.assertNotIn("innerHTML", tl.PAGE)
        self.assertNotIn("insertAdjacentHTML", tl.PAGE)
        self.assertNotIn("document.write", tl.PAGE)
        view = tl.as_json(self.report)["photos"][0]
        self.assertNotIn("\x1b", view["filename"])

    def test_served_on_loopback_with_its_data_apart_from_its_page(self):
        server = tl.make_server(self.report, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        self.addCleanup(conn.close)
        conn.request("GET", "/")
        resp = conn.getresponse()
        page = resp.read().decode()
        self.assertEqual((resp.status, page), (200, tl.PAGE))
        self.assertIn("default-src 'self'", resp.getheader("Content-Security-Policy"))
        conn.request("GET", "/report.json")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(len(data["photos"]), 1)
        conn.request("GET", "/anything-else")
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 404)


class Reading(unittest.TestCase):
    """Over SSH as the deploy users, rotated files included, into memory; never the session log (OPS-15 D20)."""

    def test_the_session_log_is_never_a_source(self):
        for name, (_, directory, filename) in tl.SOURCES.items():
            self.assertNotIn("sessions", directory + filename, name)
        with mock.patch.dict(tl.SOURCES, {"sessions": ("front", "/var/log/nginx", "coffee-cv-front.sessions")}), \
                self.assertRaises(SystemExit):
            tl.read_dir(Path("/nonexistent"))

    def test_over_ssh_each_host_by_its_alias(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return mock.Mock(returncode=0, stdout="line\n", stderr="")
        got = tl.read_ssh({"front": "webproxy-deploy", "inference": "webappremote"}, run=run)
        self.assertEqual({c[-2] for c in calls}, {"webproxy-deploy", "webappremote"})
        self.assertEqual(got, {name: ["line"] for name in tl.SOURCES})
        for cmd in calls:
            self.assertNotIn("sessions", cmd[-1])
            self.assertIn("zcat -f", cmd[-1])

    def test_a_directory_with_rotated_and_compressed_files_and_a_session_log_beside_them(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        a, b = rid(), rid()
        (root / "requests.jsonl").write_text(service(a) + "\n")
        (root / "requests.jsonl.1").write_text(service(b) + "\n")
        (root / "requests.jsonl.2.gz").write_bytes(gzip.compress((service(rid()) + "\n").encode()))
        (root / "requests.jsonl.bak").write_text(service(rid()) + "\n")   # not a rotated file: not read
        (root / "coffee-cv-front.sessions").write_text("203.0.113.9 the visitor's IP\n")
        got = tl.read_dir(root)
        self.assertEqual(len(got["service"]), 3)
        self.assertNotIn("203.0.113.9", json.dumps(got))


class TheRecordedRig(unittest.TestCase):
    """tests/fixtures/ops3_rig/: P4's browser check, as the front logged it."""

    def setUp(self):
        self.report = tl.build(tl.read_dir(RIG), DOMAIN)

    def test_its_photos_and_timelines(self):
        by = {p.request_id: p for p in self.report.photos}
        self.assertEqual(len(by), 6)
        counted = sorted((p.control, p.outcome, bool(p.timeline.get("rerun_of"))) for p in by.values() if p.timeline)
        self.assertEqual(counted, [("choose_photo", "answered", False), ("choose_photo", "answered", False),
                                   ("choose_photo", "answered", True), ("example", "answered", False)])
        self.assertEqual(sum(p.timeline is None for p in by.values()), 2)   # another page's photo, the abandoned one
        self.assertEqual(sorted(e["outcome"] for e in self.report.flagged["unverifiable"]),
                         ["abandoned", "refused_on_page"])
        self.assertEqual(self.report.days["2026-11-09"]["by_example"], {"brazil-grinder": 1})
        self.assertEqual(self.report.days["2026-11-09"]["by_format"], {"jpeg": 5, "jxl": 1})

    def test_its_busy_retry(self):
        busy = next(p for p in self.report.photos if p.timeline and len(p.timeline["attempts"]) > 1)
        self.assertEqual([a["status"] for a in busy.timeline["attempts"]], [503, 503, 200])
        self.assertEqual(next(r for r in self.report.steps if r["step"] == "6")["all"]["median"], 9000)

    def test_the_command_line_from_a_directory(self):
        out = StringIO()
        with redirect_stdout(out):
            tl.main(["summary", "--from-dir", str(RIG), "--domain", DOMAIN])
        # Headless Chromium's page loads are a crawler's.
        self.assertIn("2026-11-09  loads 0 (+3 crawlers)  photos 6  timelines 4", out.getvalue())
        self.assertIn("unverifiable", out.getvalue())


if __name__ == "__main__":
    unittest.main()
