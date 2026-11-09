"""--verify's check that its timeline is written (ticket OPS-3 D14, plan P5), against a fake front. Plain unittest.

    python -m unittest tests.test_timeline_check

webapp/deploy/timeline_check.py posts verify's timeline through the domain and finds it in the check service's log.
Here the front is the real check service on loopback, with no nginx and no inference host: what it accepts is the
schema's, and what it writes is its log file. The deploy script's own wiring is checked as text.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "deploy"))
sys.path.insert(0, str(REPO / "webapp" / "front"))
import check_service  # noqa: E402
import timeline_check  # noqa: E402

from coffeecv import timeline_schema  # noqa: E402

RID = uuid.uuid4().hex


class TimelineCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log_dir = Path(tempfile.mkdtemp(prefix="timeline-check-"))
        cls.addClassCleanup(shutil.rmtree, cls.log_dir, ignore_errors=True)
        cls.front = check_service.make_server(("127.0.0.1", 0), "http://127.0.0.1:9", cls.log_dir)
        threading.Thread(target=cls.front.serve_forever, daemon=True).start()
        cls.addClassCleanup(cls.front.server_close)
        cls.addClassCleanup(cls.front.shutdown)
        cls.url = f"http://127.0.0.1:{cls.front.server_address[1]}"

    def run_post(self, url: str | None = None) -> tuple[str | None, list[str]]:
        said = []
        return timeline_check.post(url or self.url, RID, said.append), said

    def log(self) -> list[str]:
        return (self.log_dir / check_service.LOG_FILE).read_text().splitlines()

    def test_the_timeline_is_one_the_schema_accepts_marked_as_the_deploys(self):
        sent = timeline_schema.check(timeline_check.timeline(RID), set())
        self.assertEqual(sent["probe"], timeline_schema.PROBE)
        self.assertEqual(timeline_check.PROBE_HEADER, check_service.PROBE_HEADER)

    def test_posted_then_found(self):
        written, said = self.run_post()
        self.assertEqual(said, ["ok    public POST /timeline: 204"])
        self.assertRegex(written, r"^[0-9a-f]{32}$")
        self.assertTrue(timeline_check.found(self.log(), written, RID))
        line = next(json.loads(ln) for ln in self.log() if written in ln)
        self.assertEqual(line["probe"], "verify", "the POST was not marked as the deploy's")

    def test_not_found_for_another_photo_or_another_timeline(self):
        written, _ = self.run_post()
        self.assertFalse(timeline_check.found(self.log(), written, uuid.uuid4().hex))
        self.assertFalse(timeline_check.found(self.log(), uuid.uuid4().hex, RID))
        self.assertFalse(timeline_check.found(["not json", "[]", json.dumps({"kind": "timeline"})], written, RID))

    def test_a_refused_timeline_fails(self):
        with mock.patch.object(timeline_check, "timeline", lambda rid: {"v": 1}):
            written, said = self.run_post()
        self.assertIsNone(written)
        self.assertEqual(said, ["FAIL  public POST /timeline: got '400', want '204'"])

    def test_a_front_without_timelines_or_unreachable_fails(self):
        with mock.patch.object(timeline_schema, "parse", side_effect=timeline_schema.Invalid("x")):
            self.assertIsNone(self.run_post()[0])
        written, said = self.run_post(f"http://127.0.0.1:{check_service_free_port()}")
        self.assertIsNone(written)
        self.assertTrue(said[0].startswith("FAIL  public POST /timeline: got 'no answer"), said)

    def test_the_command_line(self):
        out = subprocess.run([sys.executable, str(REPO / "webapp/deploy/timeline_check.py"), "post", "--url", self.url,
                              "--request-id", RID], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        written = re.search(r"^TIMELINE_ID=([0-9a-f]{32})$", out.stdout, re.M)[1]
        for rid, code in ((RID, 0), (uuid.uuid4().hex, 1)):
            with self.subTest(code=code):
                found = subprocess.run([sys.executable, str(REPO / "webapp/deploy/timeline_check.py"), "found",
                                        "--timeline-id", written, "--request-id", rid],
                                       input="\n".join(self.log()), capture_output=True, text=True)
                self.assertEqual(found.returncode, code, found.stdout)


def check_service_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VerifyMarksItsRequests(unittest.TestCase):
    """Every request --verify sends through the domain carries the probe marker (OPS-3 D14), so the reading script
    leaves it out; and verify checks its timeline."""

    SCRIPT = (REPO / "scripts" / "deploy_webapp.sh").read_text()
    VERIFY = SCRIPT[SCRIPT.index("verify_cors() {"):SCRIPT.index("on_verify_failure() {")]

    def test_every_curl_through_the_domain_carries_the_marker(self):
        commands = self.VERIFY.replace("\\\n", " ").splitlines()   # a command continued over lines is one
        public = [ln for ln in commands if "curl " in ln and "$PUBLIC_URL" in ln]
        self.assertGreaterEqual(len(public), 4)
        for line in public:
            with self.subTest(line=line.strip()[:60]):
                self.assertIn('"${PROBE[@]}"', line)
        self.assertIn('PROBE=(-H "X-Coffee-CV-Probe: verify")', self.SCRIPT)

    def test_verify_posts_its_timeline_and_reads_the_fronts_log(self):
        self.assertIn('timeline_check.py" post', self.VERIFY)
        self.assertIn('timeline_check.py" found', self.VERIFY)
        self.assertIn("$FRONT_LOG_DIR/requests.jsonl", self.VERIFY)
        self.assertIn("FRONT_LOG_DIR=${FRONT_LOG_DIR:-" + check_service.DEFAULT_LOG_DIR + "}", self.SCRIPT)


if __name__ == "__main__":
    unittest.main()
