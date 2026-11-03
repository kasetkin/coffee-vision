"""--verify's check of the page's example photos (ticket OPS-16 D14, D17), against a fake front. Plain unittest.

    python -m unittest tests.test_example_check

webapp/deploy/example_check.py fetches each example in the release commit's examples.json through the domain, holds
its bytes to the commit's file, posts them to /analyze and holds the answer to the example's country, or a refusal
for the negative. Here the front is a stdlib server on loopback: its examples are the commit's files unless a test
changes one, and its /analyze answers by the posted file's name.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "deploy"))
import example_check  # noqa: E402

LIST = "webapp/static/examples/examples.json"


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def part(**kw) -> str:
    return json.dumps(kw) + "\n"


def predicted(country: str) -> str:
    return part(part="preview") + part(part="classification", verdict="predicted",
                                       ranked=[{"id": country, "score": 0.9}, {"id": "Peru", "score": 0.1}])


REFUSED = part(part="preview") + part(part="classification", verdict="refused_ood", message="Unrecognized")


class FakeFront(BaseHTTPRequestHandler):
    commit = ""
    served: dict[str, bytes] = {}   # path -> bytes, instead of the commit's file
    answers: dict[str, str] = {}    # example name -> its /analyze body
    busy_first = 0                  # this many busy 503s before answering
    posted: list[str] = []

    def do_GET(self):  # noqa: N802
        path = self.path.lstrip("/")
        try:
            data = FakeFront.served.get(path) or git("show", f"{FakeFront.commit}:webapp/static/{path}")
        except subprocess.CalledProcessError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        name = body.split(b'filename="', 1)[1].split(b".jxl", 1)[0].decode()
        FakeFront.posted.append(name)
        if FakeFront.busy_first:
            FakeFront.busy_first -= 1
            data, status = b'{"error": "the classifier is busy with another photo"}', 503
        else:
            data, status = FakeFront.answers[name].encode(), 200
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class ExampleCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.front = ThreadingHTTPServer(("127.0.0.1", 0), FakeFront)
        threading.Thread(target=cls.front.serve_forever, daemon=True).start()
        cls.addClassCleanup(cls.front.server_close)
        cls.addClassCleanup(cls.front.shutdown)
        cls.url = f"http://127.0.0.1:{cls.front.server_address[1]}"
        # The commit that added the list, and its parent, which has none: a release from before OPS-16.
        cls.commit = git("log", "-1", "--format=%H", "--diff-filter=A", "--", LIST).decode().strip()
        if not cls.commit:
            raise unittest.SkipTest(f"{LIST} is not committed yet")
        cls.examples = json.loads(git("show", f"{cls.commit}:{LIST}"))["examples"]

    def setUp(self):
        FakeFront.commit = self.commit
        FakeFront.served = {}
        FakeFront.answers = {e["name"]: predicted(e["country"]) if e["country"] else REFUSED for e in self.examples}
        FakeFront.busy_first = 0
        FakeFront.posted = []
        self.enterContext(mock.patch.object(example_check.time, "sleep"))

    def run_check(self, commit: str | None = None) -> tuple[int, list[str]]:
        lines: list[str] = []
        code = example_check.check(self.url, commit or self.commit, REPO, lines.append)
        return code, lines

    def test_every_example_right_passes_and_says_so(self):
        code, lines = self.run_check()
        self.assertEqual(code, 0, lines)
        self.assertEqual(FakeFront.posted, [e["name"] for e in self.examples])
        # Three each: served, the release's bytes, its answer.
        self.assertEqual(sum(line.startswith("ok    ") for line in lines), 3 * len(self.examples), lines)
        self.assertIn("ok    example brazil-grinder: Brazil", lines)
        self.assertIn("ok    example cappuccino: refused_ood", lines)

    def test_a_wrong_country_fails(self):
        FakeFront.answers["brazil-grinder"] = predicted("Colombia")
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("FAIL  example brazil-grinder: got 'Colombia', want 'Brazil'", lines)

    def test_a_bean_photo_refused_fails(self):
        FakeFront.answers["vietnam-robusta"] = REFUSED
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("FAIL  example vietnam-robusta: got 'refused_ood', want 'Vietnam'", lines)

    def test_the_negative_not_refused_fails(self):
        FakeFront.answers["cappuccino"] = predicted("Brazil")
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("FAIL  example cappuccino: got 'Brazil', want 'refused_ood'", lines)

    def test_a_file_the_front_serves_other_than_the_commits_fails_and_is_not_posted(self):
        FakeFront.served["examples/indonesia-roasted.jxl"] = b"some other bytes"
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertTrue(any(line.startswith("FAIL  example indonesia-roasted served = the release's: got '")
                            for line in lines), lines)
        self.assertNotIn("indonesia-roasted", FakeFront.posted)

    def test_examples_the_front_does_not_serve_fail_and_are_not_posted(self):
        with mock.patch.object(FakeFront, "do_GET", lambda h: h.send_error(404)):   # a front without the route
            code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("FAIL  example ethiopia-green served: got 'HTTP 404', want 'HTTP 200'", lines)
        self.assertEqual(FakeFront.posted, [])

    def test_one_busy_answer_is_retried(self):
        FakeFront.busy_first = 1
        code, lines = self.run_check()
        self.assertEqual(code, 0, lines)
        self.assertEqual(FakeFront.posted[:2], ["brazil-grinder", "brazil-grinder"])

    def test_busy_twice_fails(self):
        FakeFront.busy_first = 2
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("FAIL  example brazil-grinder: got 'HTTP 503', want 'Brazil'", lines)

    def test_a_release_from_before_the_examples_has_nothing_to_check(self):
        code, lines = self.run_check(git("rev-parse", f"{self.commit}^").decode().strip())
        self.assertEqual((code, FakeFront.posted), (0, []))
        self.assertEqual(lines, [f"skipped: the release's commit has no {LIST}"])


class Outcome(unittest.TestCase):
    """What an /analyze answer comes to: a refusal's verdict, else the top-1 country."""

    def test_each_kind_of_answer(self):
        self.assertEqual(example_check.outcome(predicted("Kenya")), "Kenya")
        warned = part(part="preview") + part(part="classification", verdict="warned", warning="w",
                                             ranked=[{"id": "Peru", "score": 0.5}])
        self.assertEqual(example_check.outcome(warned), "Peru")
        self.assertEqual(example_check.outcome(REFUSED), "refused_ood")
        scale = part(part="preview") + part(part="classification", verdict="refused_scale", message="m")
        self.assertEqual(example_check.outcome(scale), "refused_scale")
        self.assertEqual(example_check.outcome(part(part="preview") + part(part="error", message="boom")),
                         "error: boom")
        self.assertEqual(example_check.outcome(part(part="preview")), "no classification")
        self.assertEqual(example_check.outcome("not json"), "no classification")

    def test_what_each_example_should_get(self):
        self.assertEqual(example_check.expected({"country": "Brazil"}), "Brazil")
        self.assertEqual(example_check.expected({"country": None}), "refused_ood")


if __name__ == "__main__":
    unittest.main()
