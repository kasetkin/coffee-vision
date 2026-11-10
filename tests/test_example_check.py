"""--verify's check of the page's example photos (ticket OPS-16 D14, D17), against a fake front. Plain unittest.

    python -m unittest tests.test_example_check

webapp/deploy/example_check.py fetches each example in the release commit's examples.json through the domain, holds
its bytes to the commit's file, posts them to /analyze and holds the answer to the example's country, or a refusal
for the negative. Here the front is a stdlib server on loopback: its examples are the commit's files unless a test
changes one, and its /analyze answers by the posted file's name.
"""
from __future__ import annotations

import json
import re
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

LIST = example_check.LIST


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
    probes: list[str | None] = []   # each request's X-Coffee-CV-Probe

    def do_GET(self):  # noqa: N802
        FakeFront.probes.append(self.headers.get("X-Coffee-CV-Probe"))
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
        FakeFront.probes.append(self.headers.get("X-Coffee-CV-Probe"))
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
        FakeFront.probes = []
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

    def test_every_request_is_marked_as_the_deploys(self):   # OPS-3 D14: the reading script leaves them out
        FakeFront.busy_first = 1
        self.run_check()
        self.assertEqual(len(FakeFront.probes), 2 * len(self.examples) + 1)
        self.assertEqual(set(FakeFront.probes), {"verify"})

    def test_a_wrong_country_fails(self):
        FakeFront.answers["brazil-grinder"] = predicted("Colombia")
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("FAIL  example brazil-grinder: got 'Colombia', want 'Brazil'", lines)

    def test_a_bean_photo_refused_fails(self):
        FakeFront.answers["indonesia-roasted"] = REFUSED
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("FAIL  example indonesia-roasted: got 'refused_ood', want 'Indonesia'", lines)

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

    def test_offline_is_not_retried(self):
        with mock.patch.object(FakeFront, "do_POST", lambda h: (
                h.rfile.read(int(h.headers["Content-Length"])), FakeFront.posted.append("x"), h.send_response(503),
                h.send_header("Content-Length", "39"), h.end_headers(),
                h.wfile.write(b'{"error": "the classifier is offline"}\n'))):
            code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertEqual(len(FakeFront.posted), len(self.examples), "an offline answer was retried")
        self.assertIn("FAIL  example brazil-grinder: got 'HTTP 503', want 'Brazil'", lines)

    def test_a_commit_this_repo_does_not_have_fails_rather_than_skips(self):
        for commit in ("0" * 40, ""):
            with self.subTest(commit=commit):
                lines: list[str] = []
                code = example_check.check(self.url, commit, REPO, lines.append)
                self.assertEqual(code, 1)
                self.assertTrue(lines and lines[0].startswith("FAIL  the release's commit"), lines)
                self.assertEqual(FakeFront.posted, [])

    def test_a_release_from_before_the_examples_has_nothing_to_check(self):
        code, lines = self.run_check(git("rev-parse", f"{self.commit}^").decode().strip())
        self.assertEqual((code, FakeFront.posted), (0, []))
        self.assertEqual(lines, [f"skipped: the release's commit has no {LIST}"])


def probe(answer: str, status: int = 200) -> str:
    """A release_probe.py PROBE line of an /analyze answer, as `smoke --more` writes one per photo."""
    parts = [json.loads(line) for line in answer.splitlines()]
    return "PROBE " + json.dumps({"build": {}, "analyze": {"status": status, "content_type": "application/x-ndjson",
                                                           "parts": parts}}) + "\n"


class LocalAnswers(unittest.TestCase):
    """OPS-22 D7: before a deploy, the staged app's answers here on each example as the front will send it (re-encoded
    with its settings), held to examples.json, so a changed answer stops the deploy before either host is stopped."""

    @classmethod
    def setUpClass(cls):
        cls.commit = git("log", "-1", "--format=%H", "--", LIST).decode().strip()
        if not cls.commit:
            raise unittest.SkipTest(f"{LIST} is not committed yet")
        cls.examples = json.loads(git("show", f"{cls.commit}:{LIST}"))["examples"]

    def setUp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for e in self.examples:
            self.write(e["name"], predicted(e["country"]) if e["country"] else REFUSED)

    def write(self, name: str, answer: str, status: int = 200) -> None:
        (self.dir / f"{name}.jxl.probe").write_text(probe(answer, status))

    def run_check(self) -> tuple[int, list[str]]:
        lines: list[str] = []
        return example_check.check_local(self.commit, REPO, self.dir, lines.append), lines

    def test_every_example_right_passes_and_says_so(self):
        code, lines = self.run_check()
        self.assertEqual(code, 0, lines)
        self.assertEqual(len(lines), len(self.examples))
        for e in self.examples:
            self.assertIn(f"ok    example {e['name']} on the front's photo, here: {example_check.expected(e)}", lines)

    def test_a_changed_answer_fails(self):
        bean = next(e for e in self.examples if e["country"])
        negative = next(e for e in self.examples if e["country"] is None)
        self.write(bean["name"], predicted("Peru" if bean["country"] != "Peru" else "Kenya"))
        self.write(negative["name"], predicted("Brazil"))
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertEqual(sum(line.startswith("FAIL") for line in lines), 2, lines)
        self.assertIn(f"FAIL  example {negative['name']} on the front's photo, here: got 'Brazil', want 'refused_ood'",
                      lines)

    def test_an_answer_that_is_not_200_or_missing_fails(self):
        first, second = self.examples[0]["name"], self.examples[1]["name"]
        self.write(first, part(part="error", message="boom"), status=400)
        (self.dir / f"{second}.jxl.probe").unlink()
        code, lines = self.run_check()
        self.assertEqual(code, 1)
        self.assertTrue(any(line.startswith(f"FAIL  example {first} on the front's photo, here: got 'HTTP 400'")
                            for line in lines), lines)
        self.assertTrue(any(line.startswith(f"FAIL  example {second} on the front's photo, here: got 'no answer'")
                            for line in lines), lines)

    def test_a_commit_without_the_list_has_nothing_to_check(self):
        before = git("log", "-1", "--format=%H", "--diff-filter=A", "--", LIST).decode().strip()
        lines: list[str] = []
        self.assertEqual(example_check.check_local(f"{before}^", REPO, self.dir, lines.append), 0)
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

    def test_offline_is_the_check_services_word_for_it(self):
        sys.path.insert(0, str(REPO / "webapp" / "front"))
        import check_service
        self.assertEqual(example_check.OFFLINE.decode(), check_service.OFFLINE["error"])
        self.assertNotIn(example_check.OFFLINE.decode(), check_service.BUSY["error"])

    def test_it_waits_as_long_as_the_fronts_nginx(self):
        site = (REPO / "webapp/deploy/coffee-cv-front.nginx.conf.template").read_text()
        [wait] = re.findall(r"proxy_read_timeout (\d+)s;", site.split("location = /analyze", 1)[1])
        self.assertGreaterEqual(example_check.TIMEOUT_S, int(wait))

    def test_what_each_example_should_get(self):
        self.assertEqual(example_check.expected({"country": "Brazil"}), "Brazil")
        self.assertEqual(example_check.expected({"country": None}), "refused_ood")


if __name__ == "__main__":
    unittest.main()
