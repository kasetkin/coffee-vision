"""The files that describe the web service to agents match the app and the release. Plain unittest, stdlib only.

    python -m unittest tests.test_agent_files

webapp/static/llms.txt, openapi.json and .well-known/ard.json describe POST /analyze for LLM agents (webapp/README.md,
"For agents"). Nothing in the app reads them, so nothing else fails when they drift: this checks their form fields,
routes, parts and verdicts against webapp/app.py's source, every URL in them against what the release serves, and
that every static file ships in the release.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "webapp" / "static"
APP_SRC = (REPO / "webapp" / "app.py").read_text()
sys.path.insert(0, str(REPO / "webapp" / "deploy"))
from release_manifest import APP  # noqa: E402

sys.path.insert(0, str(REPO / "webapp" / "front"))
import check_service  # noqa: E402

LLMS = (STATIC / "llms.txt").read_text()
OPENAPI = json.loads((STATIC / "openapi.json").read_text())
ARD = json.loads((STATIC / ".well-known" / "ard.json").read_text())
INDEX = (STATIC / "index.html").read_text()
# Paths nginx answers that are not files under webapp/static (the front's site, ticket OPS-15 P3).
NON_FILE_PATHS = {"/", "/analyze"}


def app_routes() -> dict[str, set[str]]:
    routes: dict[str, set[str]] = {}
    for method, path in re.findall(r'@app\.(get|post|put|patch|delete)\("([^"]+)"', APP_SRC):
        routes.setdefault(path, set()).add(method)
    return routes


def site_host() -> str:
    """The publisher every ARD identifier names: the site's host name."""
    return ARD["entries"][0]["identifier"].split(":")[2]


class AgentFilesMatchApp(unittest.TestCase):
    def test_openapi_routes_are_the_apps(self):
        documented = {path: {m for m in ops if m in {"get", "post", "put", "patch", "delete"}}
                      for path, ops in OPENAPI["paths"].items()}
        self.assertEqual(documented, app_routes())

    def test_form_fields_are_the_ones_analyze_reads(self):
        read = set(re.findall(r'request\.(?:files|form)\.get\("([^"]+)"\)', APP_SRC))
        schema = OPENAPI["paths"]["/analyze"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(set(schema["properties"]), read)
        self.assertEqual(schema["required"], re.findall(r'request\.files\.get\("([^"]+)"\)', APP_SRC))
        for field in read:
            self.assertIn(f"`{field}`", LLMS, f"llms.txt does not describe the form field {field}")

    def test_parts_and_verdicts_are_the_apps(self):
        schemas = OPENAPI["components"]["schemas"]
        parts = {s["properties"]["part"]["const"] for s in schemas.values() if "part" in s.get("properties", {})}
        self.assertEqual(parts, set(re.findall(r'"part": "([a-z]+)"', APP_SRC)))
        verdicts = set(re.findall(r'"verdict": "([a-z_]+)"', APP_SRC)) | set(re.findall(r'\["verdict"\] = "([a-z_]+)"',
                                                                                        APP_SRC))
        self.assertEqual(set(schemas["ClassificationPart"]["properties"]["verdict"]["enum"]), verdicts)
        for verdict in verdicts:
            self.assertIn(f"`{verdict}`", LLMS, f"llms.txt does not describe the verdict {verdict}")


class AgentFilesAndPageStateTheFrontsAnswers(unittest.TestCase):
    """The front refuses before the inference host sees a photo (ticket OPS-15): its cap and its offline answer are
    what the page and the agent files must state, not the app's 20 MB, which only the inference host keeps (D15)."""

    LIMIT_MB = check_service.MAX_UPLOAD_BYTES // 2**20

    def test_every_size_limit_stated_is_the_fronts(self):
        for name, text in (("llms.txt", LLMS), ("openapi.json", json.dumps(OPENAPI))):
            with self.subTest(file=name):
                limits = set(re.findall(r"(\d+) MB", text))
                self.assertEqual(limits, {str(self.LIMIT_MB)}, f"{name} states a limit other than the front's")

    def test_the_pages_limit_is_the_fronts(self):
        m = re.search(r"const MAX_UPLOAD_BYTES = (\d+) \* 1024 \* 1024;", INDEX)
        self.assertIsNotNone(m, "index.html has no MAX_UPLOAD_BYTES constant")
        self.assertEqual(int(m[1]) * 2**20, check_service.MAX_UPLOAD_BYTES)

    def test_the_page_and_the_agent_files_know_offline_from_busy(self):
        offline = check_service.OFFLINE["error"]
        self.assertIn(f"const OFFLINE_ERROR = '{offline}';", INDEX)
        self.assertIn(f'{{"error": "{offline}"}}', LLMS)
        responses = OPENAPI["paths"]["/analyze"]["post"]["responses"]
        self.assertIn(offline, json.dumps(responses["503"]))
        self.assertIn("busy", json.dumps(responses["503"]))
        self.assertEqual(responses["413"]["content"]["application/json"]["schema"],
                         {"$ref": "#/components/schemas/Error"})


class AgentFilesAreWellFormed(unittest.TestCase):
    def test_llms_txt_is_ascii_markdown_with_an_h1_and_links(self):
        # nginx serves .txt as text/plain with no charset, so a non-ASCII byte could be misread.
        self.assertTrue(LLMS.isascii())
        self.assertTrue(LLMS.startswith("# "))
        self.assertGreater(len(re.findall(r"\[[^\]]+\]\(https://[^)]+\)", LLMS)), 0)

    def test_ard_entries_have_the_specs_required_fields(self):
        # ARD v0.91 (github.com/ards-project/ard-spec, spec/schemas/ard-entry.schema.json): identifier, displayName and
        # type are required, and exactly one of url or data; 2-5 representativeQueries are recommended.
        ids = []
        for entry in ARD["entries"]:
            with self.subTest(entry=entry.get("identifier")):
                self.assertRegex(entry["identifier"], rf"^urn:air:{re.escape(site_host())}(:[a-zA-Z0-9._-]+)+$")
                self.assertTrue(entry["displayName"] and entry["type"])
                self.assertEqual(("url" in entry) + ("data" in entry), 1)
                self.assertTrue(2 <= len(entry.get("representativeQueries", [])) <= 5)
                ids.append(entry["identifier"])
        self.assertEqual(len(ids), len(set(ids)))


class AgentFilesPointAtTheRelease(unittest.TestCase):
    def test_every_site_url_is_one_the_release_serves(self):
        urls = re.findall(r"https://[^\s\"')>`]+", LLMS + json.dumps(OPENAPI) + json.dumps(ARD))
        paths = [urlsplit(u).path or "/" for u in urls if urlsplit(u).hostname == site_host()]
        paths += re.findall(r'<link rel="(?:service-desc|ard|alternate)"[^>]* href="(/[^"]*)"', INDEX)
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(path in NON_FILE_PATHS or (STATIC / path.lstrip("/")).is_file(),
                                f"{path} is neither a route nor a file under webapp/static")

    def test_every_static_file_ships_in_the_release(self):
        files = {p.relative_to(REPO).as_posix() for p in STATIC.rglob("*") if p.is_file()}
        self.assertEqual(files - set(APP), set())


if __name__ == "__main__":
    unittest.main()
