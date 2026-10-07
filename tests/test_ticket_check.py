"""coffeecv.ticket_check: the HTML issue tracker's mechanical rules are found broken in a throwaway docs/ tree,
and a tree that keeps them (plus the legacy tickets, and TOML headers quoted in prose) passes. Plain unittest.

    python -m unittest discover -s tests -p 'test_ticket_check.py'
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from coffeecv import ticket_check as tc

LABELS = """| Label in mattpocock/skills | Label in our tracker | Meaning |
| --- | --- | --- |
| `bug` | `bug` | Broken |
| `enhancement` | `enhancement` | New |
| `needs-triage` | `needs-triage` | Evaluate |
| `ready-for-agent` | `ready-for-agent` | Ready |
"""
HEAD = '<!DOCTYPE html>\n<html>\n<head>\n<title>T</title>\n<link rel="stylesheet" href="tickets.css">\n</head>\n'


def field(k: str, v: str) -> str:
    return f'<div class="field"><span class="k">{k}</span><span class="v">{v}</span></div>\n'


def ticket(tid: str, status: str = "Open", triage: str = "enhancement, needs-triage", body: str = "") -> str:
    return (HEAD + f'<body><main>\n<div class="eyebrow">Ticket {tid} · coffee-vision · area</div>\n'
            + field("Status", f'<span class="pill open">{status}</span>')
            + f'<div class="field"><span class="k">Triage</span><span class="v mono">{triage}</span></div>\n'
            + field("Blocked by", "None") + body + "</main></body>\n</html>\n")


def plan(tid: str, phases: list[tuple[str, str | None, bool]], status: str = "Open") -> str:
    """phases: (title, Status or None for no field, has a Blocked by field)."""
    s = (HEAD + f'<body><main>\n<div class="eyebrow">Plan · {tid} · coffee-vision · implementation</div>\n'
         + field("Status", f'<span class="pill open">{status}</span>') + "<h2>1. Phase order and why</h2>\n")
    for i, (title, st, blocked) in enumerate(phases, 2):
        s += f"<h2>{i}. P{i - 1} &mdash; {title}</h2>\n"
        if st:
            s += field("Status", f'<span class="pill open">{st}</span>')
        if blocked:
            s += field("Blocked by", "None")
    return s + "</main></body>\n</html>\n"


class TrackerTree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.docs = self.root / "docs"
        (self.docs / "agents").mkdir(parents=True)
        (self.docs / "adr").mkdir()
        (self.docs / "agents" / "triage-labels.md").write_text(LABELS)
        (self.docs / "tickets.css").write_text("body{}\n")
        self.write("ticket_alpha.html", ticket("ML-7", body='<p>Decided: <a href="adr/0001-x.md">ADR 0001</a></p>\n'))
        self.write("plan_alpha.html", plan("ML-7", [("Build", "Done", True), ("Ship", "Open", True)]))
        self.write("adr/0001-x.md", "# X\n\nSource: [ML-7](../ticket_alpha.html) D1.\n")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, text: str):
        (self.docs / name).write_text(text)

    def messages(self) -> list[str]:
        return [f"{f.path}: {f.message}" for f in tc.check(self.root)]

    def assertFound(self, *needles: str):
        msgs = self.messages()
        for n in needles:
            self.assertTrue(any(n in m for m in msgs), f"{n!r} not in {msgs}")

    def test_clean_tree_passes(self):
        self.assertEqual(self.messages(), [])

    def test_placeholder_and_comment_found_toml_header_passes(self):
        self.write("ticket_alpha.html", ticket("ML-7", body='<p><a href="adr/0001-x.md">a</a> [[commit]] and '
                                                           "<code>[[tool.uv.index]]</code></p>\n<!-- Tick -->\n"))
        msgs = self.messages()
        self.assertEqual(sorted(m for m in msgs if "ticket_alpha" in m),
                         ["docs/ticket_alpha.html: HTML comment left from the template",
                          "docs/ticket_alpha.html: placeholder left: [[commit]]"])

    def test_template_may_keep_placeholders_and_comments(self):
        self.write("template_ticket.html", ticket("[[ML-n]]", status="[[Open]]", body="<!-- fill in -->\n"))
        self.assertEqual(self.messages(), [])

    def test_unclosed_and_mismatched_tags_found(self):
        self.write("ticket_beta.html", ticket("ML-8", body="<div><p>open</div>\n"))
        self.write("ticket_gamma.html", ticket("ML-9", body="<section><p>open</p>\n"))
        self.write("ticket_delta.html", ticket("ML-10", body="<p>open\n"))
        # One finding per broken tag: the tags around it still nest, so nothing cascades from it.
        self.assertEqual(sorted(m for m in self.messages() if "closes" in m or "never" in m),
                         ["docs/ticket_beta.html: </div> closes <p> from line 12",
                          "docs/ticket_delta.html: </main> closes <p> from line 12",
                          "docs/ticket_gamma.html: </main> closes <section> from line 12"])

    def test_inline_style_and_missing_stylesheet_found(self):
        self.write("ticket_beta.html", ticket("ML-8").replace('<link rel="stylesheet" href="tickets.css">',
                                                               "<style>p{}</style>"))
        self.assertFound("ticket_beta.html: inline <style>", "ticket_beta.html: does not link tickets.css")

    def test_missing_link_targets_found(self):
        self.write("ticket_beta.html", ticket("ML-8", body='<a href="plan_nope.html#p2">p</a> '
                                                           '<a href="https://example.org/x">ok</a> <a href="#d1">ok</a>\n'))
        self.write("adr/0002-y.md", "# Y\n\nSee [the code](../../coffeecv/nope.py).\n")
        msgs = self.messages()
        self.assertEqual(sorted(m for m in msgs if "missing" in m),
                         ["docs/adr/0002-y.md: link to a missing file: ../../coffeecv/nope.py",
                          "docs/ticket_beta.html: link to a missing file: plan_nope.html#p2"])

    def test_duplicate_and_missing_ticket_ids_found(self):
        self.write("ticket_beta.html", ticket("ML-7"))
        self.write("ticket_gamma.html", ticket("ML-9").replace("Ticket ML-9", "ML-9"))
        self.assertFound("ticket_beta.html: ID ML-7 is already docs/ticket_alpha.html's",
                         "ticket_gamma.html: needs one 'Ticket ML-n/OPS-n' eyebrow")

    def test_plan_must_match_its_ticket(self):
        self.write("plan_alpha.html", plan("ML-8", [("Build", "Open", True)]))
        self.write("plan_orphan.html", plan("ML-9", [("Build", "Open", True)]))
        self.assertFound("plan_alpha.html: eyebrow names ['ML-8']",
                         "plan_orphan.html: no ticket for this plan")

    def test_adr_source_ticket_must_link_back(self):
        self.write("ticket_alpha.html", ticket("ML-7"))
        self.assertFound("ticket_alpha.html: docs/adr/0001-x.md names this ticket as its source")

    def test_status_and_triage_vocabulary(self):
        self.write("ticket_beta.html", ticket("ML-8", status="Done", triage="enhancement, bug"))
        self.write("ticket_gamma.html", ticket("ML-9", triage="ready-for-agent"))
        self.assertFound("ticket_beta.html: Status 'Done' is not one of Closed, In progress, Open",
                         "ticket_beta.html: Triage needs one category role",
                         "ticket_gamma.html: Triage needs one category role")

    def test_every_phase_needs_status_and_blocked_by(self):
        self.write("plan_alpha.html", plan("ML-7", [("Build", "Closed", True), ("Ship", None, False)]))
        self.assertFound("plan_alpha.html: Status 'Closed' is not one of Done, In progress, Open",
                         "plan_alpha.html: phase P2 has no Status field",
                         "plan_alpha.html: phase P2 has no Blocked by field")

    def test_legacy_tickets_and_plans_keep_their_old_format(self):
        slug = min(tc.LEGACY)
        self.write(f"ticket_{slug}.html", ticket("ML-1", status="Done · live", triage="").replace(
            '<div class="field"><span class="k">Triage</span><span class="v mono"></span></div>\n', ""))
        self.write(f"plan_{slug}.html", plan("ML-1", [("Build", None, False)], status="Done · closed"))
        self.assertEqual(self.messages(), [])


class RealTemplates(unittest.TestCase):
    def test_a_fresh_copy_of_each_template_fails_until_filled(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(tc.REPO_ROOT / "docs", root / "docs",
                            ignore=shutil.ignore_patterns("ticket_*", "plan_*", "adr"))
            (root / "docs" / "adr").mkdir()
            for kind in ("ticket", "plan"):
                shutil.copy(root / "docs" / f"template_{kind}.html", root / "docs" / f"{kind}_new.html")
            msgs = [f"{f.path}: {f.message}" for f in tc.check(root)]
            for kind in ("ticket", "plan"):
                self.assertTrue(any(f"{kind}_new.html: placeholder left" in m for m in msgs), msgs)
                self.assertTrue(any(f"{kind}_new.html: HTML comment left" in m for m in msgs), msgs)
            self.assertFalse(any("template_" in m for m in msgs), msgs)


if __name__ == "__main__":
    unittest.main()
