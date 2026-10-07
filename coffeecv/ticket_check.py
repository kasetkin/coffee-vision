"""The HTML issue tracker's mechanical rules (docs/agents/issue-tracker.md), checked over docs/ticket_*.html,
docs/plan_*.html, the two templates and docs/adr/*.md. scripts/check.sh runs it in both tiers.

    python -m coffeecv.ticket_check

Checks:

1. Every tracker file is well-formed HTML (each tag closed, in order) and takes its styles from
   docs/tickets.css, with no inline <style>.
2. Tickets and plans: no [[...]] placeholder and no HTML comment left from the template. A dotted
   [[tool.uv.index]] is a TOML table header quoted in prose, not a placeholder.
3. Relative links in tracker files and ADRs point at files that exist.
4. Each ticket names one ID in its eyebrow ("Ticket ML-n" / "Ticket OPS-n"), unique across tickets; each
   plan_<slug>.html names ("Plan · ID") the ID of ticket_<slug>.html.
5. A ticket an ADR's Source line links to links back to that ADR.
6. Tickets and plans written from the templates (every slug not in LEGACY): a ticket's Status is Open,
   In progress or Closed, and its Triage row is one category role and one state role from
   docs/agents/triage-labels.md; a plan's Status, and each phase's (<h2>N. Pk &mdash; ...), is Open,
   In progress or Done, and every phase has its own Status and Blocked by.

Exit status is non-zero on any finding.
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent  # not coffeecv.config's: that one imports torch (1.5 s)
# Written before the templates (fbdac63, 6bded8b): no Triage row, no per-phase fields, free-form Status.
# issue-tracker.md says to leave them as they are. A new ticket never joins this set.
LEGACY = {"country_classes", "retire_cross_rig", "segmentation_mask", "webapp_release_isolation"}

TICKET_STATUS = {"Open", "In progress", "Closed"}
PLAN_STATUS = {"Open", "In progress", "Done"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

_PLACEHOLDER = re.compile(r"\[\[(.*?)\]\]")
_TOML_HEADER = re.compile(r"[\w-]+(?:\.[\w-]+)+")
_HREF = re.compile(r'(?:href|src)="([^"]*)"')
_MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
_TICKET_ID = re.compile(r'class="eyebrow">Ticket ((?:ML|OPS)-\d+) ')
_PLAN_ID = re.compile(r'class="eyebrow">Plan · ((?:ML|OPS)-\d+) ')
_STATUS = re.compile(r'class="k">Status</span><span class="v"><span class="pill [^"]*">([^<]*)<')
_BLOCKED = re.compile(r'class="k">Blocked by</span>')
_TRIAGE = re.compile(r'class="k">Triage</span><span class="v[^"]*">([^<]*)<')
_PHASE = re.compile(r"<h2>\d+\. (P\d+) ")
_LABEL_ROW = re.compile(r"^\|\s*`([\w-]+)`\s*\|\s*`([\w-]+)`\s*\|", re.MULTILINE)


class Finding(NamedTuple):
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


class _Nesting(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.errors: list[tuple[int, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1][0] != tag:
            open_ = f"<{self.stack[-1][0]}> from line {self.stack[-1][1]}" if self.stack else "nothing"
            self.errors.append((self.getpos()[0], f"</{tag}> closes {open_}"))
            if any(t == tag for t, _ in self.stack):
                while self.stack.pop()[0] != tag:
                    pass
        else:
            self.stack.pop()


def nesting_errors(text: str) -> list[tuple[int, str]]:
    p = _Nesting()
    p.feed(text)
    p.close()
    return p.errors + [(line, f"<{tag}> never closed") for tag, line in p.stack]


def triage_roles(labels_md: str) -> tuple[set[str], set[str]]:
    """(category labels, state labels) from triage-labels.md's table: canonical role -> this repo's label."""
    mapping = dict(_LABEL_ROW.findall(labels_md))
    category = {mapping[r] for r in ("bug", "enhancement") if r in mapping}
    return category, set(mapping.values()) - category


def check_file(rel: str, text: str, docs: Path, category: set[str], state: set[str]) -> list[Finding]:
    """Checks 1-3 and 6 on one tracker file (path relative to the repo root, e.g. docs/ticket_x.html)."""
    out = [Finding(rel, line, msg) for line, msg in nesting_errors(text)]
    if "<style" in text:
        out.append(Finding(rel, _line(text, text.index("<style")), "inline <style>: link tickets.css instead"))
    if 'href="tickets.css"' not in text:
        out.append(Finding(rel, 1, "does not link tickets.css"))

    name = Path(rel).name
    for m in _HREF.finditer(text):
        target = m.group(1).split("#")[0].split("?")[0]
        if target and "[[" not in target and not re.match(r"[a-z]+:|//", target) \
                and not (docs / target).exists():
            out.append(Finding(rel, _line(text, m.start()), f"link to a missing file: {m.group(1)}"))
    if name.startswith("template_"):
        return out

    for m in _PLACEHOLDER.finditer(text):
        if not _TOML_HEADER.fullmatch(m.group(1)):
            out.append(Finding(rel, _line(text, m.start()), f"placeholder left: {m.group()}"))
    for m in re.finditer("<!--", text):
        out.append(Finding(rel, _line(text, m.start()), "HTML comment left from the template"))

    kind, slug = name.removesuffix(".html").split("_", 1)
    if slug in LEGACY:
        return out
    allowed = TICKET_STATUS if kind == "ticket" else PLAN_STATUS
    for m in _STATUS.finditer(text):
        if m.group(1).strip() not in allowed:
            out.append(Finding(rel, _line(text, m.start()),
                               f"Status '{m.group(1).strip()}' is not one of {', '.join(sorted(allowed))}"))
    if kind == "ticket":
        m = _TRIAGE.search(text)
        roles = [r.strip() for r in m.group(1).split(",")] if m else []
        if len(roles) != 2 or len(category & set(roles)) != 1 or len(state & set(roles)) != 1:
            out.append(Finding(rel, _line(text, m.start()) if m else 1,
                               f"Triage needs one category role ({', '.join(sorted(category))}) and one state "
                               f"role ({', '.join(sorted(state))}), got: {m.group(1) if m else 'no Triage row'}"))
    else:
        heads = [m.start() for m in re.finditer("<h2>", text)] + [len(text)]
        for a, b in pairwise(heads):
            section = text[a:b]
            phase = _PHASE.match(section)
            if phase:
                for what, rx in (("Status", _STATUS), ("Blocked by", _BLOCKED)):
                    if not rx.search(section):
                        out.append(Finding(rel, _line(text, a), f"phase {phase.group(1)} has no {what} field"))
    return out


def check(root: Path = REPO_ROOT) -> list[Finding]:
    docs = root / "docs"
    category, state = triage_roles((docs / "agents" / "triage-labels.md").read_text())
    tickets = sorted(docs.glob("ticket_*.html"))
    plans = sorted(docs.glob("plan_*.html"))
    files = tickets + plans + sorted(docs.glob("template_*.html"))
    texts = {f: f.read_text() for f in files}

    def rel(f: Path) -> str:
        return str(f.relative_to(root))

    out = [x for f in files for x in check_file(rel(f), texts[f], docs, category, state)]

    ids: dict[str, Path] = {}
    for f in tickets:
        found = _TICKET_ID.findall(texts[f])
        if len(found) != 1:
            out.append(Finding(rel(f), 1, f"needs one 'Ticket ML-n/OPS-n' eyebrow, found {found or 'none'}"))
            continue
        if found[0] in ids:
            out.append(Finding(rel(f), 1, f"ID {found[0]} is already {rel(ids[found[0]])}'s"))
        ids.setdefault(found[0], f)
    for f in plans:
        ticket = docs / f.name.replace("plan_", "ticket_", 1)
        found = _PLAN_ID.findall(texts[f])
        want = _TICKET_ID.findall(texts.get(ticket, ""))
        if not ticket.exists():
            out.append(Finding(rel(f), 1, f"no ticket for this plan: {rel(ticket)} is missing"))
        elif found != want[:1]:
            out.append(Finding(rel(f), 1, f"eyebrow names {found or 'no ID'}, {rel(ticket)} is {want}"))

    for adr in sorted((docs / "adr").glob("*.md")):
        text = adr.read_text()
        for m in _MD_LINK.finditer(text):
            target = m.group(1).split("#")[0]
            if target and not re.match(r"[a-z]+:", target) and not (adr.parent / target).exists():
                out.append(Finding(rel(adr), _line(text, m.start()), f"link to a missing file: {m.group(1)}"))
        for line in re.findall(r"(?m)^Sources?:.*$", text):
            for target in re.findall(r"\]\(\.\./(ticket_[\w-]+\.html)\)", line):
                ticket = docs / target
                if ticket.exists() and f'href="adr/{adr.name}"' not in texts[ticket]:
                    out.append(Finding(rel(ticket), 1, f"{rel(adr)} names this ticket as its source, "
                                                       f"but no D-entry links adr/{adr.name}"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    found = check()
    for f in found:
        print(f)
    if found:
        print(f"ticket_check: {len(found)} finding(s)")
        return 1
    print("ticket_check: tickets, plans, templates and ADRs clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
