"""Private-data check over everything git would publish: tracked files plus untracked ones that are not
ignored (what `git add -A` would add). scripts/check.sh runs it in both tiers.

    python -m coffeecv.leak_check [--patterns FILE]
    python -m coffeecv.leak_check --staged      # only the files staged for commit: .githooks/pre-commit

--staged reads each staged file's working-tree copy, so a file staged in part (`git add -p`) is checked
as it stands on disk. A leak that gets into a commit stays in history (it is not rewritten), which is why
the hook checks at commit time rather than only at push.

Private, by the owner's rule of 2026-10-07: public IP addresses, email addresses, credentials, and GPS or
other identifying photo metadata. Fine to publish: SSH aliases, the DVC remote URL, the VM's user name and
hostname, the live site's domain, Docker-internal and other non-global addresses.

Checks:

1. Text files: a global (public) IPv4 address, an email address, a credential shape (a private key block,
   a token with a known prefix, a URL with a password in it).
2. Images: any tag coffeecv.strip_metadata deny-lists (GPS, maker notes, serial numbers, owner fields,
   thumbnails), read with one exiftool call. Needs exiftool (in the devcontainer).
3. The owner's own patterns: one regex per line in PRIVATE_PATTERNS, matched against text and file paths.
   That file is gitignored, so the patterns themselves are never published. Skipped with a note when it
   is absent (a fresh clone has no private data to protect).

A false positive goes in ALLOWED_IPS / ALLOWED_EMAIL_RE below, with its reason. Credentials print
masked. Exit status is non-zero on any finding.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from coffeecv.config import REPO_ROOT

PRIVATE_PATTERNS = REPO_ROOT / ".private_claude/leak_patterns.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".gif", ".dng",
                    ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2"}

_IPV4 = re.compile(rb"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?!\.?\d)")
# A dotted four-part number right after these is a package version (`==5.0.0.93`, `pkg-1.1.1.2`,
# `version = "1.1.1.2"` in a lock file), not an address.
_VERSION_BEFORE = re.compile(rb'(?:==|[-v]|version\s*=\s*")$')
ALLOWED_IPS = {
    "5.0.0.93",   # opencv-python's version, in prose (docs/ticket_webapp_release_isolation.html)
}

# An address is found from its "@" (a literal, so fast), then grown to both sides.
_EMAIL_LOCAL = re.compile(rb"[\w.+-]+\Z")
_EMAIL_DOMAIN = re.compile(rb"@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
# Addresses that identify no one: commit trailers, GitHub's no-reply and SSH forms, documentation examples.
ALLOWED_EMAIL_RE = re.compile(r"^(?:noreply@anthropic\.com|.*@users\.noreply\.github\.com|git@github\.com"
                              r"|.*@example\.(?:com|org|net))$", re.IGNORECASE)

# Each starts with a literal, which lets the regex engine skip ahead to candidates: ~25 MB of text is read
# per run, and a leading class or \b made this the slowest step by far.
CREDENTIALS = {
    "private key": re.compile(rb"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),
    "GitHub fine-grained token": re.compile(rb"github_pat_[A-Za-z0-9_]{50,}"),
    "Anthropic key": re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "AWS key": re.compile(rb"AKIA[0-9A-Z]{16}(?![0-9A-Z])"),
    "Slack token": re.compile(rb"xox[abprs]-[A-Za-z0-9-]{10,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "password in URL": re.compile(rb"://[^/\s:@'\"]+:[^/\s@'\"]+@"),
}


class Finding(NamedTuple):
    path: str
    line: int          # 0 for a whole-file finding (an image tag, a file path)
    kind: str
    shown: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.kind}: {self.shown}"


def published_files(root: Path = REPO_ROOT, staged: bool = False) -> list[str]:
    """What `git add -A` would publish, or with `staged` only the files added or changed in the index."""
    cmd = (["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"] if staged
           else ["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    out = subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, check=True).stdout
    return sorted({p for p in out.decode().split("\0") if p})


def load_private_patterns(path: Path) -> list[re.Pattern[bytes]] | None:
    """The owner's regexes, or None when the file is absent. Blank lines and `#` comments are skipped."""
    if not path.is_file():
        return None
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [re.compile(ln.encode(), re.IGNORECASE) for ln in lines if ln and not ln.startswith("#")]


def _line(data: bytes, pos: int) -> int:
    return data.count(b"\n", 0, pos) + 1


def _mask(s: str) -> str:
    return s[:6] + "..." if len(s) > 6 else "..."


def text_findings(path: str, data: bytes, private: list[re.Pattern[bytes]] = ()) -> list[Finding]:
    found = []
    for m in _IPV4.finditer(data):
        ip = m.group(1).decode()
        if ip in ALLOWED_IPS or _VERSION_BEFORE.search(data[max(0, m.start() - 12):m.start()]):
            continue
        try:
            is_global = ipaddress.ip_address(ip).is_global
        except ValueError:
            continue
        if is_global:
            found.append(Finding(path, _line(data, m.start()), "public IP address", ip))
    at = data.find(b"@")
    while at != -1:
        local, domain = _EMAIL_LOCAL.search(data, max(0, at - 64), at), _EMAIL_DOMAIN.match(data, at)
        name = local.group() if local else b""
        if local and data[local.start() - 1:local.start()] == b"\\":   # "\n@decorator" in a source string
            name = name[1:]
        if name and domain:
            email = (name + domain.group()).decode(errors="replace")
            if not ALLOWED_EMAIL_RE.match(email):
                found.append(Finding(path, _line(data, at), "email address", email))
        at = data.find(b"@", at + 1)
    for kind, rx in CREDENTIALS.items():
        for m in rx.finditer(data):
            found.append(Finding(path, _line(data, m.start()), kind, _mask(m.group().decode(errors="replace"))))
    for rx in private:
        for m in rx.finditer(data):
            found.append(Finding(path, _line(data, m.start()), "private pattern", f"pattern {rx.pattern.decode()!r}"))
    return found


def image_findings(paths: list[str], root: Path = REPO_ROOT) -> list[Finding]:
    """Deny-listed tags in `paths`, one exiftool call for all of them."""
    if not paths:
        return []
    # Imported here: it pulls in torch (through coffeecv.dataset), which only the image check needs.
    from coffeecv import strip_metadata as sm
    with tempfile.NamedTemporaryFile("w", suffix=".args") as argfile:
        argfile.write("\n".join(str(root / p) for p in paths) + "\n")
        argfile.flush()
        proc = subprocess.run(["exiftool", "-j", "-a", "-G0:1", "-ee", "-u", "-q", "-q", "-@", argfile.name],
                              capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        raise RuntimeError(f"exiftool failed: {proc.stderr.strip()}")
    found = []
    for doc in json.loads(proc.stdout):
        rel = os.path.relpath(doc["SourceFile"], root)
        ext = Path(rel).suffix
        bad = sorted(k for k in doc if k != "SourceFile" and sm.is_denied(k) and not sm.accepted(k, ext))
        if bad:
            found.append(Finding(rel, 0, "private photo metadata", ", ".join(bad)))
    return found


def scan(root: Path = REPO_ROOT, private: list[re.Pattern[bytes]] = (), files: list[str] | None = None
         ) -> list[Finding]:
    found, images = [], []
    for rel in published_files(root) if files is None else files:
        for rx in private:
            if rx.search(rel.encode()):
                found.append(Finding(rel, 0, "private pattern in file path", f"pattern {rx.pattern.decode()!r}"))
        p = root / rel
        if p.is_symlink() or not p.is_file():   # symlinks (.claude/skills), submodules, deleted files
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(rel)
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            continue
        found += text_findings(rel, data, private)
    return found + image_findings(images, root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patterns", type=Path, default=PRIVATE_PATTERNS,
                    help=f"the owner's private regexes (default: {PRIVATE_PATTERNS.relative_to(REPO_ROOT)})")
    ap.add_argument("--staged", action="store_true", help="check only the files staged for commit")
    args = ap.parse_args(argv)

    private = load_private_patterns(args.patterns)
    if private is None:
        print(f"note: {args.patterns} not found, private patterns not checked")
    files = published_files(staged=args.staged)
    found = scan(REPO_ROOT, private or [], files)
    for f in found:
        print(f)
    what = f"{len(files)} {'staged' if args.staged else 'published'} files"
    if found:
        print(f"leak_check: {len(found)} finding(s) in {what}")
        return 1
    print(f"leak_check: {what} clean ({len(private or [])} private patterns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
