"""--verify's check of the page's example photos (ticket OPS-16 D14, D17), through the domain:

    python webapp/deploy/example_check.py --url https://DOMAIN --commit SHA [--repo DIR]

For each example in the release commit's webapp/static/examples/examples.json (D13): its JPEG XL fetched from the
domain, as a visitor's click fetches it, must be the commit's file byte for byte (git is the pin, D17); then it is
posted to /analyze, and the answer must be the example's country as top-1, or refused_ood where the country is null.
A busy 503 is retried once, as the smoke photo is. A commit without the list, a release from before OPS-16, has
nothing to check.

Prints one line per check in verify's own words (`ok    <what>: <got>`, `FAIL  <what>: got '<got>', want '<want>'`)
and exits 1 on any failure, which rolls a full deploy back (D14). Standard library only: deploy tooling, never in a
release. Example photos are not fixtures (D16): none is in fixtures.txt or sent to the inference host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

LIST = "webapp/static/examples/examples.json"
TIMEOUT_S = 190   # the front's nginx waits this long for an answer (proxy_read_timeout, OPS-16 D8)
BUSY_RETRY_S = 3  # as verify's smoke photo
OFFLINE = b"the classifier is offline"   # check_service.py's OFFLINE; every other 503 is busy, as the page reads it
# Every request says it is the deploy's (ticket OPS-3 D14): the logs keep the marker, and the reading script
# (coffeecv/timelines.py) leaves these requests out of the counts.
HEADERS = {"User-Agent": "coffee-cv-verify", "X-Coffee-CV-Probe": "verify"}


def expected(example: dict) -> str:
    return example["country"] if example["country"] is not None else "refused_ood"


def outcome(answer: str) -> str:
    """What an /analyze answer comes to: a refusal's verdict, else the top-1 country; or what went wrong."""
    for line in answer.splitlines():
        try:
            part = json.loads(line)
        except ValueError:
            continue
        if part.get("part") == "error":
            return f"error: {part.get('message', '?')}"
        if part.get("part") == "classification":
            verdict = part.get("verdict", "")
            if verdict.startswith("refused"):
                return verdict
            ranked = part.get("ranked") or []
            return ranked[0]["id"] if ranked else f"no ranking ({verdict})"
    return "no classification"


def _git_show(repo: Path, commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _request(req: urllib.request.Request) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(url: str, filename: str, photo: bytes) -> tuple[int, bytes]:
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="{filename}"\r\n'
            f"Content-Type: image/jxl\r\n\r\n").encode() + photo + f"\r\n--{boundary}--\r\n".encode()
    def send() -> tuple[int, bytes]:
        return _request(urllib.request.Request(
            f"{url}/analyze", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **HEADERS}))

    status, answer = send()
    if status == 503 and OFFLINE not in answer:   # busy is retried once; offline is not, it lasts minutes (OPS-15 D12)
        time.sleep(BUSY_RETRY_S)
        status, answer = send()
    return status, answer


def check(url: str, commit: str, repo: Path, say: Callable[[str], None]) -> int:
    """Runs every example's checks, saying one line each; 1 if any failed, else 0."""
    # A commit this checkout cannot resolve fails: skipping it would pass a release whose examples were never checked.
    if not commit or subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo,
                                    capture_output=True).returncode != 0:
        say(f"FAIL  the release's commit: got '{commit}', which this checkout does not have (git fetch)")
        return 1
    listing = _git_show(repo, commit, LIST)
    if listing is None:
        say(f"skipped: the release's commit has no {LIST}")
        return 0
    fails = 0

    def result(what: str, got: str, want: str) -> bool:
        nonlocal fails
        if got == want:
            say(f"ok    {what}: {got}")
            return True
        say(f"FAIL  {what}: got '{got}', want '{want}'")
        fails += 1
        return False

    for example in json.loads(listing)["examples"]:
        name, path = example["name"], f"examples/{example['name']}.jxl"
        status, served = _request(urllib.request.Request(f"{url}/{path}", headers=HEADERS))
        if not result(f"example {name} served", f"HTTP {status}", "HTTP 200"):
            continue
        release = _git_show(repo, commit, f"webapp/static/{path}") or b""
        if not result(f"example {name} served = the release's", hashlib.sha256(served).hexdigest()[:16],
                      hashlib.sha256(release).hexdigest()[:16]):
            continue
        status, answer = _post(url, f"{name}.jxl", served)
        got = outcome(answer.decode("utf-8", "replace")) if status == 200 else f"HTTP {status}"
        result(f"example {name}", got, expected(example))
    return 1 if fails else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="the domain's URL, https://DOMAIN (verify's PUBLIC_URL)")
    ap.add_argument("--commit", required=True, help="the commit the release serves (its BUILD_INFO.json)")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2], help="the git repo")
    args = ap.parse_args()
    sys.exit(check(args.url.rstrip("/"), args.commit, args.repo, lambda line: print(line, flush=True)))


if __name__ == "__main__":
    main()
