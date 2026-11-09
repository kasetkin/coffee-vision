"""--verify's check that a timeline is written (ticket OPS-3 D14, plan P5), in two halves:

    python webapp/deploy/timeline_check.py post --url https://DOMAIN --request-id RID
    <the front's log lines> | python webapp/deploy/timeline_check.py found --timeline-id TID --request-id RID

`post` sends one timeline through the domain, as the page sends one, for verify's own photo (RID, the X-Request-Id of
its POST /analyze), marked as the deploy's (X-Coffee-CV-Probe: verify, and the timeline's own probe), so the reading
script leaves it out; it must be accepted (204), and it prints TIMELINE_ID=, the X-Request-Id of the POST /timeline.
`found` reads the check service's log lines, which verify greps on the front as the deploy user, and must find that
timeline written there, naming RID.

Prints one line per check in verify's own words (`ok    <what>: <got>`, `FAIL  <what>: got '<got>', want '<want>'`)
and exits 1 on any failure, which rolls a full deploy back. Standard library only: deploy tooling, never in a release.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable

PROBE_HEADER, PROBE = "X-Coffee-CV-Probe", "verify"   # coffeecv/timeline_schema.py's PROBE, check_service.py's header
TIMEOUT_S = 30


def timeline(request_id: str) -> dict:
    """verify's own photo's timeline: Choose Photo, answered at the first attempt. No times: verify measures none."""
    return {"v": 1, "request_id": request_id, "control": "choose_photo", "outcome": "answered",
            "attempts": [{"request_id": request_id, "status": 200}], "probe": PROBE}


def post(url: str, request_id: str, say: Callable[[str], None]) -> str | None:
    """Sends the timeline; the X-Request-Id it was written under, or None if it was not accepted."""
    req = urllib.request.Request(f"{url}/timeline", data=json.dumps(timeline(request_id)).encode(), method="POST",
                                 headers={"Content-Type": "text/plain;charset=UTF-8", "User-Agent": "coffee-cv-verify",
                                          PROBE_HEADER: PROBE})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            status, written = resp.status, resp.headers.get("X-Request-Id")
    except urllib.error.HTTPError as exc:
        status, written = exc.code, None
    except OSError as exc:
        status, written = f"no answer ({type(exc).__name__})", None
    if status != 204:
        say(f"FAIL  public POST /timeline: got '{status}', want '204'")
        return None
    say(f"ok    public POST /timeline: {status}")
    return written


def found(lines: Iterable[str], timeline_id: str, request_id: str) -> bool:
    """Whether the log holds the timeline written under `timeline_id`, verify's, for photo `request_id`."""
    for raw in lines:
        try:
            line = json.loads(raw)
        except ValueError:
            continue
        if (isinstance(line, dict) and line.get("kind") == "timeline" and line.get("request_id") == timeline_id
                and line.get("probe") == PROBE and isinstance(line.get("timeline"), dict)
                and line["timeline"].get("request_id") == request_id):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("post", help="send verify's timeline through the domain")
    p.add_argument("--url", required=True, help="the domain's URL, https://DOMAIN (verify's PUBLIC_URL)")
    p.add_argument("--request-id", required=True, help="the X-Request-Id of verify's POST /analyze")
    f = sub.add_parser("found", help="find it in the check service's log lines, read from stdin")
    f.add_argument("--timeline-id", required=True, help="what `post` printed as TIMELINE_ID")
    f.add_argument("--request-id", required=True)
    args = ap.parse_args()
    say = lambda line: print(line, flush=True)   # noqa: E731
    if args.command == "post":
        written = post(args.url.rstrip("/"), args.request_id, say)
        if written is None:
            sys.exit(1)
        print(f"TIMELINE_ID={written}")
        return
    if found(sys.stdin, args.timeline_id, args.request_id):
        say(f"ok    the front's log has verify's timeline: {args.timeline_id}")
        return
    say(f"FAIL  the front's log has verify's timeline: got 'none', want '{args.timeline_id}'")
    sys.exit(1)


if __name__ == "__main__":
    main()
