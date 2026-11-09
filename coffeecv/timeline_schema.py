"""A timeline's schema (GLOSSARY.md "Timeline"; ticket OPS-3 D7, D16, D17's layer 2): what the page sends to
POST /timeline, once per photo, and what the check service writes and the reading script (coffeecv/timelines.py)
reads. Stdlib only: it ships to the front with the check service (webapp/deploy/release_manifest.py, FRONT_CODE).

The page sends only what the browser alone knows (D17's layer 1): which control sent the photo, the example if it was
one, its own times, its upload attempts and how the photo ended. The photo's name, format and size and every host's
time come from the hosts' own lines, joined by request ID. So nothing here is free text: every key is known, every
value is from a fixed set, a time, an ID in nginx's form, or an example named in examples.json. Anything else is
refused, and the refusal never repeats what was sent.

The page's times, whole milliseconds on the browser's clock (performance.now()), one per step of the ticket's §2
table that the page measures:

    pick_ms           1   the control's tap until the file arrived (not for an example or a re-run)
    download_ms       2   an example's JPEG XL downloaded
    size_check_ms     3   the size check
    to_preview_ms     13  the last attempt's upload sent until the preview part arrived
    preview_drawn_ms  14  the preview part arrived until the photo and its mask were drawn
    to_answer_ms      16  the last attempt's upload sent until the classification arrived
    answer_shown_ms   16  the classification arrived until it was shown
    total_ms          17  the tap (or the re-run's click) until the outcome was on screen

and per upload attempt, in order (steps 4-6): its request ID from the answer's X-Request-Id (none if it never got
an answer), its status (0 if none), connect_ms (DNS, TCP and TLS, from Resource Timing, when the browser gives one),
upload_ms (the attempt's start until the upload had left the browser), headers_ms (from then until the status
arrived), and wait_ms (the busy wait after it, before the next attempt).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

VERSION = 1
# The controls that send a photo (OPS-17 D7): Choose Photo (the photo picker; on iOS and the desktop it also offers a
# file browser and the camera), Select File (Android's file browser, OPS-21), the camera button, and an example.
CONTROLS = ("choose_photo", "select_file", "camera", "example")
# How the photo ended (the ticket's §7): answered (the classification shown, a refusal by the model included), failed
# (the app's error part), refused_on_page (over the size cap, before any upload), refused (a 4xx or 500 before the
# answer: the front's checks or the app's), busy (still busy after the page's retries), offline, unreachable (no
# answer at all), cut (the answer stopped before its classification), abandoned (another photo picked, or the page
# left, before the outcome), download_failed (an example's photo could not be fetched).
OUTCOMES = ("answered", "failed", "refused_on_page", "refused", "busy", "offline", "unreachable", "cut", "abandoned",
            "download_failed")
TIMES = ("pick_ms", "download_ms", "size_check_ms", "to_preview_ms", "preview_drawn_ms", "to_answer_ms",
         "answer_shown_ms", "total_ms")
ATTEMPT_TIMES = ("connect_ms", "upload_ms", "headers_ms", "wait_ms")
MAX_MS = 600_000       # 10 minutes: no step of a photo takes longer
MAX_ATTEMPTS = 8       # the page makes at most 4 (one, then three busy retries)
MAX_BODY = 4096        # bytes (D23): a real timeline is well under 2 KB
PROBE = "verify"       # the deploy's own checks (D14)
REQUEST_ID = re.compile(r"[0-9a-f]{32}")   # nginx's $request_id (D8)

_REQUIRED = {"v", "request_id", "control", "outcome", "attempts"}
_KEYS = _REQUIRED | {"rerun_of", "example", "probe", *TIMES}
_ATTEMPT_KEYS = {"request_id", "status", *ATTEMPT_TIMES}


class Invalid(ValueError):
    """A timeline refused, saying which known key is wrong, never what was sent."""


def example_names(path: Path) -> frozenset[str]:
    """The example names in examples.json (OPS-16 D13); none if it cannot be read, so no example timeline passes."""
    try:
        return frozenset(e["name"] for e in json.loads(Path(path).read_text())["examples"])
    except (OSError, ValueError, KeyError, TypeError):
        return frozenset()


def parse(body: bytes, examples: frozenset[str] | set[str]) -> dict:
    """A POST /timeline body: UTF-8 JSON, with no NaN or Infinity and no key twice, then check()."""
    def no_constant(name):
        raise Invalid("not a number")

    def no_duplicates(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise Invalid("a key given twice")
        return dict(pairs)

    try:
        data = json.loads(body.decode("utf-8"), parse_constant=no_constant, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise Invalid("not UTF-8 JSON") from None
    return check(data, examples)


def check(data: object, examples: frozenset[str] | set[str] | None) -> dict:
    """`data` if it is a timeline the schema knows; else Invalid. `examples` None takes any example name: for a line
    the check service has already held to the release's list (the reading script)."""
    if not isinstance(data, dict):
        raise Invalid("not an object")
    if set(data) - _KEYS:
        raise Invalid("an unknown key")
    for key in sorted(_REQUIRED - set(data)):
        raise Invalid(f"no {key}")
    if data["v"] != VERSION or isinstance(data["v"], bool):
        raise Invalid("v is not 1")
    _one_of(data, "control", CONTROLS)
    _one_of(data, "outcome", OUTCOMES)
    _id(data["request_id"], "request_id", none=True)
    if "rerun_of" in data:
        _id(data["rerun_of"], "rerun_of")
    if (data["control"] == "example") != ("example" in data):
        raise Invalid("example is given with, and only with, the example control")
    if "example" in data and (not isinstance(data["example"], str)
                              or examples is not None and data["example"] not in examples):
        raise Invalid("example is not in examples.json")
    if "probe" in data and data["probe"] != PROBE:
        raise Invalid("probe is not the deploy's")
    for key in TIMES:
        if key in data:
            _ms(data[key], key)
    attempts = data["attempts"]
    if not isinstance(attempts, list) or len(attempts) > MAX_ATTEMPTS:
        raise Invalid(f"attempts is not a list of at most {MAX_ATTEMPTS}")
    for attempt in attempts:
        if not isinstance(attempt, dict) or "status" not in attempt:
            raise Invalid("an attempt is not an object with a status")
        if set(attempt) - _ATTEMPT_KEYS:
            raise Invalid("an attempt has an unknown key")
        _id(attempt.get("request_id"), "an attempt's request_id", none=True)
        status = attempt["status"]
        if not _int(status) or not (status == 0 or 100 <= status <= 599):
            raise Invalid("an attempt's status is not 0 or an HTTP status")
        for key in ATTEMPT_TIMES:
            if key in attempt:
                _ms(attempt[key], f"an attempt's {key}")
    return data


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ms(value, name: str) -> None:
    if not _int(value) or not 0 <= value <= MAX_MS:
        raise Invalid(f"{name} is not whole milliseconds from 0 to {MAX_MS}")


def _one_of(data: dict, key: str, values: tuple[str, ...]) -> None:
    if data[key] not in values or not isinstance(data[key], str):
        raise Invalid(f"{key} is not one of the schema's")


def _id(value, name: str, none: bool = False) -> None:
    if value is None and none:
        return
    if not isinstance(value, str) or not REQUEST_ID.fullmatch(value):
        raise Invalid(f"{name} is not a request ID")
