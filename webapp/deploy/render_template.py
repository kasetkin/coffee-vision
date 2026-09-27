"""Render a deploy template: substitute exactly the named variables, like `envsubst '$A $B'`.

    python webapp/deploy/render_template.py DOMAIN=example.org APP_ROOT=/opt/coffee-cv < in > out

Only the names given are replaced ($NAME or ${NAME}); every other `$` is left alone, which the nginx
template needs ($host, $uri, ${request_time} are nginx's own). Refuses an empty value, and a name the
template never mentions -- either one means the caller and the template disagree. Stdlib only, so it
runs on the deploying machine, which has no envsubst.
"""
from __future__ import annotations

import re
import sys


def render(text: str, values: dict[str, str]) -> str:
    for name, value in values.items():
        if not value:
            raise SystemExit(f"render_template: {name} is empty")
        pattern = re.compile(r"\$(?:\{" + name + r"\}|" + name + r"(?![A-Za-z0-9_]))")
        if not pattern.search(text):
            raise SystemExit(f"render_template: the template never mentions ${name}")
        text = pattern.sub(lambda _m: value, text)
    return text


def main() -> None:
    values = {}
    for arg in sys.argv[1:]:
        name, sep, value = arg.partition("=")
        if not sep or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            raise SystemExit(f"render_template: expected NAME=value, got {arg!r}")
        values[name] = value
    sys.stdout.write(render(sys.stdin.read(), values))


if __name__ == "__main__":
    main()
