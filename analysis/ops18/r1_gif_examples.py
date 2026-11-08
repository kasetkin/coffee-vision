"""Ticket OPS-18 R1 (D6: a record, not a gate): the four example photos (OPS-16) sent to a local front as they are,
as a GIF and as a 24-bit BMP, and their verdicts side by side.

    python analysis/ops18/r1_gif_examples.py http://127.0.0.1:8100 > analysis/ops18/r1_gif_examples.md

The examples are the page's canonical photos (webapp/static/examples/<name>.jxl): each is read back to 8-bit sRGB
(canonical.load_canonical), then saved by Pillow as a GIF (256 colours, Pillow's adaptive palette, as an RGB image
saves to GIF) and as a BMP. A local front (webapp/front/check_service.py) forwards to a local inference host
(webapp/app.py, its LiteRT model on this machine's CPU), so the answers are the release's code, not the live site.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv import canonical  # noqa: E402
EXAMPLES = REPO / "webapp" / "static" / "examples"


def analyze(front: str, name: str, data: bytes) -> dict:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{front}/analyze", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            parts = [json.loads(line) for line in resp.read().splitlines() if line.strip()]
    except urllib.error.HTTPError as exc:
        return {"error": f"{exc.code} {json.loads(exc.read()).get('error')}"}
    return next((p for p in parts if p.get("part") != "preview"), parts[-1])


def verdict(answer: dict) -> str:
    """The answer as the page states it: the top country and its score, a warning, or the refusal."""
    if "error" in answer:
        return f"error: {answer['error']}"
    if answer.get("verdict") != "predicted" and not answer.get("ranked"):
        return f"{answer.get('verdict')}: {answer.get('message')}"
    top = answer["ranked"][0]
    warned = f" ({answer['warning']})" if answer.get("warning") else ""
    return f"{top['label']} {top['score']:.2f}{warned}"


def main() -> None:
    front = sys.argv[1].rstrip("/")
    examples = json.loads((EXAMPLES / "examples.json").read_text())["examples"]
    print("| Example | Expected | As sent (canonical JPEG XL) | GIF | BMP (24-bit) |")
    print("|---|---|---|---|---|")
    for ex in examples:
        data = (EXAMPLES / f"{ex['name']}.jxl").read_bytes()
        rgb = Image.fromarray(canonical.load_canonical(data))
        row = [verdict(analyze(front, f"{ex['name']}.jxl", data))]
        for fmt in ("GIF", "BMP"):
            buf = io.BytesIO()
            rgb.save(buf, fmt)
            row.append(verdict(analyze(front, f"{ex['name']}.{fmt.lower()}", buf.getvalue())))
        print(f"| {ex['name']} | {ex['country'] or 'refused (no beans)'} | " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
