"""The deploy's fixtures as the inference host reads them (ticket ML-19 D34): each photo fixtures.txt pins, checked
against its sha256, canonicalized here, on the machine that builds the canonical store (D51).

    python webapp/deploy/canonical_fixtures.py webapp/deploy/fixtures.txt OUT_DIR

Writes OUT_DIR/<basename>.jxl for every fixture and OUT_DIR/fixtures.canonical.txt, in fixtures.txt's own format
(`<role> <sha256> <path>`), each line the canonical file's sha256 and name: release_probe.py's --list reads it on the
host as it reads fixtures.txt. The smoke, the expected answers and --compare (PyTorch here, the release there) all read
these bytes, as the front would send them.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

from coffeecv import canonical  # noqa: E402
from coffeecv.photo_io import DATASET_MAX_PIXELS  # noqa: E402

CANONICAL_LIST = "fixtures.canonical.txt"


def main() -> None:
    fixtures, out = Path(sys.argv[1]), Path(sys.argv[2])
    Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS   # the owner's own photos, as the dataset tools read them
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in fixtures.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        role, sha, path = line.split()
        data = (REPO / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise SystemExit(f"canonical_fixtures: {path} does not match its sha256 in {fixtures}")
        name = f"{Path(path).name}.jxl"
        target = out / name
        target.write_bytes(canonical.canonicalize(data, Path(path).name))   # every run: never a stale one
        lines.append(f"{role} {hashlib.sha256(target.read_bytes()).hexdigest()} {name}")
    (out / CANONICAL_LIST).write_text("\n".join(lines) + "\n")
    print(f"canonical_fixtures: {len(lines)} fixtures in {out}")


if __name__ == "__main__":
    main()
