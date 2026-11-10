"""The deploy's fixtures as the inference host reads them (ticket ML-19 D34): each photo fixtures.txt pins, checked
against its sha256, canonicalized here, on the machine that builds the canonical store (D51). And since OPS-22, the
photos as the front sends them, made with its own settings (canonical.FRONT_SETTINGS, OPS-22 D1, D3).

    python webapp/deploy/canonical_fixtures.py webapp/deploy/fixtures.txt OUT_DIR [--examples-from SHA]

Writes OUT_DIR/<basename>.jxl for every fixture and OUT_DIR/fixtures.canonical.txt, in fixtures.txt's own format
(`<role> <sha256> <path>`), each line the canonical file's sha256 and name: release_probe.py's --list reads it on the
host as it reads fixtures.txt. The smoke, the expected answers and --compare (PyTorch here, the release there) all read
these bytes, the store's canonical photos.

OUT_DIR/front/<basename>.jxl is the smoke photo as the front makes it: --verify posts the original through the domain,
so its expected answer there is the staged app's on this one (OPS-22 D6). With --examples-from, OUT_DIR/front/examples/
<name>.jxl is each of that commit's examples (webapp/static/examples/) as the front re-encodes it, which the deploy
answers with the staged app before anything is stopped (OPS-22 D7). Neither is in fixtures.canonical.txt: they never
reach the inference host's fixtures. A commit without the example list (before OPS-16) has none to write.

Both are made by this checkout's coffeecv, as the store's fixtures are (the deploy runs from the release's commit).
That this machine makes the front's bytes is ML-19 P1's finding for libjxl at efforts 3-7; for the front's integer
colour path it is unchecked here (OPS-22 plan, finding 5), and --verify's public top-1 is the first check of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

from coffeecv import canonical  # noqa: E402
from coffeecv.photo_io import DATASET_MAX_PIXELS  # noqa: E402

CANONICAL_LIST = "fixtures.canonical.txt"
FRONT_DIR = "front"
EXAMPLES_DIR = "examples"
EXAMPLES = "webapp/static/examples"


def as_the_front_sends(data: bytes, name: str) -> bytes:
    """What the front's check service makes of an upload (OPS-22 D1, D3): its own call, with its settings, whatever
    the photo is, an already canonical one too."""
    return canonical.canonicalize_with(data, name, canonical.FRONT_SETTINGS).canonical


def _git_show(commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def write(fixtures: Path, out: Path, examples_from: str | None = None) -> int:
    """Writes OUT_DIR's files (above); returns how many fixtures."""
    Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS   # the owner's own photos, as the dataset tools read them
    front = out / FRONT_DIR
    front.mkdir(parents=True, exist_ok=True)
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
        if role == "smoke":
            (front / name).write_bytes(as_the_front_sends(data, Path(path).name))
    (out / CANONICAL_LIST).write_text("\n".join(lines) + "\n")
    if examples_from is not None:
        examples = front / EXAMPLES_DIR
        examples.mkdir(exist_ok=True)
        for old in examples.glob("*.jxl"):   # never a stale one from another commit's list
            old.unlink()
        listing = _git_show(examples_from, f"{EXAMPLES}/examples.json")
        if subprocess.run(["git", "cat-file", "-e", f"{examples_from}^{{commit}}"], cwd=REPO).returncode != 0:
            raise SystemExit(f"canonical_fixtures: {examples_from} is not a commit here")
        for example in json.loads(listing)["examples"] if listing is not None else []:
            name = f"{example['name']}.jxl"
            (examples / name).write_bytes(as_the_front_sends(_git_show(examples_from, f"{EXAMPLES}/{name}"), name))
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fixtures", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--examples-from", metavar="SHA",
                    help="also each of this commit's examples as the front sends it (OPS-22 D7)")
    args = ap.parse_args()
    n = write(args.fixtures, args.out, args.examples_from)
    print(f"canonical_fixtures: {n} fixtures in {args.out}")


if __name__ == "__main__":
    main()
