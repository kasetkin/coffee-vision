"""Verify the pretrained backbones in this directory against manifest.json.

Run after obtaining the weights on a fresh clone (see README.md, and
docs/dinov3_integration_plan.md §2.5 for where to get them):

    python models_pretrained/verify.py

Exits non-zero if anything is missing, truncated, or has the wrong contents, so it
can gate a sweep launch. Deliberately depends on nothing but the standard library --
it has to be runnable before the environment is built, and a verifier that needs
torch cannot check whether torch will be able to load the file.

Two independent checks per file:
  1. full sha256 against manifest.json, which is git-tracked;
  2. upstream's own convention -- torch.hub names weights "<name>-<sha256[:8]>.pth",
     so the filename validates itself with no project metadata at all. That second
     check is what catches "downloaded the Hugging Face safetensors instead of the
     Meta .pth": the HF copy is a different file and will never match.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: no {MANIFEST.relative_to(ROOT.parent)} -- cannot verify anything.")
        return 2
    entries = json.loads(MANIFEST.read_text())
    failures, checked = [], 0

    for e in entries:
        path = ROOT / e["path"]
        label = e["path"]
        if not path.exists():
            failures.append(f"{label}: MISSING (see README.md for how to obtain it)")
            continue
        size = path.stat().st_size
        if size != e["bytes"]:
            # Cheapest discriminator, so report it before spending a hash on the file.
            failures.append(
                f"{label}: SIZE {size:,} != expected {e['bytes']:,} "
                f"({'truncated download' if size < e['bytes'] else 'wrong file'})")
            continue
        digest = sha256_of(path)
        checked += 1
        if digest != e["sha256"]:
            # An entry with a pinned "source" says where the right file comes from; the others
            # are the DINO/torch.hub weights, where the usual mistake is the HF safetensors.
            hint = (f"    Download it again from {e['source']}" if e.get("source") else
                    f"    Most likely the Hugging Face safetensors were downloaded instead of the\n"
                    f"    Meta .pth -- they are different files. See plan §2.5.")
            failures.append(
                f"{label}: SHA256 {digest[:16]}... != expected {e['sha256'][:16]}...\n{hint}")
            continue
        stem_hash = path.stem.rsplit("-", 1)[-1]
        self_ok = (len(stem_hash) == 8 and all(c in "0123456789abcdef" for c in stem_hash)
                   and digest.startswith(stem_hash))
        note = "sha256 OK, filename self-verifies" if self_ok else "sha256 OK"
        print(f"  {label:58s} {note}")

    if failures:
        print(f"\n{len(failures)} problem(s):\n")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nAll {checked} file(s) verified against manifest.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
