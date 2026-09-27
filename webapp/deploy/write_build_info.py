"""Stamp the code being deployed: writes webapp/BUILD_INFO.json, which webapp/app.py reads at import and
puts on its startup line and on every request log line.

A release is exported from a commit (scripts/deploy_webapp.sh: git archive of the manifest), not from a
working tree, so the deploy passes the commit and the model and says where to write:

    python webapp/deploy/write_build_info.py --ref <sha> --model allrigs_dino3b16_s123 --out <release>/webapp/BUILD_INFO.json

Without --ref it describes the working tree instead (HEAD, plus whether tracked files differ from it) --
for a local dev server only. The file is gitignored.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", help="the commit the release is exported from (default: working tree)")
    ap.add_argument("--model", help="the model the release serves, e.g. allrigs_dino3b16_s123")
    ap.add_argument("--out", type=Path, default=REPO / "webapp" / "BUILD_INFO.json")
    args = ap.parse_args()

    if args.ref:
        info = {
            "commit": git("rev-parse", "--verify", f"{args.ref}^{{commit}}"),
            # Exported from the commit itself, so nothing can differ from it.
            "dirty": False,
            "dirty_files": [],
            "source": "git archive of the commit by scripts/deploy_webapp.sh",
        }
    else:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        info = {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            # Tracked files that differ from the commit: the code is then NOT exactly `commit`.
            "dirty": bool(dirty),
            "dirty_files": [line[3:] for line in dirty.splitlines()][:50],
            "source": "webapp/deploy/write_build_info.py on a working tree",
        }
    info["model"] = args.model
    info["written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=1) + "\n")
    print(f"wrote {args.out}: {info['commit'][:12]}"
          f"{' model ' + args.model if args.model else ''}"
          f"{' (DIRTY: ' + str(len(info['dirty_files'])) + ' files)' if info['dirty'] else ''}")


if __name__ == "__main__":
    main()
