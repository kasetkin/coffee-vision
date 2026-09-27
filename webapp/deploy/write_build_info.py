"""Stamp the code being deployed: writes webapp/BUILD_INFO.json on the machine the rsync starts FROM.

The web service is deployed by rsync, not by a git checkout, so the VM's own `git rev-parse HEAD`
describes whatever the VM last committed (the training sweeps commit there), not the code it is
serving. Run this immediately before the rsync, from the working tree being deployed; webapp/app.py
reads the file at import and puts it on every request log line. The file is gitignored.

    python webapp/deploy/write_build_info.py && rsync ...
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    info = {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # Tracked files that differ from the commit: the deploy is then NOT exactly `commit`.
        "dirty": bool(dirty),
        "dirty_files": [line[3:] for line in dirty.splitlines()][:50],
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "webapp/deploy/write_build_info.py at deploy time",
    }
    out = REPO / "webapp" / "BUILD_INFO.json"
    out.write_text(json.dumps(info, indent=1) + "\n")
    print(f"wrote {out.relative_to(REPO)}: {info['commit'][:12]} on {info['branch']}"
          f"{' (DIRTY: ' + str(len(info['dirty_files'])) + ' files)' if info['dirty'] else ''}")


if __name__ == "__main__":
    main()
