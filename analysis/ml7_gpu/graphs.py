"""Moved to coffeecv/litert_forms.py (ticket ML-7 plan P9). This name stays for P1-P7's scripts, which import
`graphs` from this directory: it is that module. ML7_REPO, as before, names the repo (or a release laid out like it)
whose coffeecv is imported."""
import os
import sys
from pathlib import Path

_repo = str(Path(os.environ.get("ML7_REPO") or Path(__file__).resolve().parents[2]))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from coffeecv import litert_forms  # noqa: E402

sys.modules[__name__] = litert_forms
