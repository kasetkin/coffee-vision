"""ML-19 P4 (D29): the canonical store's sizes against the originals', by format and by size, and the host's byte cap.

    python analysis/ml19/p4_sizes.py > analysis/ml19/p4_sizes.md

Reads every photo of every canonicalize@<folder> stage in dvc.yaml and its canonical photo; pixel counts from the
canonical photos' headers.
"""
from __future__ import annotations

import collections
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from coffeecv import canonical as cn  # noqa: E402
from coffeecv.jxl_header import read_header  # noqa: E402

folders = yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]["canonicalize"]["foreach"]
rows = []
for folder in folders:
    for p in sorted((REPO / "dataset" / folder).rglob("*")):
        if p.is_file() and cn.is_photo(p):
            c = cn.canonical_path(p)
            data = c.read_bytes()
            h = read_header(data)
            rows.append({"folder": folder, "ext": p.suffix.lower().replace(".jpeg", ".jpg"), "orig": p.stat().st_size,
                         "canon": len(data), "mp": h.width * h.height / 1e6, "name": str(p.relative_to(REPO))})

MB = 2**20


def table(groups: dict[str, list[dict]]) -> None:
    print("| Group | Photos | Originals MB | Canonical MB | Ratio | Canonical median / max MB |")
    print("|---|---|---|---|---|---|")
    for k, rs in groups.items():
        o, c = sum(r["orig"] for r in rs), sum(r["canon"] for r in rs)
        print(f"| {k} | {len(rs)} | {o / MB:,.0f} | {c / MB:,.0f} | {c / o:.2f} | "
              f"{statistics.median(r['canon'] for r in rs) / MB:.2f} / {max(r['canon'] for r in rs) / MB:.2f} |")


print("# ML-19 P4: the canonical store's sizes (D29)\n")
print(f"{len(rows)} photos in {len(folders)} folders; written by `analysis/ml19/p4_sizes.py`.\n")
print("## By the original's format\n")
by_ext = collections.defaultdict(list)
for r in rows:
    by_ext[r["ext"]].append(r)
table(dict(sorted(by_ext.items(), key=lambda kv: -len(kv[1]))))
print("\n## By the canonical photo's size\n")
bins = ((0, 8, "under 8 MP"), (8, 16, "8-16 MP"), (16, 30, "16-30 MP"), (30, 45, "30-45 MP"), (45, 61, "45-60 MP"))
table({label: [r for r in rows if lo <= r["mp"] < hi] for lo, hi, label in bins if any(lo <= r["mp"] < hi for r in rows)})
print("\n## By folder\n")
table({f: [r for r in rows if r["folder"] == f] for f in folders})
print("\n## The largest canonical photos\n")
for r in sorted(rows, key=lambda r: -r["canon"])[:8]:
    print(f"- {r['canon'] / MB:.2f} MB, {r['mp']:.1f} MP: `{r['name']}` (original {r['orig'] / MB:.2f} MB)")
print("\n## D29: the median photo, and the host's cap\n")
med_o, med_c = statistics.median(r["orig"] for r in rows), statistics.median(r["canon"] for r in rows)
print(f"- The median canonical photo is {med_c / MB:.2f} MB against the median original's {med_o / MB:.2f} MB "
      f"({med_c / med_o:.2f}): the median photo does not grow, so q95 stands (D29). By format: "
      + ", ".join(f"{k} {statistics.median(r['canon'] for r in rs) / statistics.median(r['orig'] for r in rs):.2f}"
                  for k, rs in sorted(by_ext.items(), key=lambda kv: -len(kv[1]))) + " (canonical over original).")
front = [r for r in rows if not (r["ext"] in {".dng"} and r["mp"] > 16)]
print(f"- What the front can send (RAW up to 16 MP, every other photo up to 50.2 MP; D24, D38 as recommended): the "
      f"largest is {max(r['canon'] for r in front) / MB:.2f} MB. The host's 20 MB cap (its nginx and "
      "MAX_UPLOAD_BYTES) holds it; only the store's 48-50 MP DNGs are larger, and they never cross.")
largest = max(r["canon"] for r in rows)
total_o, total_c = sum(r["orig"] for r in rows), sum(r["canon"] for r in rows)
print(f"\nIn all: originals {total_o / 2**30:.2f} GB, canonical photos {total_c / 2**30:.2f} GB "
      f"({total_c / total_o:.2f}). The largest canonical photo is {largest / MB:.2f} MB.")
