"""ML-5 D23: summarise point_probe.py's rows.csv per (set, model, output) into summary.txt.

    PYTHONPATH=. python analysis/ml5_point_probe/summary.py
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def share_at_least(rows: list[dict], col: str, cut: float = 0.9) -> float:
    return sum(float(r[col]) >= cut for r in rows) / len(rows)


def main() -> None:
    groups = defaultdict(list)
    for r in csv.DictReader((HERE / "rows.csv").read_text().splitlines()):
        groups[(r["set"], r["model"], r["output"])].append(r)
    lines = [f"{'set':3} {'model':14} {'out':7} {'n':>4}  replay>=.9 empty  | box-only>=.9  click1  click2  click3 "
             f"| median IoU c0..c3"]
    for key in sorted(groups):
        rs = groups[key]
        empty = sum(int(r["replay_empty"]) for r in rs) / len(rs)
        medians = " ".join(f"{statistics.median(float(r[f'click{i}']) for r in rs):.3f}" for i in range(4))
        lines.append(f"{key[0]:3} {key[1]:14} {key[2]:7} {len(rs):4}  {share_at_least(rs, 'replay_iou'):9.0%} "
                     f"{empty:6.0%} | {share_at_least(rs, 'click0'):10.0%} {share_at_least(rs, 'click1'):7.0%} "
                     f"{share_at_least(rs, 'click2'):7.0%} {share_at_least(rs, 'click3'):7.0%} | {medians}")
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
