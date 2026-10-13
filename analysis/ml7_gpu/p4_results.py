"""ML-7 P4: results.json from the comparison run (plan P4), for P5's report. Reads p4_run.sh's records, fetched
from yappy into one directory (r1/, r2/, r3/), and summarises across rounds:

  graphs   per cell and graph: the median of the rounds' warm medians, its spread between rounds, p90, the cold
           start, the time per photo (the calls a photo needs), CPU time against wall time, peak RSS and the
           MemAvailable drop, NaN or Inf in any call, the clocks and the power-limit throttle, and on a GPU cell
           whether every round proved the GPU (no GPU cell counts otherwise, plan P4). The stability run apart.
  replay   per cell: the median photo and its p90, the spread of the median between rounds, the cold start to
           the first answer, peak RSS and the MemAvailable drop with the three graphs loaded, and parity (D7) of
           each round's outputs against refs.npz, computed here with the repo's code (parity.py).

A record that is missing or unreadable is a failed round, kept by name. From the repo root, in the training venv:

    python analysis/ml7_gpu/p4_results.py --runs analysis/ml7_gpu/artifacts/outputs_p4 \\
        --refs analysis/ml7_gpu/artifacts/refs.npz --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt \\
        --out analysis/ml7_gpu/results.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = re.compile(r"(?P<cell>[a-z0-9_]+?)__(?P<what>dino_b8|dino_b40|encoder|decoder|replay)(?P<stab>__stability)?$")


def _spread(values: list[float]) -> float | None:
    """(max - min) / median: how far the rounds' medians drift apart."""
    return (max(values) - min(values)) / statistics.median(values) if len(values) > 1 else None


def _proven(records: list[dict]) -> bool | None:
    proofs = [r["gpu_proof"]["ok"] for r in records if "gpu_proof" in r]
    return all(proofs) if proofs else None


def _common(records: list[dict]) -> dict:
    c = [r.get("counters") or {} for r in records]
    return {"cpu_mhz_median": [x.get("cpu_mhz_median") for x in c],
            "gpu_mhz_median": [x.get("gpu_mhz_median") for x in c], "pl1_throttled": any(x.get("pl1_throttled") for x in c),
            "peak_rss_mb": max(r["peak_rss_mb"] for r in records),
            "mem_available_drop_mb": max((r.get("mem_available_drop_mb") or 0) for r in records),
            "gpu_proven": _proven(records)}


def _split(rounds: dict[int, dict | None]) -> tuple[dict, list[int], dict]:
    """The rounds that produced a record; the failed ones (None, or p4_run.sh's {"failed": text}); their errors."""
    ok = {k: r for k, r in sorted(rounds.items()) if r is not None and "failed" not in r}
    failed = [k for k in sorted(rounds) if k not in ok]
    errors = {str(k): rounds[k]["failed"] for k in failed if rounds[k] is not None}
    return ok, failed, errors


def summarise_graph(rounds: dict[int, dict | None]) -> dict:
    """{round: the record of p4_graph.py, None, or a failure} -> the cell and graph across rounds."""
    ok, failed, errors = _split(rounds)
    if not ok:
        return {"rounds": 0, "failed_rounds": failed, "errors": errors}
    recs = list(ok.values())
    medians = [r["timing"]["warm"]["median"] for r in recs]
    median = statistics.median(medians)
    return {"rounds": len(recs), "failed_rounds": failed, "errors": errors, "warm_median_s": median,
            "warm_medians_s": medians, "round_spread": _spread(medians),
            "warm_p90_s": statistics.median(r["timing"]["warm"]["p90"] for r in recs),
            "cold_s": [r["timing"]["cold_s"] for r in recs],
            "per_photo_s": median * recs[0].get("calls_per_photo", 1),
            "cpu_over_wall": statistics.median(r["cpu_over_wall"] for r in recs),
            "nonfinite_calls": sum(r["nonfinite_calls"] for r in recs),
            "delegate": [r["delegate"] for r in recs if r.get("delegate")][:1] or None,
            **_common(recs)}


def summarise_replay(rounds: dict[int, dict | None]) -> dict:
    """{round: the record of p4_replay.py, None, or a failure} -> the cell's photo replay across rounds."""
    ok, failed, errors = _split(rounds)
    if not ok:
        return {"rounds": 0, "failed_rounds": failed, "errors": errors}
    recs = list(ok.values())
    medians = [r["photo"]["median"] for r in recs]
    return {"rounds": len(recs), "failed_rounds": failed, "errors": errors,
            "batch": sorted({r["batch"] for r in recs}),
            "photo_median_s": statistics.median(medians), "photo_medians_s": medians,
            "round_spread": _spread(medians), "photo_p90_s": statistics.median(r["photo"]["p90"] for r in recs),
            "first_answer_s": [r["first_answer_s"] for r in recs],
            "process_to_first_answer_s": [r["process_to_first_answer_s"] for r in recs],
            "nonfinite_photos": sum(r["nonfinite_photos"] for r in recs), **_common(recs)}


def load_runs(runs: Path) -> dict:
    """{(cell, what, stability): {round: record, {"failed": text} or None}} from runs/r<k>/: each *.json, each
    *.failed that p4_run.sh leaves for a process that failed, and None for a record absent from a round in which
    the cell ran."""
    out: dict = {}
    rounds = sorted(int(d.name[1:]) for d in runs.glob("r[0-9]*") if d.is_dir())
    for k in rounds:
        for f in sorted((runs / f"r{k}").glob("*.json")) + sorted((runs / f"r{k}").glob("*.failed")):
            m = RECORD.match(f.stem)
            if not m:
                continue
            if f.suffix == ".failed":
                rec = {"failed": f.read_text()}
            else:
                try:
                    rec = json.loads(f.read_text())
                except ValueError:
                    rec = None
            out.setdefault((m["cell"], m["what"], bool(m["stab"])), {})[k] = rec
    for key, by_round in out.items():
        for k in rounds:
            if k not in by_round and any((runs / f"r{k}").glob(f"{key[0]}__*")):
                by_round[k] = None
    return out


def replay_parity(runs: Path, cell: str, refs_path: Path, context) -> dict:
    """D7 per round on the replay's first pass (refs cut to the photos the replay ran): the gate and the per-photo
    measures, the continuous ones from P1 among them (plan P4)."""
    import numpy as np
    from parity import compare
    refs = dict(np.load(refs_path, allow_pickle=False))
    out = {}
    for f in sorted(runs.glob(f"r*/{cell}__replay.npz")):
        got = dict(np.load(f, allow_pickle=False))
        n = len(got["readouts"])
        report = compare({k: v[:n] if getattr(v, "ndim", 0) and len(v) == len(refs["photos"]) else v
                          for k, v in refs.items()}, got, context)
        out[f.parent.name] = report
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from make_refs import Context

    context = Context.load(args.checkpoint)
    results: dict = {"graphs": {}, "stability": {}, "replay": {}}
    for (cell, what, stab), rounds in sorted(load_runs(args.runs).items()):
        if what == "replay":
            results["replay"][cell] = summarise_replay(rounds) | {
                "parity": replay_parity(args.runs, cell, args.refs, context)}
        else:
            results["stability" if stab else "graphs"].setdefault(cell, {})[what] = summarise_graph(rounds)
    args.out.write_text(json.dumps(results, indent=2, default=float) + "\n")
    for cell, r in results["replay"].items():
        gates = [p["gate"] for p in r.get("parity", {}).values()]
        print(f"{cell:12s} replay {r.get('photo_median_s', float('nan')):7.2f} s/photo  "
              f"parity {'PASS' if gates and all(g['pass'] for g in gates) else 'FAIL'}  "
              f"gpu {r.get('gpu_proven')}")


if __name__ == "__main__":
    main()
