# ML-14: the metadata stripper on several cores

Plan: `docs/plan_strip_metadata_parallel.html`; ticket: `docs/ticket_strip_metadata_parallel.html` (decisions
D1-D15, Activity). The scripts written while building it, 2026-10-30; each runs from the repo root.

- `factor.py`: pixel_digest's peak memory per pixel for one photo (P3), run on real photos of
  `dataset/segmenter_positives`; `DECODE_BYTES_PER_PIXEL` in `coffeecv/strip_metadata.py` is its result, rounded up.
- `measure.py`, `run_all.sh`, `results.txt`: the six dry runs of D11 (P5), before (`413f7b4`) and after, on
  `dataset/segmenter_positives` and on a `/tmp` copy of it with private tags added back; `results.txt` is the
  output the ticket's table is taken from. `run_all.sh` names the setup it needs (a worktree, the copy).
- `wait_cancel.py`: CPython 3.12's `concurrent.futures.wait` never returning for a future cancelled by
  `shutdown(cancel_futures=True)`, the cause of the Ctrl-C hang the P4 test found.
- `ctrl_c_checks.py`, `sigint_debug/`: the by-hand checks of a parallel run stopped by a Ctrl-C or an error
  (`hang`, `twice`, `early`, `pipe`), on copies of the test fixture; `sigint_debug/sitecustomize.py` makes every
  process of a run dump its stacks on SIGUSR1, which is how the hang was located.
- `parity_debug.py`: why the first parity test differed (the fixture's ICC profile carries its creation time).

Two edit scripts of P4 and P5 (one rewrote the docstring, one the pool's signal handling) are not kept: what they
wrote is in `d086222` and `68d5afc`.
