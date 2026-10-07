# Coding standards

Judgement rules for reviewing a change here (`/code-review` reads this file). Each comes from a failure this repo has had. Decisions that bind the design live in `docs/adr/`; this file covers how code is written.

## Pipeline

- **Train/inference parity.** Anything the model depends on (scale estimate, crop, mask, normalization) runs through the same code in training and in serving (`coffeecv/infer.py`, `webapp/`). An accurate method offline with a cheap one live is rejected, because it inflates every reported metric. A change to the training path updates the serving path in the same change: a stale entry point is worse than a missing one (`infer.py` once sized patches from a CLI scale ratio days after training had stopped doing so, and still answered confidently).
- **Invariance over normalization.** A capture-specific attribute (zoom, EXIF orientation, white balance, exposure) is absorbed by augmentation, never corrected per capture folder. A per-capture measurement may size an augmentation range, nothing more.
- **CV output is checked at full resolution.** A crop, mask or detection step is trusted only after a handful of its outputs, clean-marked ones included, were opened at full size with the overlay drawn; a thumbnail sheet hid a real rim leak. A window- or block-based image heuristic scales its window with the measured bean pitch, or it inverts at another magnification.
- **A tuned heuristic names what its validation set lacks.** Next to a threshold fitted on known captures, say which real-world variation (distance, lighting, device) the set does not cover, and keep an override for when it is wrong in production (`skip_crop` in the webapp).

## Config and CLI

- **An omitted flag inherits.** A CLI flag that overrides config defaults to `None`, or to the adopted recipe (`run_all_rigs.ADOPTED`, held equal to `params.yaml` by `tests/test_run_all_rigs.py`), never to a concrete value: a `0.0` default once switched off MixStyle in a "paired" sweep and cost 6.5 hours.
- **Input lists live in one committed file** that every consumer reads (`webapp/deploy/fixtures.txt`), not inline in scripts, duplicated in tests, or passed as argv or env strings.
- **Deploy files are templates.** Environment-specific values (user, paths, domain) are `$VAR` placeholders in `*.template` files with a default, rendered by `webapp/deploy/render_template.py`; a literal value is a portability bug even when only one value has ever existed.

## Proving a change

- **Run the real entry point before a long run.** A change behind a multi-hour run is run first at toy scale (`epochs: 1`, 2 patches per class) through the command that will actually run, on every branch of any conditional it touches. A smoke test of a helper says nothing about the CLI that calls it: two flags once passed every helper test and were never forwarded by `main()`.
- **A rewrite is checked against the original.** Logic moved into a new script or a new shape is run against the old one's result on the same input before it is trusted.
- **A widened invariant is tested on its old cases too**, not only on the case it was widened for.
- **String-replacement edits assert they applied.** Three bugs here came from a replacement on text that had drifted; read what follows an edited `if` block, since an inserted `else:` captures the lines after it.
- **A sandboxed service's allowlist moves with its code.** New file, directory or socket access by the webapp is added to the unit template's `ReadWritePaths` (and checked against `ProtectSystem`/`ProtectHome`) in the same change; a new log directory once crash-looped production, and no local test could see it.

## Candidates for automated checks

Mechanical enough to become checks rather than review rules, not yet built:

- A literal user name or absolute home path in `webapp/deploy/*.template`.
- An argparse override flag in `coffeecv/` whose default is a concrete number.
