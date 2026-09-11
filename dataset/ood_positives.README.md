# OOD positives

Genuine coffee-bean photos the shipped model has **never seen**. These are the population a
content guard must *not* refuse, and they exist because the checkpoint's own held-out split
turned out to be far too easy to stand in for them: photos from sessions the model trained
on score a median 1.02 on the shipped metric, while these score 1.21 and reach 1.57 — well
into the range the negatives occupy.

Used only for evaluating guard metrics (`coffeecv/ood_eval.py`), never for training.

## Provenance and checks (2026-09-10)

- **Hash-checked against the whole training tree**: SHA-256 of every file compared against
  all 938 raw photos under `dataset/*/class_*__*`, plus a capture-timestamp comparison on the
  `PXL_<date>_<ms>` stems (millisecond precision, so a re-export of the same shot would
  collide even though its bytes differ). Zero matches on either — all genuinely new.
- **EXIF stripped in place.** Every file carried GPS coordinates. Stripped by rebuilding each
  image from raw pixel data, so nothing survives rather than only the tags we thought to
  delete. Orientation was 1 throughout, so no rotation was needed.
- Contents verified by eye: all genuinely show beans (one of green/unroasted beans in a
  container, the rest tight macro shots of roasted beans).

## The two tags are not interchangeable

- **`user_beans_independent`** (7 photos) — shot on days with no training session at all
  (2026-08-23, -24, -29, -31, 2026-09-03). These are the trustworthy ones.
- **`user_beans_same_day`** (7 photos) — six shot on 2026-08-09 *35–55 minutes before* the
  `2026-08-09__pixel_cam` session began, one 5.5 h before `2026-08-30__pixel`. Same beans,
  room and lighting almost certainly. Byte-distinct from training data but **not independent
  of it**, and they score visibly lower (median 1.12 vs 1.27) — which is the point: pooling
  them with the independent photos would flatter every metric measured here.

`split` (`dev`/`holdout`) was assigned deliberately rather than randomly. With only seven
independent-day photos, a random draw could have put them all on one side and left the other
with nothing load-bearing in it.

## Sibling batch

`dataset/ood_positives_internet/` holds 12 Wikimedia-sourced photos of coffee beans under the
`internet_beans` tag — also genuine positives, but a *different* population (studio lighting,
varied roasts and framing), so they are a separate directory and a separate condition rather
than being mixed in here. Sparse arrangements (a handful of beans on a white surface) and
product/scene shots were rejected during vetting: the question is whether a guard refuses a
photo *of beans*, and a photo of four beans on a table is a different question.
