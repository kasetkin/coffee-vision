---
status: accepted
---

# A label gets seven rounds, recorded per session

On the labelling page (`coffeecv.seg_review label`), a declined mask is redrawn from the owner's points and fence and judged again. A photo still declined in the last round is dropped, with the reason logged. That limit was three rounds (ticket ML-5, D14; [ADR 0015](0015-xl0-segmenter-owner-judged-labels.md)) and is now seven (`seg_review.ROUNDS_DEFAULT`). With points and a fence, a mask often needs more than three redraws to come right: on its third decline, a photo was dropped although the next redraw might have fixed it.

- **Per session, not global**: each session file (`labels/ml5/<session>.session.json`) records its `rounds` when the session starts, and a resume with another number is refused. A file without the field reads as 3 (`LEGACY_ROUNDS`), so pass1-pass4 keep the meaning they were labelled under: pass4's drop stays a drop, not a photo waiting for round 4.
- **pass5** is the first session with seven. It had no verdicts of its own when its file gained `"rounds": 7`. A photo pass4 dropped after three rounds starts again at round 1 there, like any photo not carried.
- **Considered**: an unlimited number of rounds. Rejected because a mask the model cannot draw would never leave the queue. Seven was the owner's pick over five.

Source: the owner, 2026-10-24; amends ML-5 D14 ([ticket](../ticket_segmenter_dataset.html)) and the round limit in ADR 0015.
