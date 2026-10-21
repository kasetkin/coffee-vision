---
status: accepted
---

# The segmenter dataset groups shots taken within 3 s, not 120 s

The segmenter dataset (`labels/ml5/seg_dataset.yaml`) keeps one group in one split: byte-identical files, photos from one source URL, the owner's split units, and shots of one subject on one clock chained while each is within `seg_dataset.GROUP_SECONDS` of the last. That limit was 120 s (ticket ML-5, D13) and is now 3 s. Within one capture session the owner deliberately varies each shot (container, framing, pile), so two photos a minute apart are different photos, not near-copies; 3 s still catches what is a near-copy: a double press, or two frames of one press.

- **Considered**: dropping the time rule entirely. Rejected because a double press gives two near-identical frames that would leak a test photo into training.
- **Append-only**: every photo listed before 2026-10-21 keeps the split and group name it was given under 120 s. Only photos added from then on are grouped at 3 s. Those groups are coarser than 3 s would make them, which only merges photos within one split and never leaks across splits.
- **Why now**: at 120 s, 99 of the 1,178 pool photos the draw could pick would chain two listed groups in different splits, and the builder refuses such a photo rather than move a listed one. The append of 2026-10-21 hit one. At 3 s none does, so ML-6 D14's skip rule is not needed.
- **The OOD probe** (`fit_ood_probe.probe_beans`) uses the same groups: from its next fit, it drops a bean photo only when that photo is a test photo, a copy of one, tied to one, or shot within 3 s of one.

Source: the owner, 2026-10-21; amends ML-5 D13 ([ticket](../ticket_segmenter_dataset.html)).
