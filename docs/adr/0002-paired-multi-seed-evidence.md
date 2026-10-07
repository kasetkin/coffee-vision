# A change is adopted on paired multi-seed evidence, read by the owner

A training or pipeline change runs at seeds 42, 123 and 7 and is compared with the baseline at the same seeds, judged on whether the paired deltas agree in sign, never on one seed: seed-to-seed spread here is larger than the effects being measured, and single-seed wins have reversed on a second seed (`EXPERIMENTS_LOG.md`, Phases 7, 8 and 12). There is no automatic pass/fail rule: the work prints the paired results (per seed, val and test macro-F1 for both arms, the deltas, their mean and range, per-class F1) and stops, and the owner decides.

Sources: owner decision, 2026-08-13 (paired seeds); [ML-2 D24](../ticket_segmentation_mask.html) and [ML-3 D10](../ticket_country_classes.html) (no numeric gate).
