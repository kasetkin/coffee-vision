# ML-3: country classes and the new photos

Plan: `docs/plan_country_classes.html`; ticket: `docs/ticket_country_classes.html` (decisions D1-D15, Activity).

- `new_photos.py`, `premerge_check.py`: the 288 new photos, fixed on 2026-10-05 (P2), and the checks before
  they joined `dataset/` (P4). `labels/ml3/new_photos.csv` is the list.
- `mask_review.py`: the owner's review of ft_s123's masks on the new photos (P5, D8) and the photos the pools
  leave out (`labels/ml3/pool_exclude.csv`).
- `printout.py` → `printout.txt`: the D10 printout of the country fits exp261-263 (P7), from the archives.
  The owner picked seed 123 (exp262) from it on 2026-10-06.
- `ood_probe_country_s123.log`: `fit_frozen_head --ship 262 --seed-exps 261 262 263`, `build_ood_reference
  --store-knn-embeddings` and `fit_ood_probe --training-days 2026-09-11 --verify` on the shipped
  `models/allrigs_dino3b16_seg_country_s123.pt`, run on the VM (P8); progress lines dropped. In it, the
  per-day line for the other days reads "on other days (2026-09-11)": that label was wrong and has since been
  fixed. The group is the photos *not* shot on 2026-09-11.

**The OOD guard is not freshly validated (D11).** The probe is refitted on the new head, but its holdout
was read before, in ML-1 and ML-2. Running `--verify` on it once more is a regression check against the
live probe's figures, not an estimate of how the guard does on photos it has never met. The comparison is
also not like for like:

- The holdout now has 29 genuine photos, not 26. The owner added three, shot on 2026-10-05.
- 12 of the 22 `user_beans_independent` holdout photos were shot on 2026-09-11. Three of the new training
  sessions were shot that day, and same-day photos flatter a probe (the README's rule: "The two tags are not
  interchangeable"). `--verify` prints those 12 and the other 10 apart. 2026-09-11 is the only training day
  among the `user_beans_independent` dates.

Fresh validation needs new genuine photos and negatives that the probe has never been scored on. The owner
adds them and refits the probe once more, outside this ticket (D11).
