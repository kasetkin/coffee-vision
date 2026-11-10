---
status: accepted
---

# The front makes its own, cheaper canonical photo; training's stays as it was

The front's check service canonicalizes a visitor's photo with its own settings, not the store's: LittleCMS's optimized integer transform instead of the float one, libjxl at effort 3 instead of 7, quality 90 instead of 95, and a photo over 18 MP area-averaged to at most 15 MP instead of over 60 MP to 50 MP. Everything else about the canonical photo is the same (Rec.2020, the colour encoding as an enumeration, 16-bit lossy JPEG XL, upright, no metadata), so the inference host's check accepts both and its code does not change. The canonical store, its DVC stages, training and both models keep the store's settings and were not retrained. Every upload is re-encoded at the front, an already canonical one (each of the page's examples) included: the pass-through the front had is gone. `coffeecv/canonical.py` holds both, as `STORE_SETTINGS` (every function's default) and `FRONT_SETTINGS` (passed by the front alone).

The owner chose this for time: with the store's settings a 50 MP photo took about 20 s to canonicalize on the front's 2 vCPU and 18 s to cross to the inference host; with the front's, about 3 s and 4 s, and 2-4 s for every other kind of source (`analysis/ml19/p8/README.md`, `analysis/ops22/README.md`). The alternative, the same settings in the store and a retrain, keeps training and serving on one photo but costs a re-canonicalization of the dataset and a retrain of both models; the owner took the cheaper path knowingly.

## Consequences

- **Amends [ADR 0033](0033-photos-are-canonicalized-once-at-the-front.md):** training and serving no longer read the same canonical photo. The served photo carries less fine detail (q90 at effort 3 multiplies the error in fine detail by 2 to 7, OPS-22 §2, and a 50 MP photo keeps about 30% of its pixels), so the reported metrics describe the store's photos, not the live site's. This is a recorded departure from `CODING_STANDARDS.md`'s train/inference parity, like [ADR 0019](0019-serving-runs-litert-accelerator-per-host.md)'s precision, and unlike it unmeasured: no check of the models' answers on the front's photos beyond the deploy's examples (OPS-22 D5).
- The deploy checks the front's path: `--verify`'s public `/analyze` is held to the staged app's answer on the smoke photo as the front makes it, and before anything is stopped each example's answer on its re-encoded photo is held to its country (OPS-22 D6, D7). The host's own smoke keeps the store's photo.
- A change to `FRONT_SETTINGS` changes the live input and nothing else; a change to `STORE_SETTINGS` is still a re-canonicalization and a retrain (ADR 0033). Bringing the two back together, by re-canonicalizing the store with the front's settings and retraining, is open to a later ticket.

Source: [OPS-22](../ticket_front_fast_canonicalization.html) D1, D3, D4, D5.
