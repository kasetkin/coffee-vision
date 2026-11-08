---
status: accepted
---

# Example photos are public and in no dataset

The page offers a few example photos (five chosen in OPS-16, four since ML-16 D12), so a visitor without a photo of beans can try the service. They are lossy JPEG XL re-encodes, at full resolution, committed in `webapp/static/examples/` and released with the page, so they stay in public git history. Their originals moved out of `dataset/` into a private DVC folder, `dataset/example_photos/`, that no pipeline reads. An example photo is never in any training or evaluation set: not the classifier's pool, not the segmenter dataset, not the OOD negatives. This is a narrow exception to [ADR 0014](0014-public-repo-private-data.md): the dataset stays private, apart from these files.

We chose git over shipping them from DVC at deploy time, which would have let us withdraw them later, because a release then needs nothing outside the commit. We chose moving the photos over copying them, because a visitor sees the segmenter's mask and the OOD guard's answer as well as the country, and an answer on a photo a model trained on overstates what it does on the visitor's own. We chose lossy over lossless JPEG XL, because the lossless transcode of the 50 MP grinder photo (5.36 MB) was still over the front's upload cap, and a HEIC has no lossless transcode.

## Consequences

- Taking a photo out of a class's pool redraws that class's whole train/val/test split (`split_photos_by_class`). The five left in the same data change as ML-16's new photos, before its retrain, so the splits move once.
- A new example photo has to come from outside every set, or leave them before the next retrain, like these five.
- The re-encode is not the original's pixels. Each example ships only once the model it ships with gives it the original's answer and the owner accepts its mask, and every deploy's `--verify` checks the expected answers again.

Source: [OPS-16](../ticket_example_photos.html) D2, D3, D5; [ML-16](../ticket_next_training_session.html) D3.
