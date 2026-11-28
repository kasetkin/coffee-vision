# ML-37 D14: the segmenter, and whether ML-33's fine-tune comes before or after the linear canonical photo

Research for [ML-37](../../docs/ml-0037_ticket_linear_canonical_photo.html) D14, 2026-11-23, on the local checkout
at `01e6ac1`. A recommendation for the owner, not a decision. The options are the ticket's: (a) `xl0_v7r_s123` kept
and run on the new input, with [ML-33](../../docs/ml-0033_ticket_segmenter_on_pass17.html) fine-tuning after this
ticket on the new canonical photos; (b) ML-33 first, on today's input, then this ticket. The new numbers come from
`d14_mask_sensitivity.py` (output `d14_mask_sensitivity.txt`, per-photo rows in `tmp/ml37/d14/rows.jsonl`). Everything
else is cited from the repo's records and code, the vendored EfficientViT source and its paper.

## Summary

- **The masks barely move under ML-37's input change.** On 20 test photos of the segmenter dataset, the live
  segmenter's masks from a simulated ML-37 input (linear half-float lossy JPEG XL, float32 sRGB read-back, float
  resize) have a median IoU of 0.9999 against today's masks, for positives and for negatives alike. The lowest is
  0.9890, on the photo that already had the lowest predicted IoU. No mask became empty or stopped being empty, and
  no crop box moved more than 2 px (§2). Almost all of the change comes from the float resize alone (B). ML-29 P9
  measured an input change far larger than this one (the gain maps, about 20 levels at the median) and found a
  median IoU of at least 0.999 for every camera (§3).
- **A fine-tune on today's input would not have to be redone after ML-37.** Nothing in a fine-tune is keyed to the
  canonical photo's bytes. There is no embedding cache (ADR 0018). The labels are pinned to the original's sha256
  and a mask PNG in the canonical geometry, which ML-37 keeps. The encoder is frozen, so the embedding shift ML-37
  causes is the same for any decoder (§4). The only mechanical consequence is DVC's: the fine-tune stage depends on
  `coffeecv/canonical.py`, `canonical_store.py` and `data/canonical`, so after ML-37 it would show as changed. It
  would be frozen, as `seg_ml16_finetune` already is.
- **The re-crops and refits are the same in number either way** (§5). Each order needs two re-crops and two paired
  classifier refits, because each adoption is judged on its own change. (a) can save one release, but only if
  ML-33 ships inside ML-37's release.
- **There are concrete reasons to do ML-33 first** (§6). Its labels are done and its blocker is cleared. It fixes a
  known defect: photos with no mask are classified on the whole frame (ML-23's 11 photos). Under (b) it runs on
  today's tested uint8 training path. Under (a) it waits for all of ML-37, whose D6 to D14 are open and which is not
  planned. It would also need `seg_finetune.py` ported to float input. That module is not in ML-37's component list,
  and in float its training photos take 5.1 GB to 10.3 GB of memory instead of 2.6 GB (the D11 note, §6).
- **The reason for (a)**: its end state is a segmenter trained and chosen on the input it serves, with no 8-bit step
  in its training augmentation. That is a strict reading of D3 ("in training and in serving"). Under (b) the
  segmenter in service after ML-37 was trained on 8-bit input, as `xl0_v7r_s123` would be under (a) until ML-33
  ships.
- **Recommendation: (b)**, with moderate confidence. This departs from the ticket's provisional "(a): one fine-tune
  on the input it will serve, not two". The measurement shows (b) does not lead to a second fine-tune.
  - Conditions: ML-33's stage is frozen once it is adopted.
  - ML-37's mask floor (research D13: median IoU per camera ≥ 0.99, photos under 0.9 and gained or lost fallbacks
    opened by the owner) is then checked on ML-33's segmenter instead of `xl0_v7r_s123`.
  - Choose (a) instead if the owner reads D3 as requiring every served model to have been trained on the float
    path, or if ML-37 will be implemented soon enough that ML-33's wait costs little.

## 1. What changes for the segmenter under ML-37

Today `load_canonical` decodes the uint16 BT.709-curve Rec.2020 canonical photo and LittleCMS converts it to uint8
sRGB. `model_io.sam_resize` then resizes with PIL bilinear on uint8, and `sam_input` normalises and pads (ML-37 §2;
`coffeecv/model_io.py:62–78`). After ML-37 the canonical photo is linear Rec.2020 half-float in lossy JPEG XL (D1).
The read-back to sRGB is float32 with no 8-bit step (D2, D3), and the resize is a float resize: D11 recommends (a),
PIL bilinear in mode "F". The masks' geometry does not change (ML-37 §4, ADR 0023, ADR 0034). For photos with
nothing above white, D6's choice does not enter. For the gain-map photos, D6 (a) is recommended so that they "read
close to today" (ML-37 §6).

## 2. Measurement: `xl0_v7r_s123`'s masks under the new input

**Setup** (`d14_mask_sensitivity.py`, docstring):
- 20 photos of the segmenter dataset's test split:
  - 12 positives drawn one per folder;
  - the 4 positives with the lowest predicted IoU in the stored masks, the ones nearest a decision;
  - 4 negatives.
- The cameras include Pixel, OnePlus, iPhone, Sony, Canon and Xperia, plus internet photos.
- The live segmenter with the serving box prompt and multi3, torch at 2 threads.
- Each photo is masked five ways, and every mask is compared with A by IoU:
  - **S**: the stored mask (`data/seg_masks_ml19/xl0_v7r_s123`, 4 threads).
  - **A**: today's path.
  - **B**: A's values as float, PIL "F" resize, which isolates the resizer.
  - **C**: a float32 LittleCMS read-back of today's canonical photo, with no rounding.
  - **D**: C's photo re-encoded the way D1 stores it, then read back like C:
    - linear Rec.2020, clipped at 0 (D7 (a)), float16;
    - lossy JPEG XL at level 95, effort 7, BT2100 primaries with the linear transfer.

  D is a second lossy generation, so it somewhat overstates D1's loss.
- Input differences are given in 8-bit levels, per pixel and on 32×32 block means (ML-29 D27's measure).

**Results** (`d14_mask_sensitivity.txt`):

| IoU against A | S | B (resize) | C (float read-back) | D (ML-37 input) |
|---|---|---|---|---|
| positives (16): median | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| positives: min | 1.0000 | 0.9908 | 0.9904 | 0.9890 |
| positives: < 0.99 / < 0.9 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 |
| negatives (4): median, min | 1.0000, 1.0000 | 0.9999, 0.9997 | 0.9999, 0.9998 | 0.9999, 0.9991 |

- **Empty masks are unchanged.** Every mask that was empty under A is empty under D, and every non-empty one stays
  non-empty: 15 non-empty and 1 empty positive (an internet photo), 2 non-empty and 2 empty negatives. The largest
  shift of a crop box under D is 2 px.
- **The only positive under 0.99 under D** is `segmenter_positives/PXL_20260924_104055590`, at 0.9890. It is one of
  the 4 "hard" photos, with a stored predicted IoU among the lowest, and it scores 0.9908 under B alone.
- **Input difference, D against A:**
  - per-pixel mean: median 0.32 levels (max 0.74);
  - 32×32 block-mean difference: median 0.040 levels (max 0.094);
  - largest single block: 1.24 levels.

  C against A: per-pixel 0.24 and block mean 0.007, which is the lost rounding. C rounded to 8 bits differs from A in
  at most 503 pixels of a photo.
- **Thread count.** S against A is 1.0000 on all 20: the stored 4-thread masks equal the 2-thread ones here.
- **Is 20 photos enough?** The full per-camera comparison on every dataset photo is already a required step before
  the retrain, whatever D14 decides: ML-37 R1 and the floor in the D13 note. D14 only needs to know whether the
  segmenter is sensitive enough to change the order of the work. With a median of 0.9999 and the hardest photos at
  0.989 to 0.998, it is not. The rest of the dataset was not run.

## 3. ML-29 P9: the larger change the same segmenter already absorbed

When gain maps changed the canonical photos of 1,684 photos, by "about 20 levels at the median" (ML-29 P8, as quoted
in ML-33 §2), P9 compared the live segmenter's masks on the old and new photos:
- median IoU of 0.999 or 1.000 for every camera;
- 11 photos under 0.9;
- no verdict of the OOD probe changed (`analysis/ml29/p9_report.txt`).

`xl0_v7r_s123` was fine-tuned before gain maps and has served the gain-map photos since (ADR 0041, consequences). The
change in §2 is about two orders of magnitude smaller in block means (0.04 against about 20 levels; the two
measures are not identical). The gain-map photos themselves differ only by D6's choice. Under D6 (a) they "read close
to today". Under D6 (b), 0.3% of bean pixels in the median gain-map photo clip (ML-34 experiment A, quoted in ML-37
§2). §2 did not sample that case.

## 4. Would a fine-tune on today's input have to be redone after ML-37?

Mechanically, no:
- **No embedding cache.** "Nothing is cached, and nothing augmented is written" (`coffeecv/seg_finetune.py`,
  docstring; ADR 0018). The ML-2 and ML-5 embedding caches left `dvc.yaml` on 2026-10-21. The validation embeddings
  are recomputed in memory on every run (`Selection`, `seg_finetune.py:309–322`).
- **Labels pinned to originals, not to canonical bytes.** `data/ml5_labels/pass17/labels.csv` keys each label by
  `photo_sha256`, which is the original's. `canonical_store.canonical_for` checks `sha256_file(original)`
  (`canonical_store.py:20–34`). The mask is a PNG with its own `mask_sha256`. The only check against the canonical
  photo is its shape (`seg_finetune.py:264–269`), and ML-37 keeps the geometry. So pass17 serves either order, and
  nothing needs to be relabelled. The pictures the owner judged move by about 0.3 level per pixel (§2).
- **Frozen encoder.** Only the mask decoder trains (ADR 0015; `seg_finetune.py` docstring, "Training"). Both of
  ML-33's D2 recipes start from the same `efficientvit_sam_xl0.pt` encoder (`params.yaml` `seg_ft.weights`). So the
  change ML-37 makes to the embeddings is identical for `xl0_v7r_s123` and for any ML-33 decoder. Only the decoder's
  response to that shift could differ.
- **Augmentation far wider than ML-37's change.** The fine-tune trains through `seg_ft.augment`: ColorJitter
  brightness ±0.2, RandomGamma 80–120, HueSaturationValue, JPEG ImageCompression at quality 60–95, blur and noise
  (`params.yaml:186–199`, `augment` groups at 155–160). These perturbations are orders of magnitude larger than
  0.04-level block means. That is the reason an ML-33 decoder should respond to ML-37's input no worse than
  `xl0_v7r_s123` does. It is an inference: no ML-33 model exists to measure.
- **What DVC does.** `seg_ml16_finetune`'s deps include `coffeecv/canonical.py`, `coffeecv/canonical_store.py` and
  `data/canonical` (`dvc.yaml:449–483`). ML-37 changes all three, so ML-33's stage would show as changed. The
  precedent is to freeze it: `seg_ml16_finetune` is `frozen: true`, with the note "data/canonical has gained the 87
  photos … since". The model's card would then record a pre-ML-37 input, as ADR 0041 records for `xl0_v7r_s123`.

## 5. How ML-37's and ML-33's re-crops and classifier refits combine

The two changes and what each needs:
- **ML-37:** re-canonicalize the store, re-crop the pool, refit the classifier head and the OOD probe, and judge
  the paired three-seed comparison against D13's bar. One release of store, front, host and models (ML-37
  "Production impact", §7).
- **ML-33:** a new decoder re-crops the pool, and the classifier is refitted and compared ("Changing the segmenter"
  in `docs/agents/experiments.md`, steps 1–3: re-crop with every fallback opened, a recorded 3-seed retrain, LiteRT
  conversion). Then a release, as in ML-16 (ML-33 "Production impact").

| | (a) ML-37, then ML-33 | (b) ML-33, then ML-37 |
|---|---|---|
| Re-crops | 2: ML-37's with `xl0_v7r_s123` on the new input, then ML-33's | 2: ML-33's on today's input, then ML-37's with ML-33's decoder |
| Classifier refits and comparisons | 2: each change judged alone | 2: each change judged alone |
| Fine-tunes | 1, on float input | 1, on today's input; not redone (§4) |
| Releases | 2, or 1 if ML-33 ships inside ML-37's release (which then waits for it) | 2: ML-33 needs no front change, ML-37 does |
| ML-37's mask floor (D13 note) checked on | `xl0_v7r_s123` | ML-33's segmenter |

Each comparison has to keep the other change fixed, or D13's bar measures two things at once (the D13 note, §8:
"D14 (a) keeps `xl0_v7r_s123` … so only its input changes"). So neither order merges the two refits into one. The
owner's crop review (experiments.md step 1) happens at each re-crop either way.

## 6. Reasons to do ML-33 first

- **It is ready.** pass17 is labelled, in DVC and pushed (ML-33 §2, §7). Its blocker, ML-29 P10's runs on the
  pool, has passed (ML-29 is closed, ML-37 D5). Its D1 ("now, or after ML-29 P11") is settled by ADR 0041: gain maps
  were adopted, and ADR 0041 says ML-33 "reads the new ones".
- **It fixes a defect users see.** `xl0_v7r_s123` left 11 photos without a mask in ML-16's re-crop, six of them
  green-bean photos, and the Vietnam example left the page (ML-33 §2, ML-23). Such photos are classified on the whole
  frame. Under (a) that waits for all of ML-37: D6–D14 open, not planned, one large release (ML-37 status, §7).
- **Its training path works today, and the float port has costs nobody has planned for.**
  - `seg_finetune.py` and `augment.py` are not in ML-37's component list.
  - `sam_resize` raises on float input. `_to_chw_normalised` would silently scale a [0, 1] picture by 1/255.
  - HueSaturationValue and ImageCompression round to 8 bits inside, which affects 51% of training photos.
  - TrainSet's 1,091 photos take 2.6 GB as uint8, 5.1 GB as float16 and 10.3 GB as float32 (the D11 note, §6). This
    machine has 13.8 GB in total.

  Under (a) these have to be solved before ML-33 can run. Under (b) they move to the *next* segmenter fine-tune
  after ML-37.
- **Against (b):** after ML-37 ships, the served segmenter is one trained with 8-bit input and 8-bit steps in its
  augmentation. D3 names "the augmentation" among the things that may not be 8-bit (ML-37 §1). Under (a) the same
  holds for `xl0_v7r_s123` until ML-33 ships, but (a)'s end state meets D3 fully. §2 and §4 bound the practical cost
  of the difference at an IoU of about 1e-4. Whether D3 is read as a property of the served pipeline or also of
  every served model's training is the owner's call.
- **One machine.** A fine-tune and ML-37's re-canonicalization and refits cannot run at once without halving each
  other (ML-33 R3). Either order runs them one after the other. (b) uses the machine on ML-33 while ML-37's open
  decisions are settled.

## 7. EfficientViT-SAM's own preprocessing and training data, as context

- **Training data.** "We begin with the knowledge distillation from the SAM-ViT-H image encoder to EfficientViT.
  Subsequently, we conduct end-to-end training on the SA-1B dataset" (Zhang, Cai, Han, *EfficientViT-SAM:
  Accelerated Segment Anything Model Without Accuracy Loss*, arXiv:2402.05008 v2, abstract; the same text is in
  `third_party/efficientvit/applications/efficientvit_sam/README.md:10`).
- **The training loader** reads SA-1B's JPEGs (`sa_<n>.jpg`, `skimage.io.imread`, so 8-bit sRGB content) and casts
  them to float32 at 0–255 scale (`efficientvit/samcore/data_provider/sam.py:47–73`). It resizes in float with
  `F.interpolate(..., mode="bilinear", antialias=True)` (`samcore/data_provider/utils.py:124`) and normalises with
  ImageNet mean and std in 0–255 units (`utils.py:177`).
- **The inference transform** is `SamResize`, documented to expect uint8 and resized through PIL, then `ToTensor`,
  `Normalize` (the same mean and std divided by 255) and `SamPad` (`efficientvit/models/efficientvit/sam.py:66–90,
  208–217`). The vendored copy is upstream `de7d7733` (commit `ce4e753`).
- **What this means for D14.** The encoder only ever saw pixels that came from 8-bit JPEGs, but in training it read
  them through a float, antialiased resize. A float sRGB input in [0, 1] with no 8-bit rounding is a change below
  the quantization step. It stays inside the encoder's training distribution, and the encoder cannot depend on the
  8-bit grid. The bigger difference is the resize filter: in §2, B alone accounts for nearly all of D's mask change.
  The encoder weights and the frozen-encoder design are identical under (a) and (b), so nothing here favours either
  order.

## 8. What stays unverified

- **No ML-33 model exists.** Its sensitivity to ML-37's input is inferred from the shared frozen encoder and the
  augmentation's range (§4), not measured.
- **20 of the dataset's photos**, none with light above white. D6's effect on the gain-map photos is not in §2.
  The fallback threshold (`seg_min_area_frac` 0.0285) was not approached by any sampled mask except the empty ones.
- **D is a simulation, not ML-37's implementation.**
  - It uses the store's setting (95, effort 7), not the front's (90, effort 3).
  - It uses a LittleCMS read-back rather than D9 (a)'s numpy maths.
  - It uses no D8 intensity target.
  - It is a second lossy generation, so it is pessimistic on the coding.
- **The full per-camera comparison** (ML-37 R1 and the D13 note's floor) is still to be run on every photo, under
  either option.

## Sources

- `analysis/ml37/d14_mask_sensitivity.py`, `.txt`; `tmp/ml37/d14/rows.jsonl` (this measurement).
- `docs/ml-0037_ticket_linear_canonical_photo.html` (§1–§7, D1–D14); `docs/ml-0033_ticket_segmenter_on_pass17.html`.
- `analysis/ml29/p9_report.txt`; ADR 0011 (superseded by 0015), 0015, 0018, 0029, 0041.
- `analysis/ml37/research_d13_adoption_bar.md` (§8, the mask floor); `analysis/ml37/research_d11_float_resize_and_augment.md` (§5, §6).
- `coffeecv/seg_finetune.py` (docstring, 250–322), `coffeecv/canonical_store.py:20–39`, `coffeecv/model_io.py:62–78`,
  `coffeecv/seg_dataset.py`; `dvc.yaml` `seg_ml16_finetune`, `seg_ml16_refine`; `params.yaml` `seg_ft`, `augment`;
  `labels/ml5/pass17.*`, `data/ml5_labels/pass17/labels.csv`.
- `docs/agents/experiments.md` ("Changing the segmenter", "Reproducibility limits").
- `third_party/efficientvit` (vendored `de7d7733`): `efficientvit/models/efficientvit/sam.py`,
  `efficientvit/samcore/data_provider/sam.py`, `utils.py`, `applications/efficientvit_sam/README.md`.
- Zhang, Cai, Han, arXiv:2402.05008 (v2, 2024-05-16), abstract.
