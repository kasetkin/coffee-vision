---
status: accepted
---

# The canonical store, the crops and the test masks process only new photos, and dvc.lock says which

The `canonicalize@<folder>` and `segcrop@<session>` stages keep their outputs between runs (`persist: true`). A run reads its stage's entry in `dvc.lock` (`coffeecv.incremental`). If the stage's command, code, model, params and output listing are all as recorded and only its photo folder changed, it processes only the photos that are new or changed since the recorded run. It keeps every other photo's outputs that are on disk as recorded, and deletes the outputs of photos that are gone. Anything else, or `--full`, rebuilds the folder from nothing, as before. A photo's canonical photo and crop depend on that photo alone, so the bytes are the same either way, and tests hold an incremental run equal to a full rebuild.

We chose this because each intake re-ran every photo of the folder it touched. Adding 17 photos to `random_date_raccoon` (411 photos) took about 45 minutes, to confirm that 394 outputs had not changed. We chose `dvc.lock` as the record over a file of our own beside the outputs, because DVC already holds every input's md5 and every output's md5 as of the last run, so a second record would only be something to keep equal to it. We chose to compare every recorded input over trusting DVC's reason for the re-run, because the script runs inside `dvc repro`, which holds the repository's lock, and DVC does not tell a stage why it runs.

Amended 2026-11-11 by [ML-26](../ticket_folder_stages_only_new_photos.html) D2: `seg_ml19_predict@<model>` too. Its changing inputs are the segmenter dataset's file and the whole canonical store, whose listing DVC keeps nowhere, so it keeps a photo's mask by the photo's path, sha256, source and split in its own recorded `index.csv`. Its deps now name what a canonical photo's bytes depend on besides the original (the canonicalize stages' code), so a canonical photo is a function of its original's sha256.

## Consequences

- A change to anything a stage names in `dvc.yaml` rebuilds every folder of that stage, as before. A change to something it does not name, a module the code imports, is no longer caught by a rebuild that happens anyway: such a module goes into the stage's deps, as the stage's own code does.
- A change to the scripts that changes only which photos they skip is recorded with `dvc commit` and not re-run, as ML-6 P6 recorded a change no output depended on. The commit is made with the photo folders as `dvc.lock` records them.
- `--full` is the check that the store and the crops come out the same from nothing. It is run when the incremental path itself changes.

Source: [ML-26](../ticket_folder_stages_only_new_photos.html) D1, D2.
