# Every dataset photo is stripped of private metadata

GPS and other private metadata (maker notes, C2PA, serial numbers, owner fields, thumbnails) are removed from every photo in `dataset/`, new and old, losslessly: decoded pixels stay bit-identical, so only file hashes change. A new batch is stripped with `coffeecv.strip_metadata` before `dvc add`.

Source: [ML-3 D13](../ticket_country_classes.html).
