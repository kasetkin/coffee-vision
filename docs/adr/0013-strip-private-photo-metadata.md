# Every dataset photo is stripped of private metadata

GPS and other private metadata (maker notes, C2PA, serial numbers, owner fields, thumbnails) are removed from every photo in `dataset/`, new and old, losslessly: decoded pixels stay bit-identical, so only file hashes change. A new batch is stripped with `coffeecv.strip_metadata` before `dvc add`.

Amended 2026-11-13 by [ML-28](../ml-0028_ticket_intake_2026_11_13.html) D6: a CR3 keeps its Canon maker notes, which exiftool cannot delete from one. Its six tags that identify the camera, the shot or the region set in its menu (`InternalSerialNumber`, `ImageUniqueID`, `OwnerName`, `TimeZone`, `TimeZoneCity`, `DaylightSavings`) are written blank instead, and the check fails a CR3 in which any of them reads otherwise; its other maker notes, shooting settings and calibration, are noted, not failed, as a RAW's preview is.

Source: [ML-3 D13](../ml-0003_ticket_country_classes.html); [ML-28](../ml-0028_ticket_intake_2026_11_13.html) D6.
