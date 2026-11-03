"""ML-19 P3: the 7 photos over 60 MP (D4's measurement, analysis/ml19/downscale_answers.json) from their canonical
photos, on the live models.

    python analysis/ml19/p3/seven.py OUT_DIR          # writes OUT_DIR/<stem>.png, each canonical photo's sRGB
    PYTHONPATH=. python analysis/ml19/run_infer.py --images-dir OUT_DIR \
        --checkpoint models/dino3b16_segxl0v6_country_aug8_s42.pt --out analysis/ml19/p3/seven_infer.json

The PNG is lossless, so coffeecv.infer reads exactly what coffeecv.canonical.load_canonical gives: what the models
will see from P6 (training) and P9 (serving) on.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import imagecodecs as ic  # noqa: E402
from PIL import Image  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402
from coffeecv.photo_io import DATASET_MAX_PIXELS  # noqa: E402

Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS   # the owner's 200 MP photos, as the dataset tools read them

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for d in json.loads((REPO / "analysis/ml19/downscale_answers.json").read_text()):
    path = next((REPO / "dataset").rglob(d["photo"]))
    t = time.perf_counter()
    canon = cn.canonicalize_file(path)
    srgb = cn.load_canonical(canon)
    (out / f"{path.stem}.png").write_bytes(ic.png_encode(srgb, level=1))
    print(json.dumps({"photo": d["photo"], "canonical_bytes": len(canon), "canonical_sha256":
                      hashlib.sha256(canon).hexdigest(), "size": [srgb.shape[1], srgb.shape[0]],
                      "seconds": round(time.perf_counter() - t, 1)}), flush=True)
