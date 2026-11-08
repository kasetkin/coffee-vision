"""ML-19: the share of pixels outside sRGB in the Display-P3-tagged photos, by matrix maths (P3 D65 to linear sRGB).
Reads the exiftool CSV the README names; prints totals and the 18 photos with the most pixels outside."""
import csv

import numpy as np
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()
P3_TO_XYZ = np.array([[0.4865709, 0.2656677, 0.1982173], [0.2289746, 0.6917385, 0.0792869],
                      [0.0, 0.0451134, 1.0439444]])
SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
P3_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ) @ P3_TO_XYZ


def linear(v: np.ndarray) -> np.ndarray:
    v = v / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


rows = [r for r in csv.DictReader(open("/tmp/ml16/colour.csv")) if "P3" in r.get("ProfileDescription", "")]
total = outside = far_outside = 0
share, names = [], []
for r in rows:
    try:
        im = Image.open(r["SourceFile"])
        im.draft("RGB", (800, 800))
        im = im.convert("RGB")
        im.thumbnail((800, 800))
    except OSError:
        continue
    a = linear(np.asarray(im, dtype=np.float64).reshape(-1, 3)) @ P3_TO_SRGB.T
    out = ((a < -0.002) | (a > 1.002)).any(1)
    total += len(a)
    outside += out.sum()
    far_outside += ((a < -0.02) | (a > 1.02)).any(1).sum()
    share.append(out.mean())
    names.append(r["SourceFile"])
share = np.array(share)
print(f"{len(share)} P3 photos; pixels outside sRGB: {outside / total * 100:.2f}% "
      f"(beyond 2%: {far_outside / total * 100:.2f}%); photos with >1% out: {(share > 0.01).sum()}, "
      f">10%: {(share > 0.10).sum()}, max {share.max() * 100:.1f}%")
for i in np.argsort(-share)[:18]:
    print(f"{share[i] * 100:5.1f}%  {names[i]}")
