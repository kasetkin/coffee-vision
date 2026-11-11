"""Pack OUT/patches/<sha>.npy (float32, serving's patches) as uint8 .npz for the Pi, checking that the float32
rebuild (u / 255 - mean) / std is bitwise the original; manifest.jsonl: sha256, role, the patch file's sha256."""

import hashlib
import io
import json
import sys
from pathlib import Path
import numpy as np

out, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
rows = [json.loads(ln) for ln in (out / "patches.jsonl").read_text().splitlines()]
done = {}
for r in rows:
    if "error" in r:
        continue
    done[r["sha256"]] = r
with (dst / "manifest.jsonl").open("w") as m:
    for sha, r in done.items():
        data = (out / "patches" / f"{sha}.npy").read_bytes()
        P = np.load(io.BytesIO(data))
        u = np.rint((P.astype(np.float64) * STD + MEAN) * 255).astype(np.uint8)
        if not np.array_equal((u.astype(np.float32) / np.float32(255) - MEAN) / STD, P):
            raise SystemExit(f"{sha}: uint8 rebuild is not bitwise")
        np.savez_compressed(dst / f"{sha}.npz", u=u)
        m.write(
            json.dumps(
                {
                    "sha256": sha,
                    "role": r["role"],
                    "patch_sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            + "\n"
        )
print(len(done), "photos packed")
