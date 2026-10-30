"""ML-14 P3: pixel_digest's peak memory above the imports, per pixel, for one photo; DECODE_BYTES_PER_PIXEL comes
from it. Run once per photo, each in its own process (ru_maxrss is the process's peak).

    python analysis/ml14/factor.py PHOTO

2026-10-30, on photos of dataset/segmenter_positives (a 12, a 50 and the 200 MP JPEG, the iPhone 16 HEIC and a
49 MP HEIC, a Pixel 9 Pro DNG) and on PNG and grey JPEG copies of the 50 MP JPEG made in /tmp: JPEG 10.0 (RGB or
grey), PNG 10.0 (RGB), 12.0 (RGBA), HEIC 11.0, DNG 13.0 bytes per pixel, the same from 12 to 200 MP.
"""
import resource
import sys
import time
from pathlib import Path

import pillow_heif
import rawpy
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coffeecv import strip_metadata as sm  # noqa: E402  (also sets the dataset's pixel limit)


def rss_kb() -> int:
    return int(open("/proc/self/statm").read().split()[1]) * 4


if __name__ == "__main__":
    path = Path(sys.argv[1])
    ext = path.suffix.lower()
    if ext in sm.RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            s = raw.sizes
            px, extra = s.width * s.height, f"raw {s.raw_width}x{s.raw_height}, output {s.width}x{s.height}"
    elif ext in sm.HEIF_EXTENSIONS:
        heif = pillow_heif.open_heif(str(path))
        px = sum(i.size[0] * i.size[1] for i in heif)
        extra = f"{len(heif)} top-level {[i.size for i in heif]}"
    else:
        with Image.open(path) as im:
            px, extra = im.size[0] * im.size[1], f"{im.size} {im.mode}"
    base = rss_kb()
    t = time.time()
    sm.pixel_digest(path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"{path.name}\t{px / 1e6:.1f} MP\t{(peak - base) * 1024 / px:.2f} B/px\t"
          f"{(peak - base) / 1e6:.2f} GB above imports\t{time.time() - t:.1f} s\t{extra}")
