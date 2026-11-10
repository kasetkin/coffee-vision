"""OPS-23: analysis/ml7_gpu/parity.py with _final_mask fitted to today's BeanSegmenter (the photo's sizes are on the
segmenter itself, and _upsample takes numpy), so ML-7's D7 check runs on the Pi's outputs. From the repo root, in the
training environment:

    python analysis/pi5/parity_now.py --refs analysis/ml7_gpu/artifacts/refs.npz --got <replay>.npz \
        --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt --out parity.json
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml7_gpu"))
import parity

def _final_mask(low, input_size, original_size, seg):
    saved = seg.input_size, seg.original_size
    seg.input_size, seg.original_size = tuple(int(v) for v in input_size), tuple(int(v) for v in original_size)
    try:
        return seg._upsample(np.ascontiguousarray(low))
    finally:
        seg.input_size, seg.original_size = saved

parity._final_mask = _final_mask
parity.main()
