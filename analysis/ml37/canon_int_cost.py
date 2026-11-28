"""ML-37: encode and decode time and peak memory of today's, (I) and (H) canonical photos, full scale.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/canon_int_cost.py [--photos 1,2,17,0]

Four photos of D8's sample (by index; default: the two gain-map photos with the highest peaks, the 17 MP Sony and an
iPhone without a map) at full scale. Forms as canon_int_measure.py (today, I, Hs, Hh), intensity target 203 for the
linear forms and today's 255 for today. Each encode and decode runs with 2 threads (the front's setting), one at a
time, in this process. Time: wall clock. Memory: the rise of the process's peak RSS over its RSS just before the
call (Linux: writing 5 to /proc/self/clear_refs resets VmHWM), so the input picture already in memory is not counted;
the encoded bytes and the decoded array are. Prints a table (kept as analysis/ml37/canon_int_cost.txt).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import imagecodecs as ic
import numpy as np

from analysis.ml37 import canon_int_measure as cm
from analysis.ml37 import d8_intensity_target as d8
from coffeecv import jxl_encode as je
from coffeecv import tone_map as tm


def status_mb(field: str) -> float:
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    raise KeyError(field)


def measured(fn):
    with open("/proc/self/clear_refs", "w") as fh:
        fh.write("5")
    before = status_mb("VmRSS")
    t = time.perf_counter()
    r = fn()
    dt = time.perf_counter() - t
    return r, dt, status_mb("VmHWM") - before


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="2,9,17,0")
    args = ap.parse_args()
    photos = d8.sample(20)
    print(f"# libjxl 0.12.0 via imagecodecs {ic.__version__}; {d8.THREADS} threads per call; "
          f"load at start {os.getloadavg()[0]:.2f}; {os.cpu_count()} CPUs")
    print(f"{'photo':48s} {'MP':>5s} {'setting':7s} {'form':5s} {'bytes':>9s} {'bpp':>5s} {'enc s':>6s} "
          f"{'enc MB':>7s} {'dec s':>6s} {'dec MB':>7s}")
    for i in (int(x) for x in args.photos.split(",")):
        p = photos[i]
        light, mapped = d8.light_of(p["file"], 1)
        peak = tm.picture_peak(light.max(axis=-1).ravel().copy())
        sdr = light.copy()
        if peak > 1:
            for y in range(0, sdr.shape[0], 256):
                tm.eetf_in_place(sdr[y:y + 256], peak)
        np.clip(sdr, 0, 1, out=sdr)
        forms = {"today": d8.today_u16(light, peak), "I": np.rint(sdr * 65535).astype(np.uint16),
                 "Hs": sdr.astype(np.float16), "Hh": light.astype(np.float16)}
        sdr = light = None
        h, w = forms["I"].shape[:2]
        label = f"{p['file'].split('/')[-1][:30]} {p['_k'][0][:10]}{' map' if mapped else ''} pk{peak:.1f}"
        for name, q, e in cm.SETTINGS:
            for form, a in forms.items():
                if form == "today":
                    def enc(a=a, q=q, e=e):
                        return je.encode_rec2020(h, w, lambda y, x, ys, xs: a[y:y + ys, x:x + xs], level=q,
                                                 effort=e, numthreads=d8.THREADS, transfer="BT709", intensity_target=255)
                elif form == "I":
                    def enc(a=a, q=q, e=e):
                        return cm.encode_linear_u16(a, level=q, effort=e, intensity_target=203.0)
                else:
                    def enc(a=a, q=q, e=e):
                        return d8.encode(a, level=q, effort=e, intensity_target=203.0, half=True)
                data, te, me = measured(enc)
                _, td, md = measured(lambda data=data: ic.jpegxl_decode(data, numthreads=d8.THREADS))
                print(f"{label:48s} {h * w / 1e6:5.1f} {name:7s} {form:5s} {len(data):9d} "
                      f"{len(data) * 8 / (h * w):5.2f} {te:6.2f} {me:7.0f} {td:6.2f} {md:7.0f}", flush=True)
                data = None
        forms = None
    print(f"load at end {os.getloadavg()[0]:.2f}; process peak RSS {status_mb('VmPeak'):.0f} MB virtual")


if __name__ == "__main__":
    sys.exit(main())
