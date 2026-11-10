"""p4_replay.py unchanged, with LiteRT's CPU options overridable from the environment (Pi 5 benchmark):
XNNPACK_FLAGS (int; 4 = TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16), applied only to the graphs whose file name
contains one of FP16_GRAPHS (comma-separated, e.g. "dino"; empty = every graph), and YNNPACK=1."""
import functools
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_edge_litert import compiled_model as cm
import run_litert

base = cm.CpuOptions
flags = int(os.environ["XNNPACK_FLAGS"], 0) if os.environ.get("XNNPACK_FLAGS") else None
only = [g for g in os.environ.get("FP16_GRAPHS", "").split(",") if g]
ynn = os.environ.get("YNNPACK") == "1"
orig_init = run_litert.Graph.__init__

def init(self, path, *a, **k):
    extra = {}
    if flags is not None and (not only or any(g in path.name for g in only)):
        extra["xnnpack_flags"] = flags
    if ynn:
        extra["enable_ynnpack"] = True
    cm.CpuOptions = functools.partial(base, **extra) if extra else base
    print(f"pi_replay: {path.name} CpuOptions extra {extra}", flush=True)
    try:
        orig_init(self, path, *a, **k)
    finally:
        cm.CpuOptions = base

run_litert.Graph.__init__ = init
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("PI_SCRIPT", "p4_replay.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
