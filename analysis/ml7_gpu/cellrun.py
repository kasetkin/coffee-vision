"""ML-7 P2-P4: what every runtime cell records beside its outputs. Standard library only, so each runtime's venv
can import it as it is (run_ort.py, run_litert.py).

  Sampler      every 0.1 s, from a child process: the mean CPU clock, the package temperature, MemAvailable, and the iGPU's i915
               counters under /sys/class/drm/card*/gt/gt0/: its actual clock (rps_act_freq_mhz), its RC6 idle time
               (rc6_residency_ms) and the PL1 power-limit throttle flag (throttle_reason_pl1). The deploy user can
               read all of them without root (plan P2, read on yappy 2026-10-13).
  loaded()     what this process has mapped and opened: the GPU libraries from /proc/self/maps, the DRM render
               nodes from /proc/self/fd. The same evidence strace would give for "the device was opened".
  gpu_proof    plan P2's proof that a cell computed on the iGPU: the path's own driver loaded, a render node open,
               llvmpipe (Mesa's CPU Vulkan) not loaded, and the counters showing the GPU busy while the timed calls
               ran: a non-zero clock and RC6 idle time rising by at most IDLE_MAX of the wall time.
  timing       cold (session creation plus the first call, kernel compilation included) and warm calls.
"""
from __future__ import annotations

import os
import resource
import statistics
import time
from pathlib import Path
from typing import NamedTuple

IDLE_MAX = 0.5          # "RC6 rising by well under the wall time": the GPU idle for at most half the timed run

DRIVERS = {"vulkan": ("libvulkan_intel.so",), "opencl": ("libigdrcl.so",)}
LLVMPIPE = "libvulkan_lvp.so"
WATCHED = ("libvulkan_intel.so", "libvulkan_lvp.so", "libvulkan.so", "libigdrcl.so", "libOpenCL.so",
           "libLiteRtWebGpuAccelerator.so", "libLiteRtGpuAccelerator.so", "libLiteRtOpenClAccelerator.so",
           "libonnxruntime_providers_webgpu.so", "libopenvino_intel_gpu_plugin.so", "libopenvino_intel_cpu_plugin.so")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _int(path: Path | None) -> int | None:
    v = _read(path) if path else None
    return int(v) if v not in (None, "") else None


class Sample(NamedTuple):
    t: float
    cpu_mhz: float | None
    temp_c: float | None
    mem_available_mb: float | None
    gpu_mhz: int | None
    rc6_ms: int | None
    pl1: int | None


class Probe:
    """The counters' files, found once; read() is one Sample."""

    def __init__(self):
        self._freqs = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
        self._temp = next((z / "temp" for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
                           if _read(z / "type") == "x86_pkg_temp"), None)
        gt = next(iter(sorted(Path("/sys/class/drm").glob("card*/gt/gt0"))), None)
        self.gt = str(gt) if gt else None
        self._gpu = {k: (gt / f) if gt else None for k, f in
                     (("mhz", "rps_act_freq_mhz"), ("rc6", "rc6_residency_ms"), ("pl1", "throttle_reason_pl1"))}

    def rc6_ms(self) -> int | None:
        return _int(self._gpu["rc6"])

    def read(self) -> Sample:
        freqs = [int(v) / 1e3 for f in self._freqs if (v := _read(f)) is not None]
        temp = _int(self._temp)
        mem = next((round(int(ln.split()[1]) / 1024) for ln in (_read(Path("/proc/meminfo")) or "").splitlines()
                    if ln.startswith("MemAvailable:")), None)
        return Sample(time.perf_counter(), round(statistics.mean(freqs)) if freqs else None,
                      temp / 1e3 if temp is not None else None, mem,
                      _int(self._gpu["mhz"]), _int(self._gpu["rc6"]), _int(self._gpu["pl1"]))


class Sampler:
    """Samples every `period` s from a child process (this file run as a script), so a runtime whose native calls
    hold the interpreter lock (LiteRT's do) cannot starve it. perf_counter is CLOCK_MONOTONIC on Linux, the same
    clock in both processes, so the child's timestamps compare with the parent's."""

    def __init__(self, period: float = 0.1):
        self.period, self.samples = period, []
        self.gt = Probe().gt
        self._proc = None
        self._file = None

    def start(self) -> None:
        import subprocess
        import sys
        import tempfile
        fd, self._file = tempfile.mkstemp(prefix="ml7_samples_", suffix=".jsonl")
        os.close(fd)
        self._proc = subprocess.Popen([sys.executable, __file__, "--sample", self._file, str(self.period)])

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def _load(self) -> None:
        import json
        if self._file and os.path.exists(self._file):
            with open(self._file) as f:          # a line still being written is skipped
                self.samples = [Sample(*json.loads(line)) for line in f if line.endswith("\n")]

    def stop(self) -> None:
        self._proc.terminate()
        self._proc.wait()
        self._load()
        os.unlink(self._file)
        self._file = None

    def between(self, t0: float, t1: float) -> dict:
        self._load()                                 # the samples so far, while the child still runs
        return summarise(self.samples, t0, t1)


def _sample_forever(path: str, period: float) -> None:
    import json
    import signal
    import sys
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    probe = Probe()
    with open(path, "a", buffering=1) as f:
        while True:
            f.write(json.dumps(list(probe.read())) + "\n")
            time.sleep(period)


def summarise(samples: list[Sample], t0: float, t1: float) -> dict:
    s = [x for x in samples if t0 <= x.t <= t1]

    def vals(field):
        return [getattr(x, field) for x in s if getattr(x, field) is not None]
    rc6 = vals("rc6_ms")
    cpu, gpu, temp, mem, pl1 = vals("cpu_mhz"), vals("gpu_mhz"), vals("temp_c"), vals("mem_available_mb"), vals("pl1")
    return {"samples": len(s), "wall_ms": round((t1 - t0) * 1e3, 1),
            "cpu_mhz_median": statistics.median(cpu) if cpu else None,
            "gpu_mhz_median": statistics.median(gpu) if gpu else None, "gpu_mhz_max": max(gpu) if gpu else None,
            "rc6_ms_delta": (rc6[-1] - rc6[0]) if len(rc6) > 1 else None,
            "pl1_throttled": any(pl1) if pl1 else None, "temp_c_max": max(temp) if temp else None,
            "mem_available_mb_min": min(mem) if mem else None}


def loaded() -> dict:
    """The watched GPU libraries this process has mapped, and the DRM render nodes it has open."""
    libs = sorted({line.split()[-1] for line in (_read(Path("/proc/self/maps")) or "").splitlines()
                   if line.split() and any(Path(line.split()[-1]).name.startswith(w) for w in WATCHED)})
    nodes = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = str(fd.readlink())
        except OSError:
            continue
        if target.startswith("/dev/dri/"):
            nodes.append(target)
    return {"libs": libs, "render_nodes": sorted(set(nodes))}


def gpu_proof(seen: dict, counters: dict, path: str) -> dict:
    """Plan P2's proof for a GPU cell on `path` ("vulkan" or "opencl"), from loaded() and the counters over the
    timed calls; ok with every condition met, else the reasons."""
    names = [Path(lib).name for lib in seen["libs"]]
    why = []
    if not any(n.startswith(d) for n in names for d in DRIVERS[path]):
        why.append(f"no {path} driver ({', '.join(DRIVERS[path])}) loaded")
    if any(n.startswith(LLVMPIPE) for n in names):
        why.append("llvmpipe (Mesa's CPU Vulkan) loaded")
    if not any(n.startswith("/dev/dri/renderD") for n in seen["render_nodes"]):
        why.append("no render node open")
    idle, wall = counters.get("rc6_ms_delta"), counters.get("wall_ms")
    if idle is None or not counters.get("gpu_mhz_max"):
        why.append("no i915 counters, or the GPU clock never left 0")
    elif idle > IDLE_MAX * wall:
        why.append(f"GPU idle {idle:.0f} of {wall:.0f} ms")
    busy = None if idle is None or not wall else round(1 - idle / wall, 3)
    return {"ok": not why, "why_not": why, "gpu_busy_frac": busy, "path": path}


def timing(create_s: float, call_s: list[float], warmup: int = 0) -> dict:
    """Cold: the session's creation and its first call (kernels compile there); warm: the calls after it and
    after `warmup` more, which are not timed."""
    warm = sorted(call_s[1 + warmup:])
    q = {"n": len(warm), "median": statistics.median(warm) if warm else None,
         "p90": warm[min(len(warm) - 1, int(0.9 * (len(warm) - 1) + 0.5))] if warm else None}
    return {"create_s": round(create_s, 3), "first_call_s": round(call_s[0], 3),
            "cold_s": round(create_s + call_s[0], 3), "warm": q}


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def finite(out) -> bool:
    """No NaN or Inf in an output: an array, or a tuple or list of them."""
    import numpy as np                       # here: the module is stdlib-only for the sampler's child
    parts = out if isinstance(out, (tuple, list)) else (out,)
    return all(bool(np.isfinite(np.asarray(p)).all()) for p in parts)


def mem_available_mb() -> float | None:
    for line in (_read(Path("/proc/meminfo")) or "").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return None


def measure(make, call, items: list, sampler: Sampler, path: str | None, warmup: int = 0,
            timed: int | None = None) -> tuple[list, dict]:
    """One cell: make() the session, then call(session, item) 1 + warmup + timed times, cycling through the
    items: the first is the cold call, the warm-ups are not timed (timed: one pass over the items, P2's
    protocol, unless given; P4 gives 3 and 20). Records the timing, CPU time against wall time and the counters
    over the timed calls, the calls with a NaN or Inf in any output, peak RSS and the drop in MemAvailable,
    what the process loaded and, on a GPU `path` ("vulkan", "opencl"), gpu_proof. Returns the outputs of the
    first pass over the items, and the record."""
    probe = Probe()
    mem0 = mem_available_mb()
    t0 = time.perf_counter()
    session = make()
    create_s = time.perf_counter() - t0
    n = 1 + warmup + (len(items) - 1 if timed is None else timed)
    outputs, call_s, warm0, cpu0, nonfinite = [], [], None, None, 0
    for i in range(n):
        if i == 1 + warmup:
            warm0, cpu0, rc6_0 = time.perf_counter(), cpu_seconds(), probe.rc6_ms()
        t = time.perf_counter()
        out = call(session, items[i % len(items)])
        call_s.append(time.perf_counter() - t)
        nonfinite += not finite(out)
        if i < len(items):
            outputs.append(out)
    warm1, cpu1, rc6_1 = time.perf_counter(), cpu_seconds(), probe.rc6_ms()
    counters = sampler.between(warm0, warm1) if warm0 is not None else None
    if counters is not None and rc6_0 is not None and rc6_1 is not None:
        counters["rc6_ms_delta"] = rc6_1 - rc6_0     # cumulative: exact over the window, however few samples
    seen = loaded()
    mem_min = (counters or {}).get("mem_available_mb_min")
    record = {"timing": timing(create_s, call_s, warmup), "calls_s": [round(c, 3) for c in call_s],
              "warmup": warmup, "timed": n - 1 - warmup, "nonfinite_calls": nonfinite,
              "mem_available_mb_before": mem0,
              "mem_available_drop_mb": round(mem0 - mem_min, 1) if mem0 is not None and mem_min is not None else None,
              "cpu_over_wall": round((cpu1 - cpu0) / (warm1 - warm0), 2) if warm0 is not None else None,
              "peak_rss_mb": peak_rss_mb(), "counters": counters, "loaded": seen, "gpu_sampler": sampler.gt}
    if path is not None:
        record["gpu_proof"] = gpu_proof(seen, counters or {}, path)
    return outputs, record


if __name__ == "__main__":       # the Sampler's child: cellrun.py --sample <file> <period>
    import sys
    if sys.argv[1:2] == ["--sample"]:
        _sample_forever(sys.argv[2], float(sys.argv[3]))
