"""ML-7 P2-P4: what a runtime cell records (analysis/ml7_gpu/cellrun.py): the GPU proof, the timing summary and the
counters between two instants. Plain unittest, no runtime.

    python -m unittest tests.test_ml7_cellrun -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np

from coffeecv.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("cellrun", REPO_ROOT / "analysis" / "ml7_gpu" / "cellrun.py")
cr = importlib.util.module_from_spec(_spec)
sys.modules["cellrun"] = cr
_spec.loader.exec_module(cr)

_spec = importlib.util.spec_from_file_location("run_litert", REPO_ROOT / "analysis" / "ml7_gpu" / "run_litert.py")
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)

INTEL = "/usr/lib/x86_64-linux-gnu/libvulkan_intel.so"
LLVMPIPE = "/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so"


def _busy(frac, mhz=750):
    return {"wall_ms": 1000.0, "rc6_ms_delta": 1000.0 * (1 - frac), "gpu_mhz_max": mhz}


class GpuProof(unittest.TestCase):
    def test_the_intel_driver_the_render_node_and_a_busy_gpu(self):
        p = cr.gpu_proof({"libs": [INTEL], "render_nodes": ["/dev/dri/renderD128"]}, _busy(0.9), path="vulkan")
        self.assertTrue(p["ok"], p)

    def test_llvmpipe_is_refused_even_with_the_intel_driver_loaded(self):
        p = cr.gpu_proof({"libs": [INTEL, LLVMPIPE], "render_nodes": ["/dev/dri/renderD128"]}, _busy(0.9),
                         path="vulkan")
        self.assertFalse(p["ok"])
        self.assertIn("llvmpipe", " ".join(p["why_not"]))

    def test_an_open_device_that_stayed_idle_is_not_proof(self):
        p = cr.gpu_proof({"libs": [INTEL], "render_nodes": ["/dev/dri/renderD128"]}, _busy(0.1), path="vulkan")
        self.assertFalse(p["ok"])

    def test_no_counters_is_not_proof(self):
        p = cr.gpu_proof({"libs": [INTEL], "render_nodes": ["/dev/dri/renderD128"]},
                         {"wall_ms": 1000.0, "rc6_ms_delta": None, "gpu_mhz_max": None}, path="vulkan")
        self.assertFalse(p["ok"])

    def test_the_opencl_path_needs_neo(self):
        lib = "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"
        ok = cr.gpu_proof({"libs": [lib], "render_nodes": ["/dev/dri/renderD128"]}, _busy(0.9), path="opencl")
        self.assertTrue(ok["ok"])
        self.assertFalse(cr.gpu_proof({"libs": [INTEL], "render_nodes": ["/dev/dri/renderD128"]}, _busy(0.9),
                                      path="opencl")["ok"])


class DelegateReport(unittest.TestCase):
    """LiteRT's is_fully_accelerated() said True for a graph its WebGPU delegate had split 1 GPU op to 2 CPU ops
    (probed on yappy, plan P2): the record reads the delegate's own report from the log instead."""

    LOG = ("INFO: [gpu_environment.h:155] Created LiteRT GpuEnvironment.\n"
           "ERROR: Following operations are not supported by GPU delegate:\n"
           "BATCH_MATMUL: Not supported batched mat mul case: non-constant tensor\n"
           "PADV2: src has wrong size.\n"
           "55 operations will run on the GPU, and the remaining 255 operations will run on the CPU.\n"
           "I0000 00:00:1791913806.814831   31720 delegate_kernel.cc:984] Initializing WebGPU-based API from graph.\n")

    def test_a_split_graph(self):
        r = rl.delegate_report(self.LOG)
        self.assertEqual((r["gpu_ops"], r["cpu_ops"]), (55, 255))
        self.assertEqual(r["refused"], ["BATCH_MATMUL: Not supported batched mat mul case: non-constant tensor",
                                        "PADV2: src has wrong size."])
        self.assertFalse(r["all_on_gpu"])

    def test_no_report_is_a_whole_graph(self):
        r = rl.delegate_report("INFO: [gpu_environment.h:155] Created LiteRT GpuEnvironment.\n")
        self.assertTrue(r["all_on_gpu"])
        self.assertEqual(r["refused"], [])


class CapturedStderrSurvivesACrash(unittest.TestCase):
    """A cell that aborts inside the capture must leave its stderr behind (in the P4 rehearsal ORT's abort
    message was lost with the temporary file), and a clean one passes it on and leaves nothing."""

    def _child(self, tmp, crash: bool) -> str:
        import subprocess
        code = (f"import os, sys; sys.path.insert(0, {str(REPO_ROOT / 'analysis' / 'ml7_gpu')!r}); "
                "from run_litert import CapturedStderr; from pathlib import Path\n"
                f"with CapturedStderr(Path({str(tmp)!r})):\n"
                "    os.write(2, b'native says why\\n')\n"
                f"    {'os.abort()' if crash else 'pass'}\n")
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stderr

    def test_a_crash_leaves_the_file(self):
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp()) / "cell.stderr"
        self._child(tmp, crash=True)
        self.assertIn("native says why", tmp.read_text())

    def test_a_clean_run_passes_it_on_and_removes_the_file(self):
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp()) / "cell.stderr"
        self.assertIn("native says why", self._child(tmp, crash=False))
        self.assertFalse(tmp.exists())


class OpenVinoProof(unittest.TestCase):
    """Plan P3: the OpenCL proof for ORT's OpenVINO provider needs OpenVINO's GPU plugin as well as NEO."""

    NEO = "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"
    PLUGIN = "/home/x/venv/lib/python3.12/site-packages/onnxruntime/capi/libopenvino_intel_gpu_plugin.so"

    def test_neo_and_the_plugin(self):
        p = cr.gpu_proof({"libs": [self.NEO, self.PLUGIN], "render_nodes": ["/dev/dri/renderD128"]}, _busy(0.9),
                         path="openvino")
        self.assertTrue(p["ok"], p)

    def test_neo_without_the_plugin_is_not_openvino(self):
        p = cr.gpu_proof({"libs": [self.NEO], "render_nodes": ["/dev/dri/renderD128"]}, _busy(0.9),
                         path="openvino")
        self.assertFalse(p["ok"])
        self.assertIn("libopenvino_intel_gpu_plugin", " ".join(p["why_not"]))


class Timing(unittest.TestCase):
    def test_cold_is_creation_plus_the_first_call_and_warm_the_rest(self):
        t = cr.timing(create_s=2.0, call_s=[5.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(t["cold_s"], 7.0)
        self.assertEqual(t["warm"]["n"], 4)
        self.assertEqual(t["warm"]["median"], 2.5)


class _StubSampler:
    gt = None

    def between(self, t0, t1):
        return {}


class Measure(unittest.TestCase):
    """P4's protocol (plan P4): one cold call, `warmup` untimed calls, then `timed` calls, cycling through the
    items; every call's outputs checked for NaN and Inf; the first pass over the items kept."""

    def test_cold_warmup_and_timed_calls_cycle_the_items(self):
        seen = []
        outs, rec = cr.measure(lambda: "s", lambda s, x: seen.append(x) or np.array([x], float),
                               [0, 1, 2], _StubSampler(), None, warmup=3, timed=5)
        self.assertEqual(seen, [0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.assertEqual([int(o[0]) for o in outs], [0, 1, 2])
        self.assertEqual(rec["timing"]["warm"]["n"], 5)
        self.assertEqual((rec["warmup"], rec["timed"]), (3, 5))
        self.assertEqual(rec["nonfinite_calls"], 0)

    def test_progress_hears_of_every_call_as_it_ends(self):
        heard = []
        cr.measure(lambda: "s", lambda s, x: np.array([x], float), [0, 1, 2], _StubSampler(), None, warmup=1,
                   timed=2, progress=lambda i, n, phase, s: heard.append((i, n, phase)))
        self.assertEqual(heard, [(0, 4, "cold"), (1, 4, "warm-up"), (2, 4, "timed"), (3, 4, "timed")])

    def test_p2s_protocol_is_the_default(self):
        outs, rec = cr.measure(lambda: "s", lambda s, x: np.array([x], float), [0, 1, 2], _StubSampler(), None)
        self.assertEqual(len(outs), 3)
        self.assertEqual(rec["timing"]["warm"]["n"], 2)

    def test_a_nan_in_any_call_is_counted_even_past_the_first_pass(self):
        calls = iter(range(100))
        bad = lambda s, x: (np.array([np.nan]) if next(calls) == 4 else np.zeros(1), np.zeros(2))  # noqa: E731
        _, rec = cr.measure(lambda: "s", bad, [0, 1, 2], _StubSampler(), None, warmup=1, timed=4)
        self.assertEqual(rec["nonfinite_calls"], 1)


class SamplerProcess(unittest.TestCase):
    def test_samples_arrive_from_a_separate_process(self):
        import os
        import time
        s = cr.Sampler(period=0.05)
        s.start()
        t0 = time.perf_counter()
        time.sleep(0.5)
        t1 = time.perf_counter()
        time.sleep(0.15)
        live = s.between(t0, t1)["samples"]          # while it runs, as cellrun.measure reads it
        s.stop()
        self.assertNotEqual(s.pid, os.getpid())
        self.assertGreaterEqual(live, 5)                # the child's clock is the parent's
        self.assertEqual(s.between(t0, t1)["samples"], live)


class SamplerOrphan(unittest.TestCase):
    """A cell that dies without stopping its sampler (a crash in native code) must not leave it running: in the
    P4 rehearsal an orphaned sampler held the driver's output pipe open for 47 minutes."""

    def test_the_sampler_exits_with_its_parent_and_holds_no_pipe(self):
        import subprocess
        import time
        code = ("import os, sys; sys.path.insert(0, %r); import cellrun; s = cellrun.Sampler(0.05); s.start(); "
                "print(s.pid, flush=True); os._exit(1)") % str(REPO_ROOT / "analysis" / "ml7_gpu")
        t = time.perf_counter()
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10).stdout
        self.assertLess(time.perf_counter() - t, 5)      # the pipe closed: no child kept it open
        pid = int(out.split()[0])
        for _ in range(40):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        os.kill(pid, 9)
        self.fail("the sampler outlived its parent")


class Between(unittest.TestCase):
    def test_counters_between_two_instants(self):
        s = [cr.Sample(t=0.0, cpu_mhz=2000, temp_c=50.0, mem_available_mb=6000, gpu_mhz=0, rc6_ms=100, pl1=0),
             cr.Sample(t=1.0, cpu_mhz=1500, temp_c=60.0, mem_available_mb=5000, gpu_mhz=750, rc6_ms=200, pl1=1),
             cr.Sample(t=2.0, cpu_mhz=1400, temp_c=65.0, mem_available_mb=5500, gpu_mhz=700, rc6_ms=300, pl1=0)]
        b = cr.summarise(s, 0.0, 2.0)
        self.assertEqual(b["rc6_ms_delta"], 200)
        self.assertEqual(b["wall_ms"], 2000.0)
        self.assertEqual(b["gpu_mhz_max"], 750)
        self.assertEqual(b["mem_available_mb_min"], 5000)
        self.assertTrue(b["pl1_throttled"])


if __name__ == "__main__":
    unittest.main()
