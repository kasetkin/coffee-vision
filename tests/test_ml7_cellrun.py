"""ML-7 P2-P4: what a runtime cell records (analysis/ml7_gpu/cellrun.py): the GPU proof, the timing summary and the
counters between two instants. Plain unittest, no runtime.

    python -m unittest tests.test_ml7_cellrun -v
"""
from __future__ import annotations

import importlib.util
import sys
import unittest

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


class Timing(unittest.TestCase):
    def test_cold_is_creation_plus_the_first_call_and_warm_the_rest(self):
        t = cr.timing(create_s=2.0, call_s=[5.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(t["cold_s"], 7.0)
        self.assertEqual(t["warm"]["n"], 4)
        self.assertEqual(t["warm"]["median"], 2.5)


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
