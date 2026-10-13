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
