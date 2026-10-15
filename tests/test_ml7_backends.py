"""ML-7 P4, P7: which precision each cell's graph is compiled at (analysis/ml7_gpu/backends.py), and that the record
says so. Plain unittest, no runtime: LiteRT's Graph and ORT's make_session are stand-ins that note their arguments.

    python -m unittest tests.test_ml7_backends -v
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from coffeecv.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("backends", REPO_ROOT / "analysis" / "ml7_gpu" / "backends.py")
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)


class _Graph:
    """run_litert.Graph's stand-in: keeps its arguments."""
    made: list = []

    def __init__(self, path, threads, accelerator, fp32, runtime_dir):
        self.args = {"path": Path(path), "accelerator": accelerator, "fp32": fp32, "runtime_dir": runtime_dir}
        _Graph.made.append(self.args)
        self.input_shapes = [[8]]
        self.model = types.SimpleNamespace(is_fully_accelerated=lambda: True)


def _session(path, provider, threads, spin, precision, cache_dir):
    _session.made.append({"path": Path(path), "precision": precision})
    return types.SimpleNamespace(get_inputs=lambda: [types.SimpleNamespace(shape=[8])],
                                 get_providers=lambda: [provider])


_session.made = []


def _open(cell, graph):
    """open_graph with the stand-ins; (the runtime's arguments, the Model's info)."""
    fakes = {"run_litert": types.SimpleNamespace(Graph=_Graph), "run_ort": types.SimpleNamespace(make_session=_session)}
    _Graph.made.clear()
    _session.made.clear()
    with mock.patch.dict(sys.modules, fakes):
        model = bk.open_graph(cell, graph, Path("/m"), 4, runtime_dir=Path("/rt"))
    return (_Graph.made or _session.made)[0], model.info


class LiteRtPrecision(unittest.TestCase):
    def test_existing_cells_compile_every_graph_as_before(self):
        # Before the split cell, every LiteRT graph got enforce_f32 = (cell.precision == "fp32").
        for name, fp32 in (("litert_cpu", True), ("litert_gpu", True), ("litert_gpu_fp16", False)):
            for graph in bk.CELLS[name].graphs:
                with self.subTest(cell=name, graph=graph):
                    args, info = _open(name, graph)
                    self.assertIs(args["fp32"], fp32)
                    self.assertEqual(info["precision"], bk.CELLS[name].precision)
                    self.assertEqual(args["path"], Path("/m/tflite") / f"{graph}.tflite")

    def test_split_cell_keeps_the_segmenter_in_fp32_and_dino_in_fp16(self):
        expected = {"encoder": True, "decoder": True, "dino_b8": False, "dino_b40": False}
        self.assertEqual(set(bk.CELLS["litert_gpu_split"].graphs), set(expected))
        for graph, fp32 in expected.items():
            with self.subTest(graph=graph):
                args, info = _open("litert_gpu_split", graph)
                self.assertIs(args["fp32"], fp32)
                self.assertEqual(args["accelerator"], "gpu")
                self.assertEqual(args["runtime_dir"], Path("/rt"))
                self.assertEqual(info["precision"], "fp32" if fp32 else "fp16")
                self.assertEqual(info["cell"], "litert_gpu_split")

    def test_split_cell_is_on_vulkan(self):
        self.assertEqual(bk.CELLS["litert_gpu_split"].path, "vulkan")

    def test_fp16dec32_cell_keeps_only_the_decoder_in_fp32(self):
        expected = {"encoder": False, "decoder": True, "dino_b8": False, "dino_b40": False}
        for graph, fp32 in expected.items():
            with self.subTest(graph=graph):
                args, info = _open("litert_gpu_fp16dec32", graph)
                self.assertIs(args["fp32"], fp32)
                self.assertEqual(info["precision"], "fp32" if fp32 else "fp16")


class OrtPrecision(unittest.TestCase):
    def test_ort_cells_pass_the_cell_precision_as_before(self):
        for name, cell in bk.CELLS.items():
            if cell.runtime != "ort":
                continue
            for graph in cell.graphs:
                with self.subTest(cell=name, graph=graph):
                    args, info = _open(name, graph)
                    self.assertEqual(args["precision"], cell.precision)
                    self.assertEqual(info["precision"], cell.precision)


class PrecisionOf(unittest.TestCase):
    def test_every_cell_names_only_graphs_it_has(self):
        for name, cell in bk.CELLS.items():
            with self.subTest(cell=name):
                self.assertLessEqual(set(cell.fp16_graphs), set(cell.graphs))

    def test_a_cell_without_fp16_graphs_runs_all_at_its_precision(self):
        for name, cell in bk.CELLS.items():
            if cell.fp16_graphs:
                continue
            for graph in cell.graphs:
                with self.subTest(cell=name, graph=graph):
                    self.assertEqual(cell.precision_of(graph), cell.precision)


if __name__ == "__main__":
    unittest.main()
