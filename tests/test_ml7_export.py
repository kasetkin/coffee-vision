"""ML-7 P1: the export rewrites (analysis/ml7_gpu/graphs.py) and the parity measures (analysis/ml7_gpu/parity.py).
Each rewrite must be the same model at float rounding (plan P1; ticket §4), and the measures must say what
ticket D7 means (owner 2026-10-13: the cosine bar holds for every patch). Random weights, no real data.

    python -m unittest tests.test_ml7_export -v
"""
from __future__ import annotations

import importlib.util
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from coffeecv.config import REPO_ROOT

from tests._tiers import real_data

ML7 = REPO_ROOT / "analysis" / "ml7_gpu"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ML7 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


graphs = _load("graphs")
parity = _load("parity")
export_onnx = _load("export_onnx")


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


class BicubicAsMatmul(unittest.TestCase):
    def test_matches_torch_bicubic_for_the_neck_sizes(self):
        torch.manual_seed(0)
        for hw in ((16, 16), (32, 32), (16, 32), (64, 32)):
            x = torch.randn(2, 5, *hw)
            want = F.interpolate(x, size=(64, 64), mode="bicubic", align_corners=False)
            got = graphs.MatmulResize(hw, (64, 64))(x)
            self.assertEqual(got.shape, want.shape)
            self.assertLess(_rel_err(got, want), 1e-6, hw)

    def test_matrix_rows_sum_to_one(self):
        a = graphs.bicubic_matrix(16, 64)
        self.assertEqual(tuple(a.shape), (64, 16))
        torch.testing.assert_close(a.sum(dim=1), torch.ones(64))

    def test_neck_layers_are_replaced_and_same_size_ones_dropped(self):
        from efficientvit.models.nn.ops import UpSampleLayer

        class Neck(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Sequential(torch.nn.Identity(), UpSampleLayer(size=(64, 64)))
                self.b = torch.nn.Sequential(torch.nn.Identity(), UpSampleLayer(size=(64, 64)))

            def forward(self, x):
                return self.a(x[0]) + self.b(x[1])

        neck = Neck()
        xs = (torch.randn(1, 3, 32, 32), torch.randn(1, 3, 64, 64))
        want = neck(xs)
        replaced = graphs.matmul_neck(neck, xs)
        self.assertEqual(sorted(r["in_hw"] for r in replaced), [(32, 32), (64, 64)])
        self.assertFalse(any(isinstance(m, UpSampleLayer) for m in neck.modules()))
        self.assertLess(_rel_err(neck(xs), want), 1e-6)


class LiteRtGpuForms(unittest.TestCase):
    """LiteRT's WebGPU delegate refuses a constant on a BATCH_MATMUL's left and LiteMLA's pad with ones (PADV2;
    probed on yappy 2026-10-13, plan P2): the encoder's forms it accepts, the same model at float rounding."""

    @staticmethod
    def _encoder_like():
        from efficientvit.models.nn.ops import LiteMLA
        torch.manual_seed(0)
        m = torch.nn.Module()
        m.att = LiteMLA(in_channels=32, out_channels=32, dim=8, scales=(5,))
        m.up = graphs.MatmulResize((16, 16), (32, 32))
        m.forward = lambda x: m.att(x) + m.up(x[:, :, ::2, ::2])
        return m.eval()

    def test_same_model_at_float_rounding(self):
        m, x = self._encoder_like(), torch.randn(1, 32, 32, 32)
        with torch.no_grad():
            want = m(x)
            self.assertEqual(graphs.litert_gpu_forms(m), {"linear_attention": 1, "resize": 1})
            got = m(x)
        self.assertLess(_rel_err(got, want), 1e-6)

    def test_no_pad_with_ones_and_no_constant_on_a_matmul_left(self):
        m = self._encoder_like()
        graphs.litert_gpu_forms(m)
        ep = torch.export.export(m, (torch.randn(1, 32, 32, 32),))
        consts = set(ep.graph_signature.inputs_to_buffers) | set(ep.graph_signature.inputs_to_parameters)
        for n in ep.graph.nodes:
            if n.op != "call_function":
                continue
            name = str(n.target)
            self.assertFalse("constant_pad_nd" in name and n.args[2:3] == (1.0,), n.format_node())
            if "matmul" in name or "bmm" in name or name.endswith("mm.default"):
                self.assertNotIn(getattr(n.args[0], "name", None), consts, n.format_node())


@real_data   # builds a DINOv3 ViT-S (random weights) and traces it: a few seconds
class FourDAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import timm
        torch.manual_seed(0)
        cls.model = timm.create_model("vit_small_patch16_dinov3", pretrained=False, num_classes=0).eval()
        cls.x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            cls.want = cls.model.forward_features(cls.x)
        cls.n = graphs.four_d_attention(cls.model)
        graphs.constant_rope(cls.model, (4, 4))

    def test_same_model_at_float_rounding(self):
        self.assertEqual(self.n, len(self.model.blocks))
        with torch.no_grad():
            got = self.model.forward_features(self.x)
        self.assertLess(_rel_err(got, self.want), 1e-5)

    def test_no_tensor_above_four_dimensions(self):
        ep = torch.export.export(self.model, (self.x,))
        ranks = {n.name: len(n.meta["val"].shape) for n in ep.graph.nodes
                 if isinstance(n.meta.get("val"), torch.Tensor)}
        self.assertLessEqual(max(ranks.values()), 4, {k: r for k, r in ranks.items() if r > 4})

    def test_the_constant_table_is_for_one_grid_only(self):
        with self.assertRaises(ValueError):
            self.model.rope.get_embed(shape=(5, 5))


class AttentionScoreMatmuls(unittest.TestCase):
    """The fp16 copies keep q k^T in fp32 (plan P1): found directly before a Softmax or through a scale."""

    @staticmethod
    def _graph(*nodes):
        from types import SimpleNamespace as N
        return N(graph=N(node=[N(name=n, op_type=op, input=list(i), output=list(o)) for n, op, i, o in nodes]))

    def test_direct_and_scaled_scores_found_and_others_not(self):
        g = self._graph(("dino_qk", "MatMul", ["q", "kt"], ["s1"]), ("sm1", "Softmax", ["s1"], ["a1"]),
                        ("sam_qk", "MatMul", ["q2", "kt2"], ["s2"]), ("scale", "Div", ["s2", "d"], ["s3"]),
                        ("sm2", "Softmax", ["s3"], ["a2"]),
                        ("av", "MatMul", ["a1", "v"], ["o"]),                     # attn @ v: not a score
                        ("lin", "Gemm", ["x", "w"], ["l"]), ("sm3", "Softmax", ["l"], ["p"]))
        self.assertEqual(export_onnx.attention_score_matmuls(g), ["dino_qk", "sam_qk"])


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@unittest.skipUnless(_have("onnx") and _have("onnxruntime"), "onnx and onnxruntime: the convert_onnx environment")
class SequencesToSlices(unittest.TestCase):
    """ORT's WebGPU provider has no sequence kernels, so DINOv3's chunk and tensor_split (SplitToSequence read by
    SequenceAt) become Slices (plan P2): the same tensors, exactly."""

    @staticmethod
    def _model(split, positions):
        from onnx import TensorProto, helper
        c = lambda name, v: helper.make_node("Constant", [], [name], value=helper.make_tensor(  # noqa: E731
            name, TensorProto.INT64, [len(v)] if isinstance(v, list) else [], v if isinstance(v, list) else [v]))
        nodes = [c("split", split), helper.make_node("SplitToSequence", ["x", "split"], ["seq"], axis=-1)]
        for i, p in enumerate(positions):
            nodes += [c(f"p{i}", p), helper.make_node("SequenceAt", ["seq", f"p{i}"], [f"y{i}"])]
        nodes.append(helper.make_node("Concat", [f"y{i}" for i in reversed(range(len(positions)))], ["out"], axis=-1))
        g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 7])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, None])])
        return helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)], ir_version=10)

    @staticmethod
    def _run(model, x):
        import onnxruntime as ort
        s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        return s.run(None, {"x": x})[0]

    def test_equal_chunks_and_listed_sizes_give_the_same_tensors(self):
        x = np.arange(14, dtype=np.float32).reshape(2, 7)
        for split, positions in ((3, [0, 1, 2]), ([1, 2, 4], [2, 0, 1]), (3, [1])):
            before = self._model(split, positions)
            after = self._model(split, positions)
            self.assertEqual(export_onnx.sequences_to_slices(after), len(positions))
            self.assertFalse({n.op_type for n in after.graph.node} & {"SplitToSequence", "SequenceAt"})
            np.testing.assert_array_equal(self._run(after, x), self._run(before, x))

    def test_a_position_that_is_not_constant_is_refused(self):
        from onnx import helper
        m = self._model(3, [0])
        m.graph.node[-2].input[1] = "x"                    # SequenceAt's position from a graph input
        m.graph.node.insert(0, helper.make_node("Identity", ["x"], ["unused"]))
        with self.assertRaises(ValueError):
            export_onnx.sequences_to_slices(m)


@unittest.skipUnless(_have("onnx") and _have("onnxruntime"), "onnx and onnxruntime: the convert_onnx environment")
class RetypeCasts(unittest.TestCase):
    """onnxconverter-common typed a Cast's output fp16 and left the Cast at float (the fp16 decoder would not load;
    plan P2): the Cast follows the type its consumers were converted for."""

    def test_a_cast_left_at_float_feeding_fp16_is_retyped(self):
        import onnxruntime as ort
        from onnx import TensorProto, helper
        g = helper.make_graph(
            [helper.make_node("Constant", [], ["two"], value=helper.make_tensor("two", TensorProto.INT64, [], [2])),
             helper.make_node("Cast", ["two"], ["two_f"], to=TensorProto.FLOAT),
             helper.make_node("Mul", ["x", "two_f"], ["y"])], "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [3])],
            value_info=[helper.make_tensor_value_info("two_f", TensorProto.FLOAT16, [])])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)], ir_version=10)
        self.assertEqual(export_onnx.retype_casts(m), ["two_f"])
        s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        np.testing.assert_array_equal(s.run(None, {"x": np.ones(3, np.float16)})[0], np.full(3, 2, np.float16))


@unittest.skipUnless(_have("onnx") and _have("onnxruntime"), "onnx and onnxruntime: the convert_onnx environment")
class DropZeroBiases(unittest.TestCase):
    """The exporter gives a Conv without a bias a computed zero bias, Expand(CastLike(0), Shape(w)[:1]), which
    OpenVINO's CPU plugin refuses (plan P3): the Conv without it, the same model (ORT's Conv kernel orders its sums
    differently with a bias: 1 ulp)."""

    @staticmethod
    def _model(bias_value):
        from onnx import TensorProto, helper, numpy_helper
        w = numpy_helper.from_array(np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1, 1), "w")
        nodes = [helper.make_node("Constant", [], ["z"], value=helper.make_tensor("z", TensorProto.FLOAT, [],
                                                                                [bias_value])),
                 helper.make_node("CastLike", ["z", "x"], ["zc"]),
                 helper.make_node("Shape", ["w"], ["ws"], start=0, end=1),
                 helper.make_node("Expand", ["zc", "ws"], ["b"]),
                 helper.make_node("Conv", ["x", "w", "b"], ["y"])]
        g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])], [w])
        return helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)], ir_version=10)

    def test_a_computed_zero_bias_is_dropped_and_its_nodes_with_it(self):
        import onnxruntime as ort
        before, after = self._model(0.0), self._model(0.0)
        self.assertEqual(export_onnx.drop_zero_biases(after), 1)
        self.assertEqual([n.op_type for n in after.graph.node], ["Conv"])
        x = np.random.default_rng(0).random((1, 3, 2, 2), dtype=np.float32)
        run = lambda m: ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(  # noqa: E731
            None, {"x": x})[0]
        np.testing.assert_allclose(run(after), run(before), rtol=1e-6)

    def test_a_nonzero_bias_is_kept(self):
        m = self._model(1.0)
        self.assertEqual(export_onnx.drop_zero_biases(m), 0)
        self.assertEqual(len(m.graph.node), 5)


class Measures(unittest.TestCase):
    def test_least_cosine_over_every_patch(self):
        a = np.ones((2, 3, 4), np.float32)
        b = a.copy()
        b[1, 2] = [1, 1, 1, -1]                      # one patch at cosine 0.5
        self.assertAlmostEqual(parity.least_cosine(a, b), 0.5, places=6)
        self.assertAlmostEqual(parity.least_cosine(a, a), 1.0, places=6)

    def test_mask_iou(self):
        a = np.zeros((4, 4), bool)
        a[:2] = True                                 # 8 pixels
        b = a.copy()
        b[0, 0] = False
        self.assertAlmostEqual(parity.mask_iou(a, b), 7 / 8)
        self.assertEqual(parity.mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)), 1.0)

    def test_gate_reads_d7(self):
        rows = [{"photo": "a", "least_cosine": 0.999995, "mask_iou": 0.9995, "top1_same": True, "verdict_same": True},
                {"photo": "b", "least_cosine": 0.99999, "mask_iou": 0.999, "top1_same": False, "verdict_same": True}]
        g = parity.gate(rows)
        self.assertTrue(g["pass"])                   # the bars are inclusive; a changed answer does not block
        self.assertEqual(g["changed_answer"], ["b"])
        rows[0]["least_cosine"] = 0.99998
        self.assertFalse(parity.gate(rows)["pass"])
        rows[0]["least_cosine"], rows[1]["mask_iou"] = 1.0, 0.998
        self.assertFalse(parity.gate(rows)["pass"])

    def test_a_nan_anywhere_fails(self):
        for rows in ([{"photo": "a", "least_cosine": 1.0}, {"photo": "b", "least_cosine": float("nan")}],
                     [{"photo": "a", "least_cosine": float("nan")}, {"photo": "b", "least_cosine": 1.0}],
                     [{"photo": "a", "mask_iou": 1.0, "low_nan_or_inf": True}],
                     [{"photo": "a", "least_cosine": 1.0, "nan_or_inf": True}]):
            self.assertFalse(parity.gate(rows)["pass"], rows)

    def test_nothing_measured_fails(self):
        g = parity.gate([{"photo": "a", "least_cosine": None, "mask_iou": None}])
        self.assertFalse(g["pass"])

    def test_the_gate_is_json_plain(self):
        g = parity.gate([{"photo": "a", "least_cosine": 1.0, "mask_iou": parity.mask_iou(
            np.ones((2, 2), bool), np.ones((2, 2), bool))}])
        self.assertIs(g["pass"], True)
        self.assertIs(type(g["least_mask_iou"]), float)

    def test_the_centroid_guard_has_a_verdict(self):
        ref = {"classes": {"a": {"centroid": [0.0, 0.0], "spread": 1.0}}}
        head = {"weight": np.eye(2, dtype=np.float32), "bias": np.zeros(2, np.float32)}
        near = parity._answers(np.zeros((3, 2), np.float32), head, ref, None)
        far = parity._answers(np.full((3, 2), 10.0, np.float32), head, ref, None)
        self.assertFalse(near["refused"])
        self.assertTrue(far["refused"])

    def test_a_changed_d18_verdict_is_a_changed_answer(self):
        g = parity.gate([{"photo": "a", "least_cosine": None, "mask_iou": 0.9995, "fallback_same": False}])
        self.assertTrue(g["pass"])
        self.assertEqual(g["changed_answer"], ["a"])

    def test_a_graph_without_masks_or_readouts_is_gated_on_what_it_has(self):
        g = parity.gate([{"photo": "a", "least_cosine": None, "mask_iou": 0.9999}])
        self.assertTrue(g["pass"])
        self.assertIsNone(g["least_cosine"])


if __name__ == "__main__":
    unittest.main()
