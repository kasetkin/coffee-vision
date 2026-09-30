# Local patches to the vendored EfficientViT

Upstream: https://github.com/mit-han-lab/efficientvit at `de7d7733` (2025-09-05), vendored
unmodified in commit `ce4e753` (ticket ML-2). Only `.git/` and `assets/` were left out.

Rule (ticket ML-2, R5): patch out what CPU inference of EfficientViT-SAM-L0 never uses; install
what it does use. Each patch below only moves or drops an import, so no model code changes.
Every patched line carries a `PATCH(coffee-vision)` comment pointing here, so
`grep -rn "PATCH(coffee-vision)" third_party/efficientvit` lists them all.

The one real dependency the import chain needs, Meta's `segment_anything` (`sam.py` builds on its
`MaskDecoder`, `PromptEncoder` and `ResizeLongestSide`), is installed from PyPI as
`segment-anything==1.0` (pyproject.toml), not patched.

| # | File | Why | Change |
|---|---|---|---|
| 1 | `efficientvit/models/nn/__init__.py`, `efficientvit/models/nn/norm.py` | `triton_rms_norm.py` imports `triton` at module level, and `norm.py` imports it eagerly. The CPU torch wheel has no triton, and L0 has no `TritonRMSNorm2d` layer (it uses `bn2d`, `ln` and `ln2d`). | `nn/__init__.py` no longer star-imports `triton_rms_norm`; `norm.py` imports `TritonRMSNorm2dFunc` inside `TritonRMSNorm2d.forward`, the only code that uses it. |
| 2 | `efficientvit/models/efficientvit/__init__.py` | `from .dc_ae import *` pulls in `omegaconf` (the DC-AE autoencoder's config). SAM never uses DC-AE. | The `dc_ae` star-import is commented out. `efficientvit.ae_model_zoo` and the diffusion code, which need it, are not used here. |
| 3 | `efficientvit/apps/utils/export.py` | `import onnx` and `from onnxsim import ...` at module level; the model reaches this file through `models/nn/drop.py` → `efficientvit.apps`. onnx is not installed anywhere (ticket D23: PyTorch runtime at both ends). | Both imports moved inside `export_onnx`, the only code that uses them. |

`tests/test_sam_loader.py` builds L0 with `triton`, `omegaconf` and `onnx` blocked in
`sys.modules`, so a new upstream import of any of them fails the test rather than a deploy.

When re-vendoring a newer upstream: copy it over unmodified, re-apply the table above, rerun
`tests/test_sam_loader.py`, and treat any new import failure by the same rule.
