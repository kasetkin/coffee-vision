"""Drivers for the frozen-DINO experiments: the Experiment 1 screen, the OOD feasibility check and the
latency bench.

Plan and rationale: docs/dinov3_integration_plan.md. The backbone, the depth-0 head fit and the
classifier module moved into `coffeecv` on 2026-09-27 (plan §8.2: `coffeecv.backbones`,
`coffeecv.linear_head`, `coffeecv.dino_classifier`), because the shipped model needs them. What is left
here must import everything that has to stay identical between arms -- fold datasets, transforms,
metrics, the photo-level scorer, the archive contract, the backbone and head -- from `coffeecv` rather
than carry a copy (plan §4.2; tests/test_dino_backbone.py enforces the part of that a test can see).
"""
