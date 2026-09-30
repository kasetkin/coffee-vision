"""Drivers for the frozen-DINO experiments: the OOD feasibility check and the latency bench. (The
Experiment 1 fold screen, `screen.py`, was deleted with the fold protocol -- ticket ML-1, 2026-09-29;
its results are archived as exp220-231 and exp240-251.)

Plan and rationale: docs/dinov3_integration_plan.md. The backbone, the depth-0 head fit and the
classifier module moved into `coffeecv` on 2026-09-27 (plan §8.2: `coffeecv.backbones`,
`coffeecv.linear_head`, `coffeecv.dino_classifier`), because the shipped model needs them. What is left
here must import everything that has to stay identical between arms -- datasets, transforms,
metrics, the archive contract, the backbone and head -- from `coffeecv` rather
than carry a copy (plan §4.2; tests/test_dino_backbone.py enforces the part of that a test can see).
"""
