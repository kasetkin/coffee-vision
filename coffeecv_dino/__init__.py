"""Frozen DINO backbones for coffeecv, kept outside `coffeecv/` until they earn integration.

Plan and rationale: docs/dinov3_integration_plan.md. This package may own backbone construction,
readouts, the depth-0 head fit and its own drivers. It must import everything that has to stay
identical between arms -- fold datasets, transforms, metrics, the photo-level scorer, the archive
contract -- from `coffeecv` rather than carry a copy (plan §4.2; tests/test_dino_backbone.py
enforces the part of that a test can see).
"""
