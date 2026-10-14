# Experiments Directory

This directory stores reproducible reports and raw result artifacts for each iterative retrieval experiment.

## Naming Convention
`EXP-<ID>-<short-slug>.md`

## Index of Reports
- `EXP-001-baseline-embeddings.md`: Evaluation comparing DINOv2 (`vit_small_patch14_dinov2`) vs OpenCLIP (`ViT-B-32`).
- `EXP-002-query-preprocessing.md`: Evaluation testing Laplacian blur gating, CLAHE contrast equalization, bilateral denoising, and aspect padding.
- `EXP-003-confidence-calibration.md`: Threshold calibration and rejection boundary verification.
