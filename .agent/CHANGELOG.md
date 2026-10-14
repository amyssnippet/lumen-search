# Changelog

## [0.1.0] - 2026-10-11
### Added
- Dataset inspection pipeline (`scripts/inspect_dataset.py`) auditing 492 jewellery images across rings and necklaces.
- Identified 3 exact duplicate sets and 84 near-duplicate twin pairs.
- Dedicated Python 3.11 environment setup via `uv` with PyTorch, timm, open-clip-torch, FastAPI, and NumPy.
- Comprehensive `.agent/` context documentation system (ADRs, evaluation criteria, dataset provenance, issue tracking).
- Verified DINOv2 (`vit_small_patch14_dinov2`) embedding extraction with dynamic resolution support on CPU (~300ms forward pass).
