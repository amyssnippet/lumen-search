# Testing and Verification Protocol

## 1. Automated Test Execution
Run all test suites using `pytest`:
```bash
.venv/bin/pytest tests/ -v
```

## 2. Test Suites Overview
- `tests/test_dataset.py`: Validates dataset integrity, checks for corrupt images, and verifies hash-based duplicate detection.
- `tests/test_preprocessing.py`: Tests EXIF orientation correction, aspect ratio padding, quality assessment (Laplacian blur score), and CLAHE enhancement.
- `tests/test_embeddings.py`: Checks vector dimension, normalization ($\|v\|_2 = 1.0$), and deterministic output.
- `tests/test_index.py`: Tests cosine similarity ranking, top-k ordering, and category filtering.
- `tests/test_api.py`: Validates FastAPI endpoints (`/health`, `/api/catalog/stats`, `/api/search`, `/api/feedback`, `/api/evaluation`).

## 3. End-to-End Evaluation Reproduction
```bash
# Build manifest
.venv/bin/python scripts/build_manifest.py

# Index catalog
.venv/bin/python scripts/index_catalog.py --model dinov2

# Run evaluation benchmark
.venv/bin/python scripts/evaluate.py --model dinov2
```
