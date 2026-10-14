# Jewellery Visual Retrieval Engine — Agent Entry Point

## Project Overview
This repository implements an iterative visual retrieval engine tailored for fine jewellery (rings and necklaces). The catalog contains 492 images (301 necklaces, 191 rings).

The primary objective is exact-piece retrieval where verified identity labels exist, and high-precision visual similarity retrieval across subtle design variations (gemstone cuts, setting styles, metal types, band structures).

## System Architecture & Stack
- **Python**: 3.11.17 (`.venv/bin/python`)
- **Deep Learning / Vision**: PyTorch, torchvision, timm, open-clip-torch
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Vector Search**: Exact cosine similarity with normalized embeddings (NumPy / Torch)
- **Frontend**: Clean, responsive single-page web application (vanilla HTML/CSS/JS)
- **Quality & Testing**: Pytest, reproducible manifests and evaluation scripts

## Essential Navigation
- [Project State](file:///Users/amol/Developer/search/.agent/PROJECT_STATE.md) — Source of truth on current implementation and verified milestones
- [Architecture](file:///Users/amol/Developer/search/.agent/ARCHITECTURE.md) — System design, data flow, embedding pipeline
- [Dataset](file:///Users/amol/Developer/search/.agent/DATASET.md) — Dataset provenance, statistics, duplicates, format details
- [Experiments](file:///Users/amol/Developer/search/.agent/EXPERIMENTS.md) — Benchmark tracking and model comparisons
- [Evaluation](file:///Users/amol/Developer/search/.agent/EVALUATION.md) — Evaluation methodology, metrics, and ground truth definitions
- [Next Steps](file:///Users/amol/Developer/search/.agent/NEXT_STEPS.md) — Prioritized execution backlog

## Quick Commands
```bash
# Activate environment
source .venv/bin/activate

# Inspect dataset and generate manifest
python scripts/inspect_dataset.py
python scripts/build_manifest.py

# Run baseline indexing and evaluation
python scripts/index_catalog.py --model dinov2_vits14
python scripts/evaluate.py --model dinov2_vits14

# Component localization & region evaluation (EXP-006)
python scripts/index_catalog_regions.py
python scripts/evaluate_region_retrieval.py

# Soft spatial attention & patch reweighting (EXP-007)
python scripts/index_catalog_soft_attention.py
python scripts/evaluate_soft_attention.py

# Start demonstration server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run automated tests
pytest tests/ -v
```

