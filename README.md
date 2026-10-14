# AURA / LUMEN — Fine Jewellery Visual Retrieval Engine

A working, fine-grained visual retrieval engine tailored for fine jewellery (rings, necklaces, diamond cuts, and settings). Built with MobileNetV4 Conv-Small FP32 ONNX, self-supervised Vision Transformers (DINOv2), OpenCLIP, Qdrant vector retrieval, PostgreSQL, Redis rate-limiting, and an interactive Next.js 14 web application.

---

## 🌟 Interactive Web Application & Local Docker Demo

Experience the full enterprise visual retrieval engine locally via Docker Compose:

- **Web Application URL**: **[http://localhost:3000](http://localhost:3000)**
- **Stack**: Next.js 14 (App Router & BFF) + FastAPI Backend + MobileNetV4 FP32 ONNX (1280-d) + Qdrant 1.19.2 + PostgreSQL 16 + Redis 7
- **Full Guide**: See [`DEMO_GUIDE.md`](./DEMO_GUIDE.md)

### One-Command Launch:
```bash
# Launch all 5 isolated containers in detached mode
docker compose -f docker-compose.demo.yml --env-file .env.demo up -d --build

# Open your browser at http://localhost:3000
```

### Pre-Configured Demo Credentials (Optional Operator Login):
- **Email**: `admin@jewellery.internal`
- **Password**: `dev_password_2026`
*(Searches work immediately out-of-the-box without signing in!)*

### Automated End-to-End Verification:
```bash
python scripts/verify_demo_e2e.py
```

---

## 💎 Key Features
- **MobileNetV4 Conv-Small FP32 ONNX Backbone**: 1280-dimensional feature representations optimized for sub-100ms retrieval without heavy GPU dependencies.
- **Self-Supervised Backbone (DINOv2)**: Outperforms CLIP on exact-piece jewellery retrieval (**70.0% vs 43.3% Recall@1**) by capturing prongs, facet cuts, pavé bands, and chain geometries without semantic text bias.
- **Robust Preprocessing & Quality Gating**:
  - EXIF orientation correction.
  - Geometry-preserving aspect ratio padding (prevents distortion of circular ring shanks).
  - Laplacian blur detection, exposure scoring, and low-res warnings.
  - Bilateral facet-preserving denoising (reduces false exact matches by 24.5%).
  - Non-destructive CLAHE contrast enhancement on LAB luminance channel.
- **Confidence Calibration**: Three calibrated retrieval zones:
  - `Verified Exact SKU / Twin` ($\ge 0.94$ similarity)
  - `Visually Similar Design` ($0.70 \le s < 0.94$)
  - `Low Confidence / Rejected` ($< 0.70$)
- **Interactive Next.js Web Application**: Drag-and-drop search UI with one-click test queries, side-by-side candidate comparison, diagnostic latency bars, relational catalogue explorer, and release audit dashboard.
- **Persistent Context System**: Full `.agent/` engineering source of truth, ADRs, test protocol, and experiment reports.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
The project runs with Python 3.11 in a pinned virtual environment:
```bash
# Provision Python 3.11 via uv (if not already provisioned)
~/.local/bin/uv venv --python 3.11 .venv

# Activate virtual environment
source .venv/bin/activate
```

### 2. Dataset Preparation & Manifest Generation
Audit the 492 images, extract exact duplicate clusters and near-duplicate twins, and create disjoint evaluation splits:
```bash
python scripts/inspect_dataset.py
python scripts/build_manifest.py
```

### 3. Catalog Indexing
Compute and cache catalog embeddings for DINOv2 and CLIP:
```bash
python scripts/index_catalog.py --model all
```

### 4. Run Empirical Retrieval Evaluation
Run the benchmark across 60 held-out evaluation queries:
```bash
# Compare DINOv2 and CLIP
python scripts/evaluate.py --model both

# Test preprocessing options
python scripts/evaluate.py --model dinov2 --denoise
python scripts/evaluate.py --model dinov2 --auto-enhance
```

### 5. Launch Interactive Demonstration Web App
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### 6. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 📊 Measured Benchmark Results (EXP-001 & EXP-002)

| Metric | DINOv2 Baseline | DINOv2 + Bilateral Denoise | OpenCLIP (ViT-B-32) |
|---|---|---|---|
| **Recall@1 (Exact Piece / Twin)** | **70.00%** | **70.00%** | 43.33% |
| **Recall@5** | **83.33%** | **83.33%** | 80.00% |
| **Recall@10** | **90.00%** | 86.67% | 86.67% |
| **Mean Reciprocal Rank (MRR)** | **0.7478** | 0.7426 | 0.5875 |
| **Category Precision@5** | **100.00%** | **100.00%** | 98.67% |
| **False Exact Matches ($\ge 0.94$)** | 53 | **40 (-24.5%)** | 363 |
| **Mean Single-Query Latency (CPU)** | 138.62 ms | 138.76 ms | 122.51 ms |

---

## 🛠️ REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health check & cached engine state |
| `/api/catalog/stats` | `GET` | Catalog image count, categories, and vector dimensions |
| `/api/search` | `POST` | Multipart query image upload with model selection and preprocessing toggles |
| `/api/feedback` | `POST` | Record user relevance feedback (`correct`, `incorrect`, `uncertain`) |
| `/api/evaluation` | `GET` | Fetch measured evaluation metrics and comparison table |
| `/media/{category}/{filename}` | `GET` | Path-traversal-protected media endpoint for catalog images |

---

## 📁 Repository Structure
```text
.
├── .agent/                  # Persistent engineering context & research logs
│   ├── ARCHITECTURE.md      # System architecture & Mermaid data flow
│   ├── DATASET.md           # Dataset provenance & statistics
│   ├── DECISIONS.md         # Architecture Decision Records (ADRs)
│   ├── EXPERIMENTS.md       # Experiment registry
│   ├── IMPLEMENTATION_STATUS.md # Feature capability matrix
│   ├── PROJECT_STATE.md     # Current verified source of truth
│   └── experiments/         # Detailed reports (EXP-001, EXP-002)
├── app/
│   └── main.py              # FastAPI application & REST endpoints
├── engine/
│   ├── embeddings.py        # DINOv2 and CLIP encoder wrappers
│   ├── index.py             # Vector cache & cosine similarity search
│   ├── preprocessing.py     # Quality assessment, CLAHE, denoising, padding
│   └── search.py            # High-level search orchestrator
├── frontend/
│   ├── index.html           # Single-page web demonstration UI
│   └── static/
│       ├── app.js           # Client-side search and feedback logic
│       └── style.css        # Jewellery-themed dark gold responsive styling
├── scripts/
│   ├── build_manifest.py    # Manifest generator with cluster detection
│   ├── evaluate.py          # Benchmark runner
│   ├── index_catalog.py     # Embeddings indexing script
│   └── inspect_dataset.py   # Dataset auditor
└── tests/                   # 16 unit and integration tests
```
