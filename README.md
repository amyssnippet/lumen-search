# LUMEN Jewellery Visual Retrieval Engine

An enterprise visual search engine tailored for fine jewellery (rings, necklaces, solitaires, pavé bands, diamond cuts, and settings). Features dual retrieval backbones: **MobileNetV4 Conv-Small FP32 ONNX** (production baseline, 1280-d) and **DINOv3 ViT-S/16 INT8 ONNX** (v2 fine-grained vision POC, 384-d), powered by Qdrant HNSW vector search, PostgreSQL, Redis rate-limiting, and an interactive Next.js 14 web application.

---

## ⚡ Quickstart — Run on Any Machine (Completely Docker)

The entire stack is containerized. **No Python, Node.js, CUDA, or local machine dependencies are required** — only Docker and Docker Compose.

### 1. Clone the Repository
```bash
git clone https://github.com/amyssnippet/lumen-search.git
cd lumen-search
```

### 2. Launch the Multi-Container Stack
```bash
docker compose -f docker-compose.demo.yml --env-file .env.demo up -d --build
```
*(Or simply run `docker compose -f docker-compose.demo.yml up -d --build` — all demo environment variables have self-contained defaults).*

### 3. Open the Web Application
Navigate to:
👉 **[http://localhost:3000](http://localhost:3000)**

- **Immediate Search**: Drag and drop any jewellery image or click sample queries. Works out-of-the-box without login.
- **Pipeline Switching**: Toggle between **MobileNetV4 Baseline (1280-d)** and **DINOv3 ViT-S/16 INT8 (384-d)** in the filter bar to compare fine-grained retrieval performance live.
- **Operator Login (Optional)**:
  - **Email**: `admin@jewellery.internal`
  - **Password**: `dev_password_2026`

---

## 🏗️ What Docker Compose Bootstraps Automatically

When you launch `docker-compose.demo.yml`, five orchestrated services spin up and automatically initialize:

| Service | Container Name | Port | Description & Auto-Bootstrap Behavior |
|---|---|:---:|---|
| **Web Frontend** | `jewellery_demo_web` | **`3000`** | Next.js 14 App Router UI & BFF edge proxy. Communicates securely with backend. |
| **Backend API** | `jewellery_demo_backend` | `8000` (internal) | FastAPI engine with ONNX Runtime CPU inference. Runs migrations & data bootstrap on startup. |
| **Vector Engine** | `jewellery_demo_qdrant` | `6333` (internal) | Qdrant 1.12.1 HNSW vector database. Populates 432 catalog embeddings for both models. |
| **Relational DB** | `jewellery_demo_postgres` | `5432` (internal) | PostgreSQL 16. Runs Alembic migrations, seeds tenants, API keys, 482 products & 518 assets. |
| **Cache & Limiter**| `jewellery_demo_redis` | `6379` (internal) | Redis 7 distributed cache and sliding-window rate limiter. |

> [!NOTE]
> All ONNX model weights (`mobilenetv4_conv_small_fp32.onnx`, `dinov3_vits16_int8.onnx`) and catalog pre-computed vectors are self-contained in the repo. Container bootstrap completes in ~20-30 seconds on standard hardware without external model downloads.

---

## 🔍 Retrieval Architecture & Pipelines

### Dual Visual Backbones
1. **MobileNetV4 Conv-Small FP32 ONNX (Production Baseline)**
   - Dimension: **1280-d**
   - Inference Latency: **~12–15 ms** on CPU
   - Strength: Extremely fast, lightweight inference with solid general jewellery category and silhouette recall.
2. **DINOv3 ViT-S/16 INT8 ONNX (Visual Retrieval POC v2)**
   - Dimension: **384-d**
   - Inference Latency: **~48–55 ms** on CPU (quantized dynamic INT8 `MatMul`/`Gemm`)
   - Strength: Self-supervised Vision Transformer captures micro-geometries (prongs, pave diamonds, halo settings, chain links) with zero text bias.

### Decision & Confidence Zones
Search scores are cosine similarities mapped into calibrated decision bands:
- 🟢 **Verified Exact SKU / Match** ($\ge 88.0\%$ MobileNetV4 / $\ge 90.0\%$ DINOv3): High confidence exact piece identification.
- 🟡 **Visually Similar Design** ($65.0\% - 87.9\%$): Same silhouette, metal type, or gemstone geometry, but variations in setting or shank.
- ⚪ **No Confident Match** ($< 65.0\%$): Image rejected or flagged for low visual similarity.

---

## 🧪 Verification & Diagnostics

### Run Automated E2E Verification Inside Docker
Verify that all 5 containers, database tables, Qdrant vectors, and search endpoints are 100% operational:
```bash
docker exec -it jewellery_demo_backend python scripts/verify_demo_e2e.py
```

### Run Retrieval Test Suite
```bash
docker exec -it jewellery_demo_backend pytest tests/test_retrieval_v2.py -v
```

### Inspect Container Logs
```bash
docker compose -f docker-compose.demo.yml logs -f demo_backend
```

---

## 🛑 Teardown & Maintenance

- **Stop containers (preserve database & indices)**:
  ```bash
  docker compose -f docker-compose.demo.yml stop
  ```
- **Stop and remove containers**:
  ```bash
  docker compose -f docker-compose.demo.yml down
  ```
- **Full reset (wipe all database and Qdrant volumes)**:
  ```bash
  docker compose -f docker-compose.demo.yml down -v
  ```

---

## 📁 Repository Structure

```text
.
├── app/                     # FastAPI backend application & search endpoints
│   ├── main.py              # Main app entrypoint, routes, and authentication
│   └── ...
├── engine/                  # Core computer vision & retrieval algorithms
│   ├── embeddings.py        # MobileNetV4 & DINOv3 ONNX inference runtimes
│   ├── preprocessing.py     # Aspect-ratio padding, CLAHE, denoising, quality gates
│   ├── qdrant_index.py      # Qdrant HNSW vector index driver
│   └── search.py            # High-level search orchestrator
├── data/
│   ├── models/              # Tracked ONNX model weights (MobileNetV4, DINOv3 INT8)
│   ├── index/               # Catalog embeddings & metadata
│   └── evaluation/          # Benchmark manifests & evaluation queries
├── Jewellery_Data/          # Catalogue product photography (rings, necklaces)
├── migrations/              # Alembic relational database migrations
├── scripts/                 # Bootstrap, evaluation, indexing, and diagnostic scripts
│   ├── init_demo_stack.py   # One-shot database & Qdrant auto-initializer
│   ├── verify_demo_e2e.py   # End-to-end verification script
│   └── ...
├── tests/                   # Pytest test suite (unit, integration, retrieval quality)
├── web/                     # Next.js 14 frontend web application (App Router, Tailwind)
│   ├── src/components/      # FilterBar, SearchResults, Drag-and-drop uploader
│   └── Dockerfile           # Standalone multi-stage Next.js Dockerfile
├── Dockerfile               # Backend multi-stage production Dockerfile
├── docker-compose.demo.yml  # Complete 5-service orchestration
└── .env.demo.example        # Environment variable template
```

---

## 📄 License & Provenance

Internal Proof of Concept & Enterprise Retrieval Prototype. All benchmark results, architectural decisions (ADRs), and experimental logs are maintained under `.agent/`.
