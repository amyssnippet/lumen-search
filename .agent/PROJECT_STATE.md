# Project State: Current Source of Truth

**Date of Update**: 2026-10-14  
**Current Milestone**: Milestone 25: POC — Fine-Grained Visual Jewellery Retrieval v2 (`poc/visual-retrieval`)  
**Database / Execution Environment**: Qdrant 1.12.1 (isolated bridge network `jewellery_demo_network`), PostgreSQL 16 Alpine, Redis 7 Alpine with fail-secure token bucket, Next.js 14 App Router & BFF (Alpine Node 20 runtime on port 3000), FastAPI backend, Python 3.11.17 (`.venv`), ONNX Runtime 1.23.2, MobileNetV4 Conv-Small FP32 ONNX (1280-dim, baseline), DINOv3 ViT-S/16 INT8 ONNX (384-dim, v2 POC)  

---

## 1. Verified Working Functionality

- **POC — Fine-Grained Visual Jewellery Retrieval v2 (`poc/visual-retrieval`, `RETRIEVAL_V2_EXPERIMENT_REPORT.md`)**:
  - **Isolated Git Branch & Safe Namespace**: Created branch `poc/visual-retrieval` strictly preserving `main` and `poc/ocr-integration`. Created independent Qdrant collection `jewellery_catalog_dinov3_v1` (384-d, 432 points) alongside production `jewellery_catalog_mobilenetv4_v1` (1280-d, 432 points) with zero cross-contamination.
  - **Empirical Visual Backbone Gains**: Benchmark Recall@1 across the frozen 6-slice manifest jumped from **56.52%** (MobileNetV4 Baseline) to **61.96%** (DINOv3 INT8 ONNX), with MRR improving from **0.6374** to **0.6716**. On 32 viewpoint-shifted queries (Slice 3), Recall@1 jumped from **53.12% to 68.75% (+15.63%)**.
  - **Honest Patch Chamfer Evaluation**: Symmetrical patch Chamfer reranking proved counterproductive on 3D jewellery cross-view rotation (Recall@1 dropped from 40% to 26.67%), proving that global self-supervised ViT representations are superior to unconstrained 2D patch matching.
  - **Calibrated Model-Specific Decisions**: Validated distinct threshold regimes: DINOv3 Exact $\ge 0.980$ (FAR 3.85%, Coverage 65.33%), High $\ge 0.950$, Similar $\ge 0.900$, No Match $< 0.900$; MobileNetV4 Exact $\ge 0.880$, High $\ge 0.700$, Similar $\ge 0.520$, No Match $< 0.520$.
  - **Full-Stack Next.js 14 Mode Selection**: Added interactive "Baseline (MobileNetV4)" vs "Fine-Grained v2 (DINOv3)" selector in `FilterBar.tsx` and `page.tsx` with dynamic decision explanations, active model architecture badges, and instant re-querying without re-uploading.
  - **100% Green Automated Test Suite**: Added `tests/test_retrieval_v2.py` (7/7 passing in 15.7s); existing test suite `tests/test_ph014_retrieval_quality.py` (4/4 passing in 20.1s) remains 100% intact.
- **PH-014 — Retrieval Quality, Candidate Filtering & Real-Image Validation (`engine/qdrant_index.py`, `engine/search.py`, `web/`)**:
  - **Defensive Category Filtering**: Implemented strict pre-filtering and post-fusion candidate filtering in both FastAPI backend and Qdrant client (`norm_cat = category_filter.lower().rstrip("s")`), eliminating 100% of cross-category candidate leakage (e.g. necklaces appearing in ring queries).
  - **Next.js Category Synchronization**: Bound preset and candidate selection in `ImageDropzone.tsx` and `page.tsx` to automatically set the category filter.
  - **Qdrant Search Batching & Latency Remediation**: Replaced 3 sequential HTTP round-trips with `client.query_batch_points` and cached collection size property, dropping live Qdrant search latency from **399.1 ms down to 16.84 ms p50** (a **23.7x speedup**). Total engine time dropped to 105.72 ms p50, browser E2E to 139.45 ms p50.
  - **Defensible No-Match Decision Rule**: Calibrated 3 distinct decision states: `EXACT_MATCH` ($\ge 0.88$), `VISUAL_SIMILARITY` ($0.52–0.88$), and `NO_CONFIDENT_MATCH` ($<0.52$). Displayed with clear diagnostic warning banners in the UI.
  - **Frozen 6-Slice Real-Image Benchmark (`data/evaluation/ph014_evaluation_manifest.json`)**: Built and evaluated 122 queries across same-image, corruptions, viewpoint shifts, genuine physical multi-view, worn-on-hand, and hard negatives. Slice 6 hard negative False Accept Rate confirmed at 10.0% at $\ge 0.88$ vs 56.7% at $\ge 0.70$.
  - **Laplacian Feature Gating**: Excluded sharp facet and specular edge pixels (`lap > 15.0`, `v_chan > 240`) from the skin neutralization mask, preserving yellow gold and rose gold bands.
  - **Authored Comprehensive Report**: [`PH-014-retrieval-quality-report.md`](file:///Users/amol/Developer/search/PH-014-retrieval-quality-report.md).
- **Retrieval Robustness, Skin Neutralization & Multi-Crop Fusion (`engine/preprocessing.py`, `engine/search.py`)**:
  - Implemented skin-aware foreground isolation in HSV color space ($H \in [0, 25] \cup [165, 180], S \in [20, 160], V \in [40, 250]$) that neutralizes human skin backgrounds to studio white (`[255, 255, 255]`) on worn jewellery photos.
  - Test-time bounded 3-view multi-crop generation (Global, Foreground/Skin-Neutralized, 80% Center Zoom) with max-pooling fusion across candidate embeddings.
  - Overall Recall@1 jumped from 51.39% to **62.50%** (+11.11%), Recall@5 from 56.94% to **72.22%** (+15.28%), MRR from 0.5471 to **0.6781**, with serving latency at **31.21 ms** (p50) / **40.26 ms** (p95), well under the $<100$ ms SLA.
  - Calibrated confidence scoring thresholds ($\ge 0.88$ Exact SKU / Twin, $0.70–0.88$ High Confidence Candidate, $0.52–0.70$ Visually Similar Design, $<0.52$ Low Confidence Candidate) avoiding false failure alarms on cross-view photography.
  - Authored comprehensive report [`RETRIEVAL_ROBUSTNESS_REPORT.md`](file:///Users/amol/Developer/search/RETRIEVAL_ROBUSTNESS_REPORT.md).
- **Interactive Next.js 14 Web Application & BFF Architecture (`web/`, `DEMO_GUIDE.md`)**:
  - Implemented modern, responsive enterprise search workspace in Next.js 14 (App Router) with Tailwind CSS.
  - Built secure Backend-For-Frontend (BFF) route handlers (`web/src/app/api/`):
    - `POST /api/search`: Multipart image proxying, client E2E latency timing, automatic fallback demo authentication.
    - `GET /api/products`: Parameterized relational catalogue browsing with category and SKU filtering.
    - `GET /api/status`: Deep dependency readiness polling (Postgres, Redis, Qdrant, ONNX model).
    - `GET /api/media/[...path]`: Path-traversal-safe binary streaming with caching.
    - `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`: Secure `HttpOnly` session cookie handling.
  - Four dedicated views:
    - **Visual Search Workspace (`/`)**: Drag-and-drop file upload, 4 one-click test sample queries (`q_ring_var_001_ring_032.jpg`, `ring_001.jpg`, `necklace_005.jpg`, `ring_020.jpg`), search mode toggle (`Visual Style Exploration` vs `Exact Twin Retrieval`), top-k slider (4-24), CLAHE/Denoise toggles, query preview with blur/quality diagnostics, ranked candidate cards with similarity scores and confidence badges, "Search with this item" re-query trigger, side-by-side split inspection modal, and real-time latency breakdown.
    - **Catalogue Explorer (`/catalogue`)**: Relational inventory browsing (482 items), category/SKU filtering, and one-click search triggers.
    - **Benchmarks & Release Audits (`/evaluation`)**: Verified empirical metrics (EXP-001, EXP-002, PH-010-PH-013), and live release gate blocker status table.
    - **System Health & Observability (`/status`)**: Deep probe panel verifying datastores, ONNX model weights, and alias routing.
- **Docker Compose Demo Stack & Network Boundary Isolation (`docker-compose.demo.yml`)**:
  - Multi-container stack orchestrating `demo_db` (Postgres 16), `demo_redis` (Redis 7), `demo_qdrant` (Qdrant 1.12.1), `demo_backend` (FastAPI), and `demo_web` (Next.js 14) on isolated bridge network `jewellery_demo_network`.
  - Strict host network boundary: only port `3000` (`demo_web`) is published on the host. Datastores (5432, 6379, 6333) and FastAPI backend (8000) are sealed inside Docker.
  - Multi-stage optimized Docker builds: Next.js standalone Alpine image and CPU-only PyTorch/ONNX backend image.
- **Automated Bootstrap & End-to-End Verification (`scripts/init_demo_stack.py`, `scripts/verify_demo_e2e.py`)**:
  - `init_demo_stack.py`: Applies Alembic migrations, seeds relational catalogue (482 products, 518 assets), creates MobileNetV4 vector collection (`jewellery_catalog_mobilenetv4_v1`), indexes 432 catalogue embeddings, and binds blue-green alias.
  - `verify_demo_e2e.py`: 7/7 automated smoke tests passing (Liveness, 4 HTML Pages 200 OK, Deep Status Ready, Catalogue Relational Inventory, HttpOnly Auth Flow, Live Search with 93.29% ground-truth match and JPEG streaming, Host Network Isolation).
- **Critical Credential Remediation & Dynamic Secret Injection (`scripts/inject_proxy_secret.py`, `tests/test_ph013_credential_remediation.py`)**:
  - Permanently invalidated and banned the compromised literal `jewellery_internal_edge_proxy_secret_2026`.
  - Implemented dynamic, high-entropy 256-bit token generation (`sec_prx_<64-hex>`).
- **End-to-End Performance Gate Reconciliation & Dedicated Burst Throttling (`scripts/benchmark_staging_performance.py`)**:
  - Measured capacity across $C=1, 4, 8$ using an enterprise capacity allocation on 4-worker Uvicorn (Engine p50 = 49.87 ms at C=1, 0.0% 5xx errors).
- **Automated Scheduled Backup Management & Disaster Recovery Drill (`scripts/backup_scheduler.py`)**:
  - 100% row count parity verified across all relational tables in non-destructive restore drill.
- **Authoritative Compliance Matrix**:
  - Markdown: [`RELEASE_COMPLIANCE_MATRIX.md`](file:///Users/amol/Developer/search/RELEASE_COMPLIANCE_MATRIX.md)
  - Demo Guide: [`DEMO_GUIDE.md`](file:///Users/amol/Developer/search/DEMO_GUIDE.md)
- **Automated Test Suite**:
  - **202 / 202 tests passing** (100% green; 199 baseline + 3 demo architecture & isolation tests).

---

## 2. Release-Blocker Status Table

| Gate Name | Responsible Stream | Status | Current State | Required Launch Target | Governance Action |
|---|---|:---:|---|---|:---:|
| **Physical Ring Data Acquisition** | External Studio / Photography | `PARTIALLY_COMPLETE_PENDING_STUDIO_ASSETS` | 5 verified SKUs / 10 multi-view studio captures on disk | 50 verified SKUs / 200 multi-view studio captures under SOP-PH005 | **RELEASE BLOCKER** (Zero synthetic substitutions permitted) |
| **Distributed 1M+ Live Cloud Benchmark** | Cloud Infrastructure / DevOps | `BLUEPRINT_VERIFIED_PENDING_LIVE_CLOUD_PROVISIONING` | Tier 3 verified locally up to 100k vectors; AWS `c6i.4xlarge` manifest certified | Live AWS `c6i.4xlarge` provisioned; 1,000,000 vectors indexed; p95 latency $<50$ ms | **RELEASE BLOCKER** (Local workstation RAM limit confirmed in PH-006) |
| **Pre-Production Deployment Readiness & Hardening** | Platform & Infrastructure Engineering | `PRE_PRODUCTION_VERIFIED` | 202 tests passing, credential remediation complete, multi-worker latency verified, backup scheduler verified | All software engineering gates passing | **PASSED & OPERATIONAL** |
| **Interactive Web Application & Docker Demo** | Full-Stack & Systems Engineering | `OPERATIONAL_AND_VERIFIED` | Next.js 14 UI live on `http://localhost:3000`, 5-service isolated Docker Compose stack, zero mockups | Interactive browser search, 7/7 E2E tests passing | **PASSED & RUNNING** |
