# Implementation Status

| Feature / Capability | Status | Evidence / Notes |
|---|---|---|
| Dataset Inventory & Verification | `VERIFIED` | 492 images audited; 0 corrupt; 3 exact duplicate sets; 84 near duplicate pairs; output in `output/dataset_report.json`. |
| Python & Dependency Setup | `VERIFIED` | Python 3.11.17 via uv; PyTorch 2.2.2, timm 1.0.30, open-clip 3.3.0, NumPy 1.26.4, OpenCV 5.0.0, FastAPI 0.143.0. |
| Global Visual Embeddings (DINOv2) | `VERIFIED` | DINOv2 (`vit_small_patch14_dinov2`) indexed; Corrected v2 R@1: 72.41%, R@5: 86.21%, MRR: 0.7753, 140ms CPU latency. |
| Global Visual Embeddings (CLIP) | `VERIFIED` | OpenCLIP `ViT-B-32` indexed; Historical v1 R@1: 43.33%, R@5: 80.00%, MRR: 0.5878, 122ms CPU latency. |
| Global & Patch Embeddings (DINOv3) | `VERIFIED` | DINOv3 (`vit_small_patch16_dinov3.lvd1689m`) indexed; Corrected v2 R@1: 68.97%, R@5: 93.10%, MRR: 0.7940, 160ms CPU latency. |
| Vector Indexing & Patch Caching | `VERIFIED` | In-memory cosine matrices and spatial patch array `dinov3_patches.npy` ($432 \times 256 \times 384$) in `data/index/`. |
| Patch Cache Integrity & Versioning | `VERIFIED` | Versioned metadata `dinov3_patches.meta.json` (v1.1.0) with manifest SHA-256 and precomputed foreground masks. |
| Query Image Preprocessing & QA | `VERIFIED` | Aspect padding, Laplacian blur gating, bilateral denoise, LAB CLAHE, auto-crop implemented in `engine/preprocessing.py`. |
| Score Normalization & Hybrid Candidate Fusion | `VERIFIED` | Calibrated Min-Max fusion $0.5 \cdot \text{norm}(v_2) + 0.5 \cdot \text{norm}(v_3)$ delivering 100.0% Candidate R@20. |
| Symmetric Chamfer Patch Reranker | `VERIFIED` | Query + candidate foreground masking, $<4$ patch fallback, spatial penalty $\lambda=0.1$, score formula $(1-pw)S_{\text{global}} + pw S_{\text{chamfer}}$. |
| Ground-Truth Annotation Studio | `VERIFIED` | Interactive HTML5 synchronized zoom/pan micro-inspection at `/annotate` and `/api/annotation/*` API endpoints. |
| Balanced Benchmark v3 | `VERIFIED` | 60-query balanced test set (30 rings, 30 necklaces), 30 hard negatives, 30 singletons (`data/evaluation/benchmark_v3_balanced.json`). |
| Confidence Calibration with Hard Negatives | `VERIFIED` | Calibrated thresholds with hard negatives: $\tau_{\text{dinov2}}=0.975, \tau_{\text{dinov3}}=0.990, \tau_{\text{minmax}}=0.970, \tau_{\text{chamfer}}=0.965$ (0.0% FAR). |
| Evaluation Protocol & Benchmarks | `VERIFIED` | Triple benchmark tracking: Historical v1 ($N=30$), Corrected v2 ($N=29$), and Balanced v3 ($N=60$). |
| Interactive Web Demonstration | `VERIFIED` | HTML5/CSS/JS UI in `frontend/index.html` with preview, quality warnings, confidence badges, bounding box overlay, and 7-stage latency timing. |
| REST API Endpoints & Search Modes | `VERIFIED` | FastAPI `/api/search` supporting `model`, `mode` (`similar`, `exact`), `patch_weight`, `spatial_weight`, and 7-stage diagnostics. |
| Component Localization Engine | `VERIFIED` | Morphological saliency (necklaces) and crown setting heuristic (rings) in `engine/localization.py` (<2ms CPU, 0.7578 Mean IoU). |
| Region Annotation Studio | `VERIFIED` | Interactive canvas drag-to-draw bounding box studio at `/region-annotate` and `/api/regions/*` REST endpoints. |
| Region-Aware Retrieval & Region Indexing | `VERIFIED` | Caching of catalog region embeddings (`dinov2_region_embeddings.npy`, `dinov3_region_embeddings.npy`) and multi-mode retrieval in `engine/region_search.py`. |
| Soft Spatial Attention Engine | `VERIFIED` | Train-free token reweighting ($16 \times 16$) in `engine/soft_attention.py` and `engine/attention_search.py` with 5 continuous weighting modes. |
| Soft Attention Heatmap Overlay | `VERIFIED` | Turbo colormap heatmap rendering (`render_attention_heatmap`) returned via `/api/search` (`attention_heatmap_b64`) and toggled in UI. |
| Sequential Latency Accounting | `VERIFIED` | Rigorous separation of sequential single-query CPU latency ($B=1$, ~181–188ms) from amortized batch throughput ($B=32$, ~22–24ms). |
| Automated Tests & Regression Suite | `VERIFIED` | 55/55 passing tests across unit, integration, hybrid correctness, validation, EXP-006 localization, and EXP-007 soft attention (`pytest tests/ -v`). |
| Baseline Reproduction & Verification (EXP-008) | `VERIFIED` | Exact reproduction across DINOv2, DINOv3, Hybrid, Soft Attention, and Hard Crop negative control in `output/exp008_baseline_verification.json`. |
| Raw Ring Provenance & Identity Audit | `VERIFIED` | Forensic audit of all 191 rings; proven only 5 natural clusters exist in raw catalog; remaining pairs verified as hard negatives. |
| Production SKU Ingestion Standard | `VERIFIED` | Documented 4-gate verification standard (ERP barcode, CAD specs, 360° turntable protocol, non-transitive isolation) in `DATASET.md`. |
| Production Acceptance Criteria Audit | `VERIFIED` | Evaluated prototype actuals against quantitative production criteria (100% Candidate R@20, 0.0% FAR, 183ms latency, 50% real R@1). |
| Prototype Closure & Hardening Roadmap | `VERIFIED` | Formal R&D prototype milestone closed; documented production hardening roadmap in `EXP-008-prototype-closure.md` and `NEXT_STEPS.md`. |
| Auditable Ingestion Manifest (PH-001) | `VERIFIED` | 518 image records with SHA-256 hashes, physical identity status, 4-gate provenance metadata in `data/ingestion_manifest.json`. |
| Manifest & Identity Integrity Checker | `VERIFIED` | Automated verification script `scripts/verify_identity_integrity.py` with 0 errors across 518 records. |
| Leakage-Safe Evaluation Protocol (PH-001) | `VERIFIED` | Identity-disjoint Calibration vs Test splits, quarantined conflicting pairs, Wilson/Clopper-Pearson CIs, output in `output/ph001_evaluation_report.json`. |
| Acceptance Gate Audit & Statistical Proof | `VERIFIED` | 4-tier readiness audit; mathematically proved that $N \ge 300$ negatives required for 95% confidence on $\le 1\%$ FAR target in `PH-001-data-provenance-audit.md`. |
| Automated Regression Suite (PH-001) | `VERIFIED` | 60/60 passing tests across unit, integration, localization, soft attention, and data foundation (`pytest tests/`). |


