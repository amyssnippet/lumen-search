# EXP-006: Gemstone and Pendant Localization for Fine Jewellery Retrieval

**Date**: 2026-10-11  
**Lead Engineer**: Senior Computer Vision Engineer & ML Researcher  
**Status**: Completed  
**Hardware / Execution**: Intel x86_64 CPU-only (macOS 15.7.9), Python 3.11.17 (`.venv`)  

---

## 1. Executive Summary & Objective

In fine jewellery retrieval, global visual representations can sometimes be dominated by necklace chain silhouettes, photographic backgrounds, or ring shanks, obscuring subtle differences in the central gemstone cluster or pendant setting. 

EXP-006 investigated whether **component-level localization** (isolating necklace pendants and ring crown settings) improves exact-piece retrieval over full-image representations.

### Key Empirical Findings:
1. **Off-the-Shelf COCO Detectors Fail Completely**: Generic object detectors (e.g. SSDLite MobileNetV3) lack jewellery domain categories, hallucinating "mouse" (0.97 confidence) on ring shanks and "scissors" or "clock" on necklace loops.
2. **Morphological Saliency & Crown Heuristics are Fast & Accurate**: A CPU-efficient (<2ms) morphological opening with an elliptical kernel removes thin chain lines to isolate dense pendant blobs, while crown setting heuristics isolate ring heads, achieving **0.7578 Mean IoU** across the 60 benchmark queries (0.7056 on necklaces, 0.8101 on rings).
3. **Region-Only Retrieval Underperforms Full-Image Retrieval**:
   - DINOv2: Full-image Recall@1 is **70.0%**, whereas Region-only Recall@1 drops to **60.0%**.
   - DINOv3: Full-image Recall@1 is **71.7%**, whereas Region-only Recall@1 drops to **63.3%**.
4. **Oracle Upper Bound Disproves Isolated Crop Superiority**: Even using human-drawn, perfectly verified ground-truth bounding boxes (Oracle Upper Bound), Recall@1 is **65.0%** (DINOv2) and **68.3%** (DINOv3). Discarding global context (chain link style, clasp, proportions, shank metal) removes essential discriminative entropy: many distinct jewellery pieces share near-identical solitary stones or generic pendant drops.
5. **Real-Image vs Synthetic Robustness Divergence**:
   - On the 26 Synthetic ring queries (perspective homography, lighting shifts, sensor noise), both Full-Image and Combined models achieve **96.2%–100.0% Recall@1**.
   - On the 34 Real-image queries (independent physical captures), Recall@1 is **47.1% (DINOv2)** and **50.0% (DINOv3)**.
   - Synthetic perturbations test photometric and geometric robustness against identical images, not SKU generalization across distinct physical photographs.

---

## 2. Phase 1: Benchmark Provenance & Validity Audit

We audited all 60 queries in `data/evaluation/benchmark_v3_balanced.json` to strictly separate real photographic captures from synthetic transformations:

| Query Sub-Group | Count ($N$) | Category | Source / Provenance | SKU Generalization Validity |
|---|:---:|:---:|---|---|
| **Real Necklaces** | 30 | `necklace` | Multi-shot clusters from `Jewellery_Data/necklace/` (e.g. `necklace_10` -> `necklace_32`) | **High**: Distinct captures of identical physical pieces |
| **Real Rings** | 4 | `ring` | Multi-shot clusters from `Jewellery_Data/ring/` (`ring_081`, `ring_155`, `ring_080`, `ring_084`) | **High**: Independent captures of physical rings |
| **Synthetic Rings** | 26 | `ring` | Controlled 3D homography tilts, studio lighting shifts, smartphone noise, recompression | **Robustness Only**: Derived from same catalog images; does not prove physical SKU generalization |
| **Hard Negatives** | 30 | mixed | Lookalike items of different physical SKUs | **High**: Evaluates false-positive rejection |
| **Singletons** | 30 | mixed | Unique catalog items with no target twin | **High**: Evaluates confidence calibration |

---

## 3. Phase 2: Localization & Annotation Workflow

We implemented an end-to-end component annotation studio and automated heuristic detector:
1. **Interactive Studio (`/region-annotate`)**:
   - Canvas-based drag-and-draw viewport with zoom/pan and coordinate normalization `[ymin, xmin, ymax, xmax]`.
   - Component classes: `pendant_stone`, `centre_stone`, `halo_stones`, `full_piece_only`, `uncertain_occluded`.
   - Heuristic proposal auto-population via `POST /api/regions/propose`.
   - Structured JSON persistence at `data/region_annotations.json`.
2. **REST API Endpoints**:
   - `GET /region-annotate`: Serves studio HTML.
   - `GET /api/regions/list`: Lists benchmark queries with existing annotations.
   - `GET /api/regions/{image_id}`: Retrieves specific annotation metadata.
   - `POST /api/regions/save`: Saves expert ground truth coordinates and notes.
   - `POST /api/regions/propose`: Executes `JewelleryLocalizer` in <2ms on CPU.

---

## 4. Phase 3: Localization Architecture Comparison

| Approach | Implementation | Latency (CPU) | Jewellery Stone Detection Quality | Failure Modes |
|---|---|:---:|:---:|---|
| **Pretrained Object Detector (COCO)** | `ssdlite320_mobilenet_v3_large` | 42.5 ms | **0% (Complete Failure)** | Hallucinated `mouse` (0.97 conf), `scissors`, `clock`. Zero jewellery classes. |
| **Morphological Saliency (Necklaces)** | 7x7 elliptical opening + vertical depth weighting | 1.8 ms | **High (IoU = 0.7056)** | Solid collar chokers without thin chains default to full piece contour. |
| **Crown Setting Heuristic (Rings)** | Upper 45% crown setting bounding box | 1.2 ms | **High (IoU = 0.8101)** | Flat-lay / top-down ring perspectives without distinct shank orientation. |
| **Ground-Truth ROI Oracle** | Human expert annotation | N/A (Manual) | **1.0000 IoU (Upper Bound)** | Human annotation effort required. |

---

## 5. Phase 4 & 5: Comparative Retrieval Evaluation

All 432 catalog items were indexed for both full-image and localized region crops. Region embeddings were cached in `data/index/dinov2_region_embeddings.npy` and `data/index/dinov3_region_embeddings.npy`.

We evaluated 8 configurations across both DINOv2 and DINOv3 on the 60 benchmark queries:

### Comprehensive Evaluation Results Table:

| Model | Configuration | Overall R@1 ($N=60$) | Real R@1 ($N=34$) | Real Neck R@1 ($N=30$) | Real Ring R@1 ($N=4$) | Syn Ring R@1 ($N=26$) | Overall R@5 | Overall MRR | Mean IoU | Single-Query Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DINOv2** | Full-Image (Baseline A) | 70.0% (42/60) | 47.1% (16/34) | 50.0% (15/30) | 25.0% (1/4) | **100.0% (26/26)** | 90.0% | 0.7815 | — | 22.4 ms |
| **DINOv2** | Region-Only (Baseline B) | 60.0% (36/60) | 32.4% (11/34) | 33.3% (10/30) | 25.0% (1/4) | 96.2% (25/26) | 75.0% | 0.6733 | 0.7578 | 24.2 ms |
| **DINOv2** | Combined $w=0.10$ | 66.7% (40/60) | 41.2% (14/34) | 43.3% (13/30) | 25.0% (1/4) | **100.0% (26/26)** | 88.3% | 0.7634 | 0.7578 | 24.3 ms |
| **DINOv2** | Combined $w=0.20$ | 65.0% (39/60) | 38.2% (13/34) | 40.0% (12/30) | 25.0% (1/4) | **100.0% (26/26)** | 88.3% | 0.7485 | 0.7578 | 24.3 ms |
| **DINOv2** | Combined $w=0.35$ (Config C) | 63.3% (38/60) | 35.3% (12/34) | 36.7% (11/30) | 25.0% (1/4) | **100.0% (26/26)** | 85.0% | 0.7383 | 0.7578 | 24.3 ms |
| **DINOv2** | Combined $w=0.50$ | 60.0% (36/60) | 29.4% (10/34) | 30.0% (9/30) | 25.0% (1/4) | **100.0% (26/26)** | 81.7% | 0.7032 | 0.7578 | 24.3 ms |
| **DINOv2** | Region Rerank Top-20 (Config D) | 61.7% (37/60) | 32.4% (11/34) | 33.3% (10/30) | 25.0% (1/4) | **100.0% (26/26)** | 86.7% | 0.7328 | 0.7578 | 24.5 ms |
| **DINOv2** | Oracle GT ROI (Config E) | 65.0% (39/60) | 38.2% (13/34) | 40.0% (12/30) | 25.0% (1/4) | **100.0% (26/26)** | 85.0% | 0.7437 | 1.0000 | 24.4 ms |
| **DINOv3** | Full-Image (Baseline A) | **71.7% (43/60)** | **50.0% (17/34)** | **53.3% (16/30)** | 25.0% (1/4) | **100.0% (26/26)** | **93.3%** | **0.8142** | — | 23.1 ms |
| **DINOv3** | Region-Only (Baseline B) | 63.3% (38/60) | 38.2% (13/34) | 40.0% (12/30) | 25.0% (1/4) | 96.2% (25/26) | 76.7% | 0.6938 | 0.7578 | 24.8 ms |
| **DINOv3** | Combined $w=0.10$ | 70.0% (42/60) | 47.1% (16/34) | 50.0% (15/30) | 25.0% (1/4) | **100.0% (26/26)** | **93.3%** | 0.8049 | 0.7578 | 25.0 ms |
| **DINOv3** | Combined $w=0.20$ | 68.3% (41/60) | 47.1% (16/34) | 50.0% (15/30) | 25.0% (1/4) | 96.2% (25/26) | **93.3%** | 0.7893 | 0.7578 | 24.9 ms |
| **DINOv3** | Combined $w=0.35$ (Config C) | 65.0% (39/60) | 41.2% (14/34) | 43.3% (13/30) | 25.0% (1/4) | 96.2% (25/26) | 90.0% | 0.7658 | 0.7578 | 24.9 ms |
| **DINOv3** | Combined $w=0.50$ | 66.7% (40/60) | 44.1% (15/34) | 46.7% (14/30) | 25.0% (1/4) | 96.2% (25/26) | 86.7% | 0.7535 | 0.7578 | 24.9 ms |
| **DINOv3** | Region Rerank Top-20 (Config D) | 65.0% (39/60) | 41.2% (14/34) | 43.3% (13/30) | 25.0% (1/4) | 96.2% (25/26) | 91.7% | 0.7695 | 0.7578 | 25.1 ms |
| **DINOv3** | Oracle GT ROI (Config E) | 68.3% (41/60) | 47.1% (16/34) | 50.0% (15/30) | 25.0% (1/4) | 96.2% (25/26) | 88.3% | 0.7782 | 1.0000 | 25.0 ms |

---

## 6. Scientific Diagnosis: Why Isolated Region Retrieval Underperforms

1. **Contextual Entropy Loss**:
   When an image is cropped down to only the pendant or centre gemstone, the global geometry (chain thickness, weave pattern, clasp shape, shank metal curvature, proportions) is discarded. In a fine jewellery catalog, hundreds of pieces share standard oval cut solitaires or four-prong round brilliants. The gemstone alone has lower visual uniqueness than the complete physical assembly.
2. **Oracle Ceiling Lower than Full-Image**:
   Even with 100% human-verified ground-truth bounding boxes, the Oracle configuration reaches 65.0% (DINOv2) and 68.3% (DINOv3), trailing Full-Image (70.0% / 71.7%). This proves the degradation is **not** an artifact of noisy bounding boxes, but an inherent loss of information from isolated cropping.
3. **Contrast with Token-Level Chamfer Matching (EXP-004 / EXP-005)**:
   In EXP-004.1 and EXP-005, local patch Chamfer matching improved Recall@1 from 70.0% to 71.67%. Chamfer matching succeeds because it preserves the global spatial coordinate grid while attending to fine patch correspondences. In contrast, hard cropping zooms into the stone, discarding the global context entirely.

---

## 7. Production Architectural Recommendations

1. **Keep Global Representation as Primary Top-1 Gate**: Full-image DINOv3 with calibrated score fusion remains the optimal automated candidate generator and top-1 ranker.
2. **Deploy Component Localization for User Interaction & Explainability**:
   - The UI now renders the localized component bounding box overlay directly on the uploaded query image.
   - Users can see what region was detected (pendant or crown setting) with confidence scores.
3. **Interactive "Search by Component" Mode**: When a user explicitly wants to match a loose stone or detached pendant setting, localized crop retrieval can be enabled as an explicit opt-in filter.
4. **Automated Fallback Architecture**:
   - Whenever foreground saliency is below 0.2% or an image is degenerate (e.g. solid white backdrop), `JewelleryLocalizer` triggers `fallback_centre_crop` with $confidence=0.40$.
   - In combined retrieval, fallback flags automatically set $w_{reg}=0.0$, preventing noisy crops from harming retrieval.

---

## 8. Verification & Test Suite Status

- **Automated Tests**: **43 passed / 0 failed** in `pytest tests/ -v`.
- **New Unit Tests (`tests/test_exp006_localization.py`)**:
  - Exact IoU, zero-overlap IoU, partial IoU, and degenerate zero-area bounding boxes.
  - White backdrop graceful fallback.
  - Necklace pendant morphological contour isolation.
  - Ring crown setting isolation.
  - REST API region endpoints (`/region-annotate`, `/api/regions/list`, `/api/regions/propose`, `/api/regions/save`, `/api/regions/{image_id}`).
