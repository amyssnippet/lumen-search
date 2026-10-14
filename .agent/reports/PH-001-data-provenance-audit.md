# PH-001: Production Data and Evaluation Foundation Audit Report

**Date**: 2026-10-11  
**Lead Engineer**: Senior Computer Vision Engineer & ML Researcher  
**Status**: Completed — Production Foundation Established  
**Hardware / Execution**: Intel x86_64 CPU-only (macOS 15.7.9), Python 3.11.17 (`.venv`), PyTorch 2.2.2, NumPy 1.26.4, SciPy 1.17.1  

---

## 1. Executive Summary & Objective

The objective of **PH-001** is to start the production-hardening phase of the Jewellery Visual Retrieval Engine by establishing an auditable data foundation, preventing data leakage, and instituting statistically rigorous evaluation protocols before production deployment.

While EXP-008 successfully closed the exploratory R&D prototype milestone, it identified critical statistical and data provenance gaps:
1. **Severe Multi-View Ring Scarcity**: Only 4 real ring pairs exist in the test split; 26 of 30 ring queries were synthetic perturbations.
2. **False Accept Rate Sample Power**: Zero false accepts observed across 30 negatives was prematurely cited; statistically, $0/30$ negatives only guarantees $\text{FAR} \le 9.50\%$ with 95% confidence, failing to substantiate the production target ($\text{FAR} \le 1.0\%$).
3. **Calibration / Test Overlap & Conflicting Annotations**: In previous iterations, calibration thresholds and evaluation queries shared identical benchmark records, and two necklace pairs (`necklace_10 <-> necklace_32` and `necklace_91 <-> necklace_95`) had conflicting labels (participating as both positives and negatives).

**PH-001** resolves these foundations without modifying the frozen EXP-008 baseline:
- Built an auditable ingestion manifest ([`data/ingestion_manifest.json`](file:///Users/amol/Developer/search/data/ingestion_manifest.json)) covering all 518 images (492 raw, 26 synthetic) with SHA-256 hashes and 4-gate provenance metadata.
- Implemented automated integrity verification ([`scripts/verify_identity_integrity.py`](file:///Users/amol/Developer/search/scripts/verify_identity_integrity.py)).
- Implemented a leakage-safe evaluation harness ([`scripts/run_production_evaluation.py`](file:///Users/amol/Developer/search/scripts/run_production_evaluation.py)) with strict identity-disjoint calibration and test splits, Wilson score intervals, Clopper-Pearson FAR intervals, and explicit `INSUFFICIENT_DATA` status flags.
- Generated machine-readable audit artifacts ([`output/ph001_evaluation_report.json`](file:///Users/amol/Developer/search/output/ph001_evaluation_report.json)).
- Added automated regression tests ([`tests/test_ph001_data_foundation.py`](file:///Users/amol/Developer/search/tests/test_ph001_data_foundation.py)), bringing the full test suite to **60/60 passing tests**.

---

## 2. Workstream A — Identity Provenance & Data Ingestion Audit

### 2.1 Complete Image Inventory
The system tracks 518 total images:
- **Raw Catalog Images**: 492 images
  - `necklace`: 301 images (61.2%)
  - `ring`: 191 images (38.8%)
- **Synthetic Robustness Test Images**: 26 images (`data/evaluation/queries/rings/`)
- **Total Unique Physical SKU Identities**: 482 distinct physical identity groups
- **Genuine Independent Captures**: 488 images (492 raw minus 4 exact duplicate clones)
- **Exact Byte Clones**: 4 images (SHA-256 byte duplicates)

### 2.2 Physical Identity Verification Breakdown

| Physical Verification Status | Count | Description |
|---|:---:|---|
| `singleton_unlinked` | **446** | Catalog images representing isolated physical pieces with 0 counterpart captures in the dataset. |
| `verified_identical_sku` | **44** | Images participating in verified physical multi-view clusters (30 necklaces, 14 rings [including clones]). |
| `synthetic_transformation` | **26** | Derived geometric/photometric perturbations of catalog rings (camera angle tilts, studio lighting shifts, sensor noise). |
| `uncertain_identity` | **2** | Images with conflicting ground-truth labels (`necklace_10`, `necklace_91`) quarantined from final test splits. |
| **Total Images Audited** | **518** | **100.0% accounted for** |

### 2.3 The Four-Gate Standard Audit

Every record in `data/ingestion_manifest.json` was audited against the 4-gate standard defined in `DATASET.md`:

```
[ Ingestion Gate ]                     [ Compliance ]   [ Audit Finding ]
Gate 1: ERP/PIM Inventory Lineage          0.0%         Explicitly unassigned (sku_id=None). No manufacturer ERP metadata was provided. Zero fake SKUs invented.
Gate 2: CAD Model & Geometric Specs        0.0%         No 3D CAD/casting files accompanied the legacy catalog.
Gate 3: 360° Turntable Capture Protocol    0.0%         Legacy images are isolated single-shot photographs; no calibrated turntable session recorded.
Gate 4: Non-Transitive Pairwise Proof    100.0%         All transitive clustering chains eliminated or quarantined (e.g. ring_083 excluded). Pairwise proof required.
```

### 2.4 Real-Data Gap Analysis: Rings vs Necklaces
- **Rings ($N=191$ raw images)**:
  - An exhaustive forensic audit confirmed that the entire raw ring dataset contains **only 5 natural physical identity clusters** (10 images total):
    1. `ring_081` $\leftrightarrow$ `ring_108` (Yellow-gold diamond solitaire)
    2. `ring_155` $\leftrightarrow$ `ring_172` (Emerald-cut halo ring)
    3. `ring_080` $\leftrightarrow$ `ring_164` (Twist-shank solitaire)
    4. `ring_084` $\leftrightarrow$ `ring_181` (Split-shank pave ring)
    5. `ring_078` $\leftrightarrow$ `ring_079` (Identical comfort-fit band pair, both in catalog)
  - The remaining 181 rings are **singletons** with zero independent multi-angle physical shots.
  - In Benchmark v3, only 4 ring queries are real photographs; 26 queries are synthetic transformations.
  - **Blocker**: Validating real ring retrieval at enterprise scale ($\ge 50$ real ring queries) requires ingesting genuine external multi-angle captures under the 4-gate protocol.
- **Necklaces ($N=301$ raw images)**:
  - 3 exact duplicate sets (7 images total: `necklace_223/224/225`, `necklace_233/234`, `necklace_273/279`).
  - 28 verified multi-shot identity clusters (30 queries in Benchmark v3).
  - 25 confirmed hard negative lookalike pairs.
  - 2 conflicting pairs identified and quarantined.

---

## 3. Workstream B — Leakage-Safe Evaluation Protocol

### 3.1 EXP-008 Frozen Baseline Reproduction
The frozen EXP-008 baseline on Benchmark v3 ($N=60$) was reproduced identically from recorded artifacts and scripts:

| Configuration | Mode | Overall R@1 ($N=60$) | Real R@1 ($N=34$) | Syn R@1 ($N=26$) | Overall R@5 | Candidate R@20 | MRR | Latency p50 (ms) | Reproduction Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DINOv3 Global** | Unconstrained | 70.0% (42/60) | 47.1% (16/34) | 100.0% (26/26) | 93.3% (56/60) | 100.0% (60/60) | 0.8059 | 185.2 ms | `EXACT MATCH` |
| **DINOv3 Global** | Category Filtered | 71.7% (43/60) | 50.0% (17/34) | 100.0% (26/26) | 93.3% (56/60) | 100.0% (60/60) | 0.8142 | 187.2 ms | `EXACT MATCH` |
| **DINOv2 Global** | Unconstrained | 68.3% (41/60) | 44.1% (15/34) | 100.0% (26/26) | 90.0% (54/60) | 98.3% (59/60) | 0.7731 | 169.9 ms | `EXACT MATCH` |
| **DINOv2 Global** | Category Filtered | 70.0% (42/60) | 47.1% (16/34) | 100.0% (26/26) | 90.0% (54/60) | 98.3% (59/60) | 0.7815 | 169.4 ms | `EXACT MATCH` |

### 3.2 PH-001 Leakage-Safe Partitioned Protocol
To eliminate threshold overfitting and identity leakage, queries and lookalikes were partitioned into strictly disjoint subsets:
- **Exploratory Calibration Split**: 11 positive queries (disjoint clusters) and 23 lookalike negatives. Used exclusively to fit confidence thresholds:
  - $\tau_{\text{exact}} = 0.9909$ (calibrated against maximum negative score $0.9859$)
  - $\tau_{\text{high\_confidence}} = 0.9900$
- **Held-Out Test Split**: 47 positive queries, 7 lookalike negatives, and 30 singletons.
  - Zero identity cluster overlap with calibration ($\text{overlap} = \emptyset$).
  - Evaluated strictly using the frozen calibration thresholds.
- **Quarantined Discrepancy Split**: 2 pairs (`necklace_10 <-> necklace_32` and `necklace_91 <-> necklace_95`) isolated due to conflicting ground-truth annotations.

### 3.3 Held-Out Test Evaluation Matrix (with Wilson 95% Confidence Intervals)

| Subgroup | Queries ($N$) | Unique SKUs | Independent Captures | Recall@1 [95% CI] | Recall@5 [95% CI] | Candidate Recall@20 [95% CI] | MRR | Exact Coverage at $\tau_{\text{exact}}$ | Statistical Power Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **All Held-Out Test Queries** | 47 | 45 | 47 | **68.1%** (32/47)<br>[53.8%, 79.6%] | **93.6%** (44/47)<br>[82.8%, 97.8%] | **100.0%** (47/47)<br>[92.4%, 100.0%] | **0.7933** | 36.2% (17/47) | `SUFFICIENT` ($N \ge 30$) |
| **Real Photographic Captures** | 26 | 24 | 26 | **42.3%** (11/26)<br>[25.5%, 61.1%] | **88.5%** (23/26)<br>[71.0%, 96.0%] | **100.0%** (26/26)<br>[87.1%, 100.0%] | **0.6263** | 19.2% (5/26) | `INSUFFICIENT_DATA` ($N=26 < 30$) |
| **Real Photographic Necklaces** | 22 | 20 | 22 | **45.5%** (10/22)<br>[26.9%, 65.3%] | **86.4%** (19/22)<br>[66.7%, 95.3%] | **100.0%** (22/22)<br>[85.1%, 100.0%] | **0.6265** | 22.7% (5/22) | `INSUFFICIENT_DATA` ($N=22 < 30$) |
| **Real Photographic Rings** | 4 | 4 | 4 | **25.0%** (1/4)<br>[4.6%, 69.9%] | **100.0%** (4/4)<br>[51.0%, 100.0%] | **100.0%** (4/4)<br>[51.0%, 100.0%] | **0.6250** | 0.0% (0/4) | `INSUFFICIENT_DATA` ($N=4 \ll 30$) |
| **Synthetic Robustness Rings** | 21 | 21 | 21 | **100.0%** (21/21)<br>[84.5%, 100.0%] | **100.0%** (21/21)<br>[84.5%, 100.0%] | **100.0%** (21/21)<br>[84.5%, 100.0%] | **1.0000** | 57.1% (12/21) | `INSUFFICIENT_DATA` ($N=21 < 30$) |

### 3.4 Quarantined Discrepancy Case Study
- `necklace_10` $\leftrightarrow$ `necklace_32`: Originally clustered by dHash $\le 2$ because both feature drop pendants on round chains. However, manual inspection revealed that `necklace_10` features a three-stone tiered cluster while `necklace_32` features a solitaire bezel drop. Global DINOv3 similarity is only $0.9355$, and retrieval returns `necklace_38` at rank 1 ($0.9862$). Quarantining this pair prevented evaluation corruption.
- `necklace_91` $\leftrightarrow$ `necklace_95`: Both feature floral cluster pendants, but with differing melee petal stone counts. DINOv3 similarity is $0.9859$, successfully retrieving `necklace_95` at rank 1, but quarantined due to conflicting ground-truth annotations.

### 3.5 Stage-Level Single-Query CPU Latency Breakdown
Measured sequentially ($B=1$) via `time.perf_counter()` on Intel x86_64 CPU:
1. Query Image Read & Aspect Padding: **1.47 ms**
2. Quality Assessment & Exposure Check: **0.02 ms**
3. DINOv3 Neural Forward Pass (ViT-S/16): **187.53 ms**
4. L2 Feature Normalization: **0.07 ms**
5. Vector Similarity Dot-Product ($N=432$): **0.33 ms**
6. Thresholding & Category-Filtered Sorting: **0.06 ms**
- **Total Sequential Single-Query Latency**: **p50 = 189.53 ms (p95 = 266.12 ms)**.

---

## 4. Workstream C — Production Acceptance Gate Audit

### 4.1 Four-Tier Production Readiness Checklist

| Acceptance Gate | Production Target | Measured Prototype Metric | Compliance Status | Gap Analysis & Remediation |
|---|:---:|:---:|:---:|---|
| **Tier 1: Prototype Acceptance** | Functional end-to-end API & UI | 55/55 tests passing, FastAPI + Web UI verified | `PASSED` | Completed in EXP-008. Search, region localization, and soft attention heatmaps operational. |
| **Tier 2: Real-World Retrieval Quality** | Real Photographic Top-1 $\ge 95.0\%$ | Real R@1: **42.3% – 50.0%** | `NOT_PASSED` | Significant 45–50% gap. Constrained by lack of multi-view real training data. Candidate Recall@20 is 100.0%, confirming the correct item is always in the candidate set. Fine metric learning required. |
| **Tier 2: Candidate Recall** | Candidate Recall@20 $\ge 99.0\%$ | Candidate R@20: **100.0%** (47/47) | `PASSED` | Exceeds target with 95% confidence interval [92.4%, 100.0%]. |
| **Tier 2: Ring Dataset Sample Power** | Real Ring Queries $N \ge 30$ | Real Ring Queries: **$N=4$** | `NOT_PASSED` | Sample size severely underpowered ($N=4$). 95% CI is [4.6%, 69.9%]. Requires ingesting $\ge 50$ real physical ring captures. |
| **Tier 2: False Accept Rate Validation** | $\text{FAR} \le 1.0\%$ on Lookalikes | Observed: **0/7** (0.0%), 95% CP upper bound: **40.96%** | `NOT_STATISTICALLY_VALIDATED` | $N=7$ held-out negatives (or $N=30$ overall) cannot substantiate $\le 1.0\%$ FAR. Sample size must be expanded to $N \ge 300$. |
| **Tier 3: Query Latency (CPU)** | Sequential p50 $< 100.0$ ms | Single-query p50: **189.5 ms** | `NOT_PASSED` | Exceeds target by ~90 ms on CPU. Remediated via planned ONNX INT8 dynamic quantization targeting <40 ms. |
| **Tier 3: Catalog Scalability** | Sub-millisecond lookup on 1M items | In-memory dot-product ($N=432$): **0.33 ms** | `IN_PROGRESS` | Scales linearly ($O(N)$). Scaling beyond 50,000 items requires HNSW vector database (Qdrant/Milvus). |
| **Tier 4: Security & Input Hygiene** | 100% path traversal & corrupted file rejection | 100% rejected (tested in regression suite) | `PASSED` | Verified by `test_media_path_traversal_protection` and `test_search_endpoint_invalid_file_rejected`. |
| **Tier 4: Enterprise Operations** | Multi-tenant auth, rate limiting, Prometheus | Not yet implemented | `NOT_PASSED` | Scheduled for production deployment phase (FastAPI middleware, Redis token bucket). |

### 4.2 Statistical Proof: Sample Size Requirements for the $\le 1.0\%$ FAR Target
Observing zero false accepts across 30 negative lookalikes does **not** prove the system meets the $\le 1.0\%$ False Accept Rate target.

By the exact Clopper-Pearson binomial distribution, when $k=0$ false accepts are observed out of $n$ independent negative trials at $95\%$ confidence ($\alpha=0.05$):
$$\text{Upper Bound} = 1 - (\alpha/2)^{1/n} = 1 - (0.025)^{1/n}$$

- For $n = 7$ (held-out test negatives): Upper bound = $1 - 0.025^{1/7} = \mathbf{40.96\%}$.
- For $n = 30$ (all benchmark negatives): Upper bound = $1 - 0.025^{1/30} = \mathbf{11.59\%}$ (one-sided: $1 - 0.05^{1/30} = \mathbf{9.50\%}$).
- **Conclusion**: Observing $0/30$ false accepts is statistically consistent with a true population error rate as high as $9.5\%$, which is nearly $10\times$ higher than the production target ($1.0\%$).

#### Sample Size Required for Statistical Validation:
To guarantee that the true population FAR is $\le 1.0\%$ with $95\%$ statistical confidence when zero false accepts are observed:
$$(1 - 0.01)^N \le 0.05 \implies N \ln(0.99) \le \ln(0.05) \implies N \ge \frac{-2.9957}{-0.01005} \approx \mathbf{299}$$

**Mandate**: Validating the production $\le 1.0\%$ FAR target requires compiling a dedicated evaluation benchmark containing at least **$N \ge 300$ verified lookalike negative pairs** without a single false accept.

---

## 5. Deliverables Summary

1. **Auditable Ingestion Manifest**: [`data/ingestion_manifest.json`](file:///Users/amol/Developer/search/data/ingestion_manifest.json) containing all 518 images with SHA-256 hashes, physical identity statuses, and 4-gate provenance metadata.
2. **Automated Manifest & Leakage Integrity Checker**: [`scripts/verify_identity_integrity.py`](file:///Users/amol/Developer/search/scripts/verify_identity_integrity.py) (0 errors across 518 records).
3. **Reproducible Production Evaluation Harness**: [`scripts/run_production_evaluation.py`](file:///Users/amol/Developer/search/scripts/run_production_evaluation.py) generating [`output/ph001_evaluation_report.json`](file:///Users/amol/Developer/search/output/ph001_evaluation_report.json).
4. **Automated Regression Suite**: [`tests/test_ph001_data_foundation.py`](file:///Users/amol/Developer/search/tests/test_ph001_data_foundation.py) passing 5/5 tests; full repository regression suite passing **60/60 tests** (`pytest tests/`).

---

## 6. Next Milestone: PH-002

With the data and evaluation foundation solidified, the next immediate milestone is:
**PH-002: Real Multi-View Ring Ingestion and Quantization Feasibility Audit**
- Ingest genuine multi-angle physical ring captures to eliminate the $N=4$ sample size limitation under the 4-gate protocol.
- Benchmark ONNX Runtime INT8 dynamic quantization on Intel x86_64 CPU to empirically measure latency reduction towards the $<100$ ms target.
