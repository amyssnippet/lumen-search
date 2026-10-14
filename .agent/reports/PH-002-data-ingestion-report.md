# PH-002: Real Multi-View Ingestion & Quantization Feasibility Audit Report

**Date**: 2026-10-11  
**Lead Engineer**: Senior Computer Vision Engineer & ML Researcher  
**Status**: Completed — Production Audit & Feasibility Benchmark Established  
**Hardware / Execution**: Intel x86_64 CPU-only (macOS 15.7.9, 8 logical cores), Python 3.11.17 (`.venv`), PyTorch 2.2.2, ONNX Runtime 1.23.2, ONNX 1.23.2, NumPy 1.26.4, SciPy 1.17.1  

---

## 1. Executive Summary & Audit Scope

The objective of **PH-002** is to strengthen the production evidence for the Jewellery Visual Retrieval Engine without reopening exploratory model search or compromising the frozen EXP-008 baseline. Specifically:
1. **Gate 0 — Reconcile PH-001 Counts**: Recompute all manifest numbers directly from disk records, forensically trace every discrepancy between independent captures, identity clusters, singletons, and multi-shots, and implement symmetrical accounting rules.
2. **Workstream A — Real Multi-View Ring Ingestion & Inventory Audit**: Audit all available physical ring data against the 4-gate verification standard, document genuine multi-angle physical SKUs, and prevent synthetic variations from masquerading as physical pieces.
3. **Workstream B — ONNX Runtime INT8 Feasibility**: Export the reference DINOv3 ViT-Small model to ONNX, benchmark dynamic INT8 CPU quantization, resolve export/quantization runtime errors, verify numerical fidelity and full retrieval metrics against the frozen baseline, and measure sequential single-query latency ($B=1$).
4. **Workstream C — Hard-Negative Validation ($N \ge 300$)**: Compile a statistically powered evaluation pool of 300 independent lookalike negative query cases across distinct physical SKUs, test against predeclared frozen acceptance thresholds, compute Clopper-Pearson exact binomial confidence bounds, and document open-set population limitations.

---

## 2. Gate 0 — Reconcile PH-001 Counts & Forensic Accounting

### 2.1 Reconciled Ground-Truth Accounting
The audit recomputed all manifest records directly from [`data/ingestion_manifest.json`](file:///Users/amol/Developer/search/data/ingestion_manifest.json) and generated a complete machine-readable reconciliation at [`output/ph002_count_reconciliation.json`](file:///Users/amol/Developer/search/output/ph002_count_reconciliation.json):

| Inventory Category | Total Images | Independent Captures | Clones / Synthetic | Physical Identity Clusters | Category Breakdown |
|---|:---:|:---:|:---:|:---:|---|
| **Raw Necklaces** | 301 | 297 | 4 clones (exact byte duplicates) | 270 physical clusters | 242 singletons, 48 multi-shot (23 clusters), 7 exact dups (3 sets), 4 quarantined (2 pairs) |
| **Raw Rings** | 191 | 191 | 0 clones | 186 physical clusters | 181 singletons, 10 multi-shot (5 clusters) |
| **Raw Catalog Subtotal** | **492** | **488** | **4 byte clones** | **456 physical clusters** | **423 singletons + 28 multi-shot + 3 dup sets + 2 quarantined pairs = 456 clusters** |
| **Synthetic Ring Variants** | 26 | 0 | 26 synthetic perturbations | 26 synthetic query clusters | Dedicated `SYNTH-VAR-GROUP-001` to `026` groups |
| **Total Tracked Images** | **518** | **488** | **30 non-independent** | **482 total clusters** | **456 physical clusters + 26 synthetic test clusters = 482 clusters** |

### 2.2 Forensic Trace of PH-001 Discrepancies
The forensic investigation traced the discrepancies in PH-001 to four specific root causes in `build_ingestion_manifest.py`:

1. **Discrepancy 1: Unlinked Necklace Targets (21 records)**:
   - *Root Cause*: `build_ingestion_manifest.py` searched for counterpart links using `for c in catalog_items`. For 21 catalog targets, their counterparts were benchmark queries (not in `catalog_items`), causing the check to evaluate to `False`. Those 21 catalog targets were miscategorized as `singleton_unlinked` with individual `PHYS-SINGLE-*` IDs instead of being linked with their queries.
   - *Resolution*: Implemented symmetrical graph connected-component clustering across all verified annotations regardless of whether an item is indexed in the catalog or evaluated as a query.
2. **Discrepancy 2: Quarantined Targets Miscategorized as Singletons (2 records)**:
   - *Root Cause*: Conflicting pairs (`necklace_10 <-> necklace_32` and `necklace_91 <-> necklace_95`) only flagged the query images (`necklace_10`, `necklace_91`) as `uncertain_identity`. The catalog targets (`necklace_32`, `necklace_95`) were misclassified as `singleton_unlinked`.
   - *Resolution*: Both members of quarantined conflicting pairs now receive `physical_identity_verification_status = "uncertain_identity"` and shared group IDs (`PHYS-UNCERTAIN-necklace_10_32`, `PHYS-UNCERTAIN-necklace_91_95`).
3. **Discrepancy 3: Synthetic Variation Contamination of Physical Groups (26 records)**:
   - *Root Cause*: Synthetic variation images were assigned `identity_group_id = f"PHYS-SINGLE-{source_item}"`, which erroneously merged 25 synthetic variations into physical singleton groups, obscuring true physical cluster counts.
   - *Resolution*: Assigned synthetic variations dedicated synthetic cluster IDs (`SYNTH-VAR-GROUP-001` to `026`), guaranteeing zero group leakage between synthetic queries and physical catalog items.
4. **Discrepancy 4: The Erroneous Report Claim of "44 Images Across 22 Groups"**:
   - *Root Cause*: The PH-001 text quoted `status_breakdown["verified_identical_sku"] = 44` and assumed $44 / 2 = 22$ pairs. In reality, in the manifest, 23 of those 44 records were in groups of size 1 (unlinked queries), 18 were in groups of size 2, and 3 were in a group of size 3 (33 distinct groups total).
   - *Mathematical Proof of Previous 482 Count*: The 482 clusters reported in PH-001 resulted from an exact mathematical cancellation:
     $$456 \text{ (true physical)} + 23 \text{ (split queries)} + 2 \text{ (split quarantined)} - 25 \text{ (synthetic merged)} + 26 \text{ (synthetic groups)} = \mathbf{482}$$
     After the fix, the true physical clusters are **456**, and total clusters (including 26 synthetic) are **482**.

---

## 3. Workstream A — Real Multi-View Ring Ingestion & Inventory Audit

### 3.1 Four-Gate Audit of Available Ring Corpus
An exhaustive forensic scan of the local filesystem was conducted to identify any external physical jewellery archives:
- Scanned `/Users/amol/Downloads/poc-main/`:
  - `new_data/image/` contains 515 items, but only 18 ring images (`CRN00143` to `CRN00532`).
  - Analysis confirmed that all 18 images represent distinct singleton SKUs without multiple angles, turntable sessions, or CAD/ERP lineage.
  - `poc_inventory/` contains 10,000 synthetic or generic items (`NAT_00000.jpg` to `NAT_10000.jpg`) without SKU identity linkages.
- Scanned `Jewellery_Data/ring/` ($N=191$ raw images):
  - Exactly **5 natural physical identity clusters** exist (10 images total):
    1. `ring_081` $\leftrightarrow$ `ring_108` (Yellow-gold solitaire, crown and 45° angle)
    2. `ring_155` $\leftrightarrow$ `ring_172` (Emerald-cut halo ring, profile and top angle)
    3. `ring_080` $\leftrightarrow$ `ring_164` (Twist-shank solitaire, 45° and front angle)
    4. `ring_084` $\leftrightarrow$ `ring_181` (Split-shank pave ring, 45° and top angle)
    5. `ring_078` $\leftrightarrow$ `ring_079` (Comfort-fit wedding band pair, both in catalog)

### 3.2 Real Ring Inventory Summary
Following the requirement to report the actual physical count rather than filling gaps with synthetic variations:

| Metric | Measured Value | Compliance Status & Evidence |
|---|:---:|---|
| **Verified Physical Ring SKUs** | **5 SKUs** | Exactly 5 distinct physical pieces verified by pairwise non-transitive audit. |
| **Photographs per SKU** | **2.0** | Each verified ring SKU has exactly 2 independent photographs. |
| **Capture Sessions** | **2 sessions** | 4 pairs in benchmark test split (Session 1); 1 pair catalog-only (Session 2). |
| **Usable Query/Gallery Pairs** | **4 pairs** | In Benchmark v3 (`ring_081`, `ring_155`, `ring_080`, `ring_084`). |
| **Excluded Ring Singletons** | **181 items** | Single-shot captures with zero counterpart photographs in dataset. |
| **Disputed / Quarantined Rings** | **1 item** | `ring_083` excluded due to EXP-004 transitive error (solitaire chained to pave). |
| **Synthetic Variations Excluded** | **26 items** | Evaluated strictly as photometric/geometric robustness tests, not physical SKUs. |

**Conclusion**: The target of $\ge 50$ physical ring SKUs cannot be satisfied from the available local corpus without ingesting a new external physical capture dataset under the 4-gate standard. The actual verified count of **5 physical ring SKUs** is recorded with complete statistical integrity.

---

## 4. Workstream B — ONNX Runtime INT8 Feasibility Benchmark

### 4.1 Export & Quantization Pipeline
1. **PyTorch Reference**: `vit_small_patch16_dinov3.lvd1689m` ($256 \times 256$, embedding dimension 384).
2. **ONNX FP32 Export**: Exported with Opset 17, dynamic batch size, and L2 feature normalization wrapped into the computational graph ([`data/models/dinov3_vits16_fp32.onnx`](file:///Users/amol/Developer/search/data/models/dinov3_vits16_fp32.onnx)).
3. **Quantization Failure & Resolution**:
   - *Failure*: Default `quantize_dynamic` attempted to quantize the initial patch projection convolution to `ConvInteger(10)`. ONNX Runtime's CPUExecutionProvider failed initialization with `NotImplementedError: Could not find an implementation for ConvInteger(10) node with name '/m/patch_embed/proj/Conv_quant'`.
   - *Resolution*: Restricting `op_types_to_quantize=["MatMul", "Gemm"]` targeted the heavy Transformer multi-head attention and feed-forward weight matrices while keeping the patch projection in FP32. Generated [`data/models/dinov3_vits16_int8.onnx`](file:///Users/amol/Developer/search/data/models/dinov3_vits16_int8.onnx).

### 4.2 Model Size, Memory, and Numerical Fidelity

| Configuration | Model File Size | Compression Ratio | Peak Runtime RSS | Mean Cosine Sim vs PyTorch FP32 | Top-1 Ranking Agreement | Top-5 Mean Jaccard Overlap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PyTorch FP32** | ~85 MB (weights) | 1.00x | 380 MB | 1.000000 | 100.0% | 100.0% |
| **ONNX FP32** | 82.57 MB | 1.03x | 412 MB | 1.000000 (bit-exact) | 100.0% | 100.0% |
| **ONNX INT8** | **21.97 MB** | **3.76x** | **435 MB** | **0.939952** | **81.67%** (49/60) | **53.76%** |

### 4.3 Retrieval Metrics on Balanced Benchmark v3 ($N=60$)

| Vision Model & Quantization | Category Filter | Overall R@1 ($N=60$) | Real R@1 ($N=34$) | Syn R@1 ($N=26$) | Candidate R@20 ($N=60$) | MRR | Accuracy Finding |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **DINOv3 PyTorch FP32** | Unconstrained | 70.0% (42/60) | 47.1% (16/34) | 100.0% (26/26) | 100.0% (60/60) | 0.8059 | Frozen Baseline Reference |
| **DINOv3 PyTorch FP32** | **Category Filtered** | **71.7% (43/60)** | **50.0% (17/34)** | **100.0% (26/26)** | **100.0% (60/60)** | **0.8142** | Frozen Baseline Reference |
| **DINOv3 ONNX INT8** | Unconstrained | 65.0% (39/60) | 38.2% (13/34) | 100.0% (26/26) | 98.3% (59/60) | 0.7634 | 5.0% R@1 drop, 1 drop from R@20 |
| **DINOv3 ONNX INT8** | **Category Filtered** | **63.3% (38/60)** | **35.3% (12/34)** | **100.0% (26/26)** | **98.3% (59/60)** | **0.7691** | **8.4% R@1 regression (43 $\to$ 38)** |

> [!WARNING]
> **Accuracy Regression Identified**: Dynamic INT8 quantization degrades fine-grained jewellery retrieval. Real photographic Recall@1 drops from **50.0% to 35.3%** (-14.7%), and Candidate Recall@20 drops from **100.0% to 98.3%**. Because lookalike jewellery pieces are separated by narrow margins ($\Delta \cos \approx 0.01 - 0.03$), quantizing Transformer weights to 8 bits without metric fine-tuning introduces rank inversions for 5 out of 60 items.

### 4.4 Sequential Single-Query CPU Latency Breakdown ($B=1$, 50 Iterations)

| Pipeline Stage | PyTorch FP32 (ms) | ONNX FP32 (ms) | ONNX INT8 (ms) | INT8 Speedup vs PyTorch |
|---|:---:|:---:|:---:|:---:|
| 1. Image Read & RGB Conversion | 1.84 ms | 1.82 ms | 1.83 ms | 1.00x |
| 2. Resize & Normalization | 2.68 ms | 2.65 ms | 2.67 ms | 1.00x |
| 3. Neural Forward Pass (p50) | 156.78 ms | 173.92 ms | **139.97 ms** | **1.12x** (16.8 ms reduction) |
| ↳ *Neural Forward Pass (p95)* | *160.12 ms* | *184.23 ms* | *151.45 ms* | *1.06x* |
| 4. L2 Normalization | 0.05 ms | 0.04 ms | 0.04 ms | 1.25x |
| 5. Vector Dot-Product ($N=432$) | 0.28 ms | 0.28 ms | 0.27 ms | 1.04x |
| 6. Top-20 Sorting | 0.05 ms | 0.05 ms | 0.05 ms | 1.00x |
| **Total Sequential Latency (p50)** | **161.92 ms** | **178.87 ms** | **144.79 ms** | **1.12x** |
| **Total Sequential Latency (p95)** | **165.22 ms** | **190.39 ms** | **157.66 ms** | **1.05x** |

### 4.5 Acceptance Criteria Status for INT8
- **Primary Latency Target ($< 100.0$ ms)**: `NOT_PASSED` (Measured Total p50 = **144.79 ms**; Forward p50 = **139.97 ms**). Exceeds target by ~45 ms on CPU.
- **Stretch Latency Target ($< 40.0$ ms)**: `NOT_PASSED`.
- **Candidate Recall Retention ($\ge 99.0\%$)**: `MARGINAL / REGRESSION` (Candidate Recall@20 dropped to **98.3%**).
- **Recommendation**: Dynamic INT8 quantization should **not** replace the FP32 production backbone on CPU alone. Meeting $< 100$ ms with zero accuracy degradation requires:
  1. Distilled lightweight embedding backbones (e.g., MobileNetV4 / ViT-Tiny fine-tuned on jewellery).
  2. Static post-training quantization (PTQ) with representative calibration images.
  3. Hardware-accelerated inference (Apple Silicon Neural Engine or server GPU).

---

## 5. Workstream C — Statistically Powered Hard-Negative Validation ($N=300$)

### 5.1 Dataset Construction & Independence Guarantee
To provide the statistical power required to validate the False Accept Rate target ($\text{FAR} \le 1.0\%$), a dedicated evaluation pool of **300 independent lookalike negative query cases** was constructed:
- **Sample Size**: 300 cases (170 necklaces, 130 rings).
- **Physical Independence**: Every single query in the pool represents a distinct physical SKU group ($300 / 300$ unique `identity_group_id`s). No correlated SKU repetitions exist.
- **Candidate Selection**: For each query SKU, the hardest non-matching lookalike candidate within the same category was selected by embedding similarity.

### 5.2 Confusing Attribute Coverage
The 300 negative cases cover critical fine-grained confusing jewellery attributes:
- **Necklaces ($N=170$)**:
  - `pendant_motif_floral_vs_drop_gem_cluster` (29 cases)
  - `chain_architecture_and_bail_attachment` (68 cases)
  - `gemstone_count_and_tier_proportions` (43 cases)
  - `metal_finish_and_overall_silhouette` (30 cases)
- **Rings ($N=130$)**:
  - `solitaire_prong_vs_bezel_shank_variant` (17 cases)
  - `stone_count_solitaire_vs_three_stone_halo` (53 cases)
  - `shank_geometry_plain_vs_split_twist` (41 cases)
  - `gemstone_shape_and_carat_silhouette` (19 cases)

### 5.3 Frozen Threshold Evaluation & Statistical Bounds

| Acceptance Threshold ($\tau$) | Observed False Accepts | Sample Size ($N$) | False Accept Rate (FAR) | 95% Two-Sided CI | One-Sided 95% Upper Bound | False Reject Rate (FRR) on Positives | Validation Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| $\tau = 0.9000$ | 293 / 300 | 300 | 97.67% | [95.34%, 99.01%] | 99.06% | 0.0% | `REJECTED` (Over-permissive) |
| $\tau = 0.9500$ | 238 / 300 | 300 | 79.33% | [74.34%, 83.74%] | 83.77% | 4.3% | `REJECTED` |
| $\tau = 0.9800$ | 70 / 300 | 300 | 23.33% | [18.66%, 28.59%] | 28.54% | 23.4% | `REJECTED` |
| $\tau_{\text{high}} = 0.9900$ | 12 / 300 | 300 | 4.00% | [2.09%, 6.88%] | 6.88% | 53.2% | `REJECTED` |
| **$\tau_{\text{exact}} = 0.9909$ (Frozen)** | **10 / 300** | **300** | **3.33%** | **[1.61%, 6.03%]** | **6.05%** | **63.8%** | **`NOT_PASSED` (Target FAR $\le 1.0\%$ not met)** |
| $\tau = 0.9950$ (Post-hoc) | **0 / 300** | **300** | **0.00%** | **[0.00%, 1.22%]** | **0.99%** | **85.1%** | `STATISTICALLY_VALIDATED` (Coverage severely degraded) |

### 5.4 Key Statistical Findings
1. **The Frozen Threshold Did Not Meet the Target**: At the predeclared threshold $\tau_{\text{exact}} = 0.9909$, exactly 10 lookalike pairs falsely accepted, yielding an empirical FAR of **$3.33\%$** and an exact one-sided 95% Clopper-Pearson upper bound of **$6.05\%$**.
2. **Analysis of the 10 False Accepts**:
   - Inspection revealed near-identical designs: `necklace_18` vs `necklace_19` (0.9946), `necklace_286` vs `necklace_292` (0.9942), `necklace_62` vs `necklace_64` (0.9942), and `necklace_22` vs `necklace_23` (0.9925). These items share identical chain weaves and matching silhouette bails.
   - Global foundation models without fine-grained metric learning cannot cleanly separate these 0.005-cosine lookalikes.
3. **Threshold vs Coverage Trade-Off**:
   - Increasing the threshold to $\tau = 0.9950$ eliminates all 300 false accepts, achieving a 95% upper bound of **$0.99\% \le 1.0\%$** (confirming the Rule of Three).
   - However, at $\tau = 0.9950$, true positive coverage collapses (FRR increases to 85.1%), meaning only 14.9% of genuine exact matches qualify for automated acceptance.
4. **Official Validation Status**: Designated as `NOT_STATISTICALLY_VALIDATED_AT_FROZEN_THRESHOLD` (with `STATISTICALLY_VALIDATED_AT_TAU_0.9950_WITH_SEVERE_COVERAGE_PENALTY`).

---

## 6. Acceptance Checklist Status

| Acceptance Gate | Production Target | Measured Prototype Metric | Compliance Status | Findings & Remediation |
|---|:---:|:---:|:---:|---|
| **Gate 0: Count Reconciliation** | 100% reconciled accounting | 518 images, 492 raw, 456 physical clusters, 26 synthetic clusters, 488 independent captures | `PASSED` | Fully resolved in `ph002_count_reconciliation.json`. Regression tests passing 5/5. |
| **Workstream A: Real Ring Inventory** | $\ge 50$ physical ring SKUs | 5 verified physical ring SKUs (10 images) | `DOCUMENTED_AS_DATA_GAP` | Forensic audit proved raw catalog contains only 5 real ring multi-shot clusters. Synthetic variations kept strictly separate. |
| **Workstream B: CPU Latency SLA** | Sequential p50 $< 100.0$ ms | ONNX INT8 p50 = **144.79 ms** (PyTorch p50 = **161.92 ms**) | `NOT_PASSED` | INT8 dynamic quantization yields 1.12x speedup on CPU but misses $<100$ ms target. |
| **Workstream B: Accuracy Retention** | No unreported regressions | R@1 drops from 71.7% to 63.3% (-8.4%) | `REGRESSION_DOCUMENTED` | Fine-grained jewellery retrieval requires higher precision than dynamic INT8 weight quantization provides. |
| **Workstream C: False Accept Rate** | $\text{FAR} \le 1.0\%$ at 95% confidence | Frozen $\tau=0.9909$: **10/300 (3.33%)**, upper bound = **6.05%** | `NOT_PASSED` | Evaluated across 300 independent lookalikes. Target only met at $\tau=0.9950$ (0/300, upper bound 0.99%), where coverage collapses. |
| **Regression Test Suite** | 100% test pass rate | **72/72 tests passing** | `PASSED` | Full regression suite passes across all test modules. |

---

## 7. Deliverables Summary

1. [`output/ph002_count_reconciliation.json`](file:///Users/amol/Developer/search/output/ph002_count_reconciliation.json): Machine-readable reconciliation of all image and cluster counts.
2. [`data/models/dinov3_vits16_fp32.onnx`](file:///Users/amol/Developer/search/data/models/dinov3_vits16_fp32.onnx) & [`data/models/dinov3_vits16_int8.onnx`](file:///Users/amol/Developer/search/data/models/dinov3_vits16_int8.onnx): Exported and quantized model artifacts.
3. [`output/ph002_quantization_benchmark.json`](file:///Users/amol/Developer/search/output/ph002_quantization_benchmark.json): Empirical benchmark report for ONNX FP32 and INT8 models.
4. [`output/ph002_hard_negative_evaluation.json`](file:///Users/amol/Developer/search/output/ph002_hard_negative_evaluation.json): 300-case hard-negative evaluation report with Clopper-Pearson confidence bounds.
5. Automated test suites:
   - [`tests/test_ph002_accounting.py`](file:///Users/amol/Developer/search/tests/test_ph002_accounting.py) (5/5 passed)
   - [`tests/test_ph002_quantization.py`](file:///Users/amol/Developer/search/tests/test_ph002_quantization.py) (3/3 passed)
   - [`tests/test_ph002_hard_negatives.py`](file:///Users/amol/Developer/search/tests/test_ph002_hard_negatives.py) (4/4 passed)
6. Full test suite passing: **72/72 tests** (`pytest tests/`).
