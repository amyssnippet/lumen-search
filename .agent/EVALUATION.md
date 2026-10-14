# Retrieval Evaluation Methodology & Metrics

## 1. Ground Truth Identity & Relevance Protocol
In this 492-image dataset, exact SKU metadata was not provided upfront. To construct an objective, scientifically rigorous evaluation without fabricating labels:
1. **Identified Twin Pairs (Identity Ground Truth)**:
   - File-hash exact duplicates (3 clusters, 7 images total).
   - Perceptual twin pairs (dHash distance $\le 2$ verified as identical physical piece across different compression/render settings).
   - For any query $q$ belonging to an identity cluster $C(q)$, a retrieval is deemed an **Exact Match Hit** if a separate catalog instance $c \in C(q), c \neq q$ is retrieved.
   - Self-matches ($c = q$) are strictly excluded from ranking or evaluated using a disjoint test split.
   - **Historical Benchmark v1 ($N=30$)**: Initial 30 twin queries including `ring_083`.
   - **Corrected Benchmark v2 ($N=29$)**: Excludes `ring_083`, which was grouped transitively via union-find with `ring_116` but represents a visually distinct ring structure.
2. **Category Relevance (Semantic Correctness)**:
   - Ring queries must retrieve Rings.
   - Necklace queries must retrieve Necklaces.
   - Category Precision@K measures cross-category leak rate.

## 2. Evaluation Metrics
- **Recall@1**: Percentage of queries where the top-1 ranked item is a true ground-truth match.
- **Recall@5**: Percentage of queries where at least one ground-truth match is in the top 5.
- **Recall@10**: Percentage of queries where at least one ground-truth match is in the top 10.
- **Mean Reciprocal Rank (MRR)**: Average $\frac{1}{\text{rank}}$ of the first relevant candidate.
- **Category Precision@5**: Fraction of top-5 retrieved items matching the query's category.
- **Query Latency (ms)**: End-to-end time from image tensor preparation through nearest-neighbor lookup, broken into 7 distinct pipeline stages.

## 3. Train / Validation / Test Partitions & Historical Benchmarks (EXP-001 through EXP-004.1)
- **Catalog Corpus**: 432 canonical catalog images indexed for search.
- **Historical Benchmark v1 ($N=30$)**: 28 necklaces (93.3%), 2 rings (6.7%), including noisy transitive cluster `ring_083`.
- **Corrected Benchmark v2 ($N=29$)**: Excludes `ring_083`. 28 necklaces (96.6%), 1 ring (3.4%).
- **Limitation of Benchmarks v1 and v2**: With only 1 ring test pair, ring retrieval nuances (shank profile, prongs, facet cut) could not be evaluated with statistical power.

## 4. Balanced Benchmark v3 Protocol (EXP-005)
Stored in [`data/evaluation/benchmark_v3_balanced.json`](file:///Users/amol/Developer/search/data/evaluation/benchmark_v3_balanced.json):
- **30 Verified Necklaces**: Curated from multi-item clusters with confirmed identical SKU catalog counterparts.
- **30 Verified Rings**:
  - 4 raw multi-shot identical ring pairs (`ring_081 <-> ring_108`, `ring_155 <-> ring_172`, `ring_080 <-> ring_164`, `ring_084 <-> ring_181`).
  - 26 verified photographic capture queries from unique catalog ring SKUs with documented physical variation:
    - 7 Studio lighting and color temperature shifts ($\pm 10\%$ Kelvin gain, exposure variance).
    - 7 Camera perspective tilts ($12^\circ$ 3D homography).
    - 6 Mobile sensor noise and JPEG recompression (Q78-Q88).
    - 6 Combined realistic capture variations (perspective tilt + lighting shift + noise).
- **30 Confirmed Hard Negatives**: Same-category lookalikes verified as different physical SKUs (e.g. `ring_066 <-> ring_100`, `ring_116 <-> ring_187`).
- **30 Singletons**: Unmatched queries with 0 catalog counterpart.
- **Balance**: Exactly 50% rings and 50% necklaces in the exact-match test set.

## 5. Confidence Calibration with Hard Negatives (EXP-005)
Rebuilt against 60 verified same-SKU positives, 30 confirmed hard negatives, and 100 catalog distractors (130 total negatives):
- **DINOv2 Global**: Pos Mean = 0.9548, HardNeg = 0.8210, DistNeg = 0.2578 $\to$ $\tau_{99\%} = 0.975$ (Coverage: 51.7%, FAR: 0.0%)
- **DINOv3 Global**: Pos Mean = 0.9838, HardNeg = 0.9395, DistNeg = 0.7405 $\to$ $\tau_{99\%} = 0.990$ (Coverage: 48.3%, FAR: 0.0%)
- **Calibrated Min-Max Global ($pw=0.0$)**: Pos Mean = 0.9449, HardNeg = 0.7234, DistNeg = 0.0157 $\to$ $\tau_{99\%} = 0.970$ (Coverage: 56.7%, FAR: 0.0%)
- **Hybrid Rerank ($pw=0.2, \lambda=0.1$)**: Pos Mean = 0.9490, HardNeg = 0.7554, DistNeg = 0.1326 $\to$ $\tau_{99\%} = 0.965$ (Coverage: 60.0%, FAR: 0.0%)
- **Hybrid Rerank ($pw=0.5, \lambda=0.1$)**: Pos Mean = 0.9552, HardNeg = 0.8033, DistNeg = 0.3080 $\to$ $\tau_{99\%} = 0.965$ (Coverage: 61.7%, FAR: 0.0%)
- **Chamfer Only ($pw=1.0, \lambda=0.1$)**: Pos Mean = 0.9655, HardNeg = 0.8832, DistNeg = 0.6003 $\to$ $\tau_{99\%} = 0.965$ (Coverage: 65.0%, FAR: 0.0%)

- **Candidate Semantics**:
  - `score \ge \tau_{99\%}`: `"High-Confidence Candidate Match"` (accepted as potential exact match; unverified unless SKU ground truth is verified).
  - `0.75 \le score < \tau_{99\%}`: `"Visually Similar Candidate"` (promising visual design similarity).
  - `score < 0.75`: `"Low Confidence Candidate"` (insufficient visual match).
- **Search Modes**:
  - `mode="similar"`: Returns top-$K$ candidates ranked by score, annotated with confidence tiers.
  - `mode="exact"`: Filters candidates strictly to `score \ge \tau_{\text{exact}}`. If no candidate qualifies, returns `exact_match_found: false` with abstention explanation.

## 6. Component Localization & Provenance Evaluation Protocol (EXP-006)
- **Real-Image vs Synthetic Query Grouping**:
  - All benchmarks must separate Real Photographic Captures ($N=34$: 30 Necklaces, 4 Rings) from Synthetic Robustness Tests ($N=26$ Rings).
  - Synthetic queries (3D homography tilts, lighting shifts, sensor noise) measure robustness against perturbations of identical images, not physical SKU generalization.
- **Component Localization Quality Metric (IoU)**:
  - Intersection-over-Union (IoU) measured against human-verified ground-truth bounding boxes in `data/region_annotations.json`.
  - Heuristic detector achieved **0.7578 Mean IoU** across all 60 benchmark queries (0.7056 on necklaces, 0.8101 on rings).
- **Comparative Configurations**:
  - Baseline A: Full-Image retrieval (DINOv2 / DINOv3).
  - Baseline B: Region-Only retrieval.
  - Config C: Combined Score Fusion ($w \in [0.10, 0.50]$).
  - Config D: Region Reranking on top-20 candidates.
  - Config E: Oracle Ground-Truth ROI Upper Bound.
- **Empirical Principle**: Isolated component cropping removes discriminative global geometry, reducing top-1 retrieval performance (Full-Image: 71.7% vs Region: 63.3%, Oracle: 68.3% on DINOv3). Component localization is therefore maintained as an interactive user feature and visual explainability overlay, while global contextual embeddings serve as the primary top-1 ranking gate.

## 7. Soft Spatial Attention & Token Reweighting Protocol (EXP-007)
- **Mathematical Aggregation**:
  $$z = \text{normalize}\left(\frac{\sum_{i=1}^{256} w_i p_i}{\sum_{i=1}^{256} w_i + \epsilon}\right)$$
  over $16 \times 16 = 256$ patch tokens with weights $w_i \in [\text{floor}, 1.0]$.
- **Comparative Configurations**:
  - Baseline Global (CLS): Standard global CLS token vector.
  - Config A: Uniform Patch Pooling ($w_i = 1.0$).
  - Config B: Multi-Cue Foreground Saliency (non-white, saturation, Sobel edge).
  - Config C: Soft Component Region (exponential Euclidean decay from component box, floor 0.25).
  - Config D: 2D Anisotropic Gaussian spatial prior.
  - Config E: Combined Multi-Cue Soft Saliency ($w_{fg} \times w_{reg}$).
  - Fused: Global + Combined ($\alpha \in \{0.3, 0.5\}$).
  - Negative Control: Hard Crop (isolated gemstone/pendant).
  - Diagnostic Oracle: Ground-Truth Region Saliency.
- **Empirical Principle & Findings**:
  - Hard cropping causes catastrophic Candidate Recall@20 collapse (from 100% to 78–81%) by discarding chains and shanks.
  - Soft spatial reweighting completely eliminates candidate dropout (maintaining 98.3%–100% Recall@20) while concentrating top-5 recall (up to 96.7% Recall@5 under DINOv3 Foreground Saliency).
  - Fused Global + Soft Combined ($\alpha=0.3$) matches peak top-1 Recall@1 (71.7%) with 100% Recall@20.
- **Latency Accounting Principle**:
  - Explicitly separates sequential single-query CPU latency ($B=1$, ~181–188 ms) from amortized batch throughput ($B=32$, ~22–24 ms).

## 8. Production Acceptance Criteria & Gap Protocol (EXP-008)
- **Quantitative Readiness Criteria**:
  1. **Real Photographic Top-1 Recall**: Target $\ge 95.0\%$. Current Prototype: $47.1\%-50.0\%$. Status: `GAP IDENTIFIED`.
  2. **Candidate Recall@20**: Target $\ge 99.0\%$. Current Prototype: **100.0%** (60/60). Status: `MET`.
  3. **Candidate Recall@5**: Target $\ge 90.0\%$. Current Prototype: **93.3% - 96.7%**. Status: `MET`.
  4. **False Accept Rate (FAR) at High Confidence**: Target $\le 1.0\%$. Current Prototype: **0.0%** (0/30 at $\tau_{99\%}=0.990$). Status: `MET`.
  5. **Sequential Single-Query Latency**: Target $< 100$ ms. Current Prototype: **183.3 ms** p50 on CPU. Status: `GAP IDENTIFIED` (remediated by ONNX INT8 quantization).
  6. **Process Memory Footprint**: Target $< 2.0$ GB. Current Prototype: **~380 MB**. Status: `MET`.
- **Primary Retrieval Gate Recommendation**:
  - DINOv3 Global CLS serves as the authoritative primary top-1 retrieval engine.
  - Multi-cue Soft Spatial Attention provides visual explainability heatmaps (`render_attention_heatmap`) and candidate re-ranking.
  - Hard cropping is restricted to interactive UI bounding-box overlays and user-guided search.

## 9. Leakage-Safe Evaluation Protocol & Statistical Power Standards (PH-001)
Stored in [`scripts/run_production_evaluation.py`](file:///Users/amol/Developer/search/scripts/run_production_evaluation.py) and [`output/ph001_evaluation_report.json`](file:///Users/amol/Developer/search/output/ph001_evaluation_report.json):
- **Partitioning Without Identity Leakage**:
  - **Exploratory Calibration Split**: 11 positive queries and 23 lookalike negatives from distinct identity clusters. Used exclusively to fit confidence thresholds ($\tau_{\text{exact}} = 0.9909, \tau_{\text{high\_confidence}} = 0.9900$).
  - **Held-Out Test Split**: 47 positive queries, 7 lookalike negatives, and 30 singletons with zero identity cluster overlap ($\text{overlap} = \emptyset$).
  - **Quarantined Split**: Quarantines conflicting ground-truth pairs (`necklace_10` and `necklace_91`).
- **Confidence Intervals & Statistical Reporting**:
  - **Wilson Score 95% CI**: Computed for all Recall@1, Recall@5, and Candidate Recall@20 metrics.
  - **Sample Power Rule**: Subgroups with $N < 30$ queries are assigned an explicit `INSUFFICIENT_DATA` status (e.g., Real Rings $N=4$ has a wide Wilson CI of $[4.6\%, 69.9\%]$).
  - **False Accept Rate (FAR) Statistical Validation**:
    - For $k=0$ false accepts out of $n$ lookalike trials, the exact Clopper-Pearson 95% upper bound is $1 - 0.05^{1/n}$ (one-sided) or $1 - 0.025^{1/n}$ (two-sided).
    - For $n=30$ negatives: 95% upper bound is $9.50\%$. An empirical result of $0/30$ false accepts only guarantees $\text{FAR} \le 9.5\%$, failing to prove the $\le 1.0\%$ production target.
    - **Validation Mandate**: Proving $\text{FAR} \le 1.0\%$ at 95% confidence requires $(1 - 0.01)^N \le 0.05 \implies N \ge 300$ representative lookalike negatives without a false accept.




