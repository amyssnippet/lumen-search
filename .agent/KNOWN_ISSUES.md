# Known Issues and Limitations

## ISSUE-001: CPU Execution Environment on Intel x86_64
- **Date**: 2026-10-11
- **Severity**: `MINOR`
- **Status**: `RESOLVED` (Workaround applied)
- **Description**: The host machine is an Intel Mac (`x86_64`) without Apple Silicon MPS acceleration. High parameter vision models (e.g. ViT-Large, DINOv3-7B) would introduce unacceptable query latencies (>5 seconds).
- **Resolution**: Selected ViT-Small architectures (`vit_small_patch14_dinov2` and `vit_small_patch16_dinov3.lvd1689m`, both 21M parameters), keeping single-image CPU forward latency to 135–155ms.

## ISSUE-002: NumPy 2.x Incompatibility with PyTorch 2.2.2 Wheels
- **Date**: 2026-10-11
- **Severity**: `MAJOR`
- **Status**: `RESOLVED`
- **Description**: Default pip installation pulled `numpy==2.4.6`, causing `_ARRAY_API not found` warnings and C-API conversion crashes in PyTorch 2.2.2.
- **Resolution**: Explicitly pinned and installed `numpy==1.26.4`. Verified seamless tensor-to-numpy conversions.

## ISSUE-003: Missing Ground Truth Product SKUs in Raw Catalog
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `OPEN` (Methodologically managed)
- **Description**: Images are named sequentially (`necklace_101.jpg`, `ring_042.jpg`) without manufacturer SKU metadata.
- **Resolution**: Extracted verified twin clusters using exact hashes and perceptual hashing for ground truth identity tests. Clearly demarcate exact matches from visually similar items in the UI and API.

## ISSUE-004: Model-Specific Cosine Similarity Shift (DINOv3 vs DINOv2)
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `RESOLVED` (Calibration report generated in EXP-004.1)
- **Description**: DINOv3 global embeddings produce higher baseline cosine similarities across the catalog than DINOv2 (mean negative similarity 0.825 vs 0.445). Applying a rigid 0.94 threshold calibrated on DINOv2 yields excessive false exact matches for DINOv3.
- **Resolution**: Implemented configuration-specific rejection thresholds ($\tau_{99\%}=0.970$ for DINOv3, $\tau_{99\%}=0.905$ for DINOv2, $\tau_{99\%}=0.935$ for Hybrid Global and Hybrid Rerank) derived empirically via `scripts/calibrate_thresholds.py`.

## ISSUE-005: Transitive Closure Leak in Perceptual Twin Clustering (`ring_083`)
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `RESOLVED` (Benchmark split into v1 and v2)
- **Description**: In EXP-001/002, graph component clustering with threshold $\text{dHash} \le 2$ transitively chained `ring_083` to `ring_187` via intermediate node `ring_116` (effective dHash distance = 4 bits). Inspection confirmed `ring_083` is a single round-cut solitaire while `ring_187` is a multi-stone pave ring, making it an invalid ground-truth target.
- **Resolution**: Preserved Historical Benchmark v1 ($N=30$) for backward reproducibility and introduced Corrected Benchmark v2 ($N=29$) excluding `ring_083`.

## ISSUE-006: Evaluation Benchmark Category Skew (96.6% Necklaces in v2)
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `RESOLVED` (Benchmark v3 constructed in EXP-005)
- **Description**: The previous 29-query ground truth twin benchmark contained 28 necklaces (96.6%) and only 1 ring (3.4%), preventing statistically meaningful validation of ring retrieval nuances.
- **Resolution**: Constructed Balanced Benchmark v3 (`data/evaluation/benchmark_v3_balanced.json`) with exactly 30 necklaces (50.0%) and 30 rings (50.0%), plus 30 confirmed hard negatives and 30 singletons.

## ISSUE-007: Necklace Chain & Bail Embedding Dominance
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `RESOLVED` (Diagnosed and benchmarked in EXP-006)
- **Description**: While synthetic rings achieve 100.0% Recall@1, real photographic necklaces achieve 46.67% to 53.33% Recall@1. Many necklaces share similar link chains and oval bails, where global embeddings attend to the overall loop.
- **Resolution**: Investigated in EXP-006 via `JewelleryLocalizer` and region cropping. Discovered that hard-cropping down to only the pendant discards chain geometry and actually decreases overall Recall@1 (to 33.3%–40.0% on real necklaces) due to high false-positive clustering among generic solitaires. Spatially constrained patch Chamfer matching (EXP-004/005) remains the superior approach because it attends to gemstone facets while preserving global spatial context.

## ISSUE-008: Loss of Discriminative Global Entropy from Isolated Component Hard Cropping
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `RESOLVED` (Empirically proven with Oracle Upper Bound in EXP-006)
- **Description**: In EXP-006, isolated region retrieval dropped Recall@1 from 71.7% to 63.3% (DINOv3) and 70.0% to 60.0% (DINOv2). Even with 100% human-verified ground-truth bounding boxes (Oracle Upper Bound), Recall@1 was only 68.3%. Discarding the complete piece geometry removes vital discriminative entropy, as many distinct jewellery pieces share near-identical standard four-prong round solitaires or cluster drops.
- **Resolution**: Retained full-image global representation and calibrated score fusion as the default top-1 candidate generator, utilizing component localization for interactive user-guided search ("search by loose stone") and UI visual explainability overlay.

## ISSUE-009: Latency Discrepancy Between Batched Throughput and Single-Query Sequential Invocations
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `RESOLVED` (Diagnosed and benchmarked in EXP-007)
- **Description**: EXP-006 reported ~22–24 ms from `encode_batch(batch_size=32)`, reflecting batched CPU throughput amortized per sample. Real-world REST API invocations arrive sequentially ($B=1$), taking ~181–188 ms on CPU.
- **Resolution**: Explicitly profiled with `time.perf_counter()`, isolating preprocessing (5.2 ms), localizer (2.0 ms), encoder forward (174–179 ms), and vector scoring (0.14 ms). Both single-query sequential latency ($B=1$) and amortized batch throughput ($B=32$) are now documented and reported with precision.

## ISSUE-010: Context Preservation in Attention Weighting
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `RESOLVED` (Solved via soft spatial token reweighting in EXP-007)
- **Description**: While hard cropping collapsed Candidate Recall@20 from 100% to 78–81%, naive attention weighting with zero background weights risks similarly discarding context.
- **Resolution**: Implemented non-zero background floors ($0.05$ for foreground saliency, $0.25$ for soft component region distance decay) in `engine/soft_attention.py`. All soft attention models successfully preserved 98.3%–100% Candidate Recall@20 while elevating top-5 recall to 96.7%.

## ISSUE-011: Real Photographic Ring Data Scarcity in Raw Catalog
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `OPEN` (Formally documented in EXP-008)
- **Description**: An exhaustive forensic audit across all 191 ring images in `Jewellery_Data/ring/` revealed only 5 natural multi-shot identity clusters (4 in benchmark test split, 1 in catalog). The remaining 26 ring queries in Benchmark v3 are synthetic perturbations (homography tilts, Kelvin shifts, sensor noise). While these validate geometric/photometric robustness (100% R@1), they do not measure multi-angle real photograph generalization.
- **Resolution**: Defined a formal 4-Gate Production SKU Ingestion Standard (ERP lineage, CAD specs, 360° turntable capture protocol, and non-transitive pairwise verification) to govern incoming real photographic ring datasets.

## ISSUE-012: Real Photographic Top-1 Retrieval Gap vs Production Target
- **Date**: 2026-10-11
- **Severity**: `MAJOR`
- **Status**: `OPEN` (Production Hardening Backlog)
- **Description**: While Candidate Recall@20 is 100.0% and Candidate Recall@5 is 93.3%–96.7%, real photographic Top-1 exact retrieval stands at 47.1% (unconstrained) to 50.0% (category filtered), falling short of the enterprise production target ($\ge 95.0\%$).
- **Resolution**: Candidate generation is production-grade, but fine-grained top-1 ranking between visually similar designs requires external multi-view ingestion and metric learning fine-tuning.

## ISSUE-013: Statistical Power Limitation on False Accept Rate Validation
- **Date**: 2026-10-11
- **Severity**: `MODERATE`
- **Status**: `OPEN` (Methodologically audited in PH-001)
- **Description**: In EXP-005 and EXP-008, observing 0 false accepts across 30 negative lookalikes was cited as achieving $\text{FAR} \le 1.0\%$. Under exact Clopper-Pearson binomial statistics, $0/30$ false accepts yields a 95% upper confidence bound of $9.50\%$, failing to prove the production target ($\le 1.0\%$).
- **Resolution**: Marked FAR target status as `NOT_STATISTICALLY_VALIDATED` in the evaluation report. Formulated sample size requirement ($N \ge 300$ verified lookalike negatives without a false accept) required for statistical sign-off in PH-002.




