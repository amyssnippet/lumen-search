# Experiment Report: EXP-003 — DINOv3 Benchmark for Fine Jewellery Visual Retrieval

## Metadata
- **Experiment ID**: EXP-003
- **Date**: 2026-10-11
- **Author/Agent**: Antigravity Assistant
- **Status**: `COMPLETED`
- **Result Artifact**: `output/evaluation_report.json`

## 1. Objective & Hypothesis
- **Goal**: Benchmark Meta AI's newly released DINOv3 foundation model (`vit_small_patch16_dinov3.lvd1689m`) against DINOv2 (`vit_small_patch14_dinov2`) and OpenCLIP (`ViT-B-32`) on the identical 60-query fine jewellery retrieval benchmark.
- **Hypothesis**: DINOv3's larger pretraining corpus (LVD-1689M, 1.689B curated images) and Rotary Position Embeddings (RoPE) will improve overall candidate recall (Recall@5 / Recall@10) across diverse styling variations. However, its slightly larger patch size (16x16 vs 14x14) on low-resolution catalog thumbnails (262x262) may yield slightly coarser spatial patch tokens for micro-facets than DINOv2.

## 2. Model & Checkpoint Architecture
- **Official Repository**: `https://github.com/facebookresearch/dinov3` (Paper: *DINOv3*, Siméoni et al., Meta AI Research / FAIR, 2025).
- **Model Checkpoint**: `vit_small_patch16_dinov3.lvd1689m`
  - Parameters: 21M (matched to DINOv2 ViT-Small for controlled parameter parity).
  - Pretraining dataset: LVD-1689M (curated web vision dataset).
  - Architecture: 12 EvaBlocks with Rotary Position Embeddings (`RotaryEmbeddingDinoV3`), 384 embedding dimensions.
  - License: `dinov3-license` (Meta FAIR research license).
  - Input resolution: 256x256 native.
- **Hardware & Runtime**: macOS 15.7 Intel x86_64, CPU inference, Python 3.11.17, PyTorch 2.2.2.

## 3. Quantitative Results & 3-Way Benchmark Comparison
Evaluated on the identical held-out test split: 60 queries (30 identity twin pairs and 30 semantic singletons) against a 432-item catalog corpus:

| Metric | DINOv2 (ViT-S/14) | DINOv3 (ViT-S/16) | OpenCLIP (ViT-B/32) | DINOv3 vs DINOv2 Delta | DINOv3 vs CLIP Delta |
|---|---|---|---|---|---|
| **Recall@1 (Exact Twin)** | **70.00%** | 66.67% | 43.33% | -3.33% (1 query) | **+23.34%** |
| **Recall@5** | 83.33% | **90.00%** | 80.00% | **+6.67%** | **+10.00%** |
| **Recall@10** | 90.00% | **93.33%** | 86.67% | **+3.33%** | **+6.66%** |
| **Mean Reciprocal Rank (MRR)** | 0.7478 | **0.7650** | 0.5875 | **+0.0172** | **+0.1775** |
| **Category Precision@1** | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| **Category Precision@5** | **100.00%** | 99.67% | 98.67% | -0.33% | **+1.00%** |
| **False Matches ($\ge 0.94$)** | **53** | 424 | 363 | +371* | +61 |
| **Single-Query Latency (CPU)** | **137.56 ms** | 154.20 ms | 125.08 ms | +16.64 ms | +29.12 ms |
| **Indexing Throughput** | 10.3 img/s | 8.3 img/s | **15.4 img/s** | -2.0 img/s | -7.1 img/s |

*\*Note on False Matches*: Raw cosine similarities are distributed higher for DINOv3 global embeddings due to its pretraining distribution. The fixed $0.94$ threshold calibrated on DINOv2 requires recalibration (to $\approx 0.975$) for DINOv3 to achieve equivalent precision.

## 4. Qualitative Analysis & Findings
1. **Broader Recall Horizon**: DINOv3 significantly improves top-5 and top-10 retrieval: **Recall@5 increased from 83.33% to 90.00%**, and **Recall@10 reached 93.33%**, driving overall **MRR up to 0.7650** (best overall).
2. **Patch Granularity on Small Thumbnails**: For exact rank-1 matching, DINOv2 maintained a slight advantage (70.0% vs 66.67%). This is attributable to patch token resolution: DINOv2 uses $14 \times 14$ patches, yielding $256$ spatial tokens on $224 \times 224$ images, whereas DINOv3's $16 \times 16$ patch size covers slightly coarser surface area per token on small $262 \times 262$ catalog thumbnails.
3. **CPU Execution Feasibility**: DINOv3 ViT-Small executes on Intel CPU at **154.2ms per query**, well within practical limits for interactive search.
4. **Patch Tokens for Second-Stage Reranking**: DINOv3 exposes `model.forward_features()`, generating a $(256 \times 384)$ spatial token grid excluding register tokens, making it a prime candidate for local gem-alignment reranking.

## 5. Conclusion & Recommendation
- **Recommendation**:
  - Keep **DINOv2** as default for single-stage global exact-match retrieval when top-1 accuracy is paramount.
  - Enable **DINOv3** as the primary option for broad candidate recall (Recall@5 = 90.0%) and hybrid multi-stage retrieval.
  - Support both models interchangeably via the unified UI and API.
