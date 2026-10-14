# Experiment Report: EXP-001 — Baseline Visual Encoder Comparison (DINOv2 vs OpenCLIP)

## Metadata
- **Experiment ID**: EXP-001
- **Date**: 2026-10-11
- **Author/Agent**: Antigravity Assistant
- **Status**: `COMPLETED`
- **Result Artifact**: `output/evaluation_report.json`

## Objective & Hypothesis
- **Goal**: Evaluate pretrained visual representation models on fine-grained jewellery retrieval.
- **Hypothesis**: Self-supervised Vision Transformers (DINOv2) attend to dense geometric patch tokens (prongs, facet cuts, pavé bands, chain textures) without semantic label compression, outperforming multimodal CLIP models on exact SKU/twin retrieval.

## Configuration
- **Model A**: `vit_small_patch14_dinov2` (21M params, 384-dim, dynamic 224x224 input).
- **Model B**: `open_clip/ViT-B-32` (87M params, 512-dim, 224x224 input).
- **Dataset Manifest**: `data/manifest.json` (432 catalog items, 60 held-out test queries, 30 verified twin identity queries).
- **Hardware**: macOS 15.7 Intel x86_64, CPU inference.
- **Indexing Throughput**: DINOv2: 10.3 img/s; CLIP: 15.4 img/s.

## Quantitative Results
| Metric | DINOv2 (ViT-Small) | OpenCLIP (ViT-B-32) | Delta (DINOv2 vs CLIP) |
|---|---|---|---|
| **Recall@1** | **70.00%** | 43.33% | **+26.67%** |
| **Recall@5** | **83.33%** | 80.00% | **+3.33%** |
| **Recall@10** | **90.00%** | 86.67% | **+3.33%** |
| **MRR** | **0.7478** | 0.5875 | **+0.1603** |
| **Category Precision@1** | 100.00% | 100.00% | 0.00% |
| **Category Precision@5** | **100.00%** | 98.67% | **+1.33%** |
| **False Exact Matches** | **53** | 363 | **-310 (6.8x fewer)** |
| **Mean Latency (ms)** | 138.62 ms | 122.51 ms | +16.11 ms |

## Findings & Qualitative Diagnosis
1. **DINOv2 Superiority**: DINOv2 achieved 70.0% Recall@1 compared to CLIP's 43.33%. DINOv2 successfully isolates subtle differences in centre-stone cuts (e.g. distinguishing oval from round brilliant solitaires) and halo geometry.
2. **CLIP Clustering Failure**: CLIP models exhibited 363 false exact matches ($\ge 0.94$ similarity across different clusters) because natural-language pretraining collapses fine visual variations of "diamond ring" into the same semantic cluster.
3. **Latency Trade-off**: DINOv2 single-query CPU latency is 138.6ms vs CLIP's 122.5ms. The modest 16ms delta is well worth the +26.7% Recall@1 boost.

## Conclusion & Decision
- **Winner**: DINOv2 (`vit_small_patch14_dinov2`) is selected as the default production backbone.
- **Next Action**: Execute EXP-002 testing query enhancement (CLAHE, bilateral denoising, aspect padding).
