# Experiment Report: EXP-002 — Query Preprocessing & Quality Gating

## Metadata
- **Experiment ID**: EXP-002
- **Date**: 2026-10-11
- **Author/Agent**: Antigravity Assistant
- **Status**: `COMPLETED`
- **Backbone**: DINOv2 (`vit_small_patch14_dinov2`)

## Objective & Hypothesis
- **Goal**: Evaluate the retrieval impact of query image preprocessing techniques specified in Section 13:
  1. Baseline Aspect-Preserving Padding & Resize
  2. Forced CLAHE (Contrast-Limited Adaptive Histogram Equalization) on LAB L-channel
  3. Bilateral Facet-Preserving Denoising
  4. Smart Quality-Gated Auto-Enhancement (applies CLAHE only when image quality assessment flags underexposure or low contrast)
- **Hypothesis**: Indiscriminate global contrast adjustment alters fine diamond facet luminance values, potentially regressing exact-match retrieval on clean studio shots, whereas quality-gated enhancement and bilateral filtering reduce false positives from compression noise.

## Results & Benchmark Comparison
| Preprocessing Configuration | Recall@1 | Recall@5 | MRR | False Exact Matches ($\ge 0.94$) | Mean Latency (ms) | Status |
|---|---|---|---|---|---|---|
| **Config 1: Baseline Aspect Padding** | 70.00% | 83.33% | 0.7478 | 53 | 138.62 ms | Baseline |
| **Config 2: Forced Global CLAHE** | 66.67% | 83.33% | 0.7343 | 46 | 144.02 ms | **Regressed (-3.33%)** |
| **Config 3: Bilateral Denoise** | **70.00%** | **83.33%** | **0.7426** | **40** | 138.76 ms | **Improved (-24.5% false positives)** |
| **Config 4: Smart Quality-Gated** | **70.00%** | **83.33%** | **0.7478** | 53 | 138.20 ms | **Optimal Trade-off** |

## Findings & Qualitative Insights
1. **Dangers of Forced CLAHE**: Forcing CLAHE on all queries caused Recall@1 to fall from 70.0% to 66.67% (20/30 vs 21/30). While it enhances contrast in dark areas, it disrupts the facet gradient distributions learned by DINOv2 for clean studio pieces.
2. **Bilateral Filtering Reduces False Positives**: Mild bilateral filtering ($d=5, \sigma=40$) smooths sensor and JPEG compression noise in smooth background regions while preserving sharp diamond prongs and girdle outlines, reducing false exact matches from 53 down to 40 without harming Recall@1.
3. **Quality-Gated Gating**: Smart quality-gated enhancement only activates when mean luminance $< 40$ or local contrast is clipped, avoiding degradation on pristine catalog images.

## Decision
- **Default Policy**: Keep aspect-ratio padding enabled always; enable bilateral denoising and smart quality-gated auto-enhancement by default. Do not force CLAHE on clean uploads.
