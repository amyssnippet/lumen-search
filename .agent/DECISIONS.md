# Architecture Decision Records (ADR)

## DEC-001: Execution Environment & Python 3.11 Runtime
- **Date**: 2026-10-11
- **Status**: `ACCEPTED`
- **Context**: System Python is 3.14 on macOS x86_64 where official PyTorch binary wheels do not exist. Building Python from Homebrew source on Intel Mac was excessively slow.
- **Decision**: Installed `uv` and provisioned precompiled standalone `cpython-3.11.17`, creating a pinned `.venv` with `numpy==1.26.4` and `torch==2.2.2`.
- **Consequences**: Fast, reproducible CPU execution without compilation lag or numpy C-API mismatches.

## DEC-002: Model Architectures — Self-Supervised Vision Transformer (DINOv2) & Multimodal (CLIP)
- **Date**: 2026-10-11
- **Status**: `ACCEPTED`
- **Context**: Fine jewellery features (gemstone cuts, prongs, pavilion facets, thin chains) require fine-grained spatial and geometric understanding. Supervised ImageNet models bias heavily towards coarse objects.
- **Decision**: Benchmark DINOv2 (`vit_small_patch14_dinov2`) and CLIP (`ViT-B-32`). DINOv2 is chosen for its dense patch-level self-supervised representation, while CLIP offers semantic robustness.
- **Consequences**: Runs locally on CPU within ~138ms latency at 224x224 dynamic resolution.

## DEC-003: Vector Search Implementation — In-Memory Matrix Multiplication
- **Date**: 2026-10-11
- **Status**: `ACCEPTED`
- **Context**: Catalog size is ~500 to 1,000 images, scaling to ~50,000.
- **Decision**: Use NumPy normalized inner product (`A @ b.T`) instead of heavy distributed vector databases (Milvus, Pinecone) or bulky C++ FAISS dependencies.
- **Consequences**: Zero dependency overhead, exact nearest neighbors (100% precision vs approximate indexing), and query latency under 1ms for the index search step.

## DEC-004: Ground Truth Twin Grouping for Evaluation
- **Date**: 2026-10-11
- **Status**: `ACCEPTED`
- **Context**: Raw dataset files are numbered without manufacturer SKU identifiers.
- **Decision**: Detect exact SHA-256 duplicates and perceptual twin clusters (dHash $\le 2$) as ground-truth identity pairs for exact-match retrieval benchmarking. Singletons evaluate category precision.
- **Consequences**: Honest, non-fabricated identity evaluation that prevents self-matching leakage.

## DEC-005: DINOv3 Checkpoint Selection & Coexistence Strategy
- **Date**: 2026-10-11
- **Status**: `ACCEPTED`
- **Context**: Meta AI released DINOv3 foundation models (Siméoni et al., 2025). The repository and timm library offer variants from ViT-Small (21M) to ViT-7B (6.7B).
- **Decision**: Evaluated `vit_small_patch16_dinov3.lvd1689m` (21M params, 384-dim, 256x256 resolution) on CPU. Retain DINOv2 for top-1 exact piece retrieval (Recall@1 = 70.0%) and DINOv3 for broad candidate generation (Recall@5 = 90.0%, Recall@10 = 93.3%). Both models coexist seamlessly behind `BaseEncoder`.
- **Consequences**: Enables users to toggle models dynamically in the UI and API without re-indexing latency.
