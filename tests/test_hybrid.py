"""Unit tests for Hybrid Search and Patch Chamfer Reranking."""

import pytest
import numpy as np
from PIL import Image

from engine.reranker import (
    get_foreground_patch_indices,
    compute_chamfer_similarity,
    PatchCatalogIndex
)
from engine.hybrid_search import HybridSearchEngine

def test_foreground_patch_indices():
    # Create 256x256 image: pure white everywhere except a 64x64 dark square in the center
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    for x in range(96, 160):
        for y in range(96, 160):
            img.putpixel((x, y), (20, 20, 20))

    fg_indices = get_foreground_patch_indices(img, num_patches_side=16, threshold=240)
    # Total patches = 16 * 16 = 256. Center square is 4x4 patches = 16 patches.
    assert len(fg_indices) > 0
    assert len(fg_indices) < 256
    assert all(0 <= idx < 256 for idx in fg_indices)

def test_compute_chamfer_similarity_identical():
    # Identical normalized patch tokens should yield ~1.0
    rng = np.random.default_rng(42)
    patches = rng.standard_normal((64, 384)).astype(np.float32)
    patches /= np.linalg.norm(patches, axis=1, keepdims=True)

    score = compute_chamfer_similarity(patches, patches)
    assert abs(score - 1.0) < 1e-4

def test_compute_chamfer_similarity_orthogonal():
    # Orthogonal patches
    p1 = np.zeros((10, 20), dtype=np.float32)
    p2 = np.zeros((10, 20), dtype=np.float32)
    p1[:, :10] = 1.0 / np.sqrt(10)
    p2[:, 10:] = 1.0 / np.sqrt(10)

    score = compute_chamfer_similarity(p1, p2)
    assert abs(score) < 1e-4

def test_hybrid_search_pipeline():
    engine = HybridSearchEngine(index_dir="data/index", device="cpu")
    dummy_img = Image.new("RGB", (256, 256), (240, 240, 240))
    
    # 1. Hybrid search with rerank
    out = engine.search_hybrid(
        image=dummy_img,
        top_k=5,
        candidate_depth=10,
        enable_rerank=True,
        patch_weight=0.2
    )
    assert out["model_name"] == "hybrid_rerank"
    assert "timing_ms" in out
    assert len(out["results"]) <= 5
    for r in out["results"]:
        assert "chamfer_score" in r
        assert "similarity" in r
        assert "confidence_label" in r

    # 2. Hybrid search without rerank
    out_no_rerank = engine.search_hybrid(
        image=dummy_img,
        top_k=5,
        candidate_depth=10,
        enable_rerank=False
    )
    assert out_no_rerank["model_name"] == "hybrid"
    assert len(out_no_rerank["results"]) <= 5
