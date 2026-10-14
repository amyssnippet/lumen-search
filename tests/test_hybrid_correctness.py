"""Regression and correctness tests for EXP-004.1 hybrid retrieval, patch weights, and calibration."""

import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from engine.reranker import (
    compute_chamfer_similarity,
    get_foreground_patch_indices,
    PatchCatalogIndex
)
from engine.hybrid_search import HybridSearchEngine
from engine.index import CatalogIndex

def test_symmetric_chamfer_masking_and_fallbacks():
    # Setup dummy normalized patch grids
    rng = np.random.default_rng(123)
    q_patches = rng.standard_normal((256, 384)).astype(np.float32)
    q_patches /= np.linalg.norm(q_patches, axis=1, keepdims=True)
    
    c_patches = rng.standard_normal((256, 384)).astype(np.float32)
    c_patches /= np.linalg.norm(c_patches, axis=1, keepdims=True)

    # 1. Self similarity on identical patches
    self_score = compute_chamfer_similarity(q_patches, q_patches)
    assert abs(self_score - 1.0) < 1e-4

    # 2. Both masks present and valid
    q_fg = np.array([10, 11, 12, 13, 25, 26, 27], dtype=np.int32)
    c_fg = np.array([12, 13, 14, 15, 26, 27, 28], dtype=np.int32)
    score_masked = compute_chamfer_similarity(q_patches, c_patches, q_fg, c_fg)
    assert -1.0 <= score_masked <= 1.0

    # 3. Degenerate mask fallback (< 4 patches)
    score_fallback = compute_chamfer_similarity(q_patches, c_patches, query_fg_indices=np.array([1, 2]))
    assert -1.0 <= score_fallback <= 1.0

    # 4. Spatial constraint penalty
    score_unconstrained = compute_chamfer_similarity(q_patches, c_patches, spatial_weight=0.0)
    score_constrained = compute_chamfer_similarity(q_patches, c_patches, spatial_weight=0.2)
    assert score_constrained <= score_unconstrained

def test_fusion_formula_semantics_and_endpoints():
    engine = HybridSearchEngine(index_dir="data/index", device="cpu")
    dummy_img = Image.new("RGB", (256, 256), (220, 220, 220))

    # 1. patch_weight = 0.0 -> final_score MUST equal global_score
    out_pw0 = engine.search_hybrid(dummy_img, top_k=5, enable_rerank=True, patch_weight=0.0)
    assert out_pw0["patch_weight"] == 0.0
    for r in out_pw0["results"]:
        assert abs(r["similarity"] - r["global_score"]) < 1e-4

    # 2. patch_weight = 1.0 -> final_score MUST equal chamfer_score
    out_pw1 = engine.search_hybrid(dummy_img, top_k=5, enable_rerank=True, patch_weight=1.0)
    assert out_pw1["patch_weight"] == 1.0
    for r in out_pw1["results"]:
        assert abs(r["similarity"] - r["chamfer_score"]) < 1e-4

    # 3. Intermediate patch_weight = 0.3
    out_pw03 = engine.search_hybrid(dummy_img, top_k=5, enable_rerank=True, patch_weight=0.3)
    for r in out_pw03["results"]:
        expected = round(0.7 * r["global_score"] + 0.3 * r["chamfer_score"], 4)
        assert abs(r["similarity"] - expected) <= 1e-3

def test_patch_cache_metadata_alignment():
    patch_idx = PatchCatalogIndex(model_name="dinov3", index_dir="data/index")
    patch_idx.load()

    cat_idx = CatalogIndex(model_name="dinov3", index_dir="data/index")
    cat_idx.load()

    # Verify ID alignment between patch metadata and catalog index
    assert len(patch_idx.catalog_ids) == len(cat_idx.metadata)
    for p_id, c_meta in zip(patch_idx.catalog_ids, cat_idx.metadata):
        assert p_id == c_meta["id"]

    # Verify array dimensions
    assert patch_idx.patches.shape == (len(cat_idx.metadata), 256, 384)
    assert len(patch_idx.catalog_fg_masks) == len(cat_idx.metadata)

def test_exact_vs_similar_search_modes():
    engine = HybridSearchEngine(index_dir="data/index", device="cpu")
    dummy_img = Image.new("RGB", (256, 256), (230, 230, 230))

    # In similar mode: returns requested top_k items
    out_sim = engine.search_hybrid(dummy_img, top_k=5, search_mode="similar")
    assert len(out_sim["results"]) == 5

    # In exact mode with high threshold (e.g., 0.999): should abstain / return empty or filtered
    out_exact_high = engine.search_hybrid(
        dummy_img,
        top_k=5,
        search_mode="exact",
        exact_match_threshold=0.999
    )
    assert out_exact_high["search_mode"] == "exact"
    # Plain grey image will not match any piece with 0.999 similarity
    assert len(out_exact_high["results"]) == 0
    assert out_exact_high["exact_match_found"] is False

def test_timing_fields_presence():
    engine = HybridSearchEngine(index_dir="data/index", device="cpu")
    dummy_img = Image.new("RGB", (256, 256), (220, 220, 220))

    out = engine.search_hybrid(dummy_img, top_k=3, enable_rerank=True, patch_weight=0.2)
    timing = out["timing_ms"]
    required_fields = [
        "file_decoding_and_normalization",
        "query_enhancement",
        "dinov2_inference",
        "dinov3_inference",
        "global_candidate_generation",
        "patch_extraction",
        "patch_reranking",
        "total"
    ]
    for rf in required_fields:
        assert rf in timing
        assert isinstance(timing[rf], (int, float))
        assert timing[rf] >= 0.0

def test_confidence_labels_are_unverified():
    engine = HybridSearchEngine(index_dir="data/index", device="cpu")
    dummy_img = Image.new("RGB", (256, 256), (220, 220, 220))

    out = engine.search_hybrid(dummy_img, top_k=5, enable_rerank=True)
    for r in out["results"]:
        # Verify that labels do NOT claim verified physical SKU without ground truth
        assert "Verified Exact SKU" not in r["confidence_label"]
        assert r["confidence_label"] in [
            "High-Confidence Candidate Match",
            "Visually Similar Candidate",
            "Low Confidence Candidate"
        ]
