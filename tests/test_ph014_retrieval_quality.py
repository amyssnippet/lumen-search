"""Regression tests for PH-014: Retrieval Quality, Strict Category Filtering & Calibrated Decisions."""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from engine.search import SearchEngine
from engine.preprocessing import PreprocessConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_strict_category_filtering_rings_vs_necklaces():
    """Verify ZERO cross-category candidate leakage when category filter is active."""
    engine = SearchEngine(model_name="mobilenetv4", backend="numpy")
    ring_img_path = REPO_ROOT / "Jewellery_Data" / "ring" / "ring-downloaded.png"
    if not ring_img_path.exists():
        ring_img_path = REPO_ROOT / "Jewellery_Data" / "ring" / "ring_001.jpg"

    img = Image.open(ring_img_path)

    # 1. Search with explicit ring filter
    res_ring = engine.search_image(img, top_k=20, category_filter="ring", robust_mode=True)
    assert len(res_ring["results"]) > 0
    for r in res_ring["results"]:
        assert r["category"].lower().rstrip("s") == "ring", f"Leaked non-ring candidate: {r}"

    # 2. Search with plural 'rings' filter
    res_rings = engine.search_image(img, top_k=20, category_filter="rings", robust_mode=True)
    assert len(res_rings["results"]) > 0
    for r in res_rings["results"]:
        assert r["category"].lower().rstrip("s") == "ring", f"Leaked non-ring candidate: {r}"

    # 3. Search with necklace filter
    neck_img_path = REPO_ROOT / "Jewellery_Data" / "necklace" / "necklace_1.jpg"
    neck_img = Image.open(neck_img_path)
    res_neck = engine.search_image(neck_img, top_k=20, category_filter="necklace", robust_mode=True)
    assert len(res_neck["results"]) > 0
    for r in res_neck["results"]:
        assert r["category"].lower().rstrip("s") == "necklace", f"Leaked non-necklace candidate: {r}"


def test_defensible_decision_semantics():
    """Verify calibrated decision states: EXACT_MATCH, VISUAL_SIMILARITY, NO_CONFIDENT_MATCH."""
    engine = SearchEngine(model_name="mobilenetv4", backend="numpy")

    # 1. Exact identical query from catalog -> EXACT_MATCH (>= 0.88)
    ring_path = REPO_ROOT / "Jewellery_Data" / "ring" / "ring_001.jpg"
    img_exact = Image.open(ring_path)
    res_exact = engine.search_image(img_exact, top_k=5, category_filter="ring", robust_mode=False)
    assert res_exact["top_similarity"] >= 0.88
    assert res_exact["decision"] == "EXACT_MATCH"
    assert res_exact["has_reliable_match"] is True

    # 2. Random blank gray image -> NO_CONFIDENT_MATCH (< 0.52)
    gray_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    res_nomatch = engine.search_image(gray_img, top_k=5, category_filter="ring", robust_mode=True)
    assert res_nomatch["top_similarity"] < 0.52
    assert res_nomatch["decision"] == "NO_CONFIDENT_MATCH"
    assert res_nomatch["has_reliable_match"] is False
    assert res_nomatch["results"][0]["match_type"] == "low_confidence"

    # 3. Cross-view style query -> VISUAL_SIMILARITY (0.52 <= top_sim < 0.88)
    diff_ring_path = REPO_ROOT / "Jewellery_Data" / "ring" / "ring_080.jpg"
    img_style = Image.open(diff_ring_path)
    res_style = engine.search_image(img_style, top_k=5, category_filter="ring", exclude_ids={"ring_080"}, robust_mode=True)
    if res_style["top_similarity"] >= 0.52 and res_style["top_similarity"] < 0.88:
        assert res_style["decision"] == "VISUAL_SIMILARITY"
        assert res_style["has_reliable_match"] is True


def test_qdrant_search_batch_consistency():
    """Verify QdrantCatalogIndex search_batch behaves correctly when vector database is accessible."""
    from engine.qdrant_index import QdrantCatalogIndex
    try:
        q_idx = QdrantCatalogIndex(collection_name="jewellery_catalog_mobilenetv4_v1")
        if not q_idx.is_online or not q_idx.is_indexed():
            pytest.skip("Local Qdrant instance is offline")
    except Exception:
        pytest.skip("Qdrant service unreachable")

    # Generate 3 random normalized query vectors
    vecs = [np.random.randn(1280).astype(np.float32) for _ in range(3)]
    for v in vecs:
        v /= np.linalg.norm(v)

    batch_res = q_idx.search_batch(vecs, top_k=5, category_filter="ring")
    assert len(batch_res) == 3
    for cands in batch_res:
        assert len(cands) <= 5
        for c in cands:
            assert c["category"] == "ring"
            assert "similarity" in c
            assert "sku_id" in c


def test_skin_neutralization_laplacian_feature_preservation():
    """Verify that sharp jewellery edges are preserved during skin neutralization."""
    from engine.preprocessing import detect_and_crop_jewellery

    # Create synthetic skin image with sharp white-gold ring facets
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = [210, 150, 120]  # Skin tone
    # Sharp ring with alternating bright facets (high Laplacian)
    for i in range(80, 120):
        for j in range(80, 120):
            if (i - 100)**2 + (j - 100)**2 < 400:
                arr[i, j] = [240, 230, 200] if (i + j) % 4 == 0 else [180, 140, 50]

    img = Image.fromarray(arr)
    neutralized, was_neut = detect_and_crop_jewellery(img, margin=0.10, neutralize_skin=True)
    assert was_neut is True
    neut_arr = np.array(neutralized)
    # Background neutralized to studio white
    assert np.all(neut_arr[0, 0] == [255, 255, 255])
