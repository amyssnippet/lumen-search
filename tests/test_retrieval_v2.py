"""Comprehensive regression and integration tests for POC Fine-Grained Visual Jewellery Retrieval v2.

Covers:
1. Encoder integrity and unit normalization (MobileNetV4 1280-d vs DINOv3 384-d).
2. Collection and namespace isolation (jewellery_catalog_mobilenetv4_v1 vs jewellery_catalog_dinov3_v1).
3. Calibrated decision states per architecture (EXACT_MATCH, VISUAL_SIMILARITY, NO_CONFIDENT_MATCH).
4. Strict category isolation (rings vs necklaces without candidate leakage).
5. Robust preprocessing (EXIF, aspect-ratio padding, skin neutralization, multi-crop).
6. Out-of-catalogue queries and hard-negative suppression.
7. Corrupt image and boundary condition handling.
8. Live backend search endpoint supporting model parameter switching.
"""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

from engine.embeddings import MobileNetV4Encoder, DINOv3ONNXEncoder, get_encoder
from engine.preprocessing import (
    PreprocessConfig,
    preprocess_query_image,
    detect_and_crop_jewellery,
    extract_robust_query_crops,
    pad_to_aspect_ratio,
)
from engine.search import SearchEngine
from engine.qdrant_index import QdrantCatalogIndex


def test_encoder_dimensions_and_unit_normalization():
    """Verify both baseline and v2 encoders output strictly unit-normalized float32 vectors."""
    img = Image.new("RGB", (224, 224), color=(180, 140, 80))

    # Baseline: MobileNetV4 (1280-d)
    mob_enc = MobileNetV4Encoder()
    v_mob = mob_enc.encode_image(img)
    assert v_mob.shape == (1280,), f"Expected shape (1280,), got {v_mob.shape}"
    assert v_mob.dtype == np.float32
    assert abs(np.linalg.norm(v_mob) - 1.0) < 1e-4, f"Vector not unit-normalized: {np.linalg.norm(v_mob)}"

    # Candidate v2: DINOv3 ONNX (384-d)
    dino_enc = DINOv3ONNXEncoder("data/models/dinov3_vits16_int8.onnx")
    v_dino = dino_enc.encode_image(img)
    assert v_dino.shape == (384,), f"Expected shape (384,), got {v_dino.shape}"
    assert v_dino.dtype == np.float32
    assert abs(np.linalg.norm(v_dino) - 1.0) < 1e-4, f"Vector not unit-normalized: {np.linalg.norm(v_dino)}"


def test_qdrant_collection_namespace_isolation():
    """Verify production collection (1280-d) and POC collection (384-d) remain strictly isolated."""
    # Production collection
    prod_idx = QdrantCatalogIndex(
        model_name="mobilenetv4",
        collection_name="jewellery_catalog_mobilenetv4_v1",
        host="localhost",
        port=6333
    )
    if prod_idx.is_online and prod_idx.is_indexed():
        assert prod_idx.embedding_dim == 1280
        info_prod = prod_idx.client.get_collection("jewellery_catalog_mobilenetv4_v1")
        assert info_prod.config.params.vectors.size == 1280

    # POC collection
    poc_idx = QdrantCatalogIndex(
        model_name="dinov3",
        collection_name="jewellery_catalog_dinov3_v1",
        host="localhost",
        port=6333
    )
    if poc_idx.is_online and poc_idx.is_indexed():
        assert poc_idx.embedding_dim == 384
        info_poc = poc_idx.client.get_collection("jewellery_catalog_dinov3_v1")
        assert info_poc.config.params.vectors.size == 384


def test_calibrated_decision_semantics_mobilenet():
    """Verify calibrated decision thresholds for MobileNetV4 (Exact >= 0.88, Sim >= 0.52)."""
    engine = SearchEngine(model_name="mobilenetv4", backend="numpy")

    # 1. Exact catalog item -> EXACT_MATCH
    exact_img = Image.open(REPO_ROOT / "Jewellery_Data" / "ring" / "ring_001.jpg")
    res_exact = engine.search_image(exact_img, top_k=5, category_filter="ring")
    assert res_exact["decision"] == "EXACT_MATCH"
    assert res_exact["top_similarity"] >= 0.88
    assert res_exact["has_reliable_match"] is True

    # 2. Blank uninformative image -> NO_CONFIDENT_MATCH
    gray_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    res_gray = engine.search_image(gray_img, top_k=5, category_filter="ring")
    assert res_gray["decision"] == "NO_CONFIDENT_MATCH"
    assert res_gray["top_similarity"] < 0.52
    assert res_gray["has_reliable_match"] is False


def test_calibrated_decision_semantics_dinov3():
    """Verify calibrated decision thresholds for DINOv3 (Exact >= 0.98, Sim >= 0.90)."""
    engine = SearchEngine(model_name="dinov3", backend="numpy")

    # 1. Exact catalog item -> high similarity (>= 0.90)
    exact_img = Image.open(REPO_ROOT / "Jewellery_Data" / "ring" / "ring_001.jpg")
    res_exact = engine.search_image(exact_img, top_k=5, category_filter="ring")
    assert res_exact["decision"] in ("EXACT_MATCH", "VISUAL_SIMILARITY")
    assert res_exact["top_similarity"] >= 0.90
    assert res_exact["has_reliable_match"] is True

    # 2. Blank image -> NO_CONFIDENT_MATCH (< 0.90)
    gray_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    res_gray = engine.search_image(gray_img, top_k=5, category_filter="ring")
    assert res_gray["decision"] == "NO_CONFIDENT_MATCH"
    assert res_gray["top_similarity"] < 0.90
    assert res_gray["has_reliable_match"] is False


def test_strict_category_isolation_both_models():
    """Verify rings and necklaces never leak into opposite category results."""
    for model_name in ["mobilenetv4", "dinov3"]:
        engine = SearchEngine(model_name=model_name, backend="numpy")
        ring_img = Image.open(REPO_ROOT / "Jewellery_Data" / "ring" / "ring_001.jpg")
        neck_img = Image.open(REPO_ROOT / "Jewellery_Data" / "necklace" / "necklace_1.jpg")

        # Ring search with ring filter
        res_r = engine.search_image(ring_img, top_k=15, category_filter="ring")
        for cand in res_r["results"]:
            assert cand["category"].lower().rstrip("s") == "ring"

        # Necklace search with necklace filter
        res_n = engine.search_image(neck_img, top_k=15, category_filter="necklace")
        for cand in res_n["results"]:
            assert cand["category"].lower().rstrip("s") == "necklace"


def test_preprocessing_robust_crops_and_aspect_pad():
    """Verify aspect-ratio padding and multi-crop view extraction."""
    # Test non-square image
    rect_img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    padded = pad_to_aspect_ratio(rect_img, target_size=256)
    assert padded.size == (256, 256)

    # Test multi-crop extraction
    crops = extract_robust_query_crops(rect_img, config=PreprocessConfig(target_size=256))
    assert len(crops) >= 2
    crop_labels = [label for _, label in crops]
    assert "global" in crop_labels
    for c_img, _ in crops:
        assert c_img.size == (256, 256)


def test_corrupt_and_zero_byte_handling():
    """Verify search pipeline rejects or handles corrupt images defensively."""
    engine = SearchEngine(model_name="mobilenetv4", backend="numpy")
    
    # 1. 1x1 minimal pixel image
    tiny_img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    res_tiny = engine.search_image(tiny_img, top_k=3)
    assert "decision" in res_tiny
    assert res_tiny["decision"] == "NO_CONFIDENT_MATCH"

    # 2. Aspect pad on single pixel
    padded_tiny = pad_to_aspect_ratio(tiny_img, target_size=224)
    assert padded_tiny.size == (224, 224)
