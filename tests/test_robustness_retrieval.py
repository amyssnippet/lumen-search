"""Regression and unit tests for retrieval robustness, skin neutralization, multi-crop fusion, and score calibration."""

import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from engine.embeddings import MobileNetV4Encoder
from engine.preprocessing import (
    PreprocessConfig,
    assess_image_quality,
    detect_and_crop_jewellery,
    extract_robust_query_crops,
    preprocess_query_image,
)
from engine.search import SearchEngine
from engine.index import EXACT_MATCH_THRESHOLD, SIMILARITY_THRESHOLD, SIMILAR_DESIGN_THRESHOLD

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_skin_detection_and_neutralization():
    """Verify that images with skin backgrounds are flagged and neutralized to white."""
    # Synthetic image with skin tones (approx RGB (210, 150, 120)) and a central metallic ring
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = [210, 150, 120]  # Skin tone background
    # Add high-contrast white gold circle in the center
    rr, cc = np.ogrid[:200, :200]
    circle_mask = ((rr - 100) ** 2 + (cc - 100) ** 2) <= 30 ** 2
    arr[circle_mask] = [230, 230, 240]

    img = Image.fromarray(arr)
    quality = assess_image_quality(img)

    assert quality.skin_detected is True
    assert quality.skin_ratio > 0.50

    # Neutralize skin
    neutralized, was_neutralized = detect_and_crop_jewellery(img, margin=0.10, neutralize_skin=True)
    assert was_neutralized is True

    # The background should now be converted to studio white (255, 255, 255)
    neut_arr = np.array(neutralized)
    # Check top-left corner is studio white
    assert np.all(neut_arr[0, 0] == [255, 255, 255])


def test_robust_multi_crop_generation():
    """Verify that extract_robust_query_crops generates the 3 bounded query views."""
    img = Image.new("RGB", (300, 400), color=(240, 240, 240))
    crops = extract_robust_query_crops(img)

    assert len(crops) >= 2  # At least global and center zoom
    labels = [label for _, label in crops]
    assert "global" in labels
    assert "center_zoom_80" in labels

    # Check output sizes match target 224x224
    for crop_img, _ in crops:
        assert crop_img.size == (224, 224)


def test_score_calibration_thresholds():
    """Verify calibrated score thresholds across tiers."""
    assert EXACT_MATCH_THRESHOLD == 0.88
    assert SIMILARITY_THRESHOLD == 0.70
    assert SIMILAR_DESIGN_THRESHOLD == 0.52


def test_search_engine_robust_mode_execution():
    """Verify SearchEngine executes robust_mode with multi-crop max pooling without errors."""
    engine = SearchEngine(model_name="mobilenetv4", backend="numpy")
    dummy_img = Image.new("RGB", (256, 256), color=(250, 250, 250))

    res = engine.search_image(
        image=dummy_img,
        top_k=5,
        robust_mode=True
    )

    assert "timing_ms" in res
    assert "quality" in res
    assert "results" in res
    assert "robust_mode" in res
    assert res["robust_mode"] is True
    assert "skin_detected" in res["quality"]
    assert "robust_crops" in res["quality"]
    assert len(res["results"]) <= 5

    # Check calibrated confidence labels on candidate results
    for r in res["results"]:
        assert "similarity" in r
        assert "match_type" in r
        assert "confidence_label" in r
        sim = r["similarity"]
        if sim >= 0.88:
            assert r["match_type"] == "exact_match"
        elif sim >= 0.70:
            assert r["match_type"] == "high_confidence"
        elif sim >= 0.52:
            assert r["match_type"] == "similar_design"
        else:
            assert r["match_type"] == "low_confidence"


def test_motivating_worn_ring_retrieval():
    """Verify that the motivating real worn ring photograph achieves viable retrieval."""
    real_worn_path = REPO_ROOT / "Jewellery_Data" / "ring" / "ring-downloaded.png"
    if not real_worn_path.exists():
        pytest.skip("ring-downloaded.png not present in repository")

    engine = SearchEngine(model_name="mobilenetv4", backend="numpy")
    img = Image.open(real_worn_path)

    res = engine.search_image(
        image=img,
        top_k=10,
        robust_mode=True
    )

    assert res["quality"]["skin_detected"] is True
    assert res["quality"]["skin_ratio"] > 0.20
    assert len(res["results"]) > 0

    top_result = res["results"][0]
    # The image is either ring-downloaded itself (if in index) or matches ring_038/ring_063
    assert top_result["similarity"] >= 0.50
    # Top match should be calibrated at least as a visually similar design or verified match
    assert top_result["match_type"] in ("similar_design", "high_confidence", "exact_match")
