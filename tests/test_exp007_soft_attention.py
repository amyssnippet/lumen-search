"""Regression and validation tests for EXP-007 Soft Spatial Attention Engine."""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from engine.soft_attention import (
    GRID_SIZE,
    compute_uniform_weights,
    compute_foreground_weights,
    compute_soft_region_weights,
    compute_gaussian_weights,
    compute_combined_weights,
    aggregate_patch_tokens,
    render_attention_heatmap
)
from engine.attention_search import SoftAttentionSearchEngine
from app.main import app

client = TestClient(app)


@pytest.fixture
def synthetic_jewellery_image():
    """Create a synthetic test image with a dark foreground ring-like object on white background."""
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    pixels = img.load()
    # Draw ring-like circle
    for y in range(224):
        for x in range(224):
            dist = np.sqrt((x - 112) ** 2 + (y - 112) ** 2)
            if 40 <= dist <= 60:
                pixels[x, y] = (40, 40, 50) # Dark metallic ring shank
            elif dist < 20:
                pixels[x, y] = (20, 180, 220) # Cyan gemstone center
    return img


def test_uniform_weights():
    weights = compute_uniform_weights(grid_size=16)
    assert weights.shape == (256,)
    assert np.allclose(weights, 1.0)


def test_foreground_weights(synthetic_jewellery_image):
    weights = compute_foreground_weights(synthetic_jewellery_image, grid_size=16, min_floor=0.05)
    assert weights.shape == (256,)
    assert np.all(weights >= 0.05)
    assert np.all(weights <= 1.0)
    # The corner background should be near minimum floor (0.05)
    corner_idx = 0
    assert weights[corner_idx] < 0.2
    # The center (gemstone) should have higher weight
    center_idx = 8 * 16 + 8
    assert weights[center_idx] > weights[corner_idx]


def test_soft_region_weights():
    # Box around center: ymin=0.4, xmin=0.4, ymax=0.6, xmax=0.6
    box = [0.4, 0.4, 0.6, 0.6]
    weights = compute_soft_region_weights(box, grid_size=16, min_weight=0.25)
    assert weights.shape == (256,)
    assert np.all(weights >= 0.25)
    assert np.all(weights <= 1.0)

    # Center token should have maximum weight 1.0
    center_idx = 8 * 16 + 8
    assert np.isclose(weights[center_idx], 1.0, atol=0.05)

    # Far corner token should decay towards min_weight
    assert weights[0] < 0.5


def test_gaussian_weights():
    box = [0.4, 0.4, 0.6, 0.6]
    weights = compute_gaussian_weights(box, grid_size=16, min_weight=0.15)
    assert weights.shape == (256,)
    assert np.all(weights >= 0.15)
    assert np.all(weights <= 1.0)


def test_combined_weights(synthetic_jewellery_image):
    box = [0.4, 0.4, 0.6, 0.6]
    weights = compute_combined_weights(synthetic_jewellery_image, box, is_fallback=False, grid_size=16, min_floor=0.05)
    assert weights.shape == (256,)
    assert np.all(weights >= 0.05)
    assert np.all(weights <= 1.0)
    assert np.max(weights) == 1.0


def test_patch_aggregation_single():
    np.random.seed(42)
    patch_tokens = np.random.randn(256, 384).astype(np.float32)
    weights = np.ones(256, dtype=np.float32)
    agg = aggregate_patch_tokens(patch_tokens, weights)

    assert agg.shape == (384,)
    # Verify unit L2 norm
    norm = np.linalg.norm(agg)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_patch_aggregation_batch():
    np.random.seed(42)
    patch_tokens = np.random.randn(4, 256, 384).astype(np.float32)
    weights = np.random.rand(4, 256).astype(np.float32)
    agg = aggregate_patch_tokens(patch_tokens, weights)

    assert agg.shape == (4, 384)
    norms = np.linalg.norm(agg, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_patch_aggregation_zero_weights_stability():
    patch_tokens = np.ones((256, 64), dtype=np.float32)
    weights = np.zeros(256, dtype=np.float32)
    agg = aggregate_patch_tokens(patch_tokens, weights)
    # Should not produce NaNs or Infs
    assert not np.isnan(agg).any()
    assert not np.isinf(agg).any()


def test_render_attention_heatmap(synthetic_jewellery_image):
    weights = compute_foreground_weights(synthetic_jewellery_image, grid_size=16)
    heatmap = render_attention_heatmap(weights, synthetic_jewellery_image)

    assert isinstance(heatmap, Image.Image)
    assert heatmap.size == synthetic_jewellery_image.size
    assert heatmap.mode == "RGB"


def test_soft_attention_search_engine(synthetic_jewellery_image):
    engine = SoftAttentionSearchEngine(model_name="dinov2")
    res = engine.search(
        image=synthetic_jewellery_image,
        config_name="soft_combined",
        top_k=5
    )

    assert res["model_name"] == "dinov2"
    assert res["config_name"] == "soft_combined"
    assert "query_localization" in res
    assert "attention_weights" in res
    assert len(res["attention_weights"]) == 256
    assert "timings_ms" in res
    assert len(res["results"]) > 0

    top1 = res["results"][0]
    assert "id" in top1
    assert "score" in top1
    assert "similarity" in top1
    assert "match_type" in top1
    assert "confidence_label" in top1


def test_api_search_endpoint_soft_attention(synthetic_jewellery_image):
    buf = io.BytesIO()
    synthetic_jewellery_image.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/search",
        files={"file": ("query.jpg", buf, "image/jpeg")},
        data={
            "model": "soft_attention",
            "top_k": 5,
            "category_filter": "all",
            "attention_mode": "soft_combined"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    assert "localization" in data
    assert "attention_heatmap_b64" in data
    assert data["attention_heatmap_b64"] is not None
    assert len(data["attention_heatmap_b64"]) > 100


def test_api_evaluation_endpoint():
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "dinov2" in data
    assert "dinov3" in data
    assert "Baseline Global (CLS)" in data["dinov2"]
    assert "Baseline Global (CLS)" in data["dinov3"]
