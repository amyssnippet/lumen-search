"""Integration tests for FastAPI endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "Jewellery Visual Retrieval Engine" in data["service"]

def test_catalog_stats_endpoint():
    resp = client.get("/api/catalog/stats?model=dinov2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_images_in_dataset"] == 492
    assert "ring" in data["dataset_categories"]
    assert "necklace" in data["dataset_categories"]
    assert data["index_stats"]["indexed_count"] == 432

def test_search_endpoint_with_valid_image():
    # Create a small valid test image in memory
    img = Image.new("RGB", (100, 100), (200, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_ring.jpg", buf, "image/jpeg")}
    data = {
        "model": "dinov2",
        "search_mode": "similar",
        "category_filter": "all",
        "top_k": 5,
        "auto_enhance": True
    }
    
    resp = client.post("/api/search", files=files, data=data)
    assert resp.status_code == 200
    res_json = resp.json()
    assert res_json["model_name"] == "dinov2"
    assert "timing_ms" in res_json
    assert len(res_json["results"]) == 5
    assert "image_url" in res_json["results"][0]
    assert res_json["results"][0]["image_url"].startswith("/media/")
    # Confirm raw internal path is not exposed
    assert "rel_path" not in res_json["results"][0]

def test_search_endpoint_invalid_file_rejected():
    files = {"file": ("malicious.txt", io.BytesIO(b"not an image"), "text/plain")}
    resp = client.post("/api/search", files=files)
    assert resp.status_code == 400

def test_feedback_submission():
    payload = {
        "query_id": "test_query_001",
        "result_id": "ring_001",
        "feedback_type": "correct",
        "similarity_score": 0.965,
        "notes": "Exact match verified"
    }
    resp = client.post("/api/feedback", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

def test_media_path_traversal_protection():
    # Attempt directory traversal
    resp = client.get("/media/ring/../../etc/passwd")
    assert resp.status_code in (404, 422)

    resp2 = client.get("/media/invalid_category/ring_001.jpg")
    assert resp2.status_code == 404
