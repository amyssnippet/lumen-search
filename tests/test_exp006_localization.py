"""Unit and regression tests for EXP-006: Gemstone and component localization."""

import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from engine.localization import JewelleryLocalizer, LocalizationResult, compute_iou
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_iou_exact_match():
    box = [0.2, 0.2, 0.6, 0.6]
    assert compute_iou(box, box) == pytest.approx(1.0, rel=1e-4)

def test_iou_zero_overlap():
    box1 = [0.0, 0.0, 0.2, 0.2]
    box2 = [0.5, 0.5, 0.8, 0.8]
    assert compute_iou(box1, box2) == 0.0

def test_iou_partial_overlap():
    # Two boxes of area 0.5 x 0.5 = 0.25
    # Overlap is 0.25 x 0.5 = 0.125
    # Union is 0.25 + 0.25 - 0.125 = 0.375
    # IoU = 0.125 / 0.375 = 1/3 ~ 0.3333
    box1 = [0.0, 0.0, 0.5, 0.5]
    box2 = [0.25, 0.0, 0.75, 0.5]
    assert compute_iou(box1, box2) == pytest.approx(1.0 / 3.0, rel=1e-3)

def test_iou_degenerate_boxes():
    box_flat = [0.5, 0.5, 0.5, 0.5]
    box_normal = [0.1, 0.1, 0.6, 0.6]
    assert compute_iou(box_flat, box_normal) == 0.0
    assert compute_iou(box_flat, box_flat) == 0.0

def test_localizer_solid_white_fallback():
    """An entirely blank white studio backdrop should trigger graceful fallback."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    localizer = JewelleryLocalizer()
    res = localizer.locate(img, category="necklace")
    
    assert res.is_fallback is True
    assert res.method == "fallback_centre_crop"
    assert res.confidence == 0.40
    assert len(res.box) == 4
    # Bounding box should stay bounded in [0, 1]
    assert all(0.0 <= c <= 1.0 for c in res.box)

def test_localizer_necklace_pendant_crop():
    """Synthesize an image with a dark chain and a dense pendant at bottom center."""
    arr = np.full((300, 300, 3), 255, dtype=np.uint8)
    # Thin chain (e.g. 2px thick)
    arr[40:150, 80:82] = [40, 40, 40]
    arr[40:150, 218:220] = [40, 40, 40]
    # Dense pendant gemstone cluster at bottom centre (50x50 block)
    arr[160:220, 125:175] = [20, 30, 180] # Sapphire pendant
    
    img = Image.fromarray(arr)
    localizer = JewelleryLocalizer()
    res = localizer.locate(img, category="necklace")
    
    assert res.is_fallback is False
    assert res.label == "pendant_stone"
    assert res.confidence > 0.70
    assert res.box[0] >= 0.45 # ymin should isolate the lower half
    assert res.box[2] <= 0.85 # ymax

def test_localizer_ring_crown_crop():
    """Synthesize a ring with a circular shank and a top setting."""
    arr = np.full((300, 300, 3), 255, dtype=np.uint8)
    # Ring circular setting
    rr, cc = np.ogrid[:300, :300]
    circle_mask = ((rr - 160)**2 + (cc - 150)**2 <= 80**2) & ((rr - 160)**2 + (cc - 150)**2 >= 65**2)
    arr[circle_mask] = [180, 180, 180] # Silver shank
    # Top crown stone
    arr[70:100, 135:165] = [200, 20, 20] # Ruby crown stone
    
    img = Image.fromarray(arr)
    localizer = JewelleryLocalizer()
    res = localizer.locate(img, category="ring")
    
    assert res.is_fallback is False
    assert res.label == "centre_stone"
    assert res.confidence >= 0.80
    # Ring crown box should cover top region
    assert res.box[0] < 0.40 # Top crown ymin
    assert res.box[2] <= 0.60 # Upper half ymax

def test_api_region_endpoints(client):
    # 1. GET /region-annotate HTML view
    resp = client.get("/region-annotate")
    assert resp.status_code == 200
    assert "Jewellery Region & Gemstone Annotation Studio" in resp.text

    # 2. GET /api/regions/list
    resp = client.get("/api/regions/list")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert data["total"] >= 1

    # 3. POST /api/regions/propose
    first_item = data["items"][0]
    resp = client.post("/api/regions/propose", json={
        "image_id": first_item["id"],
        "category": first_item["category"]
    })
    assert resp.status_code == 200
    prop = resp.json()
    assert "box" in prop
    assert len(prop["box"]) == 4
    assert 0.0 <= prop["box"][0] <= 1.0

    # 4. POST /api/regions/save
    save_payload = {
        "image_id": "test_unit_item_001",
        "category": "necklace",
        "box": [0.50, 0.45, 0.75, 0.55],
        "label": "pendant_stone",
        "source": "human_verified",
        "annotator": "test_suite",
        "notes": "Unit test verified crop"
    }
    resp = client.post("/api/regions/save", json=save_payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # 5. GET /api/regions/{image_id}
    resp = client.get("/api/regions/test_unit_item_001")
    assert resp.status_code == 200
    ann = resp.json()
    assert ann["image_id"] == "test_unit_item_001"
    assert ann["box"] == [0.50, 0.45, 0.75, 0.55]

    # Clean up test artifact from data/region_annotations.json
    import json
    from pathlib import Path
    reg_path = Path("data/region_annotations.json")
    if reg_path.exists():
        with open(reg_path, "r") as fp:
            d = json.load(fp)
        if "test_unit_item_001" in d.get("annotations", {}):
            del d["annotations"]["test_unit_item_001"]
            d["total_annotated"] = len(d["annotations"])
            with open(reg_path, "w") as fp:
                json.dump(d, fp, indent=2)
