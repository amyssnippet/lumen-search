"""Tests for dataset integrity and manifest."""

import json
from pathlib import Path

def test_manifest_structure_and_counts():
    manifest_file = Path("data/manifest.json")
    assert manifest_file.exists(), "Manifest file should exist"
    
    with open(manifest_file, "r") as fp:
        data = json.load(fp)

    assert "metadata" in data
    assert "images" in data
    assert len(data["images"]) == 492
    assert data["metadata"]["catalog_count"] == 432
    assert data["metadata"]["test_count"] == 60
    assert data["metadata"]["twin_test_queries"] == 30

def test_no_identical_duplicate_leakage_in_test():
    with open("data/manifest.json", "r") as fp:
        data = json.load(fp)

    test_imgs = [img for img in data["images"] if img["split"] == "test"]
    catalog_imgs = [img for img in data["images"] if img["split"] == "catalog"]
    
    test_ids = {img["id"] for img in test_imgs}
    cat_ids = {img["id"] for img in catalog_imgs}
    
    # Strictly disjoint sets
    assert len(test_ids.intersection(cat_ids)) == 0
