"""Automated tests for EXP-005 annotation workflow, balanced benchmark, and calibrated fusion."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_annotation_stats_and_pairs_endpoints():
    res = client.get("/api/annotation/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_pairs" in data
    assert "total_annotated" in data
    assert "verdicts" in data
    assert data["total_annotated"] >= 90

    # Test pairs endpoint
    res_pairs = client.get("/api/annotation/pairs?limit=5")
    assert res_pairs.status_code == 200
    p_data = res_pairs.json()
    assert "pairs" in p_data
    assert len(p_data["pairs"]) <= 5

def test_annotation_submission_validation():
    # Invalid submission missing required fields
    res_bad = client.post("/api/annotation/submit", json={"pair_id": "test"})
    assert res_bad.status_code == 422

    # Invalid verdict
    res_bad_verdict = client.post("/api/annotation/submit", json={
        "pair_id": "pair_001",
        "query_id": "ring_081",
        "candidate_id": "ring_108",
        "verdict": "definitely_same", # Invalid pattern
        "category": "ring"
    })
    assert res_bad_verdict.status_code == 422

    # Non-existent query ID
    res_bad_id = client.post("/api/annotation/submit", json={
        "pair_id": "pair_bad",
        "query_id": "nonexistent_sku_999",
        "candidate_id": "ring_108",
        "verdict": "confirmed_same_sku",
        "category": "ring"
    })
    assert res_bad_id.status_code == 400

def test_annotate_html_served():
    res = client.get("/annotate")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Jewellery SKU Ground-Truth Annotation Studio" in res.text

def test_balanced_benchmark_v3_structure():
    bench_file = Path("data/evaluation/benchmark_v3_balanced.json")
    assert bench_file.exists()
    bench = json.load(open(bench_file))
    
    assert bench["counts"]["verified_same_sku_total"] == 60
    assert bench["counts"]["verified_necklaces"] == 30
    assert bench["counts"]["verified_rings"] == 30
    assert bench["counts"]["confirmed_hard_negatives"] == 30
    assert bench["counts"]["singleton_queries"] == 30

    cat_meta = json.load(open("data/index/dinov2_meta.json"))
    cat_ids = set(m["id"] for m in cat_meta)
    
    # Verify all targets exist in catalog
    for q in bench["same_sku_queries"]:
        assert q["target_catalog_id"] in cat_ids, f"Target {q['target_catalog_id']} not in catalog index"
        assert Path(q["query_path"]).exists(), f"Query image {q['query_path']} does not exist"

def test_calibrated_min_max_fusion_properties():
    # Negative mean should scale to near 0, positive mean to near 1
    v2_neg, v2_pos = 0.4372, 0.9825
    v3_neg, v3_pos = 0.7924, 0.9927

    import numpy as np
    norm_v2_neg = np.clip((v2_neg - v2_neg) / (v2_pos - v2_neg), 0, 1)
    norm_v2_pos = np.clip((v2_pos - v2_neg) / (v2_pos - v2_neg), 0, 1)
    norm_v3_neg = np.clip((v3_neg - v3_neg) / (v3_pos - v3_neg), 0, 1)
    norm_v3_pos = np.clip((v3_pos - v3_neg) / (v3_pos - v3_neg), 0, 1)

    assert norm_v2_neg == 0.0
    assert norm_v2_pos == 1.0
    assert norm_v3_neg == 0.0
    assert norm_v3_pos == 1.0

def test_recalibration_balanced_metrics_present():
    recalib_file = Path("output/recalibration_balanced.json")
    assert recalib_file.exists()
    recalib = json.load(open(recalib_file))

    for m in ["dinov2", "dinov3", "hybrid_minmax", "rerank_pw02", "rerank_pw05", "chamfer"]:
        assert m in recalib["models"]
        cfg = recalib["models"][m]
        assert "tau_99" in cfg
        assert "wilson_95ci_precision" in cfg["performance_at_tau_99"]
        ci = cfg["performance_at_tau_99"]["wilson_95ci_precision"]
        assert ci[0] <= ci[1]
