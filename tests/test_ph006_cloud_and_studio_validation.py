"""Tests for PH-006: Physical Ring Studio Ingestion & Distributed 1M+ Cluster Validation."""

import json
from pathlib import Path
import pytest

from scripts.ingest_studio_batch import validate_sku_batch, StudioBatchIngestionManager
from scripts.evaluate_ph006_crossview_retrieval import clopper_pearson_interval, clopper_pearson_upper_bound


def test_ph006_manifest_accounting_invariants():
    """Verify manifest audit deliverables and exact mathematical accounting parity."""
    audit_path = Path("output/ph006_manifest_integrity_audit.json")
    assert audit_path.exists(), "ph006_manifest_integrity_audit.json must exist"

    with open(audit_path, "r") as fp:
        audit = json.load(fp)

    inv = audit["accounting_invariants"]
    assert inv["invariants_passed"] is True
    assert inv["total_records"] == 518
    assert inv["raw_catalog_images"] == 492
    assert inv["synthetic_test_images"] == 26
    assert inv["physical_singletons"] == 423
    assert inv["verified_identical_images"] == 65
    assert inv["quarantined_conflicting_images"] == 4
    assert inv["physical_singletons"] + inv["verified_identical_images"] + inv["quarantined_conflicting_images"] == 492

    rings = audit["ring_dataset_accounting"]
    assert rings["verified_physical_ring_skus"] == 5
    assert rings["verified_physical_ring_captures"] == 10
    assert rings["unlinked_singleton_rings"] == 181
    assert rings["target_ring_skus"] == 50
    assert rings["acquisition_status"] == "PARTIALLY_COMPLETE_PENDING_STUDIO_ASSETS"


def test_ph006_studio_batch_validator():
    """Test studio batch validator enforces canonical 4-angle completeness and quality."""
    # Incomplete SKU (missing profile and shank)
    incomplete_sku = {
        "sku_id": "TEST-SKU-INCOMPLETE",
        "provenance_gate_status": {"gate4_nontransitive_verified": True},
        "views": [
            {"view_angle": "crown", "rel_path": "Jewellery_Data/ring/ring_078.jpg"},
            {"view_angle": "angle_45", "rel_path": "Jewellery_Data/ring/ring_079.jpg"}
        ]
    }
    res_incomplete = validate_sku_batch(incomplete_sku)
    assert res_incomplete["is_valid"] is False
    assert "missing_angles" in res_incomplete
    assert "profile" in res_incomplete["missing_angles"]
    assert "shank" in res_incomplete["missing_angles"]

    # Gate 4 violation
    gate4_violation = {
        "sku_id": "TEST-SKU-GATE4-FAIL",
        "provenance_gate_status": {"gate4_nontransitive_verified": False},
        "views": [
            {"view_angle": "crown", "rel_path": "Jewellery_Data/ring/ring_078.jpg"},
            {"view_angle": "angle_45", "rel_path": "Jewellery_Data/ring/ring_079.jpg"},
            {"view_angle": "profile", "rel_path": "Jewellery_Data/ring/ring_080.jpg"},
            {"view_angle": "shank", "rel_path": "Jewellery_Data/ring/ring_081.jpg"}
        ]
    }
    res_gate4 = validate_sku_batch(gate4_violation)
    assert res_gate4["is_valid"] is False
    assert any("Gate 4" in err for err in res_gate4["errors"])


def test_ph006_crossview_retrieval_metrics():
    """Test cross-view retrieval metrics, confidence intervals, and SKU aggregation benefit."""
    eval_path = Path("output/ph006_crossview_evaluation.json")
    assert eval_path.exists(), "ph006_crossview_evaluation.json must exist"

    with open(eval_path, "r") as fp:
        res = json.load(fp)

    # Physical Rings
    rings_eval = res["canonical_physical_ring_cross_view_evaluation"]
    assert rings_eval["verified_physical_ring_skus_count"] == 5
    sku_agg = rings_eval["sku_aggregated_retrieval"]
    assert sku_agg["recall_at_1"] == 1.0
    assert sku_agg["candidate_recall_at_20"] == 1.0
    assert sku_agg["mrr"] == 1.0
    ci = sku_agg["recall_at_1_95ci"]
    assert ci[0] == 0.4782 and ci[1] == 1.0

    # Real Necklaces
    neck_eval = res["photographic_necklace_evaluation"]
    assert neck_eval["sku_aggregated_retrieval"]["recall_at_1"] >= neck_eval["view_level_retrieval"]["recall_at_1"]
    assert neck_eval["sku_aggregation_delta_recall_at_1"] >= 0.0

    # Synthetic Rings
    assert res["synthetic_ring_perturbations_control"]["evaluation"]["recall_at_1"] == 1.0


def test_ph006_independent_hard_negatives():
    """Verify independent negative test split has zero identity leakage and FAR bound <= 1.0%."""
    eval_path = Path("output/ph006_crossview_evaluation.json")
    with open(eval_path, "r") as fp:
        res = json.load(fp)

    hn = res["independent_hard_negative_evaluation"]
    assert hn["calibration_set"]["sample_size"] == 100
    assert hn["held_out_test_set"]["sample_size"] == 300

    frozen = hn["frozen_threshold_results"]["tau_0_974"]
    assert frozen["held_out_negative_samples"] == 300
    assert frozen["observed_false_accepts"] == 0
    assert frozen["one_sided_95_upper_bound_pct"] <= 1.0
    assert frozen["status"] == "PASSED"


def test_ph006_cloud_deployment_manifest():
    """Verify cloud deployment manifest specifies c6i.4xlarge, Qdrant 1.19.2, and persistent storage."""
    manifest_path = Path("output/ph006_cloud_deployment_manifest.json")
    assert manifest_path.exists(), "ph006_cloud_deployment_manifest.json must exist"

    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    hw = manifest["cloud_hardware_specification"]
    assert hw["instance_type"] == "c6i.4xlarge"
    assert hw["vcpu_count"] == 16
    assert hw["physical_ram_gb"] == 32.0
    assert hw["primary_storage"]["volume_type"] == "gp3"

    sw = manifest["software_environment"]
    assert sw["qdrant_image"] == "qdrant/qdrant:v1.19.2"
    assert "vm.max_map_count" in sw["sysctl_optimizations"]

    costs = manifest["infrastructure_cost_model"]
    assert costs["total_monthly_cost_1yr_reserved_usd"] < costs["total_monthly_cost_on_demand_usd"]


def test_clopper_pearson_utility_math():
    """Verify statistical confidence interval and upper bound mathematics."""
    # N=300, 0 false accepts
    ub = clopper_pearson_upper_bound(0, 300, 0.95)
    assert ub < 0.010 # Bound < 1.0%
    assert round(ub * 100, 3) == 0.994

    # N=5, 5 hits
    ci = clopper_pearson_interval(5, 5, 0.95)
    assert ci[0] == 0.4782
    assert ci[1] == 1.0
