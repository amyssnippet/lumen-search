"""Automated test suite for PH-001: Production Data and Evaluation Foundation."""

import json
from pathlib import Path
import pytest
from scipy import stats

from scripts.verify_identity_integrity import verify_integrity
from scripts.run_production_evaluation import wilson_interval, clopper_pearson_interval

def test_ingestion_manifest_schema_and_counts():
    manifest_path = Path("data/ingestion_manifest.json")
    assert manifest_path.exists(), "data/ingestion_manifest.json must exist"
    
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)
        
    summary = manifest["summary"]
    assert summary["total_images"] == 518
    assert summary["raw_images_count"] == 492
    assert summary["synthetic_images_count"] == 26
    assert summary["genuine_independent_captures"] == 488
    assert summary["exact_byte_duplicate_clones"] == 4
    assert len(manifest["records"]) == 518

def test_four_gate_standards_enforced():
    manifest_path = Path("data/ingestion_manifest.json")
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    for r in manifest["records"]:
        fg = r["four_gate_status"]
        # Gate 1: No fake ERP lineage
        if r["sku_id"] is None:
            assert fg["gate1_erp_lineage"] is False, f"{r['image_id']} cannot claim gate 1 without sku_id"
            
        # Synthetic variants
        if r["physical_identity_verification_status"] == "synthetic_transformation":
            assert r["is_original_capture"] is False
            assert r["is_genuine_independent_capture"] is False
            assert r["authoritative_identity_ref"].startswith("SYNTH_")

        # Byte clones
        if r["exclusion_reason"] and "clone" in r["exclusion_reason"].lower():
            assert r["is_genuine_independent_capture"] is False

def test_verify_identity_integrity_script():
    audit_summary = verify_integrity("data/ingestion_manifest.json")
    assert audit_summary["integrity_passed"] is True
    assert audit_summary["errors_count"] == 0

def test_statistical_intervals_correctness():
    # 1. Wilson score interval
    low_w, high_w = wilson_interval(10, 20, confidence=0.95)
    assert 0.0 < low_w < 0.5 < high_w < 1.0
    
    # 2. Clopper-Pearson interval on 0/30 negatives
    low_cp_30, high_cp_30 = clopper_pearson_interval(0, 30, confidence=0.95)
    assert low_cp_30 == 0.0
    # Crucial finding: with N=30, the 95% two-sided upper bound is ~11.6% (one-sided ~9.5%)
    assert high_cp_30 > 0.09, "0/30 negatives cannot prove FAR <= 1.0% (upper bound must be > 9%)"
    
    # 3. Clopper-Pearson interval on 0/300 negatives
    low_cp_300, high_cp_300 = clopper_pearson_interval(0, 300, confidence=0.95)
    assert high_cp_300 <= 0.013, "N=300 negatives provides power for ~1% FAR upper bound"

def test_ph001_evaluation_report_structure():
    report_path = Path("output/ph001_evaluation_report.json")
    assert report_path.exists(), "output/ph001_evaluation_report.json must exist"
    
    with open(report_path, "r") as fp:
        report = json.load(fp)
        
    assert "exp008_baseline_reproduction" in report
    assert "ph001_leakage_safe_evaluation" in report
    assert "acceptance_checklist_status" in report
    
    ph001_eval = report["ph001_leakage_safe_evaluation"]
    assert "tau_exact" in ph001_eval["calibrated_thresholds"]
    assert "all_test" in ph001_eval["subgroup_metrics"]
    assert "real_photographic_rings" in ph001_eval["subgroup_metrics"]
    
    # Verify INSUFFICIENT_DATA status on real rings
    ring_stat = ph001_eval["subgroup_metrics"]["real_photographic_rings"]["statistical_power_status"]
    assert "INSUFFICIENT_DATA" in ring_stat
    
    # Verify FAR status
    far_stat = ph001_eval["negative_lookalike_evaluation"]["target_validation_status"]
    assert "NOT_STATISTICALLY_VALIDATED" in far_stat
