"""Automated regression tests for PH-002 Workstream C: Hard-Negative Validation."""

import json
from pathlib import Path
import pytest

def test_hard_negative_report_exists_and_valid():
    report_path = Path("output/ph002_hard_negative_evaluation.json")
    assert report_path.exists(), "ph002_hard_negative_evaluation.json must exist"
    
    with open(report_path, "r") as fp:
        data = json.load(fp)
        
    assert "metadata" in data
    assert "score_distribution_summary" in data
    assert "confusing_attribute_coverage" in data
    assert "frozen_threshold_evaluation" in data
    assert "precision_coverage_tradeoff_sweep" in data
    assert "statistical_limitations_and_population_drift" in data
    
    meta = data["metadata"]
    assert meta["sample_size"] >= 300, f"Expected >= 300 negative cases, got {meta['sample_size']}"
    assert meta["independent_query_skus"] == meta["sample_size"], "Queries must be 100% independent physical SKUs"
    assert meta["necklace_cases"] > 0
    assert meta["ring_cases"] > 0

def test_confusing_attributes_coverage():
    report_path = Path("output/ph002_hard_negative_evaluation.json")
    with open(report_path, "r") as fp:
        data = json.load(fp)
        
    attrs = data["confusing_attribute_coverage"]
    assert len(attrs) >= 4, "Must cover at least 4 fine-grained confusing jewellery attributes"

def test_frozen_threshold_statistical_properties():
    report_path = Path("output/ph002_hard_negative_evaluation.json")
    with open(report_path, "r") as fp:
        data = json.load(fp)
        
    eval_frozen = data["frozen_threshold_evaluation"]["tau_exact"]
    assert eval_frozen["threshold"] == 0.9909
    assert "false_accepts" in eval_frozen
    assert "one_sided_95_upper_bound_pct" in eval_frozen
    assert "two_sided_95_ci_pct" in eval_frozen
    assert eval_frozen["total_cases"] >= 300

def test_precision_coverage_sweep():
    report_path = Path("output/ph002_hard_negative_evaluation.json")
    with open(report_path, "r") as fp:
        data = json.load(fp)
        
    sweep = data["precision_coverage_tradeoff_sweep"]
    assert len(sweep) >= 5
    # Verify monotonic reduction in false accepts as threshold increases
    fa_counts = [row["false_accepts"] for row in sweep]
    for i in range(len(fa_counts) - 1):
        assert fa_counts[i] >= fa_counts[i+1], "False accepts must be monotonically non-increasing"
