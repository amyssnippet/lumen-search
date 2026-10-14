"""Automated regression tests for PH-002 Gate 0: Count Reconciliation & Accounting."""

import json
from pathlib import Path
from collections import defaultdict, Counter
import pytest

def test_manifest_reconciled_counts():
    manifest_path = Path("data/ingestion_manifest.json")
    assert manifest_path.exists()
    
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)
        
    records = manifest["records"]
    assert len(records) == 518
    
    summary = manifest["summary"]
    assert summary["total_images"] == 518
    assert summary["raw_images_count"] == 492
    assert summary["synthetic_images_count"] == 26
    assert summary["genuine_independent_captures"] == 488
    assert summary["exact_byte_duplicate_clones"] == 4
    assert summary["total_unique_physical_identities"] == 456
    assert summary["total_unique_all_identities"] == 482

def test_reconciliation_json_file():
    recon_path = Path("output/ph002_count_reconciliation.json")
    assert recon_path.exists(), "ph002_count_reconciliation.json must exist"
    
    with open(recon_path, "r") as fp:
        recon = json.load(fp)
        
    accounting = recon["reconciled_ground_truth_accounting"]
    assert accounting["total_manifest_images"] == 518
    assert accounting["raw_catalog_images"] == 492
    assert accounting["synthetic_test_images"] == 26
    assert accounting["exact_byte_duplicate_clones"] == 4
    assert accounting["genuine_independent_captures"] == 488
    assert accounting["physical_singletons"] == 423
    assert accounting["verified_multishot_physical_images"] == 58
    assert accounting["exact_byte_duplicate_images"] == 7
    assert accounting["quarantined_conflicting_images"] == 4
    assert accounting["reconciled_physical_images_sum"] == 492
    
    clusters = accounting["physical_identity_clusters"]
    assert clusters["singleton_clusters"] == 423
    assert clusters["verified_multishot_clusters"] == 28
    assert clusters["exact_byte_duplicate_clusters"] == 3
    assert clusters["quarantined_clusters"] == 2
    assert clusters["total_physical_clusters"] == 456
    assert recon["reconciled_ground_truth_accounting"]["total_all_identity_clusters"] == 482

def test_category_breakdown():
    recon_path = Path("output/ph002_count_reconciliation.json")
    with open(recon_path, "r") as fp:
        recon = json.load(fp)
        
    cats = recon["category_breakdown"]
    neck = cats["necklaces"]
    assert neck["raw_images"] == 301
    assert neck["singletons"] == 242
    assert neck["verified_multishot_images"] == 48
    assert neck["exact_duplicate_images"] == 7
    assert neck["quarantined_images"] == 4
    assert neck["check_sum"] == 301
    assert neck["physical_clusters"] == 270
    
    rings = cats["rings"]
    assert rings["raw_images"] == 191
    assert rings["synthetic_images"] == 26
    assert rings["total_ring_images"] == 217
    assert rings["raw_singletons"] == 181
    assert rings["verified_multishot_images"] == 10
    assert rings["exact_duplicate_images"] == 0
    assert rings["quarantined_images"] == 0
    assert rings["check_sum"] == 191
    assert rings["physical_clusters"] == 186

def test_synthetic_group_isolation():
    manifest_path = Path("data/ingestion_manifest.json")
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)
        
    records = manifest["records"]
    raw_groups = {r["identity_group_id"] for r in records if r["is_original_capture"]}
    synth_groups = {r["identity_group_id"] for r in records if not r["is_original_capture"]}
    
    # Crucial rule: zero group leakage between synthetic queries and raw catalog items
    assert len(raw_groups.intersection(synth_groups)) == 0, "Synthetic groups must not overlap with physical groups"
    assert len(synth_groups) == 26
    for gid in synth_groups:
        assert gid.startswith("SYNTH-VAR-GROUP-")

def test_symmetrical_multishot_grouping():
    manifest_path = Path("data/ingestion_manifest.json")
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)
        
    records = manifest["records"]
    group_map = defaultdict(list)
    for r in records:
        group_map[r["identity_group_id"]].append(r)
        
    # Check that any multi-shot cluster has identical status across members
    for gid, members in group_map.items():
        if len(members) > 1 and not gid.startswith("SYNTH"):
            statuses = set(m["physical_identity_verification_status"] for m in members)
            assert len(statuses) == 1, f"Group {gid} has mixed statuses: {statuses}"
            assert list(statuses)[0] in ["verified_identical_sku", "uncertain_identity"]
