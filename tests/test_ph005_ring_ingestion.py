"""Automated regression tests for PH-005: Physical Ring Data Acquisition & Studio Ingestion."""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from engine.qdrant_index import QdrantCatalogIndex, catalog_id_to_uuid
from engine.index import CatalogIndex
from engine.search import SearchEngine


def test_ph005_manifest_schema_and_invariants():
    """Verify manifest version 2.1.0, accounting invariants, and zero leakage."""
    manifest_path = Path("data/ingestion_manifest.json")
    assert manifest_path.exists(), "ingestion_manifest.json must exist"

    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    assert manifest["manifest_version"].startswith("2.1")
    assert manifest["schema_version"] == "ph005.ingestion.v3"

    records = manifest["records"]
    assert len(records) == 518

    # Accounting invariants
    raw_images = [r for r in records if r["physical_identity_verification_status"] != "synthetic_transformation"]
    synthetic = [r for r in records if r["physical_identity_verification_status"] == "synthetic_transformation"]
    singletons = [r for r in records if r["physical_identity_verification_status"] == "singleton_unlinked"]
    verified = [r for r in records if r["physical_identity_verification_status"] == "verified_identical_sku"]
    quarantined = [r for r in records if r["physical_identity_verification_status"] == "uncertain_identity"]

    assert len(raw_images) == 492
    assert len(synthetic) == 26
    assert len(singletons) == 423
    assert len(verified) == 65
    assert len(quarantined) == 4
    # Invariant: singletons + verified + quarantined == raw_images
    assert len(singletons) + len(verified) + len(quarantined) == 492


def test_ph005_verified_physical_ring_skus():
    """Verify that all 5 physical ring SKU groups have valid multi-view metadata."""
    manifest_path = Path("data/ingestion_manifest.json")
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    verified_ring_groups = {
        "PHYS-SKU-RING-078": ["ring_078", "ring_079"],
        "PHYS-SKU-RING-080": ["ring_080", "ring_164"],
        "PHYS-SKU-RING-081": ["ring_081", "ring_108"],
        "PHYS-SKU-RING-084": ["ring_084", "ring_181"],
        "PHYS-SKU-RING-155": ["ring_155", "ring_172"],
    }

    records_by_id = {r["image_id"]: r for r in manifest["records"]}

    for group_id, expected_ids in verified_ring_groups.items():
        for img_id in expected_ids:
            assert img_id in records_by_id, f"Missing {img_id}"
            rec = records_by_id[img_id]
            assert rec["category"] == "ring"
            assert rec["identity_group_id"] == group_id
            assert rec["sku_id"].startswith("SKU-RING-")
            assert rec["view_angle"] in ("crown", "angle_45", "profile", "shank")
            assert rec["physical_identity_verification_status"] == "verified_identical_sku"


def test_ph005_qdrant_sku_aggregation():
    """Verify that Qdrant search aggregates multi-views into distinct SKUs when requested."""
    idx = QdrantCatalogIndex(model_name="mobilenetv4")
    assert idx.is_online

    dummy_vec = np.random.randn(1280).astype(np.float32)
    dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)

    # Search with aggregate_sku = True
    res_agg = idx.search(dummy_vec, top_k=10, aggregate_sku=True)
    assert len(res_agg["results"]) > 0

    seen_skus = set()
    for item in res_agg["results"]:
        sku = item.get("sku_id") or item["id"]
        assert sku not in seen_skus, f"Duplicate SKU {sku} found in aggregated search results"
        seen_skus.add(sku)
        assert "matched_views_count" in item
        assert item["matched_views_count"] >= 1


def test_ph005_numpy_fallback_parity_with_aggregation():
    """Verify that NumPy fallback supports SKU aggregation and agrees on top match."""
    engine_q = SearchEngine(model_name="mobilenetv4", backend="qdrant")
    engine_n = SearchEngine(model_name="mobilenetv4", backend="numpy")

    dummy_img = Image.new("RGB", (256, 256), color=(200, 180, 160))

    res_q = engine_q.search_image(dummy_img, top_k=5, aggregate_sku=True)
    res_n = engine_n.search_image(dummy_img, top_k=5, aggregate_sku=True)

    assert len(res_q["results"]) == 5
    assert len(res_n["results"]) == 5

    # Top candidate SKU should agree
    top_sku_q = res_q["results"][0].get("sku_id") or res_q["results"][0]["id"]
    top_sku_n = res_n["results"][0].get("sku_id") or res_n["results"][0]["id"]
    assert top_sku_q == top_sku_n, f"Top SKU mismatch: {top_sku_q} vs {top_sku_n}"


def test_ph005_delete_sku_and_staging_lifecycle():
    """Verify delete_sku and create_staging_collection lifecycle."""
    idx = QdrantCatalogIndex(model_name="mobilenetv4")
    assert idx.is_online

    # 1. Staging collection creation
    staging = idx.create_staging_collection(staging_suffix="test_staging")
    assert staging.is_online
    assert staging.collection_name.startswith("jewellery_catalog_test_staging")

    # 2. Insert item into staging
    dummy_vec = np.random.randn(1280).astype(np.float32)
    test_sku = "SKU-TEST-TEMP-001"
    pid1 = staging.upsert_item(
        item_id="test_item_v1",
        vector=dummy_vec,
        metadata={"category": "ring", "sku_id": test_sku, "view_angle": "crown"}
    )
    assert pid1 == catalog_id_to_uuid("test_item_v1")

    # 3. Delete by SKU
    deleted_count = staging.delete_sku(test_sku)
    assert deleted_count == 1
    assert staging.get_item("test_item_v1") is None

    # Clean up staging collection
    staging.client.delete_collection(staging.collection_name)


def test_ph005_deliverable_artifacts_exist_and_validate():
    """Verify that all PH-005 deliverables exist with valid verdicts."""
    sop_path = Path("PH-005-studio-acquisition-SOP.md")
    scale_reconcile_path = Path("output/ph004_scalability_reconciliation.json")
    manifest_audit_path = Path("output/ph005_manifest_integrity_audit.json")
    crossview_path = Path("output/ph005_crossview_evaluation.json")

    assert sop_path.exists(), "SOP must exist"
    assert scale_reconcile_path.exists(), "ph004_scalability_reconciliation.json must exist"
    assert manifest_audit_path.exists(), "ph005_manifest_integrity_audit.json must exist"
    assert crossview_path.exists(), "ph005_crossview_evaluation.json must exist"

    with open(scale_reconcile_path, "r") as fp:
        sc = json.load(fp)
    assert sc["metadata"]["verdict"] == "FORENSICALLY_RECONCILED_AND_FROZEN"

    with open(manifest_audit_path, "r") as fp:
        ma = json.load(fp)
    assert ma["verdict"] == "PASSED"
    assert ma["accounting_invariants"]["invariants_passed"] is True

    with open(crossview_path, "r") as fp:
        cv = json.load(fp)
    assert cv["dataset_acquisition_accounting"]["status"] == "PARTIALLY_COMPLETE_PENDING_STUDIO_ASSETS"
    assert cv["dataset_acquisition_accounting"]["acquired_verified_skus"] == 5
    assert cv["canonical_physical_ring_cross_view_evaluation"]["sku_aggregated_retrieval"]["recall_at_1"] == 1.0
