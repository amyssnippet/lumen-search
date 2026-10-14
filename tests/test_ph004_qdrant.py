"""Automated regression tests for PH-004: Vector Database Migration & Scalability."""

import json
from pathlib import Path
import urllib.request
import numpy as np
import pytest
from PIL import Image

from engine.qdrant_index import QdrantCatalogIndex, catalog_id_to_uuid
from engine.search import SearchEngine


def test_qdrant_server_running_and_healthy():
    """Verify that the native Qdrant server daemon is responding."""
    with urllib.request.urlopen("http://127.0.0.1:6333/", timeout=2.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "qdrant" in data.get("title", "").lower()
        assert data.get("version", "").startswith("1.19")


def test_qdrant_catalog_index_properties_and_collections():
    """Verify MobileNetV4 and DINOv3 collection specifications and HNSW properties."""
    idx = QdrantCatalogIndex(model_name="mobilenetv4")
    assert idx.is_online
    assert idx.is_indexed()
    assert idx.size == 432
    assert idx.embedding_dim == 1280

    stats = idx.stats()
    assert stats["is_ready"] is True
    assert stats["points_count"] == 432
    assert stats["distance_metric"] == "COSINE"
    assert stats["hnsw_config"]["m"] == 32
    assert stats["hnsw_config"]["ef_construct"] == 200


def test_qdrant_crud_operations():
    """Verify insert, retrieve, search, and delete lifecycle for catalog items."""
    idx = QdrantCatalogIndex(model_name="mobilenetv4")
    test_id = "test_item_sku_9999"
    dummy_vec = np.random.randn(1280).astype(np.float32)
    dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)

    metadata = {
        "category": "ring",
        "metal_color": "platinum",
        "metal_purity": "950_plat",
        "gemstone_type": "sapphire",
        "setting_type": "solitaire_prong",
        "tenant_id": "test_tenant"
    }

    # 1. Upsert
    pid = idx.upsert_item(item_id=test_id, vector=dummy_vec, metadata=metadata)
    assert pid == catalog_id_to_uuid(test_id)

    # 2. Retrieve
    item = idx.get_item(test_id)
    assert item is not None
    assert item["id"] == test_id
    assert item["payload"]["metal_color"] == "platinum"
    assert item["payload"]["setting_type"] == "solitaire_prong"

    # 3. Search with filter
    res = idx.search(
        query_embedding=dummy_vec,
        top_k=5,
        filters={"metal_color": "platinum", "setting_type": "solitaire_prong"}
    )
    assert len(res["results"]) >= 1
    assert res["results"][0]["id"] == test_id
    assert res["results"][0]["similarity"] >= 0.999

    # 4. Delete
    deleted = idx.delete_item(test_id)
    assert deleted is True

    # Confirm item no longer exists
    item_post = idx.get_item(test_id)
    assert item_post is None


def test_qdrant_metadata_filtering_and_empty_handling():
    """Verify filtered search behavior, attribute matching, and empty filter safety."""
    idx = QdrantCatalogIndex(model_name="mobilenetv4")
    dummy_vec = np.random.randn(1280).astype(np.float32)
    dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)

    # Filter: rings only
    res_rings = idx.search(dummy_vec, top_k=10, category_filter="ring")
    assert len(res_rings["results"]) > 0
    for r in res_rings["results"]:
        assert r["category"] == "ring"

    # Filter: necklaces only
    res_necks = idx.search(dummy_vec, top_k=10, category_filter="necklace")
    assert len(res_necks["results"]) > 0
    for r in res_necks["results"]:
        assert r["category"] == "necklace"

    # Filter with non-matching criteria: must return empty list without crashing
    res_empty = idx.search(dummy_vec, top_k=10, filters={"gemstone_type": "nonexistent_gem_xyz"})
    assert res_empty["results"] == []
    assert res_empty["has_reliable_match"] is False
    assert res_empty["top_similarity"] == 0.0


def test_search_engine_backend_switching():
    """Verify SearchEngine orchestrates both Qdrant and NumPy backends identically."""
    engine_qdrant = SearchEngine(model_name="mobilenetv4", backend="qdrant")
    engine_numpy = SearchEngine(model_name="mobilenetv4", backend="numpy")

    assert engine_qdrant.backend_active == "qdrant"
    assert engine_numpy.backend_active == "numpy"

    dummy_img = Image.new("RGB", (256, 256), color=(220, 200, 180))

    res_q = engine_qdrant.search_image(dummy_img, top_k=5)
    res_n = engine_numpy.search_image(dummy_img, top_k=5)

    assert "results" in res_q
    assert "results" in res_n
    assert len(res_q["results"]) == 5
    assert len(res_n["results"]) == 5

    # Check that top candidate IDs agree
    q_top_id = res_q["results"][0]["id"]
    n_top_id = res_n["results"][0]["id"]
    assert q_top_id == n_top_id, f"Top match mismatch between Qdrant ({q_top_id}) and NumPy ({n_top_id})"

    # Check similarity score precision agreement
    assert abs(res_q["results"][0]["similarity"] - res_n["results"][0]["similarity"]) < 0.01


def test_ph004_deliverable_artifacts_exist_and_validate():
    """Verify machine-readable benchmark reports for Gate 0, ANN vs Exact, and Scalability."""
    gate0_path = Path("output/ph004_gate0_reconciliation.json")
    ann_path = Path("output/ph004_ann_exact_benchmark.json")
    scale_path = Path("output/ph004_scalability_benchmark.json")

    assert gate0_path.exists(), "ph004_gate0_reconciliation.json must exist"
    assert ann_path.exists(), "ph004_ann_exact_benchmark.json must exist"
    assert scale_path.exists(), "ph004_scalability_benchmark.json must exist"

    with open(gate0_path, "r") as fp:
        g0 = json.load(fp)
    assert g0["gate_0_verdict"] == "RESOLVED_AND_FROZEN"
    assert "item_1_subgroup_denominators" in g0["reconciled_discrepancies"]

    with open(ann_path, "r") as fp:
        ann = json.load(fp)
    assert "hnsw_configuration_sweeps" in ann
    assert "metadata_filtering_benchmarks" in ann
    assert ann["primary_baseline_evaluation"]["mean_ann_recall_at_20"] >= 0.99

    with open(scale_path, "r") as fp:
        sc = json.load(fp)
    assert "measured_scalability_tiers" in sc
    assert len(sc["measured_scalability_tiers"]) == 3
    assert sc["million_vector_capacity_audit"]["status"] == "NOT_VALIDATED_ON_LOCAL_WORKSTATION_DUE_TO_RAM_LIMIT"
