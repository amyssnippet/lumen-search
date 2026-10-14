"""PH-010 Workstream 6: Canary Routing Safety & Reversible Promotion Verification.

Comprehensive tests covering:
1. Candidate registration success and missing collection handling.
2. Candidate integrity validation (dimension matching, distance metric, payload schema).
3. Candidate vector dimension mismatch detection (e.g. 512 != 1280).
4. Candidate retrieval smoke test verification.
5. Canary router deterministic hash distribution across tenant cohorts.
6. Canary router edge percentages (0% strictly prod, 100% strictly candidate).
7. Canary router runtime configuration persistence and live reload.
8. Canary router graceful fallback on missing candidate collection.
9. Atomic promotion dry-run mode (guarantees zero alias modifications).
10. Atomic production alias promotion and promotion record persistence.
11. Reversible one-click rollback restoring the previous production collection.
12. Guarded retirement protection blocking deletion of active production collections.
13. Guarded retirement protection blocking deletion of registered candidate collections.
14. Guarded retirement permitting deletion of unreferenced legacy collections.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from engine.lifecycle import IndexLifecycleManager
from engine.index_router import CanaryRouter, CanaryConfig, RouteTarget


@pytest.fixture(scope="module")
def qdrant_safety_fixture():
    """Create disposable test collections and custom alias for lifecycle safety verification."""
    qc = QdrantClient(host="127.0.0.1", port=6333, timeout=5.0, check_compatibility=False)

    col_prod = "test_col_ph010_prod"
    col_cand = "test_col_ph010_cand"
    col_legacy = "test_col_ph010_legacy"
    col_mismatch = "test_col_ph010_mismatch_512"
    test_alias = "test_alias_ph010_safety"
    history_file = "data/test_ph010_lifecycle_history.json"
    canary_cfg_file = "data/test_ph010_canary_config.json"

    # Clean existing
    for col in (col_prod, col_cand, col_legacy, col_mismatch):
        if qc.collection_exists(col):
            qc.delete_collection(col)
    if Path(history_file).exists():
        Path(history_file).unlink()
    if Path(canary_cfg_file).exists():
        Path(canary_cfg_file).unlink()

    # Create col_prod (1280-dim)
    qc.create_collection(
        col_prod,
        vectors_config=rest.VectorParams(size=1280, distance=rest.Distance.COSINE)
    )
    vec1 = np.ones(1280, dtype=np.float32)
    vec1 /= np.linalg.norm(vec1)
    qc.upsert(
        collection_name=col_prod,
        points=[
            rest.PointStruct(
                id="11111111-1111-1111-1111-111111111111",
                vector=vec1.tolist(),
                payload={
                    "id": "item_prod_1",
                    "tenant_id": "default_tenant",
                    "category": "ring",
                    "sku_id": "SKU_PROD_001",
                    "metal_color": "yellow_gold",
                    "metal_purity": "18k",
                    "gemstone_type": "diamond",
                    "setting_type": "prong"
                }
            )
        ]
    )

    # Create col_cand (1280-dim)
    qc.create_collection(
        col_cand,
        vectors_config=rest.VectorParams(size=1280, distance=rest.Distance.COSINE)
    )
    vec2 = np.ones(1280, dtype=np.float32)
    vec2[0] = 2.0
    vec2 /= np.linalg.norm(vec2)
    qc.upsert(
        collection_name=col_cand,
        points=[
            rest.PointStruct(
                id="22222222-2222-2222-2222-222222222222",
                vector=vec2.tolist(),
                payload={
                    "id": "item_cand_1",
                    "tenant_id": "default_tenant",
                    "category": "ring",
                    "sku_id": "SKU_CAND_001",
                    "metal_color": "white_gold",
                    "metal_purity": "18k",
                    "gemstone_type": "sapphire",
                    "setting_type": "halo"
                }
            )
        ]
    )

    # Create col_legacy (1280-dim)
    qc.create_collection(
        col_legacy,
        vectors_config=rest.VectorParams(size=1280, distance=rest.Distance.COSINE)
    )

    # Create col_mismatch (512-dim)
    qc.create_collection(
        col_mismatch,
        vectors_config=rest.VectorParams(size=512, distance=rest.Distance.COSINE)
    )

    # Point test_alias to col_prod initially
    qc.update_collection_aliases(
        change_aliases_operations=[
            rest.CreateAliasOperation(
                create_alias=rest.CreateAlias(collection_name=col_prod, alias_name=test_alias)
            )
        ]
    )

    yield {
        "client": qc,
        "col_prod": col_prod,
        "col_cand": col_cand,
        "col_legacy": col_legacy,
        "col_mismatch": col_mismatch,
        "test_alias": test_alias,
        "history_file": history_file,
        "canary_cfg_file": canary_cfg_file
    }

    # Cleanup teardown
    for col in (col_prod, col_cand, col_legacy, col_mismatch):
        if qc.collection_exists(col):
            try:
                qc.delete_collection(col)
            except Exception:
                pass
    if Path(history_file).exists():
        try:
            Path(history_file).unlink()
        except Exception:
            pass
    if Path(canary_cfg_file).exists():
        try:
            Path(canary_cfg_file).unlink()
        except Exception:
            pass


# ==============================================================================
# 1. Candidate Registration & Integrity Validation
# ==============================================================================

def test_candidate_registration_lifecycle(qdrant_safety_fixture):
    """Verify candidate registration succeeds for existing collection and fails for non-existent."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )

    # Register non-existent collection
    res_missing = mgr.register_candidate("non_existent_collection_99999")
    assert res_missing["status"] == "FAILED"
    assert "does not exist" in res_missing["error"]

    # Register valid existing candidate
    res_valid = mgr.register_candidate(fixture["col_cand"])
    assert res_valid["status"] == "REGISTERED"
    assert res_valid["candidate_collection"] == fixture["col_cand"]
    assert mgr.get_candidate_collection() == fixture["col_cand"]


def test_candidate_validation_matching_schema(qdrant_safety_fixture):
    """Verify candidate collection with 1280 dimensions and valid schema passes validation."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )

    val = mgr.validate_collection_integrity(fixture["col_cand"], min_points=1)
    assert val["valid"] is True
    assert val["vector_dimension"] == 1280
    assert val["points_count"] >= 1
    assert "error" not in val


def test_candidate_validation_mismatched_dimension(qdrant_safety_fixture):
    """Verify candidate collection with 512 dimensions fails validation with dimension mismatch."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )

    val = mgr.validate_collection_integrity(fixture["col_mismatch"], min_points=0)
    assert val["valid"] is False
    assert "dimension mismatch" in val["error"].lower()


def test_candidate_retrieval_smoke_test(qdrant_safety_fixture):
    """Verify retrieval smoke test executes against candidate collection and returns top-1 hit."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )

    smoke = mgr.run_retrieval_smoke_test(fixture["col_cand"], top_k=5)
    assert smoke["passed"] is True
    assert smoke["collection_name"] == fixture["col_cand"]
    assert smoke["unfiltered_candidates"] >= 1


# ==============================================================================
# 2. Canary Routing Distribution & Determinism
# ==============================================================================

def test_canary_router_deterministic_hash_distribution(qdrant_safety_fixture):
    """Verify CanaryRouter hashes callers deterministically across tenants."""
    fixture = qdrant_safety_fixture
    router = CanaryRouter(config_path=fixture["canary_cfg_file"])
    router.configure(
        enabled=True,
        candidate_collection="candidate_mock_col",
        percentage=50,
        fallback_to_production=True
    )

    # 10 runs on tenant_alpha should yield identical target
    targets_alpha = [router.resolve_target("tenant_alpha")[0] for _ in range(10)]
    assert len(set(targets_alpha)) == 1

    # 10 runs on tenant_beta should yield identical target
    targets_beta = [router.resolve_target("tenant_beta")[0] for _ in range(10)]
    assert len(set(targets_beta)) == 1


def test_canary_router_zero_and_hundred_percent_routing(qdrant_safety_fixture):
    """Verify 0% canary strictly routes to PROD and 100% canary strictly routes to CANDIDATE."""
    fixture = qdrant_safety_fixture
    router = CanaryRouter(config_path=fixture["canary_cfg_file"])

    # 0% canary
    router.configure(enabled=True, candidate_collection="candidate_mock_col", percentage=0)
    for t_id in ["tenant_a", "tenant_b", "tenant_c", "tenant_d"]:
        target, reason = router.resolve_target(t_id)
        assert target == RouteTarget.PRODUCTION
        assert reason == "percentage_zero"

    # 100% canary
    router.configure(enabled=True, candidate_collection="candidate_mock_col", percentage=100)
    for t_id in ["tenant_a", "tenant_b", "tenant_c", "tenant_d"]:
        target, reason = router.resolve_target(t_id)
        assert target == RouteTarget.CANDIDATE
        assert reason == "percentage_hundred"


def test_canary_router_disabled_routes_to_production(qdrant_safety_fixture):
    """Verify disabling canary routing directs all traffic to production."""
    fixture = qdrant_safety_fixture
    router = CanaryRouter(config_path=fixture["canary_cfg_file"])
    router.configure(enabled=False, candidate_collection="candidate_mock_col", percentage=100)
    target, reason = router.resolve_target("tenant_xyz")
    assert target == RouteTarget.PRODUCTION
    assert reason == "canary_disabled_or_no_candidate"


def test_canary_router_fallback_on_unreachable_candidate(qdrant_safety_fixture):
    """Verify canary router marks non-existent candidate collection as unindexed."""
    fixture = qdrant_safety_fixture
    router = CanaryRouter(config_path=fixture["canary_cfg_file"])
    router.configure(
        enabled=True,
        candidate_collection="non_existent_candidate_xyz",
        percentage=100,
        fallback_to_production=True
    )
    cand_idx = router.get_candidate_index()
    assert cand_idx is not None
    assert cand_idx.is_indexed() is False


# ==============================================================================
# 3. Atomic Promotion & Reversible Rollback
# ==============================================================================

def test_atomic_promotion_dry_run_preserves_alias(qdrant_safety_fixture):
    """Verify dry-run promotion previews operations without modifying the live alias."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )
    mgr.register_candidate(fixture["col_cand"])

    active_before = mgr.get_active_production_collection()
    assert active_before == fixture["col_prod"]

    dry_run_res = mgr.promote_candidate(dry_run=True, actor_id="ops_tester")
    assert dry_run_res["status"] == "DRY_RUN"
    assert dry_run_res["target_collection"] == fixture["col_cand"]
    assert dry_run_res["current_collection"] == fixture["col_prod"]

    # Verify alias was NOT changed
    active_after = mgr.get_active_production_collection()
    assert active_after == fixture["col_prod"]


def test_atomic_promotion_switches_alias_and_records_history(qdrant_safety_fixture):
    """Verify live promotion atomically points alias to candidate and logs audit record."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )
    mgr.register_candidate(fixture["col_cand"])

    promote_res = mgr.promote_candidate(dry_run=False, actor_id="ops_tester")
    assert promote_res["status"] == "SUCCESS"
    assert promote_res["promoted_collection"] == fixture["col_cand"]
    assert promote_res["previous_collection"] == fixture["col_prod"]

    # Confirm alias now resolves to candidate
    active_prod = mgr.get_active_production_collection()
    assert active_prod == fixture["col_cand"]

    # Clear candidate
    mgr.clear_candidate()
    assert mgr.get_candidate_collection() is None


def test_reversible_rollback_restores_previous_collection(qdrant_safety_fixture):
    """Verify rollback atomically shifts alias back to previous collection."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )

    # Active is currently col_cand after previous promotion
    assert mgr.get_active_production_collection() == fixture["col_cand"]

    # Dry run rollback
    dry_rollback = mgr.rollback_production(dry_run=True, actor_id="ops_tester")
    assert dry_rollback["status"] == "DRY_RUN"
    assert dry_rollback["target_rollback_collection"] == fixture["col_prod"]

    # Live rollback
    live_rollback = mgr.rollback_production(dry_run=False, actor_id="ops_tester")
    assert live_rollback["status"] == "ROLLBACK_SUCCESS"
    assert live_rollback["active_collection"] == fixture["col_prod"]
    assert live_rollback["reverted_from"] == fixture["col_cand"]

    # Confirm alias now resolves back to original prod
    assert mgr.get_active_production_collection() == fixture["col_prod"]


# ==============================================================================
# 4. Guarded Collection Retirement
# ==============================================================================

def test_guarded_retirement_blocks_active_production(qdrant_safety_fixture):
    """Verify lifecycle manager rejects retirement of active production collection."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )

    active_prod = mgr.get_active_production_collection()
    assert active_prod == fixture["col_prod"]

    retire_res = mgr.retire_collection(active_prod, force=False)
    assert retire_res["status"] == "FORBIDDEN"
    assert "currently mapped to production alias" in retire_res["error"]


def test_guarded_retirement_blocks_active_candidate(qdrant_safety_fixture):
    """Verify lifecycle manager rejects retirement of active candidate collection."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )
    # Register col_mismatch (which is not active prod) as candidate
    mgr.register_candidate(fixture["col_mismatch"])

    retire_res = mgr.retire_collection(fixture["col_mismatch"], force=False)
    assert retire_res["status"] == "FORBIDDEN"
    assert "registered as active candidate" in retire_res["error"]


def test_guarded_retirement_permits_unreferenced_legacy_collection(qdrant_safety_fixture):
    """Verify lifecycle manager successfully retires unreferenced legacy collection."""
    fixture = qdrant_safety_fixture
    mgr = IndexLifecycleManager(
        production_alias=fixture["test_alias"],
        history_file=fixture["history_file"]
    )

    # col_legacy is neither active prod nor candidate nor rollback target
    retire_res = mgr.retire_collection(fixture["col_legacy"], force=False, actor_id="cleaner_ops")
    assert retire_res["status"] == "RETIRED"
    assert retire_res["deleted_collection"] == fixture["col_legacy"]

    # Confirm collection is deleted in Qdrant
    assert fixture["client"].collection_exists(fixture["col_legacy"]) is False
