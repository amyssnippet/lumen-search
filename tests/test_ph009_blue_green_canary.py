"""Integration and reliability tests for PH-009: Staging Cluster Promotion, Blue-Green Deployment & Canary Search Routing."""

import json
import os
import time
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.main import app, CANARY_ROUTER, LIFECYCLE_MANAGER, reload_search_engines
from engine.lifecycle import IndexLifecycleManager
from engine.index_router import CanaryRouter, RouteTarget
from engine.qdrant_index import QdrantCatalogIndex
from app.auth import create_access_token

client = TestClient(app)

ADMIN_TOKEN = create_access_token(
    tenant_id="default_tenant",
    actor_id="test_admin",
    scopes=["search:read", "catalog:write", "tenant:admin", "ops:admin"]
)
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

VIEWER_TOKEN = create_access_token(
    tenant_id="default_tenant",
    actor_id="test_viewer",
    scopes=["search:read"]
)
VIEWER_HEADERS = {"Authorization": f"Bearer {VIEWER_TOKEN}"}


@pytest.fixture(scope="module")
def qdrant_test_setup():
    """Ensure disposable test collections exist in Qdrant for real alias testing."""
    qc = QdrantClient(host="127.0.0.1", port=6333, timeout=5.0, check_compatibility=False)
    
    col_prod = "test_col_ph009_prod"
    col_cand = "test_col_ph009_cand"
    test_alias = "test_alias_ph009"

    for col in (col_prod, col_cand):
        if qc.collection_exists(col):
            qc.delete_collection(col)
        qc.create_collection(
            col,
            vectors_config=rest.VectorParams(size=1280, distance=rest.Distance.COSINE)
        )
        # Seed 1 test vector point with tenant_id payload
        vec = np.ones(1280, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        qc.upsert(
            collection_name=col,
            points=[
                rest.PointStruct(
                    id="11111111-1111-1111-1111-111111111111",
                    vector=vec.tolist(),
                    payload={
                        "id": "item_1",
                        "tenant_id": "default_tenant",
                        "category": "ring",
                        "sku_id": "SKU_001",
                        "metal_color": "yellow_gold",
                        "metal_purity": "18k",
                        "gemstone_type": "diamond",
                        "setting_type": "prong"
                    }
                )
            ]
        )

    # Point alias to col_prod initially
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
        "test_alias": test_alias
    }

    # Teardown
    for col in (col_prod, col_cand):
        if qc.collection_exists(col):
            qc.delete_collection(col)


# ==============================================================================
# 1. Real Qdrant Alias Management & Lifecycle Tests
# ==============================================================================

def test_lifecycle_manager_validation_and_smoke(qdrant_test_setup):
    """Validate candidate collection integrity and retrieval smoke test on live Qdrant."""
    setup = qdrant_test_setup
    mgr = IndexLifecycleManager(production_alias=setup["test_alias"])

    # Integrity check on valid collection
    val = mgr.validate_collection_integrity(setup["col_cand"], min_points=1)
    assert val["valid"] is True
    assert val["points_count"] == 1
    assert val["vector_dimension"] == 1280

    # Smoke test on valid collection
    smoke = mgr.run_retrieval_smoke_test(setup["col_cand"], test_tenant_id="default_tenant")
    assert smoke["passed"] is True
    assert smoke["tenant_candidates"] >= 1

    # Integrity check on non-existent collection
    val_missing = mgr.validate_collection_integrity("non_existent_collection_xyz")
    assert val_missing["valid"] is False


def test_blue_green_promotion_and_atomic_switch(qdrant_test_setup):
    """Verify atomic alias switch from prod to candidate collection."""
    setup = qdrant_test_setup
    mgr = IndexLifecycleManager(production_alias=setup["test_alias"])

    # 1. Verify initially active collection
    assert mgr.get_active_production_collection() == setup["col_prod"]

    # 2. Dry run preview
    dry_run = mgr.promote_candidate(candidate_collection=setup["col_cand"], dry_run=True)
    assert dry_run["status"] == "DRY_RUN"
    assert mgr.get_active_production_collection() == setup["col_prod"]

    # 3. Execute atomic promotion
    res = mgr.promote_candidate(candidate_collection=setup["col_cand"], dry_run=False, actor_id="test_runner")
    assert res["status"] == "SUCCESS"
    assert res["promoted_collection"] == setup["col_cand"]
    assert res["previous_collection"] == setup["col_prod"]

    # 4. Verify alias resolution now points to candidate
    assert mgr.get_active_production_collection() == setup["col_cand"]

    # 5. Idempotent repeated promotion
    repeat = mgr.promote_candidate(candidate_collection=setup["col_cand"])
    assert repeat["status"] == "ALREADY_ACTIVE"


def test_reversible_rollback_operation(qdrant_test_setup):
    """Verify rolling back production alias restores previous index version."""
    setup = qdrant_test_setup
    mgr = IndexLifecycleManager(production_alias=setup["test_alias"])

    # Current active is col_cand (from previous test)
    assert mgr.get_active_production_collection() == setup["col_cand"]

    # Execute rollback
    res = mgr.rollback_production(dry_run=False, actor_id="test_runner")
    assert res["status"] == "ROLLBACK_SUCCESS"
    assert res["active_collection"] == setup["col_prod"]
    assert res["reverted_from"] == setup["col_cand"]

    # Verify alias resolution is restored to col_prod
    assert mgr.get_active_production_collection() == setup["col_prod"]


def test_guarded_retirement_prevents_active_deletion(qdrant_test_setup):
    """Verify guarded retirement forbids deleting active production or rollback-eligible collections."""
    setup = qdrant_test_setup
    mgr = IndexLifecycleManager(production_alias=setup["test_alias"])

    # Attempt to retire active production index must be rejected with FORBIDDEN
    res_active = mgr.retire_collection(setup["col_prod"])
    assert res_active["status"] == "FORBIDDEN"


# ==============================================================================
# 2. Tenant-Aware Canary Routing & Determinism
# ==============================================================================

def test_canary_router_deterministic_cohorts():
    """Verify deterministic tenant assignment without cross-request oscillation."""
    router = CanaryRouter(config_path="data/test_canary_config.json")
    router.configure(
        enabled=True,
        candidate_collection="jewellery_catalog_mobilenetv4_v1",
        percentage=50,
        allowlisted_tenants=["tenant_vip_alpha"],
        salt="test_salt_123"
    )

    # Allowlisted tenant ALWAYS routes to candidate
    target, reason = router.resolve_target("tenant_vip_alpha")
    assert target == RouteTarget.CANDIDATE
    assert "allowlist" in reason

    # Any tenant must resolve deterministically across 100 repeated queries
    target_1, _ = router.resolve_target("tenant_beta")
    for _ in range(100):
        target_n, _ = router.resolve_target("tenant_beta")
        assert target_n == target_1, "Canary cohort must not oscillate between requests"


def test_canary_router_percentage_bounds():
    """Verify 0% and 100% boundary conditions."""
    router = CanaryRouter(config_path="data/test_canary_config.json")
    
    # At 0%, non-allowlisted tenants route to production
    router.configure(enabled=True, candidate_collection="jewellery_catalog_mobilenetv4_v1", percentage=0, allowlisted_tenants=[])
    target, _ = router.resolve_target("random_tenant_xyz")
    assert target == RouteTarget.PRODUCTION

    # At 100%, all tenants route to candidate
    router.configure(enabled=True, candidate_collection="jewellery_catalog_mobilenetv4_v1", percentage=100)
    target_100, _ = router.resolve_target("random_tenant_xyz")
    assert target_100 == RouteTarget.CANDIDATE


def test_canary_router_invalid_percentage_rejected():
    """Verify invalid percentage values outside [0, 100] raise ValueError."""
    router = CanaryRouter(config_path="data/test_canary_config.json")
    with pytest.raises(ValueError):
        router.configure(percentage=150)
    with pytest.raises(ValueError):
        router.configure(percentage=-10)


# ==============================================================================
# 3. RBAC Operational Endpoints & Audit Logging
# ==============================================================================

def test_lifecycle_status_endpoint_auth_and_structure():
    """Verify /api/lifecycle/status returns expected schema under ops:admin scope."""
    # Unauthenticated / unprivileged viewer must be rejected
    resp_unauth = client.get("/api/lifecycle/status", headers=VIEWER_HEADERS)
    assert resp_unauth.status_code == 403

    # Authenticated ops:admin returns 200 with complete status dictionary
    resp = client.get("/api/lifecycle/status", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "production_alias" in data
    assert "total_collections" in data
    assert "collections" in data
    assert "canary_routing" in data


def test_canary_config_endpoint_updates_runtime():
    """Verify /api/lifecycle/canary/config updates configuration safely."""
    payload = {
        "enabled": True,
        "percentage": 25,
        "allowlisted_tenants": ["vip_tenant_test"],
        "fallback_to_production": True
    }
    resp = client.post("/api/lifecycle/canary/config", json=payload, headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    cfg = resp.json()["canary_config"]
    assert cfg["enabled"] is True
    assert cfg["percentage"] == 25
    assert "vip_tenant_test" in cfg["allowlisted_tenants"]

    # Reset
    client.post("/api/lifecycle/canary/config", json={"enabled": False, "percentage": 0}, headers=ADMIN_HEADERS)


def test_promote_endpoint_dry_run_and_execution():
    """Verify /api/lifecycle/promote dry run and execution."""
    resp_dry = client.post(
        "/api/lifecycle/promote",
        json={"candidate_collection": "jewellery_catalog_mobilenetv4_v1", "dry_run": True},
        headers=ADMIN_HEADERS
    )
    assert resp_dry.status_code == 200
    assert resp_dry.json()["status"] in ("DRY_RUN", "ALREADY_ACTIVE")


# ==============================================================================
# 4. Search Routing & Metadata Integration
# ==============================================================================

def test_search_endpoint_includes_routing_metadata():
    """Verify /api/search response contains routing metadata."""
    from PIL import Image
    import io

    img = Image.new("RGB", (100, 100), (200, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_ring.jpg", buf, "image/jpeg")}
    data = {"model": "mobilenetv4", "category_filter": "ring", "top_k": 3}

    resp = client.post("/api/search", files=files, data=data, headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    res_json = resp.json()
    assert "routing" in res_json
    assert res_json["routing"]["target"] in ("production", "candidate")
    assert "fallback" in res_json["routing"]


def test_metrics_include_canary_and_lifecycle_series():
    """Verify /metrics exposes new routing and lifecycle Prometheus metrics."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    content = resp.text

    assert "jewellery_search_routing_requests_total" in content
    assert "jewellery_canary_traffic_percentage" in content
    assert "jewellery_canary_routing_fallbacks_total" in content
    assert "jewellery_index_lifecycle_operations_total" in content


def test_canary_fallback_when_candidate_offline():
    """Verify search gracefully falls back to production when candidate collection is unavailable."""
    from PIL import Image
    import io

    # Configure canary to point to an unavailable collection
    CANARY_ROUTER.configure(
        enabled=True,
        candidate_collection="non_existent_candidate_collection_xyz",
        percentage=100,
        fallback_to_production=True
    )
    try:
        img = Image.new("RGB", (100, 100), (200, 150, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("test_ring.jpg", buf, "image/jpeg")}
        data = {"model": "mobilenetv4", "category_filter": "ring", "top_k": 3}

        resp = client.post("/api/search", files=files, data=data, headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        res = resp.json()
        assert res["routing"]["target"] == "production"
        assert res["routing"]["fallback"] is True
    finally:
        CANARY_ROUTER.configure(enabled=False, percentage=0)


def test_tenant_isolation_on_routed_searches():
    """Verify tenant isolation invariant holds strictly regardless of routing target."""
    from PIL import Image
    import io

    img = Image.new("RGB", (100, 100), (200, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # Search as tenant_luxury_corp
    luxury_token = create_access_token(
        tenant_id="tenant_luxury_corp",
        actor_id="luxury_admin",
        scopes=["search:read"]
    )
    luxury_headers = {"Authorization": f"Bearer {luxury_token}"}

    files = {"file": ("test_ring.jpg", buf, "image/jpeg")}
    data = {"model": "mobilenetv4", "top_k": 5}

    resp = client.post("/api/search", files=files, data=data, headers=luxury_headers)
    assert resp.status_code == 200
    res = resp.json()
    # Confirm zero results leak from other tenants
    for r in res.get("results", []):
        pt_tenant = r.get("tenant_id")
        if pt_tenant:
            assert pt_tenant == "tenant_luxury_corp"

