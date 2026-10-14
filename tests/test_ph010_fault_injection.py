"""Automated bounded fault-injection and chaos resilience test suite for PH-010.

Tests resilience and graceful degradation under:
1. Redis connection drops, timeouts, fail-secure in-process fallback, throttling, and recovery.
2. Qdrant candidate collection absence, timeouts, connection dropouts, canary fallback, and tenant preservation.
3. Persistent vector index failures ensuring errors are not silently masked.
4. PostgreSQL connection failure, missing model weights, secret masking, and readiness probe recovery.
"""

import io
import time
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_db
from app.rate_limiter import DistributedRateLimiter, InMemoryTokenBucket
from engine.index_router import CanaryRouter, CanaryConfig
from engine.search import SearchEngine
from engine.qdrant_index import QdrantCatalogIndex


client = TestClient(app)


# ==============================================================================
# Workstream 2.1 — Redis Failures & Fail-Secure Fallback
# ==============================================================================

def test_redis_connection_loss_engages_fail_secure_in_memory_limiter():
    """Verify that Redis connection failure engages in-process rate limiting with a 50% burst budget."""
    limiter = DistributedRateLimiter(redis_url="redis://127.0.0.1:65432/0", timeout=0.1)
    
    # First check should detect Redis offline, engage in-process limiter
    allowed, remaining, retry_after = limiter.check_rate_limit(
        tenant_id="chaos_tenant",
        actor_id="actor_1",
        rate_limit_rpm=60,
        burst_capacity=10,
        fail_secure_budget_ratio=0.5
    )
    
    assert allowed is True
    # Burst capacity 10 * 0.5 = 5 max tokens; 1 consumed -> remaining <= 4
    assert remaining <= 4
    assert limiter._redis_available is False


def test_redis_fail_secure_throttles_under_burst_exhaustion():
    """Verify that during Redis outage, in-memory limiter strictly throttles after exhausting the 50% budget."""
    limiter = DistributedRateLimiter(redis_url="redis://127.0.0.1:65432/0", timeout=0.1)
    
    tenant_id = f"chaos_throttle_{uuid.uuid4().hex[:6]}"
    actor_id = "actor_burst"
    burst_cap = 6  # 50% budget = 3 tokens
    
    consumed = 0
    for _ in range(3):
        allowed, remaining, _ = limiter.check_rate_limit(
            tenant_id=tenant_id,
            actor_id=actor_id,
            rate_limit_rpm=60,
            burst_capacity=burst_cap,
            fail_secure_budget_ratio=0.5
        )
        if allowed:
            consumed += 1
            
    assert consumed == 3
    
    # 4th request must be rejected
    allowed, remaining, retry_after = limiter.check_rate_limit(
        tenant_id=tenant_id,
        actor_id=actor_id,
        rate_limit_rpm=60,
        burst_capacity=burst_cap,
        fail_secure_budget_ratio=0.5
    )
    assert allowed is False
    assert retry_after >= 1


def test_redis_recovery_restores_distributed_rate_limiting():
    """Verify that restoring Redis connectivity re-engages distributed rate limiting."""
    limiter = DistributedRateLimiter(redis_url="redis://127.0.0.1:65432/0", timeout=0.1)
    limiter._redis_available = False
    limiter._redis_checked_at = time.time() - 15.0  # Expire 10s cooldown
    
    # Mock working Redis client on next retry
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    # Lua script return: {allowed=1, remaining=29, retry_after=0}
    mock_redis.eval.return_value = [1, 29, 0]
    
    with patch("redis.Redis.from_url", return_value=mock_redis):
        allowed, remaining, retry_after = limiter.check_rate_limit(
            tenant_id="recovery_tenant",
            actor_id="actor_rec",
            rate_limit_rpm=300,
            burst_capacity=30
        )
        assert allowed is True
        assert remaining == 29
        assert limiter._redis_available is True


def test_in_memory_fallback_process_isolation_boundary():
    """Verify that in-memory fallback state is memory-local and does NOT falsely claim cross-worker sync."""
    bucket_a = InMemoryTokenBucket()
    bucket_b = InMemoryTokenBucket()
    
    key = "shared_test_key"
    allowed_a, rem_a, _ = bucket_a.consume(key=key, rate_per_sec=1.0, burst_capacity=2, tokens_requested=2.0)
    assert allowed_a is True
    assert rem_a == 0.0
    
    # Bucket B in a separate worker/memory space must be unaware of Bucket A's state
    allowed_b, rem_b, _ = bucket_b.consume(key=key, rate_per_sec=1.0, burst_capacity=2, tokens_requested=1.0)
    assert allowed_b is True
    assert rem_b >= 0.9


# ==============================================================================
# Workstream 2.2 — Qdrant Failures, Canary Fallback & Tenant Preservation
# ==============================================================================

def test_canary_routing_fallback_when_candidate_collection_missing():
    """Verify that when a candidate collection is missing, search transparently falls back to production."""
    canary_router = CanaryRouter(config_path="data/test_chaos_canary.json")
    canary_router.configure(
        enabled=True,
        candidate_collection="non_existent_collection_chaos_404",
        percentage=100,  # Force all traffic to candidate
        fallback_to_production=True
    )
    
    engine = SearchEngine(
        model_name="mobilenetv4",
        backend="qdrant",
        canary_router=canary_router
    )
    
    dummy_img = Image.new("RGB", (100, 100), color=(180, 120, 60))
    res = engine.search_image(
        image=dummy_img,
        top_k=3,
        tenant_id="chaos_tenant"
    )
    
    assert "results" in res
    assert "routing" in res
    assert res["routing"]["target"] == "production"
    assert res["routing"]["fallback"] is True
    assert "candidate_index_unavailable" in res["routing"]["fallback_reason"]


def test_canary_routing_fallback_preserves_tenant_filter():
    """Verify that tenant isolation filters are strictly preserved when canary queries fall back to production."""
    canary_router = CanaryRouter(config_path="data/test_chaos_canary.json")
    canary_router.configure(
        enabled=True,
        candidate_collection="jewellery_catalog_mobilenetv4_candidate_eval",
        percentage=100,
        fallback_to_production=True
    )
    
    engine = SearchEngine(
        model_name="mobilenetv4",
        backend="qdrant",
        canary_router=canary_router
    )
    
    # Mock candidate search raising a transient connection error
    mock_candidate_idx = MagicMock()
    mock_candidate_idx.is_online = True
    mock_candidate_idx.is_indexed.return_value = True
    mock_candidate_idx.collection_name = "jewellery_catalog_mobilenetv4_candidate_eval"
    mock_candidate_idx.search.side_effect = ConnectionError("Transient Qdrant timeout on candidate node")
    
    # Spy on production index search to assert tenant_id is preserved
    original_prod_search = engine.index.search
    prod_search_spy = MagicMock(side_effect=original_prod_search)
    engine.index.search = prod_search_spy
    
    dummy_img = Image.new("RGB", (100, 100), color=(200, 150, 80))
    
    with patch.object(canary_router, "get_candidate_index", return_value=mock_candidate_idx):
        res = engine.search_image(
            image=dummy_img,
            top_k=5,
            tenant_id="strict_tenant_alpha"
        )
        
        # Verify fallback occurred
        assert res["routing"]["fallback"] is True
        assert "Transient Qdrant timeout" in res["routing"]["fallback_reason"]
        
        # Verify production search was called with the exact tenant_id
        prod_search_spy.assert_called_once()
        _, call_kwargs = prod_search_spy.call_args
        assert call_kwargs.get("tenant_id") == "strict_tenant_alpha"


def test_persistent_qdrant_failure_is_not_masked():
    """Verify that persistent index failure (both candidate and production offline) raises HTTP error instead of fake 200."""
    canary_router = CanaryRouter(config_path="data/test_chaos_canary.json")
    canary_router.configure(
        enabled=True,
        candidate_collection="jewellery_catalog_mobilenetv4_candidate_eval",
        percentage=100,
        fallback_to_production=True
    )
    
    engine = SearchEngine(
        model_name="mobilenetv4",
        backend="qdrant",
        canary_router=canary_router
    )
    
    # Simulate both candidate and production failing
    mock_candidate_idx = MagicMock()
    mock_candidate_idx.is_online = True
    mock_candidate_idx.is_indexed.return_value = True
    mock_candidate_idx.collection_name = "jewellery_catalog_mobilenetv4_candidate_eval"
    mock_candidate_idx.search.side_effect = ConnectionError("Candidate node offline")
    
    mock_prod_idx = MagicMock()
    mock_prod_idx.search.side_effect = ConnectionError("Production cluster unreachable")
    engine.index = mock_prod_idx
    
    dummy_img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    
    with patch.object(canary_router, "get_candidate_index", return_value=mock_candidate_idx):
        with pytest.raises(ConnectionError) as exc_info:
            engine.search_image(image=dummy_img, top_k=5, tenant_id="test_tenant")
        assert "Production cluster unreachable" in str(exc_info.value)



# ==============================================================================
# Workstream 2.3 — Database & Model Weights Failure Injection on Probes
# ==============================================================================

def test_readiness_probe_database_disconnect_masks_credentials():
    """Verify /health/ready returns 503 on database failure without leaking connection strings."""
    def broken_db():
        session = MagicMock()
        session.execute.side_effect = Exception("connection to server at 'db.internal.jewellery.corp:5432', user 'admin_vault', password 'secret_pass_123' failed")
        yield session

    app.dependency_overrides[get_db] = broken_db
    try:
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["dependencies"]["database"]["status"] == "down"
        
        # Verify credentials/passwords are NOT leaked in response
        err_msg = data["dependencies"]["database"]["error"]
        assert "secret_pass_123" not in err_msg
        assert "password" not in err_msg.lower()
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_readiness_probe_missing_model_weights_returns_503():
    """Verify /health/ready returns 503 when required ONNX model weights are missing on disk."""
    with patch("pathlib.Path.exists", return_value=False):
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["dependencies"]["model"]["status"] == "missing"


def test_readiness_probe_recovery_when_dependencies_return():
    """Verify /health/ready returns 200 OK once dependencies are restored."""
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ready", "degraded")
    assert data["dependencies"]["database"]["status"] == "connected"
    assert data["dependencies"]["qdrant"]["status"] == "connected"
    assert data["dependencies"]["model"]["status"] == "available"

