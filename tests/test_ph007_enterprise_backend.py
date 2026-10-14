"""Comprehensive test suite for PH-007: Enterprise Backend, Multi-Tenancy & Observability."""

import datetime
import hashlib
import io
import json
import uuid
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.auth import create_access_token, hash_api_key, generate_new_api_key
from app.rate_limiter import RATE_LIMITER, InMemoryTokenBucket
from engine.db.session import SessionLocal
from engine.db.models import Tenant, User, ApiKey, Product, Asset, VectorPoint, IngestionJob, AuditLog
from engine.qdrant_index import QdrantCatalogIndex, catalog_id_to_uuid

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_tenants_and_keys():
    """Seed test fixtures for multi-tenant isolation and security testing."""
    db = SessionLocal()
    try:
        # 1. Tenant Alpha (Luxury Corp)
        t_alpha = db.query(Tenant).filter(Tenant.id == "test_tenant_alpha").first()
        if not t_alpha:
            t_alpha = Tenant(
                id="test_tenant_alpha",
                name="Tenant Alpha Corp",
                tier="enterprise",
                rate_limit_rpm=1200,
                rate_limit_burst=20,
                is_active=True
            )
            db.add(t_alpha)
            db.flush()

        # 2. Tenant Beta (Artisan Gems)
        t_beta = db.query(Tenant).filter(Tenant.id == "test_tenant_beta").first()
        if not t_beta:
            t_beta = Tenant(
                id="test_tenant_beta",
                name="Tenant Beta Gems",
                tier="free",
                rate_limit_rpm=60,
                rate_limit_burst=5,
                is_active=True
            )
            db.add(t_beta)
            db.flush()

        # 3. Keys for Alpha
        key_alpha_raw = "jvr_alpha_full_access_test_key_12345"
        k_alpha = db.query(ApiKey).filter(ApiKey.key_hash == hash_api_key(key_alpha_raw)).first()
        if not k_alpha:
            k_alpha = ApiKey(
                id=str(uuid.uuid4()),
                tenant_id="test_tenant_alpha",
                key_hash=hash_api_key(key_alpha_raw),
                key_prefix=key_alpha_raw[:12],
                name="Alpha Admin Key",
                scopes="search:read,catalog:write,tenant:admin,ops:admin",
                is_active=True
            )
            db.add(k_alpha)

        # Alpha Read-Only Key (lacks tenant:admin)
        key_alpha_ro = "jvr_alpha_readonly_test_key_67890"
        k_alpha_ro = db.query(ApiKey).filter(ApiKey.key_hash == hash_api_key(key_alpha_ro)).first()
        if not k_alpha_ro:
            k_alpha_ro = ApiKey(
                id=str(uuid.uuid4()),
                tenant_id="test_tenant_alpha",
                key_hash=hash_api_key(key_alpha_ro),
                key_prefix=key_alpha_ro[:12],
                name="Alpha Search Key",
                scopes="search:read",
                is_active=True
            )
            db.add(k_alpha_ro)

        # 4. Keys for Beta
        key_beta_raw = "jvr_beta_access_test_key_54321"
        k_beta = db.query(ApiKey).filter(ApiKey.key_hash == hash_api_key(key_beta_raw)).first()
        if not k_beta:
            k_beta = ApiKey(
                id=str(uuid.uuid4()),
                tenant_id="test_tenant_beta",
                key_hash=hash_api_key(key_beta_raw),
                key_prefix=key_beta_raw[:12],
                name="Beta Key",
                scopes="search:read,catalog:write,tenant:admin",
                is_active=True
            )
            db.add(k_beta)

        # 5. Products for Alpha
        p_alpha = db.query(Product).filter(Product.tenant_id == "test_tenant_alpha", Product.sku == "SKU-ALPHA-RING-999").first()
        if not p_alpha:
            p_alpha = Product(
                id=str(uuid.uuid4()),
                tenant_id="test_tenant_alpha",
                sku="SKU-ALPHA-RING-999",
                name="Alpha Royal Diamond Ring",
                category="ring",
                status="active"
            )
            db.add(p_alpha)

        # 6. Products for Beta
        p_beta = db.query(Product).filter(Product.tenant_id == "test_tenant_beta", Product.sku == "SKU-BETA-NECK-888").first()
        if not p_beta:
            p_beta = Product(
                id=str(uuid.uuid4()),
                tenant_id="test_tenant_beta",
                sku="SKU-BETA-NECK-888",
                name="Beta Sapphire Pendant Necklace",
                category="necklace",
                status="active"
            )
            db.add(p_beta)

        # 7. Users
        pwd_hash = hashlib.sha256(b"secure_password_123").hexdigest()
        u_alpha = db.query(User).filter(User.email == "lead@alpha.test").first()
        if not u_alpha:
            u_alpha = User(
                id=str(uuid.uuid4()),
                tenant_id="test_tenant_alpha",
                email="lead@alpha.test",
                hashed_password=pwd_hash,
                role="admin",
                is_active=True
            )
            db.add(u_alpha)

        db.commit()

        # Seed vector points into Qdrant for both tenants
        q_idx = QdrantCatalogIndex(model_name="mobilenetv4")
        if q_idx.is_online:
            import numpy as np
            # Seed point for Alpha
            vec_alpha = np.ones(1280, dtype=np.float32)
            vec_alpha = vec_alpha / np.linalg.norm(vec_alpha)
            q_idx.upsert_item(
                item_id="item_alpha_ring_999",
                vector=vec_alpha,
                metadata={
                    "tenant_id": "test_tenant_alpha",
                    "sku_id": "SKU-ALPHA-RING-999",
                    "category": "ring",
                    "filename": "alpha_ring.jpg"
                }
            )

            # Seed point for Beta
            vec_beta = -np.ones(1280, dtype=np.float32)
            vec_beta = vec_beta / np.linalg.norm(vec_beta)
            q_idx.upsert_item(
                item_id="item_beta_neck_888",
                vector=vec_beta,
                metadata={
                    "tenant_id": "test_tenant_beta",
                    "sku_id": "SKU-BETA-NECK-888",
                    "category": "necklace",
                    "filename": "beta_neck.jpg"
                }
            )

        yield {
            "key_alpha": key_alpha_raw,
            "key_alpha_ro": key_alpha_ro,
            "key_beta": key_beta_raw,
            "user_email": "lead@alpha.test",
            "user_pwd": "secure_password_123"
        }
    finally:
        # Teardown: delete test items from Qdrant to preserve catalog size of 432
        try:
            q_idx = QdrantCatalogIndex(model_name="mobilenetv4")
            if q_idx.is_online:
                q_idx.delete_item("item_alpha_ring_999")
                q_idx.delete_item("item_beta_neck_888")
        except Exception:
            pass
        db.close()


def create_test_image_bytes():
    """Generate in-memory test image buffer."""
    img = Image.new("RGB", (128, 128), color=(180, 140, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ==========================================
# 1. Multi-Tenant Isolation Penetration Tests
# ==========================================

def test_tenant_isolation_search_scoping(setup_tenants_and_keys):
    """Test that Tenant Alpha never receives Tenant Beta's points, and vice-versa."""
    keys = setup_tenants_and_keys
    buf_alpha = create_test_image_bytes()

    # Search as Tenant Alpha
    resp_alpha = client.post(
        "/api/search",
        headers={"X-API-Key": keys["key_alpha"]},
        files={"file": ("alpha_query.jpg", buf_alpha, "image/jpeg")},
        data={"model": "mobilenetv4", "category_filter": "all"}
    )
    assert resp_alpha.status_code == 200
    data_alpha = resp_alpha.json()
    # Confirm no Tenant Beta results leak to Alpha
    for r in data_alpha.get("results", []):
        assert r.get("tenant_id") != "test_tenant_beta"

    # Search as Tenant Beta
    buf_beta = create_test_image_bytes()
    resp_beta = client.post(
        "/api/search",
        headers={"X-API-Key": keys["key_beta"]},
        files={"file": ("beta_query.jpg", buf_beta, "image/jpeg")},
        data={"model": "mobilenetv4", "category_filter": "all"}
    )
    assert resp_beta.status_code == 200
    data_beta = resp_beta.json()
    for r in data_beta.get("results", []):
        assert r.get("tenant_id") != "test_tenant_alpha"


def test_adversarial_cross_tenant_spoofing_rejected(setup_tenants_and_keys):
    """Test that caller authenticated as Alpha attempting to query Beta's tenant is blocked with 403."""
    keys = setup_tenants_and_keys
    buf = create_test_image_bytes()

    # Attempt 1: In query string
    resp_query = client.post(
        "/api/search?tenant_id=test_tenant_beta",
        headers={"X-API-Key": keys["key_alpha"]},
        files={"file": ("spoof.jpg", buf, "image/jpeg")}
    )
    assert resp_query.status_code == 403
    assert "Cross-tenant access denied" in resp_query.json()["detail"]

    # Attempt 2: In Form body
    buf.seek(0)
    resp_form = client.post(
        "/api/search",
        headers={"X-API-Key": keys["key_alpha"]},
        files={"file": ("spoof.jpg", buf, "image/jpeg")},
        data={"tenant_id": "test_tenant_beta"}
    )
    assert resp_form.status_code == 403
    assert "Cross-tenant access denied" in resp_form.json()["detail"]

    # Attempt 3: In X-Tenant-ID header
    buf.seek(0)
    resp_hdr = client.post(
        "/api/search",
        headers={"X-API-Key": keys["key_alpha"], "X-Tenant-ID": "test_tenant_beta"},
        files={"file": ("spoof.jpg", buf, "image/jpeg")}
    )
    assert resp_hdr.status_code == 403
    assert "Cross-tenant access denied" in resp_hdr.json()["detail"]


def test_tenant_product_catalog_isolation(setup_tenants_and_keys):
    """Test that GET /api/products returns only the caller's tenant products."""
    keys = setup_tenants_and_keys

    # Alpha products
    resp_a = client.get("/api/products", headers={"X-API-Key": keys["key_alpha"]})
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a["tenant_id"] == "test_tenant_alpha"
    skus_a = [p["sku"] for p in data_a["products"]]
    assert "SKU-ALPHA-RING-999" in skus_a
    assert "SKU-BETA-NECK-888" not in skus_a

    # Beta products
    resp_b = client.get("/api/products", headers={"X-API-Key": keys["key_beta"]})
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert data_b["tenant_id"] == "test_tenant_beta"
    skus_b = [p["sku"] for p in data_b["products"]]
    assert "SKU-BETA-NECK-888" in skus_b
    assert "SKU-ALPHA-RING-999" not in skus_b


# ==========================================
# 2. Authentication, JWT & RBAC Lifecycle Tests
# ==========================================

def test_api_key_invalid_credentials_rejected():
    """Test that bogus API key returns 401 Unauthorized."""
    resp = client.get("/api/auth/me", headers={"X-API-Key": "jvr_completely_invalid_key"})
    assert resp.status_code == 401
    assert "Invalid X-API-Key" in resp.json()["detail"]


def test_jwt_token_login_and_auth_me(setup_tenants_and_keys):
    """Test user login returns JWT and /api/auth/me resolves identity."""
    keys = setup_tenants_and_keys

    # Valid login
    resp_login = client.post(
        "/api/auth/token",
        json={"email": keys["user_email"], "password": keys["user_pwd"]}
    )
    assert resp_login.status_code == 200
    token = resp_login.json()["access_token"]
    assert token is not None

    # Use Bearer token
    resp_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp_me.status_code == 200
    identity = resp_me.json()
    assert identity["tenant_id"] == "test_tenant_alpha"
    assert identity["actor_type"] == "user"
    assert "search:read" in identity["scopes"]


def test_rbac_scope_enforcement(setup_tenants_and_keys):
    """Test that a key lacking tenant:admin scope cannot create or list API keys."""
    keys = setup_tenants_and_keys

    # Read-only key attempts to list keys (requires tenant:admin)
    resp = client.get("/api/auth/keys", headers={"X-API-Key": keys["key_alpha_ro"]})
    assert resp.status_code == 403
    assert "Forbidden" in resp.json()["detail"]


def test_api_key_creation_and_revocation(setup_tenants_and_keys):
    """Test full API key generation, use, and deactivation lifecycle."""
    keys = setup_tenants_and_keys

    # 1. Create key
    resp_create = client.post(
        "/api/auth/keys",
        headers={"X-API-Key": keys["key_alpha"]},
        json={"name": "Temporary Test Key", "scopes": "search:read"}
    )
    assert resp_create.status_code == 200
    key_info = resp_create.json()
    raw_api_key = key_info["api_key"]
    key_id = key_info["key_id"]

    # 2. Verify new key works
    resp_use = client.get("/api/auth/me", headers={"X-API-Key": raw_api_key})
    assert resp_use.status_code == 200
    assert resp_use.json()["tenant_id"] == "test_tenant_alpha"

    # 3. Revoke key
    resp_del = client.delete(f"/api/auth/keys/{key_id}", headers={"X-API-Key": keys["key_alpha"]})
    assert resp_del.status_code == 200

    # 4. Confirm revoked key is rejected
    resp_revoked = client.get("/api/auth/me", headers={"X-API-Key": raw_api_key})
    assert resp_revoked.status_code == 401
    assert "disabled" in resp_revoked.json()["detail"]


# ==========================================
# 3. Rate Limiting & Fail-Secure Policy Tests
# ==========================================

def test_token_bucket_burst_exhaustion():
    """Test token bucket bursts correctly and rejects requests with 429 when exhausted."""
    bucket = InMemoryTokenBucket()
    test_key = "test_exhaustion_key"

    # Bucket with burst capacity 3 and 1 token per minute (very slow refill)
    allowed_1, rem_1, _ = bucket.consume(test_key, rate_per_sec=0.016, burst_capacity=3)
    allowed_2, rem_2, _ = bucket.consume(test_key, rate_per_sec=0.016, burst_capacity=3)
    allowed_3, rem_3, _ = bucket.consume(test_key, rate_per_sec=0.016, burst_capacity=3)
    allowed_4, rem_4, retry_after = bucket.consume(test_key, rate_per_sec=0.016, burst_capacity=3)

    assert allowed_1 is True
    assert allowed_2 is True
    assert allowed_3 is True
    assert allowed_4 is False
    assert retry_after > 0


def test_rate_limit_headers_and_status(setup_tenants_and_keys):
    """Test rate limiting HTTP headers X-RateLimit-Limit, Remaining, Reset."""
    keys = setup_tenants_and_keys
    buf = create_test_image_bytes()

    resp = client.post(
        "/api/search",
        headers={"X-API-Key": keys["key_alpha"]},
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert resp.status_code == 200
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Reset" in resp.headers


def test_fail_secure_limiter_policy():
    """Verify fail-secure policy engages conservative budget during cache/backend outage."""
    limiter = RATE_LIMITER
    # Simulate Redis offline
    limiter._redis_available = False
    limiter._redis_client = None

    allowed, rem, retry_after = limiter.check_rate_limit(
        tenant_id="tenant_failsecure_test",
        actor_id="actor_test",
        rate_limit_rpm=60,
        burst_capacity=10,
        fail_secure_budget_ratio=0.5
    )
    # Must succeed with conservative tokens (burst=5) rather than failing open
    assert allowed is True
    assert rem <= 5


# ==========================================
# 4. Observability & Telemetry Tests
# ==========================================

def test_prometheus_metrics_exposition(setup_tenants_and_keys):
    """Test Prometheus /metrics endpoint exposes standard metric series."""
    keys = setup_tenants_and_keys
    buf = create_test_image_bytes()

    # Trigger search request to record metrics
    _ = client.post(
        "/api/search",
        headers={"X-API-Key": keys["key_alpha"]},
        files={"file": ("metric_query.jpg", buf, "image/jpeg")}
    )

    resp = client.get("/metrics")
    assert resp.status_code == 200
    content = resp.text

    assert "jewellery_retrieval_requests_total" in content
    assert "jewellery_retrieval_latency_seconds" in content
    assert "jewellery_qdrant_operations_total" in content
    assert "jewellery_ratelimit_events_total" in content
    # Verify bounded cardinality invariant: no tenant IDs in metric label values
    assert 'tenant_id="test_tenant_alpha"' not in content


def test_correlation_id_and_security_headers():
    """Test X-Correlation-ID propagation and enterprise security headers."""
    custom_cid = "correl-id-unit-test-9999"
    resp = client.get("/health", headers={"X-Correlation-ID": custom_cid})

    assert resp.status_code == 200
    assert resp.headers.get("X-Correlation-ID") == custom_cid
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert "Strict-Transport-Security" in resp.headers


# ==========================================
# 5. Security & Decompression Bomb Protection Tests
# ==========================================

def test_path_traversal_media_blocked():
    """Test directory traversal in /media endpoint is strictly blocked."""
    resp = client.get("/media/ring/../../etc/passwd")
    # Must be rejected as Bad Request (400) or Not Found (404), never 200
    assert resp.status_code in (400, 404)


def test_payload_size_limit_enforced():
    """Test that payload exceeding 10MB limit is rejected."""
    oversized = io.BytesIO(b"0" * (11 * 1024 * 1024))
    resp = client.post("/api/search", files={"file": ("huge.jpg", oversized, "image/jpeg")})
    assert resp.status_code == 400
    assert "10MB size limit" in resp.json()["detail"]
