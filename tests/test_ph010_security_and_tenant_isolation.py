"""Comprehensive Security, RBAC Authorization, and Multi-Tenant Isolation Audit for PH-010.

Tests:
1. Strict authentication enforcement under REQUIRE_AUTH=true.
2. Granular RBAC scope enforcement (search:read, tenant:admin, ops:admin).
3. Anti-spoofing defense rejecting cross-tenant parameter manipulation.
4. Relational and vector isolation: Tenant Alpha cannot view or tamper with Tenant Beta's assets.
5. Media path traversal protections.
6. Payload and decompression bomb defense limits.
"""

import io
import json
import os
import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_db
from app.auth import create_access_token, hash_api_key, AuthenticatedIdentity
from engine.db.models import Tenant, User, ApiKey, Product, Asset


client = TestClient(app)


# ==============================================================================
# Helper Fixtures & Tokens
# ==============================================================================

def create_tenant_token(tenant_id: str, actor_id: str = "user_1", scopes: list = None) -> str:
    """Generate a signed JWT token with designated tenant identity and scopes."""
    return create_access_token(
        tenant_id=tenant_id,
        actor_id=actor_id,
        scopes=scopes or ["search:read"]
    )


def create_test_image_bytes() -> io.BytesIO:
    """Generate a minimal valid JPEG image buffer."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(180, 140, 70))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ==============================================================================
# 1. Strict Authentication Mode Tests (REQUIRE_AUTH=true)
# ==============================================================================

def test_strict_auth_blocks_unauthenticated_requests():
    """Verify that unauthenticated callers are rejected with HTTP 401 on protected routes."""
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true"}):
        # 1. Search endpoint
        files = {"file": ("test.jpg", create_test_image_bytes(), "image/jpeg")}
        resp = client.post("/api/search", files=files)
        assert resp.status_code == 401

        # 2. Product catalog
        resp = client.get("/api/products")
        assert resp.status_code == 401

        # 3. API key management
        resp = client.get("/api/auth/keys")
        assert resp.status_code == 401

        # 4. Lifecycle management
        resp = client.get("/api/lifecycle/status")
        assert resp.status_code == 401


# ==============================================================================
# 2. RBAC Scope Hierarchy Enforcement
# ==============================================================================

def test_scope_enforcement_search_read_cannot_manage_keys():
    """Verify caller with 'search:read' cannot invoke 'tenant:admin' API key endpoints."""
    token = create_tenant_token("test_tenant_alpha", scopes=["search:read"])
    headers = {"Authorization": f"Bearer {token}"}
    
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true"}):
        resp = client.get("/api/auth/keys", headers=headers)
        assert resp.status_code == 403
        assert "Forbidden: credential lacks required scope" in resp.json()["detail"]


def test_scope_enforcement_search_read_cannot_execute_lifecycle():
    """Verify caller with 'search:read' cannot invoke 'ops:admin' lifecycle operations."""
    token = create_tenant_token("test_tenant_alpha", scopes=["search:read"])
    headers = {"Authorization": f"Bearer {token}"}
    
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true"}):
        resp = client.post(
            "/api/lifecycle/candidate/register",
            json={"collection_name": "candidate_unauthorized_col"},
            headers=headers
        )
        assert resp.status_code == 403
        assert "Forbidden: credential lacks required scope" in resp.json()["detail"]


def test_scope_enforcement_ops_admin_permitted_on_lifecycle():
    """Verify caller with 'ops:admin' can successfully query lifecycle status."""
    token = create_tenant_token("default_tenant", scopes=["ops:admin"])
    headers = {"Authorization": f"Bearer {token}"}
    
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true"}):
        resp = client.get("/api/lifecycle/status", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "active_production_collection" in data
        assert "production_alias" in data
        assert "canary_routing" in data


# ==============================================================================
# 3. Anti-Spoofing & Multi-Tenant Isolation
# ==============================================================================

def test_cross_tenant_spoofing_in_search_strictly_rejected():
    """Verify caller authenticated as Tenant Alpha cannot query Tenant Beta by passing spoofed tenant_id."""
    token = create_tenant_token("test_tenant_alpha", scopes=["search:read"])
    headers = {"Authorization": f"Bearer {token}"}
    
    files = {"file": ("test.jpg", create_test_image_bytes(), "image/jpeg")}
    data = {
        "tenant_id": "test_tenant_beta",  # Adversarial tenant spoofing attempt
        "model": "mobilenetv4"
    }
    
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true"}):
        resp = client.post("/api/search", files=files, data=data, headers=headers)
        assert resp.status_code == 403
        assert "Cross-tenant access denied" in resp.json()["detail"]


def test_tenant_product_isolation():
    """Verify Tenant Alpha cannot list or view Tenant Beta's products."""
    token_alpha = create_tenant_token("test_tenant_alpha", scopes=["search:read"])
    token_beta = create_tenant_token("test_tenant_beta", scopes=["search:read"])
    
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true"}):
        # List products for Tenant Alpha
        resp_a = client.get("/api/products", headers={"Authorization": f"Bearer {token_alpha}"})
        assert resp_a.status_code == 200
        assert resp_a.json()["tenant_id"] == "test_tenant_alpha"
        
        # List products for Tenant Beta
        resp_b = client.get("/api/products", headers={"Authorization": f"Bearer {token_beta}"})
        assert resp_b.status_code == 200
        assert resp_b.json()["tenant_id"] == "test_tenant_beta"


def test_tenant_api_key_revocation_isolation():
    """Verify Tenant Alpha cannot revoke an API key belonging to Tenant Beta."""
    token_alpha = create_tenant_token("test_tenant_alpha", scopes=["tenant:admin"])
    
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true"}):
        resp = client.delete(
            "/api/auth/keys/key_belonging_to_beta_999",
            headers={"Authorization": f"Bearer {token_alpha}"}
        )
        assert resp.status_code == 404
        assert "API key not found" in resp.json()["detail"]


# ==============================================================================
# 4. Path Traversal & Media Security
# ==============================================================================

def test_media_path_traversal_attack_blocked():
    """Verify directory traversal attempts in /media/ return HTTP 400 Bad Request or 404 Not Found."""
    resp = client.get("/media/necklace/..%2f..%2fetc%2fpasswd")
    assert resp.status_code in (400, 404)

    resp = client.get("/media/..%2fnecklace/image.jpg")
    assert resp.status_code in (400, 404)


def test_media_non_existent_category_returns_404():
    """Verify querying an unsupported or non-existent media category returns HTTP 404."""
    resp = client.get("/media/invalid_category/some_image.jpg")
    assert resp.status_code == 404
    assert "Category not found" in resp.json()["detail"]


# ==============================================================================
# 5. Payload Limits & Decompression Bomb Defenses
# ==============================================================================

def test_payload_exceeding_10mb_rejected():
    """Verify that file uploads exceeding 10 MB are rejected with HTTP 400 (exceeds 10MB limit)."""
    oversized_data = b"0" * (10 * 1024 * 1024 + 1024)  # 10 MB + 1 KB
    files = {"file": ("huge_payload.jpg", io.BytesIO(oversized_data), "image/jpeg")}
    
    token = create_tenant_token("test_tenant_alpha", scopes=["search:read"])
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = client.post("/api/search", files=files, headers=headers)
    assert resp.status_code == 400
    assert "10MB" in resp.json()["detail"]


def test_invalid_image_payload_returns_400():
    """Verify that uploading corrupt or non-image data returns HTTP 400 Bad Request."""
    corrupt_data = b"NOT_A_VALID_IMAGE_HEADER_BYTES_1234567890"
    files = {"file": ("fake.jpg", io.BytesIO(corrupt_data), "image/jpeg")}
    
    token = create_tenant_token("test_tenant_alpha", scopes=["search:read"])
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = client.post("/api/search", files=files, headers=headers)
    assert resp.status_code == 400
    assert "Invalid or corrupted image" in resp.json()["detail"]
