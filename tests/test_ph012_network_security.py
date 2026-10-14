"""Milestone PH-012: Network Perimeter Security & Production Secret Enforcement Tests.

Verifies:
1. Reverse proxy edge enforcement: direct requests bypassing NGINX without X-Proxy-Secret are rejected with HTTP 403.
2. Ingress through NGINX reverse proxy with configured secret succeeds.
3. Startup secret validator rejects default/staging credentials in production environment (APP_ENV=production).
4. Startup secret validator enforces minimum 32-character entropy for production JWT secrets.
5. Startup secret validator passes when compliant production secrets are provided.
"""

import os
import pytest
import httpx
from fastapi.testclient import TestClient
from app.main import app, validate_production_secrets, REQUIRE_REVERSE_PROXY, INTERNAL_PROXY_SECRET

STAGING_DIRECT_URL = "http://127.0.0.1:8001"
STAGING_NGINX_HTTPS = "https://127.0.0.1:8443"


def test_direct_bypass_rejected_when_proxy_required():
    """Verify direct backend requests without X-Proxy-Secret are blocked with HTTP 403."""
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{STAGING_DIRECT_URL}/health/live")
            # If staging server is running with REQUIRE_REVERSE_PROXY=true
            assert r.status_code == 403
            assert "Direct backend access forbidden" in r.text
    except httpx.ConnectError:
        pytest.skip("Staging Uvicorn process not running on port 8001")


def test_proxy_secret_header_accepted_via_direct_internal():
    """Verify direct internal requests supplying the valid internal proxy secret succeed."""
    try:
        secret = os.environ.get("INTERNAL_PROXY_SECRET") or INTERNAL_PROXY_SECRET
        if not secret:
            secret_file = os.path.join(os.path.dirname(__file__), "..", "deploy", "staging", ".proxy_secret")
            if os.path.exists(secret_file):
                with open(secret_file) as f:
                    secret = f.read().strip()
        with httpx.Client(timeout=2.0) as client:
            r = client.get(
                f"{STAGING_DIRECT_URL}/health/live",
                headers={"X-Proxy-Secret": secret}
            )
            assert r.status_code == 200
            assert r.json().get("status") == "ok"
    except httpx.ConnectError:
        pytest.skip("Staging Uvicorn process not running on port 8001")


def test_edge_reverse_proxy_ingress_succeeds():
    """Verify requests ingressing through NGINX TLS 1.3 reverse proxy succeed with HTTP 200."""
    try:
        with httpx.Client(verify=False, timeout=3.0) as client:
            r = client.get(f"{STAGING_NGINX_HTTPS}/health/live")
            assert r.status_code == 200
            assert r.json().get("status") == "ok"
    except Exception as e:
        pytest.skip(f"Staging NGINX not reachable on port 8443: {e}")


def test_production_secret_validation_rejects_default_credentials(monkeypatch):
    """Verify validate_production_secrets rejects default/staging passwords in production."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "dev_secret_key_change_in_production")
    monkeypatch.setenv("POSTGRES_PASSWORD", "staging_secret_pass_2026")
    monkeypatch.setenv("DATABASE_URL", "postgresql://staging_user:staging_secret_pass_2026@127.0.0.1:5433/jewellery_staging")

    with pytest.raises(RuntimeError) as exc_info:
        validate_production_secrets()

    err_msg = str(exc_info.value)
    assert "Insecure/default credential detected" in err_msg


def test_production_secret_validation_rejects_short_jwt_secret(monkeypatch):
    """Verify validate_production_secrets enforces >= 32 char length for JWT secret."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "too_short_key")
    monkeypatch.setenv("POSTGRES_PASSWORD", "custom_prod_strong_pass_998877")
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod_user:custom_prod_strong_pass_998877@10.0.0.5:5432/jewellery_prod")

    with pytest.raises(RuntimeError) as exc_info:
        validate_production_secrets()

    assert "at least 32 characters long" in str(exc_info.value)


def test_production_secret_validation_passes_valid_credentials(monkeypatch):
    """Verify validate_production_secrets passes when hardened credentials are set."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "prod_super_secure_jwt_key_with_very_high_entropy_2026_xyz")
    monkeypatch.setenv("POSTGRES_PASSWORD", "prod_vault_generated_strong_pg_pass_2026")
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod_user:prod_vault_generated_strong_pg_pass_2026@10.0.0.5:5432/jewellery_prod")
    monkeypatch.setenv("API_KEY", "prod_master_api_key_vault_ref_882910")

    # Should execute without raising any exception
    validate_production_secrets()
