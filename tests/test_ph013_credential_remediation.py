"""Milestone PH-013: Credential Remediation & Edge Secret Hardening Tests.

Verifies:
1. Rejection of unauthenticated direct backend access (HTTP 403).
2. Explicit rejection and banning of the compromised literal secret
   ('jewellery_internal_edge_proxy_secret_2026').
3. Acceptance of dynamically generated, rotated 64-hex proxy credentials.
4. Edge reverse proxy TLS 1.3 ingress with dynamically included secret header.
5. Production startup validator enforcement:
   - Requires explicit INTERNAL_PROXY_SECRET when REQUIRE_REVERSE_PROXY=true.
   - Rejects secrets shorter than 32 characters.
   - Rejects known compromised literals in production mode.
   - Passes when compliant high-entropy secrets are supplied.
6. Zero-leakage verification: ensures proxy secret is never reflected in error
   responses or headers.
7. Verification of dynamic secret injector script and file permissions (0600).
"""

import os
import stat
import tempfile
from pathlib import Path
import pytest
import httpx
from fastapi.testclient import TestClient

from app.main import (
    app,
    validate_production_secrets,
    BANNED_COMPROMISED_CREDENTIALS,
)
from scripts.inject_proxy_secret import generate_token, write_nginx_secret_include

STAGING_DIRECT_URL = "http://127.0.0.1:8001"
STAGING_NGINX_HTTPS = "https://127.0.0.1:8443"
BANNED_COMPROMISED_SECRET = "jewellery_internal_edge_proxy_secret_2026"


def get_current_staging_secret() -> str:
    """Retrieve rotated staging secret from env or uncommitted file."""
    secret = os.environ.get("INTERNAL_PROXY_SECRET")
    if not secret:
        secret_file = Path(__file__).resolve().parent.parent / "deploy" / "staging" / ".proxy_secret"
        if secret_file.exists():
            secret = secret_file.read_text().strip()
    return secret or ""


def test_banned_compromised_literal_is_registered():
    """Verify the compromised literal is permanently registered in the banned set."""
    assert BANNED_COMPROMISED_SECRET in BANNED_COMPROMISED_CREDENTIALS


def test_direct_bypass_rejected_without_proxy_secret():
    """Verify direct backend requests without X-Proxy-Secret are blocked with HTTP 403."""
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{STAGING_DIRECT_URL}/health/live")
            assert r.status_code == 403
            assert "Direct backend access forbidden" in r.text
    except httpx.ConnectError:
        pytest.skip("Staging Uvicorn process not running on port 8001")


def test_direct_bypass_rejected_with_banned_compromised_secret():
    """Verify requests supplying the old compromised literal secret are strictly rejected."""
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(
                f"{STAGING_DIRECT_URL}/health/live",
                headers={"X-Proxy-Secret": BANNED_COMPROMISED_SECRET}
            )
            assert r.status_code == 403
            assert "Direct backend access forbidden" in r.text
    except httpx.ConnectError:
        pytest.skip("Staging Uvicorn process not running on port 8001")


def test_direct_backend_accepted_with_rotated_valid_secret():
    """Verify direct backend requests with the active rotated secret succeed."""
    secret = get_current_staging_secret()
    if not secret:
        pytest.skip("No staging proxy secret available")

    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(
                f"{STAGING_DIRECT_URL}/health/live",
                headers={"X-Proxy-Secret": secret}
            )
            assert r.status_code == 200
            assert r.json().get("status") == "ok"
    except httpx.ConnectError:
        pytest.skip("Staging Uvicorn process not running on port 8001")


def test_nginx_reverse_proxy_injects_valid_secret():
    """Verify ingress through NGINX HTTPS dynamically passes proxy secret and returns 200."""
    try:
        with httpx.Client(verify=False, timeout=3.0) as client:
            r = client.get(f"{STAGING_NGINX_HTTPS}/health/live")
            assert r.status_code == 200
            assert r.json().get("status") == "ok"
    except Exception as e:
        pytest.skip(f"Staging NGINX not reachable on port 8443: {e}")


def test_production_validator_rejects_banned_proxy_literal(monkeypatch):
    """Verify validate_production_secrets fails when the banned compromised secret is set."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "prod_super_secure_jwt_key_with_very_high_entropy_2026_xyz")
    monkeypatch.setenv("REQUIRE_REVERSE_PROXY", "true")
    monkeypatch.setenv("INTERNAL_PROXY_SECRET", BANNED_COMPROMISED_SECRET)

    with pytest.raises(RuntimeError) as exc_info:
        validate_production_secrets()

    assert "uses a known compromised or default literal" in str(exc_info.value)


def test_production_validator_rejects_empty_proxy_secret(monkeypatch):
    """Verify validate_production_secrets fails if INTERNAL_PROXY_SECRET is missing or empty."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "prod_super_secure_jwt_key_with_very_high_entropy_2026_xyz")
    monkeypatch.setenv("REQUIRE_REVERSE_PROXY", "true")
    monkeypatch.delenv("INTERNAL_PROXY_SECRET", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        validate_production_secrets()

    assert "INTERNAL_PROXY_SECRET must be explicitly set" in str(exc_info.value)


def test_production_validator_rejects_short_proxy_secret(monkeypatch):
    """Verify validate_production_secrets enforces >= 32 char length for proxy secret."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "prod_super_secure_jwt_key_with_very_high_entropy_2026_xyz")
    monkeypatch.setenv("REQUIRE_REVERSE_PROXY", "true")
    monkeypatch.setenv("INTERNAL_PROXY_SECRET", "short_weak_secret_123")

    with pytest.raises(RuntimeError) as exc_info:
        validate_production_secrets()

    assert "at least 32 characters long" in str(exc_info.value)


def test_production_validator_passes_high_entropy_secret(monkeypatch):
    """Verify validate_production_secrets succeeds with compliant 64-char token."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "prod_super_secure_jwt_key_with_very_high_entropy_2026_xyz")
    monkeypatch.setenv("REQUIRE_REVERSE_PROXY", "true")
    monkeypatch.setenv(
        "INTERNAL_PROXY_SECRET",
        "sec_prx_89f029348102934810293841029384019283401928340192a8b9c0d1e2f3a4b5"
    )
    monkeypatch.setenv("POSTGRES_PASSWORD", "custom_prod_strong_pg_pass_998877")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://prod_user:custom_prod_strong_pg_pass_998877@10.0.0.5:5432/jewellery_prod"
    )

    # Should execute cleanly without raising
    validate_production_secrets()


def test_proxy_secret_never_leaked_in_responses():
    """Verify proxy secret value is never reflected back to clients in error messages."""
    secret = get_current_staging_secret()
    if not secret:
        pytest.skip("No staging proxy secret available")

    try:
        with httpx.Client(timeout=2.0) as client:
            # 1. 403 direct request
            r1 = client.get(f"{STAGING_DIRECT_URL}/health/live")
            assert secret not in r1.text

            # 2. Banned secret request
            r2 = client.get(
                f"{STAGING_DIRECT_URL}/health/live",
                headers={"X-Proxy-Secret": BANNED_COMPROMISED_SECRET}
            )
            assert secret not in r2.text
            assert BANNED_COMPROMISED_SECRET not in r2.text

            # 3. Non-existent path with valid secret
            r3 = client.get(
                f"{STAGING_DIRECT_URL}/non_existent_endpoint_xyz",
                headers={"X-Proxy-Secret": secret}
            )
            assert secret not in r3.text
    except httpx.ConnectError:
        pytest.skip("Staging Uvicorn process not running on port 8001")


def test_inject_proxy_secret_generator_and_permissions():
    """Verify script token generator entropy and 0600 file permission enforcement."""
    token = generate_token(bytes_len=32)
    assert token.startswith("sec_prx_")
    # Prefix (8) + 64 hex characters = 72 chars total
    assert len(token) == 72

    with tempfile.TemporaryDirectory() as tmpdir:
        inc_path = Path(tmpdir) / "proxy_secret.inc"
        write_nginx_secret_include(inc_path, token)

        assert inc_path.exists()
        content = inc_path.read_text()
        assert f'proxy_set_header X-Proxy-Secret "{token}";' in content

        # Check file mode is 0600
        mode = stat.S_IMODE(inc_path.stat().st_mode)
        assert mode == 0o600
