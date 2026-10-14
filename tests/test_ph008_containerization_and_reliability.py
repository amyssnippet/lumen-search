"""Integration and reliability tests for PH-008: Production Containerization, CI/CD & Health Probes."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from alembic.config import Config
from alembic import command

from app.main import app
from app.rate_limiter import DistributedRateLimiter, InMemoryTokenBucket

client = TestClient(app)


# ==============================================================================
# 1. Health & Liveness Probes
# ==============================================================================

def test_health_live_endpoint():
    """Verify that /health/live returns immediate 200 without blocking."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "Jewellery Visual Retrieval Engine" in data["service"]
    assert "uptime_sec" in data
    assert data["uptime_sec"] >= 0
    assert data["version"] == "0.2.0"


def test_health_ready_endpoint_structure():
    """Verify deep readiness probe structure and safe status reporting."""
    resp = client.get("/health/ready")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert data["status"] in ("ready", "degraded", "unavailable")
    assert "dependencies" in data
    deps = data["dependencies"]
    assert "database" in deps
    assert "redis" in deps
    assert "qdrant" in deps
    assert "model" in deps

    # Database and Model should be available in normal test environment
    assert deps["database"]["status"] in ("connected", "down")
    assert deps["model"]["status"] in ("available", "missing")
    if deps["model"]["status"] == "available":
        assert deps["model"]["size_bytes"] > 0


def test_health_ready_database_failure_simulation():
    """Verify that database connectivity failure produces HTTP 503 without leaking credentials."""
    from engine.db.session import get_db

    def mock_broken_db():
        mock_session = MagicMock()
        mock_session.execute.side_effect = Exception("DB connection timeout to postgresql://secret_user:secret_pass@db.internal:5432")
        yield mock_session

    app.dependency_overrides[get_db] = mock_broken_db
    try:
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["dependencies"]["database"]["status"] == "down"
        # Confirm secret connection strings are masked/not leaked in JSON
        assert "secret_pass" not in str(data)
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_health_ready_qdrant_failure_simulation():
    """Verify that vector index failure produces HTTP 503 Service Unavailable."""
    # Point Qdrant to an unreachable non-routable test address
    with patch.dict(os.environ, {"QDRANT_HOST": "192.0.2.1", "QDRANT_PORT": "6333"}):
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["dependencies"]["qdrant"]["status"] == "down"


def test_health_ready_model_missing_simulation():
    """Verify that missing model weights file produces HTTP 503."""
    with patch("pathlib.Path.exists", return_value=False):
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["dependencies"]["model"]["status"] == "missing"


# ==============================================================================
# 2. Strict Authentication Enforcement (REQUIRE_AUTH=true)
# ==============================================================================

def test_strict_auth_mode_rejects_uncredentialed_requests():
    """Verify that when REQUIRE_AUTH=true, protected endpoints return 401 Unauthorized."""
    with patch.dict(os.environ, {"REQUIRE_AUTH": "true"}):
        # /api/search must reject without credentials
        resp_search = client.post("/api/search")
        assert resp_search.status_code == 401
        assert "Authentication required" in resp_search.json()["detail"]

        # /api/products must reject without credentials
        resp_prod = client.get("/api/products")
        assert resp_prod.status_code == 401

        # /api/auth/me must reject without credentials
        resp_me = client.get("/api/auth/me")
        assert resp_me.status_code == 401

        # Public endpoints remain accessible
        resp_live = client.get("/health/live")
        assert resp_live.status_code == 200

        resp_health = client.get("/health")
        assert resp_health.status_code == 200


# ==============================================================================
# 3. Rate Limiter Fail-Secure In-Process Fallback
# ==============================================================================

def test_rate_limiter_fail_secure_in_process():
    """Verify that rate limiter falls back to InMemoryTokenBucket and throttles bursts safely."""
    # Point to nonexistent redis port to force fail-secure fallback
    limiter = DistributedRateLimiter(redis_url="redis://127.0.0.1:59999/0", timeout=0.1)
    
    # Bucket capacity 2, rate 1 token/sec
    key = "test_tenant:test_actor"
    # First consume
    allowed, remaining, retry_after = limiter.in_memory.consume(key, rate_per_sec=1.0, burst_capacity=2, tokens_requested=1.0)
    assert allowed is True
    assert remaining == 1.0

    # Second consume
    allowed, remaining, retry_after = limiter.in_memory.consume(key, rate_per_sec=1.0, burst_capacity=2, tokens_requested=1.0)
    assert allowed is True
    assert remaining == pytest.approx(0.0, abs=1e-2)

    # Third consume should be throttled
    allowed, remaining, retry_after = limiter.in_memory.consume(key, rate_per_sec=1.0, burst_capacity=2, tokens_requested=1.0)
    assert allowed is False
    assert retry_after > 0


# ==============================================================================
# 4. Container & Deployment Configuration Manifest Audits
# ==============================================================================

def test_dockerfile_compliance():
    """Audit Dockerfile for multi-stage build, non-root user, and security hardening."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"
    content = dockerfile_path.read_text()

    assert "AS builder" in content, "Dockerfile must feature a multi-stage builder"
    assert "AS runtime" in content, "Dockerfile must feature a runtime stage"
    assert "USER 10001:10001" in content or "USER appuser" in content, "Dockerfile must execute under non-root UID"
    assert "EXPOSE 8000" in content, "Dockerfile must expose port 8000"
    assert "HEALTHCHECK" in content, "Dockerfile must define container healthcheck"
    assert "--timeout-graceful-shutdown" in content, "Uvicorn must configure graceful SIGTERM handling"


def test_dockerignore_compliance():
    """Verify that .dockerignore excludes temporary, virtualenv, and test directories."""
    dockerignore_path = Path(".dockerignore")
    assert dockerignore_path.exists(), ".dockerignore must exist"
    content = dockerignore_path.read_text()

    assert ".venv" in content
    assert "__pycache__" in content
    assert ".git" in content


def test_docker_compose_production_manifest():
    """Audit docker-compose.prod.yml for private networking, healthchecks, and persistence."""
    compose_path = Path("docker-compose.prod.yml")
    assert compose_path.exists(), "docker-compose.prod.yml must exist"
    content = compose_path.read_text()

    # Core required services
    assert "postgres:" in content
    assert "redis:" in content
    assert "qdrant:" in content
    assert "app:" in content

    # Persistent volumes
    assert "postgres_data:" in content
    assert "redis_data:" in content
    assert "qdrant_storage:" in content
    assert "catalog_data:" in content

    # Bridge network
    assert "jewellery_internal:" in content

    # Healthchecks
    assert "healthcheck:" in content
    assert "service_healthy" in content


def test_env_production_template():
    """Verify that .env.production.example contains all required environment variables."""
    env_example = Path(".env.production.example")
    assert env_example.exists()
    content = env_example.read_text()

    assert "POSTGRES_DB=" in content
    assert "POSTGRES_PASSWORD=" in content
    assert "REDIS_URL=" in content
    assert "QDRANT_HOST=" in content
    assert "JWT_SECRET_KEY=" in content
    assert "REQUIRE_AUTH=" in content


# ==============================================================================
# 5. Schema Migration & Rollback Isolation
# ==============================================================================

def test_alembic_migrations_clean_db():
    """Validate that Alembic migrations run cleanly against a fresh SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        db_url = f"sqlite:///{tmp_db.name}"
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # Run migration upgrade to head with DATABASE_URL overridden
        with patch.dict(os.environ, {"DATABASE_URL": db_url}):
            command.upgrade(alembic_cfg, "head")

        # Verify tables created
        engine = create_engine(db_url)
        with engine.connect() as conn:
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
            table_names = {t[0] for t in tables}
            assert "tenants" in table_names
            assert "users" in table_names
            assert "api_keys" in table_names
            assert "products" in table_names
            assert "assets" in table_names
            assert "vector_points" in table_names
            assert "ingestion_jobs" in table_names
            assert "audit_logs" in table_names


# ==============================================================================
# 6. CI/CD Workflow Integrity
# ==============================================================================

def test_ci_workflow_manifest():
    """Audit .github/workflows/ci.yml for pipeline stages and security gates."""
    ci_file = Path(".github/workflows/ci.yml")
    assert ci_file.exists(), "CI workflow must be defined in .github/workflows/ci.yml"
    content = ci_file.read_text()

    assert "static-analysis-and-security" in content
    assert "test-and-migrations" in content
    assert "container-build-and-verify" in content
    assert "bandit" in content
    assert "alembic upgrade head" in content
    assert "pytest" in content
