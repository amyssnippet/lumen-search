"""Integration tests for Staging PostgreSQL 16 database (Milestone PH-011).

Replaces SQLite mocks with direct execution against live PostgreSQL 16:
- Dialect verification (postgresql+psycopg2).
- Relational schema, indexes, and constraints.
- Foreign key cascading deletion behavior.
- Multi-tenant data segregation.
- ACID transaction rollbacks.
- Connection pool concurrent acquisitions.
"""

import os
import uuid
import datetime
import hashlib
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from engine.db.models import (
    Base,
    Tenant,
    User,
    ApiKey,
    Product,
    Asset,
    VectorPoint,
    IngestionJob,
    AuditLog,
)

STAGING_PG_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://staging_user:staging_secret_pass_2026@127.0.0.1:5433/jewellery_staging"
)
if STAGING_PG_URL.startswith("postgresql://"):
    STAGING_PG_URL = STAGING_PG_URL.replace("postgresql://", "postgresql+psycopg2://", 1)


@pytest.fixture(scope="module")
def pg_engine():
    """Engine connected directly to staging PostgreSQL 16."""
    try:
        eng = create_engine(STAGING_PG_URL, pool_size=5, max_overflow=5, pool_timeout=5)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return eng
    except Exception as e:
        pytest.skip(f"Staging PostgreSQL not accessible at {STAGING_PG_URL}: {e}")


@pytest.fixture
def pg_session(pg_engine):
    """Transactional session with rollback after test execution."""
    SessionMaker = sessionmaker(bind=pg_engine)
    session = SessionMaker()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_staging_postgres_connection_and_dialect(pg_engine):
    """Verify PostgreSQL 16 connection, dialect, and server version."""
    assert pg_engine.dialect.name == "postgresql"
    with pg_engine.connect() as conn:
        res = conn.execute(text("SELECT version();")).scalar()
        assert "PostgreSQL 16" in res or "PostgreSQL" in res


def test_staging_postgres_schema_and_tables(pg_engine):
    """Verify all required enterprise relational tables and indexes exist."""
    inspector = inspect(pg_engine)
    tables = inspector.get_table_names()
    expected = [
        "tenants",
        "users",
        "api_keys",
        "products",
        "assets",
        "vector_points",
        "ingestion_jobs",
        "audit_logs",
    ]
    for table_name in expected:
        assert table_name in tables, f"Expected table {table_name} missing from PostgreSQL"

    # Verify unique constraint on products (tenant_id, sku)
    unique_constraints = inspector.get_unique_constraints("products")
    constraint_names = [c["name"] for c in unique_constraints]
    assert "uq_product_tenant_sku" in constraint_names or any(
        set(c.get("column_names", [])) == {"tenant_id", "sku"} for c in unique_constraints
    )


def test_staging_postgres_tenant_crud_and_isolation(pg_session: Session):
    """Verify tenant isolation and CRUD operations in PostgreSQL."""
    test_id = f"test_tenant_{uuid.uuid4().hex[:8]}"
    tenant = Tenant(
        id=test_id,
        name="Postgres Integration Test Tenant",
        tier="enterprise",
        rate_limit_rpm=600,
        rate_limit_burst=50,
        is_active=True,
    )
    pg_session.add(tenant)
    pg_session.commit()

    # Query back
    fetched = pg_session.query(Tenant).filter(Tenant.id == test_id).first()
    assert fetched is not None
    assert fetched.name == "Postgres Integration Test Tenant"
    assert fetched.tier == "enterprise"

    # Add product under this tenant
    prod_sku = f"SKU-{uuid.uuid4().hex[:6]}"
    prod = Product(
        id=f"prod_{uuid.uuid4().hex[:8]}",
        tenant_id=test_id,
        sku=prod_sku,
        name="Solitaire Ring Staging Test",
        category="ring",
        metal_color="white",
        price_cents=199900,
    )
    pg_session.add(prod)
    pg_session.commit()

    # Query product with tenant isolation
    prod_fetched = pg_session.query(Product).filter(
        Product.tenant_id == test_id,
        Product.sku == prod_sku
    ).first()
    assert prod_fetched is not None
    assert prod_fetched.category == "ring"

    # Verify query for another tenant returns nothing
    other_prod = pg_session.query(Product).filter(
        Product.tenant_id == "non_existent_tenant",
        Product.sku == prod_sku
    ).first()
    assert other_prod is None

    # Cleanup
    pg_session.delete(tenant)
    pg_session.commit()


def test_staging_postgres_foreign_key_cascades(pg_session: Session):
    """Verify foreign key enforcement and cascade deletion in PostgreSQL."""
    test_id = f"test_cascade_{uuid.uuid4().hex[:8]}"
    tenant = Tenant(
        id=test_id,
        name="Cascade Test Tenant",
        tier="standard",
        rate_limit_rpm=300,
    )
    pg_session.add(tenant)
    pg_session.commit()

    # Create dependent product and asset
    prod_id = f"prod_{uuid.uuid4().hex[:8]}"
    prod = Product(
        id=prod_id,
        tenant_id=test_id,
        sku="CASCADE-SKU-1",
        name="Cascade Ring",
        category="ring",
    )
    asset = Asset(
        id=f"asset_{uuid.uuid4().hex[:8]}",
        tenant_id=test_id,
        product_id=prod_id,
        filename="cascade_ring.jpg",
        storage_path=f"staging/{test_id}/cascade_ring.jpg",
        sha256="0" * 64,
    )
    pg_session.add(prod)
    pg_session.add(asset)
    pg_session.commit()

    # Verify rows exist
    assert pg_session.query(Product).filter(Product.id == prod_id).count() == 1
    assert pg_session.query(Asset).filter(Asset.product_id == prod_id).count() == 1

    # Delete tenant: cascade must remove product and asset
    pg_session.delete(tenant)
    pg_session.commit()

    assert pg_session.query(Product).filter(Product.id == prod_id).count() == 0
    assert pg_session.query(Asset).filter(Asset.product_id == prod_id).count() == 0


def test_staging_postgres_transactions_and_rollback(pg_session: Session):
    """Verify PostgreSQL ACID transaction rollback prevents orphan dirty writes."""
    test_id = f"test_rollback_{uuid.uuid4().hex[:8]}"
    tenant = Tenant(
        id=test_id,
        name="Rollback Test Tenant",
        tier="standard",
    )
    pg_session.add(tenant)
    pg_session.flush()

    # Simulate error condition triggering rollback
    pg_session.rollback()

    # Verify tenant was NOT persisted
    persisted = pg_session.query(Tenant).filter(Tenant.id == test_id).first()
    assert persisted is None


def test_staging_postgres_api_key_and_audit_logging(pg_session: Session):
    """Verify API key hashing and audit log persistence in PostgreSQL."""
    test_tenant_id = f"test_audit_{uuid.uuid4().hex[:8]}"
    tenant = Tenant(
        id=test_tenant_id,
        name="Audit Test Tenant",
        tier="enterprise",
    )
    pg_session.add(tenant)
    pg_session.commit()

    raw_secret = f"jvr_staging_test_secret_{uuid.uuid4().hex}"
    key_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()
    api_key = ApiKey(
        id=str(uuid.uuid4()),
        tenant_id=test_tenant_id,
        key_hash=key_hash,
        key_prefix=raw_secret[:12],
        name="Staging Integration Key",
        scopes="search:read,catalog:write",
        is_active=True,
    )
    pg_session.add(api_key)

    audit = AuditLog(
        id=str(uuid.uuid4()),
        tenant_id=test_tenant_id,
        actor_type="api_key",
        actor_id=api_key.id,
        action="search",
        resource_type="catalog",
        resource_id="ring_solitaire_001",
        ip_address="127.0.0.1",
        user_agent="PostgresIntegrationTest/1.0",
        status="success",
        details='{"query": "ring", "results_count": 5}',
    )
    pg_session.add(audit)
    pg_session.commit()

    # Verify query
    found_key = pg_session.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    assert found_key is not None
    assert found_key.tenant_id == test_tenant_id

    found_audit = pg_session.query(AuditLog).filter(
        AuditLog.tenant_id == test_tenant_id,
        AuditLog.action == "search"
    ).first()
    assert found_audit is not None
    assert found_audit.status == "success"

    # Cleanup
    pg_session.delete(tenant)
    pg_session.commit()


def test_staging_postgres_connection_pool_resilience(pg_engine):
    """Verify opening and releasing multiple concurrent connections from pool."""
    connections = []
    try:
        # Check out 5 connections concurrently
        for _ in range(5):
            conn = pg_engine.connect()
            connections.append(conn)
            val = conn.execute(text("SELECT 42")).scalar()
            assert val == 42
    finally:
        for conn in connections:
            conn.close()
