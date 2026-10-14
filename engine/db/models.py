"""SQLAlchemy ORM models for the Jewellery Visual Retrieval Engine."""

import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Tenant(Base):
    """Multi-tenant organization account."""
    __tablename__ = "tenants"

    id = Column(String(64), primary_key=True)  # e.g., 'default_tenant', 'tenant_luxury_corp'
    name = Column(String(255), nullable=False)
    tier = Column(String(32), nullable=False, default="standard")  # free, standard, enterprise
    rate_limit_rpm = Column(Integer, nullable=False, default=300)
    rate_limit_burst = Column(Integer, nullable=False, default=30)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="tenant", cascade="all, delete-orphan")
    products = relationship("Product", back_populates="tenant", cascade="all, delete-orphan")
    assets = relationship("Asset", back_populates="tenant", cascade="all, delete-orphan")
    vector_points = relationship("VectorPoint", back_populates="tenant", cascade="all, delete-orphan")
    jobs = relationship("IngestionJob", back_populates="tenant", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    """User account within a tenant."""
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="viewer")  # admin, editor, viewer, ops
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    tenant = relationship("Tenant", back_populates="users")


class ApiKey(Base):
    """Hashed programmatic API credentials bound to a tenant."""
    __tablename__ = "api_keys"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    key_hash = Column(String(64), nullable=False, index=True)  # SHA-256 of raw secret
    key_prefix = Column(String(16), nullable=False)           # First 8 chars for identification
    name = Column(String(128), nullable=False)
    scopes = Column(String(255), nullable=False, default="search:read")  # Comma-separated scopes
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    tenant = relationship("Tenant", back_populates="api_keys")


class Product(Base):
    """Jewellery product entity (physical SKU or item)."""
    __tablename__ = "products"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    sku = Column(String(128), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(64), nullable=False, index=True)  # ring, necklace
    metal_color = Column(String(64), nullable=True)
    metal_purity = Column(String(32), nullable=True)
    gemstone_type = Column(String(64), nullable=True)
    gemstone_shape = Column(String(64), nullable=True)
    setting_type = Column(String(64), nullable=True)
    price_cents = Column(Integer, nullable=True)
    currency = Column(String(8), nullable=False, default="USD")
    status = Column(String(32), nullable=False, default="active")
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("tenant_id", "sku", name="uq_product_tenant_sku"),
        Index("ix_product_tenant_cat", "tenant_id", "category"),
    )

    tenant = relationship("Tenant", back_populates="products")
    assets = relationship("Asset", back_populates="product", cascade="all, delete-orphan")


class Asset(Base):
    """Image asset associated with a jewellery product."""
    __tablename__ = "assets"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    product_id = Column(String(64), ForeignKey("products.id", ondelete="CASCADE"), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)
    perspective = Column(String(64), nullable=False, default="unspecified")
    view_role = Column(String(64), nullable=False, default="gallery_anchor")
    provenance_tier = Column(String(64), nullable=False, default="tier_2_catalog")
    sha256 = Column(String(64), nullable=False)
    width = Column(Integer, nullable=False, default=0)
    height = Column(Integer, nullable=False, default=0)
    is_quarantined = Column(Boolean, nullable=False, default=False)
    quarantine_reason = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("tenant_id", "storage_path", name="uq_asset_tenant_path"),
        Index("ix_asset_tenant_prod", "tenant_id", "product_id"),
    )

    tenant = relationship("Tenant", back_populates="assets")
    product = relationship("Product", back_populates="assets")
    vector_point = relationship("VectorPoint", back_populates="asset", uselist=False, cascade="all, delete-orphan")


class VectorPoint(Base):
    """Vector index pointer referencing Qdrant point."""
    __tablename__ = "vector_points"

    id = Column(String(64), primary_key=True)  # Qdrant point UUID
    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_id = Column(String(64), ForeignKey("assets.id", ondelete="CASCADE"), nullable=True, index=True)
    product_id = Column(String(64), ForeignKey("products.id", ondelete="CASCADE"), nullable=True, index=True)
    collection_name = Column(String(128), nullable=False)
    vector_dim = Column(Integer, nullable=False, default=1280)
    is_indexed = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    __table_args__ = (
        Index("ix_vec_tenant_coll", "tenant_id", "collection_name"),
    )

    tenant = relationship("Tenant", back_populates="vector_points")
    asset = relationship("Asset", back_populates="vector_point")


class IngestionJob(Base):
    """Asynchronous and idempotent studio batch ingestion job."""
    __tablename__ = "ingestion_jobs"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    batch_id = Column(String(128), nullable=False, index=True)
    idempotency_key = Column(String(128), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="pending")  # pending, processing, completed, failed
    total_assets = Column(Integer, nullable=False, default=0)
    ingested_assets = Column(Integer, nullable=False, default=0)
    failed_assets = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_job_tenant_idempotency"),
    )

    tenant = relationship("Tenant", back_populates="jobs")


class AuditLog(Base):
    """Immutable audit trail for enterprise operations."""
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    actor_type = Column(String(32), nullable=False)  # api_key, user, system
    actor_id = Column(String(128), nullable=False)
    action = Column(String(64), nullable=False)      # search, ingest, promote_collection, create_key, etc.
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(128), nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False, default="success")  # success, denied, failed
    details = Column(Text, nullable=True)            # Safe JSON string (no tokens or PII)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_audit_tenant_action", "tenant_id", "action", "created_at"),
    )

    tenant = relationship("Tenant", back_populates="audit_logs")
