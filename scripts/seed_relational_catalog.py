"""Seed relational catalog database with default tenants, keys, users, and manifest assets."""

import hashlib
import json
import logging
import uuid
from pathlib import Path
from sqlalchemy.orm import Session

from engine.db.session import init_db, SessionLocal
from engine.db.models import (
    Tenant,
    User,
    ApiKey,
    Product,
    Asset,
    VectorPoint,
)
from engine.qdrant_index import catalog_id_to_uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hash_key(raw_key: str) -> str:
    """Compute SHA-256 hash of API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def seed_database():
    """Seed initial multi-tenant entities and manifest records."""
    init_db()
    db: Session = SessionLocal()

    try:
        # 1. Tenants
        tenants_data = [
            {
                "id": "default_tenant",
                "name": "Default Jewellery Organization",
                "tier": "standard",
                "rate_limit_rpm": 300,
                "rate_limit_burst": 30,
            },
            {
                "id": "tenant_luxury_corp",
                "name": "Luxury Jewellery Corp",
                "tier": "enterprise",
                "rate_limit_rpm": 1200,
                "rate_limit_burst": 100,
            },
            {
                "id": "tenant_artisan_gems",
                "name": "Artisan Gems Studio",
                "tier": "free",
                "rate_limit_rpm": 60,
                "rate_limit_burst": 10,
            },
        ]

        for td in tenants_data:
            existing = db.query(Tenant).filter(Tenant.id == td["id"]).first()
            if not existing:
                t = Tenant(**td)
                db.add(t)
                logger.info(f"Created tenant: {td['id']}")
            else:
                for k, v in td.items():
                    setattr(existing, k, v)
        db.commit()

        # 2. API Keys
        # Master dev keys with predictable values for testing
        api_keys_data = [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "key_default_dev")),
                "tenant_id": "default_tenant",
                "raw_key": "jvr_default_dev_key_2026_secret",
                "name": "Default Developer Key",
                "scopes": "search:read,catalog:write,tenant:admin,ops:admin",
            },
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "key_luxury_corp")),
                "tenant_id": "tenant_luxury_corp",
                "raw_key": "jvr_luxury_corp_key_2026_secret",
                "name": "Luxury Corp Production Key",
                "scopes": "search:read,catalog:write,tenant:admin",
            },
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "key_artisan_gems")),
                "tenant_id": "tenant_artisan_gems",
                "raw_key": "jvr_artisan_gems_key_2026_secret",
                "name": "Artisan Gems Read-Only Key",
                "scopes": "search:read",
            },
        ]

        for akd in api_keys_data:
            kh = hash_key(akd["raw_key"])
            existing_key = db.query(ApiKey).filter(ApiKey.id == akd["id"]).first()
            if not existing_key:
                ak = ApiKey(
                    id=akd["id"],
                    tenant_id=akd["tenant_id"],
                    key_hash=kh,
                    key_prefix=akd["raw_key"][:12],
                    name=akd["name"],
                    scopes=akd["scopes"],
                    is_active=True,
                )
                db.add(ak)
                logger.info(f"Created API key for tenant: {akd['tenant_id']} (prefix: {akd['raw_key'][:12]})")
            else:
                existing_key.key_hash = kh
                existing_key.scopes = akd["scopes"]
        db.commit()

        # 3. Users
        # SHA256 hashed password for "dev_password_2026"
        dev_pwd_hash = hashlib.sha256(b"dev_password_2026").hexdigest()
        users_data = [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "user_admin_default")),
                "tenant_id": "default_tenant",
                "email": "admin@jewellery.internal",
                "hashed_password": dev_pwd_hash,
                "role": "admin",
            },
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "user_ops_default")),
                "tenant_id": "default_tenant",
                "email": "ops@jewellery.internal",
                "hashed_password": dev_pwd_hash,
                "role": "ops",
            },
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "user_admin_luxury")),
                "tenant_id": "tenant_luxury_corp",
                "email": "lead@luxurycorp.com",
                "hashed_password": dev_pwd_hash,
                "role": "admin",
            },
        ]

        for ud in users_data:
            existing_user = db.query(User).filter(User.email == ud["email"]).first()
            if not existing_user:
                u = User(**ud)
                db.add(u)
                logger.info(f"Created user: {ud['email']}")
        db.commit()

        # 4. Ingest products and assets from data/ingestion_manifest.json for default_tenant
        manifest_path = Path("data/ingestion_manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r") as fp:
                manifest = json.load(fp)

            records = manifest.get("records", [])
            logger.info(f"Loading {len(records)} manifest records into relational catalog...")

            # Group by SKU or identity group
            products_created = 0
            assets_created = 0

            for r in records:
                img_id = r["image_id"]
                cat = r.get("category", "unknown")
                raw_sku = r.get("sku_id")
                if not raw_sku:
                    ident = r.get("identity_group_id")
                    if ident and not ident.startswith("PHYS-SINGLE"):
                        raw_sku = f"SKU-{cat.upper()}-{ident}"
                    else:
                        raw_sku = f"SKU-{cat.upper()}-{img_id}"

                # Product lookup or create
                prod = db.query(Product).filter(
                    Product.tenant_id == "default_tenant",
                    Product.sku == raw_sku
                ).first()

                if not prod:
                    prod = Product(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"default_tenant_{raw_sku}")),
                        tenant_id="default_tenant",
                        sku=raw_sku,
                        name=f"Jewellery {cat.title()} ({raw_sku})",
                        category=cat,
                        status="active",
                    )
                    db.add(prod)
                    db.flush()
                    products_created += 1

                # Asset lookup or create
                storage_path = r.get("rel_path", f"Jewellery_Data/{cat}/{img_id}.jpg")
                asset = db.query(Asset).filter(
                    Asset.tenant_id == "default_tenant",
                    Asset.storage_path == storage_path
                ).first()

                if not asset:
                    is_quarantined = bool(r.get("exclusion_reason"))
                    asset = Asset(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"asset_default_{img_id}")),
                        tenant_id="default_tenant",
                        product_id=prod.id,
                        filename=Path(storage_path).name,
                        storage_path=storage_path,
                        perspective=r.get("view_angle", "unspecified"),
                        view_role="gallery_anchor" if r.get("dataset_split") == "catalog_index" else "validation_query",
                        provenance_tier="tier_1_studio" if "multishot" in r.get("physical_identity_verification_status", "") else "tier_2_catalog",
                        sha256=r.get("sha256", "0" * 64),
                        width=r.get("width", 262),
                        height=r.get("height", 262),
                        is_quarantined=is_quarantined,
                        quarantine_reason=r.get("exclusion_reason"),
                    )
                    db.add(asset)
                    db.flush()
                    assets_created += 1

                # VectorPoint pointer
                pt_id = catalog_id_to_uuid(img_id)
                vp = db.query(VectorPoint).filter(VectorPoint.id == pt_id).first()
                if not vp:
                    vp = VectorPoint(
                        id=pt_id,
                        tenant_id="default_tenant",
                        asset_id=asset.id,
                        product_id=prod.id,
                        collection_name="jewellery_catalog_mobilenetv4_v1",
                        vector_dim=1280,
                        is_indexed=True,
                    )
                    db.add(vp)

            db.commit()
            logger.info(f"Relational catalog seeded successfully: {products_created} products, {assets_created} assets.")

    except Exception as e:
        db.rollback()
        logger.error(f"Seeding failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
