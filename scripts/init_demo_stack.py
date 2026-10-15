#!/usr/bin/env python3
"""Initialization and bootstrap script for the Jewellery Visual Retrieval Demo stack.

Ensures:
1. Database connectivity and executes Alembic schema migrations.
2. Relational catalogue and user credentials seeded.
3. Redis rate limiter online.
4. Qdrant vector database initialized with 432 catalogue vectors and alias bound.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo_bootstrap")

DB_URL = os.environ.get("DATABASE_URL", "postgresql://demo_user:demo_secret_pass_2026@demo_db:5432/jewellery_demo")
if DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
REDIS_URL = os.environ.get("REDIS_URL", "redis://demo_redis:6379/0")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "demo_qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
PROD_ALIAS = os.environ.get("QDRANT_PRODUCTION_ALIAS", "jewellery_catalog_mobilenetv4_staging")


def wait_for_postgres(timeout_sec: int = 60):
    logger.info("Waiting for PostgreSQL database...")
    from sqlalchemy import create_engine, text
    engine = create_engine(DB_URL)
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("PostgreSQL is connected and responding.")
            return True
        except Exception as e:
            time.sleep(2)
    logger.error("PostgreSQL connection timed out.")
    return False


def wait_for_redis(timeout_sec: int = 30):
    logger.info("Waiting for Redis cache / rate limiter...")
    import redis
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = redis.Redis.from_url(REDIS_URL)
            if r.ping():
                logger.info("Redis is connected and responding.")
                return True
        except Exception:
            time.sleep(1)
    logger.error("Redis connection timed out.")
    return False


def wait_for_qdrant(timeout_sec: int = 45):
    logger.info(f"Waiting for Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    from qdrant_client import QdrantClient
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
            qc.get_collections()
            logger.info("Qdrant vector engine is connected and responding.")
            return True
        except Exception:
            time.sleep(2)
    logger.error("Qdrant connection timed out.")
    return False


def run_migrations():
    logger.info("Applying Alembic migrations...")
    res = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Alembic migration failed:\n{res.stderr}")
        sys.exit(1)
    logger.info("Alembic schema migrations completed successfully.")


def seed_relational_data():
    logger.info("Seeding relational database (tenants, API keys, demo users, catalogue inventory)...")
    from scripts.seed_relational_catalog import seed_database
    seed_database()
    logger.info("Relational database seeded successfully.")


def setup_qdrant_collection():
    logger.info(f"Verifying Qdrant vector index and alias '{PROD_ALIAS}'...")
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from scripts.rebuild_qdrant_index import rebuild_model_index

    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10.0)
    colls = [c.name for c in qc.get_collections().collections]
    
    target_collection = "jewellery_catalog_mobilenetv4_v1"
    
    needs_rebuild = False
    if target_collection not in colls:
        needs_rebuild = True
    else:
        info = qc.get_collection(target_collection)
        if not info.points_count or info.points_count < 100:
            needs_rebuild = True

    if needs_rebuild:
        logger.info(f"Rebuilding collection '{target_collection}' with MobileNetV4 vectors...")
        success = rebuild_model_index(
            model_name="mobilenetv4",
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            recreate=True
        )
        if not success:
            logger.error("Failed to build Qdrant index.")
            sys.exit(1)
    else:
        logger.info(f"Collection '{target_collection}' already populated with vectors.")

    # Initialize DINOv3 collection if index files are present
    dino_target = "jewellery_catalog_dinov3_v1"
    dino_emb = Path("data/index/dinov3_embeddings.npy")
    if dino_emb.exists():
        dino_needs_rebuild = False
        if dino_target not in colls:
            dino_needs_rebuild = True
        else:
            dinfo = qc.get_collection(dino_target)
            if not dinfo.points_count or dinfo.points_count < 100:
                dino_needs_rebuild = True
        if dino_needs_rebuild:
            logger.info(f"Rebuilding collection '{dino_target}' with DINOv3 vectors...")
            rebuild_model_index(
                model_name="dinov3",
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                recreate=True
            )
        else:
            logger.info(f"Collection '{dino_target}' already populated with vectors.")

    # Ensure alias binds to target collection
    existing_aliases = qc.get_aliases()
    aliases_for_target = [a.alias_name for a in existing_aliases.aliases if a.collection_name == target_collection]
    
    aliases_to_bind = [PROD_ALIAS, "jewellery_catalog_mobilenetv4"]
    for alias_name in aliases_to_bind:
        if alias_name not in aliases_for_target:
            logger.info(f"Binding alias '{alias_name}' -> '{target_collection}'...")
            try:
                qc.update_collection_aliases(
                    change_aliases_operations=[
                        rest.CreateAliasOperation(
                            create_alias=rest.CreateAlias(
                                collection_name=target_collection,
                                alias_name=alias_name
                            )
                        )
                    ]
                )
                logger.info(f"Alias '{alias_name}' successfully created.")
            except Exception as e:
                logger.warning(f"Alias operation on '{alias_name}' returned: {e}")


def main():
    logger.info("=== LUMEN Jewellery Retrieval Demo Bootstrap Starting ===")
    
    if not wait_for_postgres():
        sys.exit(1)
    if not wait_for_redis():
        sys.exit(1)
    if not wait_for_qdrant():
        sys.exit(1)

    run_migrations()
    seed_relational_data()
    setup_qdrant_collection()

    logger.info("=== Bootstrap Complete: Demo Backend & Datastores Fully Operational ===")


if __name__ == "__main__":
    main()
