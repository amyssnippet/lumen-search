#!/usr/bin/env python3
"""End-to-end catalog re-indexing utility for LUMEN Visual Retrieval Engine.

Can be run entirely inside Docker with a single command:
    docker exec -it jewellery_demo_backend python scripts/reindex_all.py

Steps performed:
1. Audits Jewellery_Data/ and builds data/manifest.json.
2. Computes embeddings for all models (MobileNetV4, DINOv3) into data/index/.
3. Seeds PostgreSQL with catalog products and assets.
4. Rebuilds Qdrant HNSW vector collections for both models.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_manifest import build_manifest
from engine.index import CatalogIndex
from engine.embeddings import get_encoder
from scripts.seed_relational_catalog import seed_database
from scripts.rebuild_qdrant_index import rebuild_model_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_all")


def main():
    logger.info("======================================================================")
    logger.info("   LUMEN JEWELLERY RETRIEVAL — FULL CATALOG RE-INDEXING PIPELINE")
    logger.info("======================================================================")
    t0 = time.time()

    # Step 1: Scan images and build manifest
    logger.info("[1/4] Scanning Jewellery_Data/ and building data/manifest.json...")
    build_manifest(data_dir="Jewellery_Data", output_path="data/manifest.json")

    # Step 2: Compute vector embeddings for all models
    models = ["mobilenetv4", "dinov3"]
    for m in models:
        logger.info(f"[2/4] Computing feature embeddings for model: {m}...")
        encoder = get_encoder(m)
        idx = CatalogIndex(model_name=m, index_dir="data/index")
        count = idx.build_from_manifest(manifest_path="data/manifest.json", encoder=encoder, batch_size=32)
        logger.info(f"Indexed {count} items for {m} into data/index/{m}_embeddings.npy")

    # Step 3: Seed relational database (PostgreSQL)
    logger.info("[3/4] Updating PostgreSQL relational catalog and inventory...")
    seed_database()

    # Step 4: Rebuild Qdrant vector index collections
    qdrant_host = os.environ.get("QDRANT_HOST", "demo_qdrant")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
    for m in models:
        logger.info(f"[4/4] Rebuilding Qdrant HNSW vector collection for model: {m} at {qdrant_host}:{qdrant_port}...")
        rebuild_model_index(
            model_name=m,
            host=qdrant_host,
            port=qdrant_port,
            recreate=True
        )

    elapsed = time.time() - t0
    logger.info("======================================================================")
    logger.info(f"✓ FULL RE-INDEXING COMPLETED IN {elapsed:.2f}s!")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
