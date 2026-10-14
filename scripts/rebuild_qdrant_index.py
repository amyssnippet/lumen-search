#!/usr/bin/env python3
"""CLI utility to rebuild and migrate Qdrant vector index collections.

Usage:
    python scripts/rebuild_qdrant_index.py --model mobilenetv4 --recreate
    python scripts/rebuild_qdrant_index.py --model dinov3 --recreate
    python scripts/rebuild_qdrant_index.py --all --recreate
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.qdrant_index import QdrantCatalogIndex, MODEL_DIMENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rebuild_qdrant_index")


def rebuild_model_index(
    model_name: str,
    host: str = "127.0.0.1",
    port: int = 6333,
    m: int = 32,
    ef_construct: int = 200,
    on_disk: bool = True,
    recreate: bool = True
):
    logger.info(f"--- Rebuilding Qdrant Index for model '{model_name}' ---")
    
    index_dir = Path("data/index")
    emb_path = index_dir / f"{model_name}_embeddings.npy"
    meta_path = index_dir / f"{model_name}_meta.json"
    
    if not emb_path.exists() or not meta_path.exists():
        logger.error(f"Cannot find pre-computed index files: {emb_path} or {meta_path}")
        return False

    q_idx = QdrantCatalogIndex(
        model_name=model_name,
        host=host,
        port=port,
        m=m,
        ef_construct=ef_construct,
        on_disk=on_disk
    )
    
    if not q_idx.is_online:
        logger.error(f"Qdrant server at {host}:{port} is unreachable. Ensure the daemon is running.")
        return False

    t0 = time.time()
    count = q_idx.build_from_numpy(
        embeddings_path=str(emb_path),
        meta_path=str(meta_path),
        manifest_path="data/ingestion_manifest.json",
        recreate=recreate
    )
    elapsed = time.time() - t0
    
    stats = q_idx.stats()
    logger.info(f"Rebuild completed in {elapsed:.2f}s! Stats: {json.dumps(stats, indent=2)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Rebuild Qdrant vector index collections from disk manifests.")
    parser.add_argument("--model", type=str, default="mobilenetv4", help="Model name (mobilenetv4, dinov3, dinov2, etc.)")
    parser.add_argument("--all", action="store_true", help="Rebuild all available model indices.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Qdrant server host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant server HTTP port")
    parser.add_argument("--m", type=int, default=32, help="HNSW M parameter (number of bidirectional links per node)")
    parser.add_argument("--ef-construct", type=int, default=200, help="HNSW ef_construct parameter")
    parser.add_argument("--no-on-disk", dest="on_disk", action="store_false", help="Store vectors entirely in RAM rather than mmap on-disk")
    parser.add_argument("--no-recreate", dest="recreate", action="store_false", help="Append/update points instead of wiping and recreating collection")

    args = parser.parse_args()

    if args.all:
        models = ["mobilenetv4", "dinov3"]
        for m in models:
            rebuild_model_index(
                model_name=m,
                host=args.host,
                port=args.port,
                m=args.m,
                ef_construct=args.ef_construct,
                on_disk=args.on_disk,
                recreate=args.recreate
            )
    else:
        rebuild_model_index(
            model_name=args.model,
            host=args.host,
            port=args.port,
            m=args.m,
            ef_construct=args.ef_construct,
            on_disk=args.on_disk,
            recreate=args.recreate
        )


if __name__ == "__main__":
    main()
