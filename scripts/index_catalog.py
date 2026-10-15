#!/usr/bin/env python3
"""Build and persist catalog embeddings for jewellery visual retrieval."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.embeddings import get_encoder
from engine.index import CatalogIndex

def main():
    parser = argparse.ArgumentParser(description="Index jewellery catalog embeddings")
    parser.add_argument("--model", type=str, default="mobilenetv4", choices=["mobilenetv4", "dinov3", "dinov2", "clip", "all"], help="Embedding model")
    parser.add_argument("--manifest", type=str, default="data/manifest.json", help="Path to manifest JSON")
    parser.add_argument("--index-dir", type=str, default="data/index", help="Output directory for index")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    args = parser.parse_args()

    models = ["mobilenetv4", "dinov3"] if args.model == "all" else [args.model]

    for m in models:
        print(f"\n==========================================")
        print(f"Building index for model: {m}")
        print(f"==========================================")
        encoder = get_encoder(m)
        idx = CatalogIndex(model_name=m, index_dir=args.index_dir)
        count = idx.build_from_manifest(manifest_path=args.manifest, encoder=encoder, batch_size=args.batch_size)
        print(f"Successfully indexed {count} items for {m} into {args.index_dir}")

if __name__ == "__main__":
    main()
