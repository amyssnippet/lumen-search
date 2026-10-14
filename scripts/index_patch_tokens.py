#!/usr/bin/env python3
"""Precompute and persist spatial patch tokens for catalog reranking."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.embeddings import get_encoder
from engine.reranker import PatchCatalogIndex

def main():
    parser = argparse.ArgumentParser(description="Precompute catalog patch tokens")
    parser.add_argument("--model", type=str, default="dinov3", choices=["dinov2", "dinov3", "all"])
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--index-dir", type=str, default="data/index")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.manifest, "r") as fp:
        manifest = json.load(fp)

    catalog_items = [img for img in manifest["images"] if img.get("split") == "catalog"]
    models = ["dinov2", "dinov3"] if args.model == "all" else [args.model]

    for m in models:
        print(f"\nBuilding patch index for model '{m}'...")
        t0 = time.time()
        encoder = get_encoder(m)
        patch_idx = PatchCatalogIndex(model_name=m, index_dir=args.index_dir)
        patch_idx.build_from_catalog(catalog_items=catalog_items, encoder=encoder, batch_size=args.batch_size)
        print(f"Completed '{m}' patch indexing in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
