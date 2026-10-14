"""Index and cache catalog representations under soft spatial attention pooling.

Extracts DINOv2 & DINOv3 patch tokens and computes pre-aggregated catalog embedding matrices:
- Uniform Patch Pooling (Config A)
- Foreground Saliency Weighting (Config B)
- Soft Component Region Weighting (Config C)
- Combined Soft Weighting (Config E)

Saves versioned metadata and matrix representations in data/index/.
"""

import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

from engine.embeddings import get_encoder
from engine.localization import JewelleryLocalizer
from engine.soft_attention import (
    compute_uniform_weights,
    compute_foreground_weights,
    compute_soft_region_weights,
    compute_combined_weights,
    aggregate_patch_tokens
)

def index_soft_attention_catalog():
    index_dir = Path("data/index")
    meta_path = index_dir / "dinov2_meta.json"
    with open(meta_path, "r") as fp:
        catalog_items = json.load(fp)

    catalog_ids = [item["id"] for item in catalog_items]
    print(f"Indexing soft attention features for {len(catalog_items)} catalog items...")

    # Load catalog images
    print("Loading catalog images...")
    catalog_images = [Image.open(item["rel_path"]).convert("RGB") for item in catalog_items]

    # Check / build DINOv2 patches
    v2_patch_path = index_dir / "dinov2_patches.npy"
    if not v2_patch_path.exists():
        print("Extracting DINOv2 patch tokens for 432 catalog items...")
        t0 = time.time()
        enc_v2 = get_encoder("dinov2")
        v2_patches = enc_v2.get_patch_tokens_batch(catalog_images, batch_size=32)
        np.save(v2_patch_path, v2_patches.astype(np.float32))
        print(f"Saved DINOv2 patches: {v2_patches.shape} in {time.time() - t0:.2f}s")
    else:
        print(f"Loading existing DINOv2 patches from {v2_patch_path}...")
        v2_patches = np.load(v2_patch_path)

    # Load DINOv3 patches
    v3_patch_path = index_dir / "dinov3_patches.npy"
    print(f"Loading existing DINOv3 patches from {v3_patch_path}...")
    v3_patches = np.load(v3_patch_path)

    # Load region metadata from EXP-006
    reg_meta_path = index_dir / "region_meta.json"
    with open(reg_meta_path, "r") as fp:
        reg_meta = json.load(fp)

    # Compute weights for all catalog images
    print("Computing soft spatial weights for catalog items...")
    t_w0 = time.time()
    
    weights_uniform = []
    weights_fg = []
    weights_reg = []
    weights_comb = []

    for idx, (img, rm) in enumerate(zip(catalog_images, reg_meta)):
        box = rm.get("box", [0, 0, 1, 1])
        is_fallback = rm.get("is_fallback", False)

        w_u = compute_uniform_weights()
        w_f = compute_foreground_weights(img)
        w_r = compute_soft_region_weights(box)
        w_c = compute_combined_weights(img, box, is_fallback=is_fallback)

        weights_uniform.append(w_u)
        weights_fg.append(w_f)
        weights_reg.append(w_r)
        weights_comb.append(w_c)

    weights_uniform = np.array(weights_uniform, dtype=np.float32)
    weights_fg = np.array(weights_fg, dtype=np.float32)
    weights_reg = np.array(weights_reg, dtype=np.float32)
    weights_comb = np.array(weights_comb, dtype=np.float32)

    print(f"Weights computed in {time.time() - t_w0:.2f}s")

    # Aggregate and save for both models
    configs = [
        ("soft_uniform", weights_uniform),
        ("soft_foreground", weights_fg),
        ("soft_region", weights_reg),
        ("soft_combined", weights_comb),
    ]

    for model_name, patches in [("dinov2", v2_patches), ("dinov3", v3_patches)]:
        print(f"\nAggregating {model_name} catalog representations...")
        for cfg_name, w_matrix in configs:
            agg_embs = aggregate_patch_tokens(patches, w_matrix)
            out_path = index_dir / f"{model_name}_{cfg_name}.npy"
            np.save(out_path, agg_embs.astype(np.float32))
            print(f"  Saved {out_path.name}: {agg_embs.shape}")

    # Compute manifest hash
    manifest_path = Path("data/manifest.json")
    manifest_hash = "unknown"
    if manifest_path.exists():
        with open(manifest_path, "rb") as fp:
            manifest_hash = hashlib.sha256(fp.read()).hexdigest()

    meta_doc = {
        "version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_sha256": manifest_hash,
        "catalog_count": len(catalog_ids),
        "catalog_ids": catalog_ids,
        "models": ["dinov2", "dinov3"],
        "configurations": ["soft_uniform", "soft_foreground", "soft_region", "soft_combined"]
    }

    meta_out_path = index_dir / "soft_attention_meta.json"
    with open(meta_out_path, "w") as fp:
        json.dump(meta_doc, fp, indent=2)

    print(f"\nMetadata saved to {meta_out_path}")
    print("Catalog soft attention indexing complete.")

if __name__ == "__main__":
    index_soft_attention_catalog()
