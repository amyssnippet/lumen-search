import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

from engine.embeddings import get_encoder
from engine.localization import JewelleryLocalizer

def index_catalog_regions():
    meta_path = Path("data/index/dinov2_meta.json")
    with open(meta_path, "r") as fp:
        catalog_items = json.load(fp)

    print(f"Indexing regions for {len(catalog_items)} catalog items...")
    localizer = JewelleryLocalizer()
    
    crops = []
    region_meta = []
    t0 = time.time()
    
    fallback_count = 0
    for idx, item in enumerate(catalog_items):
        img_path = Path(item["rel_path"])
        cat = item["category"]
        img = Image.open(img_path).convert("RGB")
        
        loc_res = localizer.locate(img, category=cat)
        crops.append(loc_res.crop)
        
        if loc_res.is_fallback:
            fallback_count += 1
            
        region_meta.append({
            "id": item["id"],
            "category": cat,
            "rel_path": item["rel_path"],
            "box": [round(float(x), 4) for x in loc_res.box],
            "method": loc_res.method,
            "confidence": loc_res.confidence,
            "is_fallback": loc_res.is_fallback
        })
        
    t_loc = time.time() - t0
    print(f"Localization complete in {t_loc:.2f}s ({t_loc/len(catalog_items)*1000:.2f}ms/item). Fallbacks: {fallback_count}/{len(catalog_items)}")
    
    # Save region metadata
    with open("data/index/region_meta.json", "w") as fp:
        json.dump(region_meta, fp, indent=2)
        
    # Extract DINOv2 region embeddings
    print("Extracting DINOv2 embeddings for localized region crops...")
    t_enc_v2_start = time.time()
    enc_v2 = get_encoder("dinov2")
    embs_v2 = enc_v2.encode_batch(crops, batch_size=32)
    # L2 normalize
    norms_v2 = np.linalg.norm(embs_v2, axis=1, keepdims=True)
    norms_v2[norms_v2 == 0] = 1e-9
    embs_v2 = embs_v2 / norms_v2
    np.save("data/index/dinov2_region_embeddings.npy", embs_v2.astype(np.float32))
    print(f"DINOv2 region embeddings saved: {embs_v2.shape} in {time.time() - t_enc_v2_start:.2f}s")
    
    # Extract DINOv3 region embeddings
    print("Extracting DINOv3 embeddings for localized region crops...")
    t_enc_v3_start = time.time()
    enc_v3 = get_encoder("dinov3")
    embs_v3 = enc_v3.encode_batch(crops, batch_size=32)
    norms_v3 = np.linalg.norm(embs_v3, axis=1, keepdims=True)
    norms_v3[norms_v3 == 0] = 1e-9
    embs_v3 = embs_v3 / norms_v3
    np.save("data/index/dinov3_region_embeddings.npy", embs_v3.astype(np.float32))
    print(f"DINOv3 region embeddings saved: {embs_v3.shape} in {time.time() - t_enc_v3_start:.2f}s")

if __name__ == "__main__":
    index_catalog_regions()
