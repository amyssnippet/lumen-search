"""Region-aware visual search engine for jewellery items.

Combines full-image global features with localized gemstone/pendant crops.
Provides configurations for Full-image, Region-only, Combined Score Fusion,
and Region Reranking, with graceful fallback handling.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

from engine.embeddings import get_encoder
from engine.index import CatalogIndex
from engine.localization import JewelleryLocalizer, LocalizationResult
from engine.preprocessing import PreprocessConfig

class RegionAwareSearchEngine:
    """Retrieval engine operating over full-image and fine-grained localized component embeddings."""

    def __init__(self, model_name: str = "dinov2"):
        self.model_name = model_name.lower()
        self.encoder = get_encoder(self.model_name)
        self.catalog_index = CatalogIndex(model_name=self.model_name)
        self.catalog_index.load()
        self.localizer = JewelleryLocalizer()

        # Load catalog region embeddings
        reg_emb_path = Path(f"data/index/{self.model_name}_region_embeddings.npy")
        if not reg_emb_path.exists():
            raise FileNotFoundError(f"Catalog region embeddings not found at {reg_emb_path}. Run scripts/index_catalog_regions.py first.")
        self.catalog_region_embeddings = np.load(reg_emb_path)

        # Load region metadata
        reg_meta_path = Path("data/index/region_meta.json")
        if reg_meta_path.exists():
            import json
            with open(reg_meta_path, "r") as fp:
                self.region_meta = json.load(fp)
        else:
            self.region_meta = []

    def search(
        self,
        image: Image.Image,
        mode: str = "combined",          # "full", "region", "combined", "rerank", "oracle"
        region_weight: float = 0.35,      # weight w in: (1 - w)*global + w*region
        top_k: int = 10,
        candidate_depth: int = 20,
        category_filter: Optional[str] = None,
        custom_box: Optional[List[float]] = None, # Normalized [ymin, xmin, ymax, xmax] for Oracle mode
        preprocess_config: Optional[PreprocessConfig] = None
    ) -> Dict[str, Any]:
        """Perform region-aware search."""
        t_start = time.time()
        w_img, h_img = image.size

        # 1. Localize region
        t_loc_0 = time.time()
        if custom_box is not None:
            # Custom / Oracle ground-truth box
            y1 = max(0, int(custom_box[0] * h_img))
            x1 = max(0, int(custom_box[1] * w_img))
            y2 = min(h_img, int(custom_box[2] * h_img))
            x2 = min(w_img, int(custom_box[3] * w_img))
            if (x2 - x1) <= 2 or (y2 - y1) <= 2:
                # Degenerate custom box -> fallback to full image
                crop = image
                loc_res = LocalizationResult(
                    box=[0.0, 0.0, 1.0, 1.0],
                    box_pixels=[0, 0, h_img, w_img],
                    label="fallback",
                    confidence=0.5,
                    method="custom_box_degenerate_fallback",
                    is_fallback=True,
                    crop=crop
                )
            else:
                crop = image.crop((x1, y1, x2, y2))
                loc_res = LocalizationResult(
                    box=custom_box,
                    box_pixels=[y1, x1, y2, x2],
                    label="oracle_ground_truth",
                    confidence=1.0,
                    method="oracle_ground_truth",
                    is_fallback=False,
                    crop=crop
                )
        else:
            loc_res = self.localizer.locate(image, category=category_filter)
        t_localization = (time.time() - t_loc_0) * 1000

        # 2. Extract query embeddings
        t_enc_0 = time.time()
        # Global query embedding
        q_global = self.encoder.encode_image(image)
        norm_g = np.linalg.norm(q_global)
        if norm_g > 0:
            q_global = q_global / norm_g

        # Region query embedding
        q_region = self.encoder.encode_image(loc_res.crop)
        norm_r = np.linalg.norm(q_region)
        if norm_r > 0:
            q_region = q_region / norm_r
        t_encoding = (time.time() - t_enc_0) * 1000

        # 3. Compute similarities across catalog
        t_score_0 = time.time()
        cat_embeddings = self.catalog_index.embeddings
        global_scores = np.dot(cat_embeddings, q_global)
        region_scores = np.dot(self.catalog_region_embeddings, q_region)

        # Apply fallback weighting: if query region is fallback, gracefully ignore region score
        effective_w = 0.0 if (loc_res.is_fallback and mode != "region") else float(region_weight)

        # Category mask
        valid_mask = np.ones(len(self.catalog_index.metadata), dtype=bool)
        if category_filter and category_filter.lower() not in ("all", ""):
            cf = category_filter.lower()
            for i, meta in enumerate(self.catalog_index.metadata):
                if meta.get("category", "").lower() != cf:
                    valid_mask[i] = False

        # Compute ranking scores according to mode
        mode_lower = mode.lower()
        if mode_lower == "full":
            fused_scores = global_scores.copy()
        elif mode_lower == "region":
            fused_scores = region_scores.copy()
        elif mode_lower in ("combined", "oracle"):
            fused_scores = (1.0 - effective_w) * global_scores + effective_w * region_scores
        elif mode_lower == "rerank":
            # First select candidate_depth items by global score
            masked_global = np.where(valid_mask, global_scores, -1e9)
            top_candidates_idx = np.argsort(masked_global)[::-1][:candidate_depth]
            
            # Rerank top candidates using fused score
            fused_scores = np.full_like(global_scores, -1e9)
            for idx in top_candidates_idx:
                fused_scores[idx] = (1.0 - effective_w) * global_scores[idx] + effective_w * region_scores[idx]
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose from 'full', 'region', 'combined', 'rerank', 'oracle'.")

        # Mask out invalid category candidates
        fused_scores = np.where(valid_mask, fused_scores, -1e9)

        # Retrieve top_k
        top_indices = np.argsort(fused_scores)[::-1][:top_k]
        t_scoring = (time.time() - t_score_0) * 1000

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if fused_scores[idx] <= -1e8:
                continue
            meta = self.catalog_index.metadata[idx]
            reg_m = self.region_meta[idx] if idx < len(self.region_meta) else {}
            results.append({
                "rank": rank,
                "id": meta.get("id"),
                "category": meta.get("category"),
                "filename": meta.get("filename"),
                "score": float(fused_scores[idx]),
                "global_score": float(global_scores[idx]),
                "region_score": float(region_scores[idx]),
                "cluster_id": meta.get("cluster_id"),
                "region_box": reg_m.get("box", [0, 0, 1, 1])
            })

        t_total = (time.time() - t_start) * 1000

        return {
            "mode": mode,
            "region_weight": effective_w,
            "query_localization": {
                "box": loc_res.box,
                "box_pixels": loc_res.box_pixels,
                "label": loc_res.label,
                "confidence": loc_res.confidence,
                "method": loc_res.method,
                "is_fallback": loc_res.is_fallback
            },
            "total_candidates": len(results),
            "results": results,
            "timings_ms": {
                "localization": round(t_localization, 2),
                "encoding": round(t_encoding, 2),
                "scoring": round(t_scoring, 2),
                "total": round(t_total, 2)
            }
        }
