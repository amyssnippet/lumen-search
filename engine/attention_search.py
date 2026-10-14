"""Soft Spatial Attention Retrieval Engine for Fine Jewellery Visual Search.

Supports:
- Baseline Global Embedding (CLS)
- Uniform Patch Pooling (Config A)
- Foreground Saliency Weighting (Config B)
- Soft Component Region Weighting (Config C)
- Combined Multi-Cue Soft Weighting (Config E)
- Fused Global + Soft-Weighted Patch Representation
- Hard-Crop Retrieval (Negative Control)
- Diagnostic Region Oracle (Upper Bound)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

from engine.embeddings import get_encoder
from engine.index import CatalogIndex
from engine.localization import JewelleryLocalizer, LocalizationResult
from engine.soft_attention import (
    compute_uniform_weights,
    compute_foreground_weights,
    compute_soft_region_weights,
    compute_gaussian_weights,
    compute_combined_weights,
    aggregate_patch_tokens,
    render_attention_heatmap
)

class SoftAttentionSearchEngine:
    """Retrieval engine supporting soft spatial patch weighting and hybrid representation fusion."""

    def __init__(self, model_name: str = "dinov2"):
        self.model_name = model_name.lower()
        self.encoder = get_encoder(self.model_name)
        self.catalog_index = CatalogIndex(model_name=self.model_name)
        self.catalog_index.load()
        self.localizer = JewelleryLocalizer()

        # Load pre-aggregated catalog embedding matrices
        index_dir = Path("data/index")
        self.catalog_matrices: Dict[str, np.ndarray] = {
            "global": self.catalog_index.embeddings,
            "soft_uniform": np.load(index_dir / f"{self.model_name}_soft_uniform.npy"),
            "soft_foreground": np.load(index_dir / f"{self.model_name}_soft_foreground.npy"),
            "soft_region": np.load(index_dir / f"{self.model_name}_soft_region.npy"),
            "soft_combined": np.load(index_dir / f"{self.model_name}_soft_combined.npy"),
        }

        # Load region crops if available for negative control
        reg_emb_path = index_dir / f"{self.model_name}_region_embeddings.npy"
        if reg_emb_path.exists():
            self.catalog_matrices["hard_crop"] = np.load(reg_emb_path)

        import json
        with open(index_dir / "dinov2_meta.json", "r") as fp:
            self.metadata = json.load(fp)

    def extract_query_representation(
        self,
        image: Image.Image,
        config_name: str = "soft_combined",
        category_filter: Optional[str] = None,
        custom_box: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, LocalizationResult]:
        """Extract query feature vector and attention weights under specified configuration."""
        # 1. Localization
        if custom_box is not None:
            w, h = image.size
            y1, x1, y2, x2 = custom_box
            crop = image.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))
            loc_res = LocalizationResult(
                box=custom_box,
                box_pixels=[int(y1*h), int(x1*w), int(y2*h), int(x2*w)],
                label="oracle_ground_truth",
                confidence=1.0,
                method="oracle_ground_truth",
                is_fallback=False,
                crop=crop
            )
        else:
            loc_res = self.localizer.locate(image, category=category_filter)

        cfg_lower = config_name.lower()

        # Global CLS vector
        if cfg_lower == "global":
            q_emb = self.encoder.encode_image(image)
            q_emb = q_emb / np.linalg.norm(q_emb)
            weights = compute_uniform_weights()
            return q_emb, weights, loc_res

        # Hard crop (negative control)
        if cfg_lower == "hard_crop":
            q_emb = self.encoder.encode_image(loc_res.crop)
            q_emb = q_emb / np.linalg.norm(q_emb)
            weights = compute_uniform_weights()
            return q_emb, weights, loc_res

        # Patch extraction
        patch_tokens = self.encoder.get_patch_tokens(image) # (256, D)

        if cfg_lower == "soft_uniform":
            weights = compute_uniform_weights()
        elif cfg_lower == "soft_foreground":
            weights = compute_foreground_weights(image)
        elif cfg_lower == "soft_region":
            weights = compute_soft_region_weights(loc_res.box)
        elif cfg_lower in ("soft_combined", "soft_oracle"):
            weights = compute_combined_weights(image, loc_res.box, is_fallback=loc_res.is_fallback)
        else:
            weights = compute_uniform_weights()

        # Weighted patch aggregation
        z = aggregate_patch_tokens(patch_tokens, weights)
        return z, weights, loc_res

    def search(
        self,
        image: Image.Image,
        config_name: str = "soft_combined",
        alpha: float = 0.5, # Weight of weighted patch vector when fused with global CLS
        fuse_with_global: bool = False,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        custom_box: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Perform visual search under specified soft attention configuration."""
        t_start = time.perf_counter()

        # Extract representation
        t0 = time.perf_counter()
        q_rep, weights, loc_res = self.extract_query_representation(
            image=image,
            config_name=config_name,
            category_filter=category_filter,
            custom_box=custom_box
        )
        t_extract = (time.perf_counter() - t0) * 1000

        # Optional fusion with global CLS embedding
        if fuse_with_global and config_name not in ("global", "hard_crop"):
            q_global = self.encoder.encode_image(image)
            q_global = q_global / np.linalg.norm(q_global)
            fused_rep = (1.0 - alpha) * q_global + alpha * q_rep
            norm = np.linalg.norm(fused_rep)
            q_rep = fused_rep / norm if norm > 0 else q_rep

            # Target matrix is fused combination of global and soft catalog matrices
            cat_target = self.catalog_matrices.get(config_name, self.catalog_matrices["soft_combined"])
            cat_global = self.catalog_matrices["global"]
            cat_embs = (1.0 - alpha) * cat_global + alpha * cat_target
            # Normalize rows
            norms = np.linalg.norm(cat_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            cat_embs = cat_embs / norms
        else:
            cat_embs = self.catalog_matrices.get(config_name, self.catalog_matrices["soft_combined"])

        # Compute cosine similarity
        t0 = time.perf_counter()
        scores = np.dot(cat_embs, q_rep)

        # Apply category filter
        valid_mask = np.ones(len(self.metadata), dtype=bool)
        if category_filter and category_filter.lower() not in ("all", ""):
            cf = category_filter.lower()
            for idx, m in enumerate(self.metadata):
                if m.get("category", "").lower() != cf:
                    valid_mask[idx] = False

        filtered_scores = np.where(valid_mask, scores, -1e9)
        top_indices = np.argsort(filtered_scores)[::-1][:top_k]
        t_scoring = (time.perf_counter() - t0) * 1000

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if filtered_scores[idx] <= -1e8:
                continue
            meta = self.metadata[idx]
            sim = float(scores[idx])
            if sim >= 0.94:
                match_type = "exact_match"
                confidence_label = "Verified Exact SKU / Twin"
            elif sim >= 0.85:
                match_type = "similar_design"
                confidence_label = "Visually Similar Design"
            else:
                match_type = "low_confidence"
                confidence_label = "Low Confidence Candidate"

            results.append({
                "rank": rank,
                "id": meta.get("id"),
                "category": meta.get("category"),
                "filename": meta.get("filename"),
                "score": sim,
                "similarity": round(sim, 4),
                "match_type": match_type,
                "confidence_label": confidence_label,
                "cluster_id": meta.get("cluster_id")
            })

        t_total = (time.perf_counter() - t_start) * 1000

        return {
            "model_name": self.model_name,
            "config_name": config_name,
            "fuse_with_global": fuse_with_global,
            "alpha": alpha,
            "query_localization": {
                "box": loc_res.box,
                "label": loc_res.label,
                "confidence": loc_res.confidence,
                "method": loc_res.method,
                "is_fallback": loc_res.is_fallback
            },
            "attention_weights": weights.tolist(),
            "results": results,
            "timings_ms": {
                "extraction": round(t_extract, 2),
                "scoring": round(t_scoring, 2),
                "total": round(t_total, 2)
            }
        }
