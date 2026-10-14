"""Hybrid candidate retrieval (DINOv2 + DINOv3 RRF) and local patch Chamfer reranker."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

from engine.embeddings import get_encoder
from engine.index import CatalogIndex, EXACT_MATCH_THRESHOLD, SIMILARITY_THRESHOLD
from engine.preprocessing import PreprocessConfig, preprocess_query_image
from engine.reranker import PatchCatalogIndex, get_foreground_patch_indices

class HybridSearchEngine:
    def __init__(self, index_dir: str = "data/index", device: str = "cpu"):
        self.device = device
        self.index_dir = Path(index_dir)
        
        # Encoders
        self.v2_encoder = get_encoder("dinov2", device=device)
        self.v3_encoder = get_encoder("dinov3", device=device)
        
        # Global Indices
        self.v2_index = CatalogIndex("dinov2", index_dir=index_dir)
        self.v2_index.load()
        
        self.v3_index = CatalogIndex("dinov3", index_dir=index_dir)
        self.v3_index.load()
        
        # Patch Index for Reranking
        self.patch_index = PatchCatalogIndex("dinov3", index_dir=index_dir)
        self.patch_index.load()
        
        # Fast lookup from item ID to catalog integer index
        self._id_to_cat_idx = {item["id"]: idx for idx, item in enumerate(self.v3_index.metadata)}

    def search_hybrid(
        self,
        image: Image.Image,
        top_k: int = 10,
        candidate_depth: int = 20,
        enable_rerank: bool = True,
        patch_weight: float = 0.2,
        spatial_weight: float = 0.0,
        category_filter: Optional[str] = None,
        search_mode: str = "similar",
        preprocess_config: Optional[PreprocessConfig] = None,
        exclude_ids: Optional[set] = None,
        exact_match_threshold: float = 0.955
    ) -> Dict[str, Any]:
        """Execute hybrid search: DINOv2 + DINOv3 candidate fusion followed by local patch Chamfer reranking.
        
        Formula:
            final_score = (1 - patch_weight) * normalized_global_score + patch_weight * chamfer_score
        """
        t_start = time.time()
        
        # 1. Preprocess & Query Enhancement
        t0_prep = time.time()
        processed_img, quality_rep, prep_meta = preprocess_query_image(image, config=preprocess_config)
        prep_time_ms = round((time.time() - t0_prep) * 1000, 2)
        
        # 2. DINOv2 Inference
        t0_v2 = time.time()
        v2_query_img = processed_img.resize((224, 224), Image.Resampling.BICUBIC)
        v2_vec = self.v2_encoder.encode_image(v2_query_img)
        v2_time_ms = round((time.time() - t0_v2) * 1000, 2)

        # 3. DINOv3 Inference
        t0_v3 = time.time()
        v3_query_img = processed_img.resize((256, 256), Image.Resampling.BICUBIC)
        v3_vec = self.v3_encoder.encode_image(v3_query_img)
        v3_time_ms = round((time.time() - t0_v3) * 1000, 2)
        
        # 4. Global Retrieval & Candidate Fusion
        t0_global = time.time()
        r_v2 = self.v2_index.search(
            v2_vec,
            top_k=candidate_depth,
            category_filter=category_filter,
            exclude_ids=exclude_ids
        )["results"]
        
        r_v3 = self.v3_index.search(
            v3_vec,
            top_k=candidate_depth,
            category_filter=category_filter,
            exclude_ids=exclude_ids
        )["results"]
        
        # Collect raw cosine similarities
        v2_sims = {item["id"]: float(item["similarity"]) for item in r_v2}
        v3_sims = {item["id"]: float(item["similarity"]) for item in r_v3}

        # Reciprocal Rank Fusion (RRF) for candidate ordering
        rrf_scores = {}
        cand_dict = {}
        
        for rank, item in enumerate(r_v2, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank))
            cand_dict[cid] = item
            
        for rank, item in enumerate(r_v3, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank))
            if cid not in cand_dict:
                cand_dict[cid] = item
                
        # Sorted candidate pool
        sorted_cand_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:candidate_depth]
        
        # Normalized global score: average cosine similarity of dual representations
        # Both v2 and v3 similarities are cosine similarities in [-1, 1], geometrically compatible with Chamfer
        norm_global_scores = []
        for cid in sorted_cand_ids:
            if cid in v2_sims and cid in v3_sims:
                g_score = 0.5 * (v2_sims[cid] + v3_sims[cid])
            elif cid in v3_sims:
                g_score = v3_sims[cid]
            elif cid in v2_sims:
                g_score = v2_sims[cid]
            else:
                g_score = 0.70
            norm_global_scores.append(float(g_score))

        global_time_ms = round((time.time() - t0_global) * 1000, 2)
        
        # 5. Patch Extraction & Local Feature Reranking
        t0_patch_ext = time.time()
        patch_ext_time_ms = 0.0
        rerank_time_ms = 0.0
        
        final_results = []
        if enable_rerank and sorted_cand_ids:
            query_patches = self.v3_encoder.get_patch_tokens(v3_query_img) # (256, 384)
            patch_ext_time_ms = round((time.time() - t0_patch_ext) * 1000, 2)

            t0_rerank = time.time()
            fg_indices = get_foreground_patch_indices(v3_query_img)
            cand_cat_indices = [self._id_to_cat_idx[cid] for cid in sorted_cand_ids]
            
            reranked_tuples = self.patch_index.rerank(
                query_patches=query_patches,
                candidate_catalog_indices=cand_cat_indices,
                candidate_global_scores=norm_global_scores,
                query_fg_indices=fg_indices,
                patch_weight=patch_weight,
                spatial_weight=spatial_weight
            )
            
            for rank, (cat_idx, fused_score, ch_score) in enumerate(reranked_tuples[:top_k], start=1):
                raw_item = self.v3_index.metadata[cat_idx]
                sim = round(float(fused_score), 4)
                
                # Confidence qualification (never claim verified identity without ground truth)
                if sim >= exact_match_threshold:
                    match_type = "high_confidence_match"
                    conf_label = "High-Confidence Candidate Match"
                elif sim >= 0.75:
                    match_type = "similar_design"
                    conf_label = "Visually Similar Candidate"
                else:
                    match_type = "low_confidence"
                    conf_label = "Low Confidence Candidate"
                    
                final_results.append({
                    "rank": rank,
                    "id": raw_item["id"],
                    "filename": raw_item["filename"],
                    "category": raw_item["category"],
                    "cluster_id": raw_item["cluster_id"],
                    "similarity": sim,
                    "global_score": round(norm_global_scores[sorted_cand_ids.index(raw_item["id"])], 4),
                    "chamfer_score": round(float(ch_score), 4),
                    "match_type": match_type,
                    "confidence_label": conf_label,
                    "width": raw_item["width"],
                    "height": raw_item["height"]
                })
            rerank_time_ms = round((time.time() - t0_rerank) * 1000, 2)
        else:
            # RRF ranking without patch reranking
            for rank, (cid, g_score) in enumerate(zip(sorted_cand_ids[:top_k], norm_global_scores[:top_k]), start=1):
                item = cand_dict[cid]
                sim = round(float(g_score), 4)
                if sim >= exact_match_threshold:
                    match_type = "high_confidence_match"
                    conf_label = "High-Confidence Candidate Match"
                elif sim >= 0.75:
                    match_type = "similar_design"
                    conf_label = "Visually Similar Candidate"
                else:
                    match_type = "low_confidence"
                    conf_label = "Low Confidence Candidate"

                final_results.append({
                    "rank": rank,
                    "id": item["id"],
                    "filename": item["filename"],
                    "category": item["category"],
                    "cluster_id": item["cluster_id"],
                    "similarity": sim,
                    "global_score": sim,
                    "chamfer_score": 0.0,
                    "match_type": match_type,
                    "confidence_label": conf_label,
                    "width": item["width"],
                    "height": item["height"]
                })

        # Apply search mode semantics
        exact_match_found = False
        if search_mode == "exact":
            # In exact mode, only return candidates meeting the calibrated exact threshold
            exact_candidates = [r for r in final_results if r["similarity"] >= exact_match_threshold]
            exact_match_found = len(exact_candidates) > 0
            final_results = exact_candidates

        total_time_ms = round((time.time() - t_start) * 1000, 2)
        
        return {
            "model_name": "hybrid_rerank" if enable_rerank else "hybrid",
            "search_mode": search_mode,
            "category_filter": category_filter,
            "candidate_depth": candidate_depth,
            "patch_weight": patch_weight,
            "spatial_weight": spatial_weight,
            "exact_match_threshold": exact_match_threshold,
            "exact_match_found": exact_match_found if search_mode == "exact" else (len(final_results) > 0 and final_results[0]["similarity"] >= exact_match_threshold),
            "timing_ms": {
                "file_decoding_and_normalization": 0.0,
                "query_enhancement": prep_time_ms,
                "dinov2_inference": v2_time_ms,
                "dinov3_inference": v3_time_ms,
                "global_candidate_generation": global_time_ms,
                "patch_extraction": patch_ext_time_ms,
                "patch_reranking": rerank_time_ms,
                "total": total_time_ms
            },
            "quality": {
                "is_blurry": quality_rep.is_blurry,
                "sharpness_score": quality_rep.sharpness_score,
                "mean_luminance": quality_rep.mean_luminance,
                "warnings": quality_rep.warnings,
                "applied_steps": prep_meta["applied_steps"]
            },
            "has_reliable_match": len(final_results) > 0 and final_results[0]["similarity"] >= 0.75,
            "top_similarity": final_results[0]["similarity"] if final_results else 0.0,
            "results": final_results
        }
