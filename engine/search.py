"""High-level search orchestration engine connecting preprocessing, encoding, and index ranking.

Supports both Qdrant vector database and in-memory NumPy backends with seamless fallback.
"""

import logging
import os
import time
from typing import Dict, Optional, Any, Union
from PIL import Image

from engine.embeddings import BaseEncoder, get_encoder
from engine.index import CatalogIndex
from engine.preprocessing import PreprocessConfig, preprocess_query_image, QualityReport, extract_robust_query_crops

logger = logging.getLogger(__name__)


class SearchEngine:
    """Production visual search engine supporting Qdrant vector DB and NumPy fallback."""

    def __init__(
        self,
        model_name: str = "mobilenetv4",
        backend: str = "qdrant",  # "qdrant" or "numpy"
        index_dir: str = "data/index",
        device: str = "cpu",
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        canary_router: Optional[Any] = None
    ):
        self.model_name = model_name.lower()
        self.backend = backend.lower()
        self.encoder: BaseEncoder = get_encoder(self.model_name, device=device)
        self.index_dir = index_dir
        self.qdrant_host = qdrant_host if qdrant_host is not None else os.environ.get("QDRANT_HOST", "127.0.0.1")
        self.qdrant_port = qdrant_port if qdrant_port is not None else int(os.environ.get("QDRANT_PORT", "6333"))
        self.canary_router = canary_router

        self.index: Union[CatalogIndex, Any] = None
        self._init_index()

    def _init_index(self):
        """Initialize selected backend with graceful fallback."""
        if self.backend == "qdrant":
            try:
                from engine.qdrant_index import QdrantCatalogIndex
                # Attempt to bind to production alias if model is mobilenetv4
                if self.model_name.startswith("mobilenetv4"):
                    prod_alias = os.environ.get("QDRANT_PRODUCTION_ALIAS", "jewellery_catalog_mobilenetv4")
                    q_idx = QdrantCatalogIndex(
                        model_name=self.model_name,
                        host=self.qdrant_host,
                        port=self.qdrant_port,
                        collection_name=prod_alias
                    )
                    if q_idx.is_online and q_idx.is_indexed() and q_idx.embedding_dim == self.encoder.embedding_dim:
                        self.index = q_idx
                        self.backend_active = "qdrant"
                        logger.info(f"Initialized Qdrant backend for '{self.model_name}' using alias '{prod_alias}' ({q_idx.size} items).")
                        return

                # For other models or fallback, try model-specific collection
                q_idx_fallback = QdrantCatalogIndex(
                    model_name=self.model_name,
                    host=self.qdrant_host,
                    port=self.qdrant_port
                )
                if q_idx_fallback.is_online and q_idx_fallback.is_indexed() and q_idx_fallback.embedding_dim == self.encoder.embedding_dim:
                    self.index = q_idx_fallback
                    self.backend_active = "qdrant"
                    logger.info(f"Initialized Qdrant backend for '{self.model_name}' ({q_idx_fallback.size} items).")
                    return
                logger.warning(f"Qdrant collection for '{self.model_name}' not ready or dimension mismatch. Falling back to NumPy.")
            except Exception as e:
                logger.warning(f"Qdrant initialization error ({e}). Falling back to NumPy.")

        # NumPy index fallback or explicit selection
        self.index = CatalogIndex(model_name=self.model_name, index_dir=self.index_dir)
        self.backend_active = "numpy"
        if self.index.is_indexed():
            self.index.load()
        else:
            self.index.build_from_manifest(encoder=self.encoder)

    def search_image(
        self,
        image: Image.Image,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        search_mode: str = "similar",  # "exact" or "similar"
        preprocess_config: Optional[PreprocessConfig] = None,
        exclude_ids: Optional[set] = None,
        ef_search: int = 128,
        exact: bool = False,
        aggregate_sku: bool = False,
        tenant_id: Optional[str] = None,
        robust_mode: bool = False
    ) -> Dict[str, Any]:
        """Search catalog with an input PIL query image.
        
        Supports robust_mode with adaptive skin-neutralized foreground isolation,
        test-time multi-crop max-pooling, and calibrated confidence thresholds.
        """
        t_start = time.time()

        # Preprocessing & Quality Assessment
        t0_prep = time.time()
        processed_img, quality_report, prep_meta = preprocess_query_image(image, config=preprocess_config)
        prep_time_ms = round((time.time() - t0_prep) * 1000, 2)

        # Feature Extraction (Single or Multi-Crop)
        t0_enc = time.time()
        use_multi_crop = robust_mode
        if use_multi_crop:
            crop_tuples = extract_robust_query_crops(image, config=preprocess_config)
            crop_vectors = [self.encoder.encode_image(c) for c, _ in crop_tuples]
            crop_names = [label for _, label in crop_tuples]
            query_vec = crop_vectors[0]
        else:
            query_vec = self.encoder.encode_image(processed_img)
            crop_vectors = [query_vec]
            crop_names = ["global"]
        enc_time_ms = round((time.time() - t0_enc) * 1000, 2)

        # Vector Index Search & Routing Decision
        t0_search = time.time()
        routing_info = {"target": "production", "fallback": False}
        chosen_route = "prod"
        index_to_query = self.index

        if self.backend_active == "qdrant" and self.canary_router is not None:
            effective_tenant = tenant_id or "default_tenant"
            from engine.index_router import RouteTarget
            target, reason = self.canary_router.resolve_target(effective_tenant)
            if target == RouteTarget.CANDIDATE:
                candidate_idx = self.canary_router.get_candidate_index()
                if candidate_idx and candidate_idx.is_online and candidate_idx.is_indexed():
                    index_to_query = candidate_idx
                    chosen_route = "canary"
                    routing_info = {
                        "target": "candidate",
                        "collection": candidate_idx.collection_name,
                        "fallback": False
                    }
                else:
                    try:
                        from app.metrics import record_canary_fallback
                        record_canary_fallback("candidate_unavailable")
                    except Exception:
                        pass
                    routing_info = {
                        "target": "production",
                        "fallback": True,
                        "fallback_reason": "candidate_index_unavailable"
                    }

        norm_cat = category_filter.lower().rstrip("s") if category_filter and category_filter not in ("all", "none", "") else None

        def _execute_single_query(idx_target, vec, limit):
            if self.backend_active == "qdrant":
                return idx_target.search(
                    query_embedding=vec,
                    top_k=limit,
                    category_filter=norm_cat,
                    filters=filters,
                    exclude_ids=exclude_ids,
                    ef_search=ef_search,
                    exact=exact,
                    aggregate_sku=False,
                    tenant_id=tenant_id
                )
            else:
                return idx_target.search(
                    query_embedding=vec,
                    top_k=limit,
                    category_filter=norm_cat,
                    exclude_ids=exclude_ids,
                    aggregate_sku=False
                )

        if len(crop_vectors) > 1:
            query_depth = max(top_k * 3, 30)
            crop_candidate_maps = []
            try:
                if self.backend_active == "qdrant" and hasattr(index_to_query, "search_batch"):
                    crop_candidate_maps = index_to_query.search_batch(
                        query_embeddings=crop_vectors,
                        top_k=query_depth,
                        category_filter=norm_cat,
                        filters=filters,
                        exclude_ids=exclude_ids,
                        ef_search=ef_search,
                        exact=exact,
                        tenant_id=tenant_id
                    )
                else:
                    for cv in crop_vectors:
                        sub_res = _execute_single_query(index_to_query, cv, query_depth)
                        crop_candidate_maps.append(sub_res.get("results", []))
            except Exception as e:
                fallback_enabled = getattr(self.canary_router, "_config", None) and self.canary_router._config.fallback_to_production
                if chosen_route == "canary" and fallback_enabled:
                    logger.warning(f"Canary search failed ({e}); falling back to production index.")
                    crop_candidate_maps = []
                    if hasattr(self.index, "search_batch"):
                        crop_candidate_maps = self.index.search_batch(
                            query_embeddings=crop_vectors,
                            top_k=query_depth,
                            category_filter=norm_cat,
                            filters=filters,
                            exclude_ids=exclude_ids,
                            ef_search=ef_search,
                            exact=exact,
                            tenant_id=tenant_id
                        )
                    else:
                        for cv in crop_vectors:
                            sub_res = _execute_single_query(self.index, cv, query_depth)
                            crop_candidate_maps.append(sub_res.get("results", []))
                    chosen_route = "prod"
                    routing_info = {
                        "target": "production",
                        "fallback": True,
                        "fallback_reason": str(e)
                    }
                else:
                    raise

            # Candidate fusion logic with skin/background awareness
            skin_is_detected = getattr(quality_report, "skin_detected", False)
            fused_candidates_dict = {}

            for crop_idx, cand_list in enumerate(crop_candidate_maps):
                crop_label = crop_names[crop_idx] if crop_idx < len(crop_names) else f"crop_{crop_idx}"
                # If skin is detected, suppress global crop background bias
                weight = 0.82 if (skin_is_detected and crop_label == "global") else 1.0

                for cand in cand_list:
                    cid = cand["id"]
                    adj_sim = round(cand["similarity"] * weight, 4) if weight != 1.0 else cand["similarity"]
                    if cid not in fused_candidates_dict or adj_sim > fused_candidates_dict[cid]["similarity"]:
                        c_copy = dict(cand)
                        c_copy["similarity"] = adj_sim
                        c_copy["best_crop"] = crop_label
                        fused_candidates_dict[cid] = c_copy

            fused_candidates = list(fused_candidates_dict.values())
            # Strict defensive category post-filter
            if norm_cat:
                fused_candidates = [c for c in fused_candidates if c.get("category", "").lower().rstrip("s") == norm_cat]

            fused_candidates.sort(key=lambda x: x["similarity"], reverse=True)

            if aggregate_sku:
                sku_groups = {}
                for cand in fused_candidates:
                    sku_key = cand.get("sku_id") or cand["id"]
                    if sku_key not in sku_groups:
                        sku_groups[sku_key] = []
                    sku_groups[sku_key].append(cand)
                aggregated_results = []
                for sku_key, views in sku_groups.items():
                    best_view = max(views, key=lambda x: x["similarity"])
                    entry = dict(best_view)
                    entry["matched_views_count"] = len(views)
                    entry["matched_view_ids"] = [v["id"] for v in views]
                    entry["matched_view_angles"] = [v.get("view_angle", "unknown") for v in views]
                    aggregated_results.append(entry)
                aggregated_results.sort(key=lambda x: x["similarity"], reverse=True)
                final_results = aggregated_results[:top_k]
            else:
                final_results = fused_candidates[:top_k]

            for rank, r in enumerate(final_results, start=1):
                r["rank"] = rank

            top_sim = final_results[0]["similarity"] if final_results else 0.0
            search_res = {
                "total_candidates": len(fused_candidates),
                "has_reliable_match": len(final_results) > 0 and top_sim >= 0.70,
                "top_similarity": top_sim,
                "results": final_results
            }
        else:
            # Single-vector standard path
            if self.backend_active == "qdrant":
                try:
                    search_res = index_to_query.search(
                        query_embedding=query_vec,
                        top_k=top_k,
                        category_filter=norm_cat,
                        filters=filters,
                        exclude_ids=exclude_ids,
                        ef_search=ef_search,
                        exact=exact,
                        aggregate_sku=aggregate_sku,
                        tenant_id=tenant_id
                    )
                except Exception as e:
                    fallback_enabled = getattr(self.canary_router, "_config", None) and self.canary_router._config.fallback_to_production
                    if chosen_route == "canary" and fallback_enabled:
                        try:
                            from app.metrics import record_canary_fallback
                            record_canary_fallback("candidate_error")
                        except Exception:
                            pass
                        logger.warning(f"Canary search failed ({e}); falling back to production index.")
                        search_res = self.index.search(
                            query_embedding=query_vec,
                            top_k=top_k,
                            category_filter=norm_cat,
                            filters=filters,
                            exclude_ids=exclude_ids,
                            ef_search=ef_search,
                            exact=exact,
                            aggregate_sku=aggregate_sku,
                            tenant_id=tenant_id
                        )
                        chosen_route = "prod"
                        routing_info = {
                            "target": "production",
                            "fallback": True,
                            "fallback_reason": str(e)
                        }
                    else:
                        try:
                            from app.metrics import record_search_routing
                            record_search_routing(route=chosen_route, success=False)
                        except Exception:
                            pass
                        raise
            else:
                search_res = self.index.search(
                    query_embedding=query_vec,
                    top_k=top_k,
                    category_filter=norm_cat,
                    exclude_ids=exclude_ids,
                    aggregate_sku=aggregate_sku
                )

        search_time_ms = round((time.time() - t0_search) * 1000, 2)
        total_time_ms = round((time.time() - t_start) * 1000, 2)

        # Record route-specific metrics
        try:
            from app.metrics import record_search_routing, record_routing_stage_latency
            record_search_routing(route=chosen_route, success=True)
            record_routing_stage_latency(route=chosen_route, stage="preprocessing", duration_sec=prep_time_ms / 1000.0)
            record_routing_stage_latency(route=chosen_route, stage="encoding", duration_sec=enc_time_ms / 1000.0)
            record_routing_stage_latency(route=chosen_route, stage="search", duration_sec=search_time_ms / 1000.0)
            record_routing_stage_latency(route=chosen_route, stage="overall", duration_sec=total_time_ms / 1000.0)
        except Exception:
            pass

        results = search_res["results"]
        # Defensive post-filter on final results
        if norm_cat:
            results = [r for r in results if r.get("category", "").lower().rstrip("s") == norm_cat]

        top_sim = results[0]["similarity"] if results else 0.0

        # Calibrate match types and decision semantics
        # Decisions:
        # EXACT_MATCH: top_sim >= 0.88
        # VISUAL_SIMILARITY: 0.52 <= top_sim < 0.88
        # NO_CONFIDENT_MATCH: top_sim < 0.52 or no results
        if not results or top_sim < 0.52:
            decision = "NO_CONFIDENT_MATCH"
            has_reliable_match = False
            decision_explanation = "No catalogue candidates met the minimum visual similarity threshold (52.0%)."
        elif top_sim >= 0.88:
            decision = "EXACT_MATCH"
            has_reliable_match = True
            decision_explanation = "Candidate satisfies verified exact-match criteria (>= 88.0%)."
        else:
            decision = "VISUAL_SIMILARITY"
            has_reliable_match = True
            decision_explanation = "Retrieved candidates represent visual style similarities (heuristic score band)."

        for r in results:
            sim = r["similarity"]
            r["exact_match_candidate"] = sim >= 0.88
            if sim >= 0.88:
                r["match_type"] = "exact_match"
                r["confidence_label"] = "Verified Exact SKU / Twin"
            elif sim >= 0.70:
                r["match_type"] = "high_confidence"
                r["confidence_label"] = "High Confidence Candidate"
            elif sim >= 0.52:
                r["match_type"] = "similar_design"
                r["confidence_label"] = "Visually Similar Design"
            else:
                r["match_type"] = "low_confidence"
                r["confidence_label"] = "Low Confidence Candidate"

        detected_category = results[0].get("category") if results else None

        return {
            "model_name": self.model_name,
            "backend": self.backend_active,
            "search_mode": search_mode,
            "category_filter": category_filter,
            "category_filter_applied": norm_cat,
            "detected_category": detected_category,
            "decision": decision,
            "decision_explanation": decision_explanation,
            "robust_mode": robust_mode,
            "timing_ms": {
                "preprocessing": prep_time_ms,
                "encoding": enc_time_ms,
                "search": search_time_ms,
                "total": total_time_ms
            },
            "quality": {
                "is_blurry": quality_report.is_blurry,
                "sharpness_score": quality_report.sharpness_score,
                "mean_luminance": quality_report.mean_luminance,
                "is_underexposed": quality_report.is_underexposed,
                "is_overexposed": quality_report.is_overexposed,
                "is_low_res": quality_report.is_low_res,
                "skin_detected": getattr(quality_report, "skin_detected", False),
                "skin_ratio": getattr(quality_report, "skin_ratio", 0.0),
                "warnings": quality_report.warnings,
                "applied_steps": prep_meta["applied_steps"],
                "robust_crops": crop_names if use_multi_crop else ["global"]
            },
            "has_reliable_match": has_reliable_match,
            "top_similarity": top_sim,
            "total_candidates": search_res["total_candidates"],
            "results": results,
            "routing": routing_info
        }
