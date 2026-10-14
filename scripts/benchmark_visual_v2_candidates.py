#!/usr/bin/env python3
"""Systematic Empirical Benchmark for POC Fine-Grained Visual Jewellery Retrieval v2.

Compares:
1. baseline_mobilenetv4: Single-crop MobileNetV4 Conv-Small FP32 ONNX (1280-d) [Current Baseline].
2. robust_mobilenetv4: 3-crop adaptive skin-suppressed MobileNetV4 [PH-014 Candidate].
3. dinov3_global_int8: Single-crop DINOv3 ViT-S/16 INT8 ONNX (384-d) [Pure Vision Transformer].
4. dinov3_robust_multicrop: 3-crop adaptive skin-suppressed DINOv3 INT8 ONNX.
5. dinov3_patch_rerank: DINOv3 candidate retrieval + local spatial patch Chamfer reranking.
6. ensemble_v2: Normalized Rank-Sum Fusion of MobileNetV4 + DINOv3 Multicrop.

Evaluates all 122 queries across all 6 slices of data/evaluation/ph014_evaluation_manifest.json.
Outputs: output/retrieval_v2_benchmark_results.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.embeddings import MobileNetV4Encoder, DINOv3ONNXEncoder, DINOv3Encoder
from engine.preprocessing import PreprocessConfig, preprocess_query_image, extract_robust_query_crops
from engine.reranker import PatchCatalogIndex, get_foreground_patch_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_v2")


class VisualV2BenchmarkRunner:
    def __init__(self):
        logger.info("Initializing models and indexes...")
        self.mob_encoder = MobileNetV4Encoder()
        self.dino_int8_encoder = DINOv3ONNXEncoder("data/models/dinov3_vits16_int8.onnx")
        
        # Load catalog metadata and embeddings
        with open(REPO_ROOT / "data" / "index" / "mobilenetv4_meta.json", "r") as f:
            self.catalog_meta = json.load(f)
        self.mob_embs = np.load(REPO_ROOT / "data" / "index" / "mobilenetv4_embeddings.npy")
        self.dino_embs = np.load(REPO_ROOT / "data" / "index" / "dinov3_int8_embeddings.npy")
        
        self.patch_index = PatchCatalogIndex(model_name="dinov3", index_dir=str(REPO_ROOT / "data" / "index"))
        self.patch_index.load()
        
        self.dino_vit = None # Lazy load only if patch extraction is needed

        # Mapping
        self.id_to_idx = {item["id"]: idx for idx, item in enumerate(self.catalog_meta)}
        self.item_to_sku = {
            item["id"]: item.get("sku_id") or f"SKU-{item.get('category', 'RING').upper()[:4]}-{item['id']}"
            for item in self.catalog_meta
        }

    def _get_dino_vit(self):
        if self.dino_vit is None:
            self.dino_vit = DINOv3Encoder(device="cpu")
        return self.dino_vit

    def run_candidate(self, candidate_name: str, img_path: Path, category_filter: str) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.time()
        img = Image.open(img_path).convert("RGB")
        norm_cat = category_filter.lower().rstrip("s") if category_filter else None

        if candidate_name == "baseline_mobilenetv4":
            config = PreprocessConfig(target_size=224)
            prep_img, _, _ = preprocess_query_image(img, config)
            vec = self.mob_encoder.encode_image(prep_img)
            sims = np.dot(self.mob_embs, vec)
            ranked_indices = np.argsort(-sims)
            candidates = []
            for idx in ranked_indices:
                cat = self.catalog_meta[idx].get("category", "")
                if norm_cat and cat.lower().rstrip("s") != norm_cat:
                    continue
                candidates.append({
                    "id": self.catalog_meta[idx]["id"],
                    "sku": self.item_to_sku[self.catalog_meta[idx]["id"]],
                    "category": cat,
                    "similarity": float(sims[idx])
                })

        elif candidate_name == "robust_mobilenetv4":
            crops = extract_robust_query_crops(img)
            _, quality, _ = preprocess_query_image(img)
            skin_detected = getattr(quality, "skin_detected", False)
            crop_sims = []
            for crop_img, crop_name in crops:
                vec = self.mob_encoder.encode_image(crop_img)
                sims = np.dot(self.mob_embs, vec)
                weight = 0.82 if (skin_detected and crop_name == "global") else 1.0
                crop_sims.append(sims * weight)
            fused_sims = np.maximum.reduce(crop_sims)
            ranked_indices = np.argsort(-fused_sims)
            candidates = []
            for idx in ranked_indices:
                cat = self.catalog_meta[idx].get("category", "")
                if norm_cat and cat.lower().rstrip("s") != norm_cat:
                    continue
                candidates.append({
                    "id": self.catalog_meta[idx]["id"],
                    "sku": self.item_to_sku[self.catalog_meta[idx]["id"]],
                    "category": cat,
                    "similarity": float(fused_sims[idx])
                })

        elif candidate_name == "dinov3_global_int8":
            config = PreprocessConfig(target_size=256)
            prep_img, _, _ = preprocess_query_image(img, config)
            vec = self.dino_int8_encoder.encode_image(prep_img)
            sims = np.dot(self.dino_embs, vec)
            ranked_indices = np.argsort(-sims)
            candidates = []
            for idx in ranked_indices:
                cat = self.catalog_meta[idx].get("category", "")
                if norm_cat and cat.lower().rstrip("s") != norm_cat:
                    continue
                candidates.append({
                    "id": self.catalog_meta[idx]["id"],
                    "sku": self.item_to_sku[self.catalog_meta[idx]["id"]],
                    "category": cat,
                    "similarity": float(sims[idx])
                })

        elif candidate_name == "dinov3_robust_multicrop":
            config = PreprocessConfig(target_size=256)
            crops = extract_robust_query_crops(img, config=config)
            _, quality, _ = preprocess_query_image(img, config=config)
            skin_detected = getattr(quality, "skin_detected", False)
            crop_sims = []
            for crop_img, crop_name in crops:
                vec = self.dino_int8_encoder.encode_image(crop_img)
                sims = np.dot(self.dino_embs, vec)
                weight = 0.85 if (skin_detected and crop_name == "global") else 1.0
                crop_sims.append(sims * weight)
            fused_sims = np.maximum.reduce(crop_sims)
            ranked_indices = np.argsort(-fused_sims)
            candidates = []
            for idx in ranked_indices:
                cat = self.catalog_meta[idx].get("category", "")
                if norm_cat and cat.lower().rstrip("s") != norm_cat:
                    continue
                candidates.append({
                    "id": self.catalog_meta[idx]["id"],
                    "sku": self.item_to_sku[self.catalog_meta[idx]["id"]],
                    "category": cat,
                    "similarity": float(fused_sims[idx])
                })

        elif candidate_name == "dinov3_patch_rerank":
            # 1. Global DINOv3 candidate retrieval (top-20)
            config = PreprocessConfig(target_size=256)
            prep_img, _, _ = preprocess_query_image(img, config)
            vec = self.dino_int8_encoder.encode_image(prep_img)
            sims = np.dot(self.dino_embs, vec)
            ranked_indices = [idx for idx in np.argsort(-sims) if (not norm_cat or self.catalog_meta[idx].get("category", "").lower().rstrip("s") == norm_cat)]
            
            top20_indices = ranked_indices[:20]
            top20_scores = [float(sims[i]) for i in top20_indices]
            
            vit = self._get_dino_vit()
            q_patches = vit.get_patch_tokens(prep_img)
            q_fg = get_foreground_patch_indices(prep_img)
            
            reranked = self.patch_index.rerank(
                query_patches=q_patches,
                candidate_catalog_indices=top20_indices,
                candidate_global_scores=top20_scores,
                query_fg_indices=q_fg,
                patch_weight=0.20,
                spatial_weight=0.02
            )
            
            reranked_order = [c_idx for c_idx, _, _ in reranked]
            full_order = reranked_order + ranked_indices[20:]
            candidates = []
            rerank_dict = {c_idx: (f_score, ch_score) for c_idx, f_score, ch_score in reranked}
            for idx in full_order:
                cat = self.catalog_meta[idx].get("category", "")
                sim_val = rerank_dict[idx][0] if idx in rerank_dict else float(sims[idx])
                candidates.append({
                    "id": self.catalog_meta[idx]["id"],
                    "sku": self.item_to_sku[self.catalog_meta[idx]["id"]],
                    "category": cat,
                    "similarity": sim_val
                })

        elif candidate_name == "ensemble_v2":
            # Fusion: MobileNetV4 Multicrop + DINOv3 Multicrop (Normalized score blend 0.40 MobileNet + 0.60 DINOv3)
            # MobileNetV4 multicrop
            mob_crops = extract_robust_query_crops(img, config=PreprocessConfig(target_size=224))
            _, mob_quality, _ = preprocess_query_image(img, config=PreprocessConfig(target_size=224))
            skin_mob = getattr(mob_quality, "skin_detected", False)
            mob_crop_sims = []
            for crop_img, crop_name in mob_crops:
                vec = self.mob_encoder.encode_image(crop_img)
                s = np.dot(self.mob_embs, vec)
                w = 0.82 if (skin_mob and crop_name == "global") else 1.0
                mob_crop_sims.append(s * w)
            mob_fused = np.maximum.reduce(mob_crop_sims)
            
            # DINOv3 multicrop
            dino_crops = extract_robust_query_crops(img, config=PreprocessConfig(target_size=256))
            dino_crop_sims = []
            for crop_img, crop_name in dino_crops:
                vec = self.dino_int8_encoder.encode_image(crop_img)
                s = np.dot(self.dino_embs, vec)
                w = 0.85 if (skin_mob and crop_name == "global") else 1.0
                dino_crop_sims.append(s * w)
            dino_fused = np.maximum.reduce(dino_crop_sims)

            # Linear blend of calibrated scores: 40% MobileNetV4 + 60% DINOv3
            ensemble_sims = 0.40 * mob_fused + 0.60 * dino_fused
            ranked_indices = np.argsort(-ensemble_sims)
            candidates = []
            for idx in ranked_indices:
                cat = self.catalog_meta[idx].get("category", "")
                if norm_cat and cat.lower().rstrip("s") != norm_cat:
                    continue
                candidates.append({
                    "id": self.catalog_meta[idx]["id"],
                    "sku": self.item_to_sku[self.catalog_meta[idx]["id"]],
                    "category": cat,
                    "similarity": float(ensemble_sims[idx])
                })
        else:
            raise ValueError(f"Unknown candidate: {candidate_name}")

        latency_ms = (time.time() - t0) * 1000
        return candidates, latency_ms

    def evaluate_manifest(self, candidate_name: str, manifest_path: Path) -> Dict[str, Any]:
        with open(manifest_path) as f:
            manifest = json.load(f)

        slice_metrics = {}
        all_positive_ranks = []
        all_latencies = []

        for slice_name, queries in manifest["slices"].items():
            logger.info(f"Evaluating '{candidate_name}' on {slice_name} ({len(queries)} queries)...")

            if slice_name == "slice_6_hard_negatives":
                false_accepts_88 = 0
                false_accepts_70 = 0
                false_accepts_52 = 0
                max_sims = []
                latencies = []

                for q in queries:
                    q_path = REPO_ROOT / q["query_path"]
                    cands, lat = self.run_candidate(candidate_name, q_path, category_filter=q["category"])
                    latencies.append(lat)

                    target_id = q.get("negative_target_id") or q.get("target_catalog_id")
                    sim_to_target = 0.0
                    for c in cands:
                        if c["id"] == target_id:
                            sim_to_target = c["similarity"]
                            break

                    max_sims.append(sim_to_target)
                    if sim_to_target >= 0.88:
                        false_accepts_88 += 1
                    if sim_to_target >= 0.70:
                        false_accepts_70 += 1
                    if sim_to_target >= 0.52:
                        false_accepts_52 += 1

                n_neg = len(queries)
                slice_metrics[slice_name] = {
                    "total_pairs": n_neg,
                    "far_exact_88": round(false_accepts_88 / n_neg, 4),
                    "far_high_70": round(false_accepts_70 / n_neg, 4),
                    "far_similar_52": round(false_accepts_52 / n_neg, 4),
                    "max_negative_similarity": round(float(np.max(max_sims)), 4) if max_sims else 0.0,
                    "mean_negative_similarity": round(float(np.mean(max_sims)), 4) if max_sims else 0.0,
                    "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                    "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2)
                }
            else:
                rec1 = 0
                rec5 = 0
                rec20 = 0
                rec_rr = []
                positive_coverage_70 = 0
                positive_coverage_88 = 0
                latencies = []

                for q in queries:
                    q_path = REPO_ROOT / q["query_path"]
                    cands, lat = self.run_candidate(candidate_name, q_path, category_filter=q["category"])
                    latencies.append(lat)
                    all_latencies.append(lat)

                    target_id = q.get("target_catalog_id")
                    target_sku = q.get("target_sku")

                    rank = None
                    target_sim = 0.0
                    for r_idx, c in enumerate(cands, start=1):
                        if c["id"] == target_id or c["sku"] == target_sku:
                            rank = r_idx
                            target_sim = c["similarity"]
                            break

                    if rank is not None:
                        all_positive_ranks.append(rank)
                        if rank == 1:
                            rec1 += 1
                        if rank <= 5:
                            rec5 += 1
                        if rank <= 20:
                            rec20 += 1
                        rec_rr.append(1.0 / rank)
                        if target_sim >= 0.70:
                            positive_coverage_70 += 1
                        if target_sim >= 0.88:
                            positive_coverage_88 += 1
                    else:
                        rec_rr.append(0.0)

                n_queries = len(queries)
                slice_metrics[slice_name] = {
                    "total_queries": n_queries,
                    "recall_at_1": round(rec1 / n_queries, 4),
                    "recall_at_5": round(rec5 / n_queries, 4),
                    "recall_at_20": round(rec20 / n_queries, 4),
                    "mrr": round(float(np.mean(rec_rr)), 4),
                    "coverage_70": round(positive_coverage_70 / n_queries, 4),
                    "coverage_88": round(positive_coverage_88 / n_queries, 4),
                    "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                    "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2)
                }

        total_pos = sum(m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives")
        overall_r1 = sum(m["recall_at_1"] * m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives") / total_pos
        overall_r5 = sum(m["recall_at_5"] * m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives") / total_pos
        overall_r20 = sum(m["recall_at_20"] * m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives") / total_pos
        overall_mrr = sum(m["mrr"] * m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives") / total_pos

        return {
            "candidate": candidate_name,
            "overall_positive_summary": {
                "total_positive_queries": total_pos,
                "recall_at_1": round(overall_r1, 4),
                "recall_at_5": round(overall_r5, 4),
                "recall_at_20": round(overall_r20, 4),
                "mrr": round(overall_mrr, 4),
                "latency_p50_ms": round(float(np.percentile(all_latencies, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(all_latencies, 95)), 2)
            },
            "slices": slice_metrics
        }


def main():
    manifest_path = REPO_ROOT / "data" / "evaluation" / "ph014_evaluation_manifest.json"
    runner = VisualV2BenchmarkRunner()

    candidates = [
        "baseline_mobilenetv4",
        "robust_mobilenetv4",
        "dinov3_global_int8",
        "dinov3_robust_multicrop",
        "ensemble_v2"
    ]

    all_results = {}
    for c in candidates:
        logger.info(f"=== BENCHMARKING CANDIDATE: {c} ===")
        res = runner.evaluate_manifest(c, manifest_path)
        all_results[c] = res
        ov = res["overall_positive_summary"]
        logger.info(f"[{c}] -> Recall@1: {ov['recall_at_1']:.2%} | Recall@5: {ov['recall_at_5']:.2%} | Recall@20: {ov['recall_at_20']:.2%} | MRR: {ov['mrr']:.4f} | Latency p50: {ov['latency_p50_ms']}ms")

    out_file = REPO_ROOT / "output" / "retrieval_v2_benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Benchmark results successfully saved to {out_file}")


if __name__ == "__main__":
    main()
