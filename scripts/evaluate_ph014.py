#!/usr/bin/env python3
"""Evaluate PH-014 Retrieval Quality, Candidate Filtering & Real-Image Benchmark.

Evaluates 3 distinct configurations across all 6 slices of the frozen manifest:
1. baseline_single_crop: Raw single-image MobileNetV4 1280-dim embedding.
2. naive_multicrop: 3-crop max-pooling without skin-background suppression.
3. ph014_robust_candidate: 3-crop adaptive skin-suppressed max fusion + strict category enforcement.

Reports:
- Recall@1, Recall@5, Recall@20, MRR for Slices 1–5.
- False Accept Rate (FAR @ 0.88, FAR @ 0.70, FAR @ 0.52) for Slice 6 (Hard Negatives).
- Latency breakdown (p50, p95).

Saves structured results to:
    output/ph014_evaluation_results.json
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

from engine.embeddings import MobileNetV4Encoder
from engine.preprocessing import PreprocessConfig, preprocess_query_image, extract_robust_query_crops
from engine.index import CatalogIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_ph014")


class PH014Evaluator:
    def __init__(self, mode: str = "ph014_robust_candidate"):
        self.mode = mode
        self.encoder = MobileNetV4Encoder()
        with open(REPO_ROOT / "data" / "index" / "mobilenetv4_meta.json", "r") as f:
            self.catalog_meta = json.load(f)
        self.catalog_embeddings = np.load(REPO_ROOT / "data" / "index" / "mobilenetv4_embeddings.npy")
        self.catalog_size = len(self.catalog_meta)

        self.item_to_sku = {}
        for item in self.catalog_meta:
            cid = item["id"]
            sku = item.get("sku_id") or f"SKU-{item.get('category', 'RING').upper()[:4]}-{cid}"
            self.item_to_sku[cid] = sku

    def run_query(self, img_path: Path, category_filter: str) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.time()
        img = Image.open(img_path).convert("RGB")
        norm_cat = category_filter.lower().rstrip("s")

        if self.mode == "baseline_single_crop":
            config = PreprocessConfig(target_size=224)
            prep_img, quality, _ = preprocess_query_image(img, config)
            vec = self.encoder.encode_image(prep_img)
            # Search
            sims = np.dot(self.catalog_embeddings, vec)
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

        elif self.mode == "naive_multicrop":
            crops = extract_robust_query_crops(img)
            crop_sims = []
            for crop_img, _ in crops:
                vec = self.encoder.encode_image(crop_img)
                sims = np.dot(self.catalog_embeddings, vec)
                crop_sims.append(sims)
            # Naive max pooling across crops without skin suppression
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

        elif self.mode == "ph014_robust_candidate":
            crops = extract_robust_query_crops(img)
            _, quality, _ = preprocess_query_image(img)
            skin_detected = getattr(quality, "skin_detected", False)

            crop_candidate_dicts = []
            for crop_img, crop_name in crops:
                vec = self.encoder.encode_image(crop_img)
                sims = np.dot(self.catalog_embeddings, vec)
                weight = 0.82 if (skin_detected and crop_name == "global") else 1.0
                crop_candidate_dicts.append(sims * weight)

            fused_sims = np.maximum.reduce(crop_candidate_dicts)
            ranked_indices = np.argsort(-fused_sims)
            candidates = []
            for idx in ranked_indices:
                cat = self.catalog_meta[idx].get("category", "")
                # Strict post-filter
                if norm_cat and cat.lower().rstrip("s") != norm_cat:
                    continue
                candidates.append({
                    "id": self.catalog_meta[idx]["id"],
                    "sku": self.item_to_sku[self.catalog_meta[idx]["id"]],
                    "category": cat,
                    "similarity": float(fused_sims[idx])
                })
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        latency_ms = (time.time() - t0) * 1000
        return candidates, latency_ms

    def evaluate_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        with open(manifest_path) as f:
            manifest = json.load(f)

        slice_metrics = {}
        all_positive_ranks = []
        all_latencies = []

        for slice_name, queries in manifest["slices"].items():
            logger.info(f"Evaluating {self.mode} on {slice_name} ({len(queries)} items)...")

            if slice_name == "slice_6_hard_negatives":
                # Hard negative evaluation: False Accept Rate
                false_accepts_88 = 0
                false_accepts_70 = 0
                false_accepts_52 = 0
                max_sims = []
                latencies = []

                for q in queries:
                    q_path = REPO_ROOT / q["query_path"]
                    cands, lat = self.run_query(q_path, category_filter=q["category"])
                    latencies.append(lat)

                    # Similarity against non-matching target item
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
                # Positive query retrieval evaluation
                rec1 = 0
                rec5 = 0
                rec20 = 0
                rec_rr = []
                positive_coverage_70 = 0
                positive_coverage_88 = 0
                latencies = []

                for q in queries:
                    q_path = REPO_ROOT / q["query_path"]
                    cands, lat = self.run_query(q_path, category_filter=q["category"])
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

        # Overall positive summary
        total_pos = sum(m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives")
        overall_r1 = sum(m["recall_at_1"] * m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives") / total_pos
        overall_r5 = sum(m["recall_at_5"] * m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives") / total_pos
        overall_r20 = sum(m["recall_at_20"] * m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives") / total_pos
        overall_mrr = sum(m["mrr"] * m["total_queries"] for k, m in slice_metrics.items() if k != "slice_6_hard_negatives") / total_pos

        return {
            "mode": self.mode,
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
    modes = ["baseline_single_crop", "naive_multicrop", "ph014_robust_candidate"]
    all_results = {}

    for m in modes:
        evaluator = PH014Evaluator(mode=m)
        all_results[m] = evaluator.evaluate_manifest(manifest_path)

    out_file = REPO_ROOT / "output" / "ph014_evaluation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("PH-014 FROZEN BENCHMARK COMPARISON MATRIX (122 EVALUATION QUERIES)")
    print("="*80)
    print(f"{'Slice / Metric':<35} | {'Baseline':<12} | {'Naive Multi':<12} | {'PH-014 Robust':<12}")
    print("-" * 80)

    slices_to_show = [
        ("Slice 1: Same Image / Identity R@1", "slice_1_same_image_near_duplicate", "recall_at_1"),
        ("Slice 2: Corruptions (Noise/Blur) R@1", "slice_2_image_corruptions", "recall_at_1"),
        ("Slice 3: Viewpoint Shifts (±15°) R@1", "slice_3_viewpoint_crops", "recall_at_1"),
        ("Slice 4: Genuine Physical Multi-View R@1", "slice_4_genuine_physical_multiview", "recall_at_1"),
        ("Slice 5: Worn on Hand / Skin R@1", "slice_5_worn_on_hand", "recall_at_1"),
        ("Overall Positive Recall@1", "overall", "recall_at_1"),
        ("Overall Positive Recall@5", "overall", "recall_at_5"),
        ("Overall Positive MRR", "overall", "mrr"),
        ("Slice 6: Hard Negative FAR @ 0.88", "slice_6_hard_negatives", "far_exact_88"),
        ("Slice 6: Hard Negative FAR @ 0.70", "slice_6_hard_negatives", "far_high_70"),
        ("Serving Latency p50 (ms)", "overall", "latency_p50_ms"),
    ]

    for label, s_key, metric_key in slices_to_show:
        if s_key == "overall":
            val_base = all_results["baseline_single_crop"]["overall_positive_summary"][metric_key]
            val_naive = all_results["naive_multicrop"]["overall_positive_summary"][metric_key]
            val_ph014 = all_results["ph014_robust_candidate"]["overall_positive_summary"][metric_key]
        else:
            val_base = all_results["baseline_single_crop"]["slices"][s_key][metric_key]
            val_naive = all_results["naive_multicrop"]["slices"][s_key][metric_key]
            val_ph014 = all_results["ph014_robust_candidate"]["slices"][s_key][metric_key]

        if "far" in metric_key or "recall" in metric_key or "mrr" in metric_key:
            print(f"{label:<35} | {val_base*100:>10.1f}% | {val_naive*100:>10.1f}% | {val_ph014*100:>10.1f}%")
        else:
            print(f"{label:<35} | {val_base:>10.1f}   | {val_naive:>10.1f}   | {val_ph014:>10.1f}  ")

    print("="*80)
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
