#!/usr/bin/env python3
"""Evaluate Retrieval Robustness Benchmark across all 5 slices.

Supports comparing:
- Baseline single-image MobileNetV4.
- Candidate Preprocessing pipelines (CLAHE, Denoising, Autocrop).
- Candidate Multi-Crop / Test-Time Augmentation pipelines.

Outputs comprehensive JSON artifact to output/robustness_benchmark_results_<mode>.json.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.embeddings import MobileNetV4Encoder
from engine.preprocessing import PreprocessConfig, preprocess_query_image, pad_to_aspect_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_robustness")


class RobustnessEvaluator:
    def __init__(self, mode: str = "baseline"):
        self.mode = mode
        self.encoder = MobileNetV4Encoder()
        with open(REPO_ROOT / "data" / "index" / "mobilenetv4_meta.json", "r") as f:
            self.catalog_meta = json.load(f)
        self.catalog_embeddings = np.load(REPO_ROOT / "data" / "index" / "mobilenetv4_embeddings.npy")
        self.catalog_size = len(self.catalog_meta)

        # Precompute sku_id for each catalog item
        self.item_to_sku = {}
        for item in self.catalog_meta:
            cid = item["id"]
            sku = item.get("sku_id") or f"SKU-{item.get('category', 'RING').upper()[:4]}-{cid}"
            self.item_to_sku[cid] = sku

    def encode_query(self, img: Image.Image) -> Tuple[np.ndarray, float, float]:
        """Encode query image according to the specified pipeline mode.
        
        Returns:
            (query_vector_or_vectors, prep_time_ms, enc_time_ms)
        """
        t0_prep = time.time()

        if self.mode == "baseline":
            config = PreprocessConfig(target_size=224)
            prep, qual, meta = preprocess_query_image(img, config)
            prep_time_ms = (time.time() - t0_prep) * 1000

            t0_enc = time.time()
            vec = self.encoder.encode_image(prep)
            enc_time_ms = (time.time() - t0_enc) * 1000
            return [vec], prep_time_ms, enc_time_ms

        elif self.mode == "clahe_denoise":
            config = PreprocessConfig(target_size=224, enable_clahe=True, enable_denoise=True)
            prep, qual, meta = preprocess_query_image(img, config)
            prep_time_ms = (time.time() - t0_prep) * 1000

            t0_enc = time.time()
            vec = self.encoder.encode_image(prep)
            enc_time_ms = (time.time() - t0_enc) * 1000
            return [vec], prep_time_ms, enc_time_ms

        elif self.mode == "robust_multicrop":
            from engine.preprocessing import extract_robust_query_crops
            crops = extract_robust_query_crops(img)
            prep_time_ms = (time.time() - t0_prep) * 1000

            t0_enc = time.time()
            crop_vecs = [self.encoder.encode_image(c) for c, _ in crops]
            enc_time_ms = (time.time() - t0_enc) * 1000

            return crop_vecs, prep_time_ms, enc_time_ms

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def search(self, query_vectors: List[np.ndarray], category_filter: Optional[str] = None, exclude_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search catalog with single or multi-crop vectors using max-pooling fusion."""
        # Compute cosine similarities for each crop vector against catalog
        all_sims = []
        for vec in query_vectors:
            sims = np.dot(self.catalog_embeddings, vec)
            all_sims.append(sims)

        # Max-pool across query crops: best view representation matches candidate
        fused_sims = np.max(np.stack(all_sims, axis=0), axis=0)

        results = []
        for idx, sim in enumerate(fused_sims):
            item = self.catalog_meta[idx]
            cid = item["id"]
            if exclude_id and cid == exclude_id:
                continue
            cat = item.get("category", "")
            if category_filter and category_filter != "all" and cat != category_filter:
                continue

            results.append({
                "id": cid,
                "sku_id": self.item_to_sku.get(cid, cid),
                "category": cat,
                "similarity": float(sim)
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def evaluate_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        slice_metrics = {}
        all_latencies = []
        total_p_queries = 0
        total_r1 = 0
        total_r5 = 0
        total_r20 = 0
        total_rr = 0.0

        for slice_name, queries in manifest["slices"].items():
            if slice_name == "slice_e_hard_negatives":
                continue

            s_r1 = 0
            s_r5 = 0
            s_r20 = 0
            s_rr = 0.0
            query_records = []

            for q in queries:
                q_path = REPO_ROOT / q["query_path"]
                if not q_path.exists():
                    logger.warning(f"File missing: {q_path}")
                    continue

                q_img = Image.open(q_path)
                t0 = time.time()
                q_vecs, t_prep, t_enc = self.encode_query(q_img)
                t0_s = time.time()

                # For cross-view tests, exclude self-matching the exact source image if target is distinct
                target_id = q.get("target_catalog_id")
                source_id = Path(q["query_path"]).stem
                exclude = source_id if (source_id in self.item_to_sku and source_id != target_id) else None

                cat_filter = q.get("category")
                results = self.search(q_vecs, category_filter=cat_filter, exclude_id=exclude)
                t_search = (time.time() - t0_s) * 1000
                total_latency = (time.time() - t0) * 1000
                all_latencies.append(total_latency)

                # Find ground truth rank
                gt_rank = None
                gt_sim = None
                for rank, res in enumerate(results, 1):
                    if res["id"] == target_id:
                        gt_rank = rank
                        gt_sim = res["similarity"]
                        break

                r1 = gt_rank == 1
                r5 = gt_rank is not None and gt_rank <= 5
                r20 = gt_rank is not None and gt_rank <= 20
                rr = (1.0 / gt_rank) if gt_rank is not None else 0.0

                s_r1 += int(r1)
                s_r5 += int(r5)
                s_r20 += int(r20)
                s_rr += rr

                query_records.append({
                    "query_id": q["query_id"],
                    "target_id": target_id,
                    "gt_rank": gt_rank,
                    "gt_similarity": round(gt_sim, 4) if gt_sim is not None else None,
                    "top_match_id": results[0]["id"] if results else None,
                    "top_match_similarity": round(results[0]["similarity"], 4) if results else None,
                    "is_r1": r1,
                    "is_r5": r5,
                    "timings_ms": {
                        "prep": round(t_prep, 2),
                        "enc": round(t_enc, 2),
                        "search": round(t_search, 2),
                        "total": round(total_latency, 2)
                    }
                })

            n_queries = len(query_records)
            slice_metrics[slice_name] = {
                "count": n_queries,
                "recall_at_1": round(s_r1 / n_queries, 4) if n_queries else 0.0,
                "recall_at_5": round(s_r5 / n_queries, 4) if n_queries else 0.0,
                "recall_at_20": round(s_r20 / n_queries, 4) if n_queries else 0.0,
                "mrr": round(s_rr / n_queries, 4) if n_queries else 0.0,
                "queries": query_records
            }

            total_p_queries += n_queries
            total_r1 += s_r1
            total_r5 += s_r5
            total_r20 += s_r20
            total_rr += s_rr

        # Evaluate Hard Negatives (Slice E)
        hard_negatives = manifest["slices"].get("slice_e_hard_negatives", [])
        neg_evals = []
        tau_thresholds = [0.94, 0.88, 0.80, 0.70]
        false_accepts = {str(tau): 0 for tau in tau_thresholds}

        for neg in hard_negatives:
            q_path = REPO_ROOT / neg["query_path"]
            if not q_path.exists():
                continue
            q_img = Image.open(q_path)
            q_vecs, _, _ = self.encode_query(q_img)
            # Find similarity to the specific negative target
            target_id = neg["negative_target_id"]
            results = self.search(q_vecs, category_filter=neg.get("category"), exclude_id=Path(neg["query_path"]).stem)
            sim_to_neg = 0.0
            for r in results:
                if r["id"] == target_id:
                    sim_to_neg = r["similarity"]
                    break

            for tau in tau_thresholds:
                if sim_to_neg >= tau:
                    false_accepts[str(tau)] += 1

            neg_evals.append({
                "query_id": neg["query_id"],
                "target_neg_id": target_id,
                "similarity": round(sim_to_neg, 4)
            })

        n_neg = len(neg_evals)
        far_by_tau = {
            str(tau): {
                "false_accepts": false_accepts[str(tau)],
                "far_rate": round(false_accepts[str(tau)] / max(1, n_neg), 4)
            }
            for tau in tau_thresholds
        }

        slice_metrics["slice_e_hard_negatives"] = {
            "count": n_neg,
            "far_by_threshold": far_by_tau,
            "negative_evaluations": neg_evals
        }

        # Overall summary
        latencies_sorted = sorted(all_latencies)
        p50_lat = latencies_sorted[int(len(latencies_sorted) * 0.50)] if latencies_sorted else 0.0
        p95_lat = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0.0

        summary = {
            "mode": self.mode,
            "backbone": "MobileNetV4 Conv-Small FP32 ONNX (1280-d)",
            "total_positive_queries": total_p_queries,
            "overall_recall_at_1": round(total_r1 / total_p_queries, 4) if total_p_queries else 0.0,
            "overall_recall_at_5": round(total_r5 / total_p_queries, 4) if total_p_queries else 0.0,
            "overall_recall_at_20": round(total_r20 / total_p_queries, 4) if total_p_queries else 0.0,
            "overall_mrr": round(total_rr / total_p_queries, 4) if total_p_queries else 0.0,
            "latency_p50_ms": round(p50_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
            "slice_metrics": slice_metrics
        }

        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "clahe_denoise", "robust_multicrop"])
    parser.add_argument("--manifest", type=str, default="data/evaluation/robustness_benchmark_manifest.json")
    args = parser.parse_args()

    evaluator = RobustnessEvaluator(mode=args.mode)
    manifest_path = REPO_ROOT / args.manifest
    report = evaluator.evaluate_manifest(manifest_path)

    out_file = REPO_ROOT / "output" / f"robustness_benchmark_{args.mode}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(report, fp, indent=2)

    print("=" * 80)
    print(f"ROBUSTNESS BENCHMARK RESULTS: MODE = {args.mode.upper()}")
    print("=" * 80)
    print(f"Overall Recall@1:  {report['overall_recall_at_1'] * 100:.2f}%")
    print(f"Overall Recall@5:  {report['overall_recall_at_5'] * 100:.2f}%")
    print(f"Overall Recall@20: {report['overall_recall_at_20'] * 100:.2f}%")
    print(f"Overall MRR:       {report['overall_mrr']:.4f}")
    print(f"Latency p50:       {report['latency_p50_ms']} ms")
    print(f"Latency p95:       {report['latency_p95_ms']} ms")
    print("-" * 80)
    print("Slice Breakdown:")
    for s_name in ["slice_a_real_multiview", "slice_b_worn_on_hand", "slice_c_viewpoint_shifts", "slice_d_image_corruptions"]:
        sm = report["slice_metrics"][s_name]
        print(f"  {s_name:28s} | Count: {sm['count']:2d} | R@1: {sm['recall_at_1']*100:5.1f}% | R@5: {sm['recall_at_5']*100:5.1f}% | MRR: {sm['mrr']:.4f}")
    print("-" * 80)
    sm_neg = report["slice_metrics"]["slice_e_hard_negatives"]
    print("Hard Negative False Accept Rates (FAR):")
    for tau, far in sm_neg["far_by_threshold"].items():
        print(f"  Threshold tau >= {tau}: False Accepts = {far['false_accepts']}/{sm_neg['count']} (FAR = {far['far_rate']*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
