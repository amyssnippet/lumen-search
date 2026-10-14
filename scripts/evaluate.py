#!/usr/bin/env python3
"""Repeatable retrieval evaluation pipeline measuring Recall@1/5/10, Category Precision, and Latency."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.embeddings import get_encoder
from engine.index import CatalogIndex, EXACT_MATCH_THRESHOLD, SIMILARITY_THRESHOLD
from engine.preprocessing import PreprocessConfig, preprocess_query_image

def evaluate_retrieval(
    model_name: str = "dinov2",
    prep_config: PreprocessConfig = PreprocessConfig(),
    manifest_path: str = "data/manifest.json",
    index_dir: str = "data/index"
) -> Dict[str, Any]:
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    encoder = get_encoder(model_name)
    index = CatalogIndex(model_name=model_name, index_dir=index_dir)
    index.load()

    test_items = [img for img in manifest["images"] if img.get("split") == "test"]
    twin_queries = [img for img in test_items if manifest["metadata"]["total_clusters"] and 
                    sum(1 for c in manifest["images"] if c["cluster_id"] == img["cluster_id"] and c["split"] == "catalog") > 0]
    singleton_queries = [img for img in test_items if img["id"] not in {t["id"] for t in twin_queries}]

    print(f"\n--- Evaluating Model: '{model_name}' ---")
    print(f"Total test queries: {len(test_items)} (Twin/Identity queries: {len(twin_queries)}, Semantic singletons: {len(singleton_queries)})")

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    latencies_ms = []
    category_matches_top1 = 0
    category_precision_top5 = []
    false_exact_matches = 0 # Similarity >= 0.94 on non-matching cluster

    twin_results = []
    hard_failures = []

    for item in test_items:
        img_path = Path(item["rel_path"])
        if not img_path.exists():
            img_path = Path("Jewellery_Data") / item["category"] / item["filename"]
        
        with Image.open(img_path) as raw_img:
            t0 = time.time()
            prep_img, quality_rep, _ = preprocess_query_image(raw_img, config=prep_config)
            q_vec = encoder.encode_image(prep_img)
            # Search catalog excluding self
            search_out = index.search(q_vec, top_k=10, exclude_ids={item["id"]})
            latency = (time.time() - t0) * 1000.0
            latencies_ms.append(latency)

        results = search_out["results"]
        target_cluster = item["cluster_id"]
        is_twin_query = any(c["cluster_id"] == target_cluster and c["split"] == "catalog" for c in manifest["images"])

        # Check Category Precision
        if results:
            if results[0]["category"] == item["category"]:
                category_matches_top1 += 1
            cat_p5 = sum(1 for r in results[:5] if r["category"] == item["category"]) / min(5, len(results))
            category_precision_top5.append(cat_p5)

        # Check False Exact Matches
        for r in results:
            if r["similarity"] >= EXACT_MATCH_THRESHOLD and r["cluster_id"] != target_cluster:
                false_exact_matches += 1

        # Check Exact Twin Hit if ground truth counterpart exists
        if is_twin_query:
            found_rank = None
            for idx, r in enumerate(results):
                if r["cluster_id"] == target_cluster:
                    found_rank = idx + 1
                    break

            if found_rank == 1:
                hits_at_1 += 1
            if found_rank is not None and found_rank <= 5:
                hits_at_5 += 1
            if found_rank is not None and found_rank <= 10:
                hits_at_10 += 1
            if found_rank is not None:
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

            twin_record = {
                "query_id": item["id"],
                "category": item["category"],
                "cluster_id": target_cluster,
                "found_rank": found_rank,
                "top_similarity": results[0]["similarity"] if results else 0.0,
                "top_retrieved_id": results[0]["id"] if results else None
            }
            twin_results.append(twin_record)
            if found_rank is None or found_rank > 1:
                hard_failures.append(twin_record)

    num_twins = len(twin_queries) if twin_queries else 1
    num_total = len(test_items) if test_items else 1

    r1 = round((hits_at_1 / num_twins) * 100, 2)
    r5 = round((hits_at_5 / num_twins) * 100, 2)
    r10 = round((hits_at_10 / num_twins) * 100, 2)
    mrr = round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4)
    cat_acc = round((category_matches_top1 / num_total) * 100, 2)
    mean_cat_p5 = round(float(np.mean(category_precision_top5)) * 100, 2) if category_precision_top5 else 0.0
    mean_latency = round(float(np.mean(latencies_ms)), 2)

    report = {
        "model_name": model_name,
        "preprocessing": {
            "enable_clahe": prep_config.enable_clahe,
            "enable_denoise": prep_config.enable_denoise,
            "enable_autocrop": prep_config.enable_autocrop,
            "auto_enhance": prep_config.auto_enhance
        },
        "num_evaluated_queries": num_total,
        "num_identity_queries": num_twins,
        "metrics": {
            "recall_at_1": r1,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "mrr": mrr,
            "category_precision_at_1": cat_acc,
            "category_precision_at_5": mean_cat_p5,
            "false_exact_match_count": false_exact_matches,
            "mean_query_latency_ms": mean_latency
        },
        "hard_failures_sample": hard_failures[:10]
    }

    print(f"Results Summary for '{model_name}':")
    print(f"  Recall@1: {r1}% | Recall@5: {r5}% | Recall@10: {r10}%")
    print(f"  MRR: {mrr} | Category Precision@1: {cat_acc}% | Category Precision@5: {mean_cat_p5}%")
    print(f"  Mean Latency: {mean_latency}ms | False Exact Matches: {false_exact_matches}")
    return report

def main():
    parser = argparse.ArgumentParser(description="Evaluate jewellery visual retrieval")
    parser.add_argument("--model", type=str, default="dinov2", choices=["dinov2", "clip", "dinov3", "both", "all"])
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE")
    parser.add_argument("--denoise", action="store_true", help="Enable bilateral denoising")
    parser.add_argument("--crop", action="store_true", help="Enable auto-cropping")
    parser.add_argument("--auto-enhance", action="store_true", help="Enable quality-gated auto-enhancement")
    parser.add_argument("--output", type=str, default="output/evaluation_report.json")
    args = parser.parse_args()

    if args.model == "all":
        models = ["dinov2", "clip", "dinov3"]
    elif args.model == "both":
        models = ["dinov2", "clip"]
    else:
        models = [args.model]
    config = PreprocessConfig(
        enable_clahe=args.clahe,
        enable_denoise=args.denoise,
        enable_autocrop=args.crop,
        auto_enhance=args.auto_enhance
    )

    all_reports = {}
    for m in models:
        rep = evaluate_retrieval(model_name=m, prep_config=config)
        all_reports[m] = rep

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(all_reports, fp, indent=2)
    print(f"\nEvaluation saved to {args.output}")

if __name__ == "__main__":
    main()
