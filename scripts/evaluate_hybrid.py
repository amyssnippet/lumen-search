#!/usr/bin/env python3
"""EXP-004.1 Comprehensive Evaluation Runner: Corrected Ground Truth, Symmetric Masking, and Patch Weight Sweeps."""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.embeddings import get_encoder
from engine.index import CatalogIndex
from engine.preprocessing import PreprocessConfig, preprocess_query_image
from engine.hybrid_search import HybridSearchEngine

def run_evaluation():
    manifest_path = "data/manifest.json"
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    test_items = [img for img in manifest["images"] if img.get("split") == "test"]
    
    # Historical Benchmark v1: all 30 twin queries (including ring_083)
    v1_twin_queries = [
        img for img in test_items
        if any(c["cluster_id"] == img["cluster_id"] and c["split"] == "catalog" for c in manifest["images"])
    ]
    
    # Corrected Benchmark v2: exclude ring_083 (noisy transitive clustering with ring_187 via ring_116)
    v2_twin_queries = [
        img for img in v1_twin_queries if img["id"] != "ring_083"
    ]

    print(f"Total test queries: {len(test_items)}")
    print(f"  Historical Benchmark v1 twins: {len(v1_twin_queries)} (28 necklaces, 2 rings)")
    print(f"  Corrected Benchmark v2 twins: {len(v2_twin_queries)} (28 necklaces, 1 ring: ring_080)")

    # Initialize Hybrid Engine
    print("\nInitializing HybridSearchEngine...")
    hybrid_engine = HybridSearchEngine(index_dir="data/index", device="cpu")

    # Load calibrated thresholds if available
    calib_thresholds = {
        "dinov2": 0.935,
        "dinov3": 0.975,
        "hybrid_global": 0.955,
        "hybrid_rerank_pw02": 0.955
    }
    calib_file = Path("output/calibration_report.json")
    if calib_file.exists():
        with open(calib_file) as f:
            c_data = json.load(f)
            for k in calib_thresholds:
                if k in c_data and c_data[k].get("calibrated_threshold_99_precision"):
                    calib_thresholds[k] = c_data[k]["calibrated_threshold_99_precision"]
    print(f"Loaded 99% precision calibrated thresholds: {calib_thresholds}")

    configs = [
        {"key": "dinov2_global", "name": "A: DINOv2 Global Baseline", "mode": "v2", "patch_weight": 0.0, "spatial_weight": 0.0},
        {"key": "dinov3_global", "name": "B: DINOv3 Global Baseline", "mode": "v3", "patch_weight": 0.0, "spatial_weight": 0.0},
        {"key": "hybrid_global", "name": "C: Hybrid Global (patch_weight=0.0)", "mode": "hybrid", "patch_weight": 0.0, "spatial_weight": 0.0},
        {"key": "hybrid_rerank_sym_pw02", "name": "D: Hybrid + Symmetric Masked Rerank (patch_weight=0.2)", "mode": "hybrid_rerank", "patch_weight": 0.2, "spatial_weight": 0.0},
        {"key": "hybrid_rerank_spatial_pw02", "name": "E: Hybrid + Spatially Constrained Rerank (pw=0.2, spat=0.1)", "mode": "hybrid_rerank", "patch_weight": 0.2, "spatial_weight": 0.1},
        {"key": "hybrid_rerank_chamfer_only", "name": "F: Chamfer Correspondence Only (patch_weight=1.0)", "mode": "hybrid_rerank", "patch_weight": 1.0, "spatial_weight": 0.0},
    ]

    all_results = {}

    for cfg in configs:
        cfg_key = cfg["key"]
        print(f"\n==========================================")
        print(f"Evaluating {cfg['name']}...")
        print(f"==========================================")

        v1_hits_1, v1_hits_5, v1_hits_10, v1_hits_20 = 0, 0, 0, 0
        v2_hits_1, v2_hits_5, v2_hits_10, v2_hits_20 = 0, 0, 0, 0
        v1_mrr_list = []
        v2_mrr_list = []
        latencies_ms = []
        timing_stats = {
            "query_enhancement": [],
            "dinov2_inference": [],
            "dinov3_inference": [],
            "global_candidate_generation": [],
            "patch_extraction": [],
            "patch_reranking": [],
            "total": []
        }
        category_matches_top1 = 0
        category_precision_top5 = []
        false_exact_matches = 0

        # Calibrated threshold for this config
        tau = calib_thresholds.get("dinov3" if "dinov3" in cfg_key else "dinov2" if "dinov2" in cfg_key else "hybrid_rerank_pw02", 0.955)

        for item in test_items:
            img_path = Path(item["rel_path"])
            if not img_path.exists():
                img_path = Path("Jewellery_Data") / item["category"] / item["filename"]

            with Image.open(img_path) as raw_img:
                t0 = time.time()
                
                if cfg["mode"] == "v2":
                    prep_img, _, _ = preprocess_query_image(raw_img)
                    t_prep = (time.time() - t0) * 1000.0
                    t_inf0 = time.time()
                    q_vec = hybrid_engine.v2_encoder.encode_image(prep_img.resize((224, 224)))
                    t_inf = (time.time() - t_inf0) * 1000.0
                    t_s0 = time.time()
                    out = hybrid_engine.v2_index.search(q_vec, top_k=20, exclude_ids={item["id"]})
                    t_search = (time.time() - t_s0) * 1000.0
                    tot_latency = (time.time() - t0) * 1000.0
                    results = out["results"]
                    timing_stats["query_enhancement"].append(t_prep)
                    timing_stats["dinov2_inference"].append(t_inf)
                    timing_stats["dinov3_inference"].append(0.0)
                    timing_stats["global_candidate_generation"].append(t_search)
                    timing_stats["patch_extraction"].append(0.0)
                    timing_stats["patch_reranking"].append(0.0)
                    timing_stats["total"].append(tot_latency)

                elif cfg["mode"] == "v3":
                    prep_img, _, _ = preprocess_query_image(raw_img)
                    t_prep = (time.time() - t0) * 1000.0
                    t_inf0 = time.time()
                    q_vec = hybrid_engine.v3_encoder.encode_image(prep_img.resize((256, 256)))
                    t_inf = (time.time() - t_inf0) * 1000.0
                    t_s0 = time.time()
                    out = hybrid_engine.v3_index.search(q_vec, top_k=20, exclude_ids={item["id"]})
                    t_search = (time.time() - t_s0) * 1000.0
                    tot_latency = (time.time() - t0) * 1000.0
                    results = out["results"]
                    timing_stats["query_enhancement"].append(t_prep)
                    timing_stats["dinov2_inference"].append(0.0)
                    timing_stats["dinov3_inference"].append(t_inf)
                    timing_stats["global_candidate_generation"].append(t_search)
                    timing_stats["patch_extraction"].append(0.0)
                    timing_stats["patch_reranking"].append(0.0)
                    timing_stats["total"].append(tot_latency)

                elif cfg["mode"] == "hybrid":
                    out = hybrid_engine.search_hybrid(
                        image=raw_img,
                        top_k=20,
                        candidate_depth=20,
                        enable_rerank=False,
                        patch_weight=0.0,
                        exclude_ids={item["id"]},
                        exact_match_threshold=tau
                    )
                    results = out["results"]
                    for k_t in timing_stats:
                        timing_stats[k_t].append(out["timing_ms"].get(k_t, 0.0))

                elif cfg["mode"] == "hybrid_rerank":
                    out = hybrid_engine.search_hybrid(
                        image=raw_img,
                        top_k=20,
                        candidate_depth=20,
                        enable_rerank=True,
                        patch_weight=cfg["patch_weight"],
                        spatial_weight=cfg["spatial_weight"],
                        exclude_ids={item["id"]},
                        exact_match_threshold=tau
                    )
                    results = out["results"]
                    for k_t in timing_stats:
                        timing_stats[k_t].append(out["timing_ms"].get(k_t, 0.0))

            latencies_ms.append(timing_stats["total"][-1])
            target_cluster = item["cluster_id"]

            # Category Precision across all 60 queries
            if results:
                if results[0]["category"] == item["category"]:
                    category_matches_top1 += 1
                cat_p5 = sum(1 for r in results[:5] if r["category"] == item["category"]) / min(5, len(results))
                category_precision_top5.append(cat_p5)

            # False Exact Matches
            for r in results[:10]:
                if r["similarity"] >= tau and r["cluster_id"] != target_cluster:
                    false_exact_matches += 1

            # Match rank calculation
            found_rank = None
            for idx, r in enumerate(results):
                if r["cluster_id"] == target_cluster:
                    found_rank = idx + 1
                    break

            # Benchmark v1 tracking (30 queries)
            if item["id"] in {t["id"] for t in v1_twin_queries}:
                if found_rank == 1: v1_hits_1 += 1
                if found_rank is not None and found_rank <= 5: v1_hits_5 += 1
                if found_rank is not None and found_rank <= 10: v1_hits_10 += 1
                if found_rank is not None and found_rank <= 20: v1_hits_20 += 1
                v1_mrr_list.append(1.0 / found_rank if found_rank else 0.0)

            # Benchmark v2 tracking (29 queries, excluding ring_083)
            if item["id"] in {t["id"] for t in v2_twin_queries}:
                if found_rank == 1: v2_hits_1 += 1
                if found_rank is not None and found_rank <= 5: v2_hits_5 += 1
                if found_rank is not None and found_rank <= 10: v2_hits_10 += 1
                if found_rank is not None and found_rank <= 20: v2_hits_20 += 1
                v2_mrr_list.append(1.0 / found_rank if found_rank else 0.0)

        n_v1 = len(v1_twin_queries)
        n_v2 = len(v2_twin_queries)
        n_tot = len(test_items)

        res_entry = {
            "name": cfg["name"],
            "patch_weight": cfg["patch_weight"],
            "spatial_weight": cfg["spatial_weight"],
            "benchmark_v1_historical": {
                "num_queries": n_v1,
                "recall_at_1": round(v1_hits_1 / n_v1 * 100, 2),
                "recall_at_5": round(v1_hits_5 / n_v1 * 100, 2),
                "recall_at_10": round(v1_hits_10 / n_v1 * 100, 2),
                "candidate_recall_at_20": round(v1_hits_20 / n_v1 * 100, 2),
                "mrr": round(float(np.mean(v1_mrr_list)), 4)
            },
            "benchmark_v2_corrected": {
                "num_queries": n_v2,
                "recall_at_1": round(v2_hits_1 / n_v2 * 100, 2),
                "recall_at_5": round(v2_hits_5 / n_v2 * 100, 2),
                "recall_at_10": round(v2_hits_10 / n_v2 * 100, 2),
                "candidate_recall_at_20": round(v2_hits_20 / n_v2 * 100, 2),
                "mrr": round(float(np.mean(v2_mrr_list)), 4)
            },
            "category_precision_top1": round(category_matches_top1 / n_tot * 100, 2),
            "category_precision_top5": round(float(np.mean(category_precision_top5)) * 100, 2),
            "false_exact_matches": false_exact_matches,
            "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
            "timing_breakdown_ms": {k: round(float(np.mean(v)), 2) for k, v in timing_stats.items()}
        }
        all_results[cfg_key] = res_entry

        print(f"Results for {cfg['name']}:")
        print(f"  [Corrected v2] Recall@1: {res_entry['benchmark_v2_corrected']['recall_at_1']}% ({v2_hits_1}/{n_v2}) | R@5: {res_entry['benchmark_v2_corrected']['recall_at_5']}% | MRR: {res_entry['benchmark_v2_corrected']['mrr']}")
        print(f"  [Historical v1] Recall@1: {res_entry['benchmark_v1_historical']['recall_at_1']}% ({v1_hits_1}/{n_v1}) | R@5: {res_entry['benchmark_v1_historical']['recall_at_5']}% | MRR: {res_entry['benchmark_v1_historical']['mrr']}")
        print(f"  Latency: {res_entry['mean_latency_ms']} ms | False exact matches: {false_exact_matches}")

    # Patch Weight Sweep on Corrected Benchmark v2
    print(f"\n==========================================")
    print("Evaluating patch_weight sweep on Corrected v2...")
    print(f"==========================================")
    pw_sweep_results = []
    for pw in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        hits_1, hits_5, hits_10 = 0, 0, 0
        mrrs = []
        for item in v2_twin_queries:
            img_path = Path(item["rel_path"])
            if not img_path.exists():
                img_path = Path("Jewellery_Data") / item["category"] / item["filename"]
            with Image.open(img_path) as raw_img:
                out = hybrid_engine.search_hybrid(
                    image=raw_img,
                    top_k=20,
                    candidate_depth=20,
                    enable_rerank=(pw > 0.0),
                    patch_weight=pw,
                    spatial_weight=0.0,
                    exclude_ids={item["id"]}
                )
            results = out["results"]
            target_cluster = item["cluster_id"]
            rank = None
            for idx, r in enumerate(results):
                if r["cluster_id"] == target_cluster:
                    rank = idx + 1
                    break
            if rank == 1: hits_1 += 1
            if rank is not None and rank <= 5: hits_5 += 1
            if rank is not None and rank <= 10: hits_10 += 1
            mrrs.append(1.0 / rank if rank else 0.0)

        entry = {
            "patch_weight": pw,
            "recall_at_1": round(hits_1 / len(v2_twin_queries) * 100, 2),
            "recall_at_5": round(hits_5 / len(v2_twin_queries) * 100, 2),
            "recall_at_10": round(hits_10 / len(v2_twin_queries) * 100, 2),
            "mrr": round(float(np.mean(mrrs)), 4)
        }
        pw_sweep_results.append(entry)
        print(f"  pw={pw:.1f} | Recall@1: {entry['recall_at_1']}% ({hits_1}/{len(v2_twin_queries)}) | Recall@5: {entry['recall_at_5']}% | MRR: {entry['mrr']}")

    output_doc = {
        "configurations": all_results,
        "patch_weight_sweep": pw_sweep_results
    }

    out_file = Path("output/eval_hybrid.json")
    with open(out_file, "w") as fp:
        json.dump(output_doc, fp, indent=2)
    print(f"\nAll evaluation outputs successfully saved to {out_file}")

if __name__ == "__main__":
    run_evaluation()
