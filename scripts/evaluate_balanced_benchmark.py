#!/usr/bin/env python3
"""EXP-005 Phase 5: Comprehensive evaluation of candidate generation and local patch reranking
on the balanced 60-query benchmark (30 necklaces, 30 rings) with 30 hard negatives and 30 singletons.

Compares:
1. DINOv2 Global
2. DINOv3 Global
3. Calibrated Min-Max Hybrid Global (pw=0.0)
4. DINOv3 Patch Chamfer (Unmasked, spatial=0.0, pw=0.2)
5. Symmetric Foreground Masked Chamfer (spatial=0.0, pw=0.2)
6. Spatially Constrained Symmetric Chamfer (spatial=0.1, pw=0.2)
7. Spatially Constrained Symmetric Chamfer (spatial=0.1, pw=0.5) [Best validated]
8. Chamfer Only (spatial=0.1, pw=1.0)
"""

import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Any

from engine.embeddings import DINOv2Encoder, DINOv3Encoder
from engine.reranker import compute_chamfer_similarity, get_foreground_patch_indices

def run_evaluation():
    # 1. Load balanced benchmark
    bench = json.load(open("data/evaluation/benchmark_v3_balanced.json"))
    same_sku = bench["same_sku_queries"]
    hard_negs = bench["hard_negatives"]
    singletons = bench["singletons"]
    
    # 2. Load catalog index
    cat_meta = json.load(open("data/index/dinov2_meta.json"))
    cat_ids = [m["id"] for m in cat_meta]
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}
    v2_index = np.load("data/index/dinov2_embeddings.npy")
    v3_index = np.load("data/index/dinov3_embeddings.npy")
    v3_patches = np.load("data/index/dinov3_patches.npy")
    patch_meta = json.load(open("data/index/dinov3_patches.meta.json"))
    cat_fg_masks = {
        cid: np.array(mask, dtype=np.int32)
        for cid, mask in zip(patch_meta["catalog_ids"], patch_meta["foreground_masks"])
    }
    
    # Validation score normalization parameters
    v2_neg_mean, v2_pos_mean = 0.4372, 0.9825
    v3_neg_mean, v3_pos_mean = 0.7924, 0.9927
    
    # Load calibrated tau_99 from recalibration report
    recalib = json.load(open("output/recalibration_balanced.json"))
    tau_map = {k: v["tau_99"] for k, v in recalib["models"].items()}
    
    print("Initializing vision backbones...")
    v2_enc = DINOv2Encoder()
    v3_enc = DINOv3Encoder()
    
    # Pre-encode all 60 benchmark queries
    print("Pre-encoding 60 benchmark queries...")
    query_data = []
    for q in same_sku:
        t0 = time.perf_counter()
        with Image.open(q["query_path"]) as img:
            q_rgb = img.convert("RGB")
            t_decode = (time.perf_counter() - t0) * 1000
            
            t1 = time.perf_counter()
            q_v2 = v2_enc.encode_image(q_rgb)
            t_enc_v2 = (time.perf_counter() - t1) * 1000
            
            t2 = time.perf_counter()
            q_v3 = v3_enc.encode_image(q_rgb)
            q_patch = v3_enc.get_patch_tokens(q_rgb)
            t_enc_v3 = (time.perf_counter() - t2) * 1000
            
            q_fg = get_foreground_patch_indices(q_rgb)
            
        query_data.append({
            "meta": q,
            "v2": q_v2,
            "v3": q_v3,
            "patch": q_patch,
            "fg": q_fg,
            "timing": {"decode_ms": t_decode, "enc_v2_ms": t_enc_v2, "enc_v3_ms": t_enc_v3}
        })
        
    configurations = [
        ("dinov2_global", "DINOv2 Global", 0.0, 0.0, False, "v2_only"),
        ("dinov3_global", "DINOv3 Global", 0.0, 0.0, False, "v3_only"),
        ("hybrid_minmax_global", "Calibrated Min-Max Global (pw=0.0)", 0.0, 0.0, False, "minmax"),
        ("chamfer_unmasked", "DINOv3 Chamfer (Unmasked, spat=0.0, pw=0.2)", 0.2, 0.0, False, "minmax"),
        ("chamfer_masked", "Symmetric Masked Chamfer (spat=0.0, pw=0.2)", 0.2, 0.0, True, "minmax"),
        ("chamfer_spatial_pw02", "Spatially Constrained Chamfer (spat=0.1, pw=0.2)", 0.2, 0.1, True, "minmax"),
        ("chamfer_spatial_pw05", "Spatially Constrained Chamfer (spat=0.1, pw=0.5)", 0.5, 0.1, True, "minmax"),
        ("chamfer_only", "Chamfer Only (spat=0.1, pw=1.0)", 1.0, 0.1, True, "minmax"),
    ]
    
    report = {"overall": {}, "necklaces": {}, "rings": {}, "configurations": []}
    
    print("\n" + "="*85)
    print(f"{'Configuration':<38} | {'Overall R@1':<11} | {'R@5':<7} | {'MRR':<7} | {'Neck R@1':<8} | {'Ring R@1':<8} | {'Latency':<7}")
    print("="*85)
    
    for cfg_key, cfg_name, pw, spat, use_fg, fusion_type in configurations:
        results_all = []
        latencies = []
        
        for qd in query_data:
            q_meta = qd["meta"]
            target_id = q_meta["target_catalog_id"]
            q_v2 = qd["v2"]
            q_v3 = qd["v3"]
            q_patch = qd["patch"]
            q_fg = qd["fg"]
            
            t_start = time.perf_counter()
            # 1. Global scoring across 432 catalog items
            cos_v2 = np.dot(v2_index, q_v2)
            cos_v3 = np.dot(v3_index, q_v3)
            
            if fusion_type == "v2_only":
                global_scores = cos_v2
            elif fusion_type == "v3_only":
                global_scores = cos_v3
            else: # minmax
                norm_v2 = np.clip((cos_v2 - v2_neg_mean) / (v2_pos_mean - v2_neg_mean), 0, 1)
                norm_v3 = np.clip((cos_v3 - v3_neg_mean) / (v3_pos_mean - v3_neg_mean), 0, 1)
                global_scores = 0.5 * norm_v2 + 0.5 * norm_v3
                
            # Candidate union (top-20)
            if pw == 0.0:
                final_scores = global_scores
                top_indices = np.argsort(-final_scores)
            else:
                cand_k = 20
                cand_v2 = np.argsort(-cos_v2)[:cand_k]
                cand_v3 = np.argsort(-cos_v3)[:cand_k]
                cand_union = list(dict.fromkeys(list(cand_v2) + list(cand_v3)))
                
                # Rerank candidates
                rerank_scores = np.zeros(len(cat_ids), dtype=np.float32)
                for c_idx in cand_union:
                    c_id = cat_ids[c_idx]
                    c_patch = v3_patches[c_idx]
                    c_fg = cat_fg_masks.get(c_id) if use_fg else None
                    q_mask = q_fg if use_fg else None
                    
                    s_chamfer = compute_chamfer_similarity(
                        q_patch, c_patch,
                        query_fg_indices=q_mask,
                        cand_fg_indices=c_fg,
                        spatial_weight=spat
                    )
                    
                    s_fused = (1.0 - pw) * global_scores[c_idx] + pw * s_chamfer
                    rerank_scores[c_idx] = s_fused
                    
                top_indices = sorted(cand_union, key=lambda idx: rerank_scores[idx], reverse=True)
                
            t_total_ms = (time.perf_counter() - t_start) * 1000 + qd["timing"]["enc_v2_ms"] + qd["timing"]["enc_v3_ms"]
            latencies.append(t_total_ms)
            
            ranked_ids = [cat_ids[idx] for idx in top_indices]
            rank = None
            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1
                
            results_all.append({
                "query_id": q_meta["query_id"],
                "category": q_meta["category"],
                "target_id": target_id,
                "rank": rank,
                "reciprocal_rank": 1.0 / rank if rank else 0.0,
                "hit_at_1": rank == 1 if rank else False,
                "hit_at_5": (rank is not None and rank <= 5),
                "hit_at_10": (rank is not None and rank <= 10)
            })
            
        # Metric computations
        def compute_split_metrics(items: List[Dict]):
            n = len(items)
            r1 = sum(1 for x in items if x["hit_at_1"]) / n * 100
            r5 = sum(1 for x in items if x["hit_at_5"]) / n * 100
            r10 = sum(1 for x in items if x["hit_at_10"]) / n * 100
            mrr = float(np.mean([x["reciprocal_rank"] for x in items]))
            return {"R@1": round(r1, 2), "R@5": round(r5, 2), "R@10": round(r10, 2), "MRR": round(mrr, 4)}
            
        overall_m = compute_split_metrics(results_all)
        neck_m = compute_split_metrics([x for x in results_all if x["category"] == "necklace"])
        ring_m = compute_split_metrics([x for x in results_all if x["category"] == "ring"])
        avg_lat = round(float(np.mean(latencies)), 1)
        
        cfg_summary = {
            "key": cfg_key,
            "name": cfg_name,
            "overall": overall_m,
            "necklaces": neck_m,
            "rings": ring_m,
            "mean_latency_ms": avg_lat
        }
        report["configurations"].append(cfg_summary)
        
        print(f"{cfg_name:<38} | {overall_m['R@1']:>5.2f}% ({sum(1 for x in results_all if x['hit_at_1'])}/60) | {overall_m['R@5']:>5.2f}% | {overall_m['MRR']:>6.4f} | {neck_m['R@1']:>5.2f}% ({sum(1 for x in results_all if x['category']=='necklace' and x['hit_at_1'])}/30) | {ring_m['R@1']:>5.2f}% ({sum(1 for x in results_all if x['category']=='ring' and x['hit_at_1'])}/30) | {avg_lat:>5.1f}ms")
        
    out_eval = Path("output/eval_balanced_benchmark.json")
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    with open(out_eval, "w") as fp:
        json.dump(report, fp, indent=2)
    print("="*85)
    print(f"\nDetailed evaluation results saved to {out_eval}")

if __name__ == "__main__":
    run_evaluation()
