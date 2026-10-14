#!/usr/bin/env python3
"""EXP-005 Phase 4: Revalidate confidence calibration on balanced dataset with hard negatives.

Evaluates score distributions, false acceptance rate, coverage, and precision
across DINOv2, DINOv3, Calibrated Min-Max Global, and Hybrid Rerank configurations.
"""

import json
import math
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

from engine.embeddings import DINOv2Encoder, DINOv3Encoder
from engine.reranker import compute_chamfer_similarity, get_foreground_patch_indices

def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + (z**2) / (4 * total)) / total) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))

def run_recalibration():
    # Load balanced benchmark
    bench = json.load(open("data/evaluation/benchmark_v3_balanced.json"))
    same_sku = bench["same_sku_queries"]
    hard_negs = bench["hard_negatives"]
    singletons = bench["singletons"]
    
    # Catalog index
    cat_meta = json.load(open("data/index/dinov2_meta.json"))
    cat_id_to_idx = {m["id"]: idx for idx, m in enumerate(cat_meta)}
    v2_index = np.load("data/index/dinov2_embeddings.npy")
    v3_index = np.load("data/index/dinov3_embeddings.npy")
    v3_patches = np.load("data/index/dinov3_patches.npy")
    patch_meta = json.load(open("data/index/dinov3_patches.meta.json"))
    cat_fg_masks = {
        cid: np.array(mask, dtype=np.int32)
        for cid, mask in zip(patch_meta["catalog_ids"], patch_meta["foreground_masks"])
    }
    
    # Validation normalization parameters
    v2_neg_mean, v2_pos_mean = 0.4372, 0.9825
    v3_neg_mean, v3_pos_mean = 0.7924, 0.9927
    
    print("Loading vision encoders...")
    v2_enc = DINOv2Encoder()
    v3_enc = DINOv3Encoder()
    
    def score_pair(q_path: str, cat_id: str, patch_weight: float = 0.2, spatial_weight: float = 0.1) -> Dict[str, float]:
        with Image.open(q_path) as q_img:
            q_rgb = q_img.convert("RGB")
            q_v2 = v2_enc.encode_image(q_rgb)
            q_v3 = v3_enc.encode_image(q_rgb)
            q_patch = v3_enc.get_patch_tokens(q_rgb)
            
        if cat_id in cat_id_to_idx:
            c_idx = cat_id_to_idx[cat_id]
            c_v2 = v2_index[c_idx]
            c_v3 = v3_index[c_idx]
            c_patch = v3_patches[c_idx]
            c_fg_idx = cat_fg_masks.get(cat_id)
        else:
            c_path = id_to_rel_path(cat_id)
            with Image.open(c_path) as c_img:
                c_rgb = c_img.convert("RGB")
                c_v2 = v2_enc.encode_image(c_rgb)
                c_v3 = v3_enc.encode_image(c_rgb)
                c_patch = v3_enc.get_patch_tokens(c_rgb)
                c_fg_idx = get_foreground_patch_indices(c_rgb)
        
        s_v2 = float(np.dot(q_v2, c_v2))
        s_v3 = float(np.dot(q_v3, c_v3))
        
        norm_v2 = np.clip((s_v2 - v2_neg_mean) / (v2_pos_mean - v2_neg_mean), 0, 1)
        norm_v3 = np.clip((s_v3 - v3_neg_mean) / (v3_pos_mean - v3_neg_mean), 0, 1)
        s_minmax = float(0.5 * norm_v2 + 0.5 * norm_v3)
        s_raw_avg = float(0.5 * s_v2 + 0.5 * s_v3)
        
        # Patch chamfer
        q_fg_idx = get_foreground_patch_indices(q_rgb)
        s_chamfer = compute_chamfer_similarity(
            q_patch, c_patch,
            query_fg_indices=q_fg_idx,
            cand_fg_indices=c_fg_idx,
            spatial_weight=spatial_weight
        )
        
        s_rerank_02 = float((1.0 - 0.2) * s_minmax + 0.2 * s_chamfer)
        s_rerank_05 = float((1.0 - 0.5) * s_minmax + 0.5 * s_chamfer)
        
        return {
            "dinov2": s_v2,
            "dinov3": s_v3,
            "hybrid_minmax": s_minmax,
            "hybrid_raw_avg": s_raw_avg,
            "chamfer": s_chamfer,
            "rerank_pw02": s_rerank_02,
            "rerank_pw05": s_rerank_05
        }

    print("\nExtracting scores for Verified Same-SKU Positives (N=60)...")
    pos_scores = {k: [] for k in ["dinov2", "dinov3", "hybrid_minmax", "hybrid_raw_avg", "chamfer", "rerank_pw02", "rerank_pw05"]}
    for q in same_sku:
        sc = score_pair(q["query_path"], q["target_catalog_id"])
        for k, v in sc.items():
            pos_scores[k].append(v)
            
    print("Extracting scores for Confirmed Hard Negatives (N=30)...")
    hard_neg_scores = {k: [] for k in ["dinov2", "dinov3", "hybrid_minmax", "hybrid_raw_avg", "chamfer", "rerank_pw02", "rerank_pw05"]}
    for hn in hard_negs:
        sc = score_pair(id_to_rel_path(hn["q"]), hn["target"])
        for k, v in sc.items():
            hard_neg_scores[k].append(v)
            
    print("Extracting scores for Distractor Negatives (N=100)...")
    distractor_scores = {k: [] for k in ["dinov2", "dinov3", "hybrid_minmax", "hybrid_raw_avg", "chamfer", "rerank_pw02", "rerank_pw05"]}
    # 100 disjoint pairs
    calib_cache = json.load(open("output/calibration_report.json"))
    # Or sample randomly from catalog
    rng = np.random.RandomState(42)
    cat_items = list(cat_meta)
    for _ in range(100):
        i, j = rng.choice(len(cat_items), 2, replace=False)
        if cat_items[i]["category"] == cat_items[j]["category"]:
            continue
        sc = score_pair(cat_items[i]["rel_path"], cat_items[j]["id"])
        for k, v in sc.items():
            distractor_scores[k].append(v)

    # Combine negatives: 30 hard negatives + 100 distractors = 130 total negatives
    total_neg_scores = {}
    for k in pos_scores.keys():
        total_neg_scores[k] = hard_neg_scores[k] + distractor_scores[k]

    report = {"models": {}}
    configs = [
        ("dinov2", "DINOv2 Global"),
        ("dinov3", "DINOv3 Global"),
        ("hybrid_minmax", "Calibrated Min-Max Hybrid Global (pw=0.0)"),
        ("rerank_pw02", "Hybrid Rerank (pw=0.2, spat=0.1)"),
        ("rerank_pw05", "Hybrid Rerank (pw=0.5, spat=0.1)"),
        ("chamfer", "Chamfer Only (pw=1.0, spat=0.1)")
    ]

    print("\n=== CALIBRATION REPORT SUMMARY ===")
    for key, name in configs:
        pos_arr = np.array(pos_scores[key])
        hard_neg_arr = np.array(hard_neg_scores[key])
        dist_neg_arr = np.array(distractor_scores[key])
        all_neg_arr = np.array(total_neg_scores[key])
        
        # Threshold sweeps
        thresholds = np.linspace(0.70, 0.99, 59)
        best_tau = 0.70
        best_precision = 0.0
        calib_stats = []
        
        for tau in thresholds:
            tp = int((pos_arr >= tau).sum())
            fp = int((all_neg_arr >= tau).sum())
            fn = int((pos_arr < tau).sum())
            tn = int((all_neg_arr < tau).sum())
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / len(pos_arr) if len(pos_arr) > 0 else 0.0
            far = fp / len(all_neg_arr) if len(all_neg_arr) > 0 else 0.0
            abstention = (fn + tn) / (len(pos_arr) + len(all_neg_arr))
            
            calib_stats.append({
                "threshold": round(float(tau), 3),
                "precision": round(prec * 100, 2),
                "coverage": round(rec * 100, 2),
                "false_acceptance_rate": round(far * 100, 3),
                "abstention_rate": round(abstention * 100, 2),
                "tp": tp, "fp": fp
            })
            
        # Select threshold for 95% and 99% precision
        tau_95 = next((s["threshold"] for s in calib_stats if s["precision"] >= 95.0 and s["coverage"] > 0), 0.90)
        tau_99 = next((s["threshold"] for s in calib_stats if s["precision"] >= 99.0 and s["coverage"] > 0), 0.95)
        
        stat_at_99 = next((s for s in calib_stats if s["threshold"] == tau_99), calib_stats[-1])
        ci_prec = wilson_score_interval(stat_at_99["tp"], stat_at_99["tp"] + stat_at_99["fp"])
        
        report["models"][key] = {
            "name": name,
            "pos_distribution": {
                "mean": round(float(pos_arr.mean()), 4),
                "std": round(float(pos_arr.std()), 4),
                "count": len(pos_arr)
            },
            "hard_neg_distribution": {
                "mean": round(float(hard_neg_arr.mean()), 4),
                "std": round(float(hard_neg_arr.std()), 4),
                "count": len(hard_neg_arr)
            },
            "distractor_distribution": {
                "mean": round(float(dist_neg_arr.mean()), 4),
                "std": round(float(dist_neg_arr.std()), 4),
                "count": len(dist_neg_arr)
            },
            "tau_95": tau_95,
            "tau_99": tau_99,
            "performance_at_tau_99": {
                "threshold": tau_99,
                "precision": stat_at_99["precision"],
                "coverage": stat_at_99["coverage"],
                "false_acceptance_rate": stat_at_99["false_acceptance_rate"],
                "abstention_rate": stat_at_99["abstention_rate"],
                "wilson_95ci_precision": [round(ci_prec[0] * 100, 2), round(ci_prec[1] * 100, 2)]
            },
            "sweep": calib_stats
        }
        
        print(f"\n[{name}]")
        print(f"  Pos: {pos_arr.mean():.4f} +/- {pos_arr.std():.4f} | HardNeg: {hard_neg_arr.mean():.4f} | DistNeg: {dist_neg_arr.mean():.4f}")
        print(f"  tau_99={tau_99}: Precision={stat_at_99['precision']}% (95% CI: [{ci_prec[0]*100:.1f}%, {ci_prec[1]*100:.1f}%]), Coverage={stat_at_99['coverage']}%, FAR={stat_at_99['false_acceptance_rate']}%")

    out_file = Path("output/recalibration_balanced.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nRecalibration report saved to {out_file}")

def id_to_rel_path(img_id: str) -> str:
    manifest = json.load(open("data/manifest.json"))
    for img in manifest["images"]:
        if img["id"] == img_id:
            return img["rel_path"]
    for p in Path("Jewellery_Data").rglob(f"{img_id}.*"):
        if p.is_file():
            return str(p)
    cat = "ring" if "ring" in img_id else "necklace"
    return f"Jewellery_Data/{cat}/{img_id}.jpg"

if __name__ == "__main__":
    run_recalibration()
