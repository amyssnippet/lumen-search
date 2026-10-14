#!/usr/bin/env python3
"""EXP-005 Phase 1: Audit score fusion strategies before adding new models.

Compares:
1. DINOv2 Global
2. DINOv3 Global
3. Rank-based Reciprocal Rank Fusion (RRF)
4. Raw Cosine Average: 0.5 * cos_v2 + 0.5 * cos_v3
5. Calibrated Min-Max Score Fusion: 0.5 * norm(cos_v2) + 0.5 * norm(cos_v3)
6. Calibrated Logistic / Z-Score Fusion

Measures Candidate Recall@5, @10, @20 and MRR before local reranking.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def audit_fusion():
    manifest = json.load(open("data/manifest.json"))
    images = manifest["images"]
    id_to_img = {img["id"]: img for img in images}
    
    # Load catalog indices
    catalog_meta = json.load(open("data/index/dinov2_meta.json"))
    catalog_ids = [item["id"] for item in catalog_meta]
    v2_index = np.load("data/index/dinov2_embeddings.npy")
    v3_index = np.load("data/index/dinov3_embeddings.npy")
    
    # Load validation calibration parameters from output/calibration_report.json
    calib_data = json.load(open("output/calibration_report.json"))
    v2_pos_mean = calib_data["dinov2"]["score_distribution"]["pos_mean"]
    v2_neg_mean = calib_data["dinov2"]["score_distribution"]["neg_mean"]
    v3_pos_mean = calib_data["dinov3"]["score_distribution"]["pos_mean"]
    v3_neg_mean = calib_data["dinov3"]["score_distribution"]["neg_mean"]
    
    print("=== Validation Calibration Stats ===")
    print(f"DINOv2: pos_mean={v2_pos_mean:.4f}, neg_mean={v2_neg_mean:.4f}, span={v2_pos_mean - v2_neg_mean:.4f}")
    print(f"DINOv3: pos_mean={v3_pos_mean:.4f}, neg_mean={v3_neg_mean:.4f}, span={v3_pos_mean - v3_neg_mean:.4f}")
    
    # Load test queries (Benchmark v1 and Corrected v2)
    test_imgs = [img for img in images if img.get("split") == "test"]
    cluster_to_members = {}
    for img in images:
        cid = img["cluster_id"]
        cluster_to_members.setdefault(cid, []).append(img["id"])
        
    twin_queries_v1 = [img for img in test_imgs if len(cluster_to_members[img["cluster_id"]]) > 1]
    twin_queries_v2 = [img for img in twin_queries_v1 if img["id"] != "ring_083"]
    
    print(f"\nEvaluating on Benchmark v1 (N={len(twin_queries_v1)}) and Corrected v2 (N={len(twin_queries_v2)})")
    
    # Pre-extract query embeddings for test queries using index or encoder
    from engine.embeddings import DINOv2Encoder, DINOv3Encoder
    v2_enc = DINOv2Encoder()
    v3_enc = DINOv3Encoder()
    
    query_feats_v2 = {}
    query_feats_v3 = {}
    for q in twin_queries_v1:
        from PIL import Image
        with Image.open(q["rel_path"]) as img:
            img_rgb = img.convert("RGB")
            query_feats_v2[q["id"]] = v2_enc.encode_image(img_rgb)
            query_feats_v3[q["id"]] = v3_enc.encode_image(img_rgb)
            
    def evaluate_strategy(strategy_name: str, score_fn, queries: List[Dict]):
        r1, r5, r10, r20 = 0, 0, 0, 0
        rr_list = []
        
        for q in queries:
            qid = q["id"]
            q_cid = q["cluster_id"]
            # Target IDs in catalog
            target_ids = set(m for m in cluster_to_members[q_cid] if m != qid and m in catalog_ids)
            if not target_ids:
                continue
                
            q_v2 = query_feats_v2[qid]
            q_v3 = query_feats_v3[qid]
            
            cos_v2 = np.dot(v2_index, q_v2)
            cos_v3 = np.dot(v3_index, q_v3)
            
            # Compute scores across all catalog items
            final_scores = score_fn(cos_v2, cos_v3)
            
            ranked_indices = np.argsort(-final_scores)
            ranked_ids = [catalog_ids[i] for i in ranked_indices]
            
            # Find rank of first target
            rank = None
            for idx, cid in enumerate(ranked_ids):
                if cid in target_ids:
                    rank = idx + 1
                    break
                    
            if rank is not None:
                rr_list.append(1.0 / rank)
                if rank == 1:
                    r1 += 1
                if rank <= 5:
                    r5 += 1
                if rank <= 10:
                    r10 += 1
                if rank <= 20:
                    r20 += 1
            else:
                rr_list.append(0.0)
                
        n = len(queries)
        return {
            "strategy": strategy_name,
            "queries": n,
            "R@1": round(r1 / n * 100, 2),
            "R@5": round(r5 / n * 100, 2),
            "R@10": round(r10 / n * 100, 2),
            "R@20": round(r20 / n * 100, 2),
            "MRR": round(float(np.mean(rr_list)), 4)
        }
        
    strategies = [
        ("DINOv2 Global", lambda c2, c3: c2),
        ("DINOv3 Global", lambda c2, c3: c3),
        ("Raw Cosine Average", lambda c2, c3: 0.5 * c2 + 0.5 * c3),
        ("Rank-based RRF (k=60)", lambda c2, c3: (
            np.zeros_like(c2) +
            (1.0 / (60.0 + np.argsort(np.argsort(-c2)) + 1)) +
            (1.0 / (60.0 + np.argsort(np.argsort(-c3)) + 1))
        )),
        ("Calibrated Min-Max Fusion", lambda c2, c3: (
            0.5 * np.clip((c2 - v2_neg_mean) / (v2_pos_mean - v2_neg_mean), 0, 1) +
            0.5 * np.clip((c3 - v3_neg_mean) / (v3_pos_mean - v3_neg_mean), 0, 1)
        )),
        ("Calibrated Z-Score Fusion", lambda c2, c3: (
            0.5 * ((c2 - v2_neg_mean) / 0.084) +
            0.5 * ((c3 - v3_neg_mean) / 0.046)
        )),
    ]
    
    results = {"benchmark_v1": [], "corrected_v2": []}
    
    print("\n--- Benchmark v1 (Historical, N=30) ---")
    for name, s_fn in strategies:
        res = evaluate_strategy(name, s_fn, twin_queries_v1)
        results["benchmark_v1"].append(res)
        print(f"{res['strategy']:<26}: R@1={res['R@1']}% R@5={res['R@5']}% R@10={res['R@10']}% R@20={res['R@20']}% MRR={res['MRR']}")
        
    print("\n--- Corrected Benchmark v2 (N=29) ---")
    for name, s_fn in strategies:
        res = evaluate_strategy(name, s_fn, twin_queries_v2)
        results["corrected_v2"].append(res)
        print(f"{res['strategy']:<26}: R@1={res['R@1']}% R@5={res['R@5']}% R@10={res['R@10']}% R@20={res['R@20']}% MRR={res['MRR']}")
        
    out_path = Path("output/audit_fusion.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nAudit results saved to {out_path}")

if __name__ == "__main__":
    audit_fusion()
