"""Optimized evaluation protocol for EXP-006: Gemstone & Pendant Localization for Jewellery Retrieval.

Precomputes query embeddings in batch mode for instantaneous evaluation of all 8 configurations across DINOv2 and DINOv3.

Configurations:
1. Baseline A: Full-image retrieval only (DINOv2 / DINOv3)
2. Baseline B: Region-only retrieval (DINOv2 / DINOv3)
3. Config C1: Combined score fusion w=0.10
4. Config C2: Combined score fusion w=0.20
5. Config C3: Combined score fusion w=0.35
6. Config C4: Combined score fusion w=0.50
7. Config D: Region reranking on top-20 global candidates (w=0.35)
8. Config E: Oracle Ground-Truth ROI Upper Bound (w=0.35)

Sub-group breakdown:
- All Same-SKU Queries (N=60)
- Real Necklaces (N=30)
- Real Rings (N=4)
- Overall Real Queries (N=34)
- Synthetic Robustness Rings (N=26)
- Hard Negatives (N=30)
- Singletons (N=30)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image

from engine.region_search import RegionAwareSearchEngine
from engine.localization import JewelleryLocalizer, compute_iou

def run_evaluation():
    bench_file = Path("data/evaluation/benchmark_v3_balanced.json")
    with open(bench_file, "r") as fp:
        bench = json.load(fp)

    reg_ann_file = Path("data/region_annotations.json")
    with open(reg_ann_file, "r") as fp:
        reg_annotations = json.load(fp)["annotations"]

    queries = bench["same_sku_queries"]
    hard_negs = bench.get("hard_negatives", [])
    singletons = bench.get("singletons", [])

    def get_group(q):
        qid = q["query_id"]
        cat = q["category"]
        prov = q.get("provenance", "")
        if "synthetic" in prov or "var_" in qid:
            return "synthetic_ring"
        elif cat == "necklace":
            return "real_necklace"
        else:
            return "real_ring"

    models_to_test = ["dinov2", "dinov3"]
    configurations = [
        {"name": "Full-Image (Baseline A)", "mode": "full", "weight": 0.0, "use_oracle": False},
        {"name": "Region-Only (Baseline B)", "mode": "region", "weight": 1.0, "use_oracle": False},
        {"name": "Combined w=0.10", "mode": "combined", "weight": 0.10, "use_oracle": False},
        {"name": "Combined w=0.20", "mode": "combined", "weight": 0.20, "use_oracle": False},
        {"name": "Combined w=0.35 (Config C)", "mode": "combined", "weight": 0.35, "use_oracle": False},
        {"name": "Combined w=0.50", "mode": "combined", "weight": 0.50, "use_oracle": False},
        {"name": "Region Rerank Top-20 (Config D)", "mode": "rerank", "weight": 0.35, "use_oracle": False},
        {"name": "Oracle Ground-Truth ROI (Config E)", "mode": "oracle", "weight": 0.35, "use_oracle": True},
    ]

    # Pre-load query images, crops, and annotations
    localizer = JewelleryLocalizer()
    query_imgs = []
    query_heuristic_crops = []
    query_oracle_crops = []
    query_loc_info = []
    
    t_loc_start = time.time()
    for q in queries:
        qid = q["query_id"]
        cat = q["category"]
        qpath = Path(q["query_path"])
        img = Image.open(qpath).convert("RGB")
        w, h = img.size
        query_imgs.append(img)

        # Heuristic crop
        loc_res = localizer.locate(img, category=cat)
        query_heuristic_crops.append(loc_res.crop)

        # Oracle crop
        if qid in reg_annotations:
            gt_box = reg_annotations[qid]["ground_truth_box"]
            y1 = max(0, int(gt_box[0] * h))
            x1 = max(0, int(gt_box[1] * w))
            y2 = min(h, int(gt_box[2] * h))
            x2 = min(w, int(gt_box[3] * w))
            oracle_crop = img.crop((x1, y1, x2, y2))
            iou = compute_iou(loc_res.box, gt_box)
        else:
            oracle_crop = loc_res.crop
            iou = 1.0
        query_oracle_crops.append(oracle_crop)

        query_loc_info.append({
            "is_fallback": loc_res.is_fallback,
            "pred_box": loc_res.box,
            "iou": iou
        })

    t_loc_total = time.time() - t_loc_start
    avg_loc_ms = (t_loc_total / len(queries)) * 1000

    all_results = {}

    for model_name in models_to_test:
        print(f"\n==========================================")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"==========================================")
        engine = RegionAwareSearchEngine(model_name=model_name)
        cat_global_embs = engine.catalog_index.embeddings # (432, D)
        cat_reg_embs = engine.catalog_region_embeddings   # (432, D)
        meta = engine.catalog_index.metadata
        cat_ids = [m["id"] for m in meta]
        cat_categories = [m.get("category", "") for m in meta]

        # Batch encode queries
        print(f"Encoding {len(queries)} query full-images in batch...")
        t_enc_0 = time.time()
        q_global_embs = engine.encoder.encode_batch(query_imgs, batch_size=32)
        q_reg_embs = engine.encoder.encode_batch(query_heuristic_crops, batch_size=32)
        q_oracle_embs = engine.encoder.encode_batch(query_oracle_crops, batch_size=32)
        t_enc_total = time.time() - t_enc_0
        avg_enc_ms = (t_enc_total / (3 * len(queries))) * 1000

        # L2 normalize query embeddings
        def normalize(e):
            n = np.linalg.norm(e, axis=1, keepdims=True)
            n[n == 0] = 1e-9
            return e / n

        q_global_embs = normalize(q_global_embs)
        q_reg_embs = normalize(q_reg_embs)
        q_oracle_embs = normalize(q_oracle_embs)

        # Precompute similarity matrices: (N_queries, N_catalog)
        sim_global = np.dot(q_global_embs, cat_global_embs.T)
        sim_reg = np.dot(q_reg_embs, cat_reg_embs.T)
        sim_oracle = np.dot(q_oracle_embs, cat_reg_embs.T)

        model_results = {}

        for cfg in configurations:
            cfg_name = cfg["name"]
            mode = cfg["mode"]
            weight = cfg["weight"]
            use_oracle = cfg["use_oracle"]

            t_score_start = time.time()
            per_query_records = []
            
            for q_idx, q in enumerate(queries):
                qid = q["query_id"]
                target_id = q["target_catalog_id"]
                cat = q["category"]
                group = get_group(q)
                loc_info = query_loc_info[q_idx]
                iou = loc_info["iou"]

                g_scores = sim_global[q_idx]
                r_scores = sim_oracle[q_idx] if use_oracle else sim_reg[q_idx]

                # Fallback check: if fallback was triggered and not region-only, weight is 0.0
                effective_w = 0.0 if (loc_info["is_fallback"] and mode != "region") else float(weight)

                # Category filtering mask
                valid_mask = np.array([c.lower() == cat.lower() for c in cat_categories])

                if mode == "full":
                    fused = g_scores.copy()
                elif mode == "region":
                    fused = r_scores.copy()
                elif mode in ("combined", "oracle"):
                    fused = (1.0 - effective_w) * g_scores + effective_w * r_scores
                elif mode == "rerank":
                    # Masked global top 20
                    masked_g = np.where(valid_mask, g_scores, -1e9)
                    top_cand_idx = np.argsort(masked_g)[::-1][:20]
                    fused = np.full_like(g_scores, -1e9)
                    for c_i in top_cand_idx:
                        fused[c_i] = (1.0 - effective_w) * g_scores[c_i] + effective_w * r_scores[c_i]
                else:
                    fused = g_scores.copy()

                # Apply category mask
                fused = np.where(valid_mask, fused, -1e9)

                # Rank candidates
                ranked_idx = np.argsort(fused)[::-1]
                ranked_ids = [cat_ids[i] for i in ranked_idx[:20]]

                rank = -1
                if target_id in ranked_ids:
                    rank = ranked_ids.index(target_id) + 1

                reciprocal_rank = 1.0 / rank if rank > 0 else 0.0
                r1 = 1 if rank == 1 else 0
                r5 = 1 if 0 < rank <= 5 else 0
                r10 = 1 if 0 < rank <= 10 else 0

                per_query_records.append({
                    "query_id": qid,
                    "target_id": target_id,
                    "group": group,
                    "category": cat,
                    "rank": rank,
                    "r1": r1,
                    "r5": r5,
                    "r10": r10,
                    "mrr": reciprocal_rank,
                    "iou": iou
                })

            avg_score_ms = ((time.time() - t_score_start) / len(queries)) * 1000

            def calc_metrics(records):
                if not records:
                    return {"count": 0, "r1": 0.0, "r5": 0.0, "r10": 0.0, "mrr": 0.0}
                n = len(records)
                return {
                    "count": n,
                    "r1": round(sum(r["r1"] for r in records) / n, 4),
                    "r5": round(sum(r["r5"] for r in records) / n, 4),
                    "r10": round(sum(r["r10"] for r in records) / n, 4),
                    "mrr": round(sum(r["mrr"] for r in records) / n, 4)
                }

            neck_records = [r for r in per_query_records if r["group"] == "real_necklace"]
            ring_real_records = [r for r in per_query_records if r["group"] == "real_ring"]
            real_all_records = neck_records + ring_real_records
            syn_records = [r for r in per_query_records if r["group"] == "synthetic_ring"]

            ious_list = [r["iou"] for r in per_query_records]

            summary = {
                "config_name": cfg_name,
                "mode": mode,
                "weight": weight,
                "use_oracle": use_oracle,
                "overall_all_60": calc_metrics(per_query_records),
                "real_queries_all_34": calc_metrics(real_all_records),
                "real_necklaces_30": calc_metrics(neck_records),
                "real_rings_4": calc_metrics(ring_real_records),
                "synthetic_rings_26": calc_metrics(syn_records),
                "mean_iou": round(float(np.mean(ious_list)), 4) if ious_list else 0.0,
                "timings_ms": {
                    "localization": round(avg_loc_ms, 2),
                    "encoding": round(avg_enc_ms, 2),
                    "scoring": round(avg_score_ms, 2),
                    "total": round(avg_loc_ms + 2 * avg_enc_ms + avg_score_ms, 2)
                }
            }

            model_results[cfg_name] = summary
            print(f"{cfg_name:35s} | Overall R@1: {summary['overall_all_60']['r1']:5.1%} | Real R@1: {summary['real_queries_all_34']['r1']:5.1%} (Neck: {summary['real_necklaces_30']['r1']:5.1%}, Rings: {summary['real_rings_4']['r1']:5.1%}) | Syn R@1: {summary['synthetic_rings_26']['r1']:5.1%} | Total Lat: {summary['timings_ms']['total']}ms")

        # Evaluate Hard Negatives & Singletons for calibration check
        # (Using top-1 score separation in Combined w=0.35 mode)
        print(f"\nEvaluating calibration on Hard Negatives (N={len(hard_negs)}) and Singletons (N={len(singletons)})...")
        hn_top1_scores = []
        for hn in hard_negs:
            q_file = Path("Jewellery_Data") / hn["cat"] / f"{hn['q']}.jpg"
            if q_file.exists():
                img = Image.open(q_file).convert("RGB")
                loc = localizer.locate(img, category=hn["cat"])
                q_g = engine.encoder.encode_image(img)
                q_r = engine.encoder.encode_image(loc.crop)
                q_g = q_g / np.linalg.norm(q_g)
                q_r = q_r / np.linalg.norm(q_r)
                g_sc = np.dot(cat_global_embs, q_g)
                r_sc = np.dot(cat_reg_embs, q_r)
                fused = 0.65 * g_sc + 0.35 * r_sc
                # exclude self if in catalog
                for c_i, c_id in enumerate(cat_ids):
                    if c_id == hn["q"]:
                        fused[c_i] = -1e9
                hn_top1_scores.append(float(np.max(fused)))

        singleton_top1_scores = []
        for s in singletons:
            q_file = Path(s["query_path"])
            if q_file.exists():
                img = Image.open(q_file).convert("RGB")
                loc = localizer.locate(img, category=s["category"])
                q_g = engine.encoder.encode_image(img)
                q_r = engine.encoder.encode_image(loc.crop)
                q_g = q_g / np.linalg.norm(q_g)
                q_r = q_r / np.linalg.norm(q_r)
                g_sc = np.dot(cat_global_embs, q_g)
                r_sc = np.dot(cat_reg_embs, q_r)
                fused = 0.65 * g_sc + 0.35 * r_sc
                singleton_top1_scores.append(float(np.max(fused)))

        calibration_stats = {
            "hard_negative_count": len(hn_top1_scores),
            "mean_hn_top1_score": round(float(np.mean(hn_top1_scores)), 4) if hn_top1_scores else 0.0,
            "max_hn_top1_score": round(float(np.max(hn_top1_scores)), 4) if hn_top1_scores else 0.0,
            "singleton_count": len(singleton_top1_scores),
            "mean_singleton_top1_score": round(float(np.mean(singleton_top1_scores)), 4) if singleton_top1_scores else 0.0
        }
        print(f"Calibration: Hard Neg Mean Top-1={calibration_stats['mean_hn_top1_score']:.4f}, Singletons Mean Top-1={calibration_stats['mean_singleton_top1_score']:.4f}")
        model_results["calibration_diagnostics"] = calibration_stats
        all_results[model_name] = model_results

    out_file = Path("output/eval_region_retrieval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(all_results, fp, indent=2)

    print(f"\n==========================================")
    print(f"EVALUATION COMPLETE. Saved to {out_file}")
    print(f"==========================================")

if __name__ == "__main__":
    run_evaluation()
