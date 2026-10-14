"""Evaluation protocol for EXP-007: Soft Spatial Attention & Patch Token Reweighting.

Controlled empirical comparison across:
1. Baseline Global CLS (DINOv2 / DINOv3)
2. Uniform Patch Pooling (Config A)
3. Foreground Saliency Weighting (Config B)
4. Soft Component Region Weighting (Config C)
5. Combined Soft Weighting (Config E)
6. Fused Global + Soft Combined (alpha=0.3, alpha=0.5)
7. Hard-Crop Retrieval (Negative Control)
8. Diagnostic Region Oracle (Diagnostic Upper Bound)

Reporting:
- R@1, R@5, R@10, Candidate R@20, MRR (with numerators & denominators)
- Subgroups: All 60, Real 34, Real Necklaces 30, Real Rings 4, Synthetic Rings 26
- Hard Negatives (N=30) and Singletons (N=30) calibration separation
- Sequential p50/p95 latency and batch throughput
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image

from engine.embeddings import get_encoder
from engine.localization import JewelleryLocalizer
from engine.soft_attention import (
    compute_uniform_weights,
    compute_foreground_weights,
    compute_soft_region_weights,
    compute_combined_weights,
    aggregate_patch_tokens
)

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

    index_dir = Path("data/index")
    with open(index_dir / "dinov2_meta.json", "r") as fp:
        catalog_meta = json.load(fp)
    cat_ids = [m["id"] for m in catalog_meta]
    cat_categories = [m.get("category", "") for m in catalog_meta]

    localizer = JewelleryLocalizer()
    query_imgs = []
    query_boxes = []
    query_oracle_boxes = []
    query_is_fallback = []
    query_crops = []

    print(f"Pre-processing and localizing {len(queries)} queries...")
    loc_fallbacks = 0
    for q in queries:
        qid = q["query_id"]
        cat = q["category"]
        qpath = Path(q["query_path"])
        img = Image.open(qpath).convert("RGB")
        query_imgs.append(img)

        loc_res = localizer.locate(img, category=cat)
        query_boxes.append(loc_res.box)
        query_is_fallback.append(loc_res.is_fallback)
        query_crops.append(loc_res.crop)
        if loc_res.is_fallback:
            loc_fallbacks += 1

        gt_box = reg_annotations.get(qid, {}).get("ground_truth_box", loc_res.box)
        query_oracle_boxes.append(gt_box)

    print(f"Localization complete. Fallbacks: {loc_fallbacks}/{len(queries)}")

    models = ["dinov2", "dinov3"]
    all_results = {}

    for model_name in models:
        print(f"\n==========================================")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"==========================================")
        encoder = get_encoder(model_name)

        # Batch encode queries
        print(f"Encoding query images for {model_name}...")
        t0 = time.time()
        q_global_embs = encoder.encode_batch(query_imgs, batch_size=32)
        q_patch_tokens = encoder.get_patch_tokens_batch(query_imgs, batch_size=32) # (60, 256, 384)
        q_crop_embs = encoder.encode_batch(query_crops, batch_size=32)
        t_batch_enc = time.time() - t0
        amortized_ms = (t_batch_enc / (3 * len(queries))) * 1000

        # Normalize query global and crop embs
        def norm_mat(m):
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            return m / norms

        q_global_embs = norm_mat(q_global_embs)
        q_crop_embs = norm_mat(q_crop_embs)

        # Compute query spatial weights
        q_w_uniform = np.array([compute_uniform_weights() for _ in query_imgs], dtype=np.float32)
        q_w_fg = np.array([compute_foreground_weights(img) for img in query_imgs], dtype=np.float32)
        q_w_reg = np.array([compute_soft_region_weights(box) for box in query_boxes], dtype=np.float32)
        q_w_comb = np.array([compute_combined_weights(img, box, is_fallback=fb) for img, box, fb in zip(query_imgs, query_boxes, query_is_fallback)], dtype=np.float32)
        q_w_oracle = np.array([compute_combined_weights(img, box, is_fallback=False) for img, box in zip(query_imgs, query_oracle_boxes)], dtype=np.float32)

        # Aggregate query representations
        q_rep_uniform = aggregate_patch_tokens(q_patch_tokens, q_w_uniform)
        q_rep_fg = aggregate_patch_tokens(q_patch_tokens, q_w_fg)
        q_rep_reg = aggregate_patch_tokens(q_patch_tokens, q_w_reg)
        q_rep_comb = aggregate_patch_tokens(q_patch_tokens, q_w_comb)
        q_rep_oracle = aggregate_patch_tokens(q_patch_tokens, q_w_oracle)

        # Load catalog matrices
        cat_global = np.load(index_dir / f"{model_name}_embeddings.npy")
        cat_uniform = np.load(index_dir / f"{model_name}_soft_uniform.npy")
        cat_fg = np.load(index_dir / f"{model_name}_soft_foreground.npy")
        cat_reg = np.load(index_dir / f"{model_name}_soft_region.npy")
        cat_comb = np.load(index_dir / f"{model_name}_soft_combined.npy")
        cat_crops = np.load(index_dir / f"{model_name}_region_embeddings.npy")

        # Configurations to evaluate
        configs = [
            ("Baseline Global (CLS)", q_global_embs, cat_global),
            ("Config A: Uniform Patch Pooling", q_rep_uniform, cat_uniform),
            ("Config B: Foreground Saliency", q_rep_fg, cat_fg),
            ("Config C: Soft Component Region", q_rep_reg, cat_reg),
            ("Config E: Combined Soft Weighting", q_rep_comb, cat_comb),
            ("Fused Global + Combined (alpha=0.3)", norm_mat(0.70 * q_global_embs + 0.30 * q_rep_comb), norm_mat(0.70 * cat_global + 0.30 * cat_comb)),
            ("Fused Global + Combined (alpha=0.5)", norm_mat(0.50 * q_global_embs + 0.50 * q_rep_comb), norm_mat(0.50 * cat_global + 0.50 * cat_comb)),
            ("Negative Control: Hard-Crop", q_crop_embs, cat_crops),
            ("Diagnostic Oracle: Ground-Truth Region", q_rep_oracle, cat_comb)
        ]

        model_results = {}

        for cfg_name, Q_mat, C_mat in configs:
            sim_matrix = np.dot(Q_mat, C_mat.T) # (60, 432)
            per_query_records = []

            for q_idx, q in enumerate(queries):
                qid = q["query_id"]
                target_id = q["target_catalog_id"]
                cat = q["category"]
                group = get_group(q)

                scores = sim_matrix[q_idx].copy()
                valid_mask = np.array([c.lower() == cat.lower() for c in cat_categories])
                scores = np.where(valid_mask, scores, -1e9)

                ranked_idx = np.argsort(scores)[::-1]
                ranked_ids = [cat_ids[i] for i in ranked_idx]

                rank = -1
                if target_id in ranked_ids:
                    rank = ranked_ids.index(target_id) + 1

                reciprocal_rank = 1.0 / rank if rank > 0 else 0.0
                r1 = 1 if rank == 1 else 0
                r5 = 1 if 0 < rank <= 5 else 0
                r10 = 1 if 0 < rank <= 10 else 0
                r20 = 1 if 0 < rank <= 20 else 0

                per_query_records.append({
                    "query_id": qid,
                    "target_id": target_id,
                    "group": group,
                    "category": cat,
                    "rank": rank,
                    "r1": r1,
                    "r5": r5,
                    "r10": r10,
                    "r20": r20,
                    "mrr": reciprocal_rank
                })

            def calc_metrics(recs):
                n = len(recs)
                if n == 0:
                    return {"count": 0, "r1": 0.0, "r5": 0.0, "r10": 0.0, "r20": 0.0, "mrr": 0.0,
                            "r1_num": "0/0", "r5_num": "0/0", "r20_num": "0/0"}
                c1 = sum(r["r1"] for r in recs)
                c5 = sum(r["r5"] for r in recs)
                c10 = sum(r["r10"] for r in recs)
                c20 = sum(r["r20"] for r in recs)
                mrr = sum(r["mrr"] for r in recs) / n
                return {
                    "count": n,
                    "r1": round(c1 / n, 4),
                    "r5": round(c5 / n, 4),
                    "r10": round(c10 / n, 4),
                    "r20": round(c20 / n, 4),
                    "mrr": round(mrr, 4),
                    "r1_num": f"{c1}/{n}",
                    "r5_num": f"{c5}/{n}",
                    "r10_num": f"{c10}/{n}",
                    "r20_num": f"{c20}/{n}"
                }

            neck_recs = [r for r in per_query_records if r["group"] == "real_necklace"]
            ring_real_recs = [r for r in per_query_records if r["group"] == "real_ring"]
            real_all_recs = neck_recs + ring_real_recs
            syn_recs = [r for r in per_query_records if r["group"] == "synthetic_ring"]

            summary = {
                "config_name": cfg_name,
                "overall_all_60": calc_metrics(per_query_records),
                "real_queries_all_34": calc_metrics(real_all_recs),
                "real_necklaces_30": calc_metrics(neck_recs),
                "real_rings_4": calc_metrics(ring_real_recs),
                "synthetic_rings_26": calc_metrics(syn_recs),
                "amortized_batch_latency_ms": round(amortized_ms, 2)
            }

            model_results[cfg_name] = summary
            print(f"{cfg_name:38s} | Overall R@1: {summary['overall_all_60']['r1_num']} ({summary['overall_all_60']['r1']:5.1%}) | Real: {summary['real_queries_all_34']['r1_num']} ({summary['real_queries_all_34']['r1']:5.1%}) [Neck: {summary['real_necklaces_30']['r1_num']}, Ring: {summary['real_rings_4']['r1_num']}] | Syn: {summary['synthetic_rings_26']['r1_num']} ({summary['synthetic_rings_26']['r1']:5.1%}) | R@20: {summary['overall_all_60']['r20']:5.1%} | MRR: {summary['overall_all_60']['mrr']:.4f}")

        all_results[model_name] = model_results

    out_file = Path("output/eval_soft_attention.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(all_results, fp, indent=2)

    print(f"\nSaved full evaluation results to {out_file}")

if __name__ == "__main__":
    run_evaluation()
