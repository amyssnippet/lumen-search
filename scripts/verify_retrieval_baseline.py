#!/usr/bin/env python3
"""EXP-008: Baseline Retrieval Verification & Comparison Harness.

Reproduces and verifies exact retrieval metrics for:
- DINOv2 Global (CLS)
- DINOv3 Global (CLS)
- Calibrated Hybrid Global Fusion (Min-Max)
- Best Soft Spatial Attention (DINOv3 Combined & Foreground)
- Negative Control (Hard-Crop)

Generates exact numerators, denominators, calibration thresholds,
and latency measurements separated across Real Photographic Captures
and Synthetic Robustness Tests.
"""

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
from engine.soft_attention import compute_combined_weights, aggregate_patch_tokens


def load_benchmark(bench_path: str = "data/evaluation/benchmark_v3_balanced.json") -> Dict[str, Any]:
    with open(bench_path, "r") as fp:
        return json.load(fp)


def evaluate_model(
    model_name: str,
    benchmark: Dict[str, Any],
    mode: str = "global", # "global", "soft_attention", "hard_crop", "hybrid"
    alpha: float = 0.3
) -> Dict[str, Any]:
    """Evaluate a specific retrieval configuration against benchmark queries."""
    index_dir = Path("data/index")
    
    # Load catalog metadata and embeddings
    with open(index_dir / "dinov2_meta.json", "r") as fp:
        catalog_meta = json.load(fp)
    
    if mode == "global":
        encoder = get_encoder(model_name)
        cat_embeddings = np.load(index_dir / f"{model_name}_embeddings.npy")
    elif mode == "soft_attention":
        encoder = get_encoder(model_name)
        cat_embeddings = np.load(index_dir / f"{model_name}_soft_combined.npy")
        if alpha > 0:
            cat_global = np.load(index_dir / f"{model_name}_embeddings.npy")
            cat_embeddings = (1.0 - alpha) * cat_global + alpha * cat_embeddings
            norms = np.linalg.norm(cat_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            cat_embeddings = cat_embeddings / norms
    elif mode == "hard_crop":
        encoder = get_encoder(model_name)
        cat_embeddings = np.load(index_dir / f"{model_name}_region_embeddings.npy")
    elif mode == "hybrid":
        enc_v2 = get_encoder("dinov2")
        enc_v3 = get_encoder("dinov3")
        cat_v2 = np.load(index_dir / "dinov2_embeddings.npy")
        cat_v3 = np.load(index_dir / "dinov3_embeddings.npy")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    queries = benchmark["same_sku_queries"]
    
    results = {
        "all_60": {"hits_1": 0, "hits_5": 0, "hits_10": 0, "hits_20": 0, "mrr_sum": 0.0, "total": 0},
        "real_34": {"hits_1": 0, "hits_5": 0, "hits_10": 0, "hits_20": 0, "mrr_sum": 0.0, "total": 0},
        "real_neck_30": {"hits_1": 0, "hits_5": 0, "hits_10": 0, "hits_20": 0, "mrr_sum": 0.0, "total": 0},
        "real_ring_4": {"hits_1": 0, "hits_5": 0, "hits_10": 0, "hits_20": 0, "mrr_sum": 0.0, "total": 0},
        "synthetic_26": {"hits_1": 0, "hits_5": 0, "hits_10": 0, "hits_20": 0, "mrr_sum": 0.0, "total": 0},
    }
    
    latencies_ms = []

    for q in queries:
        qid = q["query_id"]
        target_id = q["target_catalog_id"]
        cat = q["category"]
        is_synthetic = "synthetic" in q.get("provenance", "") or "verified_photographic_variation" in q.get("provenance", "")
        is_ring = (cat == "ring")

        img_path = Path(q["query_path"])
        if not img_path.exists():
            continue
        
        img = Image.open(img_path).convert("RGB")
        
        t0 = time.perf_counter()
        if mode == "global":
            q_emb = encoder.encode_image(img)
            scores = np.dot(cat_embeddings, q_emb)
        elif mode == "soft_attention":
            patches = encoder.get_patch_tokens(img)
            from engine.localization import JewelleryLocalizer
            loc = JewelleryLocalizer()
            loc_res = loc.locate(img, category=cat)
            w = compute_combined_weights(img, loc_res.box, is_fallback=loc_res.is_fallback)
            q_soft = aggregate_patch_tokens(patches, w)
            if alpha > 0:
                q_glob = encoder.encode_image(img)
                q_emb = (1.0 - alpha) * q_glob + alpha * q_soft
                q_emb = q_emb / np.linalg.norm(q_emb)
            else:
                q_emb = q_soft
            scores = np.dot(cat_embeddings, q_emb)
        elif mode == "hard_crop":
            from engine.localization import JewelleryLocalizer
            loc = JewelleryLocalizer()
            loc_res = loc.locate(img, category=cat)
            q_emb = encoder.encode_image(loc_res.crop)
            scores = np.dot(cat_embeddings, q_emb)
        elif mode == "hybrid":
            q_v2 = enc_v2.encode_image(img)
            q_v3 = enc_v3.encode_image(img)
            s_v2 = np.dot(cat_v2, q_v2)
            s_v3 = np.dot(cat_v3, q_v3)
            # Min-Max calibrated score normalization
            s_v2_norm = (s_v2 - 0.20) / (0.95 - 0.20 + 1e-6)
            s_v3_norm = (s_v3 - 0.70) / (0.99 - 0.70 + 1e-6)
            scores = 0.5 * s_v2_norm + 0.5 * s_v3_norm

        t_elapsed = (time.perf_counter() - t0) * 1000
        latencies_ms.append(t_elapsed)

        top_indices = np.argsort(scores)[::-1][:20]
        ranked_ids = [catalog_meta[idx]["id"] for idx in top_indices]

        # Determine rank of true target
        rank = None
        if target_id in ranked_ids:
            rank = ranked_ids.index(target_id) + 1

        # Helper to accumulate
        def accumulate(sub_dict):
            sub_dict["total"] += 1
            if rank is not None:
                if rank == 1: sub_dict["hits_1"] += 1
                if rank <= 5: sub_dict["hits_5"] += 1
                if rank <= 10: sub_dict["hits_10"] += 1
                if rank <= 20: sub_dict["hits_20"] += 1
                sub_dict["mrr_sum"] += 1.0 / rank

        accumulate(results["all_60"])
        if is_synthetic:
            accumulate(results["synthetic_26"])
        else:
            accumulate(results["real_34"])
            if is_ring:
                accumulate(results["real_ring_4"])
            else:
                accumulate(results["real_neck_30"])

    # Format output metrics
    formatted = {}
    for k, v in results.items():
        tot = v["total"]
        if tot == 0: continue
        formatted[k] = {
            "count": tot,
            "r1": round(v["hits_1"] / tot, 4),
            "r5": round(v["hits_5"] / tot, 4),
            "r10": round(v["hits_10"] / tot, 4),
            "r20": round(v["hits_20"] / tot, 4),
            "mrr": round(v["mrr_sum"] / tot, 4),
            "r1_num": f"{v['hits_1']}/{tot}",
            "r5_num": f"{v['hits_5']}/{tot}",
            "r10_num": f"{v['hits_10']}/{tot}",
            "r20_num": f"{v['hits_20']}/{tot}",
        }

    formatted["latency_p50_ms"] = round(float(np.median(latencies_ms)), 2)
    formatted["latency_p95_ms"] = round(float(np.percentile(latencies_ms, 95)), 2)
    return formatted


def main():
    bench = load_benchmark()
    print("=== Running EXP-008 Baseline Reproduction & Verification ===")

    configs = [
        ("DINOv2 Global (CLS)", "dinov2", "global", 0.0),
        ("DINOv3 Global (CLS)", "dinov3", "global", 0.0),
        ("Calibrated Hybrid Global", "hybrid", "hybrid", 0.0),
        ("DINOv3 Fused Soft Attention (alpha=0.3)", "dinov3", "soft_attention", 0.3),
        ("Negative Control (DINOv3 Hard Crop)", "dinov3", "hard_crop", 0.0)
    ]

    summary = {}
    for label, model, mode, alpha in configs:
        print(f"\nEvaluating: {label}...")
        res = evaluate_model(model, bench, mode=mode, alpha=alpha)
        summary[label] = res
        print(f"  Overall R@1: {res['all_60']['r1_num']} ({res['all_60']['r1']*100:.1f}%) | R@5: {res['all_60']['r5_num']} | R@20: {res['all_60']['r20_num']} | MRR: {res['all_60']['mrr']}")
        print(f"  Real R@1 (N=34): {res['real_34']['r1_num']} ({res['real_34']['r1']*100:.1f}%) [Neck: {res['real_neck_30']['r1_num']}, Ring: {res['real_ring_4']['r1_num']}]")
        print(f"  Synthetic R@1 (N=26): {res['synthetic_26']['r1_num']} ({res['synthetic_26']['r1']*100:.1f}%)")
        print(f"  Sequential Latency: p50={res['latency_p50_ms']}ms, p95={res['latency_p95_ms']}ms")

    out_file = Path("output/exp008_baseline_verification.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nVerification summary saved to: {out_file}")


if __name__ == "__main__":
    main()
