#!/usr/bin/env python3
"""Confidence threshold calibration for all scoring configurations on held-out catalog pairs."""

import io
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.embeddings import get_encoder
from engine.reranker import compute_chamfer_similarity, get_foreground_patch_indices

def generate_augmented_positive(img: Image.Image, seed: int) -> Image.Image:
    """Generate realistic photographic variation (compression, mild brightness, crop) without altering SKU identity."""
    rng = random.Random(seed)
    w, h = img.size
    
    # Mild random crop (94% to 98% area)
    scale = rng.uniform(0.94, 0.98)
    cw, ch = int(w * scale), int(h * scale)
    x = rng.randint(0, w - cw)
    y = rng.randint(0, h - ch)
    aug = img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.Resampling.LANCZOS)
    
    # Mild brightness / contrast shift
    enh = ImageEnhance.Brightness(aug)
    aug = enh.enhance(rng.uniform(0.92, 1.08))
    
    # JPEG recompression artifact
    buf = io.BytesIO()
    aug.save(buf, format="JPEG", quality=rng.randint(65, 85))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def evaluate_sweep(pos_sims: np.ndarray, neg_sims: np.ndarray, thresholds: np.ndarray) -> Dict:
    sweep_results = []
    calibrated_tau_95 = None
    calibrated_tau_99 = None

    for tau in thresholds:
        tp = np.sum(pos_sims >= tau)
        fp = np.sum(neg_sims >= tau)
        fn = np.sum(pos_sims < tau)
        tn = np.sum(neg_sims < tau)
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        abstention = float((fn + tn) / (len(pos_sims) + len(neg_sims)))
        
        entry = {
            "threshold": round(float(tau), 4),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "false_acceptance_rate": round(far * 100, 3),
            "abstention_rate": round(abstention * 100, 2)
        }
        sweep_results.append(entry)
        
        if precision >= 0.95 and calibrated_tau_95 is None and (tp + fp) > 0:
            calibrated_tau_95 = round(float(tau), 4)
        if precision >= 0.99 and calibrated_tau_99 is None and (tp + fp) > 0:
            calibrated_tau_99 = round(float(tau), 4)

    return {
        "score_distribution": {
            "pos_mean": round(float(np.mean(pos_sims)), 4),
            "pos_std": round(float(np.std(pos_sims)), 4),
            "neg_mean": round(float(np.mean(neg_sims)), 4),
            "neg_std": round(float(np.std(neg_sims)), 4),
        },
        "calibrated_threshold_95_precision": calibrated_tau_95,
        "calibrated_threshold_99_precision": calibrated_tau_99,
        "sweep": sweep_results
    }

def run_calibration(num_items: int = 100, num_negatives: int = 300, seed: int = 42):
    with open("data/manifest.json") as f:
        manifest = json.load(f)

    # Use only catalog items for calibration (strictly disjoint from test set!)
    catalog_items = [img for img in manifest["images"] if img.get("split") == "catalog"]
    rng = random.Random(seed)
    sample_items = rng.sample(catalog_items, min(num_items, len(catalog_items)))

    print(f"Loading encoders on CPU...")
    v2_encoder = get_encoder("dinov2")
    v3_encoder = get_encoder("dinov3")

    print(f"Extracting features for {len(sample_items)} original and positive augmented catalog items...")
    orig_imgs = []
    aug_imgs = []
    for idx, item in enumerate(sample_items):
        img_path = Path(item["rel_path"])
        if not img_path.exists():
            img_path = Path("Jewellery_Data") / item["category"] / item["filename"]
        raw = Image.open(img_path).convert("RGB")
        aug = generate_augmented_positive(raw, seed=idx + 1000)
        orig_imgs.append(raw)
        aug_imgs.append(aug)

    # Global vectors
    print("Computing DINOv2 vectors...")
    v2_orig = v2_encoder.encode_batch([im.resize((224, 224)) for im in orig_imgs])
    v2_aug = v2_encoder.encode_batch([im.resize((224, 224)) for im in aug_imgs])

    print("Computing DINOv3 vectors...")
    v3_orig = v3_encoder.encode_batch([im.resize((256, 256)) for im in orig_imgs])
    v3_aug = v3_encoder.encode_batch([im.resize((256, 256)) for im in aug_imgs])

    print("Computing DINOv3 patch tokens and foreground masks for subset...")
    v3_patches_orig = v3_encoder.get_patch_tokens_batch([im.resize((256, 256)) for im in orig_imgs])
    v3_patches_aug = v3_encoder.get_patch_tokens_batch([im.resize((256, 256)) for im in aug_imgs])
    fg_orig = [get_foreground_patch_indices(im.resize((256, 256))) for im in orig_imgs]
    fg_aug = [get_foreground_patch_indices(im.resize((256, 256))) for im in aug_imgs]

    # Sample negative pairs from different clusters
    neg_pairs = []
    while len(neg_pairs) < num_negatives:
        i, j = rng.sample(range(len(sample_items)), 2)
        if sample_items[i]["cluster_id"] != sample_items[j]["cluster_id"]:
            neg_pairs.append((i, j))

    print(f"\nEvaluating configurations on {len(sample_items)} positive pairs and {len(neg_pairs)} negative pairs...")
    configs = ["dinov2", "dinov3", "hybrid_global", "hybrid_rerank_pw02", "hybrid_rerank_pw03"]
    thresholds = np.linspace(0.70, 0.995, 60)

    calibration_report = {}

    for cfg in configs:
        pos_sims = []
        neg_sims = []

        # 1. Positives
        for idx in range(len(sample_items)):
            sim_v2 = float(np.dot(v2_orig[idx], v2_aug[idx]))
            sim_v3 = float(np.dot(v3_orig[idx], v3_aug[idx]))
            sim_g = 0.5 * (sim_v2 + sim_v3)

            if cfg == "dinov2":
                score = sim_v2
            elif cfg == "dinov3":
                score = sim_v3
            elif cfg == "hybrid_global":
                score = sim_g
            elif cfg == "hybrid_rerank_pw02":
                ch = compute_chamfer_similarity(v3_patches_orig[idx], v3_patches_aug[idx], fg_orig[idx], fg_aug[idx])
                score = (1.0 - 0.2) * sim_g + 0.2 * ch
            elif cfg == "hybrid_rerank_pw03":
                ch = compute_chamfer_similarity(v3_patches_orig[idx], v3_patches_aug[idx], fg_orig[idx], fg_aug[idx])
                score = (1.0 - 0.3) * sim_g + 0.3 * ch

            pos_sims.append(score)

        # 2. Negatives
        for i, j in neg_pairs:
            sim_v2 = float(np.dot(v2_orig[i], v2_orig[j]))
            sim_v3 = float(np.dot(v3_orig[i], v3_orig[j]))
            sim_g = 0.5 * (sim_v2 + sim_v3)

            if cfg == "dinov2":
                score = sim_v2
            elif cfg == "dinov3":
                score = sim_v3
            elif cfg == "hybrid_global":
                score = sim_g
            elif cfg == "hybrid_rerank_pw02":
                ch = compute_chamfer_similarity(v3_patches_orig[i], v3_patches_orig[j], fg_orig[i], fg_orig[j])
                score = (1.0 - 0.2) * sim_g + 0.2 * ch
            elif cfg == "hybrid_rerank_pw03":
                ch = compute_chamfer_similarity(v3_patches_orig[i], v3_patches_orig[j], fg_orig[i], fg_orig[j])
                score = (1.0 - 0.3) * sim_g + 0.3 * ch

            neg_sims.append(score)

        pos_sims = np.array(pos_sims)
        neg_sims = np.array(neg_sims)

        res = evaluate_sweep(pos_sims, neg_sims, thresholds)
        res["calibration_set_size"] = {"positive_pairs": len(pos_sims), "negative_pairs": len(neg_sims)}
        calibration_report[cfg] = res

        print(f"\nConfiguration: '{cfg}'")
        print(f"  Pos: {res['score_distribution']['pos_mean']} +/- {res['score_distribution']['pos_std']} | Neg: {res['score_distribution']['neg_mean']} +/- {res['score_distribution']['neg_std']}")
        print(f"  Calibrated Tau (95% Prec): {res['calibrated_threshold_95_precision']}")
        print(f"  Calibrated Tau (99% Prec): {res['calibrated_threshold_99_precision']}")

    Path("output").mkdir(exist_ok=True)
    out_file = Path("output/calibration_report.json")
    with open(out_file, "w") as fp:
        json.dump(calibration_report, fp, indent=2)
    print(f"\nSaved calibrated thresholds to {out_file}")

if __name__ == "__main__":
    run_calibration()
