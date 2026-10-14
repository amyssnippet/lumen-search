#!/usr/bin/env python3
"""Workstream 1: Diagnostic Failure Reproduction Script.

Evaluates the production MobileNetV4 baseline under:
1. Original catalogue photograph.
2. Slight rotation and angle changes (±5°, ±10°, ±15°).
3. Crop and zoom variations (80% crop, 1.2x zoom).
4. Mild noise (Gaussian sigma=15) and JPEG compression (Q=40).
5. Brightness, exposure, and white-balance changes.
6. Real query photographs with worn-on-hand presentation (ring-downloaded.png -> ring_063.jpg).

Outputs detailed diagnostic breakdown to output/ph_robustness_diagnosis.json.
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.embeddings import MobileNetV4Encoder
from engine.preprocessing import PreprocessConfig, preprocess_query_image


def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image around center with white background fill."""
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255))


def apply_zoom_crop(img: Image.Image, scale: float = 0.8) -> Image.Image:
    """Center crop to fraction of size and resize back."""
    w, h = img.size
    cw, ch = int(w * scale), int(h * scale)
    x1, y1 = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((x1, y1, x1 + cw, y1 + ch))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)


def apply_gaussian_noise(img: Image.Image, sigma: float = 15.0) -> Image.Image:
    """Add Gaussian sensor noise."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_jpeg_compression(img: Image.Image, quality: int = 40) -> Image.Image:
    """Simulate severe JPEG compression artifacts."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_exposure_change(img: Image.Image, factor: float) -> Image.Image:
    """Alter brightness/exposure."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_white_balance_shift(img: Image.Image, temp: str = "warm") -> Image.Image:
    """Shift white balance towards warm (amber/yellow) or cool (blue)."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    if temp == "warm":
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.15, 0, 255)  # Red boost
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.85, 0, 255)  # Blue cut
    else:
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.85, 0, 255)  # Red cut
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.15, 0, 255)  # Blue boost
    return Image.fromarray(arr.astype(np.uint8))


def run_diagnostics():
    np.random.seed(42)
    encoder = MobileNetV4Encoder()
    with open("data/index/mobilenetv4_meta.json", "r") as f:
        meta = json.load(f)
    embs = np.load("data/index/mobilenetv4_embeddings.npy")

    # Map catalog ID to index
    id_to_idx = {item["id"]: i for i, item in enumerate(meta)}

    # Test cases: (base_image_path, ground_truth_id, name)
    cases = [
        ("Jewellery_Data/ring/ring_063.jpg", "ring_063", "Floral Diamond Halo Ring"),
        ("Jewellery_Data/ring/ring_032.jpg", "ring_032", "Solitaire Diamond Ring"),
        ("Jewellery_Data/ring/ring_081.jpg", "ring_081", "Yellow Gold Solitaire Ring"),
        ("Jewellery_Data/ring/ring_155.jpg", "ring_155", "Emerald Cut Halo Ring"),
    ]

    # Include the worn-on-hand case: ring-downloaded is a real photo of ring_063 on finger
    worn_cases = [
        ("Jewellery_Data/ring/ring-downloaded.png", "ring_063", "Worn on Hand — Floral Cluster Ring")
    ]

    results = []

    def evaluate_query(query_img: Image.Image, gt_id: str, condition_name: str, src_path: str, exclude_self: bool = False):
        t0_prep = time.time()
        config = PreprocessConfig()
        prep, qual, pmeta = preprocess_query_image(query_img, config)
        t_prep_ms = (time.time() - t0_prep) * 1000

        t0_enc = time.time()
        q_emb = encoder.encode_image(prep)
        t_enc_ms = (time.time() - t0_enc) * 1000

        t0_search = time.time()
        sims = np.dot(embs, q_emb)

        # Build ranking list
        scored = []
        for idx, sim_val in enumerate(sims):
            cid = meta[idx]["id"]
            if exclude_self and cid == Path(src_path).stem:
                continue
            scored.append((cid, float(sim_val), meta[idx]))

        scored.sort(key=lambda x: x[1], reverse=True)
        t_search_ms = (time.time() - t0_search) * 1000

        # Find GT rank and score
        gt_rank = None
        gt_sim = None
        for rank, (cid, s, _) in enumerate(scored, 1):
            if cid == gt_id:
                gt_rank = rank
                gt_sim = s
                break

        top_hit = scored[0]
        top_cid, top_sim, _ = top_hit
        margin = (gt_sim - scored[1][1]) if (gt_rank == 1 and len(scored) > 1) else (gt_sim - top_sim)

        top_3 = [{"id": cid, "similarity": round(s, 4)} for cid, s, _ in scored[:3]]

        res = {
            "condition": condition_name,
            "source_image": src_path,
            "ground_truth_id": gt_id,
            "gt_rank": gt_rank,
            "gt_similarity": round(gt_sim, 4) if gt_sim is not None else None,
            "top_match_id": top_cid,
            "top_match_similarity": round(top_sim, 4),
            "score_margin": round(margin, 4) if margin is not None else None,
            "top_3": top_3,
            "is_in_top_1": gt_rank == 1,
            "is_in_top_5": gt_rank is not None and gt_rank <= 5,
            "is_in_top_20": gt_rank is not None and gt_rank <= 20,
            "timings_ms": {
                "preprocessing": round(t_prep_ms, 2),
                "encoding": round(t_enc_ms, 2),
                "search": round(t_search_ms, 2),
                "total": round(t_prep_ms + t_enc_ms + t_search_ms, 2)
            }
        }
        return res

    for img_path, gt_id, label in cases:
        base_img = Image.open(img_path).convert("RGB")
        # 1. Original (baseline self-check, note self match)
        results.append(evaluate_query(base_img, gt_id, f"1_original_catalogue ({label})", img_path))

        # 2. Rotations
        for ang in [5, 10, 15, -10]:
            rot_img = apply_rotation(base_img, ang)
            results.append(evaluate_query(rot_img, gt_id, f"2_rotation_{ang}deg ({label})", img_path))

        # 3. Zoom / Crop
        for scale in [0.85, 0.70]:
            crop_img = apply_zoom_crop(base_img, scale=scale)
            results.append(evaluate_query(crop_img, gt_id, f"3_zoom_crop_{int(scale*100)}pct ({label})", img_path))

        # 4. Noise & Compression
        noisy_img = apply_gaussian_noise(base_img, sigma=15.0)
        results.append(evaluate_query(noisy_img, gt_id, f"4_gaussian_noise_s15 ({label})", img_path))

        jpeg_img = apply_jpeg_compression(base_img, quality=40)
        results.append(evaluate_query(jpeg_img, gt_id, f"4_jpeg_compression_q40 ({label})", img_path))

        # 5. Brightness & White Balance
        dim_img = apply_exposure_change(base_img, 0.70)
        results.append(evaluate_query(dim_img, gt_id, f"5_underexposed_30pct ({label})", img_path))

        bright_img = apply_exposure_change(base_img, 1.30)
        results.append(evaluate_query(bright_img, gt_id, f"5_overexposed_30pct ({label})", img_path))

        warm_img = apply_white_balance_shift(base_img, "warm")
        results.append(evaluate_query(warm_img, gt_id, f"5_white_balance_warm ({label})", img_path))

        cool_img = apply_white_balance_shift(base_img, "cool")
        results.append(evaluate_query(cool_img, gt_id, f"5_white_balance_cool ({label})", img_path))

    # 6. Real Worn-on-Hand condition
    for img_path, gt_id, label in worn_cases:
        worn_img = Image.open(img_path).convert("RGB")
        # Exclude self ("ring-downloaded") from candidates so we evaluate retrieval of the actual catalogue SKU "ring_063"
        results.append(evaluate_query(worn_img, gt_id, f"6_worn_on_hand ({label})", img_path, exclude_self=True))

    out_file = Path("output/ph_robustness_diagnosis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump({"total_evaluations": len(results), "evaluations": results}, fp, indent=2)

    print(f"Diagnostics complete. Saved {len(results)} condition evaluations to {out_file}")

    # Summary
    print("\n" + "=" * 80)
    print("WORKSTREAM 1 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    for r in results:
        status = "PASS (Rank 1)" if r["is_in_top_1"] else f"DEGRADED (Rank {r['gt_rank']}, Sim {r['gt_similarity']}, Top: {r['top_match_id']}@{r['top_match_similarity']})"
        print(f"[{r['condition'][:40]:40s}] GT: {r['ground_truth_id']:10s} -> {status}")


if __name__ == "__main__":
    run_diagnostics()
