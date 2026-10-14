#!/usr/bin/env python3
"""Build Frozen PH-014 Real-Image Evaluation Manifest across 6 distinct slices.

Slices:
1. slice_1_same_image_near_duplicate: Same image / near-duplicate retrieval (10 rings, 10 necklaces).
2. slice_2_image_corruptions: Sensor noise, blur, JPEG compression, exposure shifts (20 queries).
3. slice_3_viewpoint_crops: Controlled angle rotation (±5°, ±10°, ±15°), tilt, zoom (32 queries).
4. slice_4_genuine_physical_multiview: Real studio multi-view photographs of the same physical SKU (15 queries).
5. slice_5_worn_on_hand: Real worn and realistic skin-background ring queries (5 queries).
6. slice_6_hard_negatives: Visually similar but distinct SKUs to measure False Accept Rate (30 pairs).

Output:
    data/evaluation/ph014_evaluation_manifest.json
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def build_manifest():
    out_manifest_path = REPO_ROOT / "data" / "evaluation" / "ph014_evaluation_manifest.json"
    existing_robustness_path = REPO_ROOT / "data" / "evaluation" / "robustness_benchmark_manifest.json"

    with open(existing_robustness_path, "r") as f:
        robustness_data = json.load(f)

    existing_slices = robustness_data.get("slices", {})

    # Slice 1: Same-image / near-duplicate (10 rings + 10 necklaces)
    slice_1 = []
    ring_ids = [f"ring_00{i}" for i in range(1, 10)] + ["ring_013"]
    neck_ids = ["necklace_1", "necklace_9", "necklace_10", "necklace_20", "necklace_30", "necklace_40", "necklace_50", "necklace_60", "necklace_70", "necklace_80"]

    for cid in ring_ids:
        slice_1.append({
            "query_id": f"same_img_{cid}",
            "query_path": f"Jewellery_Data/ring/{cid}.jpg",
            "target_sku": f"SKU-RING-{cid}",
            "target_catalog_id": cid,
            "category": "ring",
            "slice": "slice_1_same_image_near_duplicate",
            "description": f"Exact same-image identity retrieval for {cid}"
        })

    for cid in neck_ids:
        slice_1.append({
            "query_id": f"same_img_{cid}",
            "query_path": f"Jewellery_Data/necklace/{cid}.jpg",
            "target_sku": f"SKU-NECK-{cid}",
            "target_catalog_id": cid,
            "category": "necklace",
            "slice": "slice_1_same_image_near_duplicate",
            "description": f"Exact same-image identity retrieval for {cid}"
        })

    manifest = {
        "version": "ph014_frozen_v1.0",
        "description": "PH-014 Definitive Jewellery Visual Retrieval Quality & Robustness Benchmark",
        "slices": {
            "slice_1_same_image_near_duplicate": slice_1,
            "slice_2_image_corruptions": existing_slices.get("slice_d_image_corruptions", []),
            "slice_3_viewpoint_crops": existing_slices.get("slice_c_viewpoint_shifts", []),
            "slice_4_genuine_physical_multiview": existing_slices.get("slice_a_real_multiview", []),
            "slice_5_worn_on_hand": existing_slices.get("slice_b_worn_on_hand", []),
            "slice_6_hard_negatives": existing_slices.get("slice_e_hard_negatives", [])
        },
        "slice_counts": {
            "slice_1_same_image_near_duplicate": len(slice_1),
            "slice_2_image_corruptions": len(existing_slices.get("slice_d_image_corruptions", [])),
            "slice_3_viewpoint_crops": len(existing_slices.get("slice_c_viewpoint_shifts", [])),
            "slice_4_genuine_physical_multiview": len(existing_slices.get("slice_a_real_multiview", [])),
            "slice_5_worn_on_hand": len(existing_slices.get("slice_b_worn_on_hand", [])),
            "slice_6_hard_negatives": len(existing_slices.get("slice_e_hard_negatives", []))
        },
        "total_queries": len(slice_1) + sum(len(existing_slices.get(k, [])) for k in existing_slices)
    }

    with open(out_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created {out_manifest_path} with {manifest['total_queries']} total queries across 6 slices:")
    for s_name, cnt in manifest["slice_counts"].items():
        print(f"  - {s_name}: {cnt} queries")

if __name__ == "__main__":
    build_manifest()
