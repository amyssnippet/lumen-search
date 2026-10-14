#!/usr/bin/env python3
"""Build a reproducible dataset manifest with verified identity clusters and evaluation splits."""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from PIL import Image

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            p_left = pixels[row * (hash_size + 1) + col]
            p_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(p_left > p_right)
    decimal_val = 0
    hex_string = []
    for index, val in enumerate(difference):
        if val:
            decimal_val += 2 ** (index % 4)
        if (index % 4) == 3:
            hex_string.append(hex(decimal_val)[2:])
            decimal_val = 0
    return ''.join(hex_string)

def hamming_dist(s1: str, s2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def build_manifest(data_dir: str = "Jewellery_Data", output_path: str = "data/manifest.json"):
    data_path = Path(data_dir)
    images = []
    sha_map = defaultdict(list)
    dhash_map = {}

    all_files = sorted([f for f in data_path.rglob("*") if f.is_file() and not f.name.startswith(".")])

    for f in all_files:
        rel_path = str(f.relative_to(data_path.parent))
        cat = f.parent.name
        with open(f, "rb") as fp:
            sha256 = hashlib.sha256(fp.read()).hexdigest()

        with Image.open(f) as img:
            w, h = img.size
            fmt = img.format
            dh = compute_dhash(img)

        img_record = {
            "id": f.stem,
            "filename": f.name,
            "rel_path": rel_path,
            "category": cat,
            "sha256": sha256,
            "dhash": dh,
            "width": w,
            "height": h,
            "format": fmt
        }
        images.append(img_record)
        sha_map[sha256].append(img_record["id"])
        dhash_map[img_record["id"]] = dh

    # Connected components for identity clusters
    parent = {img["id"]: img["id"] for img in images}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Union exact duplicates
    for sha, ids in sha_map.items():
        if len(ids) > 1:
            for i in range(len(ids) - 1):
                union(ids[i], ids[i+1])

    # Union near duplicates (dHash distance <= 2) within same category
    by_category = defaultdict(list)
    for img in images:
        by_category[img["category"]].append(img)

    for cat, cat_imgs in by_category.items():
        for i in range(len(cat_imgs)):
            for j in range(i + 1, len(cat_imgs)):
                id1, id2 = cat_imgs[i]["id"], cat_imgs[j]["id"]
                if hamming_dist(dhash_map[id1], dhash_map[id2]) <= 2:
                    union(id1, id2)

    clusters = defaultdict(list)
    for img in images:
        root = find(img["id"])
        clusters[root].append(img["id"])

    # Assign cluster labels
    cluster_labels = {}
    cluster_sizes = {}
    for idx, (root, members) in enumerate(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)):
        cid = f"cluster_{idx:03d}"
        cluster_sizes[cid] = len(members)
        for m in members:
            cluster_labels[m] = cid

    for img in images:
        img["cluster_id"] = cluster_labels[img["id"]]

    # Split into Catalog and Test
    # For every cluster with >= 2 members, pick one for test query and keep remainder in catalog
    test_ids = set()
    catalog_ids = set()

    multi_item_clusters = [cid for cid, size in cluster_sizes.items() if size >= 2]
    # Select up to 30 twin queries
    twin_queries = 0
    for cid in multi_item_clusters:
        members = [img["id"] for img in images if img["cluster_id"] == cid]
        if twin_queries < 30 and len(members) >= 2:
            test_ids.add(members[0])
            for m in members[1:]:
                catalog_ids.add(m)
            twin_queries += 1
        else:
            for m in members:
                catalog_ids.add(m)

    # Pick 30 singletons for non-twin test queries
    singletons = [img["id"] for img in images if cluster_sizes[img["cluster_id"]] == 1]
    import random
    rng = random.Random(42)
    sample_singletons = set(rng.sample(singletons, min(30, len(singletons))))
    for sid in sample_singletons:
        test_ids.add(sid)

    for img in images:
        if img["id"] not in test_ids:
            catalog_ids.add(img["id"])
        img["split"] = "test" if img["id"] in test_ids else "catalog"

    manifest_data = {
        "metadata": {
            "total_images": len(images),
            "catalog_count": len(catalog_ids),
            "test_count": len(test_ids),
            "total_clusters": len(clusters),
            "multi_item_clusters": len(multi_item_clusters),
            "twin_test_queries": twin_queries,
            "singleton_test_queries": len(sample_singletons)
        },
        "images": images
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(manifest_data, fp, indent=2)

    print(f"Manifest written to {output_path}")
    print(f"Catalog images: {len(catalog_ids)}, Test queries: {len(test_ids)} (Twins: {twin_queries}, Singletons: {len(sample_singletons)})")
    print(f"Total clusters: {len(clusters)} (Multi-item: {len(multi_item_clusters)})")
    return manifest_data

if __name__ == "__main__":
    build_manifest()
