#!/usr/bin/env python3
"""Dataset inspection and integrity analysis for Jewellery Visual Retrieval Engine."""

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute difference hash for near-duplicate detection."""
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)
    decimal_val = 0
    hex_string = []
    for index, val in enumerate(difference):
        if val:
            decimal_val += 2 ** (index % 4)
        if (index % 4) == 3:
            hex_string.append(hex(decimal_val)[2:])
            decimal_val = 0
    return ''.join(hex_string)

def hamming_distance(s1: str, s2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def inspect_dataset(data_dir: str = "Jewellery_Data") -> dict:
    data_path = Path(data_dir)
    images_info = []
    hash_to_files = defaultdict(list)
    dhash_map = {}
    corrupted_files = []
    categories = defaultdict(int)
    formats = defaultdict(int)
    modes = defaultdict(int)
    sizes = []

    all_files = sorted(list(data_path.rglob("*")))
    image_files = [f for f in all_files if f.is_file() and not f.name.startswith(".")]

    for p in image_files:
        rel_path = str(p.relative_to(data_path.parent))
        cat = p.parent.name
        categories[cat] += 1
        file_size = p.stat().st_size

        with open(p, "rb") as f:
            content = f.read()
            sha256 = hashlib.sha256(content).hexdigest()
            hash_to_files[sha256].append(rel_path)

        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                fmt = img.format
                mode = img.mode
                w, h = img.size
                formats[fmt] += 1
                modes[mode] += 1
                sizes.append((w, h))
                exif = img.getexif()
                exif_tags = {k: str(v)[:50] for k, v in exif.items()} if exif else {}
                dh = compute_dhash(img)
                dhash_map[rel_path] = dh

                images_info.append({
                    "id": p.stem,
                    "filename": p.name,
                    "rel_path": rel_path,
                    "category": cat,
                    "format": fmt,
                    "mode": mode,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 3) if h else 0,
                    "file_size_bytes": file_size,
                    "sha256": sha256,
                    "dhash": dh,
                    "exif": exif_tags
                })
        except Exception as e:
            corrupted_files.append({"path": rel_path, "error": str(e)})

    # Detect exact duplicates
    exact_duplicates = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}

    # Detect near duplicates (hamming distance <= 2 on 64-bit dHash)
    near_duplicates = []
    paths_list = list(dhash_map.keys())
    for i in range(len(paths_list)):
        for j in range(i + 1, len(paths_list)):
            p1, p2 = paths_list[i], paths_list[j]
            dist = hamming_distance(dhash_map[p1], dhash_map[p2])
            if dist <= 2:
                # Exclude exact duplicates
                if dhash_map[p1] == dhash_map[p2] and p2 in exact_duplicates.get(images_info[i]["sha256"], []):
                    continue
                near_duplicates.append({"p1": p1, "p2": p2, "distance": dist})

    summary = {
        "total_files": len(image_files),
        "valid_images": len(images_info),
        "corrupted_images": len(corrupted_files),
        "categories": dict(categories),
        "formats": dict(formats),
        "modes": dict(modes),
        "exact_duplicate_sets": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": len(near_duplicates),
        "near_duplicates": near_duplicates[:20],
        "width_range": [min(s[0] for s in sizes), max(s[0] for s in sizes)] if sizes else [],
        "height_range": [min(s[1] for s in sizes), max(s[1] for s in sizes)] if sizes else [],
    }

    return {"summary": summary, "images": images_info, "corrupted": corrupted_files}

if __name__ == "__main__":
    report = inspect_dataset()
    os.makedirs("output", exist_ok=True)
    with open("output/dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== DATASET INSPECTION SUMMARY ===")
    print(f"Total files: {report['summary']['total_files']}")
    print(f"Valid images: {report['summary']['valid_images']}")
    print(f"Corrupted images: {report['summary']['corrupted_images']}")
    print(f"Categories: {report['summary']['categories']}")
    print(f"Formats: {report['summary']['formats']}")
    print(f"Modes: {report['summary']['modes']}")
    print(f"Exact duplicate sets: {report['summary']['exact_duplicate_sets']}")
    print(f"Near duplicate pairs: {report['summary']['near_duplicate_pairs']}")
    print(f"Width range: {report['summary']['width_range']}")
    print(f"Height range: {report['summary']['height_range']}")
    if report['summary']['exact_duplicates']:
        print("Exact duplicates found:")
        for h, paths in report['summary']['exact_duplicates'].items():
            print(f"  Hash {h[:10]}...: {paths}")
