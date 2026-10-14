#!/usr/bin/env python3
"""EXP-005 Phase 2: Generate candidate pairs queue for ground-truth annotation."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def hamming_dist(s1: str, s2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def build_queue():
    manifest = json.load(open("data/manifest.json"))
    images = manifest["images"]
    id_to_img = {img["id"]: img for img in images}
    
    # Load embeddings to compute pairwise similarities
    # We can load catalog embeddings and compute query embeddings, or load precomputed matrices
    # Let us check if all embeddings are available
    cat_meta = json.load(open("data/index/dinov2_meta.json"))
    v2_index = np.load("data/index/dinov2_embeddings.npy")
    v3_index = np.load("data/index/dinov3_embeddings.npy")
    cat_ids = [m["id"] for m in cat_meta]
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}
    
    pairs = []
    seen = set()
    
    # 1. Add all existing multi-item cluster pairs from manifest
    clusters = defaultdict(list)
    for img in images:
        clusters[img["cluster_id"]].append(img)
        
    for cid, members in clusters.items():
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    id1, id2 = members[i]["id"], members[j]["id"]
                    cat = members[i]["category"]
                    pair_key = tuple(sorted([id1, id2]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        dh = hamming_dist(members[i]["dhash"], members[j]["dhash"])
                        # similarity if in catalog
                        s2, s3 = None, None
                        if id1 in cat_id_to_idx and id2 in cat_id_to_idx:
                            s2 = float(np.dot(v2_index[cat_id_to_idx[id1]], v2_index[cat_id_to_idx[id2]]))
                            s3 = float(np.dot(v3_index[cat_id_to_idx[id1]], v3_index[cat_id_to_idx[id2]]))
                        pairs.append({
                            "pair_id": f"{pair_key[0]}__{pair_key[1]}",
                            "query_id": id1,
                            "candidate_id": id2,
                            "category": cat,
                            "query_path": f"/media/{cat}/{members[i]['filename']}",
                            "candidate_path": f"/media/{cat}/{members[j]['filename']}",
                            "dhash_distance": dh,
                            "dinov2_sim": round(s2, 4) if s2 is not None else None,
                            "dinov3_sim": round(s3, 4) if s3 is not None else None,
                            "priority": 1,
                            "source_note": f"Cluster {cid} (size {len(members)}), dHash={dh}"
                        })

    # 2. Add high-similarity ring pairs (dHash <= 6 and high visual similarity)
    rings = [img for img in images if img["category"] == "ring"]
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            r1, r2 = rings[i], rings[j]
            pair_key = tuple(sorted([r1["id"], r2["id"]]))
            if pair_key in seen:
                continue
            dh = hamming_dist(r1["dhash"], r2["dhash"])
            s2, s3 = None, None
            if r1["id"] in cat_id_to_idx and r2["id"] in cat_id_to_idx:
                s2 = float(np.dot(v2_index[cat_id_to_idx[r1["id"]]], v2_index[cat_id_to_idx[r2["id"]]]))
                s3 = float(np.dot(v3_index[cat_id_to_idx[r1["id"]]], v3_index[cat_id_to_idx[r2["id"]]]))
            
            # Prioritize close dHash or high similarity
            if dh <= 4 or (s2 is not None and s2 >= 0.85) or (s3 is not None and s3 >= 0.95):
                seen.add(pair_key)
                pairs.append({
                    "pair_id": f"{pair_key[0]}__{pair_key[1]}",
                    "query_id": r1["id"],
                    "candidate_id": r2["id"],
                    "category": "ring",
                    "query_path": f"/media/ring/{r1['filename']}",
                    "candidate_path": f"/media/ring/{r2['filename']}",
                    "dhash_distance": dh,
                    "dinov2_sim": round(s2, 4) if s2 is not None else None,
                    "dinov3_sim": round(s3, 4) if s3 is not None else None,
                    "priority": 2 if dh <= 3 else 3,
                    "source_note": f"Ring candidate pair: dHash={dh}, DINOv2={s2}, DINOv3={s3}"
                })

    # 3. Add high-similarity necklace pairs (dHash <= 4 not yet in clusters)
    necklaces = [img for img in images if img["category"] == "necklace"]
    for i in range(len(necklaces)):
        for j in range(i + 1, len(necklaces)):
            n1, n2 = necklaces[i], necklaces[j]
            pair_key = tuple(sorted([n1["id"], n2["id"]]))
            if pair_key in seen:
                continue
            dh = hamming_dist(n1["dhash"], n2["dhash"])
            if dh <= 4:
                s2, s3 = None, None
                if n1["id"] in cat_id_to_idx and n2["id"] in cat_id_to_idx:
                    s2 = float(np.dot(v2_index[cat_id_to_idx[n1["id"]]], v2_index[cat_id_to_idx[n2["id"]]]))
                    s3 = float(np.dot(v3_index[cat_id_to_idx[n1["id"]]], v3_index[cat_id_to_idx[n2["id"]]]))
                seen.add(pair_key)
                pairs.append({
                    "pair_id": f"{pair_key[0]}__{pair_key[1]}",
                    "query_id": n1["id"],
                    "candidate_id": n2["id"],
                    "category": "necklace",
                    "query_path": f"/media/necklace/{n1['filename']}",
                    "candidate_path": f"/media/necklace/{n2['filename']}",
                    "dhash_distance": dh,
                    "dinov2_sim": round(s2, 4) if s2 is not None else None,
                    "dinov3_sim": round(s3, 4) if s3 is not None else None,
                    "priority": 2 if dh <= 3 else 3,
                    "source_note": f"Necklace candidate pair: dHash={dh}, DINOv2={s2}, DINOv3={s3}"
                })

    # Sort queue by priority, then dhash distance ascending
    pairs.sort(key=lambda p: (p["priority"], p["dhash_distance"], -(p["dinov2_sim"] or 0)))
    
    out_path = Path("data/annotation_queue.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(pairs, fp, indent=2)
        
    print(f"Annotation queue created with {len(pairs)} pairs ({sum(1 for p in pairs if p['category'] == 'ring')} rings, {sum(1 for p in pairs if p['category'] == 'necklace')} necklaces).")

if __name__ == "__main__":
    build_queue()
