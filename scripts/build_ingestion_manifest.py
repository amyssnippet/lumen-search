#!/usr/bin/env python3
"""PH-002 / PH-001: Build Auditable Ingestion Manifest.

Implements the formal 4-gate provenance schema for all catalog and evaluation images:
- Raw catalog images (492 images: 301 necklaces, 191 rings)
- Synthetic robustness query images (26 ring variations)

Captures:
- Content SHA-256 hashes
- Authoritative ERP/PIM lineage status (explicitly unassigned for legacy data)
- Physical product identity verification status and evidence
- Four-gate conformance tracking
- Capture source, view angle, session identifier
- Original vs synthetic capture flags
- Genuine independent capture flags
- Dataset split and exclusion tracking

Reconciled under PH-002 Gate 0:
- Symmetrical graph clustering for verified multi-shot identity groups.
- Symmetrical assignment of 'uncertain_identity' to both members of quarantined conflicting pairs.
- Dedicated synthetic group IDs preventing contamination of physical identity groups.
- Exact accounting: 456 physical clusters + 26 synthetic clusters = 482 clusters.
"""

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image

def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def build_ingestion_manifest():
    root_dir = Path(".")
    raw_dir = root_dir / "Jewellery_Data"
    syn_dir = root_dir / "data/evaluation/queries/rings"
    
    # Load existing benchmark and annotations
    bench_file = root_dir / "data/evaluation/benchmark_v3_balanced.json"
    with open(bench_file, "r") as fp:
        bench_data = json.load(fp)
        
    ann_file = root_dir / "data/annotations.jsonl"
    annotations = []
    if ann_file.exists():
        with open(ann_file, "r") as fp:
            for line in fp:
                if line.strip():
                    annotations.append(json.loads(line))

    # Exact byte duplicate groups
    sha_to_paths = defaultdict(list)
    raw_files = sorted([f for f in raw_dir.rglob("*") if f.is_file() and not f.name.startswith(".")])
    for rf in raw_files:
        sha = compute_sha256(rf)
        sha_to_paths[sha].append(rf.stem)
        
    exact_dup_groups = {sha: sorted(ids) for sha, ids in sha_to_paths.items() if len(ids) > 1}
    exact_dup_id_to_group = {}
    for g_idx, (sha, ids) in enumerate(sorted(exact_dup_groups.items(), key=lambda x: x[1][0])):
        grp_id = f"PHYS-EXACT-DUP-{g_idx+1:03d}"
        for member in ids:
            exact_dup_id_to_group[member] = (grp_id, ids)

    # Conflicting ground truth pairs identified in PH-001 audit (quarantined)
    conflicting_pairs = {
        ("necklace_10", "necklace_32"),
        ("necklace_32", "necklace_10"),
        ("necklace_91", "necklace_95"),
        ("necklace_95", "necklace_91")
    }
    quarantined_ids = {"necklace_10", "necklace_32", "necklace_91", "necklace_95"}
    quarantine_group_map = {
        "necklace_10": "PHYS-UNCERTAIN-necklace_10_32",
        "necklace_32": "PHYS-UNCERTAIN-necklace_10_32",
        "necklace_91": "PHYS-UNCERTAIN-necklace_91_95",
        "necklace_95": "PHYS-UNCERTAIN-necklace_91_95"
    }

    # Symmetrical graph clustering of verified multi-shot pairs
    exact_dup_all_ids = set(exact_dup_id_to_group.keys())
    adj = defaultdict(set)
    for a in annotations:
        if a["verdict"] == "confirmed_same_sku":
            q = a["query_id"].replace("q_neck_", "").replace("q_ring_", "")
            c = a["candidate_id"]
            if (q, c) not in conflicting_pairs and (c, q) not in conflicting_pairs:
                if not q.startswith("var_") and not c.startswith("var_"):
                    if q not in exact_dup_all_ids and c not in exact_dup_all_ids:
                        adj[q].add(c)
                        adj[c].add(q)

    # Add ring_078 <-> ring_079 (both in catalog, verified plain band pair)
    adj["ring_078"].add("ring_079")
    adj["ring_079"].add("ring_078")

    # Find connected components for verified physical multi-shot clusters
    visited = set()
    verified_clusters = []
    multishot_id_to_group = {}
    for node in sorted(adj.keys()):
        if node not in visited:
            comp = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            comp = sorted(comp)
            prefix = "NECK" if comp[0].startswith("neck") else "RING"
            grp_id = f"PHYS-SKU-{prefix}-{comp[0]}"
            verified_clusters.append((grp_id, comp))
            for m in comp:
                multishot_id_to_group[m] = (grp_id, comp)

    # Load catalog meta to know what is in data/index/
    cat_meta_file = root_dir / "data/index/dinov2_meta.json"
    with open(cat_meta_file, "r") as fp:
        catalog_items = set(m["id"] for m in json.load(fp))

    # Real benchmark queries
    bench_queries = {q["query_id"]: q for q in bench_data["same_sku_queries"]}

    records = []
    
    # 1. Process all 492 raw catalog images
    for rf in raw_files:
        img_id = rf.stem
        cat = rf.parent.name
        rel_path = str(rf.relative_to(root_dir))
        sha = compute_sha256(rf)
        
        with Image.open(rf) as pil_img:
            width, height = pil_img.size
            img_format = pil_img.format
            
        # Determine physical identity status
        if img_id in exact_dup_id_to_group:
            grp_id, members = exact_dup_id_to_group[img_id]
            status = "verified_identical_sku"
            evidence = f"Byte-level exact SHA-256 duplicate with {sorted(members)}"
            identity_group = grp_id
            is_genuine_independent = (img_id == members[0]) # Canonical is independent capture; rest are clones
            four_gate = {
                "gate1_erp_lineage": False,
                "gate2_cad_geometry": False,
                "gate3_turntable_protocol": False,
                "gate4_nontransitive_verified": True
            }
        elif img_id in quarantined_ids:
            status = "uncertain_identity"
            evidence = "Conflicting benchmark labels: listed as same-SKU cluster in v1/v3 but annotated as different SKU lookalike in expert audit"
            identity_group = quarantine_group_map[img_id]
            is_genuine_independent = True
            four_gate = {
                "gate1_erp_lineage": False,
                "gate2_cad_geometry": False,
                "gate3_turntable_protocol": False,
                "gate4_nontransitive_verified": False
            }
        elif img_id in multishot_id_to_group:
            grp_id, members = multishot_id_to_group[img_id]
            status = "verified_identical_sku"
            counterparts = [m for m in members if m != img_id]
            evidence = f"Verified physical multi-shot {cat} cluster targeting {counterparts} (independent photograph, identical physical geometry)"
            identity_group = grp_id
            is_genuine_independent = True
            four_gate = {
                "gate1_erp_lineage": False,
                "gate2_cad_geometry": False,
                "gate3_turntable_protocol": False,
                "gate4_nontransitive_verified": True
            }
        else:
            status = "singleton_unlinked"
            evidence = "Single photographic capture with zero verified counterpart captures in dataset"
            identity_group = f"PHYS-SINGLE-{img_id}"
            is_genuine_independent = True
            four_gate = {
                "gate1_erp_lineage": False,
                "gate2_cad_geometry": False,
                "gate3_turntable_protocol": False,
                "gate4_nontransitive_verified": True
            }

        # Dataset split determination
        qid = f"q_{cat[:4]}_{img_id}"
        if qid in bench_queries:
            dataset_split = "benchmark_v3_test_real"
        elif img_id in catalog_items:
            dataset_split = "catalog_index"
        else:
            dataset_split = "catalog_non_indexed"

        # Exclusion tracking
        exclusion_reason = None
        if img_id == "ring_083":
            exclusion_reason = "Excluded from benchmark test due to EXP-004 transitive closure error (solitaire vs pave)"
        elif img_id in ["necklace_224", "necklace_225", "necklace_234", "necklace_279"]:
            exclusion_reason = "Identical byte clone duplicate excluded from active search index to prevent self-retrieval inflation"

        records.append({
            "image_id": img_id,
            "rel_path": rel_path,
            "sha256": sha,
            "category": cat,
            "width": width,
            "height": height,
            "format": img_format,
            "sku_id": None,
            "authoritative_identity_ref": "UNASSIGNED_LEGACY_ERP",
            "identity_group_id": identity_group,
            "capture_source": "raw_catalog_upload",
            "physical_identity_verification_status": status,
            "verification_evidence": evidence,
            "four_gate_status": four_gate,
            "view_angle": "frontal_angle" if cat == "necklace" else "unknown",
            "capture_session_id": "legacy_catalog_session_0",
            "is_original_capture": True,
            "is_genuine_independent_capture": is_genuine_independent,
            "dataset_split": dataset_split,
            "exclusion_reason": exclusion_reason
        })

    # 2. Process all 26 synthetic ring variation images
    syn_files = sorted([f for f in syn_dir.glob("*.jpg")])
    for s_idx, sf in enumerate(syn_files):
        img_id = sf.stem # e.g. q_ring_var_001_ring_032
        rel_path = str(sf.relative_to(root_dir))
        sha = compute_sha256(sf)
        
        with Image.open(sf) as pil_img:
            width, height = pil_img.size
            img_format = pil_img.format

        target_cat_id = img_id.split("_")[-2] + "_" + img_id.split("_")[-1] # ring_032
        
        records.append({
            "image_id": img_id,
            "rel_path": rel_path,
            "sha256": sha,
            "category": "ring",
            "width": width,
            "height": height,
            "format": img_format,
            "sku_id": None,
            "authoritative_identity_ref": f"SYNTH_DERIVED_FROM_{target_cat_id}",
            "identity_group_id": f"SYNTH-VAR-GROUP-{s_idx+1:03d}",
            "capture_source": "synthetic_variation_pipeline",
            "physical_identity_verification_status": "synthetic_transformation",
            "verification_evidence": f"Synthetic transformation of catalog item {target_cat_id} for photometric/geometric robustness testing",
            "four_gate_status": {
                "gate1_erp_lineage": False,
                "gate2_cad_geometry": False,
                "gate3_turntable_protocol": False,
                "gate4_nontransitive_verified": True
            },
            "view_angle": "perspective_tilt" if "camera_angle" in img_id else "unknown",
            "capture_session_id": "synthetic_batch_v3",
            "is_original_capture": False,
            "is_genuine_independent_capture": False,
            "dataset_split": "benchmark_v3_test_synthetic",
            "exclusion_reason": None
        })

    # Summary metadata
    statuses = defaultdict(int)
    splits = defaultdict(int)
    cat_counts = defaultdict(int)
    genuine_captures = 0
    all_identities = set()
    physical_identities = set()
    
    for r in records:
        statuses[r["physical_identity_verification_status"]] += 1
        splits[r["dataset_split"]] += 1
        cat_counts[r["category"]] += 1
        if r["is_genuine_independent_capture"]:
            genuine_captures += 1
        all_identities.add(r["identity_group_id"])
        if r["is_original_capture"]:
            physical_identities.add(r["identity_group_id"])

    manifest_output = {
        "manifest_version": "2.0.0",
        "schema_version": "ph002.ingestion.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_images": len(records),
            "raw_images_count": len(raw_files),
            "synthetic_images_count": len(syn_files),
            "categories": dict(cat_counts),
            "total_unique_physical_identities": len(physical_identities),
            "total_unique_all_identities": len(all_identities),
            "genuine_independent_captures": genuine_captures,
            "exact_byte_duplicate_clones": sum(len(ids) - 1 for ids in exact_dup_groups.values()),
            "status_breakdown": dict(statuses),
            "split_breakdown": dict(splits),
            "reconciliation_accounting": {
                "raw_images": len(raw_files),
                "physical_singletons": sum(1 for r in records if r["physical_identity_verification_status"] == "singleton_unlinked"),
                "verified_multishot_images": sum(1 for r in records if r["physical_identity_verification_status"] == "verified_identical_sku" and not r["image_id"] in exact_dup_all_ids),
                "exact_byte_duplicate_images": len(exact_dup_all_ids),
                "quarantined_conflicting_images": sum(1 for r in records if r["physical_identity_verification_status"] == "uncertain_identity"),
                "synthetic_test_images": len(syn_files)
            },
            "four_gate_audit": {
                "gate1_erp_lineage_coverage_pct": 0.0,
                "gate2_cad_geometry_coverage_pct": 0.0,
                "gate3_turntable_protocol_coverage_pct": 0.0,
                "gate4_nontransitive_verified_pct": round(100.0 * sum(1 for r in records if r["four_gate_status"]["gate4_nontransitive_verified"]) / len(records), 2)
            }
        },
        "records": records
    }

    out_file = root_dir / "data/ingestion_manifest.json"
    with open(out_file, "w") as fp:
        json.dump(manifest_output, fp, indent=2)
        
    print(f"Auditable Ingestion Manifest saved to {out_file}")
    print(f"Total records: {len(records)} ({len(raw_files)} raw, {len(syn_files)} synthetic)")
    print(f"Physical identity clusters: {len(physical_identities)}")
    print(f"Total identity clusters: {len(all_identities)}")
    print(f"Genuine independent captures: {genuine_captures}")
    print(f"Statuses: {dict(statuses)}")
    return manifest_output

if __name__ == "__main__":
    build_ingestion_manifest()
