"""Catalog vector indexing, persistence, and cosine similarity search engine."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

from engine.embeddings import BaseEncoder, get_encoder

EXACT_MATCH_THRESHOLD = 0.88
SIMILARITY_THRESHOLD = 0.70
SIMILAR_DESIGN_THRESHOLD = 0.52

class CatalogIndex:
    def __init__(self, model_name: str = "dinov2", index_dir: str = "data/index"):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_path = self.index_dir / f"{model_name}_embeddings.npy"
        self.meta_path = self.index_dir / f"{model_name}_meta.json"
        
        self.embeddings: Optional[np.ndarray] = None # Shape (N, D), float32
        self.metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}

    def is_indexed(self) -> bool:
        return self.embeddings_path.exists() and self.meta_path.exists()

    def load(self):
        """Load cached embeddings and metadata from disk."""
        if not self.is_indexed():
            raise FileNotFoundError(f"Index files not found at {self.index_dir} for model {self.model_name}")
        self.embeddings = np.load(str(self.embeddings_path))
        with open(self.meta_path, "r") as fp:
            self.metadata = json.load(fp)
        self._id_to_idx = {item["id"]: idx for idx, item in enumerate(self.metadata)}

    def build_from_manifest(
        self,
        manifest_path: str = "data/manifest.json",
        encoder: Optional[BaseEncoder] = None,
        batch_size: int = 32
    ) -> int:
        """Compute embeddings for all catalog items and persist to disk."""
        if encoder is None:
            encoder = get_encoder(self.model_name)

        with open(manifest_path, "r") as fp:
            manifest = json.load(fp)

        # Index items marked for catalog split
        catalog_items = [img for img in manifest["images"] if img.get("split") == "catalog"]
        if not catalog_items:
            # Fallback to all images if split wasn't marked
            catalog_items = manifest["images"]

        print(f"Indexing {len(catalog_items)} catalog items with model '{self.model_name}'...")
        t0 = time.time()

        images = []
        meta = []
        for item in catalog_items:
            img_path = Path(item["rel_path"])
            if not img_path.exists():
                img_path = Path("Jewellery_Data") / item["category"] / item["filename"]
            
            with Image.open(img_path) as img:
                images.append(img.convert("RGB"))
            
            meta.append({
                "id": item["id"],
                "filename": item["filename"],
                "rel_path": item["rel_path"],
                "category": item["category"],
                "cluster_id": item.get("cluster_id", item["id"]),
                "width": item["width"],
                "height": item["height"]
            })

        embeddings = encoder.encode_batch(images, batch_size=batch_size)
        elapsed = time.time() - t0
        print(f"Computed {len(embeddings)} embeddings in {elapsed:.2f}s ({len(embeddings)/elapsed:.1f} img/s)")

        # Persist
        np.save(str(self.embeddings_path), embeddings)
        with open(self.meta_path, "w") as fp:
            json.dump(meta, fp, indent=2)

        self.embeddings = embeddings
        self.metadata = meta
        self._id_to_idx = {item["id"]: idx for idx, item in enumerate(self.metadata)}
        return len(catalog_items)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        exclude_ids: Optional[set] = None,
        aggregate_sku: bool = False,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform exact cosine similarity search over catalog embeddings.
        
        If aggregate_sku is True, candidates are grouped by physical SKU (or item id),
        returning the highest-scoring view per distinct product.
        """
        if self.embeddings is None:
            self.load()

        if exclude_ids is None:
            exclude_ids = set()

        # Filter indices by category, exclusion, and tenant
        candidate_indices = []
        for idx, item in enumerate(self.metadata):
            if item["id"] in exclude_ids:
                continue
            if tenant_id and item.get("tenant_id") and item.get("tenant_id") != tenant_id:
                continue
            if category_filter and category_filter.lower() != "all":
                if item["category"].lower() != category_filter.lower():
                    continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return {"total_candidates": 0, "has_reliable_match": False, "results": []}

        cand_indices_arr = np.array(candidate_indices, dtype=np.int32)
        cand_embeddings = self.embeddings[cand_indices_arr] # Shape (M, D)

        # Dot product for unit-normalized vectors equals cosine similarity
        similarities = np.dot(cand_embeddings, query_embedding) # Shape (M,)

        raw_candidates = []
        for rank_idx, orig_idx in enumerate(cand_indices_arr):
            sim = float(similarities[rank_idx])
            item = self.metadata[orig_idx]

            # Confidence determination
            if sim >= EXACT_MATCH_THRESHOLD:
                match_type = "exact_match"
                confidence_label = "Verified Exact SKU / Twin"
            elif sim >= SIMILARITY_THRESHOLD:
                match_type = "high_confidence"
                confidence_label = "High Confidence Candidate"
            elif sim >= SIMILAR_DESIGN_THRESHOLD:
                match_type = "similar_design"
                confidence_label = "Visually Similar Design"
            else:
                match_type = "low_confidence"
                confidence_label = "Low Confidence Candidate"

            raw_candidates.append({
                "id": item["id"],
                "filename": item.get("filename", f"{item['id']}.jpg"),
                "rel_path": item.get("rel_path", f"Jewellery_Data/{item['category']}/{item.get('filename', item['id'] + '.jpg')}"),
                "category": item.get("category", "unknown"),
                "cluster_id": item.get("cluster_id", item["id"]),
                "sku_id": item.get("sku_id") or f"SKU-{item.get('category', 'item').upper()[:4]}-{item['id']}",
                "view_angle": item.get("view_angle", "frontal_angle"),
                "similarity": round(sim, 4),
                "match_type": match_type,
                "confidence_label": confidence_label,
                "metal_color": item.get("metal_color", "unknown"),
                "metal_purity": item.get("metal_purity", "unknown"),
                "gemstone_type": item.get("gemstone_type", "unknown"),
                "gemstone_shape": item.get("gemstone_shape", "unknown"),
                "setting_type": item.get("setting_type", "unknown"),
                "tenant_id": item.get("tenant_id", "default_tenant"),
                "width": item.get("width", 0),
                "height": item.get("height", 0)
            })

        if aggregate_sku:
            # Group by sku_id
            sku_groups = {}
            for cand in raw_candidates:
                sku_key = cand.get("sku_id") or cand["id"]
                if sku_key not in sku_groups:
                    sku_groups[sku_key] = []
                sku_groups[sku_key].append(cand)

            aggregated_results = []
            for sku_key, views in sku_groups.items():
                best_view = max(views, key=lambda x: x["similarity"])
                entry = dict(best_view)
                entry["matched_views_count"] = len(views)
                entry["matched_view_ids"] = [v["id"] for v in views]
                entry["matched_view_angles"] = [v.get("view_angle", "unknown") for v in views]
                aggregated_results.append(entry)

            aggregated_results.sort(key=lambda x: x["similarity"], reverse=True)
            results = aggregated_results[:top_k]
            for rank, r in enumerate(results, start=1):
                r["rank"] = rank
        else:
            raw_candidates.sort(key=lambda x: x["similarity"], reverse=True)
            results = raw_candidates[:top_k]
            for rank, r in enumerate(results, start=1):
                r["rank"] = rank

        has_reliable = len(results) > 0 and results[0]["similarity"] >= SIMILARITY_THRESHOLD

        return {
            "total_candidates": len(candidate_indices),
            "has_reliable_match": has_reliable,
            "top_similarity": results[0]["similarity"] if results else 0.0,
            "results": results
        }

    def stats(self) -> Dict[str, Any]:
        """Return catalog stats."""
        if self.embeddings is None and self.is_indexed():
            self.load()
        
        categories = {}
        for item in self.metadata:
            cat = item.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "model_name": self.model_name,
            "indexed_count": len(self.metadata) if self.metadata else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "categories": categories,
            "is_ready": self.embeddings is not None
        }
