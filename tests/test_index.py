"""Tests for catalog index, cosine similarity ranking, and confidence thresholds."""

import numpy as np
import pytest

from engine.index import CatalogIndex, EXACT_MATCH_THRESHOLD, SIMILARITY_THRESHOLD

def test_index_cosine_similarity_ranking(tmp_path):
    index = CatalogIndex(model_name="test_model", index_dir=str(tmp_path))
    
    # Create 3 synthetic unit-normalized vectors of dim 4
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.95, 0.312, 0.0, 0.0], dtype=np.float32)
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    
    embeddings = np.vstack([v1, v2, v3])
    metadata = [
        {"id": "ring_001", "filename": "ring_001.jpg", "category": "ring", "cluster_id": "c1", "width": 262, "height": 262},
        {"id": "ring_002", "filename": "ring_002.jpg", "category": "ring", "cluster_id": "c1", "width": 262, "height": 262},
        {"id": "neck_001", "filename": "neck_001.jpg", "category": "necklace", "cluster_id": "c2", "width": 262, "height": 262},
    ]

    index.embeddings = embeddings
    index.metadata = metadata
    index._id_to_idx = {m["id"]: i for i, m in enumerate(metadata)}

    # Query with exact v1
    res = index.search(query_embedding=v1, top_k=3)
    results = res["results"]
    assert len(results) == 3
    assert results[0]["id"] == "ring_001"
    assert pytest.approx(results[0]["similarity"], 0.001) == 1.0
    assert results[0]["match_type"] == "exact_match"
    
    assert results[1]["id"] == "ring_002"
    assert results[1]["similarity"] >= SIMILARITY_THRESHOLD

    assert results[2]["id"] == "neck_001"
    assert results[2]["match_type"] == "low_confidence"

def test_index_category_filtering(tmp_path):
    index = CatalogIndex(model_name="test_filter", index_dir=str(tmp_path))
    v = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    metadata = [
        {"id": "r1", "filename": "r1.jpg", "category": "ring", "cluster_id": "c1", "width": 262, "height": 262},
        {"id": "n1", "filename": "n1.jpg", "category": "necklace", "cluster_id": "c2", "width": 262, "height": 262},
    ]
    index.embeddings = v
    index.metadata = metadata
    index._id_to_idx = {"r1": 0, "n1": 1}

    q = np.array([1.0, 0.0], dtype=np.float32)
    # Search rings only
    res = index.search(query_embedding=q, top_k=5, category_filter="ring")
    assert len(res["results"]) == 1
    assert res["results"][0]["category"] == "ring"

def test_index_exclusion(tmp_path):
    index = CatalogIndex(model_name="test_excl", index_dir=str(tmp_path))
    v = np.array([[1.0, 0.0], [0.9, 0.4]], dtype=np.float32)
    metadata = [
        {"id": "r1", "filename": "r1.jpg", "category": "ring", "cluster_id": "c1", "width": 262, "height": 262},
        {"id": "r2", "filename": "r2.jpg", "category": "ring", "cluster_id": "c1", "width": 262, "height": 262},
    ]
    index.embeddings = v
    index.metadata = metadata
    index._id_to_idx = {"r1": 0, "r2": 1}

    q = np.array([1.0, 0.0], dtype=np.float32)
    # Exclude r1 (self query)
    res = index.search(query_embedding=q, top_k=5, exclude_ids={"r1"})
    assert len(res["results"]) == 1
    assert res["results"][0]["id"] == "r2"
