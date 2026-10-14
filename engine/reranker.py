"""Local patch-token reranking and symmetric Chamfer geometric correspondence for fine jewellery."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

def compute_chamfer_similarity(
    query_patches: np.ndarray,
    candidate_patches: np.ndarray,
    query_fg_indices: Optional[np.ndarray] = None,
    cand_fg_indices: Optional[np.ndarray] = None,
    spatial_weight: float = 0.0
) -> float:
    """Compute symmetric foreground-masked Chamfer similarity between normalized patch token grids.
    
    Formula:
      Chamfer(Q, C) = 0.5 * ( (1/|Q_fg|) * sum_{i in Q_fg} max_{j in C_fg} Sim(q_i, c_j)
                            + (1/|C_fg|) * sum_{j in C_fg} max_{i in Q_fg} Sim(q_i, c_j) )
    
    Parameters:
      query_patches: (P, D) normalized float32
      candidate_patches: (P, D) normalized float32
      query_fg_indices: optional 1D array of patch indices belonging to the query foreground
      cand_fg_indices: optional 1D array of patch indices belonging to the candidate foreground
      spatial_weight: penalty coefficient (>= 0.0) applied to normalized 2D distance between patch coordinates
    """
    assert query_patches.ndim == 2, f"Expected 2D query patches, got {query_patches.shape}"
    assert candidate_patches.ndim == 2, f"Expected 2D candidate patches, got {candidate_patches.shape}"
    P, D = query_patches.shape
    assert candidate_patches.shape == (P, D), f"Dimension mismatch: {query_patches.shape} vs {candidate_patches.shape}"

    # Safe fallback if masks are missing or degenerate (< 4 patches detected)
    if query_fg_indices is None or len(query_fg_indices) < 4:
        query_fg_indices = np.arange(P, dtype=np.int32)
    else:
        query_fg_indices = np.asarray(query_fg_indices, dtype=np.int32)

    if cand_fg_indices is None or len(cand_fg_indices) < 4:
        cand_fg_indices = np.arange(P, dtype=np.int32)
    else:
        cand_fg_indices = np.asarray(cand_fg_indices, dtype=np.int32)

    # Sub-select foreground patches
    Q_fg = query_patches[query_fg_indices] # (M_q, D)
    C_fg = candidate_patches[cand_fg_indices] # (M_c, D)

    # Pairwise cosine similarity matrix
    sim_mat = np.dot(Q_fg, C_fg.T) # (M_q, M_c)

    # Apply spatial distance constraint if requested
    if spatial_weight > 0.0:
        side = int(round(np.sqrt(P)))
        qr, qc = query_fg_indices // side, query_fg_indices % side
        cr, cc = cand_fg_indices // side, cand_fg_indices % side
        # Normalized Euclidean distance in [0, 1] across the grid
        max_dist = np.sqrt(2.0 * ((side - 1) ** 2))
        dist_mat = np.sqrt((qr[:, None] - cr[None, :]) ** 2 + (qc[:, None] - cc[None, :]) ** 2) / max_dist
        sim_mat = sim_mat - spatial_weight * dist_mat

    # Symmetrically matched Chamfer correspondence
    max_q_to_c = np.max(sim_mat, axis=1) # (M_q,) best match in candidate for each query patch
    max_c_to_q = np.max(sim_mat, axis=0) # (M_c,) best match in query for each candidate patch

    q_score = float(np.mean(max_q_to_c))
    c_score = float(np.mean(max_c_to_q))
    return float(0.5 * (q_score + c_score))

def get_foreground_patch_indices(
    image: Image.Image,
    num_patches_side: int = 16,
    threshold: int = 240,
    min_fg_patches: int = 4
) -> np.ndarray:
    """Detect which patch positions in the grid contain jewellery foreground pixels rather than studio background."""
    img_gray = np.array(image.convert("L").resize((num_patches_side * 16, num_patches_side * 16)))
    h, w = img_gray.shape
    patch_size = h // num_patches_side
    
    fg_indices = []
    for r in range(num_patches_side):
        for c in range(num_patches_side):
            patch = img_gray[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size]
            # Foreground if darker than white threshold or contains textural variance
            if np.mean(patch) < threshold or np.std(patch) > 10.0:
                fg_indices.append(r * num_patches_side + c)

    # Fallback to all patches if detection failed or under threshold
    if len(fg_indices) < min_fg_patches:
        return np.arange(num_patches_side * num_patches_side, dtype=np.int32)
    return np.array(fg_indices, dtype=np.int32)

class PatchCatalogIndex:
    def __init__(self, model_name: str = "dinov3", index_dir: str = "data/index"):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.patch_file = self.index_dir / f"{model_name}_patches.npy"
        self.meta_file = self.index_dir / f"{model_name}_patches.meta.json"
        
        self.patches: Optional[np.ndarray] = None # Shape (N, 256, 384)
        self.meta: Optional[Dict[str, Any]] = None
        self.catalog_ids: List[str] = []
        self.catalog_fg_masks: List[np.ndarray] = []

    def is_cached(self) -> bool:
        return self.patch_file.exists() and self.meta_file.exists()

    def load(self):
        if not self.patch_file.exists():
            raise FileNotFoundError(f"Patch array cache not found: {self.patch_file}")
        self.patches = np.load(str(self.patch_file), mmap_mode="r")

        if self.meta_file.exists():
            with open(self.meta_file, "r") as fp:
                self.meta = json.load(fp)
            self.catalog_ids = self.meta.get("catalog_ids", [])
            raw_masks = self.meta.get("foreground_masks", [])
            self.catalog_fg_masks = [np.array(m, dtype=np.int32) for m in raw_masks]
        else:
            self.meta = {}
            self.catalog_ids = []
            self.catalog_fg_masks = [np.arange(self.patches.shape[1], dtype=np.int32) for _ in range(len(self.patches))]

    def get_candidate_fg_mask(self, cat_idx: int) -> Optional[np.ndarray]:
        if 0 <= cat_idx < len(self.catalog_fg_masks):
            return self.catalog_fg_masks[cat_idx]
        return None

    def rerank(
        self,
        query_patches: np.ndarray,
        candidate_catalog_indices: List[int],
        candidate_global_scores: List[float],
        query_fg_indices: Optional[np.ndarray] = None,
        patch_weight: float = 0.2,
        spatial_weight: float = 0.0
    ) -> List[Tuple[int, float, float]]:
        """Rerank candidates using exact fusion formula:
        
            final_score = (1 - patch_weight) * normalized_global_score + patch_weight * chamfer_score
        
        Where:
            patch_weight = 0.0 -> global score only
            patch_weight = 1.0 -> Chamfer score only
            
        Returns: list of (catalog_index, final_score, chamfer_score) sorted descending by final_score.
        """
        if self.patches is None:
            self.load()

        assert 0.0 <= patch_weight <= 1.0, f"patch_weight must be in [0.0, 1.0], got {patch_weight}"

        results = []
        for idx, g_score in zip(candidate_catalog_indices, candidate_global_scores):
            c_patches = self.patches[idx]
            c_fg = self.get_candidate_fg_mask(idx)

            if patch_weight > 0.0:
                ch_score = compute_chamfer_similarity(
                    query_patches=query_patches,
                    candidate_patches=c_patches,
                    query_fg_indices=query_fg_indices,
                    cand_fg_indices=c_fg,
                    spatial_weight=spatial_weight
                )
            else:
                ch_score = 0.0

            final_score = (1.0 - patch_weight) * float(g_score) + patch_weight * float(ch_score)
            results.append((idx, final_score, ch_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
