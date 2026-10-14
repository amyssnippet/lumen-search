"""Soft Spatial Attention and Patch Token Reweighting for Jewellery Visual Retrieval.

Provides train-free, deterministic spatial token weighting strategies:
- Uniform patch pooling (Baseline)
- Multi-cue Foreground Saliency (non-white backdrop, color saturation, edge gradient)
- Soft Component-Region Weighting (continuous distance decay from bounding box)
- Anisotropic Gaussian Spatial Weighting
- Combined Soft Weighting with graceful fallback
- Normalized weighted feature aggregation
- Attention heatmap rendering for UI inspection
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import cv2

GRID_SIZE = 16  # 16x16 = 256 patches

def get_grid_centers(grid_size: int = GRID_SIZE) -> np.ndarray:
    """Return (grid_size*grid_size, 2) normalized [y, x] patch centers in [0, 1]."""
    coords = []
    for r in range(grid_size):
        for c in range(grid_size):
            y = (r + 0.5) / float(grid_size)
            x = (c + 0.5) / float(grid_size)
            coords.append((y, x))
    return np.array(coords, dtype=np.float32)

def compute_uniform_weights(grid_size: int = GRID_SIZE) -> np.ndarray:
    """Configuration A: Equal weight 1.0 for all patches."""
    return np.ones(grid_size * grid_size, dtype=np.float32)

def compute_foreground_weights(
    image: Image.Image,
    grid_size: int = GRID_SIZE,
    bg_threshold: int = 240,
    min_floor: float = 0.10
) -> np.ndarray:
    """Configuration B: Multi-cue foreground saliency.
    
    Combines:
    1. Studio non-white background contrast
    2. Chromatic saturation (metals, gemstones)
    3. Structural texture & gradient magnitude
    """
    img_rgb = np.array(image.convert("RGB"))
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    # Edge gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    cell_h = h / float(grid_size)
    cell_w = w / float(grid_size)
    scores = []

    for r in range(grid_size):
        for c in range(grid_size):
            y1, y2 = max(0, int(r * cell_h)), min(h, int((r + 1) * cell_h))
            x1, x2 = max(0, int(c * cell_w)), min(w, int((c + 1) * cell_w))
            
            c_gray = gray[y1:y2, x1:x2]
            c_sat = sat[y1:y2, x1:x2]
            c_grad = grad_mag[y1:y2, x1:x2]

            if c_gray.size == 0:
                scores.append(min_floor)
                continue

            non_white = float(np.mean(c_gray < bg_threshold))
            chromatic = float(np.mean(c_sat > 20))
            texture = float(np.mean(c_grad > 15))

            # Multi-signal fused saliency
            cell_score = 0.40 * non_white + 0.30 * chromatic + 0.30 * texture
            scores.append(cell_score)

    scores_arr = np.array(scores, dtype=np.float32)
    max_val = float(np.max(scores_arr))
    if max_val > 0.05:
        norm_scores = scores_arr / max_val
        weights = np.maximum(min_floor, np.minimum(1.0, norm_scores))
    else:
        weights = np.ones_like(scores_arr)

    return weights.astype(np.float32)

def compute_soft_region_weights(
    box: List[float],
    grid_size: int = GRID_SIZE,
    sigma: float = 0.15,
    min_weight: float = 0.25
) -> np.ndarray:
    """Configuration C: Soft component-region weighting.
    
    Assigns 1.0 inside box, smooth exponential decay outside, retaining min_weight
    for surrounding jewellery structure (chains, bails, ring shanks).
    box: normalized [ymin, xmin, ymax, xmax] in [0, 1].
    """
    coords = get_grid_centers(grid_size)
    y, x = coords[:, 0], coords[:, 1]
    ymin, xmin, ymax, xmax = box

    # Distance to box boundary
    dy = np.maximum(0.0, np.maximum(ymin - y, y - ymax))
    dx = np.maximum(0.0, np.maximum(xmin - x, x - xmax))
    dist = np.sqrt(dy**2 + dx**2)

    weights = np.maximum(min_weight, np.exp(-dist / max(0.01, sigma)))
    return weights.astype(np.float32)

def compute_gaussian_weights(
    box: List[float],
    grid_size: int = GRID_SIZE,
    min_weight: float = 0.15
) -> np.ndarray:
    """Configuration D: 2D Anisotropic Gaussian spatial prior centered on component."""
    coords = get_grid_centers(grid_size)
    y, x = coords[:, 0], coords[:, 1]
    ymin, xmin, ymax, xmax = box

    yc = (ymin + ymax) / 2.0
    xc = (xmin + xmax) / 2.0
    sig_y = max(0.12, (ymax - ymin) / 1.5)
    sig_x = max(0.12, (xmax - xmin) / 1.5)

    d2 = ((y - yc)**2 / (2.0 * sig_y**2)) + ((x - xc)**2 / (2.0 * sig_x**2))
    weights = np.maximum(min_weight, np.exp(-d2))
    return weights.astype(np.float32)

def compute_combined_weights(
    image: Image.Image,
    box: Optional[List[float]],
    is_fallback: bool = False,
    grid_size: int = GRID_SIZE,
    min_floor: float = 0.05
) -> np.ndarray:
    """Configuration E: Combined multi-cue foreground and soft component region weighting."""
    fg_w = compute_foreground_weights(image, grid_size=grid_size, min_floor=min_floor)

    if is_fallback or box is None:
        # Fallback to foreground saliency alone
        return fg_w

    reg_w = compute_soft_region_weights(box, grid_size=grid_size, min_weight=0.25)
    combined = fg_w * reg_w
    # Normalize relative to max to preserve dynamic range
    max_val = float(np.max(combined))
    if max_val > 1e-5:
        combined = combined / max_val
    return np.maximum(min_floor, np.minimum(1.0, combined)).astype(np.float32)

def aggregate_patch_tokens(
    patch_tokens: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Aggregate patch tokens with soft spatial weights:
    
    z = normalize( (sum_i w_i * p_i) / (sum_i w_i + eps) )
    patch_tokens: shape (256, D) or (B, 256, D)
    weights: shape (256,) or (B, 256,)
    Returns unit L2-normalized float32 vector of shape (D,) or (B, D).
    """
    if patch_tokens.ndim == 2:
        # Single image: (256, D)
        w = weights[:, np.newaxis] # (256, 1)
        w_sum = np.sum(w) + eps
        agg = np.sum(patch_tokens * w, axis=0) / w_sum
        norm = np.linalg.norm(agg)
        if norm > 0:
            agg = agg / norm
        return agg.astype(np.float32)
    elif patch_tokens.ndim == 3:
        # Batch: (B, 256, D)
        w = weights[:, :, np.newaxis] # (B, 256, 1)
        w_sum = np.sum(w, axis=1, keepdims=True) + eps # (B, 1, 1)
        agg = np.sum(patch_tokens * w, axis=1) / w_sum.squeeze(-1) # (B, D)
        norms = np.linalg.norm(agg, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        agg = agg / norms
        return agg.astype(np.float32)
    else:
        raise ValueError(f"Invalid patch_tokens shape {patch_tokens.shape}")

def render_attention_heatmap(
    weights: np.ndarray,
    base_image: Image.Image,
    grid_size: int = GRID_SIZE,
    alpha: float = 0.45
) -> Image.Image:
    """Render 2D attention heatmap blended over query image for visual explainability."""
    w_2d = weights.reshape((grid_size, grid_size))
    # Normalize to [0, 255]
    w_min, w_max = np.min(w_2d), np.max(w_2d)
    if w_max > w_min:
        w_norm = ((w_2d - w_min) / (w_max - w_min) * 255).astype(np.uint8)
    else:
        w_norm = np.full((grid_size, grid_size), 128, dtype=np.uint8)

    w_img, h_img = base_image.size
    heatmap_resized = cv2.resize(w_norm, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_TURBO)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    base_rgb = np.array(base_image.convert("RGB"))
    blended = (alpha * colormap_rgb + (1.0 - alpha) * base_rgb).astype(np.uint8)
    return Image.fromarray(blended)
