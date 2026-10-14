"""Jewellery component localization (gemstones, settings, pendants) for fine-grained visual search."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from PIL import Image

@dataclass
class LocalizationResult:
    box: List[float]             # Normalized [ymin, xmin, ymax, xmax] in [0, 1]
    box_pixels: List[int]       # Integer pixel coordinates [ymin, xmin, ymax, xmax]
    label: str                  # "pendant_stone", "centre_stone", "setting", "fallback"
    confidence: float           # Confidence score [0, 1]
    method: str                 # "morphological_saliency", "crown_setting_heuristic", "fallback"
    is_fallback: bool
    crop: Image.Image           # Cropped image of the region

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [ymin, xmin, ymax, xmax]."""
    y1 = max(box_a[0], box_b[0])
    x1 = max(box_a[1], box_b[1])
    y2 = min(box_a[2], box_b[2])
    x2 = min(box_a[3], box_b[3])
    
    inter_area = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)

class JewelleryLocalizer:
    """Lightweight, CPU-efficient localizer for centre stones, settings, and pendant gemstone clusters."""
    
    def __init__(self, bg_threshold: int = 240, min_area_pct: float = 0.002, max_area_pct: float = 0.90):
        self.bg_threshold = bg_threshold
        self.min_area_pct = min_area_pct
        self.max_area_pct = max_area_pct

    def locate(self, image: Image.Image, category: Optional[str] = None) -> LocalizationResult:
        """Locate the most distinctive jewellery region (pendant or ring centre stone)."""
        w, h = image.size
        im_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        
        # Segment foreground: non-white studio background or chromatic saturation
        fg_mask = (gray < self.bg_threshold) | (sat > 25)
        fg_uint8 = (fg_mask.astype(np.uint8)) * 255
        total_pixels = w * h
        fg_pixels = int(np.sum(fg_mask))
        
        # Normalize category input: strictly avoid guessing category from aspect ratio
        cat_norm = (category or "").strip().lower()
        if cat_norm in ("all", "none", "unknown", ""):
            cat_norm = None

        if cat_norm == "necklace":
            # Necklaces: use morphological opening to eliminate thin chain lines (< 5px)
            kernel_size = max(5, int(min(w, h) * 0.025))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened = cv2.morphologyEx(fg_uint8, cv2.MORPH_OPEN, kernel)
            
            cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                cnts, _ = cv2.findContours(fg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
            if cnts:
                # Rank contours: prefer dense area located in vertical lower half
                def score_contour(cnt):
                    area = cv2.contourArea(cnt)
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    centroid_y = by + bh / 2.0
                    # Weight area by vertical depth (pendants hang low)
                    return area * (1.0 + 0.5 * (centroid_y / h))

                best_cnt = max(cnts, key=score_contour)
                x, y, bw, bh = cv2.boundingRect(best_cnt)
                
                # Check area sanity
                area_ratio = (bw * bh) / total_pixels
                if self.min_area_pct <= area_ratio <= self.max_area_pct:
                    pad_x = int(bw * 0.15)
                    pad_y = int(bh * 0.15)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w, x + bw + pad_x)
                    y2 = min(h, y + bh + pad_y)
                    
                    crop = image.crop((x1, y1, x2, y2))
                    norm_box = [y1 / h, x1 / w, y2 / h, x2 / w]
                    return LocalizationResult(
                        box=norm_box,
                        box_pixels=[y1, x1, y2, x2],
                        label="pendant_stone",
                        confidence=round(min(1.0, 0.70 + 0.30 * (bw * bh / max(1, fg_pixels))), 3),
                        method="morphological_dense_contour",
                        is_fallback=False,
                        crop=crop
                    )

        elif cat_norm == "ring":
            # Rings: centre stone / setting is at the crown (upper-centre of the ring)
            cnts, _ = cv2.findContours(fg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                all_pts = np.vstack([c for c in cnts])
                rx, ry, rw, rh = cv2.boundingRect(all_pts)
                
                # Center stone setting is typically in the top 45% of the overall ring contour
                cy1 = max(0, ry - int(rh * 0.05))
                cy2 = min(h, ry + int(rh * 0.50))
                cx1 = max(0, rx + int(rw * 0.15))
                cx2 = min(w, rx + int(rw * 0.85))
                
                crop_w = cx2 - cx1
                crop_h = cy2 - cy1
                if crop_w > 10 and crop_h > 10:
                    crop = image.crop((cx1, cy1, cx2, cy2))
                    norm_box = [cy1 / h, cx1 / w, cy2 / h, cx2 / w]
                    return LocalizationResult(
                        box=norm_box,
                        box_pixels=[cy1, cx1, cy2, cx2],
                        label="centre_stone",
                        confidence=0.85,
                        method="crown_setting_heuristic",
                        is_fallback=False,
                        crop=crop
                    )

        else:
            # Unknown category: generic component saliency (dense foreground cluster) without aspect-ratio guessing
            kernel_size = max(5, int(min(w, h) * 0.025))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened = cv2.morphologyEx(fg_uint8, cv2.MORPH_OPEN, kernel)
            cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                best_cnt = max(cnts, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(best_cnt)
                area_ratio = (bw * bh) / total_pixels
                if self.min_area_pct <= area_ratio <= self.max_area_pct:
                    pad_x = int(bw * 0.15)
                    pad_y = int(bh * 0.15)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w, x + bw + pad_x)
                    y2 = min(h, y + bh + pad_y)
                    crop = image.crop((x1, y1, x2, y2))
                    return LocalizationResult(
                        box=[y1 / h, x1 / w, y2 / h, x2 / w],
                        box_pixels=[y1, x1, y2, x2],
                        label="jewellery_component",
                        confidence=0.75,
                        method="generic_dense_contour",
                        is_fallback=False,
                        crop=crop
                    )

        # Fallback: centre crop when detection is degenerate or ambiguous
        pad_y = int(h * 0.15)
        pad_x = int(w * 0.15)
        x1, y1, x2, y2 = pad_x, pad_y, w - pad_x, h - pad_y
        crop = image.crop((x1, y1, x2, y2))
        return LocalizationResult(
            box=[y1 / h, x1 / w, y2 / h, x2 / w],
            box_pixels=[y1, x1, y2, x2],
            label="fallback",
            confidence=0.40,
            method="fallback_centre_crop",
            is_fallback=True,
            crop=crop
        )
