"""Query Image Enhancement, Quality Assessment, and Robust Preprocessing for Jewellery Retrieval."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageOps
import cv2

@dataclass
class QualityReport:
    is_valid: bool = True
    width: int = 0
    height: int = 0
    sharpness_score: float = 0.0
    is_blurry: bool = False
    mean_luminance: float = 0.0
    is_underexposed: bool = False
    is_overexposed: bool = False
    is_low_res: bool = False
    jewellery_bbox_ratio: float = 1.0
    skin_detected: bool = False
    skin_ratio: float = 0.0
    warnings: List[str] = field(default_factory=list)

@dataclass
class PreprocessConfig:
    target_size: int = 224
    enable_denoise: bool = False
    enable_clahe: bool = False
    enable_autocrop: bool = False
    auto_enhance: bool = False  # Applies enhancement conditionally if quality check indicates poor exposure
    enable_robust_crops: bool = False
    neutralize_skin: bool = True

def correct_exif_orientation(image: Image.Image) -> Image.Image:
    """Transpose image if EXIF orientation flag exists."""
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image

def assess_image_quality(image: Image.Image) -> QualityReport:
    """Assess sharpness, exposure, resolution, and visible object proportion."""
    w, h = image.size
    report = QualityReport(width=w, height=h)

    if w < 100 or h < 100:
        report.is_low_res = True
        report.warnings.append(f"Low resolution image ({w}x{h}). Micro-details may be degraded.")

    # Convert to grayscale numpy for CV analysis
    img_gray = np.array(image.convert("L"))

    # Sharpness via Laplacian variance
    lap_var = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    report.sharpness_score = round(lap_var, 1)
    if lap_var < 50.0:
        report.is_blurry = True
        report.warnings.append(f"Image may be blurry (sharpness score {report.sharpness_score} < 50).")

    # Luminance / exposure check
    mean_lum = float(np.mean(img_gray))
    report.mean_luminance = round(mean_lum, 1)
    if mean_lum < 40.0:
        report.is_underexposed = True
        report.warnings.append("Image appears dark / underexposed. Metal settings may be obscured.")
    elif mean_lum > 225.0:
        report.is_overexposed = True
        report.warnings.append("Image appears overexposed / washed out. Highlights may clip stone facets.")

    # Foreground object estimation & skin tone detection
    try:
        # Check for skin tones (worn jewellery detection) in HSV color space
        hsv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2HSV)
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        skin_mask = ((h_chan < 25) | (h_chan > 165)) & (s_chan > 20) & (s_chan < 160) & (v_chan > 40) & (v_chan < 250)
        skin_pct = float(np.mean(skin_mask))
        report.skin_ratio = round(skin_pct, 3)
        if skin_pct > 0.15:
            report.skin_detected = True
            report.warnings.append(f"Skin tones detected in background ({skin_pct*100:.1f}% area). Likely worn jewellery.")

        # Otsu thresholding after Gaussian blur
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        # Background is usually bright in jewelry studio shots; invert if so
        if mean_lum > 128:
            _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            _, _, cw, ch = cv2.boundingRect(c)
            bbox_area = (cw * ch) / (w * h)
            report.jewellery_bbox_ratio = round(bbox_area, 2)
            if bbox_area < 0.15:
                report.warnings.append("Jewellery occupies a small portion of the image.")
    except Exception:
        pass

    return report

def pad_to_aspect_ratio(image: Image.Image, target_size: int = 224, fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Pad image to square with neutral background to preserve geometry of rings and chains."""
    image = image.convert("RGB")
    w, h = image.size
    max_dim = max(w, h)
    
    padded = Image.new("RGB", (max_dim, max_dim), fill_color)
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    padded.paste(image, (paste_x, paste_y))

    if max_dim != target_size:
        padded = padded.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return padded

def apply_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
    """Apply CLAHE on LAB luminance channel, strictly preserving A/B color channels (metal color fidelity)."""
    img_rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    rgb_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_enhanced)

def apply_bilateral_denoise(image: Image.Image, d: int = 5, sigma_color: int = 40, sigma_space: int = 40) -> Image.Image:
    """Mild bilateral filter smoothing background noise while preserving sharp gem facets and prong edges."""
    img_rgb = np.array(image.convert("RGB"))
    denoised = cv2.bilateralFilter(img_rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return Image.fromarray(denoised)

def detect_and_crop_jewellery(image: Image.Image, margin: float = 0.10, neutralize_skin: bool = True) -> Tuple[Image.Image, bool]:
    """Detect jewellery bounding box and neutralize skin background if present. Returns (cropped_image, skin_neutralized)."""
    img_rgb = np.array(image.convert("RGB"))
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    raw_skin = ((h_chan < 25) | (h_chan > 165)) & (s_chan > 20) & (s_chan < 160) & (v_chan > 40) & (v_chan < 250)
    # Exclude high-gradient edges (facets, prongs, engraved text) and specular glints so gold bands are preserved
    jewellery_features = (lap > 15.0) | (v_chan > 240)
    skin_mask = raw_skin & (~jewellery_features)
    skin_ratio = float(np.mean(skin_mask))

    # If skin/hand background detected, neutralize skin pixels to studio white (255, 255, 255)
    if skin_ratio > 0.15 and neutralize_skin:
        masked_rgb = img_rgb.copy()
        masked_rgb[skin_mask] = [255, 255, 255]
        non_white = (masked_rgb[:, :, 0] < 240) | (masked_rgb[:, :, 1] < 240) | (masked_rgb[:, :, 2] < 240)
        cnts, _ = cv2.findContours(non_white.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            box_area = (bw * bh) / (w * h)
            if 0.04 <= box_area <= 0.90:
                mx = int(bw * margin)
                my = int(bh * margin)
                x1 = max(0, bx - mx)
                y1 = max(0, by - my)
                x2 = min(w, bx + bw + mx)
                y2 = min(h, by + bh + my)
                return Image.fromarray(masked_rgb[y1:y2, x1:x2]), True
        return Image.fromarray(masked_rgb), True

    # Standard studio background check
    corners = [gray[0:20, 0:20], gray[0:20, w-20:w], gray[h-20:h, 0:20], gray[h-20:h, w-20:w]]
    bg_mean = np.mean([np.mean(c) for c in corners])
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if bg_mean > 180:
        _, thresh = cv2.threshold(blurred, int(bg_mean - 20), 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(blurred, int(bg_mean + 20), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, False

    c = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)
    
    # Only crop if bounding box is significant but not full frame
    box_area = (bw * bh) / (w * h)
    if box_area < 0.15 or box_area > 0.90:
        return image, False

    # Add margin
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, bx - mx)
    y1 = max(0, by - my)
    x2 = min(w, bx + bw + mx)
    y2 = min(h, by + bh + my)

    cropped = img_rgb[y1:y2, x1:x2]
    return Image.fromarray(cropped), False

def extract_robust_query_crops(image: Image.Image, config: Optional[PreprocessConfig] = None) -> List[Tuple[Image.Image, str]]:
    """Generate compact 3-crop test-time query views: global, foreground-isolated, and central zoom."""
    if config is None:
        config = PreprocessConfig()

    img = correct_exif_orientation(image)
    if img.mode != "RGB":
        img = img.convert("RGB")

    crops = []
    # 1. Global View (always present)
    crops.append((pad_to_aspect_ratio(img, target_size=config.target_size), "global"))

    # 2. Foreground / Skin-Neutralized Crop
    fg_img, was_neutralized = detect_and_crop_jewellery(img, margin=0.10, neutralize_skin=config.neutralize_skin)
    if was_neutralized or fg_img.size != img.size:
        crops.append((pad_to_aspect_ratio(fg_img, target_size=config.target_size), "foreground_isolated"))

    # 3. Center Zoom Crop (80% scale to emphasize center stone / setting details)
    w, h = img.size
    cw, ch = int(w * 0.80), int(h * 0.80)
    x1, y1 = (w - cw) // 2, (h - ch) // 2
    center_img = img.crop((x1, y1, x1 + cw, y1 + ch))
    crops.append((pad_to_aspect_ratio(center_img, target_size=config.target_size), "center_zoom_80"))

    return crops

def preprocess_query_image(image: Image.Image, config: Optional[PreprocessConfig] = None) -> Tuple[Image.Image, QualityReport, Dict[str, Any]]:
    """Complete preprocessing pipeline for query image, returning (processed_image, quality_report, metadata)."""
    if config is None:
        config = PreprocessConfig()

    # Step 1: EXIF correction
    img = correct_exif_orientation(image)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Step 2: Quality Assessment
    quality = assess_image_quality(img)

    applied_steps = ["exif_correction"]

    # Step 3: Optional jewellery crop or auto-neutralize skin if detected
    if config.enable_autocrop or (config.neutralize_skin and quality.skin_detected):
        cropped, was_skin_neut = detect_and_crop_jewellery(img, margin=0.10, neutralize_skin=config.neutralize_skin)
        if cropped.size != img.size or was_skin_neut:
            img = cropped
            applied_steps.append("skin_neutralize_crop" if was_skin_neut else "autocrop")

    # Step 4: Conditional or forced CLAHE
    if config.enable_clahe or (config.auto_enhance and (quality.is_underexposed or quality.mean_luminance < 100)):
        img = apply_clahe(img, clip_limit=2.0)
        applied_steps.append("clahe_contrast")

    # Step 5: Conditional or forced denoising
    if config.enable_denoise:
        img = apply_bilateral_denoise(img, d=5, sigma_color=40, sigma_space=40)
        applied_steps.append("bilateral_denoise")

    # Step 6: Aspect padding & resize
    img = pad_to_aspect_ratio(img, target_size=config.target_size)
    applied_steps.append("aspect_pad_resize")

    meta = {
        "applied_steps": applied_steps,
        "sharpness": quality.sharpness_score,
        "mean_luminance": quality.mean_luminance,
        "skin_detected": quality.skin_detected,
        "skin_ratio": quality.skin_ratio,
        "warnings": quality.warnings
    }

    return img, quality, meta
