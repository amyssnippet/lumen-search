"""Tests for image preprocessing, orientation, and quality assessment."""

import numpy as np
import pytest
from PIL import Image

from engine.preprocessing import (
    correct_exif_orientation,
    assess_image_quality,
    pad_to_aspect_ratio,
    apply_clahe,
    apply_bilateral_denoise,
    PreprocessConfig,
    preprocess_query_image
)

def test_pad_to_aspect_ratio():
    # Non-square image (100x200)
    img = Image.new("RGB", (100, 200), (255, 0, 0))
    padded = pad_to_aspect_ratio(img, target_size=224)
    assert padded.size == (224, 224)
    # Check padding has occurred (should not distort image)
    arr = np.array(padded)
    assert arr.shape == (224, 224, 3)

def test_assess_image_quality_sharp_vs_blurry():
    # Sharp pattern (checkerboard)
    arr = np.zeros((200, 200), dtype=np.uint8)
    arr[::2, ::2] = 255
    sharp_img = Image.fromarray(arr)
    report_sharp = assess_image_quality(sharp_img)
    assert report_sharp.sharpness_score > 50
    assert not report_sharp.is_blurry

    # Completely flat/blurry image
    flat_arr = np.ones((200, 200), dtype=np.uint8) * 128
    flat_img = Image.fromarray(flat_arr)
    report_flat = assess_image_quality(flat_img)
    assert report_flat.sharpness_score < 10
    assert report_flat.is_blurry
    assert any("blurry" in w.lower() for w in report_flat.warnings)

def test_assess_image_quality_exposure():
    # Very dark image
    dark_img = Image.new("RGB", (200, 200), (10, 10, 10))
    rep_dark = assess_image_quality(dark_img)
    assert rep_dark.is_underexposed
    assert rep_dark.mean_luminance < 40

    # Very washed out image
    bright_img = Image.new("RGB", (200, 200), (250, 250, 250))
    rep_bright = assess_image_quality(bright_img)
    assert rep_bright.is_overexposed
    assert rep_bright.mean_luminance > 225

def test_clahe_preserves_shape_and_color_mode():
    img = Image.new("RGB", (150, 150), (200, 180, 100))
    clahe_img = apply_clahe(img, clip_limit=2.0)
    assert clahe_img.size == (150, 150)
    assert clahe_img.mode == "RGB"

def test_preprocess_query_image_pipeline():
    img = Image.new("RGB", (180, 120), (150, 150, 150))
    cfg = PreprocessConfig(target_size=224, enable_clahe=True, enable_denoise=True)
    out_img, quality, meta = preprocess_query_image(img, config=cfg)
    assert out_img.size == (224, 224)
    assert "aspect_pad_resize" in meta["applied_steps"]
    assert "clahe_contrast" in meta["applied_steps"]
    assert "bilateral_denoise" in meta["applied_steps"]
