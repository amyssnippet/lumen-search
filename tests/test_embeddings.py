"""Tests for visual embedding encoders (DINOv2, DINOv3, and CLIP)."""

import numpy as np
import pytest
from PIL import Image

from engine.embeddings import get_encoder

def test_dinov2_encoder_properties():
    enc = get_encoder("dinov2")
    assert enc.embedding_dim == 384
    assert enc.model_name == "vit_small_patch14_dinov2"
    
    img = Image.new("RGB", (224, 224), (200, 150, 100))
    vec = enc.encode_image(img)
    assert vec.shape == (384,)
    assert pytest.approx(np.linalg.norm(vec), 0.001) == 1.0

def test_dinov3_encoder_properties():
    enc = get_encoder("dinov3")
    assert enc.embedding_dim == 384
    assert enc.model_name == "dinov3"
    
    img = Image.new("RGB", (256, 256), (150, 100, 50))
    vec = enc.encode_image(img)
    assert vec.shape == (384,)
    assert pytest.approx(np.linalg.norm(vec), 0.001) == 1.0
    
    # Test patch tokens extraction
    patch_tokens = enc.get_patch_tokens(img)
    assert patch_tokens.shape == (256, 384)
    # Check unit normalization on tokens
    assert pytest.approx(np.linalg.norm(patch_tokens[0]), 0.001) == 1.0

def test_clip_encoder_properties():
    enc = get_encoder("clip")
    assert enc.embedding_dim == 512
    
    img = Image.new("RGB", (224, 224), (100, 100, 100))
    vec = enc.encode_image(img)
    assert vec.shape == (512,)
    assert pytest.approx(np.linalg.norm(vec), 0.001) == 1.0
