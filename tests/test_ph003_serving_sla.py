"""Automated regression tests for PH-003: Model Serving & Latency SLA Hardening."""

import json
import os
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from PIL import Image

from engine.embeddings import (
    BaseEncoder,
    MobileNetV4Encoder,
    DINOv3ONNXEncoder,
    get_encoder
)
from engine.search import SearchEngine


def test_ph003_model_artifacts_exist():
    models_dir = Path("data/models")
    dinov3_fp32 = models_dir / "dinov3_vits16_fp32.onnx"
    dinov3_int8 = models_dir / "dinov3_vits16_int8.onnx"
    dinov3_static = models_dir / "dinov3_vits16_static_int8.onnx"
    mobilenetv4 = models_dir / "mobilenetv4_conv_small_fp32.onnx"

    assert dinov3_fp32.exists(), "DINOv3 FP32 ONNX must exist"
    assert dinov3_int8.exists(), "DINOv3 Dynamic INT8 ONNX must exist"
    assert dinov3_static.exists(), "DINOv3 Static INT8 ONNX must exist"
    assert mobilenetv4.exists(), "MobileNetV4 Conv-Small FP32 ONNX must exist"

    # Verify MobileNetV4 size is under 15 MB
    mb_size = os.path.getsize(mobilenetv4) / (1024 * 1024)
    assert mb_size < 15.0, f"Expected MobileNetV4 size < 15 MB, got {mb_size:.2f} MB"

    # Verify model integrity with onnx.checker
    onnx_mb = onnx.load(str(mobilenetv4))
    onnx.checker.check_model(onnx_mb)


def test_encoder_factory_and_invariants():
    # Test MobileNetV4 encoder
    enc_mb = get_encoder("mobilenetv4")
    assert isinstance(enc_mb, MobileNetV4Encoder)
    assert enc_mb.embedding_dim == 1280
    assert enc_mb.model_name == "mobilenetv4"

    # Test dummy image encoding and normalization invariant
    dummy_img = Image.new("RGB", (256, 256), color=(200, 180, 150))
    vec_mb = enc_mb.encode_image(dummy_img)
    assert vec_mb.shape == (1280,)
    assert np.all(np.isfinite(vec_mb))
    norm_mb = np.linalg.norm(vec_mb)
    assert 0.99 <= norm_mb <= 1.01, f"Expected unit norm, got {norm_mb}"

    # Test DINOv3 Dynamic INT8 encoder
    enc_int8 = get_encoder("dinov3_int8")
    assert isinstance(enc_int8, DINOv3ONNXEncoder)
    assert enc_int8.embedding_dim == 384
    vec_int8 = enc_int8.encode_image(dummy_img)
    assert vec_int8.shape == (384,)
    assert np.all(np.isfinite(vec_int8))
    norm_int8 = np.linalg.norm(vec_int8)
    assert 0.99 <= norm_int8 <= 1.01

    # Test DINOv3 Static INT8 encoder
    enc_static = get_encoder("dinov3_static")
    assert isinstance(enc_static, DINOv3ONNXEncoder)
    assert enc_static.embedding_dim == 384
    vec_static = enc_static.encode_image(dummy_img)
    assert vec_static.shape == (384,)
    assert np.all(np.isfinite(vec_static))


def test_mobilenetv4_serving_latency_sla():
    """Verify that MobileNetV4 satisfies the sequential latency SLA (<100ms primary, <40ms stretch)."""
    enc = get_encoder("mobilenetv4")
    dummy_img = Image.new("RGB", (256, 256), color=(220, 200, 180))

    # Warmup
    for _ in range(5):
        _ = enc.encode_image(dummy_img)

    # Measure 20 iterations
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = enc.encode_image(dummy_img)
        latencies.append((time.perf_counter() - t0) * 1000)

    p50_ms = float(np.percentile(latencies, 50))
    p95_ms = float(np.percentile(latencies, 95))

    # Assert SLA compliance
    assert p50_ms < 100.0, f"Primary SLA failed: p50={p50_ms:.2f} ms >= 100.0 ms"
    assert p50_ms < 40.0, f"Stretch SLA failed: p50={p50_ms:.2f} ms >= 40.0 ms"
    # MobileNetV4 is typically sub-10 ms on CPU
    assert p50_ms < 15.0, f"MobileNetV4 performance degraded: p50={p50_ms:.2f} ms"


def test_ph003_benchmark_artifacts_structure_and_metrics():
    backbone_path = Path("output/ph003_backbone_comparison.json")
    latency_path = Path("output/ph003_latency_benchmark.json")

    assert backbone_path.exists(), "ph003_backbone_comparison.json must exist"
    assert latency_path.exists(), "ph003_latency_benchmark.json must exist"

    with open(backbone_path, "r") as fp:
        bb_data = json.load(fp)

    assert "metadata" in bb_data
    assert "candidates" in bb_data
    assert "executive_comparisons" in bb_data

    cands = bb_data["candidates"]
    assert "mobilenetv4_fp32" in cands
    assert "dinov3_fp32" in cands
    assert "dinov3_int8_dynamic" in cands
    assert "dinov3_int8_static" in cands

    # MobileNetV4 retrieval accuracy checks
    mb_retrieval = cands["mobilenetv4_fp32"]["retrieval_metrics"]["category_filtered"]
    assert mb_retrieval["recall_at_1"] >= 0.65, f"Expected MobileNetV4 R@1 >= 65%, got {mb_retrieval['recall_at_1']}"
    assert mb_retrieval["candidate_recall_at_20"] >= 0.95

    # Check Hard Negative evaluation fields
    mb_hn = cands["mobilenetv4_fp32"]["hard_negative_evaluation"]
    assert mb_hn["sample_size"] == 300
    assert "operational_threshold_for_sub_1pct_far" in mb_hn
    assert mb_hn["operational_threshold_for_sub_1pct_far"]["observed_false_accepts"] == 0

    # Latency artifact checks
    with open(latency_path, "r") as fp:
        lat_data = json.load(fp)

    assert "sequential_profiles_b1" in lat_data
    assert "concurrency_benchmarks" in lat_data
    assert "sla_compliance_audit" in lat_data

    mb_lat = lat_data["sequential_profiles_b1"]["mobilenetv4_fp32"]
    assert mb_lat["total_ms"]["p50"] < 40.0, f"Expected total p50 < 40 ms, got {mb_lat['total_ms']['p50']}"
    assert mb_lat["forward_ms"]["p50"] < 15.0


def test_search_engine_mobilenetv4_integration():
    """Verify that high-level SearchEngine orchestrates MobileNetV4 index and queries."""
    engine = SearchEngine(model_name="mobilenetv4")
    assert engine.index.is_indexed()
    assert len(engine.index.metadata) == 432
    assert len(engine.index.embeddings) == 432

    dummy_img = Image.new("RGB", (256, 256), color=(210, 190, 160))
    res = engine.search_image(dummy_img, top_k=5)

    assert "results" in res
    assert len(res["results"]) == 5
    assert "quality" in res
    assert res["results"][0]["similarity"] > 0.0
