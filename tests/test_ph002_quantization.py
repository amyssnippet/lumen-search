"""Automated regression tests for PH-002 Workstream B: ONNX Runtime INT8 Benchmark."""

import json
import os
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest

def test_onnx_models_exist_and_validate():
    models_dir = Path("data/models")
    fp32_path = models_dir / "dinov3_vits16_fp32.onnx"
    int8_path = models_dir / "dinov3_vits16_int8.onnx"
    
    assert fp32_path.exists(), "FP32 ONNX model must exist"
    assert int8_path.exists(), "INT8 ONNX model must exist"
    
    # Check ONNX structure
    onnx_fp32 = onnx.load(str(fp32_path))
    onnx.checker.check_model(onnx_fp32)
    
    # Check sizes and compression ratio
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    compression = fp32_size / int8_size
    assert compression >= 3.0, f"Expected >= 3.0x compression, got {compression:.2f}x"

def test_onnx_inference_numerical_properties():
    int8_path = Path("data/models/dinov3_vits16_int8.onnx")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(int8_path), sess_options=opts, providers=["CPUExecutionProvider"])
    
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    output = sess.run(None, {"input": dummy_input})[0]
    
    assert output.shape == (1, 384)
    assert np.all(np.isfinite(output))
    norm = np.linalg.norm(output[0])
    assert 0.95 <= norm <= 1.05

def test_quantization_benchmark_artifact():
    artifact_path = Path("output/ph002_quantization_benchmark.json")
    assert artifact_path.exists(), "ph002_quantization_benchmark.json must exist"
    
    with open(artifact_path, "r") as fp:
        data = json.load(fp)
        
    assert "metadata" in data
    assert "model_compression" in data
    assert "quantization_failures_and_resolutions" in data
    assert "embedding_fidelity" in data
    assert "retrieval_metrics_comparison" in data
    assert "latency_profiles_sequential_b1" in data
    assert "acceptance_criteria_audit" in data
    
    # Check compression ratio
    comp = data["model_compression"]
    assert comp["compression_ratio"] >= 3.0
    
    # Check that ConvInteger failure was documented
    failures = data["quantization_failures_and_resolutions"]
    assert len(failures) >= 1
    assert "ConvInteger" in failures[0]["failure_reason"]
    
    # Check that accuracy regression is transparently recorded
    retrieval = data["retrieval_metrics_comparison"]
    pt_filt = retrieval["pytorch_fp32_cat_filtered"]
    int8_filt = retrieval["onnx_int8_cat_filtered"]
    assert pt_filt["r1"] >= int8_filt["r1"], "Regression between FP32 and INT8 documented"
    
    # Check latency breakdown fields
    latencies = data["latency_profiles_sequential_b1"]
    int8_lat = latencies["onnx_int8"]
    assert "forward_ms_p50" in int8_lat
    assert "total_ms_p50" in int8_lat
