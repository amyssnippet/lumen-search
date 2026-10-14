#!/usr/bin/env python3
"""Profile Live Docker Demo Latency across granular timing boundaries (Workstream 6).

Distinguishes:
1. Browser / Client E2E (HTTP round-trip to Next.js on port 3000)
2. Next.js server-side proxy overhead
3. Backend Request Handling
4. Image Preprocessing (EXIF, skin detection, aspect-pad)
5. ONNX Embedding (MobileNetV4 1280-dim FP32)
6. Qdrant HNSW vector search (batched query_batch_points)
7. Candidate formatting, localization & JSON delivery

Runs:
- 3 warm-up requests
- 20 benchmark requests
- Reports min, max, mean, p50, p95
"""

import io
import json
import statistics
import time
from pathlib import Path
import urllib.request
import urllib.parse

REPO_ROOT = Path(__file__).resolve().parent.parent

def run_profiling(sample_count: int = 20):
    url = "http://localhost:3000/api/search"
    query_file = REPO_ROOT / "Jewellery_Data" / "ring" / "ring-downloaded.png"
    if not query_file.exists():
        query_file = REPO_ROOT / "Jewellery_Data" / "ring" / "ring_001.jpg"

    with open(query_file, "rb") as f:
        file_bytes = f.read()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body_parts = []
    # File part
    body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"query.png\"\r\nContent-Type: image/png\r\n\r\n".encode("utf-8") + file_bytes + b"\r\n")
    # Category filter
    body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"category_filter\"\r\n\r\nring\r\n".encode("utf-8"))
    # Robust mode
    body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"robust_mode\"\r\n\r\ntrue\r\n".encode("utf-8"))
    # End boundary
    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    payload = b"".join(body_parts)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(payload))
    }

    def execute_request():
        t0 = time.time()
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            e2e_ms = (time.time() - t0) * 1000
            data = json.loads(raw.decode("utf-8"))
            return e2e_ms, data

    print("Warming up live Docker container stack (3 queries)...")
    for _ in range(3):
        execute_request()

    print(f"Executing {sample_count} live benchmark queries against {url}...")
    e2e_latencies = []
    engine_latencies = []
    prep_latencies = []
    enc_latencies = []
    search_latencies = []
    proxy_overhead_latencies = []

    for i in range(sample_count):
        e2e_ms, data = execute_request()
        timing = data.get("timing_ms", {})
        engine_total = timing.get("total", 0.0)
        prep = timing.get("preprocessing", 0.0)
        enc = timing.get("encoding", 0.0)
        q_search = timing.get("search", 0.0)
        proxy_overhead = max(0.0, e2e_ms - engine_total)

        e2e_latencies.append(e2e_ms)
        engine_latencies.append(engine_total)
        prep_latencies.append(prep)
        enc_latencies.append(enc)
        search_latencies.append(q_search)
        proxy_overhead_latencies.append(proxy_overhead)

    def stats(arr):
        return {
            "min": round(min(arr), 2),
            "max": round(max(arr), 2),
            "mean": round(statistics.mean(arr), 2),
            "p50": round(statistics.median(arr), 2),
            "p95": round(statistics.quantiles(arr, n=20)[18], 2) if len(arr) >= 20 else round(max(arr), 2)
        }

    results = {
        "environment": {
            "machine": "mac (Apple Silicon)",
            "service_stack": "Docker Compose (Next.js + FastAPI + Qdrant 1.12.1 + Redis 7 + Postgres 16)",
            "index_collection": "jewellery_catalog_mobilenetv4_v1 (alias: jewellery_catalog_mobilenetv4_staging)",
            "index_size_points": 432,
            "embedding_dimension": 1280,
            "sample_count": sample_count,
            "concurrency": 1
        },
        "breakdown": {
            "browser_e2e_ms": stats(e2e_latencies),
            "nextjs_proxy_overhead_ms": stats(proxy_overhead_latencies),
            "engine_core_total_ms": stats(engine_latencies),
            "image_preprocessing_ms": stats(prep_latencies),
            "onnx_inference_ms": stats(enc_latencies),
            "qdrant_hnsw_batch_search_ms": stats(search_latencies)
        }
    }

    out_file = REPO_ROOT / "output" / "live_latency_profile.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("PH-014 LIVE DOCKER DEMO LATENCY PROFILING REPORT")
    print("="*80)
    print(f"Machine: {results['environment']['machine']}")
    print(f"Stack: {results['environment']['service_stack']}")
    print(f"Samples: {sample_count} queries | Concurrency: 1 | Indexed vectors: 432 (1280-dim)")
    print("-" * 80)
    print(f"{'Timing Boundary Stage':<35} | {'Min (ms)':<9} | {'Mean (ms)':<9} | {'p50 (ms)':<9} | {'p95 (ms)':<9}")
    print("-" * 80)
    for stage_name, s in results["breakdown"].items():
        clean_name = stage_name.replace("_ms", "").replace("_", " ").title()
        print(f"{clean_name:<35} | {s['min']:<9} | {s['mean']:<9} | {s['p50']:<9} | {s['p95']:<9}")
    print("="*80)
    print(f"Detailed output saved to: {out_file}\n")

if __name__ == "__main__":
    run_profiling(sample_count=20)
