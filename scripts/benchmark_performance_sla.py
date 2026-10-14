#!/usr/bin/env python3
"""PH-010: Production Release Readiness, Performance Hardening & SLA Benchmark.

Workstream 5:
- Measures sequential & concurrent search latency (p50, p90, p95, p99) under production-like conditions.
- Profiles latency across routes:
  1. Active production route (Qdrant alias 'jewellery_catalog_mobilenetv4')
  2. Canary candidate route ('jewellery_catalog_mobilenetv4_candidate_eval')
  3. Canary fallback route (missing candidate -> graceful production fallback)
- Profiles health probes (/health/live, /health/ready).
- Measures internal stage latency breakdown (preprocessing, inference, Qdrant search, heatmap).
- Evaluates statistical hard-negative lookalike pairs (N=300) with Clopper-Pearson 95% confidence bounds.
- Validates operational false-accept rate (FAR <= 1.0%) and positive coverage trade-off.

Outputs:
- output/ph010_performance_sla_benchmark.json
"""

import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import stats
from starlette.testclient import TestClient

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app, CANARY_ROUTER, LIFECYCLE_MANAGER
from app.auth import create_access_token
from engine.embeddings import MobileNetV4Encoder, get_encoder
from engine.preprocessing import PreprocessConfig, preprocess_query_image


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    alpha = 1.0 - confidence
    if k == 0:
        lower = 0.0
        upper = 1.0 - (alpha / 2.0) ** (1.0 / n)
    elif k == n:
        lower = (alpha / 2.0) ** (1.0 / n)
        upper = 1.0
    else:
        lower = float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
        upper = float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lower, upper)


def clopper_pearson_upper_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper confidence bound."""
    if n == 0:
        return 0.0
    alpha = 1.0 - confidence
    if k == 0:
        return float(1.0 - alpha ** (1.0 / n))
    elif k == n:
        return 1.0
    return float(stats.beta.ppf(1.0 - alpha, k + 1, n - k))


def compute_percentiles(durations_ms: List[float]) -> Dict[str, float]:
    arr = np.array(durations_ms, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "std": round(float(np.std(arr)), 2),
        "samples": len(arr),
    }


def load_query_images(benchmark_path: str = "data/evaluation/benchmark_v3_balanced.json", count: int = 100) -> List[Tuple[str, bytes]]:
    """Load query image bytes from benchmark manifest."""
    with open(benchmark_path, "r") as fp:
        bench_data = json.load(fp)
    queries = bench_data.get("same_sku_queries", [])
    
    loaded = []
    for q in queries:
        img_path = Path(q["query_path"])
        if img_path.exists():
            with open(img_path, "rb") as f:
                loaded.append((q["query_id"], f.read()))
    
    if not loaded:
        # Fallback to creating synthetic test image
        img = Image.new("RGB", (224, 224), color=(180, 150, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        loaded = [("fallback_dummy", buf.getvalue())]

    # Cycle to reach desired query count
    result = []
    while len(result) < count:
        result.extend(loaded)
    return result[:count]


def main():
    print("=" * 70)
    print("PH-010: Production Release Readiness & Performance SLA Hardening")
    print("=" * 70)

    # 1. Setup client and auth token with dedicated benchmark tenant (enterprise tier)
    from engine.db.session import SessionLocal
    from engine.db.models import Tenant
    db = SessionLocal()
    bench_tenant = db.query(Tenant).filter(Tenant.id == "perf_bench_tenant").first()
    if not bench_tenant:
        bench_tenant = Tenant(
            id="perf_bench_tenant",
            name="Performance Benchmark Tenant",
            tier="enterprise",
            rate_limit_rpm=100000,
            rate_limit_burst=100000,
            is_active=True
        )
        db.add(bench_tenant)
    else:
        bench_tenant.rate_limit_rpm = 100000
        bench_tenant.rate_limit_burst = 100000
    db.commit()
    db.close()

    client = TestClient(app)
    auth_token = create_access_token("perf_bench_tenant", "perf_tester", ["search:read", "ops:admin"])
    headers = {"Authorization": f"Bearer {auth_token}"}

    # 2. Load query images
    queries = load_query_images(count=100)
    print(f"Loaded {len(queries)} query workloads for benchmarking.")

    # 3. Warmup runs
    print("Running 10 warmup requests...")
    for q_id, q_bytes in queries[:10]:
        files = {"file": ("warmup.jpg", io.BytesIO(q_bytes), "image/jpeg")}
        data = {"model": "mobilenetv4", "tenant_id": "perf_bench_tenant"}
        resp = client.post("/api/search", files=files, data=data, headers=headers)
        if resp.status_code != 200:
            print(f"Warmup warning: status {resp.status_code}, {resp.text}")

    # =========================================================================
    # Workstream 5.1: Sequential Latency Profiling by Route (N=100)
    # =========================================================================
    print("\n--- 5.1 Sequential Latency Profiling (C=1, N=100 per route) ---")

    # Route A: Active Production
    CANARY_ROUTER.configure(enabled=False)
    prod_durations = []
    stage_latencies = {"preprocessing": [], "encoding": [], "search": [], "total_engine": []}

    for q_id, q_bytes in queries:
        files = {"file": ("query.jpg", io.BytesIO(q_bytes), "image/jpeg")}
        data = {"model": "mobilenetv4", "tenant_id": "perf_bench_tenant"}
        t0 = time.perf_counter()
        resp = client.post("/api/search", files=files, data=data, headers=headers)
        dur = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, f"Prod search failed: {resp.text}"
        prod_durations.append(dur)
        
        resp_json = resp.json()
        timing = resp_json.get("timing_ms", {})
        if timing:
            stage_latencies["preprocessing"].append(timing.get("preprocessing", 0.0))
            stage_latencies["encoding"].append(timing.get("encoding", 0.0))
            stage_latencies["search"].append(timing.get("search", 0.0))
            stage_latencies["total_engine"].append(timing.get("total", 0.0))

    prod_stats = compute_percentiles(prod_durations)
    print(f"Route A (Active Production): p50={prod_stats['p50']}ms, p95={prod_stats['p95']}ms, p99={prod_stats['p99']}ms")

    # Route B: Candidate Route
    CANARY_ROUTER.configure(
        enabled=True,
        percentage=100.0,
        candidate_collection="jewellery_catalog_mobilenetv4_candidate_eval",
        fallback_to_production=True
    )
    cand_durations = []
    for q_id, q_bytes in queries:
        files = {"file": ("query.jpg", io.BytesIO(q_bytes), "image/jpeg")}
        data = {"model": "mobilenetv4", "tenant_id": "perf_bench_tenant"}
        t0 = time.perf_counter()
        resp = client.post("/api/search", files=files, data=data, headers=headers)
        dur = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, f"Candidate search failed: {resp.text}"
        cand_durations.append(dur)
    cand_stats = compute_percentiles(cand_durations)
    print(f"Route B (Candidate Staging): p50={cand_stats['p50']}ms, p95={cand_stats['p95']}ms, p99={cand_stats['p99']}ms")

    # Route C: Fallback Route (missing candidate triggers graceful fallback to production)
    CANARY_ROUTER.configure(
        enabled=True,
        percentage=100.0,
        candidate_collection="non_existent_collection_fallback_test",
        fallback_to_production=True
    )
    fallback_durations = []
    for q_id, q_bytes in queries:
        files = {"file": ("query.jpg", io.BytesIO(q_bytes), "image/jpeg")}
        data = {"model": "mobilenetv4", "tenant_id": "perf_bench_tenant"}
        t0 = time.perf_counter()
        resp = client.post("/api/search", files=files, data=data, headers=headers)
        dur = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, f"Fallback search failed: {resp.text}"
        fallback_durations.append(dur)
    fallback_stats = compute_percentiles(fallback_durations)
    print(f"Route C (Canary Fallback):  p50={fallback_stats['p50']}ms, p95={fallback_stats['p95']}ms, p99={fallback_stats['p99']}ms")

    # Restore canary router to disabled for subsequent clean testing
    CANARY_ROUTER.configure(enabled=False)

    # Compute stage stats
    stage_breakdown = {k: compute_percentiles(v) for k, v in stage_latencies.items()}

    # =========================================================================
    # Workstream 5.2: Concurrency Scaling Benchmark (C=1, 2, 4, 8)
    # =========================================================================
    print("\n--- 5.2 Concurrency Scaling Benchmark ---")
    concurrency_results = {}
    for conc in [1, 2, 4, 8]:
        c_durations = []
        t_start_all = time.perf_counter()

        def do_query(item):
            _, q_bytes_item = item
            files = {"file": ("query.jpg", io.BytesIO(q_bytes_item), "image/jpeg")}
            data = {"model": "mobilenetv4", "tenant_id": "perf_bench_tenant"}
            t0 = time.perf_counter()
            r = client.post("/api/search", files=files, data=data, headers=headers)
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            return r.status_code, t_elapsed

        with ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [executor.submit(do_query, q) for q in queries]
            for f in as_completed(futures):
                status_code, dur = f.result()
                assert status_code == 200
                c_durations.append(dur)

        total_elapsed_sec = time.perf_counter() - t_start_all
        qps = round(len(queries) / total_elapsed_sec, 2)
        c_stats = compute_percentiles(c_durations)
        c_stats["throughput_qps"] = qps
        c_stats["concurrency"] = conc
        concurrency_results[f"c_{conc}"] = c_stats
        print(f"Concurrency C={conc}: Throughput={qps} QPS | p50={c_stats['p50']}ms, p95={c_stats['p95']}ms, p99={c_stats['p99']}ms")

    # =========================================================================
    # Workstream 5.3: Health & Readiness Probe Latency Profiling (N=50)
    # =========================================================================
    print("\n--- 5.3 Health Probe Latency Profiling (N=50) ---")
    live_durations = []
    ready_durations = []

    for _ in range(50):
        t0 = time.perf_counter()
        r = client.get("/health/live")
        dur = (time.perf_counter() - t0) * 1000.0
        assert r.status_code == 200
        live_durations.append(dur)

    for _ in range(50):
        t0 = time.perf_counter()
        r = client.get("/health/ready")
        dur = (time.perf_counter() - t0) * 1000.0
        assert r.status_code == 200
        ready_durations.append(dur)

    live_stats = compute_percentiles(live_durations)
    ready_stats = compute_percentiles(ready_durations)
    print(f"Liveness Probe (/health/live):   p50={live_stats['p50']}ms, p95={live_stats['p95']}ms, max={live_stats['max']}ms")
    print(f"Readiness Probe (/health/ready): p50={ready_stats['p50']}ms, p95={ready_stats['p95']}ms, max={ready_stats['max']}ms")

    # =========================================================================
    # Workstream 5.4: Hard-Negative Evaluation & Operational Threshold (N=300)
    # =========================================================================
    print("\n--- 5.4 Hard-Negative Evaluation & Statistical Confidence (N=300) ---")

    # Load 300 hard-negative lookalike pairs
    manifest_path = Path("data/ingestion_manifest.json")
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)
    records = manifest["records"]
    rec_by_id = {r["image_id"]: r for r in records}

    with open("data/index/dinov2_meta.json", "r") as fp:
        catalog_meta = json.load(fp)
    cat_ids = [m["id"] for m in catalog_meta]
    cat_categories = [m.get("category", "") for m in catalog_meta]
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}

    # Baseline DINOv3 catalog embeddings used for lookalike pair selection
    dinov3_embs = np.load("data/index/dinov3_embeddings.npy")
    raw_images = [r for r in records if r["is_original_capture"] and r["physical_identity_verification_status"] != "uncertain_identity"]
    neck_records = [r for r in raw_images if r["category"] == "necklace"]
    ring_records = [r for r in raw_images if r["category"] == "ring"]

    def find_hardest_lookalikes(query_records, max_cases=180):
        pairs = []
        used_query_groups = set()
        sorted_queries = sorted(query_records, key=lambda x: x["image_id"])
        for qr in sorted_queries:
            q_id = qr["image_id"]
            q_group = qr["identity_group_id"]
            if q_group in used_query_groups:
                continue
            if q_id not in cat_id_to_idx:
                continue
            q_vec = dinov3_embs[cat_id_to_idx[q_id]]
            sims = np.dot(dinov3_embs, q_vec)
            ranked_indices = np.argsort(sims)[::-1]
            for c_idx in ranked_indices:
                c_id = cat_ids[c_idx]
                c_rec = rec_by_id.get(c_id)
                if not c_rec or c_rec["category"] != qr["category"] or c_rec["identity_group_id"] == q_group:
                    continue
                pairs.append({
                    "query_id": q_id,
                    "query_group": q_group,
                    "candidate_id": c_id,
                    "candidate_group": c_rec["identity_group_id"],
                    "category": qr["category"]
                })
                used_query_groups.add(q_group)
                break
            if len(pairs) >= max_cases:
                break
        return pairs

    neck_pairs = find_hardest_lookalikes(neck_records, max_cases=170)
    ring_pairs = find_hardest_lookalikes(ring_records, max_cases=130)
    all_neg_pairs = neck_pairs + ring_pairs
    assert len(all_neg_pairs) == 300, f"Expected 300 lookalike pairs, got {len(all_neg_pairs)}"

    # Load MobileNetV4 catalog embeddings
    mobilenet_embs = np.load("data/index/mobilenetv4_embeddings.npy")
    neg_scores = []
    for pair in all_neg_pairs:
        q_idx = cat_id_to_idx[pair["query_id"]]
        c_idx = cat_id_to_idx[pair["candidate_id"]]
        q_vec = mobilenet_embs[q_idx]
        c_vec = mobilenet_embs[c_idx]
        score = float(np.dot(q_vec, c_vec))
        neg_scores.append(score)
    neg_scores = np.array(neg_scores, dtype=np.float64)

    max_neg = float(np.max(neg_scores))
    p95_neg = float(np.percentile(neg_scores, 95))
    median_neg = float(np.median(neg_scores))
    min_neg = float(np.min(neg_scores))

    print(f"MobileNetV4 Hard Negative Similarity: Max={max_neg:.6f}, p95={p95_neg:.6f}, Median={median_neg:.6f}, Min={min_neg:.6f}")

    # Threshold sweep
    threshold_sweep = []
    for tau in [0.90, 0.92, 0.95, 0.97, 0.974, 0.98, 0.99]:
        fa = int(np.sum(neg_scores >= tau))
        far = fa / 300.0
        ci_l, ci_u = clopper_pearson_interval(fa, 300)
        u_bound = clopper_pearson_upper_bound(fa, 300)
        threshold_sweep.append({
            "threshold": tau,
            "false_accepts": fa,
            "far_pct": round(far * 100, 3),
            "ci95_two_sided_pct": [round(ci_l * 100, 3), round(ci_u * 100, 3)],
            "one_sided_95_upper_bound_pct": round(u_bound * 100, 3),
            "sub_1pct_target_met": (u_bound <= 0.01)
        })

    # Load benchmark v3 to measure positive coverage at tau = 0.974
    with open("data/evaluation/benchmark_v3_balanced.json", "r") as fp:
        bench_data = json.load(fp)
    bench_queries = bench_data.get("same_sku_queries", [])

    # Encode query images with MobileNetV4
    encoder = MobileNetV4Encoder()
    prep_cfg = PreprocessConfig(target_size=224)
    pos_scores = []
    for bq in bench_queries:
        q_path = Path(bq["query_path"])
        if q_path.exists():
            img = Image.open(q_path).convert("RGB")
            proc, _, _ = preprocess_query_image(img, prep_cfg)
            q_vec = encoder.encode_image(proc)
            target_idx = cat_id_to_idx[bq["target_catalog_id"]]
            c_vec = mobilenet_embs[target_idx]
            pos_scores.append(float(np.dot(q_vec, c_vec)))
    pos_scores = np.array(pos_scores)

    tau_op = 0.974
    fa_op = int(np.sum(neg_scores >= tau_op))
    upper_bound_op = clopper_pearson_upper_bound(fa_op, 300)
    pos_cov_op = float(np.sum(pos_scores >= tau_op) / len(pos_scores) * 100.0) if len(pos_scores) > 0 else 0.0

    print(f"Operational Threshold tau={tau_op}: False Accepts={fa_op}/300, 95% Upper Bound={upper_bound_op*100:.3f}%, Positive Coverage={pos_cov_op:.2f}%")

    # =========================================================================
    # Workstream 5.5: SLA Compliance Verification
    # =========================================================================
    sla_compliance = {
        "sequential_p50_under_100ms": {
            "target": "< 100 ms",
            "measured": f"{prod_stats['p50']} ms",
            "passed": bool(prod_stats["p50"] < 100.0)
        },
        "sequential_p95_under_150ms": {
            "target": "< 150 ms",
            "measured": f"{prod_stats['p95']} ms",
            "passed": bool(prod_stats["p95"] < 150.0)
        },
        "concurrent_c4_p50_under_100ms": {
            "target": "< 100 ms",
            "measured": f"{concurrency_results['c_4']['p50']} ms",
            "passed": bool(concurrency_results["c_4"]["p50"] < 100.0)
        },
        "readiness_probe_p95_under_100ms": {
            "target": "< 100 ms",
            "measured": f"{ready_stats['p95']} ms",
            "passed": bool(ready_stats["p95"] < 100.0)
        },
        "hard_negative_far_upper_bound_sub_1pct": {
            "target": "<= 1.0% (at 95% confidence)",
            "measured": f"{upper_bound_op * 100:.3f}%",
            "passed": bool(upper_bound_op <= 0.01)
        }
    }

    all_sla_passed = all(v["passed"] for v in sla_compliance.values())
    print("\n--- SLA Compliance Summary ---")
    for k, v in sla_compliance.items():
        status_str = "PASS" if v["passed"] else "FAIL"
        print(f"[{status_str}] {k}: measured {v['measured']} (target {v['target']})")

    # =========================================================================
    # Output JSON Compilation
    # =========================================================================
    output_data = {
        "metadata": {
            "evaluation_stage": "PH-010: Production Release Readiness & SLA Verification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_reference": "MobileNetV4 Conv-Small FP32 ONNX (1280-dim)",
            "backend": "Qdrant Vector DB (persistent native daemon)",
            "query_count": len(queries),
            "warmup_runs": 10,
            "hard_negative_sample_size": 300,
            "concurrency_tested": [1, 2, 4, 8]
        },
        "sequential_latency_by_route": {
            "production_route": prod_stats,
            "candidate_route": cand_stats,
            "fallback_route": fallback_stats
        },
        "stage_latency_breakdown": stage_breakdown,
        "concurrency_scaling": concurrency_results,
        "health_probe_latency": {
            "liveness_probe": live_stats,
            "readiness_probe": ready_stats
        },
        "hard_negative_evaluation": {
            "sample_size": 300,
            "max_negative_similarity": round(max_neg, 6),
            "p95_negative_similarity": round(p95_neg, 6),
            "median_negative_similarity": round(median_neg, 6),
            "min_negative_similarity": round(min_neg, 6),
            "operational_threshold_tau": tau_op,
            "observed_false_accepts": fa_op,
            "one_sided_95_upper_bound_pct": round(upper_bound_op * 100, 3),
            "positive_coverage_pct": round(pos_cov_op, 2),
            "threshold_sweep": threshold_sweep
        },
        "sla_compliance_matrix": sla_compliance,
        "overall_status": "SLA_VERIFIED_PASS" if all_sla_passed else "SLA_FAILED"
    }

    out_path = Path("output/ph010_performance_sla_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(output_data, fp, indent=2)

    print(f"\nSaved performance benchmark report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
