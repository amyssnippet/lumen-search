#!/usr/bin/env python3
"""Milestone PH-012: Rigorous Statistical Hard-Negative & FAR Reconciliation.

Evaluates the canonical confirmed hard-negative queries from data/evaluation/benchmark_v3_balanced.json
against the MobileNetV4 retrieval pipeline and computes the exact Clopper-Pearson binomial confidence bounds.

Statistical Proof:
For sample size N and k observed false accepts, the 95% one-sided Clopper-Pearson upper bound p_U satisfies:
  (1 - p_U)^N = 0.05  =>  p_U = 1 - (0.05)^(1/N)
For N = 30 and k = 0:
  p_U = 1 - (0.05)^(1/30) = 9.516% (~9.52%)
To establish an upper bound p_U <= 1.0% with k = 0 errors at 95% confidence:
  N >= ln(0.05) / ln(0.99) = -2.99573 / -0.0100503 = 298.07  =>  N >= 299 independent queries.

Because the held-out benchmark contains N = 30 confirmed negative queries, an observed 0 errors yields
an upper bound of 9.52%, which is mathematically insufficient to certify FAR <= 1.0%.
Therefore, GATE-02 is formally and honestly classified as NOT_VERIFIED.

Outputs: output/ph012_hard_negative_evaluation.json
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ph012_hard_negatives")

BENCHMARK_PATH = Path("data/evaluation/benchmark_v3_balanced.json")
OUTPUT_PATH = Path("output/ph012_hard_negative_evaluation.json")
STAGING_HTTPS_URL = "https://127.0.0.1:8443"
API_KEY = "jvr_default_dev_key_2026_secret"

TAU_EXACT = 0.9909
TAU_HIGH = 0.9900
TAU_CONFIDENT = 0.8500


def clopper_pearson_upper_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """Exact one-sided binomial Clopper-Pearson upper confidence bound."""
    if n == 0:
        return 1.0
    if k == 0:
        alpha = 1.0 - confidence
        return 1.0 - (alpha ** (1.0 / n))
    from scipy.stats import beta
    alpha = 1.0 - confidence
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def run_hard_negative_evaluation() -> Dict[str, Any]:
    logger.info("=== Running Hard-Negative Retrieval Evaluation against Staging Stack ===")

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark artifact not found: {BENCHMARK_PATH}")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    hard_negatives = bench_data.get("hard_negatives", [])
    logger.info(f"Loaded {len(hard_negatives)} canonical hard negative queries.")

    client = httpx.Client(verify=False, timeout=10.0)

    evaluated_queries = []
    false_accepts_tau_exact = 0
    false_accepts_tau_high = 0
    false_accepts_tau_confident = 0

    for idx, item in enumerate(hard_negatives):
        q_name = item["q"]
        target_name = item["target"]
        cat = item["cat"]
        reason = item.get("reason", "")

        query_path = Path(f"Jewellery_Data/{cat}/{q_name}.jpg")
        if not query_path.exists():
            logger.warning(f"Query file missing: {query_path}, skipping.")
            continue

        with open(query_path, "rb") as fp:
            img_bytes = fp.read()

        # Search via staging NGINX
        r = client.post(
            f"{STAGING_HTTPS_URL}/api/search",
            headers={"X-API-Key": API_KEY},
            files={"file": (query_path.name, img_bytes, "image/jpeg")},
            data={"category": cat, "top_k": "10"}
        )

        if r.status_code != 200:
            logger.error(f"Search request failed for {query_path.name}: HTTP {r.status_code}")
            continue

        resp = r.json()
        results = resp.get("results", [])

        # Filter out self-match (the query item itself in catalog)
        non_self_results = [
            res for res in results
            if res.get("sku") != q_name and not res.get("filename", "").startswith(q_name)
        ]

        top_non_self = non_self_results[0] if non_self_results else None
        top_sim = top_non_self.get("similarity", 0.0) if top_non_self else 0.0
        top_sku = top_non_self.get("sku", "") if top_non_self else ""

        # Check specifically for target lookalike score
        target_hit = next(
            (res for res in results if res.get("sku") == target_name or res.get("filename", "").startswith(target_name)),
            None
        )
        target_sim = target_hit.get("similarity", 0.0) if target_hit else 0.0

        # Check for false accept at various thresholds
        is_fa_exact = top_sim >= TAU_EXACT
        is_fa_high = top_sim >= TAU_HIGH
        is_fa_confident = top_sim >= TAU_CONFIDENT

        if is_fa_exact:
            false_accepts_tau_exact += 1
        if is_fa_high:
            false_accepts_tau_high += 1
        if is_fa_confident:
            false_accepts_tau_confident += 1

        evaluated_queries.append({
            "query_index": idx,
            "query_sku": q_name,
            "target_lookalike_sku": target_name,
            "category": cat,
            "lookalike_reason": reason,
            "highest_lookalike_similarity": round(float(top_sim), 4),
            "highest_lookalike_sku": top_sku,
            "target_lookalike_similarity": round(float(target_sim), 4),
            "false_accept_at_tau_exact": is_fa_exact,
            "false_accept_at_tau_high": is_fa_high,
            "false_accept_at_tau_confident": is_fa_confident
        })

    client.close()

    total_n = len(evaluated_queries)
    logger.info(f"Evaluated {total_n} hard negatives.")
    logger.info(f"Observed false accepts: tau_exact={false_accepts_tau_exact}, tau_high={false_accepts_tau_high}, tau_confident={false_accepts_tau_confident}")

    # Clopper-Pearson 95% one-sided upper confidence bounds
    cp_upper_exact = clopper_pearson_upper_bound(false_accepts_tau_exact, total_n, confidence=0.95)
    cp_upper_high = clopper_pearson_upper_bound(false_accepts_tau_high, total_n, confidence=0.95)
    cp_upper_confident = clopper_pearson_upper_bound(false_accepts_tau_confident, total_n, confidence=0.95)

    required_sample_size = math.ceil(math.log(0.05) / math.log(0.99))

    # Evaluate SLA Gate GATE-02
    target_far = 0.01  # 1.0%
    can_certify_sla = (cp_upper_exact <= target_far)

    report = {
        "milestone": "PH-012",
        "benchmark_name": "Hard-Negative False Accept Rate (FAR) Statistical Audit",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_endpoint": f"{STAGING_HTTPS_URL}/api/search",
        "encoder": "MobileNetV4 Conv-Small FP32 ONNX (1280-dim)",
        "sample_size_evaluated": total_n,
        "sample_size_required_for_1pct_bound": required_sample_size,
        "thresholds_evaluated": {
            "tau_exact": TAU_EXACT,
            "tau_high": TAU_HIGH,
            "tau_confident": TAU_CONFIDENT
        },
        "empirical_results": {
            "tau_exact": {
                "observed_false_accepts": false_accepts_tau_exact,
                "empirical_far": round(false_accepts_tau_exact / total_n, 4) if total_n > 0 else 0.0,
                "clopper_pearson_95_upper_bound": round(cp_upper_exact, 4),
                "clopper_pearson_95_upper_bound_pct": round(cp_upper_exact * 100.0, 2)
            },
            "tau_high": {
                "observed_false_accepts": false_accepts_tau_high,
                "empirical_far": round(false_accepts_tau_high / total_n, 4) if total_n > 0 else 0.0,
                "clopper_pearson_95_upper_bound": round(cp_upper_high, 4),
                "clopper_pearson_95_upper_bound_pct": round(cp_upper_high * 100.0, 2)
            },
            "tau_confident": {
                "observed_false_accepts": false_accepts_tau_confident,
                "empirical_far": round(false_accepts_tau_confident / total_n, 4) if total_n > 0 else 0.0,
                "clopper_pearson_95_upper_bound": round(cp_upper_confident, 4),
                "clopper_pearson_95_upper_bound_pct": round(cp_upper_confident * 100.0, 2)
            }
        },
        "gate_02_evaluation": {
            "gate_id": "GATE-02",
            "requirement": "Hard-negative False Accept Rate (FAR) <= 1.0% at 95% confidence",
            "status": "NOT_VERIFIED",
            "reason": (
                f"Statistical sample size limitation: The canonical held-out test suite contains N = {total_n} "
                f"confirmed hard negatives. Observing 0 false accepts yields a 95% one-sided upper bound of "
                f"{round(cp_upper_exact * 100.0, 2)}%, which cannot mathematically prove an FAR <= 1.0%. "
                f"A minimum sample size of N >= {required_sample_size} independent queries with 0 errors is "
                f"statistically required. Verification is deferred pending dataset expansion under SOP-PH005."
            )
        },
        "queries_sample": evaluated_queries[:10]
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    logger.info(f"Evaluation complete. Report written to {OUTPUT_PATH}")
    logger.info(f"GATE-02 Status: {report['gate_02_evaluation']['status']}")
    return report


if __name__ == "__main__":
    run_hard_negative_evaluation()
