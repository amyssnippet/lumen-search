#!/usr/bin/env python3
"""End-to-End Staging Verification & Smoke Test for Jewellery Visual Retrieval Engine.

Milestone PH-011: Staging Deployment, End-to-End Verification & Operational Burn-In.
Verifies the complete staging stack using httpx:
- NGINX TLS 1.3 Edge Proxy & HTTP->HTTPS Redirection.
- Health & Readiness with PostgreSQL 16, Redis 7, Qdrant, and ONNX serving.
- Authentication (JWT tokens and API keys).
- Visual search retrieval with MobileNetV4 and Qdrant alias routing.
- Rate limiting enforcement via Redis token bucket.
- Prometheus scraping & target health.
- Edge security (10MB upload limits).
"""

import json
import logging
import os
import sys
import time
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("staging_e2e_verifier")

STAGING_HTTP_URL = "http://127.0.0.1:8080"
STAGING_HTTPS_URL = "https://127.0.0.1:8443"
PROMETHEUS_URL = "http://127.0.0.1:9091"
OUTPUT_PATH = "output/ph011_staging_e2e_verification.json"

DEV_API_KEY = "jvr_default_dev_key_2026_secret"
ADMIN_EMAIL = "admin@jewellery.internal"
ADMIN_PASSWORD = "dev_password_2026"
TEST_QUERY_IMAGE = "data/evaluation/queries/rings/q_ring_var_001_ring_032.jpg"


def run_staging_verification():
    os.makedirs("output", exist_ok=True)
    report = {
        "milestone": "PH-011",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_endpoints": {
            "http": STAGING_HTTP_URL,
            "https": STAGING_HTTPS_URL,
            "prometheus": PROMETHEUS_URL,
        },
        "tests": {},
        "overall_status": "PENDING"
    }

    all_passed = True

    # 1. HTTP to HTTPS Redirection
    logger.info("--- Step 1: Testing HTTP to HTTPS 301 Redirection ---")
    try:
        with httpx.Client(follow_redirects=False, timeout=5.0) as client:
            r = client.get(f"{STAGING_HTTP_URL}/health/live")
            redirect_ok = r.status_code == 301 and "https://127.0.0.1:8443/health/live" in r.headers.get("location", "")
            report["tests"]["http_redirect"] = {
                "status": "PASS" if redirect_ok else "FAIL",
                "status_code": r.status_code,
                "location_header": r.headers.get("location")
            }
            if not redirect_ok:
                all_passed = False
    except Exception as e:
        logger.error(f"HTTP redirect test failed: {e}")
        report["tests"]["http_redirect"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # 2. TLS Handshake & HTTPS Health
    logger.info("--- Step 2: Testing TLS Handshake & HTTPS Health ---")
    try:
        with httpx.Client(verify=False, timeout=5.0) as client:
            r_live = client.get(f"{STAGING_HTTPS_URL}/health/live")
            r_ready = client.get(f"{STAGING_HTTPS_URL}/health/ready")
            live_data = r_live.json()
            ready_data = r_ready.json()

            health_ok = (
                r_live.status_code == 200
                and live_data.get("status") == "ok"
                and r_ready.status_code == 200
                and ready_data.get("status") == "ready"
                and ready_data.get("dependencies", {}).get("database", {}).get("status") == "connected"
                and ready_data.get("dependencies", {}).get("redis", {}).get("status") == "connected"
                and ready_data.get("dependencies", {}).get("qdrant", {}).get("status") == "connected"
                and ready_data.get("dependencies", {}).get("model", {}).get("status") == "available"
            )
            report["tests"]["health_and_readiness"] = {
                "status": "PASS" if health_ok else "FAIL",
                "live_status": live_data,
                "ready_status": ready_data
            }
            if not health_ok:
                all_passed = False
    except Exception as e:
        logger.error(f"Health test failed: {e}")
        report["tests"]["health_and_readiness"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # 3. Authentication & RBAC
    logger.info("--- Step 3: Testing Authentication & RBAC ---")
    try:
        with httpx.Client(verify=False, timeout=5.0) as client:
            # 3a. Unauthenticated request rejection
            r_unauth = client.get(f"{STAGING_HTTPS_URL}/api/auth/me")
            unauth_rejected = r_unauth.status_code == 401

            # 3b. Invalid credentials rejection
            r_invalid_login = client.post(
                f"{STAGING_HTTPS_URL}/api/auth/token",
                json={"email": ADMIN_EMAIL, "password": "wrong_password_xyz"}
            )
            invalid_login_rejected = r_invalid_login.status_code == 401

            # 3c. Valid JWT login
            r_valid_login = client.post(
                f"{STAGING_HTTPS_URL}/api/auth/token",
                json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
            )
            jwt_ok = r_valid_login.status_code == 200
            jwt_token = r_valid_login.json().get("access_token") if jwt_ok else None

            # 3d. Me endpoint with JWT
            me_data = None
            if jwt_token:
                r_me = client.get(
                    f"{STAGING_HTTPS_URL}/api/auth/me",
                    headers={"Authorization": f"Bearer {jwt_token}"}
                )
                if r_me.status_code == 200:
                    me_data = r_me.json()

            # 3e. API key validation
            r_api_key_valid = client.get(
                f"{STAGING_HTTPS_URL}/api/auth/me",
                headers={"X-API-Key": DEV_API_KEY}
            )
            api_key_ok = r_api_key_valid.status_code == 200

            auth_ok = (
                unauth_rejected
                and invalid_login_rejected
                and jwt_ok
                and me_data is not None
                and api_key_ok
            )
            report["tests"]["authentication"] = {
                "status": "PASS" if auth_ok else "FAIL",
                "unauthenticated_status": r_unauth.status_code,
                "invalid_login_status": r_invalid_login.status_code,
                "jwt_login_status": r_valid_login.status_code,
                "jwt_claims": me_data,
                "api_key_status": r_api_key_valid.status_code
            }
            if not auth_ok:
                all_passed = False
    except Exception as e:
        logger.error(f"Auth test failed: {e}")
        report["tests"]["authentication"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # 4. End-to-End Visual Retrieval Search
    logger.info("--- Step 4: Testing End-to-End Visual Retrieval Search ---")
    try:
        with httpx.Client(verify=False, timeout=15.0) as client:
            with open(TEST_QUERY_IMAGE, "rb") as fp:
                file_bytes = fp.read()

            files = {"file": ("query.jpg", file_bytes, "image/jpeg")}
            data = {"category": "ring", "top_k": "5"}
            headers = {"X-API-Key": DEV_API_KEY}

            r_search = client.post(
                f"{STAGING_HTTPS_URL}/api/search",
                headers=headers,
                files=files,
                data=data
            )

            search_ok = r_search.status_code == 200
            search_data = r_search.json() if search_ok else {}
            results = search_data.get("results", [])

            top_match_id = results[0].get("id") if results else None
            top_similarity = results[0].get("similarity") if results else 0.0
            expected_match = (top_match_id == "ring_032" and top_similarity > 0.90)

            retrieval_ok = (
                search_ok
                and len(results) >= 1
                and expected_match
                and search_data.get("model_name") == "mobilenetv4"
                and search_data.get("backend") == "qdrant"
            )
            report["tests"]["retrieval_search"] = {
                "status": "PASS" if retrieval_ok else "FAIL",
                "status_code": r_search.status_code,
                "top_match_id": top_match_id,
                "top_similarity": top_similarity,
                "total_results": len(results),
                "timing_ms": search_data.get("timing_ms"),
                "model_name": search_data.get("model_name"),
                "backend": search_data.get("backend"),
            }
            if not retrieval_ok:
                all_passed = False
    except Exception as e:
        logger.error(f"Search test failed: {e}")
        report["tests"]["retrieval_search"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # 5. Rate Limiting via Redis
    logger.info("--- Step 5: Testing Redis Rate Limiting & 429 Responses ---")
    try:
        artisan_key = "jvr_artisan_gems_key_2026_secret"
        statuses = []
        with httpx.Client(verify=False, timeout=5.0) as client:
            with open(TEST_QUERY_IMAGE, "rb") as fp:
                file_bytes = fp.read()
            for _ in range(16):
                files = {"file": ("query.jpg", file_bytes, "image/jpeg")}
                data = {"category": "ring", "top_k": "1"}
                r = client.post(
                    f"{STAGING_HTTPS_URL}/api/search",
                    headers={"X-API-Key": artisan_key},
                    files=files,
                    data=data
                )
                statuses.append(r.status_code)
                if r.status_code == 429:
                    break

        rate_limit_triggered = 429 in statuses
        report["tests"]["rate_limiting"] = {
            "status": "PASS" if rate_limit_triggered else "FAIL",
            "sample_statuses": statuses,
            "rate_limited_count": statuses.count(429),
            "success_count": statuses.count(200)
        }
        if not rate_limit_triggered:
            all_passed = False
    except Exception as e:
        logger.error(f"Rate limiting test failed: {e}")
        report["tests"]["rate_limiting"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # 6. Observability: Prometheus Exporter & Target Scraping
    logger.info("--- Step 6: Testing Observability & Prometheus Scraper ---")
    try:
        with httpx.Client(timeout=5.0) as client:
            r_prom = client.get(f"{PROMETHEUS_URL}/api/v1/targets")
            prom_data = r_prom.json()
            active_targets = prom_data.get("data", {}).get("activeTargets", [])
            staging_target = next((t for t in active_targets if t.get("labels", {}).get("job") == "jewellery-staging-app"), None)
            target_healthy = staging_target is not None and staging_target.get("health") == "up"

            report["tests"]["prometheus_scrape"] = {
                "status": "PASS" if target_healthy else "FAIL",
                "target_found": staging_target is not None,
                "target_health": staging_target.get("health") if staging_target else None,
                "last_scrape_duration": staging_target.get("lastScrapeDuration") if staging_target else None,
            }
            if not target_healthy:
                all_passed = False
    except Exception as e:
        logger.error(f"Prometheus test failed: {e}")
        report["tests"]["prometheus_scrape"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    # 7. Edge Security: Upload Limit Enforcement (10MB limit)
    logger.info("--- Step 7: Testing Edge Security & 10MB Body Limit ---")
    try:
        oversized_data = b"X" * (11 * 1024 * 1024)
        with httpx.Client(verify=False, timeout=10.0) as client:
            r_oversized = client.post(
                f"{STAGING_HTTPS_URL}/api/search",
                headers={"X-API-Key": DEV_API_KEY},
                files={"file": ("large.jpg", oversized_data, "image/jpeg")},
                data={"category": "ring"}
            )
            limit_enforced = r_oversized.status_code == 413
            report["tests"]["edge_upload_limit"] = {
                "status": "PASS" if limit_enforced else "FAIL",
                "status_code": r_oversized.status_code,
                "expected_code": 413
            }
            if not limit_enforced:
                all_passed = False
    except Exception as e:
        logger.error(f"Edge limit test failed: {e}")
        report["tests"]["edge_upload_limit"] = {"status": "FAIL", "error": str(e)}
        all_passed = False

    report["overall_status"] = "PASS" if all_passed else "FAIL"

    with open(OUTPUT_PATH, "w") as fp:
        json.dump(report, fp, indent=2)

    logger.info(f"Verification completed with overall status: {report['overall_status']}")
    logger.info(f"Audit artifact written to: {OUTPUT_PATH}")
    return all_passed


if __name__ == "__main__":
    success = run_staging_verification()
    sys.exit(0 if success else 1)
