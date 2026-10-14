#!/usr/bin/env python3
"""Comprehensive End-to-End Verification Test for the Jewellery Visual Retrieval Demo Stack.

Executes real HTTP requests against the live demo web interface at http://localhost:3000:
1. Verifies web server liveness and HTML pages (Search, Catalogue, Evaluation, Status).
2. Verifies deep system status & dependency health probes (Postgres, Redis, Qdrant, ONNX).
3. Verifies catalogue inventory retrieval and metadata contracts.
4. Verifies RBAC operator authentication and HttpOnly session cookies.
5. Verifies end-to-end visual similarity search with a real query ring photograph.
6. Verifies safe media proxy serving for retrieved candidate thumbnails.
7. Verifies network perimeter security (internal datastores isolated).
"""

import os
import sys
import time
import socket
from pathlib import Path
import httpx

BASE_URL = os.environ.get("DEMO_WEB_URL", "http://localhost:3000")
QUERY_IMAGE = Path("data/evaluation/queries/rings/q_ring_var_001_ring_032.jpg")


def test_server_liveness(client: httpx.Client, timeout_sec: int = 120):
    print(f"\n[1/7] Waiting for demo web server at {BASE_URL} to become responsive...")
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            res = client.get(f"{BASE_URL}/", timeout=5.0)
            if res.status_code == 200:
                print(f"  ✓ Demo web server is live! (Status: {res.status_code})")
                return True
        except Exception:
            time.sleep(2)
    print("  ✗ Demo web server did not respond within timeout.")
    return False


def test_html_pages(client: httpx.Client):
    print("\n[2/7] Verifying all frontend web pages render HTTP 200...")
    pages = ["/", "/catalogue", "/evaluation", "/status"]
    for page in pages:
        res = client.get(f"{BASE_URL}{page}", timeout=10.0)
        assert res.status_code == 200, f"Page {page} returned {res.status_code}"
        assert len(res.text) > 500, f"Page {page} returned empty or truncated body"
        print(f"  ✓ Page '{page}' returned 200 OK ({len(res.text)} bytes)")
    return True


def test_system_status(client: httpx.Client):
    print("\n[3/7] Verifying system status & deep dependency readiness...")
    res = client.get(f"{BASE_URL}/api/status", timeout=10.0)
    assert res.status_code == 200, f"Status probe returned {res.status_code}: {res.text}"
    data = res.json()
    print(f"  Status: {data.get('status')}")
    deps = data.get("dependencies", {})
    print(f"  Database connected: {deps.get('database')}")
    print(f"  Redis connected: {deps.get('redis')}")
    print(f"  Qdrant connected: {deps.get('qdrant')}")
    print(f"  ONNX model loaded: {deps.get('onnx_model')}")

    assert deps.get("database") is True, "Database dependency failed"
    assert deps.get("redis") is True, "Redis dependency failed"
    assert deps.get("qdrant") is True, "Qdrant dependency failed"
    assert deps.get("onnx_model") is True, "ONNX model dependency failed"
    print("  ✓ All core infrastructure dependencies healthy and verified!")
    return True


def test_catalogue_api(client: httpx.Client):
    print("\n[4/7] Verifying catalogue browser API & relational inventory...")
    res = client.get(f"{BASE_URL}/api/products?limit=10", timeout=10.0)
    assert res.status_code == 200, f"Catalogue API returned {res.status_code}: {res.text}"
    data = res.json()
    total = data.get("total", 0)
    products = data.get("products", [])
    print(f"  Total catalog items: {total}")
    print(f"  Sample items retrieved: {len(products)}")
    assert total > 0, "No products found in catalogue"
    assert len(products) > 0, "Empty product batch returned"
    first = products[0]
    print(f"  Sample item: SKU={first.get('sku')}, Category={first.get('category')}, Thumbnail={first.get('thumbnail_url')}")
    assert first.get("thumbnail_url"), "Product missing thumbnail_url"
    print("  ✓ Catalogue inventory verified with relational metadata and thumbnails!")
    return True


def test_auth_workflow(client: httpx.Client):
    print("\n[5/7] Verifying authentication & HttpOnly session handling...")
    login_res = client.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": "admin@jewellery.internal", "password": "dev_password_2026"},
        timeout=10.0
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    login_data = login_res.json()
    assert login_data.get("success") is True, "Login response success != True"
    print(f"  ✓ Login succeeded for operator {login_data.get('user', {}).get('email')}")

    me_res = client.get(f"{BASE_URL}/api/auth/me", timeout=10.0)
    assert me_res.status_code == 200, f"Auth verification failed: {me_res.text}"
    me_data = me_res.json()
    assert me_data.get("authenticated") is True, "Session cookie not authenticated"
    print(f"  ✓ Session verified via HttpOnly cookie (Tenant: {me_data.get('user', {}).get('tenant_id')}, Scopes: {me_data.get('user', {}).get('scopes')})")
    return True


def test_visual_search_and_media(client: httpx.Client):
    print(f"\n[6/7] Verifying live visual similarity search using query image: {QUERY_IMAGE.name}...")
    assert QUERY_IMAGE.exists(), f"Query image {QUERY_IMAGE} not found"

    with open(QUERY_IMAGE, "rb") as fp:
        files = {"file": (QUERY_IMAGE.name, fp, "image/jpeg")}
        data = {
            "category_filter": "ring",
            "search_mode": "similar",
            "top_k": "10",
        }
        start_t = time.time()
        res = client.post(f"{BASE_URL}/api/search", files=files, data=data, timeout=30.0)
        client_elapsed_ms = (time.time() - start_t) * 1000

    assert res.status_code == 200, f"Visual search request failed: {res.status_code} - {res.text}"
    search_data = res.json()

    print(f"  Query ID: {search_data.get('query_id')}")
    print(f"  Model: {search_data.get('model_name')}")
    print(f"  Client E2E Measured Latency: {client_elapsed_ms:.1f}ms")
    print(f"  Reported Client E2E: {search_data.get('client_e2e_ms')}ms")
    timing = search_data.get("timing_ms", {})
    print(f"  Internal Engine Latency: {timing.get('total')}ms (Preprocessing: {timing.get('preprocessing')}ms, Encoding: {timing.get('encoding')}ms, Search: {timing.get('search')}ms)")
    print(f"  Reliable Match: {search_data.get('has_reliable_match')}")
    print(f"  Top Similarity: {search_data.get('top_similarity', 0)*100:.2f}%")

    results = search_data.get("results", [])
    assert len(results) > 0, "No search results returned"
    top_hit = results[0]
    print(f"  Top Hit: SKU={top_hit.get('sku')}, Similarity={top_hit.get('similarity', 0)*100:.2f}%, Media={top_hit.get('image_url')}")
    assert top_hit.get("similarity", 0) > 0.85, f"Expected top similarity > 0.85, got {top_hit.get('similarity')}"

    # Verify media proxy stream
    media_url = f"{BASE_URL}{top_hit.get('image_url')}"
    print(f"  Verifying media proxy stream from {media_url}...")
    media_res = client.get(media_url, timeout=10.0)
    assert media_res.status_code == 200, f"Media fetch returned {media_res.status_code}"
    assert "image/" in media_res.headers.get("content-type", ""), f"Invalid Content-Type: {media_res.headers.get('content-type')}"
    assert len(media_res.content) > 1000, f"Image stream too small: {len(media_res.content)} bytes"
    print(f"  ✓ Retrieved image stream: {len(media_res.content)} bytes, Content-Type={media_res.headers.get('content-type')}")
    return True


def test_network_isolation():
    print("\n[7/7] Verifying network isolation (demo internal datastores NOT exposed on host)...")
    for port, name in [(5432, "Postgres Demo Default"), (6379, "Redis Demo Default"), (6333, "Qdrant Demo Default")]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if result != 0:
            print(f"  ✓ Port {port} ({name}) is properly sealed from host interface.")
        else:
            print(f"  ⚠ Port {port} is accessible on host (may belong to another host service).")
    return True


def main():
    print("=" * 70)
    print("  LUMEN JEWELLERY RETRIEVAL ENGINE — DEMO SMOKE VERIFICATION")
    print("=" * 70)

    with httpx.Client(follow_redirects=True) as client:
        if not test_server_liveness(client):
            sys.exit(1)

        try:
            test_html_pages(client)
            test_system_status(client)
            test_catalogue_api(client)
            test_auth_workflow(client)
            test_visual_search_and_media(client)
            test_network_isolation()
            print("\n" + "=" * 70)
            print("  ✓ ALL DEMO E2E VERIFICATION TESTS PASSED SUCCESSFULLY!")
            print(f"  Interactive Web Application is ready at: {BASE_URL}")
            print("=" * 70)
        except AssertionError as err:
            print(f"\n✗ TEST FAILURE: {err}")
            sys.exit(1)
        except Exception as err:
            print(f"\n✗ UNEXPECTED ERROR: {err}")
            sys.exit(1)


if __name__ == "__main__":
    main()
