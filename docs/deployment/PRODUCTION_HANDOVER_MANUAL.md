# Production Handover and Operational Runbook Manual

**System**: Jewellery Visual Retrieval Engine  
**Milestone**: PH-010: Production Release Readiness, Performance Hardening & Operational Audit Closure  
**Revision**: 1.0.0-rc1  
**Last Updated**: 2026-10-12  

---

## 1. System Architecture & Topology

The Jewellery Visual Retrieval Engine is a high-performance visual search service designed for fine-grained multi-view jewellery matching.

```
                      Internet / Client Applications
                                    │
                                    ▼ (HTTPS / TLS 1.3)
             ┌──────────────────────────────────────────────┐
             │         NGINX Edge Reverse Proxy             │
             │   - TLS Termination (Strict modern ciphers)  │
             │   - 10MB Request Body Perimeter Limit        │
             │   - Media Cache & HTTP Strict Transport Sec  │
             │   - Perimeter Connection Throttling          │
             └──────────────────────┬───────────────────────┘
                                    │
                 Internal Docker Network (jewellery_net)
                                    │
                                    ▼ (HTTP / Port 8000)
             ┌──────────────────────────────────────────────┐
             │         FastAPI Search Application           │
             │   - JWT / Scoped API Key Authentication      │
             │   - Strict Anti-Spoofing & Tenant Isolation  │
             │   - MobileNetV4 Conv-Small FP32 ONNX Runtime │
             │   - Tenant-Aware Canary Index Router         │
             │   - Fail-Secure In-Process Rate Limiter      │
             │   - Prometheus Metrics Exporter (/metrics)   │
             └───────┬──────────────┬──────────────┬────────┘
                     │              │              │
                     ▼              ▼              ▼
           ┌────────────────┐ ┌──────────┐ ┌────────────────┐
           │ Qdrant Vector  │ │  Redis   │ │ Relational DB  │
           │ Database 6333  │ │ Cache &  │ │ (Postgres /    │
           │ - HNSW Index   │ │ Lua Rate │ │  SQLite)       │
           │ - Cosine Dist  │ │ Limiter  │ │ - Tenants      │
           │ - Alias Switch │ │ 6379     │ │ - API Keys     │
           └────────────────┘ └──────────┘ └────────────────┘
```

---

## 2. Quickstart & Deployment Procedure

### 2.1 Bootstrapping Containers
```bash
# 1. Clone repository and navigate to root
cd /opt/jewellery-retrieval

# 2. Configure production environment variables
cp deploy/.env.production .env
# Ensure SECRET_KEY, DATABASE_URL, and QDRANT_URL are configured

# 3. Start services via Docker Compose
docker-compose -f deploy/docker/docker-compose.yml up -d

# 4. Verify healthy startup
docker-compose -f deploy/docker/docker-compose.yml ps
```

### 2.2 Relational Database Migration & Seeding
```bash
# Seed initial tenants and relational catalog entries
python scripts/seed_relational_catalog.py
```

### 2.3 Vector Database Index Verification
```bash
# Verify active Qdrant alias and collections
curl -s http://127.0.0.1:6333/aliases | jq .
curl -s http://127.0.0.1:6333/collections | jq .
```

### 2.4 Deep Readiness Probe Check
```bash
curl -s http://localhost:8000/health/ready | jq .
# Expect HTTP 200 OK with all dependencies "connected" / "available":
# {
#   "status": "ready",
#   "dependencies": {
#     "database": {"status": "connected"},
#     "redis": {"status": "connected"},
#     "qdrant": {"status": "connected"},
#     "model": {"status": "available"}
#   }
# }
```

---

## 3. Blue-Green Index Promotion & Canary Playbook

The index lifecycle state machine enforces safe zero-downtime releases:
$$\text{BUILD} \longrightarrow \text{VALIDATE} \longrightarrow \text{STAGE} \longrightarrow \text{CANARY} \longrightarrow \text{PROMOTE} \longrightarrow \text{MONITOR} \longrightarrow \text{RETIRE}$$

### Step 1: Ingest Candidate Collection
Create a new versioned collection (e.g. `jewellery_catalog_mobilenetv4_v2`) in Qdrant with 1280 dimensions and Cosine distance.

### Step 2: Register Candidate in Lifecycle Manager
```bash
curl -X POST http://localhost:8000/api/lifecycle/candidate/register \
  -H "Authorization: Bearer $OPS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "jewellery_catalog_mobilenetv4_v2"}'
```

### Step 3: Run Automated Integrity Validation
```bash
curl -X POST http://localhost:8000/api/lifecycle/candidate/validate \
  -H "Authorization: Bearer $OPS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"min_points": 100, "run_smoke_test": true}'
```
> [!IMPORTANT]
> The candidate validation checks:
> 1. Vector dimension equals exactly 1280.
> 2. Distance metric is Cosine.
> 3. Required payload fields exist (`category`, `sku_id`, `tenant_id`, etc.).
> 4. Point count meets minimum threshold.
> 5. Retrieval smoke test executes without cross-tenant leakage.

### Step 4: Configure Canary Traffic Split
```bash
# Shift 10% of search queries to candidate collection
curl -X POST http://localhost:8000/api/lifecycle/canary/config \
  -H "Authorization: Bearer $OPS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "candidate_collection": "jewellery_catalog_mobilenetv4_v2",
    "percentage": 10,
    "fallback_to_production": true
  }'
```

### Step 5: Side-by-Side Candidate Evaluation
Run the distinct evaluation harness to compare recall and agreement:
```bash
python scripts/evaluate_index_candidate.py \
  --candidate jewellery_catalog_mobilenetv4_v2 \
  --output output/candidate_v2_evaluation.json
```
> [!NOTE]
> The evaluation script enforces a strict self-evaluation guard: if the candidate collection is identical to the active production collection, the gate decision is marked `BLOCKED_IDENTICAL_COLLECTION` (`is_promotion_approved = False`).

### Step 6: Atomic Production Promotion
Execute dry-run preview first:
```bash
curl -X POST http://localhost:8000/api/lifecycle/promote \
  -H "Authorization: Bearer $OPS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```
Execute live atomic alias promotion:
```bash
curl -X POST http://localhost:8000/api/lifecycle/promote \
  -H "Authorization: Bearer $OPS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'
```

### Step 7: Instant Reversible Rollback (Emergency Procedure)
If latency or error regressions are detected post-promotion:
```bash
curl -X POST http://localhost:8000/api/lifecycle/rollback \
  -H "Authorization: Bearer $OPS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'
```

### Step 8: Guarded Retirement of Obsolete Collections
```bash
curl -X POST http://localhost:8000/api/lifecycle/retire \
  -H "Authorization: Bearer $OPS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "jewellery_catalog_mobilenetv4_v1", "force": false}'
```
> [!CAUTION]
> Retirement strictly forbids deleting:
> 1. The collection currently mapped to the production alias.
> 2. The registered candidate collection.
> 3. The immediate rollback target (unless `force=true` is explicitly provided).

---

## 4. Incident Response & Troubleshooting Runbooks

### Incident 1: `SearchLatencyP95Regression` (> 100 ms)
- **Symptom**: Prometheus alert fires indicating end-to-end latency p95 > 100 ms.
- **Diagnostic Steps**:
  1. Inspect Grafana panel **Stage Latency p95** (`dashboards/search_service_health.json`).
  2. If `search` stage is high: Qdrant CPU saturation or missing index. Check `htop` or Qdrant container CPU.
  3. If `encoding` stage is high: Check thread contention on ONNX Runtime CPU threads.
  4. If `preprocessing` stage is high: Check if oversized images (> 4K resolution) are consuming PIL resize cycles.
- **Mitigation**:
  - Scale out FastAPI search worker replicas.
  - Tune Qdrant search parameter `ef_search` from 128 down to 64 if latency relief is urgently needed.

### Incident 2: `CanaryCandidateErrorRateHigh` (> 2%)
- **Symptom**: Candidate index errors on canary traffic exceed safety threshold.
- **Diagnostic Steps**:
  1. Check canary fallbacks counter: `jewellery_canary_routing_fallbacks_total`.
  2. Inspect candidate collection health in Qdrant: `GET /collections/{candidate_name}`.
- **Immediate Mitigation**:
  - Immediately disable canary routing to restore 100% traffic to production:
    ```bash
    curl -X POST http://localhost:8000/api/lifecycle/canary/config \
      -H "Authorization: Bearer $OPS_ADMIN_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"enabled": false}'
    ```

### Incident 3: Redis Connection Partition
- **Symptom**: Redis connectivity is lost or degraded.
- **Expected System Behavior**:
  - System automatically transitions to in-process token bucket rate limiter.
  - Fail-secure burst budget (50% of tenant limit) prevents traffic storms.
  - Readiness probe reports Redis as `"degraded"`, returning HTTP 200 to keep search online while warning orchestrators.
- **Action**:
  - Restart Redis container (`docker-compose restart redis`).
  - Upon reconnection, distributed Lua rate limiting automatically resumes without restarting the search application.

---

## 5. Security & Multi-Tenancy Governance

1. **Authentication Enforcement**:
   - Production deployments MUST set `REQUIRE_AUTH=true`.
   - Never run unauthenticated in publicly exposed perimeters.
2. **Key Rotation**:
   - API keys are hashed with SHA-256 (`key_hash`). The raw secret is only displayed once upon generation.
   - Revoke compromised keys immediately via `DELETE /api/auth/keys/{key_id}`.
3. **Multi-Tenant Anti-Spoofing**:
   - The API layer strictly rejects any payload or form field `tenant_id` that does not match the caller's authenticated identity (HTTP 403 Forbidden).
   - Qdrant queries always inject `rest.FieldCondition(key="tenant_id", match=rest.MatchValue(value=caller_tenant_id))` to guarantee zero cross-tenant vector visibility.
4. **Media Perimeter**:
   - All `/media/{category}/{filename}` requests enforce path traversal normalization and extension whitelisting.

---

## 6. Open Commercial Blockers

Before commercial general availability sign-off:
1. **Physical Ring Studio Acquisition**:
   - Complete capture and ingestion of the remaining 45 physical ring SKUs following [PH-005-studio-acquisition-SOP.md](file:///Users/amol/Developer/search/docs/PH-005-studio-acquisition-SOP.md).
2. **Cloud-Scale 1M Vector Cluster**:
   - Execute the terraform / cloud formation template on AWS to deploy multi-node Qdrant cluster with 1M vectors and measure p95 latency under live cloud network roundtrips.
