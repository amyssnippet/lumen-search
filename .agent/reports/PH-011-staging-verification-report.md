# PH-011: Staging Deployment, End-to-End Verification & Operational Burn-In

**Milestone**: PH-011  
**Project**: Jewellery Visual Retrieval Engine  
**Date**: 2026-10-12  
**Baseline Test Count**: 171 passing tests  
**New Test Count**: **182 passing tests** (+11 new integration tests; 0 failures)  
**Overall Status**: **PASSED (Software Engineering & Staging Verification)**  
**Production Release Status**: **BLOCKED PENDING EXTERNAL PHYSICAL ASSETS & CLOUD PROVISIONING**  

---

## 1. Executive Summary

Milestone **PH-011** bridges the gap between passing unit/mock tests and an operationally verified, production-like staging deployment. All core components of the Jewellery Visual Retrieval Engine—FastAPI backend, PostgreSQL 16 relational storage, Redis 7 caching and rate limiting, Qdrant vector index, NGINX TLS 1.3 edge reverse proxy, and Prometheus metrics scraper—have been deployed together and rigorously verified through automated end-to-end smoke tests, empirical retrieval benchmarking, bounded-concurrency load testing, disaster recovery drills, and incident alert simulations.

### Adherence to Milestone Scope Boundaries
- **No AWS infrastructure was provisioned** (preserved release blocker).
- **The 1-million-vector cloud cluster benchmark was not implemented** (preserved release blocker).
- **No physical ring photographs were fabricated or synthesized** (preserved release blocker: 5/50 verified physical ring SKUs under SOP-PH005).
- **The MobileNetV4 ONNX 1280-dimensional encoder was preserved without alteration**.
- **No unverified candidate indexes were promoted to production**.
- **Staging verification is strictly documented as staging readiness, not commercial general availability**.

---

## 2. Workstream Execution & Empirical Findings

### Workstream 1: Handover Audit & Status Tiering
The PH-010 audit artifacts and 171 baseline tests were inspected and confirmed 100% green. To prevent conflation of mock-backed tests with live system verification, the 4-tier verification status schema was formally instituted:
1. *Verified in Automated Tests* (e.g. SQLite unit fixtures).
2. *Verified in Local Deployment* (e.g. native Qdrant / single daemons).
3. *Verified in Staging* (e.g. containerized multi-tier stack with NGINX, PostgreSQL 16, Redis 7).
4. *Not Verified (External Blockers)* (physical ring acquisition and cloud 1M cluster).

### Workstream 2: Staging Environment Isolation & Provisioning
An isolated Docker Compose environment was established under `deploy/staging/` using dedicated loopback-only port bindings:
- **PostgreSQL 16**: `127.0.0.1:5433` (`jewellery_staging` database, `staging_user`).
- **Redis 7**: `127.0.0.1:6380`.
- **Qdrant Vector DB**: `127.0.0.1:6335` (isolated storage volume).
- **NGINX Edge Proxy**: `127.0.0.1:8080` (HTTP) and `127.0.0.1:8443` (HTTPS with TLS 1.3).
- **Prometheus Scraper**: `127.0.0.1:9091` (5s scrape and evaluation intervals).

### Workstream 3: Relational Migrations & PostgreSQL 16 Integration Tests
- **Alembic Migrations**: Upgraded to head against live PostgreSQL 16 using transactional DDL (`psycopg2-binary` driver).
- **Relational Seeding**: Populated `jewellery_staging` with 482 products, 518 assets, 3 multi-tenant accounts (`default_tenant`, `tenant_luxury_corp`, `tenant_artisan_gems`), 3 users, and 3 API keys.
- **Integration Test Suite**: Created [tests/test_ph011_postgres_integration.py](file:///Users/amol/Developer/search/tests/test_ph011_postgres_integration.py) (7 tests). Verified dialect connection, schema tables, unique constraints, multi-tenant CRUD, foreign key cascading deletion, transaction rollback semantics, and connection pool saturation resilience directly against PostgreSQL 16. All 7 tests passed.

### Workstream 4: End-to-End Application & Staging Verification
The automated verification script [scripts/verify_staging_e2e.py](file:///Users/amol/Developer/search/scripts/verify_staging_e2e.py) tested the live application through the NGINX HTTPS proxy on port 8443:
- **HTTP 301 Redirect**: `http://127.0.0.1:8080/health/live` redirects to `https://127.0.0.1:8443/health/live` (`PASS`).
- **Health & Readiness**: `/health/ready` reports all 4 dependencies connected (`database`, `redis`, `qdrant`, `model`) (`PASS`).
- **Authentication**: Unauthenticated access rejected with 401; invalid password rejected with 401; valid JWT login succeeds with RBAC scopes; master API key access succeeds (`PASS`).
- **Visual Search Retrieval**: `q_ring_var_001_ring_032.jpg` matches `ring_032` at rank 1 with 0.9329 similarity (`PASS`).
- **Redis Rate Limiting**: Artisan Gems tier exhausts burst capacity (10) and returns HTTP 429 Too Many Requests with retry headers (`PASS`).
- **Prometheus Scraping**: Active target `jewellery-staging-app` reports `health: up` (`PASS`).
- **10MB Body Limit**: Uploading an 11MB payload returns HTTP 413 Request Entity Too Large (`PASS`).
- **Audit Artifact**: [output/ph011_staging_e2e_verification.json](file:///Users/amol/Developer/search/output/ph011_staging_e2e_verification.json).

### Workstream 5: Staging Retrieval Benchmark (Benchmark v3 Balanced)
The official Benchmark v3 balanced evaluation was executed against the staging Qdrant collection `jewellery_catalog_mobilenetv4_v1` via staging alias `jewellery_catalog_mobilenetv4_staging`:
- **Query Counts**: 60 queries (34 real photographs: 30 necklaces, 4 verified rings; 26 synthetic ring variations; 30 confirmed hard negatives).
- **Retrieval Metrics**:
  - **Recall@1**: **0.8667** (86.67%, Wilson 95% CI: [0.7583, 0.9309])
  - **Recall@5**: **0.9333** (93.33%, Wilson 95% CI: [0.8407, 0.9738])
  - **Recall@20**: **0.9833** (98.33%, Wilson 95% CI: [0.9114, 0.9971])
  - **MRR**: **0.8954**
- **Threshold Calibration**:
  - Exact-Match Threshold ($\tau = 0.96$): Positive coverage **41.67%**, False-Accept Rate **0.0%** (`ZERO_FALSE_ACCEPT`).
  - Similarity Threshold ($\tau = 0.85$): Positive coverage **95.0%**, False-Accept Rate **0.0%**.
- **Segregation Audit**: Status `PASS`. Zero query images leaked into the catalog indexing.
- **Audit Artifact**: [output/ph011_staging_retrieval_benchmark.json](file:///Users/amol/Developer/search/output/ph011_staging_retrieval_benchmark.json).

### Workstream 6: Bounded-Concurrency Performance Benchmark ($C=1, 4, 8, 16$)
Executed multi-threaded search traffic over HTTPS port 8443 using [scripts/benchmark_staging_performance.py](file:///Users/amol/Developer/search/scripts/benchmark_staging_performance.py):

| Concurrency | Total Requests | Successful | 429 Throttled | Throughput (RPS) | Round-Trip p50 (ms) | Round-Trip p95 (ms) | Mean Engine Stage (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$C=1$** | 30 | 30 | 0 | 4.83 | 185.8 | 368.3 | 70.4 |
| **$C=4$** | 40 | 40 | 0 | 7.22 | 531.9 | 642.9 | 51.2 |
| **$C=8$** | 40 | 40 | 0 | 6.08 | 1225.6 | 1760.4 | 67.5 |
| **$C=16$** | 48 | 30 | 18 | 7.15 | 1749.9 | 2026.6 | 41.4 |

**Observations**: Internal search engine pipeline execution (preprocessing, ONNX inference, Qdrant search) remained stable at 41-70 ms. End-to-end round trip reflects network TLS encryption, multipart upload parsing, request correlation, and rate-limiting synchronization. At $C=16$, the default tenant burst limit (30) was reached, properly throttling 18 surplus requests with HTTP 429 without service degradation.
- **Audit Artifact**: [output/ph011_staging_performance_benchmark.json](file:///Users/amol/Developer/search/output/ph011_staging_performance_benchmark.json).

### Workstream 7: NGINX Edge Security & TLS 1.3 Verification
- Created [tests/test_ph011_edge_security_and_tls.py](file:///Users/amol/Developer/search/tests/test_ph011_edge_security_and_tls.py) (4 tests).
- Verified TLS 1.3 only (`TLS_AES_256_GCM_SHA384`), rejecting legacy TLS 1.0/1.1 handshakes.
- Verified HSTS (`max-age=31536000; includeSubDomains`), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`.
- Verified incoming `X-Correlation-ID` preservation across reverse proxy.
- Verified 10MB payload size limit enforcement (HTTP 413). All 4 tests passed.

### Workstream 8: Disaster Recovery Drill (PostgreSQL & Qdrant)
Executed an automated backup and restoration drill using [scripts/drill_backup_recovery.py](file:///Users/amol/Developer/search/scripts/drill_backup_recovery.py):
- **PostgreSQL 16**: Dumped 138 KB archive (`pg_dump -Fc` in 0.91s); restored into temporary target `jewellery_restore_test` (1.41s); verified 100% row count match across all 5 relational tables (`tenants: 3, products: 482, assets: 518, api_keys: 3, users: 3`).
- **Qdrant**: Created collection snapshot (5.7 MB in 0.82s); restored snapshot into `jewellery_catalog_restored_test` (0.91s); verified 432 / 432 points restored. Executed vector query parity test: identical top-3 nearest neighbours returned.
- **RTO & RPO**: Restores completed in < 3 seconds total (well within 15-minute RTO and 24-hour RPO SLAs).
- **Runbook**: Documented in [docs/deployment/BACKUP_AND_RECOVERY_DRILL.md](file:///Users/amol/Developer/search/docs/deployment/BACKUP_AND_RECOVERY_DRILL.md).
- **Audit Artifact**: [output/ph011_backup_recovery_drill.json](file:///Users/amol/Developer/search/output/ph011_backup_recovery_drill.json).

### Workstream 9: Observability & Incident Simulation Drill
Executed an automated operational incident drill using [scripts/simulate_staging_incident.py](file:///Users/amol/Developer/search/scripts/simulate_staging_incident.py):
- **Rule Verification**: Verified rule `StagingIncidentSimulationActive` loaded and evaluating every 5s in Prometheus.
- **Alert Lifecycle**:
  1. *Baseline*: State reported as `inactive`.
  2. *Trigger*: Incident injected via authenticated API (`/api/lifecycle/simulate-incident`).
  3. *Pending*: Transitioned to `pending` within 4.2 seconds.
  4. *Firing*: Transitioned to `firing` after 14.3 seconds (satisfying `for: 10s`).
  5. *Recovery*: Incident condition cleared; transitioned back to `inactive` (RESOLVED) within 4.1 seconds.
- **Audit Artifact**: [output/ph011_observability_incident_simulation.json](file:///Users/amol/Developer/search/output/ph011_observability_incident_simulation.json).

---

## 3. Test Suite Progression

| Test Suite | Milestone Baseline | Added in PH-011 | Final Passing | Failures |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Functional & Regression Tests | 171 | 0 | 171 | 0 |
| Staging PostgreSQL 16 Integration Suite | 0 | +7 | 7 | 0 |
| Staging NGINX Edge Security & TLS Suite | 0 | +4 | 4 | 0 |
| **Total Automated Regression Tests** | **171** | **+11** | **182** | **0** |

---

## 4. Preserved External Blockers Status

Per milestone scope requirements, the two external blockers remain strictly preserved:
1. **Physical Ring Studio Acquisition**: **5 of 50 physical SKUs acquired** under SOP-PH005. 45 physical SKUs remain pending studio scheduling and photographic delivery. Gate status remains **`NOT_VERIFIED`**.
2. **Cloud-Scale 1M+ Vector Cluster**: Automated provisioning scripts and capacity models exist, but live AWS infrastructure deployment remains deferred pending budget authorization. Gate status remains **`NOT_VERIFIED`**.

---

## 5. Summary of Created & Updated Artifacts

- `deploy/staging/docker-compose.staging.yml`: Staging Compose specification with loopback isolation.
- `deploy/staging/staging.env`: Staging environment variables.
- `deploy/staging/nginx/conf.d/jewellery_staging.conf`: NGINX reverse-proxy configuration enforcing TLS 1.3.
- `deploy/staging/nginx/certs/staging_fullchain.pem`: Staging TLS certificate.
- `deploy/staging/prometheus/prometheus.yml`: Prometheus staging scraper config.
- `tests/test_ph011_postgres_integration.py`: Live PostgreSQL 16 integration tests (7 tests).
- `tests/test_ph011_edge_security_and_tls.py`: Staging NGINX edge and TLS tests (4 tests).
- `scripts/verify_staging_e2e.py`: Automated staging smoke & E2E verification script.
- `scripts/benchmark_staging_retrieval.py`: Benchmark v3 balanced staging evaluation script.
- `scripts/benchmark_staging_performance.py`: Bounded-concurrency performance benchmark script.
- `scripts/drill_backup_recovery.py`: Disaster recovery backup & restore drill script.
- `scripts/simulate_staging_incident.py`: Incident alert lifecycle simulation script.
- `docs/deployment/STAGING_OPERATIONS_RUNBOOK.md`: Comprehensive staging operations manual.
- `docs/deployment/BACKUP_AND_RECOVERY_DRILL.md`: Backup & disaster recovery procedures and report.
- `docs/deployment/RELEASE_COMPLIANCE_MATRIX.md`: Updated 22-gate compliance matrix.
- `output/ph011_compliance_matrix.json`: Machine-readable compliance matrix JSON.
- `output/ph011_staging_e2e_verification.json`: E2E verification results.
- `output/ph011_staging_retrieval_benchmark.json`: Staging retrieval benchmark results.
- `output/ph011_staging_performance_benchmark.json`: Concurrency performance results.
- `output/ph011_backup_recovery_drill.json`: Backup & recovery drill results.
- `output/ph011_observability_incident_simulation.json`: Observability alert transition results.
