# PH-013 Milestone Report: Release Candidate Remediation & Pre-Production Gate Verification

**Milestone Identifier**: PH-013  
**Evaluation Status**: Software Engineering Pre-Production Verified | Commercial Production Launch Blocked by External Physical Assets & Cloud Benchmark  
**Date of Completion**: October 13, 2026  
**Baseline Test Suite**: 199 / 199 Passing Tests (100% Green)  

---

## 1. Executive Summary

Milestone **PH-013** advances the Jewellery Visual Retrieval Engine from the PH-012 release-candidate baseline to an independently verifiable, secure, pre-production release state. All outstanding engineering issues identified in the release audit have been systematically remediated with empirical validation:

1. **Compromised Credential Remediation**:
   - The literal reverse-proxy secret `jewellery_internal_edge_proxy_secret_2026` was permanently retired, purged from tracked files, and explicitly registered in `BANNED_COMPROMISED_CREDENTIALS`.
   - Default fallbacks were stripped from source code and NGINX configurations.
   - Dynamic injection via uncommitted, file-permission-hardened (`0600`) `proxy_secret.inc` files was established, with strict `.gitignore` and `.dockerignore` enforcement.
   - Middleware was upgraded to constant-time `hmac.compare_digest` verification.

2. **Performance Gate Reconciliation**:
   - Distinguished accepted capacity tests from token-bucket rate-limiting drops.
   - Benchmarked the 4-worker staging stack under rate-limit-aware conditions ($C=1, 4, 8$):
     - $C=1$: Client p50 = 87.47 ms (target < 100 ms), Engine p50 = 49.87 ms (SLA < 100 ms).
     - $C=4$: Client p50 = 167.83 ms, Engine p50 = 99.19 ms (SLA < 150 ms), Engine p95 = 181.50 ms.
     - $C=8$: Client p50 = 240.65 ms, Engine p50 = 148.75 ms, Engine p95 = 245.04 ms.
     - 0.0% unexpected 5xx server errors across all workloads.
   - Conducted a separate 429 burst throttling drill (30 requests at $C=8$ against a 10-burst token bucket): verified 10/10 200 OK + 20/20 429 Too Many Requests with 100% header compliance (`Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining: 0`, `X-RateLimit-Reset`).

3. **False-Accept Statistical Integrity (`GATE-02`)**:
   - Re-evaluated the canonical held-out test set ($N=30$) against the live staging cluster.
   - Recomputed the exact Clopper-Pearson 95% one-sided upper bound ($9.52\%$).
   - Kept `GATE-02` strictly marked as `NOT_VERIFIED` because proving $\text{FAR} \le 1.0\%$ mathematically requires $N \ge 299$ independent negative queries with 0 errors.

4. **Production Network Topology & Listener Isolation (`GATE-25`)**:
   - Validated that `deploy/production/docker-compose.prod.yml` publishes ZERO host ports for internal datastores (PostgreSQL, Redis, Qdrant) and application backend.
   - Confirmed NGINX TLS 1.3 is the sole internet-facing ingress container (ports 80/443).
   - Quoted all YAML environment variable expressions to prevent parsing syntax errors.

5. **Scheduled Backup Execution & Disaster Recovery (`GATE-21`)**:
   - Implemented `scripts/backup_scheduler.py` supporting recurring interval-based execution, persistent audit tracking in `data/backup_schedule_history.json`, SHA-256 checksums, and rolling retention (5 copies).
   - Demonstrated recurring execution over multiple cycles and verified non-destructive isolated restore into disposable database `jewellery_restore_test_ph013` with 100% table row parity across all domain entities.
   - Connected failure alerting to Alertmanager (`DatabaseBackupFailed`).

6. **Continuous Monitoring Alert Delivery (`GATE-22`)**:
   - Validated live notification delivery from Alertmanager 0.27.0 to a host webhook receiver for `DatabaseBackupFailed` and `TLSCertificateExpiringSoon`.

7. **Test Suite Expansion**:
   - Added `tests/test_ph013_credential_remediation.py` (11 new tests), bringing the full regression suite to **199 passed tests (100% Green)**.

---

## 2. Workstream 1: Exposed Secret Invalidation & Dynamic NGINX Include Architecture

### Threat Model & Problem Statement
In previous iterations, the internal edge proxy secret literal `jewellery_internal_edge_proxy_secret_2026` was committed to tracked NGINX configurations and hardcoded as a fallback in `app/main.py`. This exposed the secret in git history and created a risk of replay attacks or unauthorized edge bypass.

### Remediations Implemented
1. **Permanent Invalidation & Runtime Banning**:
   - Added `BANNED_COMPROMISED_CREDENTIALS` set in `app/main.py` containing `jewellery_internal_edge_proxy_secret_2026` and legacy test strings.
   - `validate_production_secrets()` immediately raises `RuntimeError` if any environment variable or proxy secret matches a banned literal.
   - `verify_reverse_proxy_middleware()` explicitly checks `proxy_secret in BANNED_COMPROMISED_CREDENTIALS` and returns HTTP 403 Forbidden even if accidentally configured.
2. **Fallback Stripping**:
   - Removed all default literal fallbacks:
     ```python
     # PH-013 Remediated: No literal fallback
     REQUIRE_REVERSE_PROXY = os.environ.get("REQUIRE_REVERSE_PROXY", "false").lower() in ("true", "1")
     INTERNAL_PROXY_SECRET = os.environ.get("INTERNAL_PROXY_SECRET", "")
     ```
3. **Timing-Safe Constant-Time Comparison**:
   - Replaced string equality with `hmac.compare_digest(proxy_secret, INTERNAL_PROXY_SECRET)`.
4. **Dynamic NGINX Include Tooling (`scripts/inject_proxy_secret.py`)**:
   - Generates cryptographically secure 256-bit entropy tokens (`sec_prx_<64-hex>`).
   - Writes uncommitted include file `proxy_secret.inc`:
     ```nginx
     # Auto-generated reverse proxy secret include — DO NOT COMMIT
     proxy_set_header X-Proxy-Secret "sec_prx_fa84f505654dd0421aea9d785672a7c6bfcb634875d3371ee2e28c7945cb81b9";
     ```
   - Sets strict user-only file permissions (`chmod 0600`).
5. **Git and Docker Leak Prevention**:
   - Added `.proxy_secret*`, `*.inc`, and `proxy_secret.inc` to `.gitignore` and `.dockerignore`.
   - Updated `scripts/audit_secrets.py` to continuously scan for hardcoded header literals in tracked `.conf` files.

### Empirical Validation (`tests/test_ph013_credential_remediation.py`)
- Direct access without secret: **HTTP 403 Forbidden** (PASS).
- Direct access with old banned literal: **HTTP 403 Forbidden** (PASS).
- Direct access with valid rotated secret: **HTTP 200 OK** (PASS).
- Ingress through NGINX HTTPS (8443): **HTTP 200 OK** (PASS).
- Leak test: verified that proxy secret values are never reflected in error traces or response bodies (PASS).

---

## 3. Workstream 2: End-to-End Performance Gate Reconciliation & Dedicated Burst Throttling

### Context & Clarification
In PH-012, running multi-threaded load ($C=4, 8, 16$) quickly consumed the standard API key's token bucket burst (30 tokens), resulting in 35%–82% HTTP 429 responses. To properly evaluate system serving capacity:
1. Normal accepted capacity was measured under rate-limit-aware conditions using an enterprise-provisioned capacity allocation (burst 120, rate 3600 RPM).
2. Rate-limiting mechanics were independently evaluated via a dedicated burst test.

### Benchmark Execution Matrix (`output/ph013_staging_performance_benchmark.json`)

#### Capacity Benchmark (MobileNetV4 ONNX, 4 Uvicorn Workers, NGINX TLS 1.3)
| Concurrency ($C$) | Total Requests ($N$) | Internal Engine p50 | Client E2E p50 | Client E2E p95 | Throughput (RPS) | 429 Rate Limits | 5xx Errors |
|---|---|---|---|---|---|---|---|
| **$C=1$** | 30 | **49.87 ms** | **87.47 ms** | 125.35 ms | 10.67 | 0 (0.0%) | **0 (0.0%)** |
| **$C=4$** | 40 | **99.19 ms** | **167.83 ms** | 316.44 ms | 21.48 | 0 (0.0%) | **0 (0.0%)** |
| **$C=8$** | 40 | **148.75 ms** | **240.65 ms** | 321.11 ms | 29.17 | 0 (0.0%) | **0 (0.0%)** |

#### Dedicated 429 Token-Bucket Burst Throttling Drill
- **Target Key**: `jvr_artisan_gems_key_2026_secret` (burst limit: 10 tokens, rate: 60 RPM).
- **Workload**: 30 rapid requests dispatched at concurrency $C=8$.
- **Observed Results**:
  - Exactly 10 requests allowed: **HTTP 200 OK**.
  - Exactly 20 requests throttled: **HTTP 429 Too Many Requests**.
  - 20/20 responses verified with standard headers:
    - `Retry-After: 1`
    - `X-RateLimit-Limit: 60`
    - `X-RateLimit-Remaining: 0`
    - `X-RateLimit-Reset: 1`
  - Detail message: `"Rate limit exceeded for tenant 'tenant_artisan_gems'. Capacity: 60 RPM. Try again in 1 seconds."`
  - Unexpected 5xx errors: **0 (0.0%)**.
  - Status: **PASS**.

#### Latency Breakdown Analysis
- **Engine Processing**: Preprocessing ($\sim 6\text{--}9\text{ ms}$), MobileNetV4 ONNX inference ($\sim 10\text{--}74\text{ ms}$ depending on CPU thread contention), Qdrant vector search ($\sim 35\text{--}75\text{ ms}$). Engine p50 satisfies all SLAs ($\le 100\text{ ms}$ at $C=1$, $\le 150\text{ ms}$ at $C=4$).
- **Client E2E Overhead**: Client round-trip latency includes NGINX TLS 1.3 termination, multipart/form-data upload transfer, HTTP request parsing, Uvicorn process dispatch, and JSON serialization. At $C=1$, client p50 is **87.47 ms**, meeting the $< 100\text{ ms}$ target. At $C=4$, client p50 is **167.83 ms**, reflecting normal CPU queuing across the 4 worker processes.

---

## 4. Workstream 3: False-Accept Rate Statistical Rigor (`GATE-02`)

### Mathematical Derivation
For sample size $N$ with $k$ observed false accepts, the exact Clopper-Pearson 95% one-sided binomial upper bound $p_U$ is defined by:
$$(1 - p_U)^N = 1 - 0.95 = 0.05 \implies p_U = 1 - (0.05)^{1/N}$$
For the held-out benchmark suite (`benchmark_v3_balanced.json`), $N = 30$. Observing $k = 0$ false accepts:
$$p_U = 1 - (0.05)^{1/30} = 1 - 0.90484 = 9.516\% \approx 9.52\%$$
To statistically prove an upper bound $p_U \le 1.0\%$ at 95% confidence with 0 errors:
$$N \ge \frac{\ln(0.05)}{\ln(0.99)} = \frac{-2.99573}{-0.0100503} \approx 298.07 \implies N \ge 299 \text{ independent queries}$$

### Live Evaluation Results (`output/ph013_hard_negative_evaluation.json`)
- Evaluated $N = 30$ canonical hard-negative queries against the live staging cluster.
- Observed false accepts at $\tau_{\text{exact}} = 0.9909$: **0 (0.0%)**.
- Observed false accepts at $\tau_{\text{high}} = 0.9900$: **0 (0.0%)**.
- Observed false accepts at $\tau_{\text{confident}} = 0.8500$: 12 (40.0%).
- 95% One-Sided Clopper-Pearson Upper Bound: **9.52%**.
- **Formal Status**: **NOT_VERIFIED**. In accordance with strict engineering integrity, `GATE-02` remains unverified until dataset expansion under SOP-PH005 provides at least $N = 299$ held-out hard-negative queries.

---

## 5. Workstream 4: Production Network Configuration & Listener Isolation (`GATE-25`)

### Topology Verification (`scripts/verify_production_network_exposure.py`)
- **Internal Services**:
  - `prod_postgres` (PostgreSQL 16 Alpine): zero host ports published. Network: `jewellery_prod_net` internal bridge.
  - `prod_redis` (Redis 7 Alpine): zero host ports published. Network: `jewellery_prod_net` internal bridge.
  - `prod_qdrant` (Qdrant Vector Database): zero host ports published. Network: `jewellery_prod_net` internal bridge.
  - `prod_app` (FastAPI 4-Worker Uvicorn): zero host ports published. Network: `jewellery_prod_net` internal bridge.
- **Edge Ingress**:
  - `prod_nginx`: Only service publishing host ports (`80:80`, `443:443`).
  - Enforces TLS 1.3, modern ciphers, HSTS, 10MB upload limits, and includes uncommitted `proxy_secret.inc`.
- **Compose Syntax Hardening**:
  - Fixed unquoted colon expressions in `deploy/production/docker-compose.prod.yml` (`POSTGRES_PASSWORD`, `JWT_SECRET_KEY`, `INTERNAL_PROXY_SECRET`) to ensure strict YAML parser compliance.
- Status: **PASS** (`output/ph013_network_exposure_audit.json`).

---

## 6. Workstream 5: Automated Recurring Backup Scheduling & DR Verification (`GATE-21`)

### Implementation & Architecture (`scripts/backup_scheduler.py`)
- **Recurring Scheduler**: Supports automated periodic execution with timestamped archives and execution logging in `data/backup_schedule_history.json`.
- **PostgreSQL**: Automated compressed dumps (`pg_dump -Fc`) copied to `deploy/staging/backup/`.
- **Qdrant**: Automated vector collection snapshots via REST API downloaded to host.
- **Integrity Checks**: Cryptographic SHA-256 hash computed and recorded for every archive.
- **Automated Retention**: Keeps the most recent 5 backups of each type, automatically pruning older archives.
- **Disaster Recovery Restore Drill**:
  - Verified SHA-256 integrity prior to restoration.
  - Restored PostgreSQL archive into isolated disposable database `jewellery_restore_test_ph013`.
  - Validated 100% row count parity across all core domain tables:
    - `tenants`: 6 / 6
    - `products`: 525 / 525
    - `assets`: 600 / 600
    - `api_keys`: 22 / 22
    - `users`: 3 / 3
  - Safely tore down disposable test database.
- **Alertmanager Integration**: Failure handlers emit `DatabaseBackupFailed` alerts to Alertmanager.
- Status: **PASS** (`output/ph013_backup_verification.json`).

---

## 7. Workstream 6: Continuous Monitoring Alert Delivery Pipeline (`GATE-22`)

### Verification Drill (`scripts/verify_alertmanager_delivery.py`)
- Tested live Alertmanager 0.27.0 running on staging port 9093.
- Injected critical alert `DatabaseBackupFailed` and warning alert `TLSCertificateExpiringSoon`.
- Validated that Alertmanager grouped alerts and dispatched the webhook payload to the host sink (`http://host.docker.internal:9999/alert-sink`).
- Status: **PASS** (`output/ph013_alertmanager_delivery.json`).

---

## 8. Release Compliance Gates Summary

| Gate ID | Domain | Requirement | Status | Evidence / Measured Value |
|---|---|---|---|---|
| **GATE-01** | Serving SLA | Sequential single-query CPU latency p50 < 100 ms | **PASS** | Engine p50 = 49.87 ms, Client p50 = 87.47 ms |
| **GATE-02** | Retrieval SLA | Hard-negative FAR $\le 1.0\%$ at 95% confidence | **NOT_VERIFIED** | $N=30$ yields 9.52% Clopper-Pearson bound ($N \ge 299$ required) |
| **GATE-03** | Serving SLA | Readiness probe (`/health/ready`) p95 < 100 ms | **PASS** | Readiness p95 < 10 ms |
| **GATE-04** | Data Quality | 50 independently verified physical ring SKUs | **NOT_VERIFIED** | 5/50 acquired; 45 pending studio acquisition (SOP-PH005) |
| **GATE-05** | Data Quality | Cross-view retrieval evaluation on real ring photos | **NOT_VERIFIED** | Pending physical ring multi-view asset delivery |
| **GATE-06** | Vector Database | Persistent vector DB with HNSW & metadata filter | **PASS** | Qdrant HNSW $M=32$, $ef=200$ with tenant filters |
| **GATE-07** | Scalability | Distributed cloud validation at 1,000,000+ vectors | **NOT_VERIFIED** | AWS cloud cluster deployment deferred |
| **GATE-08** | Security | Strict authentication & RBAC scope enforcement | **PASS** | JWT Bearer & hashed API key enforcement |
| **GATE-09** | Security | Multi-tenant anti-spoofing & data isolation | **PASS** | Tenant isolation verified across DB, Qdrant, and rate limits |
| **GATE-10** | Perimeter Security | 10MB upload limit, TLS 1.3, HSTS headers | **PASS** | Enforced at NGINX edge reverse proxy |
| **GATE-11** | Chaos Resilience | Redis partition handling & fail-secure fallback | **PASS** | Fail-secure in-process token bucket verified |
| **GATE-12** | Chaos Resilience | Canary routing graceful fallback to production | **PASS** | Fallback verified under fault injection |
| **GATE-13** | Blue-Green Lifecycle | Distinct candidate evaluation & self-eval guard | **PASS** | Guarded registration & candidate evaluation verified |
| **GATE-14** | Blue-Green Lifecycle | Atomic alias promotion, rollback, safe retirement | **PASS** | Atomic alias switching & rollback verified |
| **GATE-15** | Observability | Prometheus metrics, bounded labels, live scraping | **PASS** | Prometheus live scraping target status `up` |
| **GATE-16** | Staging Stack | Staging Environment Isolation & Provisioning | **PASS** | Isolated loopback ports & persistent staging volumes |
| **GATE-17** | Relational DB | PostgreSQL 16 Migration & ORM Dialect Check | **PASS** | PostgreSQL 16 verified across all models & migrations |
| **GATE-18** | End-to-End Verification | Full Staging Application Smoke & Pipeline Check | **PASS** | End-to-end ingest, search, and audit logging verified |
| **GATE-19** | Retrieval Benchmark | Benchmark v3 Verification on Live Staging Stack | **PASS** | 96.67% Top-1 SKU retrieval accuracy |
| **GATE-20** | Concurrency SLA | Bounded-Concurrency Benchmark & Throttling | **PASS** | $C=1$ Engine p50=49.87ms; $C=4$ Engine p50=99.19ms; burst test verified |
| **GATE-21** | Disaster Recovery | Scheduled Backups & Isolated Restore Drill | **PASS** | Recurring scheduler demonstrated; 100% table row parity |
| **GATE-22** | Incident Response | Live Alertmanager Notification Pipeline | **PASS** | Alertmanager 0.27.0 delivery verified |
| **GATE-23** | Perimeter Security | Reverse Proxy Bypass Protection & Secret Hardening | **PASS** | Missing/banned secret blocked (403); rotated token accepted |
| **GATE-24** | Secret Hardening | Production Secret Isolation & Startup Validator | **PASS** | Startup validator fails on default/banned credentials |
| **GATE-25** | Network Topology | Production Network Isolation & Listener Check | **PASS** | Zero host port exposure for datastores & backend |

**Overall Gate Counts**:
- Total Gates: **25**
- Passed: **21**
- Failed: **0**
- Not Verified (External Blockers): **4** (`GATE-02`, `GATE-04`, `GATE-05`, `GATE-07`)

---

## 9. Pre-Production Release Sign-off & Pre-Cutover Checklist

### Engineering Verdict
The Jewellery Visual Retrieval Engine software backend, vector search engine, edge reverse proxy, multi-tenancy enforcement, observability, and backup recovery pipeline are **fully verified and release-ready from a software engineering standpoint**.

Commercial production launch remains strictly gated on the two external dependencies:
1. Physical studio photography acquisition for 45 remaining ring SKUs (SOP-PH005).
2. Cloud infrastructure provisioning for the 1,000,000-vector cluster benchmark.

### Mandatory Pre-Cutover Execution Checklist
Prior to production traffic cutover:
- [ ] Run `scripts/inject_proxy_secret.py` in the production environment to mount fresh, unique `proxy_secret.inc`.
- [ ] Ensure `.env.production` is provisioned via AWS Secrets Manager / KMS and contains compliant $>32$-char secrets.
- [ ] Verify production TLS certificates are placed in `deploy/production/nginx/certs/`.
- [ ] Deploy production stack with `docker compose -f deploy/production/docker-compose.prod.yml up -d`.
- [ ] Execute `python scripts/verify_production_network_exposure.py` on the production host.
- [ ] Start recurring backup scheduler via systemd daemon or cron (`python scripts/backup_scheduler.py`).
