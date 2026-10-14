# Next Steps & Post-Milestone Roadmap

**Milestone 24: PH-014 — Retrieval Quality, Candidate Filtering & Real-Image Validation** is **COMPLETED**.  
The interactive enterprise visual search application is fully operational and verified at **`http://localhost:3000`** with Next.js 14, isolated Docker Compose stack (`docker-compose.demo.yml`), strict category filtering, batched Qdrant search (16.84 ms p50 vector retrieval), defensible No-Match decisions, live MobileNetV4 ONNX (1280-d) inference, and **192 passing automated tests (100% green)**.

Detailed findings, benchmark results across 6 slices, and latency profiling breakdown are recorded in [`PH-014-retrieval-quality-report.md`](file:///Users/amol/Developer/search/PH-014-retrieval-quality-report.md).

Prior pre-production software engineering issues, credential remediation, performance gate reconciliation, network topology isolation, automated backup scheduling, and Alertmanager delivery drills remain verified.

The remaining roadmap forward consists exclusively of external physical data acquisition and cloud infrastructure milestones:

---

## 1. Physical Ring Studio Photoshoot Receipt & Ingestion (`PENDING EXTERNAL STUDIO ASSETS`)
- **Studio Batch Ingestion of 45 Pending Physical Ring SKUs**:
  - Ingest the remaining 45 physical ring SKUs (180 multi-angle images: `crown`, `angle_45`, `profile`, `shank`) produced according to [`PH-005-studio-acquisition-SOP.md`](file:///Users/amol/Developer/search/PH-005-studio-acquisition-SOP.md) using [`scripts/ingest_studio_batch.py`](file:///Users/amol/Developer/search/scripts/ingest_studio_batch.py).
  - Enforce 4-gate provenance validation and idempotent job tracking in `ingestion_jobs`.
  - Update `data/ingestion_manifest.json` and expand the canonical held-out test suite from $N = 30$ to $N \ge 299$ to statistically certify `GATE-02` (False Accept Rate $\le 1.0\%$ at 95% Clopper-Pearson confidence).
  - Promote new physical ring collection via `IndexLifecycleManager` blue-green alias switch in staging prior to production promotion.

---

## 2. Live Cloud Cluster Provisioning & Distributed 1M+ Benchmark (`PENDING CLOUD PROVISIONING`)
- **AWS `c6i.4xlarge` Deployment**:
  - Deploy Dockerized Qdrant 1.19.2 cluster on AWS `c6i.4xlarge` (16 vCPU, 32 GB RAM, 100 GB gp3 SSD) following [`output/ph006_cloud_deployment_manifest.json`](file:///Users/amol/Developer/search/output/ph006_cloud_deployment_manifest.json) and [`.agent/reports/PH-006-cloud-operations-guide.md`](file:///Users/amol/Developer/search/.agent/reports/PH-006-cloud-operations-guide.md).
  - Execute live progressive scale benchmarks to 1,000,000 vectors on dedicated cloud hardware with Linux kernel parameters (`vm.max_map_count=262144`, `fs.file-max=2097152`, `net.core.somaxconn=4096`).
  - Measure live concurrent QPS, segment optimizer completion, and p95/p99 search latencies under multi-tenant load to satisfy `GATE-07`.

---

## 3. Production Deployment Execution & Day-2 Operations
- **Preflight and Launch via Runbook**:
  - Follow [`docs/deployment/PRODUCTION_DEPLOYMENT_RUNBOOK.md`](file:///Users/amol/Developer/search/docs/deployment/PRODUCTION_DEPLOYMENT_RUNBOOK.md) to initialize `.env.production` from AWS Secrets Manager / Vault.
  - Run [`scripts/inject_proxy_secret.py`](file:///Users/amol/Developer/search/scripts/inject_proxy_secret.py) to dynamically mount unique `proxy_secret.inc` in production NGINX.
  - Deploy [`deploy/production/docker-compose.prod.yml`](file:///Users/amol/Developer/search/deploy/production/docker-compose.prod.yml) with 4-worker Uvicorn and NGINX edge TLS 1.3.
  - Verify datastore loopback isolation via [`scripts/verify_production_network_exposure.py`](file:///Users/amol/Developer/search/scripts/verify_production_network_exposure.py) (zero published host ports for PostgreSQL, Redis, Qdrant, backend).
  - Connect Alertmanager to enterprise PagerDuty / Slack webhooks.
  - Run recurring backup scheduler daemon via systemd or cron: `python scripts/backup_scheduler.py --interval 86400`.
