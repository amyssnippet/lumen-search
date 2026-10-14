# Jewellery Visual Retrieval Engine — Staging Operations Runbook

**Milestone**: PH-011: Staging Deployment, End-to-End Verification & Operational Burn-In  
**Target Environment**: Staging Isolated Docker Stack (`deploy/staging/`)  
**Domain**: `staging.jewellery-search.local` (Local Loopback)  
**Last Updated**: 2026-10-12

---

## 1. Environment Architecture & Port Allocations

The staging environment operates with complete isolation from host native daemons using non-colliding loopback port bindings:

| Service | Docker Container | Host Loopback Port | Internal Port | Health Probe |
| :--- | :--- | :--- | :--- | :--- |
| **NGINX Proxy (HTTP)** | `jewellery_staging_nginx` | `127.0.0.1:8080` | 80 | `http://127.0.0.1:8080/` (301 redirect) |
| **NGINX Proxy (HTTPS)**| `jewellery_staging_nginx` | `127.0.0.1:8443` | 443 | `https://127.0.0.1:8443/health/live` |
| **FastAPI Backend** | Host Daemon / Staging App | `0.0.0.0:8001` | 8001 | `http://127.0.0.1:8001/health/ready` |
| **PostgreSQL 16** | `jewellery_staging_postgres`| `127.0.0.1:5433` | 5432 | `pg_isready -h 127.0.0.1 -p 5433` |
| **Redis 7** | `jewellery_staging_redis` | `127.0.0.1:6380` | 6379 | `redis-cli -h 127.0.0.1 -p 6380 ping` |
| **Qdrant Vector DB** | `jewellery_staging_qdrant` | `127.0.0.1:6335` | 6333 | `curl -f http://127.0.0.1:6335/readyz` |
| **Prometheus** | `jewellery_staging_prometheus`| `127.0.0.1:9091` | 9090 | `http://127.0.0.1:9091/-/healthy` |

---

## 2. Starting & Stopping the Staging Stack

### Start Staging Infrastructure:
```bash
docker compose -f deploy/staging/docker-compose.staging.yml up -d
```

### Start Staging Application Backend:
```bash
PYTHONPATH=. \
DATABASE_URL=postgresql://staging_user:staging_secret_pass_2026@127.0.0.1:5433/jewellery_staging \
REDIS_URL=redis://127.0.0.1:6380/0 \
QDRANT_HOST=127.0.0.1 \
QDRANT_PORT=6335 \
QDRANT_PRODUCTION_ALIAS=jewellery_catalog_mobilenetv4_staging \
REQUIRE_AUTH=true \
JWT_SECRET_KEY=staging_super_secret_jwt_key_2026_secure \
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Stop Infrastructure:
```bash
docker compose -f deploy/staging/docker-compose.staging.yml down
```

---

## 3. Database Migration & Catalog Seeding

### Apply Alembic Migrations to Staging PostgreSQL:
```bash
DATABASE_URL=postgresql://staging_user:staging_secret_pass_2026@127.0.0.1:5433/jewellery_staging \
.venv/bin/alembic upgrade head
```

### Seed Relational Entities (Tenants, Users, API Keys, Products, Assets):
```bash
PYTHONPATH=. \
DATABASE_URL=postgresql://staging_user:staging_secret_pass_2026@127.0.0.1:5433/jewellery_staging \
.venv/bin/python scripts/seed_relational_catalog.py
```

### Populate Staging Qdrant Collection & Alias:
```bash
PYTHONPATH=. \
.venv/bin/python scripts/rebuild_qdrant_index.py --model mobilenetv4 --port 6335

# Set alias
PYTHONPATH=. .venv/bin/python -c "
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
client = QdrantClient(host='127.0.0.1', port=6335)
client.update_collection_aliases(change_aliases_operations=[
    rest.CreateAliasOperation(create_alias=rest.CreateAlias(collection_name='jewellery_catalog_mobilenetv4_v1', alias_name='jewellery_catalog_mobilenetv4_staging'))
])
"
```

---

## 4. Operational Health & Verification Procedures

### Check Deep Health & Readiness:
```bash
curl -k -s https://127.0.0.1:8443/health/ready | python3 -m json.tool
```
Expected output:
```json
{
  "status": "ready",
  "dependencies": {
    "database": {"status": "connected"},
    "redis": {"status": "connected"},
    "qdrant": {"status": "connected"},
    "model": {"status": "available"}
  }
}
```

### Execute Automated Staging Smoke & E2E Verification:
```bash
.venv/bin/python scripts/verify_staging_e2e.py
```

### Execute Retrieval Benchmark Verification:
```bash
.venv/bin/python scripts/benchmark_staging_retrieval.py
```

### Execute Bounded-Concurrency Performance Benchmark:
```bash
.venv/bin/python scripts/benchmark_staging_performance.py
```

### Execute Backup & Disaster Recovery Drill:
```bash
.venv/bin/python scripts/drill_backup_recovery.py
```

### Execute Prometheus Observability & Alert Simulation Drill:
```bash
.venv/bin/python scripts/simulate_staging_incident.py
```

---

## 5. Security Controls & Credentials Reference

- **Disposable Staging TLS Certificate**: `deploy/staging/nginx/certs/staging_fullchain.pem`
- **Default Master Dev Key (Full scopes)**: `jvr_default_dev_key_2026_secret`
- **Luxury Corp Production Key**: `jvr_luxury_corp_key_2026_secret`
- **Artisan Gems Limited Key (Burst=10, RPM=60)**: `jvr_artisan_gems_key_2026_secret`
- **Default Admin User**: `admin@jewellery.internal` / `dev_password_2026`
- **Default Ops User**: `ops@jewellery.internal` / `dev_password_2026`

---

## 6. Incident Response & Troubleshooting Playbook

### Issue: NGINX returns 502 Bad Gateway
1. Verify backend daemon is listening on `0.0.0.0:8001`: `lsof -i :8001`.
2. Inspect NGINX container error log: `docker logs jewellery_staging_nginx`.
3. Check network reachability from container to host: `docker exec jewellery_staging_nginx nc -zv host.docker.internal 8001`.

### Issue: 429 Too Many Requests Spike
1. Inspect Prometheus rate limit counters: query `sum(rate(jewellery_ratelimit_events_total[1m])) by (action)`.
2. Verify Redis container connectivity: `redis-cli -h 127.0.0.1 -p 6380 ping`.
3. Check tenant tier burst configuration: Free tier burst is 10; enterprise is 100.

### Issue: Candidate Canary Failure or Alert Firing
1. Check Prometheus alert rules: `http://127.0.0.1:9091/api/v1/alerts`.
2. Trigger immediate production rollback:
   ```bash
   curl -k -X POST https://127.0.0.1:8443/api/lifecycle/rollback \
     -H "X-API-Key: $OPS_ADMIN_KEY" \
     -H "Content-Type: application/json" \
     -d '{"reason": "Automated incident response rollback"}'
   ```
