# Production Deployment & Operations Runbook (Milestone PH-012)

## 1. System Architecture Overview

The Jewellery Visual Retrieval Engine production topology consists of:
- **Edge Reverse Proxy**: NGINX 1.25 Alpine with TLS 1.3, strict HSTS, 10MB upload ceiling, error masking, and cryptographic internal proxy secret injection (`X-Proxy-Secret`). Sole public ingress point on ports 80/443.
- **Application Backend**: FastAPI running under Uvicorn multi-worker daemon (`--workers 4`) inside an isolated Docker bridge network. Binds exclusively to the private network.
- **Vector Database**: Qdrant running HNSW index (MobileNetV4 1280-dim, cosine distance).
- **Relational Datastore**: PostgreSQL 16 Alpine for multi-tenant identities, RBAC scopes, product metadata, and audit logs.
- **Cache & Rate Limiter**: Redis 7 Alpine for sliding-window token-bucket rate limiting.
- **Continuous Monitoring**: Prometheus 2.50.1 scraping internal metrics with Alertmanager 0.27.0 evaluating latency, error rates, certificate expiry, and backup alerts.

---

## 2. Preflight Deployment Checklist

Before deploying to production, execute the following preflight checks:

1. **Verify No Default/Staging Credentials**:
   ```bash
   python3 scripts/audit_secrets.py
   ```
   Confirm that the audit status is `PASS`. Ensure `APP_ENV=production` is declared and that `POSTGRES_PASSWORD`, `JWT_SECRET_KEY`, and `INTERNAL_PROXY_SECRET` are generated with $\ge 32$ characters of high-entropy randomness.

2. **Verify Edge TLS Certificates**:
   Ensure valid production TLS certificates are placed in `deploy/production/nginx/certs/`:
   - `production_fullchain.pem`
   - `production_privkey.pem`
   Check remaining validity:
   ```bash
   openssl x509 -in deploy/production/nginx/certs/production_fullchain.pem -noout -enddate
   ```

3. **Verify Database Migration State**:
   Run Alembic check to ensure schema migrations are up to date:
   ```bash
   alembic upgrade head
   ```

4. **Run Regression & Security Test Suite**:
   ```bash
   pytest tests/ -v
   ```
   Confirm 100% passing tests.

---

## 3. Production Deployment Execution

1. **Prepare Environment File**:
   Copy `.env.production.template` to `.env.production` (or inject secrets via AWS Secrets Manager / Vault):
   ```bash
   cp deploy/production/.env.production.template deploy/production/.env.production
   chmod 600 deploy/production/.env.production
   # Populate secrets with production credentials
   ```

2. **Launch Isolated Container Stack**:
   ```bash
   cd deploy/production
   docker compose -f docker-compose.prod.yml --env-file .env.production up -d
   ```

3. **Verify Container Health & Isolation**:
   ```bash
   docker ps --filter "name=jewellery_prod_"
   ```
   Confirm that all containers (`prod_postgres`, `prod_redis`, `prod_qdrant`, `prod_app`, `prod_nginx`) report `healthy` or `running`.
   
   **Verify Datastore Isolation**:
   Ensure ports 5432, 6379, 6333, and 8000 are NOT listed under `PORTS` with host bindings (`0.0.0.0:*` or `127.0.0.1:*`). Only NGINX ports 80 and 443 should be exposed.

4. **Verify Reverse-Proxy Perimeter Protection**:
   Execute a direct test request bypassing NGINX:
   ```bash
   # Direct curl to backend port (if attempted from within host network) must be rejected with HTTP 403:
   curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health/live
   # Expected: Connection refused (port not exposed) or 403 Forbidden
   ```
   Verify public ingress through NGINX:
   ```bash
   curl -k -s -o /dev/null -w "%{http_code}\n" https://api.jewellery-search.production.internal/health/live
   # Expected: 200 OK
   ```

---

## 4. Rollback Runbook

If regression or instability is detected during or after deployment:

1. **Vector Index Rollback**:
   Use the lifecycle rollback endpoint or script:
   ```bash
   python3 scripts/drill_canary_rollback.py
   # Or via API:
   curl -X POST https://api.jewellery-search.production.internal/api/lifecycle/rollback \
     -H "Authorization: Bearer <ADMIN_JWT>" \
     -H "Content-Type: application/json" -d '{}'
   ```
   This atomically updates the Qdrant alias to the previous verified production collection.

2. **Container Stack Rollback**:
   To revert to the previous container image release:
   ```bash
   cd deploy/production
   docker compose -f docker-compose.prod.yml down
   # Point image tags in docker-compose.prod.yml to previous release (e.g. 0.1.9)
   docker compose -f docker-compose.prod.yml up -d
   ```

3. **Relational Database Rollback**:
   Restore from the most recent verified backup archive:
   ```bash
   python3 scripts/backup_manager.py
   ```

---

## 5. Secret Rotation Procedures

1. **JWT Secret Key Rotation**:
   - Generate new 32-character secret: `openssl rand -hex 32`.
   - Update `JWT_SECRET_KEY` in production secrets manager.
   - Perform rolling restart of `prod_app` containers. Active tokens issued under the previous key will expire according to their 24h TTL; clients will re-authenticate against `/api/auth/token`.

2. **Internal Proxy Secret Rotation**:
   - Generate new proxy secret: `openssl rand -hex 32`.
   - Update both `INTERNAL_PROXY_SECRET` in `.env.production` and `proxy_set_header X-Proxy-Secret` in `jewellery_production.conf`.
   - Reload NGINX (`nginx -s reload`) and restart backend workers simultaneously.
