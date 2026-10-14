# Jewellery Visual Retrieval Engine — Backup & Disaster Recovery Runbook and Drill Report

**Milestone**: PH-011: Staging Deployment, End-to-End Verification & Operational Burn-In  
**Verification Date**: 2026-10-12  
**Status**: VERIFIED & REPEATABLE (`PASS`)  
**Targets**: PostgreSQL 16 (Port 5433), Qdrant Vector DB (Port 6335), Redis 7 (Port 6380)

---

## 1. Objectives & SLA Targets

- **Recovery Point Objective (RPO)**: <= 24 hours (nightly automated snapshots).
- **Recovery Time Objective (RTO)**: <= 15 minutes (tested restore time: < 3 seconds total).
- **Data Parity Guarantee**: 100% exact match on relational row counts, schema constraints, and nearest-neighbour search ranking.

---

## 2. Backup Procedures

### A. PostgreSQL 16 Relational Catalog
Generate a binary custom-format dump (`-Fc`) using `pg_dump`:
```bash
docker exec jewellery_staging_postgres pg_dump \
  -U staging_user \
  -d jewellery_staging \
  -Fc \
  -f /tmp/jewellery_staging.dump

docker cp jewellery_staging_postgres:/tmp/jewellery_staging.dump deploy/staging/backup/
```

### B. Qdrant Vector Collection
Trigger an atomic collection snapshot via Qdrant's REST API:
```bash
# 1. Trigger snapshot
curl -s -X POST http://127.0.0.1:6335/collections/jewellery_catalog_mobilenetv4_v1/snapshots

# 2. Download snapshot archive
curl -s http://127.0.0.1:6335/collections/jewellery_catalog_mobilenetv4_v1/snapshots/{SNAPSHOT_NAME} \
  -o deploy/staging/backup/{SNAPSHOT_NAME}
```

---

## 3. Disaster Recovery Drill Verification Results

The automated recovery drill (`scripts/drill_backup_recovery.py`) executed on 2026-10-12:

| Component | Operation | Target Entity | Duration | Measured Parity | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PostgreSQL 16** | `pg_dump` | `jewellery_staging` | 0.91s | 138 KB archive | PASS |
| **PostgreSQL 16** | `pg_restore` | `jewellery_restore_test` | 1.41s | 3 tenants, 482 products, 518 assets, 3 api_keys, 3 users (100% parity) | PASS |
| **Qdrant** | Snapshot creation | `jewellery_catalog_mobilenetv4_v1` | 0.82s | 5.7 MB archive | PASS |
| **Qdrant** | Snapshot upload/restore | `jewellery_catalog_restored_test` | 0.91s | 432 / 432 points restored (100% parity) | PASS |
| **Vector Search Parity** | 1280-dim query | Primary vs Restored | 0.05s | Top-3 IDs identical (`da07b312...`, `cfb10643...`, `c0cfc7f1...`) | PASS |

---

## 4. Operational Recovery Playbook

In the event of database failure or corrupted index data:

1. **Stop upstream application traffic** or enable maintenance mode:
   ```bash
   curl -k -X POST https://127.0.0.1:8443/api/lifecycle/maintenance -H "X-API-Key: $OPS_ADMIN_KEY"
   ```
2. **Re-create PostgreSQL database**:
   ```bash
   docker exec jewellery_staging_postgres dropdb -U staging_user --if-exists jewellery_staging
   docker exec jewellery_staging_postgres createdb -U staging_user jewellery_staging
   docker exec jewellery_staging_postgres pg_restore -U staging_user -d jewellery_staging /tmp/jewellery_staging.dump
   ```
3. **Restore Qdrant snapshot**:
   ```bash
   curl -X POST "http://127.0.0.1:6335/collections/jewellery_catalog_mobilenetv4_v1/snapshots/upload?priority=snapshot" \
     -F "snapshot=@deploy/staging/backup/qdrant_staging.snapshot"
   ```
4. **Verify Health Probes**:
   ```bash
   curl -k -s https://127.0.0.1:8443/health/ready
   ```
