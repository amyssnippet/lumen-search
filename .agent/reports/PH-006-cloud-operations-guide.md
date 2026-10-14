# Qdrant Cloud Deployment, Operations, and Disaster Recovery Guide (PH-006)

**Document Version**: 1.0.0  
**Target Environment**: AWS EC2 `c6i.4xlarge` (16 vCPU, 32 GB RAM, 100 GB gp3 SSD)  
**Database**: Qdrant 1.19.2 (Pinned Official Container Image)  
**Security Tier**: VPC Private Application Subnet with IAM Role & Security Group Isolation  

---

## 1. Cloud Architecture Overview

To support a 1,000,000-vector production catalog at 1280 dimensions (MobileNetV4 Conv-Small FP32) with sub-2 ms p95 search latency and high concurrency, the vector indexing layer is deployed on dedicated, compute-optimized cloud instances.

```
+---------------------------------------------------------------------------------------------------+
|                                  AWS VPC (10.0.0.0/16) ARCHITECTURE                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   +--------------------------+          +-----------------------------------------------------+   |
|   | Public Subnet (10.0.0.0) |          | Private Subnet (10.0.1.0/24)                         |   |
|   |                          |          |                                                     |   |
|   |  AWS ALB (HTTPS 443)     | -------> |  FastAPI Web Cluster (Port 8000)                    |   |
|   |  ACM TLS Termination     |          |  (MobileNetV4 ONNX Preprocessing & Feature Extraction)|  |
|   +--------------------------+          +--------------------------+--------------------------+   |
|                                                                    |                              |
|                                            Internal VPC Traffic    | gRPC: 6334 / REST: 6333      |
|                                            (Zero Public Ingress)   v                              |
|                                         +-----------------------------------------------------+   |
|                                         | Dedicated Vector DB: AWS c6i.4xlarge                 |   |
|                                         | - 16 vCPUs (Intel Xeon Ice Lake AVX-512)            |   |
|                                         | - 32 GB RAM (DDR4-3200)                             |   |
|                                         | - Docker: qdrant/qdrant:v1.19.2                     |   |
|                                         | - Attached gp3 SSD: 100 GB (3000 IOPS, 125 MB/s)    |   |
|                                         +--------------------------+--------------------------+   |
|                                                                    |                              |
|                                            VPC Gateway Endpoint    | Daily Snapshots              |
|                                                                    v                              |
|                                         +-----------------------------------------------------+   |
|                                         | AWS S3: s3://jewellery-retrieval-backups/           |   |
|                                         +-----------------------------------------------------+   |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Host Provisioning & Kernel Optimization

### 2.1 Sysctl & OS Tuning (Ubuntu 22.04 LTS)
Qdrant relies on memory-mapped files (`mmap`) for vector and HNSW graph indexing. To prevent memory allocation failures and socket bottlenecks:

```bash
# Append to /etc/sysctl.d/99-qdrant.conf
sudo tee /etc/sysctl.d/99-qdrant.conf <<EOF
# Increase max virtual memory map areas for mmap
vm.max_map_count = 262144

# Increase file descriptor ceiling
fs.file-max = 2097152

# Network connection queue depth
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 4096

# Memory overcommit handling
vm.overcommit_memory = 1
EOF

# Apply sysctl settings immediately
sudo sysctl --system

# Set user limits in /etc/security/limits.d/qdrant.conf
sudo tee /etc/security/limits.d/qdrant.conf <<EOF
* soft nofile 1048576
* hard nofile 1048576
* soft memlock unlimited
* hard memlock unlimited
EOF
```

---

## 3. Containerized Deployment (`docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.19.2
    container_name: qdrant-production
    restart: unless-stopped
    ports:
      - "10.0.1.50:6333:6333"   # REST API bound strictly to private VPC IP
      - "10.0.1.50:6334:6334"   # gRPC endpoint bound strictly to private VPC IP
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__SERVICE__ENABLE_CORS=false
      - QDRANT__SERVICE__TELEMETRY_DISABLED=true
      - QDRANT__STORAGE__ON_DISK_PAYLOAD=true
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
    volumes:
      - /var/lib/qdrant/data:/qdrant/storage:rw
      - /var/lib/qdrant/snapshots:/qdrant/snapshots:rw
    ulimits:
      nofile:
        soft: 1048576
        hard: 1048576
      memlock:
        soft: -1
        hard: -1
    deploy:
      resources:
        limits:
          cpus: '14.0'
          memory: 28G
        reservations:
          cpus: '8.0'
          memory: 16G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 20s
```

---

## 4. Blue-Green Staging & Promotion Workflow

To prevent query downtime or catalog corruption during bulk ingestion of new physical ring studio batches or CAD catalog updates:

### Step 1: Create Versioned Staging Collection
```bash
curl -X PUT "http://10.0.1.50:6333/collections/jewellery_catalog_mobilenetv4_staging" \
  -H "api-key: $QDRANT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1280,
      "distance": "Cosine",
      "on_disk": true
    },
    "hnsw_config": {
      "m": 32,
      "ef_construct": 200,
      "on_disk": true
    }
  }'
```

### Step 2: Batch Ingest Vectors & Await Optimizer Completion
```bash
# Ingest vectors via batch script
python scripts/ingest_studio_batch.py --batch-manifest data/studio_batches/batch_001.json --staging

# Poll optimizer status until green and fully indexed
curl -s "http://10.0.1.50:6333/collections/jewellery_catalog_mobilenetv4_staging" | jq '{
  status: .result.status,
  optimizer: .result.optimizer_status,
  points: .result.points_count,
  indexed: .result.indexed_vectors_count
}'
```

### Step 3: Atomic Alias Switch (Zero-Downtime Cutover)
```bash
curl -X POST "http://10.0.1.50:6333/collections/aliases" \
  -H "api-key: $QDRANT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {
        "swap_alias": {
          "alias_name": "jewellery_catalog_active",
          "new_collection_name": "jewellery_catalog_mobilenetv4_staging"
        }
      }
    ]
  }'
```

---

## 5. Backup, Disaster Recovery & Rollback Runbook

### 5.1 Automated Snapshot Schedule (Cron)
```bash
# /etc/cron.daily/qdrant-snapshot-s3
#!/bin/bash
set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_RES=$(curl -s -X POST "http://localhost:6333/collections/jewellery_catalog_active/snapshots" -H "api-key: $QDRANT_API_KEY")
SNAPSHOT_NAME=$(echo $SNAPSHOT_RES | jq -r .result.name)

# Copy snapshot file to S3
aws s3 cp /var/lib/qdrant/snapshots/jewellery_catalog_active/${SNAPSHOT_NAME} \
  s3://jewellery-retrieval-backups/qdrant/${TIMESTAMP}_${SNAPSHOT_NAME}

# Retain only last 30 snapshots on S3
aws s3 ls s3://jewellery-retrieval-backups/qdrant/ | sort -r | awk 'NR>30 {print $4}' | while read f; do
  aws s3 rm s3://jewellery-retrieval-backups/qdrant/$f
done
```

### 5.2 Emergency Rollback Runbook
If an ingested studio photoshoot contains mislabeled SKUs, corrupt vectors, or retrieval regression:

1. **Step 1 — Point alias back to previous stable collection**:
   ```bash
   curl -X POST "http://10.0.1.50:6333/collections/aliases" \
     -H "api-key: $QDRANT_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "actions": [
         {
           "swap_alias": {
             "alias_name": "jewellery_catalog_active",
             "new_collection_name": "jewellery_catalog_mobilenetv4_v1"
           }
         }
       ]
     }'
   ```
2. **Step 2 — Verify search response on rollback alias**:
   ```bash
   curl -s "http://10.0.1.50:8000/health" | jq .
   ```
3. **Step 3 — Snapshot Restore from S3 (Total Disaster Recovery)**:
   ```bash
   curl -X POST "http://10.0.1.50:6333/collections/jewellery_catalog_mobilenetv4_v1/snapshots/recover" \
     -H "api-key: $QDRANT_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "location": "s3://jewellery-retrieval-backups/qdrant/20261011_backup.snapshot"
     }'
   ```

---

## 6. Infrastructure Cost Analysis & Budgeting

| Component | AWS Resource | Sizing & Throughput | Monthly Cost (On-Demand) | Monthly Cost (1-Yr Reserved) | Cost Savings |
|---|---|---|:---:|:---:|:---:|
| **Compute** | EC2 `c6i.4xlarge` | 16 vCPU, 32 GB RAM, 12.5 Gbps network | \$489.60 | \$308.16 | **37.1%** |
| **Primary Disk** | EBS `gp3` SSD | 100 GB, 3,000 IOPS, 125 MB/s throughput | \$8.00 | \$8.00 | 0.0% |
| **Backup Storage** | AWS S3 Standard | 50 GB snapshot rolling retention | \$1.15 | \$1.15 | 0.0% |
| **Network Egress** | VPC Gateway S3 | Free internal AWS data transfer | \$0.00 | \$0.00 | — |
| **Total Operational Cost** | — | — | **\$498.75 / mo** | **\$317.31 / mo** | **36.4%** |

- **Cost Per Search Query**: At 10,000,000 queries per month, cloud compute costs equate to **\$0.0000317 per query** (\$0.032 per 1,000 queries).
