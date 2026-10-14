# Production Edge Security, Reverse-Proxy & TLS Hardening Guide

**Document Version**: 1.0.0  
**Target Architecture**: NGINX / Envoy Edge Reverse-Proxy fronting Jewellery Visual Retrieval Engine  
**Classification**: Engineering Operations Standard  

---

## 1. Overview & Edge Architecture

In production deployments, the FastAPI ASGI application service (`app:8000`) MUST NOT be directly exposed to the public Internet. Instead, an edge reverse-proxy (NGINX or Envoy) terminates TLS, enforces transport security, applies HTTP-to-HTTPS redirection, filters perimeter request bodies, protects sensitive administrative routes, and buffers static asset delivery.

```mermaid
graph TD
    PublicClient["Public HTTPS Client<br/>(Browsers / Mobile Apps / Webhooks)"] -->|TCP 443 / TLS 1.3| NGINX["Edge Reverse Proxy<br/>(NGINX / Envoy)"]
    AdminClient["Internal Operator / Prometheus<br/>(Internal Subnet / VPN)"] -->|TCP 443| NGINX

    subgraph InternalDMZ [Private Docker Bridge Network: jewellery_internal]
        NGINX -->|HTTP (Cleartext 8000)| App["jewellery_app<br/>(FastAPI / Uvicorn Non-Root)"]
        App -->|TCP 5432| DB[("PostgreSQL 16")]
        App -->|TCP 6379| Redis[("Redis 7")]
        App -->|TCP 6333| Qdrant[("Qdrant 1.19.2")]
    end
```

> [!CAUTION]
> **Zero Public Exposure for Internal Databases**:  
> PostgreSQL (`5432`), Redis (`6379`), and Qdrant (`6333`) MUST bind strictly to the private bridge network (`jewellery_internal`) or loopback interface (`127.0.0.1`). They MUST NEVER expose public ports (`0.0.0.0`) on host firewalls.

---

## 2. TLS 1.3 Configuration & Cryptographic Standards

### 2.1 Cipher Suites & Protocols
The reference NGINX configuration (`deploy/nginx/conf.d/jewellery_retrieval.conf`) enforces strict TLS protocols:
- **Supported Protocols**: `TLSv1.2`, `TLSv1.3`. Legacy protocols (`SSLv3`, `TLSv1.0`, `TLSv1.1`) are explicitly disabled.
- **TLS 1.3 Ciphers**: Handled automatically by OpenSSL 1.1.1+ / OpenSSL 3.0 (`TLS_AES_128_GCM_SHA256`, `TLS_AES_256_GCM_SHA384`, `TLS_CHACHA20_POLY1305_SHA256`).
- **TLS 1.2 Ciphers**: ECDHE key exchange with GCM authenticated encryption:
  ```nginx
  ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384';
  ssl_prefer_server_ciphers off;
  ```

### 2.2 Certificate Provisioning & Automated Renewal
1. **Automated Provisioning (Certbot / Let's Encrypt)**:
   ```bash
   certbot certonly --webroot -w /var/www/certbot -d search.jewellery.internal --rsa-key-size 4096 --agree-tos
   ```
2. **Renewal Automation**: Run via cron or systemd timer twice daily:
   ```bash
   0 3,15 * * * certbot renew --post-hook "nginx -s reload"
   ```
3. **Expiry Monitoring**: Integrate Prometheus blackbox exporter (`probe_ssl_earliest_cert_expiry`) alerting if certificate expiration is $< 21$ days.

---

## 3. Perimeter Request Limits & DoS Protection

### 3.1 Request Body & Upload Sizing
- **`client_max_body_size 10M;`**: Mirrors the application-level 10 MB payload ceiling (`MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024`). NGINX rejects requests exceeding 10 MB with `HTTP 413 Payload Too Large` at the perimeter before any bytes reach the Python ASGI process.
- **PIL Decompression Bomb Defense**: Although images are checked in application code (`Image.MAX_IMAGE_PIXELS = 10_000_000`), NGINX drops malformed multipart chunks that stream abnormally slow using `client_body_timeout 30s`.

### 3.2 Timeouts & Keepalive
| Directive | Value | Purpose |
|---|:---:|---|
| `client_header_timeout` | 10s | Prevents Slowloris header-drip attacks |
| `client_body_timeout` | 30s | Drops slow-upload body drips |
| `proxy_connect_timeout` | 5s | Rapid failover if backend container is restarting |
| `proxy_read_timeout` | 60s | Accommodates heavy multi-view ANN search bursts |
| `proxy_send_timeout` | 30s | Bounds upstream socket transmission |
| `keepalive_timeout` | 65s | Aligned with standard cloud load balancer idle timers |

---

## 4. Rate-Limiting Division of Responsibilities

| Responsibility Layer | Enforcement Mechanism | Granularity | Objective |
|---|---|---|---|
| **Edge Reverse-Proxy** | NGINX `limit_req_zone` (IP-based) | Per IP address (e.g. 100 req/s burst 50) | Shields edge from volumetric DDoS and TCP floods |
| **Application Layer** | `DistributedRateLimiter` (Redis Token Bucket) | Per Authenticated Tenant + Actor | Enforces contractual SaaS tenant quotas & anti-abuse |
| **Outage Fallback** | In-Process Token Bucket (Memory-local) | Per Tenant + Actor (50% burst budget) | Prevents denial-of-service during Redis partition |

---

## 5. Sensitive Endpoint Protection

1. **Prometheus Scraping (`/metrics`)**:
   - Contains operational internal metrics (point counts, latencies, tenant distribution).
   - **Rule**: Blocked from public access; allowed only for internal monitoring subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.1`).
2. **Index Lifecycle Administration (`/api/lifecycle/*`)**:
   - Manages promotion, canary routing, rollback, and index retirement.
   - **Rule**: Blocked from public access at the edge, AND protected at the application layer by OAuth2 scope `ops:admin` with mandatory PostgreSQL audit logging.

---

## 6. Static Media Asset Caching Policy (`/media/`)

The application serves catalog thumbnails via `GET /media/{category}/{filename}`.
1. **Public Catalog Assets**:
   - Cached for 24 hours (`proxy_cache_valid 200 24h`).
   - Served with `Cache-Control: public, no-transform`.
   - Stale cache serving enabled during upstream reboots (`proxy_cache_use_stale error timeout updating`).
2. **Strict Prohibition on Authenticated Content**:
   - NGINX configuration explicitly bypasses cache if request contains `Authorization` or `X-API-Key`:
     ```nginx
     proxy_cache_bypass $http_authorization $http_x_api_key;
     proxy_no_cache $http_authorization $http_x_api_key;
     ```
   - Prevents cross-tenant asset cache pollution.

---

## 7. Mandatory HTTP Security Headers

All responses must emit the following headers:
- `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none';`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
