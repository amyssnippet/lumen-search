# Enterprise Route Security & Access Control Matrix

**Service**: Jewellery Visual Retrieval Engine  
**Release**: v0.2.0 (Milestone PH-008)  
**Security Standard**: Dual Authentication (JWT + SHA-256 API Key), RBAC, Multi-Tenant Isolation, Distributed Rate Limiting  

---

## 1. Route Security Matrix

| HTTP Method | Route Endpoint | Authentication Required | Required RBAC Scope | Distributed Rate Limiting | Tenant Isolation Enforcement | Payload & Security Guards |
|---|---|:---:|:---:|:---:|:---:|---|
| `GET` | `/` | No (Public) | None | No | None | Static HTML delivery |
| `GET` | `/static/{path}` | No (Public) | None | No | None | Static CSS/JS assets |
| `GET` | `/health` | No (Public) | None | No | None | Legacy ping response |
| `GET` | `/health/live` | No (Public) | None | No | None | Liveness probe ($<5$ ms execution, non-blocking) |
| `GET` | `/health/ready` | No (Public) | None | No | None | Deep readiness (PostgreSQL, Redis, Qdrant, Model file; bounded timeouts; zero secret leakage) |
| `GET` | `/metrics` | No (Ops) | None | No | None | Prometheus format; strictly bounded label cardinality (no PII, no tenant IDs) |
| `POST` | `/api/auth/token` | No (Public Login) | None | Standard IP | None | Email & SHA-256 password authentication; issues scoped HS256 JWT |
| `GET` | `/api/auth/me` | **Yes** (JWT or API Key) | Valid Credential | Yes (Tiered) | Self-Tenant | Introspects caller tenant, actor ID, and granted scopes |
| `POST` | `/api/auth/keys` | **Yes** (JWT or API Key) | `tenant:admin` | Yes (Tiered) | Self-Tenant | Generates SHA-256 hashed API key bound strictly to caller's tenant |
| `GET` | `/api/auth/keys` | **Yes** (JWT or API Key) | `tenant:admin` | Yes (Tiered) | Self-Tenant | Lists active API keys matching caller `tenant_id` |
| `DELETE` | `/api/auth/keys/{id}` | **Yes** (JWT or API Key) | `tenant:admin` | Yes (Tiered) | Self-Tenant | Immediate deactivation; returns 404 if key belongs to another tenant |
| `GET` | `/api/products` | **Yes** (JWT or API Key) | `search:read` | Yes (Tiered) | Self-Tenant | Relational SQL filtered strictly by `Product.tenant_id == caller_tenant_id` |
| `GET` | `/api/catalog/stats` | Permissive / Read | Optional | Yes (Tiered) | System | Reports vector count and catalog category aggregates |
| `POST` | `/api/search` | **Yes** (JWT or API Key) | `search:read` | **Yes (Token Bucket)** | **Anti-Spoofing + Qdrant Payload Filter** | - **Anti-Spoofing Guard**: Caller cannot supply foreign `tenant_id`<br/>- **Payload Ceiling**: Max 10 MB upload size<br/>- **Decompression Bomb**: `Image.MAX_IMAGE_PIXELS = 10,000,000`<br/>- **Extension Whitelist**: `.jpg`, `.jpeg`, `.png`, `.webp`<br/>- **Path Sanitization**: `/media/{cat}/{fname}` URLs without raw disk paths<br/>- **Audit Logging**: Recorded in PostgreSQL with IP and user agent |
| `POST` | `/api/feedback` | Optional / Read | `search:read` | Yes | System | Pydantic regex schema: `feedback_type` in `[correct, incorrect, uncertain]` |
| `GET` | `/api/evaluation` | No (Research) | None | Yes | None | Reads static offline benchmark JSON reports |
| `GET` | `/media/{cat}/{file}` | No (Public Media) | None | Yes | None | **Directory Traversal Defense**: Rejects `..`, `/`, `\\`; Category whitelist (`necklace`, `ring`, `query_ring`) |
| `GET` | `/annotate` | No (Tooling UI) | None | No | None | Annotation workflow frontend |
| `GET` | `/api/annotation/pairs` | Optional / Tooling | None | Yes | None | Reads annotation queue |
| `POST` | `/api/annotation/submit` | Optional / Tooling | None | Yes | None | Regex verdict validation: `confirmed_same_sku`, `confirmed_diff_sku`, etc. |
| `GET` | `/region-annotate` | No (Tooling UI) | None | No | None | Region annotation workflow frontend |
| `POST` | `/api/regions/save` | Optional / Tooling | None | Yes | None | Bounding box array validation: `min_length=4, max_length=4` |
| `POST` | `/api/regions/propose` | Optional / Tooling | None | Yes | None | Safe path lookup with category bounding box computation |

---

## 2. Authentication Enforcement Modes

The system supports two environment-governed authentication modes:

1. **Permissive Development Mode (`REQUIRE_AUTH=false`, default)**:
   - Designed for local testing, integration suites, and backwards compatibility.
   - Unauthenticated requests default to `tenant_id="default_tenant"` with developer scopes.
   - Anti-spoofing guards remain active: callers attempting to pass an explicit `tenant_id` different from `default_tenant` are rejected with `HTTP 403 Forbidden`.

2. **Strict Production Mode (`REQUIRE_AUTH=true`, Docker Compose / Production)**:
   - Enforced in all production environments via `.env.production`.
   - Every protected route (`/api/search`, `/api/products`, `/api/auth/me`, `/api/auth/keys`) immediately rejects uncredentialed callers with `HTTP 401 Unauthorized`.
   - Requires valid `X-API-Key` or `Authorization: Bearer <JWT>` header.
