"""FastAPI application for Jewellery Visual Retrieval Engine."""

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Response, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
import io
import time
import uuid
import numpy as np
from sqlalchemy.orm import Session

# Decompression bomb protection (max 10 megapixels)
Image.MAX_IMAGE_PIXELS = 10_000_000

from engine.embeddings import get_encoder
from engine.index import CatalogIndex
from engine.preprocessing import PreprocessConfig
from engine.search import SearchEngine
from engine.localization import JewelleryLocalizer
from engine.soft_attention import compute_combined_weights, render_attention_heatmap

from app.auth import (
    AuthenticatedIdentity,
    get_current_identity,
    require_scope,
    create_access_token,
    generate_new_api_key,
)
from app.rate_limiter import enforce_rate_limit
from app.metrics import (
    get_metrics_response,
    record_request_metric,
    record_stage_latency,
    record_qdrant_op,
    record_ratelimit_event,
    record_lifecycle_op,
    set_canary_percentage,
    set_incident_simulation,
)
from app.logging_conf import RequestCorrelationMiddleware, configure_structured_logging
from engine.db.session import get_db, init_db
from engine.db.models import Tenant, User, ApiKey, Product, Asset, AuditLog
from engine.lifecycle import IndexLifecycleManager
from engine.index_router import CanaryRouter, RouteTarget

# Ensure database tables exist and logging is configured
init_db()
configure_structured_logging()
SERVER_START_TIME = time.time()

# Banned / compromised credentials list (PH-013)
BANNED_COMPROMISED_CREDENTIALS = {
    "jewellery_internal_edge_proxy_secret_2026",
    "dev_secret_key_change_in_production",
    "staging_super_secret_jwt_key_2026_secure",
    "staging_secret_pass_2026",
    "jvr_default_dev_key_2026_secret",
}

# Startup Secret Validation for Production Hardening (PH-013 Remediated)
def validate_production_secrets():
    app_env = os.environ.get("APP_ENV", "development").lower()
    if app_env == "production":
        insecure_patterns = [
            ("JWT_SECRET_KEY", "dev_secret_key_change_in_production"),
            ("JWT_SECRET_KEY", "staging_super_secret_jwt_key_2026_secure"),
            ("POSTGRES_PASSWORD", "staging_secret_pass_2026"),
            ("DATABASE_URL", "staging_secret_pass_2026"),
            ("API_KEY", "jvr_default_dev_key_2026_secret"),
            ("INTERNAL_PROXY_SECRET", "jewellery_internal_edge_proxy_secret_2026"),
        ]
        errors = []
        for var, bad_val in insecure_patterns:
            val = os.environ.get(var, "")
            if bad_val in val:
                errors.append(f"Insecure/default credential detected in {var} for production environment.")

        jwt_key = os.environ.get("JWT_SECRET_KEY", "")
        if len(jwt_key) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters long in production.")

        if os.environ.get("REQUIRE_REVERSE_PROXY", "false").lower() in ("true", "1"):
            proxy_secret = os.environ.get("INTERNAL_PROXY_SECRET", "")
            if not proxy_secret:
                errors.append("INTERNAL_PROXY_SECRET must be explicitly set when REQUIRE_REVERSE_PROXY=true in production.")
            elif len(proxy_secret) < 32:
                errors.append("INTERNAL_PROXY_SECRET must be at least 32 characters long in production.")
            elif proxy_secret in BANNED_COMPROMISED_CREDENTIALS:
                errors.append("INTERNAL_PROXY_SECRET uses a known compromised or default literal.")

        if errors:
            raise RuntimeError("FATAL: Production secret validation failed:\n" + "\n".join(errors))

validate_production_secrets()

# Reverse Proxy & Edge Perimeter Enforcement (PH-013 Remediated — no literal fallback)
REQUIRE_REVERSE_PROXY = os.environ.get("REQUIRE_REVERSE_PROXY", "false").lower() in ("true", "1")
INTERNAL_PROXY_SECRET = os.environ.get("INTERNAL_PROXY_SECRET", "")

def _record_audit_log_background(
    tenant_id: str,
    actor_type: str,
    actor_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    ip_address: Optional[str],
    user_agent: Optional[str],
    status: str,
    details: str
):
    try:
        from engine.db.session import SessionLocal
        with SessionLocal() as session:
            audit = AuditLog(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                actor_type=actor_type,
                actor_id=actor_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                status=status,
                details=details
            )
            session.add(audit)
            session.commit()
    except Exception:
        pass

app = FastAPI(
    title="Jewellery Visual Retrieval Engine",
    description="Fine-grained visual search engine for rings, necklaces, and gemstones with enterprise multi-tenancy and observability.",
    version="0.2.0"
)

@app.middleware("http")
async def verify_reverse_proxy_middleware(request: Request, call_next):
    if REQUIRE_REVERSE_PROXY:
        proxy_secret = request.headers.get("x-proxy-secret")
        # Enforce constant-time comparison, reject missing/empty secret or known compromised secret
        if (
            not proxy_secret
            or not INTERNAL_PROXY_SECRET
            or proxy_secret in BANNED_COMPROMISED_CREDENTIALS
            or not hmac.compare_digest(proxy_secret, INTERNAL_PROXY_SECRET)
        ):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Direct backend access forbidden. Requests must ingress via the edge reverse proxy."}
            )
    response = await call_next(request)
    return response

app.add_middleware(RequestCorrelationMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine Cache & Lifecycle Singletons
ENGINES: Dict[str, SearchEngine] = {}
HYBRID_ENGINE = None
CANARY_ROUTER = CanaryRouter()
LIFECYCLE_MANAGER = IndexLifecycleManager()

def get_search_engine(model_name: str = "mobilenetv4", backend: str = "qdrant") -> SearchEngine:
    engine_key = f"{model_name.lower()}_{backend.lower()}"
    if engine_key not in ENGINES:
        ENGINES[engine_key] = SearchEngine(
            model_name=model_name,
            backend=backend,
            canary_router=CANARY_ROUTER if backend.lower() == "qdrant" and model_name.lower() == "mobilenetv4" else None
        )
    return ENGINES[engine_key]

def reload_search_engines():
    """Clear cached search engine instances so alias updates take immediate effect."""
    global ENGINES
    ENGINES.clear()

def get_hybrid_search_engine():
    global HYBRID_ENGINE
    if HYBRID_ENGINE is None:
        from engine.hybrid_search import HybridSearchEngine
        HYBRID_ENGINE = HybridSearchEngine(index_dir="data/index")
    return HYBRID_ENGINE

LOCALIZER: Optional[JewelleryLocalizer] = None

def get_localizer() -> JewelleryLocalizer:
    global LOCALIZER
    if LOCALIZER is None:
        LOCALIZER = JewelleryLocalizer()
    return LOCALIZER

ATTENTION_ENGINES: Dict[str, Any] = {}

def get_attention_search_engine(model_name: str = "dinov3"):
    model_key = model_name.lower()
    if model_key not in ATTENTION_ENGINES:
        from engine.attention_search import SoftAttentionSearchEngine
        ATTENTION_ENGINES[model_key] = SoftAttentionSearchEngine(model_name=model_key)
    return ATTENTION_ENGINES[model_key]

class FeedbackRequest(BaseModel):
    query_id: Optional[str] = "unspecified"
    result_id: str
    feedback_type: str = Field(..., pattern="^(correct|incorrect|uncertain)$")
    notes: Optional[str] = None
    similarity_score: Optional[float] = None
    model_name: Optional[str] = "mobilenetv4"

@app.get("/metrics")
def metrics():
    """Prometheus exposition format metrics exporter."""
    return get_metrics_response()


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateKeyRequest(BaseModel):
    name: str
    scopes: Optional[str] = "search:read"


@app.post("/api/auth/token")
def login(login_req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with user email and password; issue JWT bearer token."""
    pwd_hash = hashlib.sha256(login_req.password.encode("utf-8")).hexdigest()
    user = db.query(User).filter(User.email == login_req.email, User.hashed_password == pwd_hash).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
    if not tenant or not tenant.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant organization is deactivated")

    role_scopes_map = {
        "admin": ["search:read", "catalog:write", "tenant:admin", "ops:admin"],
        "ops": ["search:read", "catalog:write", "ops:admin"],
        "editor": ["search:read", "catalog:write"],
        "viewer": ["search:read"],
    }
    scopes = role_scopes_map.get(user.role, ["search:read"])
    token = create_access_token(
        tenant_id=user.tenant_id,
        actor_id=user.id,
        scopes=scopes,
        user_email=user.email
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "tenant_id": user.tenant_id,
        "role": user.role,
        "scopes": scopes
    }


@app.get("/api/auth/me")
def get_caller_identity(identity: AuthenticatedIdentity = Depends(get_current_identity)):
    """Return caller's verified tenant, identity, and RBAC scopes."""
    return {
        "tenant_id": identity.tenant_id,
        "actor_id": identity.actor_id,
        "actor_type": identity.actor_type,
        "scopes": sorted(list(identity.scopes)),
        "tier": identity.tier,
        "rate_limit_rpm": identity.rate_limit_rpm,
        "rate_limit_burst": identity.rate_limit_burst,
        "email": identity.user_email
    }


@app.post("/api/auth/keys")
def create_api_key(
    req: CreateKeyRequest,
    identity: AuthenticatedIdentity = Depends(require_scope("tenant:admin")),
    db: Session = Depends(get_db)
):
    """Generate a new programmatic API key for the caller's tenant."""
    raw_key, kh, prefix = generate_new_api_key()
    new_key = ApiKey(
        id=str(uuid.uuid4()),
        tenant_id=identity.tenant_id,
        key_hash=kh,
        key_prefix=prefix,
        name=req.name,
        scopes=req.scopes or "search:read",
        is_active=True
    )
    db.add(new_key)
    db.commit()
    return {
        "key_id": new_key.id,
        "name": new_key.name,
        "key_prefix": new_key.key_prefix,
        "api_key": raw_key,  # Raw secret only displayed once!
        "scopes": new_key.scopes,
        "tenant_id": identity.tenant_id
    }


@app.get("/api/auth/keys")
def list_api_keys(
    identity: AuthenticatedIdentity = Depends(require_scope("tenant:admin")),
    db: Session = Depends(get_db)
):
    """List active API keys belonging to the caller's tenant."""
    keys = db.query(ApiKey).filter(ApiKey.tenant_id == identity.tenant_id).all()
    return {
        "tenant_id": identity.tenant_id,
        "keys": [
            {
                "id": k.id,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "scopes": k.scopes,
                "is_active": k.is_active,
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None
            }
            for k in keys
        ]
    }


@app.delete("/api/auth/keys/{key_id}")
def revoke_api_key(
    key_id: str,
    identity: AuthenticatedIdentity = Depends(require_scope("tenant:admin")),
    db: Session = Depends(get_db)
):
    """Revoke an API key belonging to the caller's tenant."""
    key = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.tenant_id == identity.tenant_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = False
    db.commit()
    return {"status": "success", "message": f"API key {key_id} revoked."}


@app.get("/api/products")
def list_products(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    identity: AuthenticatedIdentity = Depends(require_scope("search:read")),
    db: Session = Depends(get_db)
):
    """List catalogue products scoped strictly to the authenticated tenant."""
    query = db.query(Product).filter(Product.tenant_id == identity.tenant_id)
    if category and category.lower() not in ("all", ""):
        query = query.filter(Product.category == category.lower())
    if sku and sku.strip():
        query = query.filter(Product.sku.ilike(f"%{sku.strip()}%"))
    total = query.count()
    products = query.order_by(Product.sku).offset(offset).limit(limit).all()
    return {
        "tenant_id": identity.tenant_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "products": [
            {
                "id": p.id,
                "sku": p.sku,
                "name": p.name,
                "category": p.category,
                "status": p.status,
                "metal_color": p.metal_color,
                "gemstone_type": p.gemstone_type,
                "setting_type": p.setting_type,
                "price_cents": p.price_cents,
                "currency": p.currency,
                "asset_count": len(p.assets),
                "thumbnail_url": f"/media/{p.category}/{p.assets[0].filename}" if p.assets else None,
                "assets": [
                    {
                        "id": a.id,
                        "filename": a.filename,
                        "image_url": f"/media/{p.category}/{a.filename}",
                        "view_role": a.view_role,
                        "perspective": a.perspective,
                        "provenance_tier": a.provenance_tier
                    }
                    for a in p.assets
                ]
            }
            for p in products
        ]
    }


# ==============================================================================
# Index Lifecycle & Canary Management Endpoints (PH-009)
# ==============================================================================

class CandidateRegisterRequest(BaseModel):
    collection_name: str


class CandidateValidateRequest(BaseModel):
    collection_name: Optional[str] = None
    min_points: Optional[int] = 1
    run_retrieval_smoke: Optional[bool] = True


class CanaryConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    candidate_collection: Optional[str] = None
    percentage: Optional[int] = Field(None, ge=0, le=100)
    allowlisted_tenants: Optional[List[str]] = None
    fallback_to_production: Optional[bool] = None


class PromoteRequest(BaseModel):
    candidate_collection: Optional[str] = None
    dry_run: Optional[bool] = False
    skip_validation: Optional[bool] = False


class RollbackRequest(BaseModel):
    dry_run: Optional[bool] = False


class RetireRequest(BaseModel):
    collection_name: str
    force: Optional[bool] = False


@app.get("/api/lifecycle/status")
def get_lifecycle_status(identity: AuthenticatedIdentity = Depends(require_scope("ops:admin"))):
    """Inspect active production index, candidate collection, and canary routing state."""
    status_data = LIFECYCLE_MANAGER.get_status_summary()
    status_data["canary_routing"] = CANARY_ROUTER.get_config()
    return status_data


@app.post("/api/lifecycle/candidate/register")
def register_index_candidate(
    req: CandidateRegisterRequest,
    identity: AuthenticatedIdentity = Depends(require_scope("ops:admin")),
    db: Session = Depends(get_db)
):
    """Register a new candidate index collection for staging and validation."""
    res = LIFECYCLE_MANAGER.register_candidate(req.collection_name)
    if res.get("status") == "FAILED":
        raise HTTPException(status_code=400, detail=res.get("error"))

    CANARY_ROUTER.configure(candidate_collection=req.collection_name, actor_id=identity.actor_id)
    
    try:
        audit = AuditLog(
            id=str(uuid.uuid4()),
            tenant_id=identity.tenant_id,
            actor_type=identity.actor_type,
            actor_id=identity.actor_id,
            action="register_candidate",
            resource_type="vector_collection",
            resource_id=req.collection_name,
            status="success"
        )
        db.add(audit)
        db.commit()
    except Exception:
        db.rollback()

    return res


@app.post("/api/lifecycle/candidate/validate")
def validate_index_candidate(
    req: CandidateValidateRequest,
    identity: AuthenticatedIdentity = Depends(require_scope("ops:admin"))
):
    """Run comprehensive integrity and retrieval smoke tests against a candidate collection."""
    target_col = req.collection_name or LIFECYCLE_MANAGER.get_candidate_collection()
    if not target_col:
        raise HTTPException(status_code=400, detail="No candidate collection specified or registered.")

    val_res = LIFECYCLE_MANAGER.validate_collection_integrity(target_col, min_points=req.min_points or 1)
    smoke_res = {"passed": True}
    if req.run_retrieval_smoke:
        smoke_res = LIFECYCLE_MANAGER.run_retrieval_smoke_test(target_col)

    passed = val_res.get("valid", False) and smoke_res.get("passed", False)
    record_lifecycle_op("validate", "success" if passed else "failure")

    return {
        "status": "PASSED" if passed else "FAILED",
        "collection_name": target_col,
        "integrity": val_res,
        "smoke_test": smoke_res
    }


@app.post("/api/lifecycle/canary/config")
def update_canary_config(
    req: CanaryConfigRequest,
    identity: AuthenticatedIdentity = Depends(require_scope("ops:admin")),
    db: Session = Depends(get_db)
):
    """Update runtime canary percentage and tenant allowlists."""
    updated = CANARY_ROUTER.configure(
        enabled=req.enabled,
        candidate_collection=req.candidate_collection,
        percentage=req.percentage,
        allowlisted_tenants=req.allowlisted_tenants,
        fallback_to_production=req.fallback_to_production,
        actor_id=identity.actor_id
    )
    if req.percentage is not None:
        set_canary_percentage(req.percentage)

    try:
        audit = AuditLog(
            id=str(uuid.uuid4()),
            tenant_id=identity.tenant_id,
            actor_type=identity.actor_type,
            actor_id=identity.actor_id,
            action="canary_config_update",
            resource_type="canary_router",
            resource_id="canary_router",
            status="success",
            details=json.dumps(updated)
        )
        db.add(audit)
        db.commit()
    except Exception:
        db.rollback()

    return {"status": "SUCCESS", "canary_config": updated}


@app.post("/api/lifecycle/promote")
def promote_index(
    req: PromoteRequest,
    identity: AuthenticatedIdentity = Depends(require_scope("ops:admin")),
    db: Session = Depends(get_db)
):
    """Atomically switch production alias to candidate collection."""
    res = LIFECYCLE_MANAGER.promote_candidate(
        candidate_collection=req.candidate_collection,
        dry_run=req.dry_run or False,
        actor_id=identity.actor_id,
        skip_validation=req.skip_validation or False
    )
    if res.get("status") == "FAILED":
        record_lifecycle_op("promote", "failure")
        raise HTTPException(status_code=400, detail=res.get("error"))

    if res.get("status") == "DRY_RUN":
        record_lifecycle_op("promote", "dry_run")
        return res

    if res.get("status") == "SUCCESS":
        record_lifecycle_op("promote", "success")
        CANARY_ROUTER.disable(actor_id=identity.actor_id)
        set_canary_percentage(0)
        reload_search_engines()

        try:
            audit = AuditLog(
                id=str(uuid.uuid4()),
                tenant_id=identity.tenant_id,
                actor_type=identity.actor_type,
                actor_id=identity.actor_id,
                action="promote_index",
                resource_type="vector_alias",
                resource_id=LIFECYCLE_MANAGER.production_alias,
                status="success",
                details=json.dumps(res)
            )
            db.add(audit)
            db.commit()
        except Exception:
            db.rollback()

    return res


@app.post("/api/lifecycle/rollback")
def rollback_index(
    req: RollbackRequest,
    identity: AuthenticatedIdentity = Depends(require_scope("ops:admin")),
    db: Session = Depends(get_db)
):
    """Revert production alias to the previous collection."""
    res = LIFECYCLE_MANAGER.rollback_production(
        dry_run=req.dry_run or False,
        actor_id=identity.actor_id
    )
    if res.get("status") == "FAILED":
        record_lifecycle_op("rollback", "failure")
        raise HTTPException(status_code=400, detail=res.get("error"))

    if res.get("status") == "DRY_RUN":
        record_lifecycle_op("rollback", "dry_run")
        return res

    if res.get("status") == "ROLLBACK_SUCCESS":
        record_lifecycle_op("rollback", "success")
        CANARY_ROUTER.disable(actor_id=identity.actor_id)
        set_canary_percentage(0)
        reload_search_engines()

        try:
            audit = AuditLog(
                id=str(uuid.uuid4()),
                tenant_id=identity.tenant_id,
                actor_type=identity.actor_type,
                actor_id=identity.actor_id,
                action="rollback_index",
                resource_type="vector_alias",
                resource_id=LIFECYCLE_MANAGER.production_alias,
                status="success",
                details=json.dumps(res)
            )
            db.add(audit)
            db.commit()
        except Exception:
            db.rollback()

    return res


@app.post("/api/lifecycle/retire")
def retire_index(
    req: RetireRequest,
    identity: AuthenticatedIdentity = Depends(require_scope("ops:admin")),
    db: Session = Depends(get_db)
):
    """Safely delete an obsolete index collection after verifying guardrails."""
    res = LIFECYCLE_MANAGER.retire_collection(
        collection_name=req.collection_name,
        force=req.force or False,
        actor_id=identity.actor_id
    )
    if res.get("status") == "FORBIDDEN":
        record_lifecycle_op("retire", "forbidden")
        raise HTTPException(status_code=403, detail=res.get("error"))
    if res.get("status") == "FAILED":
        record_lifecycle_op("retire", "failure")
        raise HTTPException(status_code=400, detail=res.get("error"))

    record_lifecycle_op("retire", "success")
    try:
        audit = AuditLog(
            id=str(uuid.uuid4()),
            tenant_id=identity.tenant_id,
            actor_type=identity.actor_type,
            actor_id=identity.actor_id,
            action="retire_index",
            resource_type="vector_collection",
            resource_id=req.collection_name,
            status="success"
        )
        db.add(audit)
        db.commit()
    except Exception:
        db.rollback()

    return res


class IncidentSimulationRequest(BaseModel):
    active: bool = True


@app.post("/api/lifecycle/simulate-incident")
def trigger_incident_simulation(
    req: IncidentSimulationRequest,
    identity: AuthenticatedIdentity = Depends(require_scope("ops:admin"))
):
    """Toggle operational incident simulation gauge for Prometheus alert verification drills."""
    set_incident_simulation(req.active)
    return {"status": "ok", "incident_simulation_active": req.active}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Jewellery Visual Retrieval Engine",
        "engines_cached": list(ENGINES.keys()),
        "version": "0.2.0"
    }


@app.get("/health/live")
def health_live():
    """Lightweight process liveness probe for orchestrators (Docker / Kubernetes)."""
    return {
        "status": "ok",
        "service": "Jewellery Visual Retrieval Engine",
        "uptime_sec": round(time.time() - SERVER_START_TIME, 2),
        "version": "0.2.0"
    }


@app.get("/health/ready")
def health_ready(response: Response, db: Session = Depends(get_db)):
    """Deep readiness probe verifying critical dependencies.
    
    Checks:
    - Relational database connectivity via SELECT 1 (bounded 2.0s timeout)
    - Redis distributed cache / rate limiter ping (bounded 1.0s timeout)
    - Qdrant vector database collection listing (bounded 2.0s timeout)
    - Model weights file existence on disk
    
    Returns HTTP 200 when ready (or degraded if only Redis is down due to fail-secure fallback).
    Returns HTTP 503 if PostgreSQL, Qdrant, or Model weights are unavailable.
    """
    dependencies = {}
    
    # 1. Relational Database check
    t0 = time.time()
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        dependencies["database"] = {
            "status": "connected",
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }
    except Exception:
        dependencies["database"] = {
            "status": "down",
            "error": "Database connectivity check failed"
        }

    # 2. Redis Cache / Rate Limiter check
    t0 = time.time()
    try:
        import redis
        redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        r = redis.Redis.from_url(redis_url, socket_timeout=1.0, socket_connect_timeout=1.0)
        r.ping()
        dependencies["redis"] = {
            "status": "connected",
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }
    except Exception:
        dependencies["redis"] = {
            "status": "degraded",
            "warning": "Redis unavailable; fail-secure in-process rate limiter active"
        }

    # 3. Qdrant Vector DB check
    t0 = time.time()
    try:
        from qdrant_client import QdrantClient
        q_host = os.environ.get("QDRANT_HOST", "127.0.0.1")
        q_port = int(os.environ.get("QDRANT_PORT", "6333"))
        qc = QdrantClient(host=q_host, port=q_port, timeout=2.0, check_compatibility=False)
        colls = qc.get_collections()
        dependencies["qdrant"] = {
            "status": "connected",
            "collections_count": len(colls.collections),
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }
    except Exception:
        dependencies["qdrant"] = {
            "status": "down",
            "error": "Vector database connectivity check failed"
        }

    # 4. Reference Model Weights check
    model_path = Path("data/models/mobilenetv4_conv_small_fp32.onnx")
    if model_path.exists() and model_path.is_file():
        dependencies["model"] = {
            "status": "available",
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size
        }
    else:
        dependencies["model"] = {
            "status": "missing",
            "error": "MobileNetV4 ONNX model weights not found on disk"
        }

    is_critical_ready = (
        dependencies["database"]["status"] == "connected" and
        dependencies["qdrant"]["status"] == "connected" and
        dependencies["model"]["status"] == "available"
    )

    if not is_critical_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "unavailable"
    elif dependencies["redis"]["status"] != "connected":
        response.status_code = status.HTTP_200_OK
        overall_status = "degraded"
    else:
        response.status_code = status.HTTP_200_OK
        overall_status = "ready"

    return {
        "status": overall_status,
        "service": "Jewellery Visual Retrieval Engine",
        "uptime_sec": round(time.time() - SERVER_START_TIME, 2),
        "dependencies": dependencies
    }


@app.get("/api/catalog/stats")
def catalog_stats(model: str = "mobilenetv4", backend: str = "qdrant"):
    try:
        engine = get_search_engine(model_name=model, backend=backend)
        stats = engine.index.stats()
        # Total dataset stats
        total_raw = 0
        cat_counts = {}
        data_dir = Path("Jewellery_Data")
        if data_dir.exists():
            for c_dir in data_dir.iterdir():
                if c_dir.is_dir() and not c_dir.name.startswith("."):
                    count = len([f for f in c_dir.glob("*") if f.is_file() and not f.name.startswith(".")])
                    cat_counts[c_dir.name] = count
                    total_raw += count
        
        return {
            "total_images_in_dataset": total_raw,
            "dataset_categories": cat_counts,
            "index_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
def search_endpoint(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("mobilenetv4"),
    backend: str = Form("qdrant"),
    search_mode: str = Form("similar"), # "exact" or "similar"
    category_filter: Optional[str] = Form("all"),
    metal_color_filter: Optional[str] = Form(None),
    gemstone_filter: Optional[str] = Form(None),
    setting_filter: Optional[str] = Form(None),
    ef_search: int = Form(128),
    exact: bool = Form(False),
    aggregate_sku: bool = Form(False),
    top_k: int = Form(10),
    patch_weight: float = Form(0.2),
    spatial_weight: float = Form(0.0),
    attention_mode: Optional[str] = Form("none"),
    enable_clahe: bool = Form(False),
    enable_denoise: bool = Form(False),
    enable_crop: bool = Form(False),
    auto_enhance: bool = Form(False),
    robust_mode: bool = Form(True),
    tenant_id: Optional[str] = Form(None),
    identity: AuthenticatedIdentity = Depends(require_scope("search:read")),
):
    # Reject cross-tenant spoofing attempts in form data
    if tenant_id and tenant_id != identity.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cross-tenant access denied: caller authenticated as '{identity.tenant_id}' cannot operate on tenant '{tenant_id}'"
        )
    effective_tenant_id = identity.tenant_id

    # Enforce distributed rate limiting
    enforce_rate_limit(
        request=request,
        response=response,
        tenant_id=effective_tenant_id,
        actor_id=identity.actor_id,
        rate_limit_rpm=identity.rate_limit_rpm,
        rate_limit_burst=identity.rate_limit_burst
    )
    record_ratelimit_event("allowed")

    # Validate extension & content type
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    if file_ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{file_ext}'. Allowed: JPG, PNG, WEBP.")

    contents = file.file.read()
    # Size check: 10MB limit
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file exceeds 10MB size limit.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        # Re-open after verify
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {str(e)}")

    prep_config = PreprocessConfig(
        target_size=224,
        enable_clahe=enable_clahe,
        enable_denoise=enable_denoise,
        enable_autocrop=enable_crop,
        auto_enhance=auto_enhance
    )

    try:
        is_attention_mode = (
            model.lower() in ("soft_attention", "attention")
            or (attention_mode and attention_mode.lower() not in ("none", "", "off", "false"))
        )

        if is_attention_mode:
            backbone = "dinov2" if "dinov2" in model.lower() else "dinov3"
            att_engine = get_attention_search_engine(model_name=backbone)
            cfg = attention_mode.lower() if (attention_mode and attention_mode.lower() not in ("none", "", "off", "false")) else "soft_combined"
            att_res = att_engine.search(
                image=image,
                config_name=cfg,
                alpha=0.3,
                fuse_with_global=True,
                top_k=top_k,
                category_filter=None if category_filter in (None, "all", "") else category_filter
            )
            search_output = {
                "model_name": f"{backbone}_soft_attention",
                "search_mode": search_mode,
                "category_filter": category_filter,
                "timing_ms": {
                    "preprocessing": 0.0,
                    "encoding": att_res["timings_ms"]["extraction"],
                    "search": att_res["timings_ms"]["scoring"],
                    "total": att_res["timings_ms"]["total"]
                },
                "quality": {
                    "is_blurry": False,
                    "sharpness_score": 100.0,
                    "mean_luminance": 128.0,
                    "is_underexposed": False,
                    "is_overexposed": False,
                    "is_low_res": False,
                    "warnings": [],
                    "applied_steps": []
                },
                "has_reliable_match": len(att_res["results"]) > 0 and att_res["results"][0]["similarity"] >= 0.85,
                "top_similarity": att_res["results"][0]["similarity"] if att_res["results"] else 0.0,
                "total_candidates": len(att_res["results"]),
                "results": att_res["results"],
                "localization": att_res["query_localization"],
                "attention_weights": att_res["attention_weights"]
            }
        elif model.lower() in ("hybrid", "hybrid_rerank"):
            h_engine = get_hybrid_search_engine()
            search_output = h_engine.search_hybrid(
                image=image,
                top_k=top_k,
                candidate_depth=20,
                enable_rerank=(model.lower() == "hybrid_rerank"),
                patch_weight=patch_weight,
                spatial_weight=spatial_weight,
                category_filter=None if category_filter in (None, "all", "") else category_filter,
                search_mode=search_mode,
                preprocess_config=prep_config
            )
        else:
            filters = {}
            if metal_color_filter and metal_color_filter != "all":
                filters["metal_color"] = metal_color_filter
            if gemstone_filter and gemstone_filter != "all":
                filters["gemstone_type"] = gemstone_filter
            if setting_filter and setting_filter != "all":
                filters["setting_type"] = setting_filter

            engine = get_search_engine(model_name=model, backend=backend)
            search_output = engine.search_image(
                image=image,
                top_k=top_k,
                category_filter=None if category_filter in (None, "all", "") else category_filter,
                filters=filters if filters else None,
                search_mode=search_mode,
                preprocess_config=prep_config,
                ef_search=ef_search,
                exact=exact,
                aggregate_sku=aggregate_sku,
                tenant_id=effective_tenant_id,
                robust_mode=robust_mode
            )

        norm_cat = None if category_filter in (None, "all", "") else category_filter.lower().rstrip("s")
        if norm_cat:
            search_output["results"] = [r for r in search_output["results"] if r.get("category", "").lower().rstrip("s") == norm_cat]

        top_s = search_output["results"][0]["similarity"] if search_output["results"] else 0.0
        if "decision" not in search_output:
            if not search_output["results"] or top_s < 0.52:
                search_output["decision"] = "NO_CONFIDENT_MATCH"
                search_output["has_reliable_match"] = False
                search_output["decision_explanation"] = "No catalogue candidates met the minimum visual similarity threshold (52.0%)."
            elif top_s >= 0.88:
                search_output["decision"] = "EXACT_MATCH"
                search_output["has_reliable_match"] = True
                search_output["decision_explanation"] = "Candidate satisfies verified exact-match criteria (>= 88.0%)."
            else:
                search_output["decision"] = "VISUAL_SIMILARITY"
                search_output["has_reliable_match"] = True
                search_output["decision_explanation"] = "Retrieved candidates represent visual style similarities."
            search_output["category_filter_applied"] = norm_cat
            search_output["detected_category"] = search_output["results"][0].get("category") if search_output["results"] else None

        # Sanitize internal filesystem paths into public media URLs
        for r in search_output["results"]:
            cat = r["category"]
            fname = r["filename"]
            r["image_url"] = f"/media/{cat}/{fname}"
            # Remove raw local relative path
            r.pop("rel_path", None)

        # Query hash for feedback reference
        q_hash = hashlib.sha256(contents).hexdigest()[:16]
        search_output["query_id"] = q_hash

        # Gemstone / Component Localization overlay (if not already populated)
        if "localization" not in search_output or not search_output["localization"]:
            try:
                localizer = get_localizer()
                loc_res = localizer.locate(
                    image,
                    category=category_filter if category_filter in ("necklace", "ring") else None
                )
                search_output["localization"] = {
                    "box": [round(float(x), 4) for x in loc_res.box],
                    "box_pixels": loc_res.box_pixels,
                    "label": loc_res.label,
                    "confidence": float(loc_res.confidence),
                    "method": loc_res.method,
                    "is_fallback": bool(loc_res.is_fallback)
                }
            except Exception as e:
                search_output["localization"] = {
                    "box": [0.0, 0.0, 1.0, 1.0],
                    "box_pixels": [0, 0, image.size[1], image.size[0]],
                    "label": "fallback",
                    "confidence": 0.0,
                    "method": f"error: {str(e)}",
                    "is_fallback": True
                }

        # Attention Heatmap Overlay for visual explainability
        # Only compute heatmap if attention mode is explicitly requested
        if is_attention_mode or (attention_mode and attention_mode.lower() not in ("none", "", "off", "false")):
            try:
                if "attention_weights" in search_output and search_output["attention_weights"]:
                    weights = np.array(search_output["attention_weights"], dtype=np.float32)
                else:
                    loc_box = search_output.get("localization", {}).get("box")
                    is_fb = search_output.get("localization", {}).get("is_fallback", False)
                    weights = compute_combined_weights(image, loc_box, is_fallback=is_fb)
                hm_img = render_attention_heatmap(weights, image)
                buf = io.BytesIO()
                hm_img.save(buf, format="JPEG", quality=85)
                search_output["attention_heatmap_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                search_output["attention_heatmap_b64"] = None
        else:
            search_output["attention_heatmap_b64"] = None

        # Metrics and Audit Logging
        record_request_metric(endpoint="/api/search", status_code=200, model=model)
        if "timing_ms" in search_output:
            record_stage_latency("preprocessing", search_output["timing_ms"].get("preprocessing", 0) / 1000.0)
            record_stage_latency("encoding", search_output["timing_ms"].get("encoding", 0) / 1000.0)
            record_stage_latency("search", search_output["timing_ms"].get("search", 0) / 1000.0)
            record_stage_latency("overall", search_output["timing_ms"].get("total", 0) / 1000.0)
        record_qdrant_op("search", True)

        details_json = json.dumps({
            "model": model,
            "results_count": len(search_output.get("results", [])),
            "routing": search_output.get("routing")
        })
        background_tasks.add_task(
            _record_audit_log_background,
            effective_tenant_id,
            identity.actor_type,
            identity.actor_id,
            "search",
            "catalog",
            q_hash,
            request.client.host if request.client else None,
            request.headers.get("user-agent"),
            "success",
            details_json
        )

        return search_output
    except HTTPException:
        raise
    except Exception as e:
        record_request_metric(endpoint="/api/search", status_code=500, model=model)
        record_qdrant_op("search", False)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/feedback")
def submit_feedback(fb: FeedbackRequest):
    record = fb.model_dump()
    record["timestamp"] = int(time.time())
    
    feedback_file = Path("data/feedback.jsonl")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_file, "a") as fp:
        fp.write(json.dumps(record) + "\n")
        
    return {"status": "success", "message": "Feedback recorded."}

@app.get("/api/evaluation")
def get_evaluation(exp: Optional[str] = None):
    if exp in ("exp007", "soft_attention") or not exp:
        eval_soft = Path("output/eval_soft_attention.json")
        if eval_soft.exists():
            with open(eval_soft, "r") as fp:
                return json.load(fp)
    eval_hybrid = Path("output/eval_hybrid.json")
    if eval_hybrid.exists():
        with open(eval_hybrid, "r") as fp:
            return json.load(fp)
    eval_file = Path("output/evaluation_report.json")
    if not eval_file.exists():
        return {
            "status": "pending",
            "message": "Evaluation report not yet generated."
        }
    with open(eval_file, "r") as fp:
        return json.load(fp)

# Safe Media Serving: strictly prevents path traversal
@app.get("/media/{category}/{filename}")
def serve_jewellery_media(category: str, filename: str):
    if ".." in filename or "/" in filename or "\\" in filename or ".." in category:
        raise HTTPException(status_code=400, detail="Invalid path: traversal characters detected.")
    safe_name = Path(filename).name
    if category in ("necklace", "ring"):
        img_path = Path("Jewellery_Data") / category / safe_name
    elif category == "query_ring":
        img_path = Path("data/evaluation/queries/rings") / safe_name
    else:
        raise HTTPException(status_code=404, detail="Category not found")
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    
    media_type = "image/png" if safe_name.endswith(".png") else "image/jpeg"
    return FileResponse(str(img_path), media_type=media_type)


class AnnotationSubmission(BaseModel):
    pair_id: str
    query_id: str
    candidate_id: str
    verdict: str = Field(..., pattern="^(confirmed_same_sku|confirmed_diff_sku|identity_uncertain|insufficient_evidence)$")
    category: str
    image_source: Optional[str] = "catalog"
    query_viewpoint: Optional[str] = "frontal_angle"
    candidate_viewpoint: Optional[str] = "frontal_angle"
    visibility: Optional[str] = "clear_full"
    explanation: Optional[str] = None
    annotator: Optional[str] = "expert_annotator"
    provenance: Optional[str] = "synchronized_zoom_expert_audit"
    timestamp: Optional[str] = None

@app.get("/annotate")
def serve_annotate():
    return FileResponse("frontend/annotate.html")

@app.get("/api/annotation/pairs")
def get_annotation_pairs(category: Optional[str] = None, status: str = "pending", offset: int = 0, limit: int = 50):
    queue_path = Path("data/annotation_queue.json")
    if not queue_path.exists():
        return {"pairs": [], "total": 0}
    with open(queue_path, "r") as fp:
        all_pairs = json.load(fp)
    
    annotated_pair_ids = set()
    ann_path = Path("data/annotations.jsonl")
    if ann_path.exists():
        with open(ann_path, "r") as fp:
            for line in fp:
                if line.strip():
                    try:
                        record = json.loads(line)
                        annotated_pair_ids.add(record["pair_id"])
                    except Exception:
                        pass

    filtered = []
    for p in all_pairs:
        if category and p.get("category") != category:
            continue
        is_annotated = p["pair_id"] in annotated_pair_ids
        if status == "pending" and is_annotated:
            continue
        if status == "annotated" and not is_annotated:
            continue
        filtered.append(p)
        
    paged = filtered[offset : offset + limit]
    return {
        "pairs": paged,
        "total": len(filtered),
        "offset": offset,
        "limit": limit
    }

@app.post("/api/annotation/submit")
def submit_annotation(sub: AnnotationSubmission):
    import datetime
    ann_path = Path("data/annotations.jsonl")
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path("Jewellery_Data")
    q_matches = list(data_dir.rglob(f"{sub.query_id}.*"))
    c_matches = list(data_dir.rglob(f"{sub.candidate_id}.*"))
    if not q_matches or not c_matches:
        raise HTTPException(status_code=400, detail="Invalid query_id or candidate_id")

    payload = sub.dict()
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
    with open(ann_path, "a") as fp:
        fp.write(json.dumps(payload) + "\n")
        
    return {"status": "success", "message": "Annotation saved."}

@app.get("/api/annotation/stats")
def get_annotation_stats():
    queue_path = Path("data/annotation_queue.json")
    total_queue = 0
    if queue_path.exists():
        with open(queue_path, "r") as fp:
            total_queue = len(json.load(fp))
            
    ann_path = Path("data/annotations.jsonl")
    records = []
    if ann_path.exists():
        with open(ann_path, "r") as fp:
            for line in fp:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

    verdicts = {}
    categories = {}
    for r in records:
        v = r.get("verdict", "unknown")
        c = r.get("category", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1
        categories[c] = categories.get(c, 0) + 1

    return {
        "total_pairs": total_queue,
        "total_annotated": len(records),
        "verdicts": verdicts,
        "categories": categories
    }

@app.get("/api/annotation/export")
def export_annotations():
    ann_path = Path("data/annotations.jsonl")
    records = []
    if ann_path.exists():
        with open(ann_path, "r") as fp:
            for line in fp:
                if line.strip():
                    records.append(json.loads(line))
    return {"count": len(records), "annotations": records}

# ==========================================
# EXP-006 Region & Gemstone Annotation Routes
# ==========================================

class RegionSaveRequest(BaseModel):
    image_id: str
    category: str
    box: List[float] = Field(..., min_length=4, max_length=4)
    label: str = "pendant_stone"
    source: str = "human_verified"
    annotator: str = "senior_cv_lead"
    notes: Optional[str] = ""

class RegionProposeRequest(BaseModel):
    image_id: str
    category: Optional[str] = None

@app.get("/region-annotate")
def serve_region_annotate():
    return FileResponse("frontend/region_annotate.html")

@app.get("/api/regions/list")
def list_region_items():
    reg_file = Path("data/region_annotations.json")
    existing_ann = {}
    if reg_file.exists():
        with open(reg_file, "r") as fp:
            data = json.load(fp)
            existing_ann = data.get("annotations", {})
            
    bench_file = Path("data/evaluation/benchmark_v3_balanced.json")
    items = []
    if bench_file.exists():
        with open(bench_file, "r") as fp:
            bench = json.load(fp)
            for q in bench.get("same_sku_queries", []):
                qid = q["query_id"]
                cat = q["category"]
                qpath = Path(q["query_path"])
                fname = qpath.name
                is_query_ring = "queries/rings" in str(qpath)
                url = f"/media/query_ring/{fname}" if is_query_ring else f"/media/{cat}/{fname}"
                existing = existing_ann.get(qid)
                items.append({
                    "id": qid,
                    "category": cat,
                    "url": url,
                    "width": 262,
                    "height": 262,
                    "existing_region": {
                        "box": existing["box"],
                        "label": existing.get("label", "pendant_stone"),
                        "source": existing.get("source", "human_verified"),
                        "notes": existing.get("notes", "")
                    } if existing else None
                })
    return {"total": len(items), "items": items}

@app.get("/api/regions/{image_id}")
def get_region_item(image_id: str):
    reg_file = Path("data/region_annotations.json")
    if reg_file.exists():
        with open(reg_file, "r") as fp:
            data = json.load(fp)
            ann = data.get("annotations", {}).get(image_id)
            if ann:
                return ann
    raise HTTPException(status_code=404, detail=f"No region annotation found for {image_id}")

@app.post("/api/regions/save")
def save_region_annotation(req: RegionSaveRequest):
    reg_file = Path("data/region_annotations.json")
    data = {"version": "1.0", "annotations": {}}
    if reg_file.exists():
        with open(reg_file, "r") as fp:
            data = json.load(fp)
            
    box = [round(float(x), 4) for x in req.box]
    record = {
        "image_id": req.image_id,
        "category": req.category,
        "box": box,
        "label": req.label,
        "source": req.source,
        "annotator": req.annotator,
        "notes": req.notes
    }
    data["annotations"][req.image_id] = record
    data["total_annotated"] = len(data["annotations"])
    reg_file.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_file, "w") as fp:
        json.dump(data, fp, indent=2)
        
    return {"status": "success", "message": f"Region saved for {req.image_id}"}

@app.post("/api/regions/propose")
def propose_region(req: RegionProposeRequest):
    clean_id = req.image_id
    for pfx in ["q_neck_", "q_ring_"]:
        if clean_id.startswith(pfx):
            clean_id = clean_id[len(pfx):]
            
    found_path = None
    cat = req.category or "necklace"
    for ext in [".jpg", ".png", ".jpeg"]:
        p1 = Path("Jewellery_Data/necklace") / f"{clean_id}{ext}"
        if p1.exists():
            found_path, cat = p1, "necklace"
            break
        p2 = Path("Jewellery_Data/ring") / f"{clean_id}{ext}"
        if p2.exists():
            found_path, cat = p2, "ring"
            break
        p3 = Path("data/evaluation/queries/rings") / f"{req.image_id}{ext}"
        if p3.exists():
            found_path, cat = p3, "ring"
            break
        p4 = Path("data/evaluation/queries/rings") / f"{clean_id}{ext}"
        if p4.exists():
            found_path, cat = p4, "ring"
            break

    if not found_path or not found_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found for ID {req.image_id}")

    try:
        img = Image.open(found_path).convert("RGB")
        loc = get_localizer()
        res = loc.locate(img, category=cat)
        return {
            "box": [round(float(x), 4) for x in res.box],
            "label": res.label,
            "confidence": float(res.confidence),
            "method": res.method,
            "is_fallback": bool(res.is_fallback)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proposal failed: {str(e)}")


# Mount frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("frontend/index.html")

