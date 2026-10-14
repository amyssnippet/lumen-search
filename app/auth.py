"""Authentication, API Key verification, JWT issuance, and RBAC authorization."""

import datetime
import hashlib
import os
import secrets
from typing import Optional, Set, List
import jwt
from fastapi import Header, HTTPException, Request, Depends, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from engine.db.session import get_db
from engine.db.models import ApiKey, Tenant, User

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jvr_enterprise_secret_signing_key_2026_dev")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


class AuthenticatedIdentity(BaseModel):
    """Authoritative identity resolved from verified credentials."""
    tenant_id: str
    actor_id: str
    actor_type: str  # "api_key", "user", "system"
    scopes: Set[str]
    tier: str = "standard"
    rate_limit_rpm: int = 300
    rate_limit_burst: int = 30
    user_email: Optional[str] = None


def hash_api_key(raw_key: str) -> str:
    """Compute SHA-256 hash of API key."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def generate_new_api_key(prefix: str = "jvr") -> tuple[str, str, str]:
    """Generate a cryptographically secure API key.
    
    Returns:
        (raw_key, key_hash, key_prefix)
    """
    token = secrets.token_urlsafe(32)
    raw_key = f"{prefix}_{token}"
    key_hash = hash_api_key(raw_key)
    key_prefix = raw_key[:12]
    return raw_key, key_hash, key_prefix


def create_access_token(
    tenant_id: str,
    actor_id: str,
    scopes: List[str],
    user_email: Optional[str] = None,
    expires_delta: Optional[datetime.timedelta] = None
) -> str:
    """Issue a signed JWT bearer token."""
    now = datetime.datetime.utcnow()
    expire = now + (expires_delta or datetime.timedelta(hours=JWT_EXPIRATION_HOURS))
    payload = {
        "sub": actor_id,
        "tenant_id": tenant_id,
        "scopes": scopes,
        "email": user_email,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Validate and decode a JWT bearer token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


def verify_api_key(raw_key: str, db: Session) -> Optional[AuthenticatedIdentity]:
    """Verify an API key and resolve the authoritative tenant identity."""
    kh = hash_api_key(raw_key)
    api_key_rec = (
        db.query(ApiKey)
        .join(Tenant, ApiKey.tenant_id == Tenant.id)
        .filter(ApiKey.key_hash == kh)
        .first()
    )

    if not api_key_rec:
        return None

    if not api_key_rec.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key is disabled")

    if api_key_rec.expires_at and api_key_rec.expires_at < datetime.datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has expired")

    tenant = api_key_rec.tenant
    if not tenant or not tenant.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant organization is deactivated")

    # Update last_used_at throttled to once per 60s to prevent row-lock contention under concurrency
    now = datetime.datetime.utcnow()
    if not api_key_rec.last_used_at or (now - api_key_rec.last_used_at).total_seconds() > 60:
        api_key_rec.last_used_at = now
        try:
            db.commit()
        except Exception:
            db.rollback()

    scopes_set = set(s.strip() for s in api_key_rec.scopes.split(",") if s.strip())

    return AuthenticatedIdentity(
        tenant_id=tenant.id,
        actor_id=api_key_rec.id,
        actor_type="api_key",
        scopes=scopes_set,
        tier=tenant.tier,
        rate_limit_rpm=tenant.rate_limit_rpm,
        rate_limit_burst=tenant.rate_limit_burst,
    )


async def get_current_identity(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    db: Session = Depends(get_db)
) -> AuthenticatedIdentity:
    """Authoritative credential resolver.
    
    Inspects X-API-Key or Authorization Bearer header.
    Never trusts caller-provided tenant_id claims without verification.
    """
    # 1. Inspect API Key Header
    if x_api_key:
        identity = verify_api_key(x_api_key, db)
        if not identity:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-API-Key credentials"
            )
        # Check against spoofed tenant_id in query or body
        _assert_no_tenant_spoofing(request, identity.tenant_id)
        return identity

    # 2. Inspect Authorization Header (Bearer JWT)
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            payload = decode_access_token(token)
            tenant_id = payload.get("tenant_id")
            actor_id = payload.get("sub")
            scopes_list = payload.get("scopes", [])
            email = payload.get("email")

            tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
            if not tenant or not tenant.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Tenant organization is invalid or inactive"
                )

            identity = AuthenticatedIdentity(
                tenant_id=tenant.id,
                actor_id=actor_id,
                actor_type="user",
                scopes=set(scopes_list),
                tier=tenant.tier,
                rate_limit_rpm=tenant.rate_limit_rpm,
                rate_limit_burst=tenant.rate_limit_burst,
                user_email=email,
            )
            _assert_no_tenant_spoofing(request, identity.tenant_id)
            return identity
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header scheme. Expected 'Bearer <token>'"
            )

    # 3. Missing credentials
    require_auth = os.environ.get("REQUIRE_AUTH", "false").lower() in ("true", "1")
    if require_auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide 'X-API-Key' or 'Authorization: Bearer <token>' header.",
            headers={"WWW-Authenticate": "Bearer, ApiKey"}
        )

    # In permissive development/test mode, resolve default tenant
    identity = AuthenticatedIdentity(
        tenant_id="default_tenant",
        actor_id="dev_anonymous_actor",
        actor_type="system",
        scopes={"search:read", "catalog:write", "tenant:admin", "ops:admin"},
        tier="standard",
        rate_limit_rpm=300,
        rate_limit_burst=30,
        user_email="dev@jewellery.internal"
    )
    _assert_no_tenant_spoofing(request, identity.tenant_id)
    return identity


def _assert_no_tenant_spoofing(request: Request, authoritative_tenant_id: str):
    """Enforce security invariant: reject cross-tenant spoofing attempts."""
    # Check query params
    caller_tenant_param = request.query_params.get("tenant_id")
    if caller_tenant_param and caller_tenant_param != authoritative_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cross-tenant access denied: caller authenticated as '{authoritative_tenant_id}' cannot operate on tenant '{caller_tenant_param}'"
        )
    # Check header
    header_tenant = request.headers.get("X-Tenant-ID")
    if header_tenant and header_tenant != authoritative_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cross-tenant access denied: caller authenticated as '{authoritative_tenant_id}' cannot spoof tenant '{header_tenant}'"
        )


def require_scope(required_scope: str):
    """Dependency factory checking that the authenticated caller has the specified scope."""
    def scope_checker(identity: AuthenticatedIdentity = Depends(get_current_identity)) -> AuthenticatedIdentity:
        if required_scope not in identity.scopes and "ops:admin" not in identity.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: credential lacks required scope '{required_scope}'"
            )
        return identity
    return scope_checker
