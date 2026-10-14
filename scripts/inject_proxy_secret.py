#!/usr/bin/env python3
"""Milestone PH-013: Dynamic Reverse Proxy Secret Injector & Rotation Tool.

Securely generates or injects the internal reverse-proxy shared secret
into uncommitted NGINX include files (proxy_secret.inc) and environment files.

Guarantees:
1. Proxy secret is never hardcoded in tracked NGINX configuration files.
2. The compromised literal 'jewellery_internal_edge_proxy_secret_2026' is strictly blocked from generation or injection.
3. Secret files are created with restricted file permissions (0600).
4. Generates cryptographically secure 64-character hex tokens when generating new secrets.
"""

import os
import secrets
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BANNED_COMPROMISED_SECRETS = {
    "jewellery_internal_edge_proxy_secret_2026",
    "dev_proxy_secret_literal",
    "staging_proxy_secret_insecure",
}

STAGING_INCLUDE_PATH = REPO_ROOT / "deploy" / "staging" / "nginx" / "conf.d" / "proxy_secret.inc"
PRODUCTION_INCLUDE_PATH = REPO_ROOT / "deploy" / "production" / "nginx" / "conf.d" / "proxy_secret.inc"
SECRET_ENV_FILE = REPO_ROOT / "deploy" / "staging" / ".proxy_secret"


def generate_token(bytes_len: int = 32) -> str:
    """Generate cryptographically secure high-entropy token."""
    return f"sec_prx_{secrets.token_hex(bytes_len)}"


def get_or_create_secret(custom_secret: str = None) -> str:
    """Retrieve secret from environment, file, or generate a fresh secure token."""
    secret = custom_secret or os.environ.get("INTERNAL_PROXY_SECRET")
    
    if not secret and SECRET_ENV_FILE.exists():
        secret = SECRET_ENV_FILE.read_text(encoding="utf-8").strip()

    if not secret:
        # Generate fresh high-entropy rotated credential (256-bit entropy, 64 hex chars)
        secret = generate_token(32)
        SECRET_ENV_FILE.write_text(secret, encoding="utf-8")
        SECRET_ENV_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)

    if secret in BANNED_COMPROMISED_SECRETS:
        raise ValueError(f"FATAL: Attempted to inject known compromised secret: {secret}")

    if len(secret) < 32:
        raise ValueError("FATAL: INTERNAL_PROXY_SECRET must be at least 32 characters long.")

    return secret


def write_nginx_secret_include(target_path: Path, secret: str):
    """Write NGINX include snippet setting the X-Proxy-Secret header."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# Auto-generated reverse proxy secret include — DO NOT COMMIT\nproxy_set_header X-Proxy-Secret \"{secret}\";\n"
    target_path.write_text(content, encoding="utf-8")
    target_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"Wrote NGINX proxy secret include to {target_path} (mode 0600)")


def main():
    secret = get_or_create_secret()
    write_nginx_secret_include(STAGING_INCLUDE_PATH, secret)
    write_nginx_secret_include(PRODUCTION_INCLUDE_PATH, secret)
    print("Proxy secret injection completed successfully.")
    print("Masked secret preview:", secret[:10] + "..." + secret[-6:])


if __name__ == "__main__":
    main()
