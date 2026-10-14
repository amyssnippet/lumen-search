import fs from "fs";
import path from "path";

// Internal backend URL (used by Next.js Server Route Handlers)
export const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.INTERNAL_BACKEND_URL ||
  "http://127.0.0.1:8001";

// Resolve internal proxy secret from environment or uncommitted secret file
export function getInternalProxySecret(): string {
  if (process.env.INTERNAL_PROXY_SECRET) {
    return process.env.INTERNAL_PROXY_SECRET;
  }

  // Attempt to read from deploy/demo/.proxy_secret or deploy/staging/.proxy_secret
  const candidatePaths = [
    path.resolve(process.cwd(), "../deploy/demo/.proxy_secret"),
    path.resolve(process.cwd(), "../deploy/staging/.proxy_secret"),
    path.resolve(process.cwd(), "./.proxy_secret"),
  ];

  for (const p of candidatePaths) {
    if (fs.existsSync(p)) {
      try {
        const secret = fs.readFileSync(p, "utf-8").trim();
        if (secret) return secret;
      } catch {
        // ignore read errors
      }
    }
  }

  return "";
}

export const DEMO_USER = {
  email: process.env.DEMO_USER_EMAIL || "admin@jewellery.internal",
  password: process.env.DEMO_USER_PASSWORD || "dev_password_2026",
  tenant: "default_tenant",
};
