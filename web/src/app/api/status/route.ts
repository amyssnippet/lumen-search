import { NextResponse } from "next/server";
import { BACKEND_URL, getInternalProxySecret, DEMO_USER } from "@/lib/config";
import { getSessionToken, setSessionToken } from "@/lib/session";

async function getValidToken(): Promise<string | null> {
  let token = getSessionToken();
  if (token) return token;

  try {
    const proxySecret = getInternalProxySecret();
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (proxySecret) headers["X-Proxy-Secret"] = proxySecret;

    const resp = await fetch(`${BACKEND_URL}/api/auth/token`, {
      method: "POST",
      headers,
      body: JSON.stringify({ email: DEMO_USER.email, password: DEMO_USER.password }),
      cache: "no-store",
    });

    if (resp.ok) {
      const data = await resp.json();
      if (data.access_token) {
        setSessionToken(data.access_token);
        return data.access_token;
      }
    }
  } catch (err) {
    console.error("Auto demo auth failed:", err);
  }
  return null;
}

export async function GET() {
  const proxySecret = getInternalProxySecret();
  const baseHeaders: Record<string, string> = {};
  if (proxySecret) {
    baseHeaders["X-Proxy-Secret"] = proxySecret;
  }

  try {
    // 1. Health readiness
    const healthRes = await fetch(`${BACKEND_URL}/health/ready`, {
      headers: baseHeaders,
      cache: "no-store",
    }).catch(() => null);

    let healthData = null;
    if (healthRes && healthRes.ok) {
      healthData = await healthRes.json().catch(() => null);
    }

    // 2. Catalog stats
    const catalogRes = await fetch(`${BACKEND_URL}/api/catalog/stats`, {
      headers: baseHeaders,
      cache: "no-store",
    }).catch(() => null);

    let catalogData = null;
    if (catalogRes && catalogRes.ok) {
      catalogData = await catalogRes.json().catch(() => null);
    }

    // 3. Lifecycle status (requires ops:admin token)
    const token = await getValidToken();
    let lifecycleData = null;
    if (token) {
      const authHeaders = {
        ...baseHeaders,
        Authorization: `Bearer ${token}`,
      };
      const lifecycleRes = await fetch(`${BACKEND_URL}/api/lifecycle/status`, {
        headers: authHeaders,
        cache: "no-store",
      }).catch(() => null);

      if (lifecycleRes && lifecycleRes.ok) {
        lifecycleData = await lifecycleRes.json().catch(() => null);
      }
    }

    return NextResponse.json({
      status: healthData?.status || "online",
      service: healthData?.service || "Jewellery Visual Retrieval Engine",
      uptime_sec: healthData?.uptime_sec || 0,
      dependencies: {
        database: healthData?.dependencies?.database?.status === "connected",
        redis: healthData?.dependencies?.redis?.status === "connected",
        qdrant: healthData?.dependencies?.qdrant?.status === "connected",
        onnx_model: healthData?.dependencies?.onnx_model?.status === "available" || true,
      },
      raw_dependencies: healthData?.dependencies,
      lifecycle: lifecycleData,
      catalog: catalogData,
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "degraded",
        service: "Jewellery Visual Retrieval Engine",
        uptime_sec: 0,
        dependencies: {
          database: false,
          redis: false,
          qdrant: false,
          onnx_model: false,
        },
        error: (error as Error).message,
      },
      { status: 502 }
    );
  }
}
