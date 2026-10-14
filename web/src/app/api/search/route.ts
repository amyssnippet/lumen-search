import { NextRequest, NextResponse } from "next/server";
import { BACKEND_URL, getInternalProxySecret, DEMO_USER } from "@/lib/config";
import { getSessionToken, setSessionToken } from "@/lib/session";

async function getValidToken(): Promise<string | null> {
  let token = getSessionToken();
  if (token) return token;

  // Auto-authenticate with demo user credentials if no session exists
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

export async function POST(req: NextRequest) {
  const startTime = performance.now();

  try {
    const token = await getValidToken();
    if (!token) {
      return NextResponse.json(
        { error: "Authentication required. Backend service unavailable or credentials invalid." },
        { status: 401 }
      );
    }

    const formData = await req.formData();
    const proxySecret = getInternalProxySecret();

    const headers: Record<string, string> = {
      Authorization: `Bearer ${token}`,
    };
    if (proxySecret) {
      headers["X-Proxy-Secret"] = proxySecret;
    }

    const backendRes = await fetch(`${BACKEND_URL}/api/search`, {
      method: "POST",
      headers,
      body: formData,
      cache: "no-store",
    });

    if (!backendRes.ok) {
      const errBody = await backendRes.json().catch(() => ({ detail: "Search request failed" }));
      return NextResponse.json(
        { error: errBody.detail || "Search request failed" },
        { status: backendRes.status }
      );
    }

    const data = await backendRes.json();
    const client_e2e_ms = Math.round(performance.now() - startTime);
    data.client_e2e_ms = client_e2e_ms;

    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: `Internal gateway error: ${(error as Error).message}` },
      { status: 502 }
    );
  }
}
