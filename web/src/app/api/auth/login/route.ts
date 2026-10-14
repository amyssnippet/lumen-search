import { NextRequest, NextResponse } from "next/server";
import { BACKEND_URL, getInternalProxySecret, DEMO_USER } from "@/lib/config";
import { setSessionToken } from "@/lib/session";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const email = body.email || DEMO_USER.email;
    const password = body.password || DEMO_USER.password;

    const proxySecret = getInternalProxySecret();
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (proxySecret) {
      headers["X-Proxy-Secret"] = proxySecret;
    }

    const resp = await fetch(`${BACKEND_URL}/api/auth/token`, {
      method: "POST",
      headers,
      body: JSON.stringify({ email, password }),
      cache: "no-store",
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: "Authentication failed" }));
      return NextResponse.json(
        { error: err.detail || "Invalid credentials" },
        { status: resp.status }
      );
    }

    const data = await resp.json();
    setSessionToken(data.access_token);

    return NextResponse.json({
      success: true,
      user: {
        email: email,
        tenant_id: data.tenant_id,
        role: data.role,
        scopes: data.scopes,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Backend service unavailable: ${(error as Error).message}` },
      { status: 502 }
    );
  }
}
