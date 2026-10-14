import { NextRequest, NextResponse } from "next/server";
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

export async function GET(req: NextRequest) {
  try {
    const token = await getValidToken();
    if (!token) {
      return NextResponse.json(
        { error: "Authentication required to browse catalogue." },
        { status: 401 }
      );
    }

    const { searchParams } = new URL(req.url);
    const category = searchParams.get("category") || "";
    const sku = searchParams.get("sku") || "";
    const limit = searchParams.get("limit") || "50";
    const offset = searchParams.get("offset") || "0";

    const params = new URLSearchParams();
    if (category && category !== "all") params.set("category", category);
    if (sku) params.set("sku", sku);
    params.set("limit", limit);
    params.set("offset", offset);

    const proxySecret = getInternalProxySecret();
    const headers: Record<string, string> = {
      Authorization: `Bearer ${token}`,
    };
    if (proxySecret) {
      headers["X-Proxy-Secret"] = proxySecret;
    }

    const backendRes = await fetch(`${BACKEND_URL}/api/products?${params.toString()}`, {
      headers,
      cache: "no-store",
    });

    if (!backendRes.ok) {
      const err = await backendRes.json().catch(() => ({ detail: "Failed to fetch products" }));
      return NextResponse.json(
        { error: err.detail || "Failed to fetch products" },
        { status: backendRes.status }
      );
    }

    const data = await backendRes.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: `Internal gateway error: ${(error as Error).message}` },
      { status: 502 }
    );
  }
}
