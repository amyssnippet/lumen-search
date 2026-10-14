import { NextResponse } from "next/server";
import { BACKEND_URL, getInternalProxySecret } from "@/lib/config";
import { getSessionToken, clearSessionToken } from "@/lib/session";

export async function GET() {
  const token = getSessionToken();
  if (!token) {
    return NextResponse.json({ authenticated: false });
  }

  try {
    const proxySecret = getInternalProxySecret();
    const headers: Record<string, string> = {
      Authorization: `Bearer ${token}`,
    };
    if (proxySecret) {
      headers["X-Proxy-Secret"] = proxySecret;
    }

    const resp = await fetch(`${BACKEND_URL}/api/auth/me`, {
      headers,
      cache: "no-store",
    });

    if (!resp.ok) {
      if (resp.status === 401 || resp.status === 403) {
        clearSessionToken();
        return NextResponse.json({ authenticated: false });
      }
      return NextResponse.json(
        { authenticated: false, error: "Failed to verify session" },
        { status: resp.status }
      );
    }

    const user = await resp.json();
    return NextResponse.json({ authenticated: true, user });
  } catch (error) {
    return NextResponse.json(
      { authenticated: false, error: (error as Error).message },
      { status: 502 }
    );
  }
}
