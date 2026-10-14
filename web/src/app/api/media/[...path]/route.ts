import { NextRequest, NextResponse } from "next/server";
import { BACKEND_URL, getInternalProxySecret } from "@/lib/config";

export async function GET(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const pathParts = params.path;
  if (!pathParts || pathParts.length === 0) {
    return new NextResponse("Not Found", { status: 404 });
  }

  // Prevent directory traversal
  for (const part of pathParts) {
    if (part.includes("..") || part.includes("/") || part.includes("\\")) {
      return new NextResponse("Invalid Path", { status: 400 });
    }
  }

  const subpath = pathParts.join("/");
  const proxySecret = getInternalProxySecret();
  const headers: Record<string, string> = {};
  if (proxySecret) {
    headers["X-Proxy-Secret"] = proxySecret;
  }

  try {
    const backendRes = await fetch(`${BACKEND_URL}/media/${subpath}`, {
      headers,
      cache: "force-cache",
    });

    if (!backendRes.ok) {
      return new NextResponse("Not Found", { status: backendRes.status });
    }

    const contentType = backendRes.headers.get("content-type") || "image/jpeg";
    const buffer = await backendRes.arrayBuffer();

    return new NextResponse(buffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400, stale-while-revalidate=604800",
      },
    });
  } catch (error) {
    return new NextResponse(`Proxy error: ${(error as Error).message}`, {
      status: 502,
    });
  }
}
