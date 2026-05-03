import { NextRequest } from "next/server";

const SPELLBOOK_SERVER =
  process.env.SPELLBOOK_SERVER ?? "http://localhost:8101";

// The playground timeline endpoint can take ~45s on a cold replay of large
// sessions. Keep a generous ceiling so the route doesn't drop slow requests.
const PROXY_TIMEOUT_MS = 5 * 60 * 1000;

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

async function proxy(req: NextRequest, path: string[]): Promise<Response> {
  const url = new URL(req.url);
  const target = `${SPELLBOOK_SERVER}/${path.join("/")}${url.search}`;

  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), PROXY_TIMEOUT_MS);

  const init: RequestInit = {
    method: req.method,
    headers: { "content-type": req.headers.get("content-type") ?? "application/json" },
    signal: ctrl.signal,
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }

  try {
    const res = await fetch(target, init);
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return new Response(
      JSON.stringify({ error: `proxy: ${message}`, target }),
      { status: 502, headers: { "content-type": "application/json" } },
    );
  } finally {
    clearTimeout(t);
  }
}

interface RouteContext {
  params: Promise<{ path: string[] }>;
}

export async function GET(req: NextRequest, ctx: RouteContext) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function POST(req: NextRequest, ctx: RouteContext) {
  const { path } = await ctx.params;
  return proxy(req, path);
}
