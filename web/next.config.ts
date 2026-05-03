import type { NextConfig } from "next";

// Note: the Spellbook playground proxy lives at /api/sb/[...path]/route.ts
// rather than as a Next rewrite — the timeline endpoint can take 45s+ on the
// cold path and the dev rewrite proxy times out before that.

const nextConfig: NextConfig = {};

export default nextConfig;
