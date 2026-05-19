import type { NextConfig } from "next";

// When building inside the Tauri desktop shell, we produce a static export
// (no Next.js server, no API routes). The desktop app talks directly to the
// Python FastAPI backend; route handlers in /app/api/* are only used by the
// web/marketing context.
const isTauriBuild = process.env.NEXT_PUBLIC_TAURI_BUILD === "true";

const config: NextConfig = {
  reactStrictMode: true,
  ...(isTauriBuild && {
    output: "export",
    images: { unoptimized: true },
    // Trailing slashes make Tauri's webview happy when serving static files
    // from the bundle. Don't enable for the regular web build.
    trailingSlash: true,
  }),
  async rewrites() {
    // Rewrites don't apply in static-export mode anyway; skip them in the
    // Tauri build to avoid Next.js warnings.
    if (isTauriBuild) return [];
    const pmcApi = process.env.PMC_API_URL ?? "http://localhost:8000";
    return [
      {
        source: "/pmc-api/:path*",
        destination: `${pmcApi}/:path*`,
      },
    ];
  },
};

export default config;
