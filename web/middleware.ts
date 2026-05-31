import { NextResponse, type NextRequest } from "next/server";

/**
 * Auth guard for app routes.
 *
 * The Tauri Mac-app surfaces (/welcome, /reading, /right-now,
 * /settings/agent, /knowledge-update, plus the dormant
 * /connect|/curate|/train|/eval|/first-meeting|/chat|/actions routes)
 * all run inside a webview that authenticates via a localStorage
 * session token, not a cookie. Cookie-based middleware redirects
 * would just bounce them to /sign-in on first launch and infinite-
 * loop, so this middleware doesn't touch those routes — the
 * client-side `useUser` hook handles redirects within the app.
 *
 * Today the middleware is effectively a no-op; it's kept around so
 * that if/when we add a marketing-site route that needs a real cookie
 * session, there's an obvious place to wire it.
 */

export function middleware(_req: NextRequest) {
  return NextResponse.next();
}

// Empty matcher = middleware never runs. Add specific patterns here
// (e.g. /admin/:path*) when there's a route that genuinely needs a
// server-side cookie check.
export const config = {
  matcher: [],
};
