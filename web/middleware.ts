import { NextResponse, type NextRequest } from "next/server";

/**
 * Auth guard for app routes.
 *
 * Any request to a protected app route without the session cookie is
 * redirected to /sign-in?next=<original-path> so the user lands back on the
 * page they wanted after sign-in.
 *
 * We do NOT validate the session in middleware — that would mean a DB query
 * on every request. Cookie presence is enough at this layer; routes that
 * need the real user identity call /api/auth/me (or read the cookie via
 * getSessionByToken on the server).
 *
 * Protected routes (Mac app surfaces): /welcome /connect /curate /train
 * /eval /first-meeting /chat /actions
 *
 * Public routes (marketing + auth): / /sign-in /download /privacy /terms
 * /contact /other-platforms /api/*
 */

const PROTECTED_PATHS = [
  "/welcome",
  "/connect",
  "/curate",
  "/train",
  "/eval",
  "/first-meeting",
  "/chat",
  "/actions",
];

const SESSION_COOKIE = "pmc_session";

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  const isProtected = PROTECTED_PATHS.some(
    (p) => pathname === p || pathname.startsWith(`${p}/`),
  );
  if (!isProtected) return NextResponse.next();

  const hasSession = Boolean(req.cookies.get(SESSION_COOKIE)?.value);
  if (hasSession) return NextResponse.next();

  const signIn = new URL("/sign-in", req.url);
  signIn.searchParams.set("next", pathname + req.nextUrl.search);
  return NextResponse.redirect(signIn);
}

// Limit matcher so we don't run on static assets, _next, favicon, etc.
export const config = {
  matcher: [
    "/welcome",
    "/welcome/:path*",
    "/connect",
    "/connect/:path*",
    "/curate",
    "/curate/:path*",
    "/train",
    "/train/:path*",
    "/eval",
    "/eval/:path*",
    "/first-meeting",
    "/first-meeting/:path*",
    "/chat",
    "/chat/:path*",
    "/actions",
    "/actions/:path*",
  ],
};
