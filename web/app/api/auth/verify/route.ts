import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import {
  SESSION_COOKIE,
  SESSION_TTL_MS,
  verifyMagicLink,
} from "@/lib/auth";

/**
 * GET /api/auth/verify?token=<token>&next=<path>
 *
 * Endpoint linked from the magic-link email. Consumes the token (single
 * use, 15 min expiry), creates a session, sets the HttpOnly session cookie,
 * and redirects the user into the app.
 *
 * On failure (missing / expired / already used) we redirect to /sign-in
 * with ?expired=1 so the page can render the right message.
 *
 * `next` is an optional safe redirect target. We allow only same-origin
 * paths starting with "/" to prevent open-redirect abuse.
 */

export const runtime = "nodejs";

export async function GET(req: Request) {
  const url = new URL(req.url);
  const token = url.searchParams.get("token") ?? "";
  const nextParam = url.searchParams.get("next") ?? "/welcome";
  const next = nextParam.startsWith("/") && !nextParam.startsWith("//")
    ? nextParam
    : "/welcome";

  const result = await verifyMagicLink(token);
  if (!result) {
    return NextResponse.redirect(new URL("/sign-in?expired=1", req.url));
  }

  const response = NextResponse.redirect(new URL(next, req.url));
  const isProd = process.env.NODE_ENV === "production";

  response.cookies.set({
    name: SESSION_COOKIE,
    value: result.sessionToken,
    httpOnly: true,
    secure: isProd,
    sameSite: "lax",
    path: "/",
    maxAge: Math.floor(SESSION_TTL_MS / 1000),
  });

  // The `next/headers` cookies() store is preferred in Server Components,
  // but for a Route Handler the response cookies above are what stick on
  // the redirect. Importing cookies() ensures the route is treated as
  // dynamic in Next 15 builds.
  await cookies();

  return response;
}
