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

  // Caller may pass `next=` to override smart routing — used for protected-
  // route redirects (middleware sets next=<original-path>). Otherwise we
  // ask the backend what state this user is in and route accordingly.
  const explicit = url.searchParams.get("next");
  const explicitSafe =
    explicit && explicit.startsWith("/") && !explicit.startsWith("//")
      ? explicit
      : null;

  const result = await verifyMagicLink(token);
  if (!result) {
    return NextResponse.redirect(new URL("/sign-in?expired=1", req.url));
  }

  const next = explicitSafe ?? (await smartLanding(result.user.pmcUserId));
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

  await cookies();
  return response;
}

/**
 * Where should this user land after sign-in? Picked from backend state:
 *   - Has a registered adapter → /chat (their model is ready)
 *   - Has ingested raw items but no adapter → /train (resume the journey)
 *   - Fresh / no state → /welcome (the Hello moment)
 *
 * Backend unreachable falls back to /welcome — getting them through the
 * door beats sending them to a broken page.
 */
async function smartLanding(pmcUserId: string): Promise<string> {
  const apiUrl = process.env.PMC_API_URL ?? "http://localhost:8000";
  try {
    // Has the user been registered with an adapter?
    const modelRes = await fetch(
      `${apiUrl}/v1/models/${encodeURIComponent(pmcUserId)}`,
      { cache: "no-store", signal: AbortSignal.timeout(2000) },
    );
    if (modelRes.ok) return "/chat";

    // Otherwise check whether they've ingested any data.
    const statusRes = await fetch(
      `${apiUrl}/v1/users/${encodeURIComponent(pmcUserId)}/status`,
      { cache: "no-store", signal: AbortSignal.timeout(2000) },
    );
    if (statusRes.ok) {
      const data = (await statusRes.json()) as { raw_item_count?: number };
      if ((data.raw_item_count ?? 0) > 0) return "/train";
    }
  } catch {
    /* backend offline — fall through to /welcome */
  }
  return "/welcome";
}
