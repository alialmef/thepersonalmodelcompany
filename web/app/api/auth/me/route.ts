import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { SESSION_COOKIE, getSessionByToken } from "@/lib/auth";

/**
 * GET /api/auth/me
 *
 * Returns the current user JSON, or 401 if not signed in. Used by the
 * client-side `useUser()` hook so every app route can read its own user
 * without each page being a Server Component.
 */

export const runtime = "nodejs";

export async function GET() {
  const jar = await cookies();
  const token = jar.get(SESSION_COOKIE)?.value;
  const user = await getSessionByToken(token);
  if (!user) {
    return NextResponse.json({ error: "not_authenticated" }, { status: 401 });
  }
  return NextResponse.json({
    id: user.id,
    email: user.email,
    pmcUserId: user.pmcUserId,
  });
}
