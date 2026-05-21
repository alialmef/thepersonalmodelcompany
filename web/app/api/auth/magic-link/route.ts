import { NextResponse } from "next/server";
import { z } from "zod";
import { sendMagicLink } from "@/lib/auth";

/**
 * POST /api/auth/magic-link
 * Body: { email: string }
 *
 * Sends a sign-in link to the email. Always returns 200 — we don't leak
 * whether the email exists in our users table.
 *
 * In dev mode (no RESEND_API_KEY set), the response includes the link in
 * `dev.link` so the developer can click it directly without setting up
 * email infrastructure.
 */

export const runtime = "nodejs";

const Body = z.object({ email: z.string().email() });

export async function POST(req: Request) {
  let parsed: { email: string };
  try {
    parsed = Body.parse(await req.json());
  } catch {
    return NextResponse.json({ error: "invalid email" }, { status: 400 });
  }

  try {
    const { devLink } = await sendMagicLink(parsed.email);
    return NextResponse.json({
      ok: true,
      ...(devLink ? { dev: { link: devLink } } : {}),
    });
  } catch (e) {
    console.error("[magic-link] failed:", e);
    // Still return 200 so callers can't probe delivery failures.
    return NextResponse.json({ ok: true });
  }
}
