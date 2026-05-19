import { NextResponse } from "next/server";
import { z } from "zod";

/**
 * Stub for the magic-link endpoint.
 *
 * Full implementation lands when we wire Resend:
 *   1. Validate email
 *   2. Insert into magic_links table with a fresh token + 15min expiry
 *   3. Send email via Resend with link to /api/auth/verify?token=...
 *   4. Return 200 (always — don't leak whether the email exists)
 */

const Body = z.object({ email: z.string().email() });

export async function POST(req: Request) {
  let body;
  try {
    body = Body.parse(await req.json());
  } catch {
    return NextResponse.json({ error: "invalid email" }, { status: 400 });
  }

  // TODO: persist magic link + send via Resend
  console.log(`[magic-link] would send to ${body.email}`);

  // Always 200 — don't leak account existence.
  return NextResponse.json({ ok: true });
}
