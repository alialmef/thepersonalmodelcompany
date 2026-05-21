/**
 * Magic-link auth.
 *
 * Flow:
 *   1. User enters email on /sign-in
 *   2. sendMagicLink(email) — inserts magic_links row + emails the link
 *   3. User clicks the link → /api/auth/verify?token=xxx
 *   4. verifyMagicLink(token) — consumes the row, creates a session
 *   5. The route handler sets the pmc_session cookie + redirects to /
 *   6. Subsequent requests carry the cookie; getSessionByToken() returns the user
 *
 * Storage:
 *   - Web app: Postgres tables (users, sessions, magic_links) in db/schema.ts
 *   - PMC backend: filesystem under storage_root/users/<pmcUserId>/
 *     The web `users.pmcUserId` is the user_id passed to the FastAPI endpoints.
 *
 * Security:
 *   - Tokens are 32 random bytes (64 hex). Not guessable.
 *   - Magic links expire in 15 min and are single-use (used_at set on consume).
 *   - Sessions expire in 30 days. No refresh; expired users just get a new link.
 *   - Cookies are HttpOnly, Secure (in production), SameSite=Lax.
 *
 * Dev mode: if RESEND_API_KEY isn't set, we log the magic-link URL to the
 * server console instead of sending email. Useful for local dev with no DNS.
 */

import { db } from "./db";
import { magicLinks, sessions, users } from "./db/schema";
import { and, eq, gt, isNull } from "drizzle-orm";
import { Resend } from "resend";

const SESSION_COOKIE = "pmc_session";
const SESSION_TTL_MS = 1000 * 60 * 60 * 24 * 30; // 30 days
const MAGIC_LINK_TTL_MS = 1000 * 60 * 15; // 15 minutes

export type SessionUser = {
  id: string;
  email: string;
  pmcUserId: string;
};

// ---------------------------------------------------------------------------
// User upsert — called from verifyMagicLink; the email is treated as the
// canonical identity. pmcUserId is generated at first sight and is the id
// passed to the Python backend's per-user filesystem.
// ---------------------------------------------------------------------------

export async function getOrCreateUser(email: string): Promise<SessionUser> {
  const normalized = email.trim().toLowerCase();
  const existing = await db
    .select()
    .from(users)
    .where(eq(users.email, normalized))
    .limit(1);

  if (existing.length > 0) {
    return {
      id: existing[0].id,
      email: existing[0].email,
      pmcUserId: existing[0].pmcUserId,
    };
  }

  const [created] = await db
    .insert(users)
    .values({
      email: normalized,
      pmcUserId: crypto.randomUUID(),
    })
    .returning();

  return {
    id: created.id,
    email: created.email,
    pmcUserId: created.pmcUserId,
  };
}

// ---------------------------------------------------------------------------
// Tokens — cryptographically random 32-byte hex strings.
// ---------------------------------------------------------------------------

function generateToken(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

// ---------------------------------------------------------------------------
// Send magic link
// ---------------------------------------------------------------------------

export async function sendMagicLink(email: string): Promise<{ devLink?: string }> {
  const normalized = email.trim().toLowerCase();
  const token = generateToken();
  const expiresAt = new Date(Date.now() + MAGIC_LINK_TTL_MS);

  await db.insert(magicLinks).values({
    email: normalized,
    token,
    expiresAt,
  });

  const baseUrl =
    process.env.NEXT_PUBLIC_SITE_URL ?? process.env.AUTH_URL ?? "http://localhost:3000";
  const verifyUrl = `${baseUrl}/api/auth/verify?token=${token}`;

  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    // Dev mode — log to server console so the developer can click it.
    // Also returned to the caller for /sign-in to render in dev.
    console.log(`[magic-link] (dev) ${normalized} -> ${verifyUrl}`);
    return { devLink: verifyUrl };
  }

  const from = process.env.EMAIL_FROM ?? "hello@thepersonalmodelcompany.com";
  const resend = new Resend(apiKey);

  await resend.emails.send({
    from,
    to: normalized,
    subject: "Sign in to The Personal Model Company",
    text: [
      "Tap the link below to sign in. It expires in 15 minutes.",
      "",
      verifyUrl,
      "",
      "If you didn't request this, ignore it. Nothing happens until you click.",
    ].join("\n"),
    html: emailHtml(verifyUrl),
  });

  return {};
}

function emailHtml(link: string): string {
  return `<!doctype html>
<html><body style="margin:0;padding:48px 24px;background:#FFFFFF;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','SF Pro Text',Helvetica,Arial,sans-serif;color:#1D1D1F;">
  <div style="max-width:480px;margin:0 auto;">
    <p style="font-size:14px;color:#6E6E73;margin:0 0 32px;">The Personal Model Company</p>
    <h1 style="font-size:28px;line-height:1.15;letter-spacing:-0.025em;font-weight:500;margin:0 0 16px;">Sign in.</h1>
    <p style="font-size:15px;line-height:1.55;color:#1D1D1F;margin:0 0 28px;">
      Tap the button below to sign in. It expires in 15 minutes.
    </p>
    <p style="margin:0 0 36px;">
      <a href="${link}" style="display:inline-block;padding:12px 22px;background:#1D1D1F;color:#FFFFFF;text-decoration:none;font-size:14px;font-weight:500;border-radius:999px;">Sign in</a>
    </p>
    <p style="font-size:12px;line-height:1.55;color:#6E6E73;margin:0 0 8px;">Or paste this link into your browser:</p>
    <p style="font-size:12px;line-height:1.55;color:#6E6E73;word-break:break-all;margin:0 0 32px;">
      <a href="${link}" style="color:#6E6E73;">${link}</a>
    </p>
    <p style="font-size:11px;color:#6E6E73;margin:0;">
      If you didn't request this, ignore it. Nothing happens until you click.
    </p>
  </div>
</body></html>`;
}

// ---------------------------------------------------------------------------
// Verify magic link → mark used, upsert user, create session.
// ---------------------------------------------------------------------------

export async function verifyMagicLink(token: string): Promise<{
  sessionToken: string;
  user: SessionUser;
} | null> {
  if (!token) return null;

  // Look up the link, fail closed on missing / expired / already-used.
  const rows = await db
    .select()
    .from(magicLinks)
    .where(
      and(
        eq(magicLinks.token, token),
        gt(magicLinks.expiresAt, new Date()),
        isNull(magicLinks.usedAt),
      ),
    )
    .limit(1);

  if (rows.length === 0) return null;
  const link = rows[0];

  // Mark used (single-use).
  await db
    .update(magicLinks)
    .set({ usedAt: new Date() })
    .where(eq(magicLinks.id, link.id));

  // Resolve / create the user, mark email verified.
  const user = await getOrCreateUser(link.email);
  await db
    .update(users)
    .set({ emailVerifiedAt: new Date() })
    .where(eq(users.id, user.id));

  // Create a session.
  const sessionToken = generateToken();
  const expiresAt = new Date(Date.now() + SESSION_TTL_MS);
  await db.insert(sessions).values({
    id: sessionToken,
    userId: user.id,
    expiresAt,
  });

  return { sessionToken, user };
}

// ---------------------------------------------------------------------------
// Session reader. Pass the cookie value (e.g. from cookies().get(...).value).
// Returns null when the cookie is missing, the session is gone, or expired.
// ---------------------------------------------------------------------------

export async function getSessionByToken(
  token: string | undefined,
): Promise<SessionUser | null> {
  if (!token) return null;
  const rows = await db
    .select({
      sessionExpiresAt: sessions.expiresAt,
      userId: users.id,
      userEmail: users.email,
      pmcUserId: users.pmcUserId,
    })
    .from(sessions)
    .innerJoin(users, eq(sessions.userId, users.id))
    .where(eq(sessions.id, token))
    .limit(1);

  if (rows.length === 0) return null;
  const row = rows[0];
  if (row.sessionExpiresAt < new Date()) return null;

  return {
    id: row.userId,
    email: row.userEmail,
    pmcUserId: row.pmcUserId,
  };
}

// ---------------------------------------------------------------------------
// Sign out
// ---------------------------------------------------------------------------

export async function destroySession(token: string | undefined): Promise<void> {
  if (!token) return;
  await db.delete(sessions).where(eq(sessions.id, token));
}

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

export {
  SESSION_COOKIE,
  SESSION_TTL_MS,
  MAGIC_LINK_TTL_MS,
  magicLinks,
  sessions,
  users,
};
