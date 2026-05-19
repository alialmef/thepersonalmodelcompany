/**
 * Auth — magic-link sign-in. Stubbed for the scaffold; full implementation
 * lands when we wire Resend + cookies + the /api/auth routes.
 *
 * V1 plan:
 *   sendMagicLink(email)   → create magic_links row, send email via Resend
 *   verifyMagicLink(token) → consume row, create session, set cookie
 *   getSession(req)        → read cookie, validate session row
 *   signOut(req)           → delete session row, clear cookie
 */

import { db } from "./db";
import { magicLinks, sessions, users } from "./db/schema";
import { eq } from "drizzle-orm";

const SESSION_COOKIE = "pmc_session";
const SESSION_TTL_MS = 1000 * 60 * 60 * 24 * 30; // 30 days
const MAGIC_LINK_TTL_MS = 1000 * 60 * 15; // 15 minutes

export type SessionUser = {
  id: string;
  email: string;
  pmcUserId: string;
};

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

export function generateToken(): string {
  // 32 bytes → 64 hex chars. Cryptographically random.
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export { SESSION_COOKIE, SESSION_TTL_MS, MAGIC_LINK_TTL_MS, magicLinks, sessions, users };
