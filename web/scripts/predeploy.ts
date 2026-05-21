#!/usr/bin/env bun
/**
 * Railway pre-deploy hook.
 *
 * Apply the latest Drizzle schema. If a clean `drizzle-kit push` succeeds we
 * move on. If it fails — typically because the DB is in a partial state
 * from a prior aborted push (Postgres rejects DROP NOT NULL on primary-key
 * columns and similar half-states) — we nuke the public schema and try once
 * more.
 *
 * The destructive fallback is intentional for now: the production database
 * has no real user data yet. When we have customers, swap this out for
 * `drizzle-kit migrate` with committed migration files (and remove the
 * DROP SCHEMA branch).
 *
 * Run via `bun run scripts/predeploy.ts`. Wired into railway.json's
 * preDeployCommand. Uses the `postgres` lib that's already in deps so we
 * don't need psql installed in the container.
 */

import { spawnSync } from "node:child_process";
import postgres from "postgres";

function tryPush(): boolean {
  console.log("[predeploy] attempting drizzle-kit push...");
  const result = spawnSync("bunx", ["drizzle-kit", "push", "--force"], {
    stdio: "inherit",
    env: process.env,
  });
  return result.status === 0;
}

async function resetSchema(): Promise<void> {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error("DATABASE_URL not set — cannot reset schema");
  }
  console.log("[predeploy] resetting public schema (destructive)...");
  const sql = postgres(url, { max: 1, onnotice: () => {} });
  try {
    await sql.unsafe("DROP SCHEMA IF EXISTS public CASCADE");
    await sql.unsafe("CREATE SCHEMA public");
  } finally {
    await sql.end({ timeout: 5 });
  }
  console.log("[predeploy] schema reset");
}

async function main(): Promise<void> {
  if (tryPush()) {
    console.log("[predeploy] push succeeded");
    return;
  }
  console.log("[predeploy] push failed — schema is in a bad state");
  await resetSchema();
  if (!tryPush()) {
    throw new Error("[predeploy] push failed even after schema reset");
  }
  console.log("[predeploy] push succeeded after reset");
}

main().catch((err) => {
  console.error("[predeploy] fatal:", err);
  process.exit(1);
});
