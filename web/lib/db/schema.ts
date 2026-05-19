import {
  pgTable,
  text,
  timestamp,
  uuid,
  jsonb,
  integer,
  boolean,
  index,
} from "drizzle-orm/pg-core";

/**
 * Web-app database schema.
 *
 * The PMC Python backend has its own per-user filesystem storage for raw data,
 * curated datasets, adapters, and bundles. This database holds only what the
 * web app needs: user accounts, sessions, magic-link tokens, and a pointer to
 * each user's PMC tenant.
 */

export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  email: text("email").notNull().unique(),
  name: text("name"),
  emailVerifiedAt: timestamp("email_verified_at"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  // The user_id used in the PMC backend's UserStore / ArtifactStore / etc.
  // We default to the same uuid as the web user id for clean joins.
  pmcUserId: text("pmc_user_id").notNull(),
});

export const sessions = pgTable(
  "sessions",
  {
    id: text("id").primaryKey(), // session token
    userId: uuid("user_id")
      .references(() => users.id, { onDelete: "cascade" })
      .notNull(),
    expiresAt: timestamp("expires_at").notNull(),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => ({
    userIdx: index("sessions_user_idx").on(table.userId),
  }),
);

export const magicLinks = pgTable(
  "magic_links",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    email: text("email").notNull(),
    token: text("token").notNull().unique(),
    expiresAt: timestamp("expires_at").notNull(),
    usedAt: timestamp("used_at"),
    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => ({
    tokenIdx: index("magic_links_token_idx").on(table.token),
  }),
);

/**
 * Pipeline jobs the user has kicked off (ingest / curate / train / eval).
 * The actual orchestration runs in the Python backend; this table just gives
 * us a place to attach UI state (last viewed, dismissed, etc.) without
 * round-tripping to the backend on every page render.
 */
export const jobs = pgTable(
  "jobs",
  {
    id: text("id").primaryKey(), // matches Python orchestrator's job_id
    userId: uuid("user_id")
      .references(() => users.id, { onDelete: "cascade" })
      .notNull(),
    kind: text("kind").notNull(), // "ingest" | "curate" | "train" | "eval" | "pipeline"
    status: text("status").notNull(), // "queued" | "running" | "completed" | "failed"
    progress: integer("progress").default(0).notNull(), // 0-100 best-effort
    submittedAt: timestamp("submitted_at").defaultNow().notNull(),
    completedAt: timestamp("completed_at"),
    summary: jsonb("summary"), // most recent PipelineResult / similar
    dismissed: boolean("dismissed").default(false).notNull(),
  },
  (table) => ({
    userIdx: index("jobs_user_idx").on(table.userId),
  }),
);

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
export type Session = typeof sessions.$inferSelect;
export type MagicLink = typeof magicLinks.$inferSelect;
export type Job = typeof jobs.$inferSelect;
