/**
 * Until auth is wired (task #13), the web app operates against a single
 * hard-coded user_id. This lets the upload + status + chat flows work locally
 * against a real PMC backend without a session layer in the way.
 *
 * When auth lands, replace `DEMO_USER_ID` with `(await getSession()).pmcUserId`
 * in server components, or pass it through cookies for client components.
 */

export const DEMO_USER_ID = "demo";
