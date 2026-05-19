# The Personal Model Company — Web

Next.js 15 frontend for thepersonalmodelcompany.com.

Sibling to the `pmc/` Python backend in this repo. Talks to it via HTTP
(`PMC_API_URL`) and proxies browser requests through `/pmc-api/*` rewrites.

## Stack

- **Next.js 15** App Router + React 19 + TypeScript
- **Tailwind CSS 4** (CSS-first config, no JS config file)
- **Drizzle** + Postgres for sessions and job state
- **Resend** for magic-link emails (wired in V1)
- **Bun** as package manager (`bun.lock` is committed)
- **Railway** for hosting, Railpack builder, Postgres sibling service
- **Cloudflare** for DNS (proxied), TLS via Railway's Let's Encrypt

Matches the conventions from `token-street`.

## Local dev

```bash
# 1. Install deps
bun install

# 2. Postgres — easiest with Docker
docker run --name pmc-pg -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 -d postgres:16

# 3. Env
cp .env.example .env.local
# edit DATABASE_URL if you used different creds

# 4. Migrate
bun run db:migrate

# 5. Start the PMC backend in a separate terminal
cd ..   # back to repo root
uv run --extra dev --extra serve uvicorn pmc.serve.api:create_app --factory

# 6. Start the web app
cd web
bun dev
```

Open [http://localhost:3000](http://localhost:3000).

## Project layout

```
web/
├── app/
│   ├── layout.tsx                Root layout (fonts, metadata)
│   ├── page.tsx                  Act 1 — landing
│   ├── globals.css               Tailwind 4 entry + theme tokens
│   ├── sign-in/page.tsx          Act 1 — sign-in
│   └── api/
│       └── auth/
│           └── magic-link/       POST: send magic link (stubbed)
├── components/
│   ├── ui/button.tsx             Apple-style button (3 variants, 2 sizes)
│   └── brand.tsx                 Wordmark — "The Personal Model Company"
├── lib/
│   ├── db/                       Drizzle schema + client
│   ├── api/client.ts             Typed client for the PMC backend
│   ├── auth.ts                   Magic link + sessions (stub helpers)
│   └── utils.ts                  cn() helper
├── drizzle.config.ts
├── next.config.ts                /pmc-api/* rewrite to PMC_API_URL
├── package.json                  Scripts: dev / build / start / db:*
├── railway.json                  Build = RAILPACK, preDeploy = migrate
└── README.md
```

## Visual system

- **Black, white, and one warm grey** — `#1d1d1f` on `#ffffff`, with `#6e6e73`
  for secondary text. Dark mode flips to `#f5f5f7` on `#000000`. No accent color
  in V1 — Apple-restrained.
- **Type** — SF Pro Display / SF Pro Text on Apple devices, falls back through
  the system stack. No web font download.
- **Voice** — short sentences as statements of fact. No exclamation marks. No
  emoji in product UI. Headlines lowercase punctuation; sentences end with
  periods, not bullet points.

## Deploy to Railway

1. Push to GitHub.
2. In Railway: New Project → Deploy from GitHub repo, pick this repo.
3. Set the service root to `web/` (Railway auto-detects Next.js via Railpack).
4. Add a Postgres sibling service. Railway injects `${{Postgres.DATABASE_URL}}`.
5. Add env vars: `PMC_API_URL`, `RESEND_API_KEY`, `EMAIL_FROM`, `AUTH_SECRET`.
6. Point Cloudflare CNAME for `thepersonalmodelcompany.com` at the Railway
   service target. Full (Strict) SSL.

## What's built vs not

**Built (scaffold):**
- ✓ Landing page (Act 1)
- ✓ Sign-in page (form + magic-link API stub)
- ✓ Database schema (users, sessions, magic_links, jobs)
- ✓ PMC backend client (chat / chatStream / listModels / exportBundle / deleteModel)
- ✓ Tailwind 4 + theme tokens + dark mode
- ✓ Button + Brand primitives
- ✓ Railway config + preDeployCommand for migrations

**Pending (next):**
- Connect data flow (Act 2)
- Curate + train progress (Act 3)
- Bundle reveal (Act 4)
- Chat with streaming (Act 5)
- Magic-link email delivery + session cookies (wire Resend)
- Auth middleware to protect `/(app)/*` routes
