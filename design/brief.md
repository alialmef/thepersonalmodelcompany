# The Personal Model Company — Design Brief

> Read this top to bottom. The product, the user, the constraints, the screens.
> When you're done you should know enough to design any screen we haven't built yet,
> and to push back on any of mine that don't fit.

---

## The product in one sentence

A native Mac app that trains a personal AI agent on your own writing —
your messages, notes, email, documents — and gives you a model you own.

---

## The longer version

Most "personal AI" today is a wrapper. You write a prompt, GPT-4 or Claude
responds in *its* voice, and the data you fed it has been hoovered up to train
the next foundation model. Nothing belongs to you.

PMC inverts that:

- You connect your data sources. We never read them.
- We train a personal **LoRA adapter** on top of an open-weights base model.
- You own the resulting bundle — weights, style profile, training manifest,
  eval report, audit log. Export it anytime.
- You use it through PMC (hosted) or take it and run it anywhere.

The product is the **agent**, not the chat. The model isn't there to answer
trivia — it's there to draft your emails, code in your voice, schedule things
on your behalf, read articles and summarize them the way you'd describe them
to a friend.

---

## Who this is for

Not the mass consumer. The target user is:

- A **developer, founder, writer, or knowledge worker** who already pays for
  Claude Pro or ChatGPT Plus and finds them generic
- On a **Mac** (so native data access works)
- Cares enough about *ownership* to pay a premium over the free wrapper layer
- Has **enough writing online** (email, code, notes, messages) that a model
  trained on it would actually sound like them

Their pain: "I have a generic AI assistant. I want one that knows me, has my
context, and that I can take with me."

---

## Brand voice

Institutional. Declarative. **Apple product-page energy, not startup-launch
energy.**

The brand name itself sets the register: *The Personal Model Company.* Not
"PersonAI" or "Mimi" or some single-word coinage. It reads as a company that
makes a thing, not a tool with a logo. Steve Jobs introducing a product line,
not a Twitter founder doing a viral demo.

Specific copy rules:

- **Short declarative sentences.** Fewer commas than you think.
- **No exclamation marks.** None. Anywhere.
- **No emoji** in any product UI. No 🚀, no 👋, no 🎯.
- **No "we"-talk that performs friendliness.** Don't write "We're so excited
  to help you train…" Write "Train your model."
- **No urgency theatre.** No "Only 27 spots left!" No countdowns. The
  fact that the first 100 users are free is tracked silently on the backend.
- **The verb is "train" or "make" or "own."** Not "create" or "build."
- **Headlines as statements of fact.** "Your model is ready." Not "Your model
  is ready! 🎉"
- **Lowercase punctuation in headlines.** Sentence case, terminal periods.
  Not Title Case, not bare phrases.

Examples of the voice working:

> The Personal Model Company.
> Train an AI model on your own writing. You own it. Host it. Take it anywhere.

> Let's start with your writing. The more we have, the better it sounds like you.

> Your model is ready.

Examples of voice failures (avoid):

> Welcome to PMC! 🚀 We're so excited to help you build YOUR personal AI agent!
> ⚡ Only 27 spots left in our Founder program — claim yours now!
> Get started in 3 easy steps ✨

---

## Visual system

### Color
- **Light mode**: white background (`#FFFFFF`), near-black foreground
  (`#1D1D1F`), warm grey for secondary text (`#6E6E73`), subtle grey for
  panel fills (`#F5F5F7`), thin border (`rgba(29,29,31,0.08)`).
- **Dark mode**: pure black background (`#000000`), warm white foreground
  (`#F5F5F7`), grey-warm secondary (`#86868B`).
- **No accent color in V1.** No blue button, no purple gradient. Apple's
  product pages routinely use only black/white/grey — we should too. We can
  introduce one accent later if there's a real reason.

### Typography
- **SF Pro Display** for headlines, **SF Pro Text** for body. System fallback
  through `-apple-system, BlinkMacSystemFont, ...`. No web-font download.
- Headlines: tight tracking (`-0.03em` to `-0.02em`), `font-medium` (500), not
  bold.
- Body: regular weight, comfortable line-height (`leading-relaxed`).
- Reading-width copy capped at `max-w-[58ch]`. Don't let lines run wider.

### Density + rhythm
- **Generous whitespace.** When in doubt, more.
- **Two text sizes per screen** is the ceiling. A headline and a body. Maybe
  one tertiary label. Anything more is noise.
- **One primary action per screen.** Never two buttons fighting for attention.
- **Rule of one thought per screen.** If a screen has two ideas, it's two
  screens.

### Components
- Buttons: pill shape (full rounded), three variants
  (`primary` = filled foreground, `secondary` = bordered, `ghost` = text-only)
  and two sizes (default / large).
- Inputs: subtle filled background, no harsh borders, focus ring is a soft
  foreground tint.
- Cards: minimal — no shadows by default. Border or background fill is the
  separator.
- Lists: single thin divider between rows. Apple settings-pane style.

### Motion
- The **counter ticking up** during ingestion
- The **loss curve animating** during training
- The **style profile assembling itself** word by word
- These are the *only* animated things in V1. Everything else is still.

No springs going wild. No "wow" page transitions. No skeleton shimmers.
Just calm, factual movement on the few things that genuinely change.

---

## The five-act flow

A user's complete journey from "never heard of this" to "talking to my model"
breaks into five acts. Each act is one *feeling*, expressed across one or
more screens.

### Act 1 — Promise

**Feeling:** the moment of "wait, you can train an AI to be *me*?"

**Screens:**
- Landing page (`/`)
- Sign-in (`/sign-in`)
- Download for Mac (the marketing-only entry point)

**What the user does:** reads the proposition, enters their email, gets a
magic link, signs in. (Or downloads the Mac app from the landing page.)

**Current state:** landing page and sign-in form built. Magic-link email is
stubbed (no Resend wiring yet).

**Open design questions:**
- Should the landing page have a 2-minute scrollytelling section explaining
  how the model gets trained, or stay one-screen-one-thought?
- How do we handle the "download for Mac" CTA in a way that doesn't make
  non-Mac visitors feel rejected?
- Below-the-fold sections (How it works / What you receive / Privacy /
  Pricing) — are these on the landing page, or each their own page?

### Act 2 — Gather

**Feeling:** "I'm handing over my writing. I need to feel safe."

**Screens:**
- Connect data (`/connect`)
- Each source's connection flow (OAuth modal, Full Disk Access prompt,
  per-source picker)

**What the user does:** clicks "Connect" on Gmail, iMessage, Apple Notes,
Mail, WhatsApp, etc. Sees a live counter of items collected. Reads the
privacy promise.

**Current state:** the page exists. In Tauri (Mac app) mode, the iMessage
row uses native ingestion via Rust (`imessage_status` + `ingest_imessage`).
In web mode, it's all file upload. **Apple Notes, Mail, WhatsApp native
modules are not yet built.** OAuth flows for Gmail/Drive/Notion are not yet
built.

**Open design questions:**
- How do we communicate "this stays on your Mac" when some of the rows are
  native (Mac-only) and some are upload (cloud-bound)?
- What does the Full Disk Access prompt UI look like? Native macOS handles
  the actual permission grant — what do we show *before* deep-linking to
  System Settings?
- Per-source exclusion (skip these contacts, this folder) — modal? Inline?
- Time-range filter (default 24 months) — where does this live?

### Act 3 — Wait, watching the work

**Feeling:** "the model is being made *for me*, and I can see it happening."

This is where most products would put a spinner and a "training your AI…"
message. PMC shouldn't. Curation and training take real time (minutes for
curate, 30-60 min for training). That wait is **the moment the user starts
believing the model is theirs.** Show the work.

**Screens:**
- Curate (`/curate`) — live event stream as the pipeline runs
- Train (`/train`) — animated loss curve, ETA, "we'll email you" framing

**What the user does:** watches things happen. Reads through the events.
Sees their style profile being assembled. Eventually closes the tab and
goes about their day; comes back when the email arrives.

**Current state:** **neither page is built.** Backend SSE event stream is
done (`/v1/users/{id}/runs/{job_id}/events`). The JobScheduler runs the
pipeline. We just need the frontend.

**Open design questions:**
- What do the live curate events look like rendered? List with checkmarks
  appearing? Or running text? Or progress bars per stage?
- The style profile preview: when curation finishes, we have things like
  "warm, direct, ~14 words/sentence, you say 'honestly' a lot." How is
  this shown? Card? Sentence? Animated text?
- The training loss curve — what scale, how prominent? Real animated SVG
  or canvas? With axis labels or naked aesthetic?
- ETA countdown: precise to the minute, or "about 12 minutes left"?
- The "we'll email you when it's ready" handoff — banner? Footer note?

### Act 4 — Reveal

**Feeling:** unboxing. The model is theirs. There it is, in a folder.

**Screens:**
- Ready (`/ready`) — the bundle reveal

**What the user does:** sees their model laid out as files in a folder. Reads
the eval scores ("sounds like you · 78%, privacy check · passed"). Picks one
of two equal-weight actions: **Talk to it** or **Export bundle**.

The fact that "Export bundle" sits *next to* "Talk to it" — same weight,
same prominence — is the whole ownership thesis made visible.

**Current state:** **not built.** Backend bundle artifact exists
(`ArtifactBundle.to_zip()` works).

**Open design questions:**
- The folder visualization — literal Finder-like file list, or stylized?
  ASCII-ish or skeuomorphic icons?
- The eval scores below — table, sentence, or pill badges?
- Entrance animation — should the folder fade in, slide in, build up file
  by file? Or just appear?
- Where does the model's *name* go? Default is something like
  `alex_personal_model/`. Do we let the user rename?

### Act 5 — Use

**Feeling:** "this is mine and I can talk to it."

**Screens:**
- Chat (`/chat`)
- Settings panel (slides in from the right, not a separate page)

**What the user does:** chats with their model. Maybe asks it to draft an
email. Maybe gives it a tool (web, code, calendar) and watches it act. Opens
the settings panel to retrain, export, get API key, manage sources, delete.

**Current state:** **not built.** Backend streaming chat works
(`/v1/chat/completions` with `stream: true` SSE).

**Open design questions:**
- What's the chat UI shape — iMessage-quiet (bubbles, monospace-ish
  timestamps), or Linear-cleaner (no bubbles, just text), or Claude.ai-like
  (block-formatted)?
- Token streaming — letter-by-letter? Word-by-word? Chunked phrases?
- When the agent takes an action (sends an email, runs code), how is that
  shown in the chat stream? Inline card? Sidebar log?
- The settings panel — slide-in from right, full-screen modal, or inline
  pane? What's in it (vs the chat)?
- Connected sources — manageable inline in chat ("attach this email"), or
  only in settings?

---

## What's committed in memory (do not relitigate)

These are decisions already made. Design within them; if you want to push
back, do so explicitly with a reason.

1. **Native Mac app**, distributed as a signed `.dmg` outside the App Store.
   Tauri + Next.js webview. Web stays for marketing + billing only.
2. **Three pricing tiers, monthly only.** Try $19 (Llama 3.1 8B),
   Personal $79 (Qwen 3.6 27B Dense), Frontier $299 (Kimi K2.6). No
   lifetime SKU. No annual prepay shown by default.
3. **First 100 users get free Try training.** Backend counter. No urgency
   tactics on the page.
4. **OAuth + native ingestion, not file upload.** Upload exists as a fallback
   only for sources without APIs (iMessage exception is handled natively).
5. **Personal agent, not personal chat.** The model is expected to *do*
   things (browse, code, send emails, schedule). Chat is just the surface
   the agent thinks through.
6. **No emoji, no exclamation marks, no urgency, no startup-launch energy.**
   Apple-quiet always.
7. **Black/white + one warm grey.** No accent color in V1.

---

## Anti-patterns to avoid

- **Dashboard sprawl.** No "Your AI Models" grid view, no widgets, no
  "weekly stats." One model, one chat, one settings drawer.
- **Confetti, celebrations, "Congratulations!" toasts.** When training
  finishes, the screen *is* the celebration — don't add fireworks.
- **Onboarding tooltips.** If you need a tooltip to explain a button, the
  button is wrong.
- **Skeleton loaders.** Either show real content or show nothing. No
  shimmer placeholders.
- **Progress percentages without meaning.** Don't show "32%" if you don't
  know that it actually represents 32% of anything. Use "step 3 of 7" or
  "14 min remaining" or just dots.
- **Two CTAs on the same screen competing.** Pick one.
- **Generic AI iconography.** No brain, no chip, no neural-net mesh, no
  sparkle, no robot. The brand has no logo — it's a wordmark.

---

## Inspiration / reference points

- **Apple product pages** (mac.com/personal-model-company-vibe) — the
  rhythm of a hero, then below-the-fold thought blocks.
- **Linear's settings pane** — the slide-in style, the lack of nav chrome.
- **Things 3** — the way it lets one thought live on screen at a time.
- **Stripe Atlas** — the institutional confidence in a long, dry product
  name done well.
- **Arc Browser's downloads** — for how to present a Mac app .dmg without
  acting like a startup.
- **iMessage** — for what "quiet chat" feels like (the model the user
  trained should feel like a person they're texting, not a chatbot).

---

## What I'd like from this exercise

Walking through with you, I want to settle:

1. **The visual identity components** — once SF Pro + black/white is set,
   what does the wordmark look like across the app? Tray icon? `.icns`?
2. **The five act-screens, drawn** — even text mockups would help, but
   ideally low-fi visual mocks. The Wait screens especially.
3. **The library of components** — buttons, inputs, source rows, the
   slide-in panel. Tokenized in Tailwind (the project uses TW 4 with
   CSS-first config; theme tokens live in `web/app/globals.css`).
4. **The motion vocabulary** — three reusable animation primitives, no
   more (counter, loss-curve, fade-in-text).
5. **The component-vs-content split** — what's a reusable shadcn-style
   component vs what's a one-off arrangement.

Start with the act you find weakest in the current state. Move from there.

---

## How to find what's already there

- Backend code: `pmc/` (Python)
- Frontend (web + in-Tauri webview): `web/` (Next.js 15 + React 19 +
  Tailwind 4 + shadcn-style primitives in `web/components/`)
- Mac native shell + Rust ingestion: `desktop/` (Tauri 2)
- Memory notes (every product decision): `~/.claude/projects/-Users-alialmeflehi-Desktop-Sites-thepersonalmodelcompany/memory/`
- Run it locally: `./scripts/dev.sh`

---

## One more thing

The product's whole reason to exist is the line *"you own it."*

That phrase must show up somewhere on every screen — explicit, implied, or
felt through the affordances. The Export button in Act 4 sits next to Talk
to it because of this. The settings panel surfaces "Download bundle" first.
The privacy line on the Connect screen is part of the page, not a footer.

If a screen doesn't reinforce ownership, it's wrong.
