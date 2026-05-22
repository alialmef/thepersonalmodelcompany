# The Personal Model Company — Complete App Screens

Every screen in the app, from first launch to first conversation, in one
package. Read `APP-FLOW.md` first — it's the sequence and the why. This file
is just the file map and setup.

Stack: Next.js 15 + React 19 + Tailwind 4 + lucide-react.

## Structure

```
pmc-app-complete/
├── APP-FLOW.md                  ← read first: the full journey, sequenced
├── screens/
│   ├── 0-download-page.tsx      web page after "Download for Mac" (pre-app)
│   ├── 1-first-launch.tsx       "Hello." — the welcome
│   ├── 2-connect.tsx            Step 1 · bring your writing
│   ├── 3-curate.tsx            Step 2 · reading you
│   ├── 4-5-train.tsx            Step 3 · pick a base + training (two exports)
│   ├── 6-eval.tsx               does this sound like you? ×5
│   ├── 7-first-meeting.tsx      the magic — model's first words
│   └── 8-chat.tsx               the working chat
├── shared/
│   ├── brand-mark.tsx           the circle + red-dot heartbeat
│   └── letter-cascade.tsx       text that generates token-by-token
└── styles/
    ├── globals-app-additions.css        keyframes for screens 0–5
    └── globals-eval-chat-additions.css  keyframes for screens 6–8
```

Note: `4-5-train.tsx` contains BOTH the pick-a-base screen (export
`TrainPickBase`) and the training/loss-curve screen (export
`TrainInProgress`). They're two states of step 3.

## Install

1. Drop `shared/*` into `web/components/shared/`.
2. Drop `screens/*` into `web/components/app/` (rename as you like — the
   numbering is just for reading order).
3. `0-download-page.tsx` is a web route — it goes to `web/app/download/page.tsx`.
4. Append BOTH files in `styles/` to `web/app/globals.css`. Order: app
   additions first, then eval-chat additions.
5. `npm install lucide-react` if not already present.

Fix the `@/` import paths if your alias config differs.

## Dependency map

```
Every screen          → shared/brand-mark.tsx  (the red-dot mark)
Screens 1,3,6,7       → shared/letter-cascade.tsx  (generating text)
Screens 0–5           → styles/globals-app-additions.css
Screens 6–8           → styles/globals-eval-chat-additions.css
```

Both shared primitives and both CSS files are required for the full set to
render correctly.

## The four through-lines (don't break these)

1. **The red-dot mark** appears in every screen's header and pulses on a 2.6s
   breath. Same `<BrandMark>` component everywhere.
2. **The letter-cascade** (text arriving token-by-token) recurs at every
   emotional beat — it's the product demonstrating how the model works.
3. **The voice** — lowercase, short, no exclamations, no emoji. Model output
   reflects the user's own style, not a house voice.
4. **White, quiet, generous** — one thought per screen, one primary action,
   lots of whitespace.

## The one transition that must not be a route change

Screen 6 (Eval) → Screen 7 (First Meeting) is **one continuous animation** —
the swirl. The eval UI falls away and the model swirls into being in a single
fluid move. If built as two pages with a navigation between them, the magic
dies. Timing is in `APP-FLOW.md` and the eval-chat CSS. This is the single
most important implementation detail in the package.

## What's deferred (not in this package)

- Settings drawer (retrain, export bundle, manage sources, API key, delete)
- Agent action card (inline confirm-before-send for Standard/Frontier)
- The folder/bundle "Reveal" — recommend folding into Settings; see APP-FLOW.md

## Per-screen detail

Deeper implementation notes, backend contracts (SSE endpoints, judgment
payloads, streaming), and exact choreography tables live in the two original
READMEs this package was assembled from. The essentials are reproduced in
`APP-FLOW.md`; if you need the full backend contract spec, that's where it is.
