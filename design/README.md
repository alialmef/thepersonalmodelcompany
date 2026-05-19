# PMC Landing — Implementation Handoff

Drop-in files for the marketing landing page at `thepersonalmodel.com`.
Stack: Next.js 15 (App Router) + React 19 + Tailwind 4 + lucide-react.

## Files

| File | Goes to |
|------|---------|
| `page.tsx` | `web/app/page.tsx` |
| `landing-sections.tsx` | `web/components/landing/landing-sections.tsx` |
| `use-in-view.ts` | `web/hooks/use-in-view.ts` |
| `globals.css` | **append to** `web/app/globals.css` (do not replace) |

Adjust import paths as needed if your alias config differs from `@/`.

## Dependencies

Already in the project except possibly `lucide-react`:

```bash
npm install lucide-react
```

## Page structure

Five sections, top to bottom:

1. **Hero** (dark) — fusion-mark choreography on page load
2. **How it works** (dark) — diagram choreography on scroll-into-view
3. **A lab. In a folder.** (light) — file listing types in on scroll-into-view
4. **Privacy** (light tertiary) — single declarative sentence
5. **Closer** (light) — static single-circle mark, final CTA

Plus a minimal footer.

The dark band runs continuously from hero through how-it-works. White begins at the folder section.

## Animations — overview

Two big choreographies, several minor reveals. **All are gated behind `prefers-reduced-motion: no-preference`** — users with reduce see final states only.

### Hero (page load)

| t (s) | event |
|-------|-------|
| 0.1 | nav fades in |
| 0.2–0.8 | mark circles fade in (separated) |
| 0.8–2.8 | drift toward center |
| 2.5–3.25 | flash pulse |
| 2.8 | fusion complete + headline letter cascade begins (35ms stagger) |
| 4.0 | sub-line fades in |
| 4.5 | CTA fades in |
| ~5.0 | settled |

### How it works (scroll into view, fires once per session)

| t (s) | event |
|-------|-------|
| 0.2 | section title fades in |
| 0.7–1.0 | left icons stagger in (100ms apart) |
| 1.1 | left lines draw in (stroke-dashoffset) |
| 1.8 | center mark grows in |
| 2.1 | left → center particles fire (SMIL `animateMotion`, freeze at end) |
| 2.4 | mark label `your model` fades in |
| 3.1 | right lines draw in |
| 3.3–3.6 | right icons stagger in |
| 3.7 | center → right particles fire |
| 5.0 | tagline fades in |

The mark in this section is a **single** circle. Fusion happened in the hero; this section shows the model existing and acting.

### Folder, Privacy, Closer

Each section uses `useInView` from `use-in-view.ts` to add a `pmc-in-view` class when scrolled into view. CSS keyframes scoped to `.pmc-in-view` trigger the reveals (folder file lines type in, privacy fades up, closer mark + heading + button fade up sequentially).

## Production assets still needed

These are the build-side blockers — they don't affect the design, but they affect the launch.

- `public/brand-mark.webm` and `public/brand-mark.mp4` — the iridescent fusion loop you have. 5–6 seconds. Loops continuously.
- `public/brand-mark-poster.jpg` — a single frame from the video for `<video poster>`. Use a frame from the iridescent peak.
- `public/mark.svg` — single-circle outline. Source for favicon, .icns, OG image.
- `public/favicon.ico`, `public/apple-touch-icon.png`, `public/icon.png` — generated from `mark.svg`.
- `public/og-image.png` — 1200×630, single circle centered on black, with wordmark below.
- Mac `.dmg` and download URL — `/download` route should serve or redirect.

The hero's SVG fusion choreography is a **fallback** for users on slow connections or before the video buffers. Once the video file exists, replace the SVG with a `<video>` element. See the TODO comment in `Hero()`. The SVG can stay as a `<noscript>` or `<video>` poster behind the video, or be removed entirely depending on how aggressively you want to optimize.

## Optional / open decisions

These are choices that aren't blockers but are worth considering before launch:

1. **The closer's static circle bookend** — keep, or strip to just the heading + button? Currently kept (the page opens with two circles fusing and ends with the one that resulted; bookend on the brand visual).
2. **Pricing nav link** — removed. Pricing now happens in-app, after Curate runs and before Train. If you want a `/pricing` page for the curious, add it back in the nav.
3. **Reduced-motion behavior of the iridescent video** — pause on first frame, or hide and fall back to the SVG? Current code falls back to the SVG.
4. **Particles in How it works** — currently SMIL `animateMotion`, fine in modern browsers but deprecated long-term. Upgrade path: CSS `offset-path` or Framer Motion. Not blocking.
5. **Hover-driven path highlighting** in How it works — nice-to-have, not implemented. Would require coordinating HTML hover state with SVG path classes via React state.

## Notes on the brand voice

The copy throughout is institutional and quiet by design — see the original brief at `web/docs/personal-model-company-brief.md` for the full rules. Specifically:

- No exclamation marks anywhere.
- No emoji in product UI.
- Headlines as statements of fact, sentence case, terminal periods.
- "You own it" must show up somewhere on every screen (currently: hero sub, folder sub, "make one of your own" closer, single-circle mark bookend).

If new copy is added, run it past those rules first.
