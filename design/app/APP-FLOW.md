# The Personal Model Company — App Flow

The complete in-app journey, in order, from the moment the user first opens
the app to the moment they talk to their model for the first time.

This is the map. Each screen has its own component file (referenced below).
This document is the sequence, the transitions, and the why.

---

## The arc, in one line

A ceremonial welcome → ask for trust → show the work → one real choice → the
wait → judge the result → meet the model → talk.

Eight screens. The emotional shape is a held breath that releases at the
first conversation.

---

## The through-lines (constant across every screen)

**The red-dot mark.** A single thin circle with a red dot at its center,
pulsing on a 2.6s breath. It appears in every header from the first launch
onward. It's the visual proof that it's the same model the whole way through.
The dot is the model's heartbeat — it means *alive, active, yours*.

**The letter-cascade.** Text that arrives letter-by-letter, like a model
generating tokens. It recurs at every emotional beat: the welcome greeting,
the style profile assembling, the model's first words. It's the product
demonstrating itself — the interface generates text the same way the model
does.

**The voice.** Lowercase, short, declarative. No exclamation marks. No emoji.
Every line the model "speaks" is in the user's own style, not a house voice.
The product's copy and the model's output share one register.

**White, quiet, generous.** Light background throughout the app (the dark
hero belongs to the marketing site only). One thought per screen. One primary
action per screen. When in doubt, more whitespace.

---

## SCREEN 1 — First Launch

**File:** `first-launch-screen.tsx`
**Feeling:** arrival. You crossed a threshold.

Pure white. The user has just opened the app for the first time.

**Choreography:**

| t (s) | event |
|-------|-------|
| 0.0–0.6 | empty white — held silence |
| 0.6 | mark scales in |
| 1.7 | red dot ignites with overshoot |
| 2.5 | dot begins its forever pulse |
| 2.8 | `Hello.` fades up |
| 2.8–5.5 | **held silence** — just "Hello." and the breathing mark |
| 5.5–6.5 | `Let's start with your writing.` cascades in, letter by letter |
| 7.5 | `Begin` button arrives |

**The one rule:** do not shorten the 2.8s → 5.5s silence. Almost three
seconds where the screen shows one word and a heartbeat. Every instinct will
say cut it. It's the most important pause in the product — it tells the user
this thing respects them enough not to rush. A welcome, not an onboarding.

**Exit:** click `Begin` → Screen 2.

---

## SCREEN 2 — Connect

**File:** `connect-screen.tsx`
**Feeling:** "I'm handing over my writing. I need to feel safe."
**Step 1 of 3.**

`Bring your writing.` — the verb makes the user the active party. Below it,
the privacy promise as a real sentence, not a footer: *Everything stays on
your Mac. The audit log shows every byte.*

Four source rows in an Apple Settings-pane list (hairline dividers, rounded
container): **Messages**, **Notes**, **Documents**, **Mail**. Each row: icon
chip, name, a scope line (`last 24 months`, `Gmail · sign in to connect`),
and a button — `Connect` (filled, for permissioned sources) or `Add`
(outlined, for drag-and-drop documents).

`Continue` is disabled until at least one source connects. Message to its
left: `Connect at least one to continue.`

**Choreography:** header → heading → privacy line → source rows cascade in
(100ms stagger) → footer. Subdued. The ceremony was Screen 1; this is work.

**Exit:** click `Continue` → Screen 3.

---

## SCREEN 3 — Curate

**File:** `curate-screen.tsx`
**Feeling:** "the model is being made for me, and I can see it happening."
**Step 2 of 3.**

`Reading you.` — the model is the active party now; it's reading everything
you've written. *This takes a few minutes. You can close the window — we'll
keep going.*

Two columns:
- **Left — Events.** A monospace, timestamped field-note stream as the
  pipeline runs: `Read 12,431 messages.` `Filtered 2,108 short replies.`
  `Excluded 47 contacts.` `Style profile built.` Lab-notebook aesthetic.
  Specificity is trust — real numbers make the user feel seen.
- **Right — Style profile.** Assembles itself line by line, each observation
  typing in via letter-cascade: *You write in lowercase. Short sentences. You
  use "honestly" a lot. You sign off without a closing. Average sentence: 12
  words.* This is the most satisfying moment so far — the user watches their
  own voice described back to them, written by software.

**No spinner. No percentage.** The events ARE the progress. The screen
settles when the last event fires; then a `Continue` button appears (it does
not auto-advance — the user gets to see the finished picture).

**The bar to hold:** the style-profile observations must be specific and
*measured* (`Average sentence: 12 words.`), never generic descriptors
(`Warm and direct.`). The magic depends on it. Push back on the backend if it
produces vague output.

**Exit:** click `Continue` → Screen 4.

---

## SCREEN 4 — Train, Pick a Base

**File:** `train-screen.tsx` (export: `TrainPickBase`)
**Feeling:** the one real choice.
**Step 3 of 3.**

`Pick a base.` with the framing line that defuses the pricing-feels-icky
problem: *Your writing is the same. The base is what we train it onto.* The
user isn't picking a tier of themselves — they're picking the metal
underneath. Their data is constant.

Three cards, described by **capability**, not parameters:

- **Try** — *Free* — `A model you can chat with.`
  Drafts in your voice. Answers in your voice. Just chat.
- **Standard** — *$79/mo* — `An assistant that can do things.`
  Reads your mail, browses, takes notes, books, schedules. Light agent tasks.
- **Frontier** — *$299/mo* — `A model that can code.`
  Everything Standard does, plus writes software at frontier quality.

Try is pre-selected and genuinely free — no trial, no asterisk. The
conversion to paid happens when the user hits a capability ceiling, not a
paywall. `Cancel anytime. Your bundle is yours.` sits left of the `Train`
button.

**Exit:** click `Train` → Screen 5. (If a paid tier is selected, Stripe
Checkout happens between, returning to Screen 5 on success. Request macOS
notification permission at this click.)

---

## SCREEN 5 — Train, In Progress

**File:** `train-screen.tsx` (export: `TrainInProgress`)
**Feeling:** the wait that makes it theirs.
**Step 3 of 3.**

`Training your model.` *About 38 minutes. We'll notify you when it's ready.*
Rough ETA, not minute-precise — invites checking back, not watching.

A **naked loss curve** descends across the screen — no axis labels, no
tooltip, just the shape, drawing itself in over 6 seconds. It's the universal
visual signature of training a real model; showing it is the institutional
flex that says *this is not a chatbot wrapper.* Two big tabular numbers flank
it: `Loss 2.317` and `Step 428 / 1,200`, ticking live from the SSE stream.

A handoff card at the bottom with a small red dot: *You can close this
window. The model is training in our enclave — your Mac will ping when it's
ready.* No email. macOS notification fires on completion.

**Exit:** the user leaves. When the model finishes, a macOS notification
(`Your model is ready.`) brings them back → Screen 6.

---

## SCREEN 6 — Evaluation

**File:** `eval-screen.tsx`
**Feeling:** "let me check it actually sounds like me."
**Shown the moment training completes.**

`Does this sound like you?` Five rounds. Each round: a grounded situation
(drawn from the user's own patterns where possible) → the model responds live,
streaming → the user judges: `Not quite` / `That's me` / `Edit instead`.

**What it really is:** preference-data collection disguised as quality
control. Every judgment is DPO-grade training signal. The user is doing RLHF
on their own model and it feels like verifying it. It also completes the
"personal AI lab" arc — they've now run curation, training, AND evaluation.

`Edit instead` is the highest-value signal (a user-corrected response = "here
is exactly what I'd have said"), offered quietly. Reject reasons are optional
chips that appear after a reject and never block advancing.

Progress shown as five segmented dots, top right: `3 of 5`.

**Exit:** final round judged → the swirl transition into Screen 7.

---

## THE TRANSITION — Eval folds into the Meeting

**This is ONE continuous animation, not a route change.** It is the "loads
like magic" moment. If it's built as two pages with a navigation between them,
the magic dies.

| t (s) | event |
|-------|-------|
| 0.0 | the eval UI (headline, buttons, progress) fades and falls away; screen washes to clean white |
| 0.3 | the mark **swirls in** — scales up from 0.6 while rotating out of a −40° tilt, settling upright (**the only rotation anywhere in the entire product** — reserved for this single moment) |
| 1.3 | a ring expands outward from the mark once and dissipates — the energy of arrival |
| 1.4 | the red dot **blooms** with overshoot — the heartbeat begins |
| 2.2 | the dot settles into its forever pulse |
| 2.4 | the first line arrives, centered, at 22px |
| 4.2 | the composer rises from the bottom |

---

## SCREEN 7 — First Meeting

**File:** `first-meeting.tsx`
**Feeling:** meeting someone for the first time who has known you forever.

After the swirl, the model speaks its first words — alone, centered, on
white, with no chat chrome and no history. No transcript is shown, because
the history isn't in the transcript — it's in the model. They already know
you. The first line carries all of it.

The line, generated and conditioned on the style profile (structure constant:
*recognition → "known you forever" → humility → hand over control*):

> there you are. i've read everything you've ever written — i think i get
> you. where do you want to start?

It's larger than chat text (22px) because it isn't a chat message yet — it's
THE FIRST THING SAID, and it gets its own moment floating in space before any
chat UI exists. The composer waits below.

**Exit:** the user types their first reply → the ceremonial space resolves
into the working chat (Screen 8). The ceremony is for the meeting; the work is
for the chat. The magic is precious because it ends — do not keep the airy
centered format forever.

---

## SCREEN 8 — Chat

**File:** `chat-screen.tsx`
**Feeling:** "this is mine and I can talk to it."

The working surface. Quiet iMessage-style bubbles — the model feels like a
person you text (the eval established the texting metaphor). Tight radius,
system greys, the model in near-black to carry weight. NOT playful iMessage
blue.

Header: mark + dot + `Your model` left, a single settings gear right. That
gear opens the settings drawer (retrain, export bundle, manage sources, API
key, delete — where the "you own it" thesis lives concretely). No nav chrome.

Model responses stream token-by-token from the chat SSE endpoint — the same
generation feel as everywhere else. When the model takes an *action*
(Standard/Frontier: sends email, runs code, schedules), it renders inline as
a compact action card; irreversible actions require explicit confirmation
before executing.

This is where the journey ends and the relationship begins.

---

## Files map

| Screen | Component file | Folder handed off in |
|--------|---------------|---------------------|
| 1 First Launch | `first-launch-screen.tsx` | `pmc-app/` |
| 2 Connect | `connect-screen.tsx` | `pmc-app/` |
| 3 Curate | `curate-screen.tsx` | `pmc-app/` |
| 4 Train (pick) | `train-screen.tsx` | `pmc-app/` |
| 5 Train (run) | `train-screen.tsx` | `pmc-app/` |
| 6 Eval | `eval-screen.tsx` | `pmc-eval-chat/` |
| 7 First Meeting | `first-meeting.tsx` | `pmc-eval-chat/` |
| 8 Chat | `chat-screen.tsx` | `pmc-eval-chat/` |

Shared primitives (`brand-mark.tsx`, `letter-cascade.tsx`) and the CSS
additions live in those same two folders. The two READMEs in `pmc-app/` and
`pmc-eval-chat/` carry the per-screen implementation detail, backend
contracts, and choreography specs. This document is the sequence that ties
them together.

## What's NOT in this flow (deliberately deferred)

- **The Settings drawer** (retrain, export bundle, manage sources, API key,
  delete) — reachable from the chat header gear. Where "you own it" gets its
  concrete home. Designed separately.
- **The agent action card** — the inline confirm-before-send UI for when
  Standard/Frontier models take actions. Stubbed in `chat-screen.tsx`.
- **The folder/bundle "Reveal"** — the original brief's Act 4. Recommendation:
  the First Meeting IS the reveal; demote the bundle to a Settings feature
  rather than a mandatory screen. Flagged as a product call, not just design.
