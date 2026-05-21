'use client';

import { LetterCascade } from '@/components/shared/letter-cascade';

/**
 * The Mac app's first launch. Pure white. The user has just opened the app
 * for the first time. This is the bridge moment from web to product.
 *
 * Critical design decision: ~3 seconds of held silence at the most
 * vulnerable point in onboarding. The user sees "Hello." and a breathing
 * red dot, with nothing else on screen, for almost three full seconds
 * before the second line arrives.
 *
 * DO NOT shorten this. The held quiet is the entire point.
 *
 * Choreography:
 *   0.0–0.6s  empty white (held silence)
 *   0.6s      mark scales in
 *   1.7s      red dot ignites with overshoot
 *   2.5s      dot begins indefinite 2.6s pulse
 *   2.8s      "Hello." fades up
 *   5.5–6.5s  "Let's start with your writing." cascades in
 *   7.5s      "Begin" button arrives
 *
 * On `Begin`, navigates to /app/connect (Step 1 of onboarding).
 */
export default function FirstLaunchScreen({ onBegin }: { onBegin?: () => void }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-white px-7">
      {/* Mark scaled for full-screen takeover. clamp() so it's right at
          1024px AND on a 32" display. */}
      <div className="mb-16">
        <svg
          viewBox="0 0 100 100"
          className="h-[clamp(120px,16vw,200px)] w-[clamp(120px,16vw,200px)]"
          aria-hidden="true"
        >
          <circle
            cx="50"
            cy="50"
            r="38"
            fill="none"
            stroke="#1D1D1F"
            strokeWidth="0.5"
            className="pmc-fl-mark"
          />
          <circle
            cx="50"
            cy="50"
            r="3"
            fill="#FF3B30"
            className="pmc-fl-dot"
          />
        </svg>
      </div>

      {/* "Hello." has to land like a moment on a 27" display — 44px would
          look lost. clamp() scales from 72px in dev to 140px on large. */}
      <h1
        className="mb-10 font-medium leading-[0.95] tracking-[-0.035em] text-neutral-900 pmc-fl-hello"
        style={{ fontSize: 'clamp(72px, 10vw, 140px)' }}
      >
        Hello.
      </h1>

      <p
        className="mb-16 min-h-[1.5em] tracking-[-0.015em] text-neutral-500"
        style={{ fontSize: 'clamp(18px, 2vw, 26px)' }}
      >
        <LetterCascade
          text="Let's start with your writing."
          startMs={5500}
          perLetterMs={35}
        />
      </p>

      <button
        onClick={onBegin}
        className="rounded-full bg-neutral-900 px-12 py-4 text-[15px] font-medium text-white pmc-anim-fade transition-transform hover:scale-[1.03] active:scale-[0.98]"
        style={{ animationDelay: '7.5s' }}
      >
        Begin
      </button>
    </div>
  );
}
