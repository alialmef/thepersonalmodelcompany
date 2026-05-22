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
      <div className="mb-14">
        <svg viewBox="0 0 100 100" width="100" height="100" aria-hidden="true">
          <circle
            cx="50"
            cy="50"
            r="38"
            fill="none"
            stroke="#1D1D1F"
            strokeWidth="0.75"
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

      <h1 className="mb-7 text-[44px] font-medium tracking-[-0.03em] text-neutral-900 pmc-fl-hello">
        Hello.
      </h1>

      <p className="mb-14 min-h-[1.5em] text-[16px] tracking-[-0.01em] text-neutral-500">
        <LetterCascade
          text="Let's start with your writing."
          startMs={5500}
          perLetterMs={35}
        />
      </p>

      <button
        onClick={onBegin}
        className="rounded-full bg-neutral-900 px-9 py-[11px] text-[13px] font-medium text-white pmc-anim-fade"
        style={{ animationDelay: '7.5s' }}
      >
        Begin
      </button>
    </div>
  );
}
