"use client";

import { useEffect, useState } from "react";
import { LetterCascade } from "@/components/shared/letter-cascade";

const DMG_PATH = "/downloads/PersonalModelCompany.dmg";

/**
 * The download moment — the bridge from web to product.
 *
 * Full-bleed white. The mark scales in, the red dot blooms with overshoot
 * (the choreography is defined in globals-app-additions.css as `.pmc-dl-mark`
 * and `.pmc-dl-dot`), and a single-sentence headline arrives. The actual
 * .dmg download fires the moment the page mounts so the user's browser
 * shows their native download chrome while they read the install line.
 *
 * Voice / layout match the rest of the designed app (first-launch, welcome,
 * etc.): one thought, one image, no chrome. The page deliberately has no
 * back-to-home link — this is a moment, not a navigation hub. The browser's
 * back button gets them home if they want.
 *
 * Choreography (from globals-app-additions.css):
 *   0.3s   mark scales in (.pmc-dl-mark via pmc-mark-scale-in)
 *   1.2s   red dot ignites with overshoot (.pmc-dl-dot via pmc-dot-appear)
 *   2.0s   dot settles into its 2.6s forever breath (pmc-dot-pulse)
 *   2.4s   headline + install line fade up (via pmc-anim-fade-up)
 */
export default function DownloadPage() {
  const [fallback, setFallback] = useState(false);

  useEffect(() => {
    // Auto-fire the .dmg download. The browser handles it natively; this
    // page stays put so the user sees the install message while bytes
    // arrive. If after a few seconds nothing seems to have happened the
    // fallback link appears in case the auto-trigger was blocked.
    const a = document.createElement("a");
    a.href = DMG_PATH;
    a.download = "PersonalModelCompany.dmg";
    a.rel = "noopener noreferrer";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    const t = setTimeout(() => setFallback(true), 4000);
    return () => clearTimeout(t);
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-white px-7 text-neutral-900">
      {/* Mark — uses the existing .pmc-dl-* download-screen choreography */}
      <div className="mb-12">
        <svg
          viewBox="0 0 100 100"
          className="h-[clamp(96px,12vw,160px)] w-[clamp(96px,12vw,160px)]"
          aria-hidden="true"
        >
          <circle
            cx="50"
            cy="50"
            r="38"
            fill="none"
            stroke="#1D1D1F"
            strokeWidth="0.5"
            className="pmc-dl-mark"
          />
          <circle
            cx="50"
            cy="50"
            r="3"
            fill="#FF3B30"
            className="pmc-dl-dot"
          />
        </svg>
      </div>

      {/* Headline — lowercase, terminal period, no exclamation, sized to
          feel like a Mac product page, not a tiny modal */}
      <h1
        className="mb-6 text-center font-medium leading-[0.98] tracking-[-0.035em] pmc-anim-fade-up"
        style={{
          fontSize: "clamp(40px, 6vw, 88px)",
          animationDelay: "2.4s",
        }}
      >
        your model is downloading.
      </h1>

      {/* Install instruction — single sentence, no separate "after
          downloading" panel. Less scaffolding, more brand voice. */}
      <p
        className="mb-14 max-w-[36ch] text-center text-neutral-500 pmc-anim-fade-up"
        style={{
          fontSize: "clamp(15px, 1.6vw, 19px)",
          animationDelay: "2.8s",
        }}
      >
        <LetterCascade
          text="open the .dmg, then drag personal model company to applications."
          startMs={2800}
          perLetterMs={22}
        />
      </p>

      {/* Fallback — only after 4s, and only if the auto-trigger didn't
          seem to fire. Quiet, undertstated link. */}
      {fallback && (
        <a
          href={DMG_PATH}
          download="PersonalModelCompany.dmg"
          className="pmc-anim-fade text-[13px] text-neutral-500 underline decoration-neutral-300 underline-offset-[3px] transition-colors hover:text-neutral-900"
        >
          if nothing happened, tap here.
        </a>
      )}
    </main>
  );
}
