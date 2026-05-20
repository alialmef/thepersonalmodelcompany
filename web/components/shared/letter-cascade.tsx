'use client';

/**
 * Renders text where each character animates in individually, simulating a
 * model generating tokens. Used in the hero, the download screen, the
 * first-launch greeting, and the curate screen's style profile.
 *
 * The animation itself is defined in globals.css as `.pmc-letter` plus its
 * `pmc-letter-arrive` keyframe. This component just stamps the spans with
 * the appropriate inline animation-delay.
 *
 * Honors `prefers-reduced-motion: reduce` — the CSS keyframes are gated, so
 * users who prefer reduced motion will see the full string immediately
 * without animation.
 */

interface LetterCascadeProps {
  text: string;
  startMs?: number;
  perLetterMs?: number;
  className?: string;
}

export function LetterCascade({
  text,
  startMs = 0,
  perLetterMs = 35,
  className = '',
}: LetterCascadeProps) {
  return (
    <span className={className}>
      {[...text].map((char, i) => (
        <span
          key={i}
          className="pmc-letter"
          style={{ animationDelay: `${startMs + i * perLetterMs}ms` }}
        >
          {char === ' ' ? '\u00A0' : char}
        </span>
      ))}
    </span>
  );
}
