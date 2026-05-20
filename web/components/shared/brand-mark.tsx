'use client';

/**
 * The brand mark. A single circle with an optional red dot at its center.
 *
 * The dot signals "your model is active." It appears on every in-app screen
 * once the user has begun their personal model — from the download screen
 * forward. It pulses on a 2.6s breath cycle.
 *
 * Sizes used in the app:
 *   - 32px: in step headers (Connect, Curate, Train)
 *   - 48px:  in closer / static brand placements
 *   - 100px: in the first-launch hero and download screen
 */

interface BrandMarkProps {
  size?: number;
  withDot?: boolean;
  className?: string;
}

export function BrandMark({
  size = 32,
  withDot = true,
  className = '',
}: BrandMarkProps) {
  const r = size * 0.38;
  const dotR = Math.max(1.5, size * 0.03);

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      className={className}
      aria-hidden="true"
    >
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke="#1D1D1F"
        strokeWidth="0.75"
      />
      {withDot && (
        <circle
          cx={size / 2}
          cy={size / 2}
          r={dotR}
          fill="#FF3B30"
          className="pmc-mark-dot"
        />
      )}
    </svg>
  );
}
