/**
 * The PMC brand mark — a thin circle with a red dot at center.
 *
 * Used everywhere from the 92px ceremonial mark at first-meeting down to
 * the 16px inline mark on chat bubbles. The dot's pulse animation is
 * defined in globals.css (`.pmc-mark-dot` → `pmc-dot-pulse` keyframes),
 * not here — the static SVG just renders the geometry.
 *
 * Stroke width scales mildly with size so small versions don't disappear
 * and large versions don't look chunky.
 */
export function BrandMark({
  size = 24,
  pulsing = true,
  className,
}: {
  size?: number;
  /** When false, the dot renders static (no pulse) — for the very first
   *  frame of first-meeting before the bloom animation takes over. */
  pulsing?: boolean;
  className?: string;
}) {
  // Scale stroke + dot relative to size so a 16px mark and a 92px mark
  // both read as "thin ring + small red center".
  const stroke = Math.max(0.5, size * 0.012);
  const dotR = Math.max(1.5, size * 0.07);

  return (
    <svg
      viewBox="0 0 120 120"
      width={size}
      height={size}
      aria-hidden="true"
      className={className}
    >
      <circle
        cx="60"
        cy="60"
        r="44"
        fill="none"
        stroke="#1D1D1F"
        strokeWidth={stroke * (120 / size)}
        vectorEffect="non-scaling-stroke"
      />
      <circle
        cx="60"
        cy="60"
        r={dotR * (120 / size)}
        fill="#FF3B30"
        className={pulsing ? "pmc-mark-dot" : undefined}
      />
    </svg>
  );
}
