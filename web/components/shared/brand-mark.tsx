/**
 * The PMC brand mark — circle + red center dot, matching the app icon.
 *
 * Proportions taken directly from design/icon/app-icon-1024-bold.svg:
 *   - viewBox 1024x1024
 *   - circle r=270, stroke=20 (≈1.95% of viewBox)
 *   - dot r=42
 *
 * Used everywhere from the 92px ceremonial mark at first-meeting down to
 * the 16px inline mark on chat bubbles. The dot's pulse animation is
 * defined in globals.css (`.pmc-mark-dot` → `pmc-dot-pulse` keyframes),
 * not here — the static SVG just renders the geometry.
 *
 * Note: this is the transparent in-UI variant — no white rounded-square
 * background. That backing is reserved for the app-icon treatment in
 * the Dock / Finder / favicon (web/public/icon.svg).
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
  return (
    <svg
      viewBox="0 0 1024 1024"
      width={size}
      height={size}
      aria-hidden="true"
      className={className}
    >
      <circle
        cx="512"
        cy="512"
        r="270"
        fill="none"
        stroke="#1D1D1F"
        strokeWidth="20"
      />
      <circle
        cx="512"
        cy="512"
        r="42"
        fill="#FF3B30"
        className={pulsing ? "pmc-mark-dot" : undefined}
      />
    </svg>
  );
}
