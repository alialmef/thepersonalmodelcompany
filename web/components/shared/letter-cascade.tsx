/**
 * Letter-by-letter entrance animation. Used wherever text arrives word-by-
 * word the way the model "types" — landing hero, first meeting, etc.
 *
 * Each character is its own span with a calculated animation-delay. Pair
 * with the `.pmc-letter` class in globals.css (animation keyframes there).
 */
export function LetterCascade({
  text,
  startMs = 0,
  perLetterMs = 28,
  className = "",
}: {
  text: string;
  startMs?: number;
  perLetterMs?: number;
  className?: string;
}) {
  return (
    <span className={className} aria-label={text}>
      {Array.from(text).map((ch, i) => (
        <span
          key={i}
          className="pmc-letter"
          aria-hidden="true"
          style={{ animationDelay: `${startMs + i * perLetterMs}ms` }}
        >
          {ch === " " ? " " : ch}
        </span>
      ))}
    </span>
  );
}
