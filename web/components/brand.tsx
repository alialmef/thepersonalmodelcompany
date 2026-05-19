import { cn } from "@/lib/utils";

interface BrandProps {
  className?: string;
  size?: "small" | "medium" | "large";
}

const sizeClasses = {
  small: "text-[15px] tracking-tight",
  medium: "text-2xl tracking-tight",
  large: "text-5xl md:text-7xl tracking-[-0.03em] leading-[0.95]",
} as const;

/**
 * The wordmark. Used as the homepage hero AND as a small top-left signature on
 * inner pages. Keep it as type, not a logo. Apple does this with "Mac" and
 * "iPhone" — they're words first.
 */
export function Brand({ className, size = "medium" }: BrandProps) {
  return (
    <span
      className={cn(
        "font-medium text-foreground",
        sizeClasses[size],
        className,
      )}
    >
      The Personal Model Company
    </span>
  );
}
