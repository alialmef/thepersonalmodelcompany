'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * Fires once when the referenced element enters the viewport at the given
 * threshold of visibility. Used to scroll-trigger section choreographies on
 * the landing page. After firing, the observer disconnects — animations
 * never replay in the same session.
 *
 * Pair with a className like `pmc-in-view` on the section, and have CSS
 * keyframes apply only when that class is present:
 *
 *   .pmc-in-view .my-element { animation: ... forwards; }
 */
export function useInView<T extends Element>(threshold = 0.15) {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node || inView) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      {
        threshold,
        // Fire slightly before the bottom of the section reaches the bottom of
        // the viewport — animations start the moment the section is meaningfully
        // visible, not only when it dominates the screen.
        rootMargin: '0px 0px -10% 0px',
      },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [threshold, inView]);

  return [ref, inView] as const;
}
