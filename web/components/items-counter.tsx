"use client";

import { useEffect, useState } from "react";
import { getUserStatus } from "@/lib/api/client";

interface ItemsCounterProps {
  userId: string;
  refreshKey: number;
}

/**
 * Live "X items collected" pill. Refetches user status whenever `refreshKey`
 * changes (parent bumps it after each upload).
 */
export function ItemsCounter({ userId, refreshKey }: ItemsCounterProps) {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    getUserStatus(userId)
      .then((status) => {
        if (!cancelled) setCount(status.raw_item_count);
      })
      .catch(() => {
        if (!cancelled) setCount(null);
      });
    return () => {
      cancelled = true;
    };
  }, [userId, refreshKey]);

  if (count === null) return null;
  if (count === 0) {
    return (
      <span className="text-muted">No items collected yet.</span>
    );
  }
  return (
    <span>
      <span className="text-foreground tabular-nums">
        {count.toLocaleString()}
      </span>{" "}
      <span className="text-muted">items collected.</span>
    </span>
  );
}
