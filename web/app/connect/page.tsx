"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * /connect — redirect-only as of the auto-opt-in redesign.
 *
 * The old per-source picker (a grid of Connect buttons for Messages,
 * Notes, Mail, Documents) is gone. The product is now auto-opt-in:
 * sign in, land on /reading, the macOS Full Disk Access prompt fires
 * once, all sources start ingesting in parallel.
 *
 * The redact + manage surface (what was the *idea* behind a source
 * picker — letting users control what's read) now lives at
 * /knowledge-update where it belongs: after the fact, queryable,
 * with pause + private + forget primitives.
 *
 * This route exists for back-compat — any old in-app deep link or
 * stored "next" URL pointing here just bounces to /reading.
 */
export default function ConnectPage() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/reading");
  }, [router]);
  return <main className="min-h-screen w-full bg-background" />;
}
