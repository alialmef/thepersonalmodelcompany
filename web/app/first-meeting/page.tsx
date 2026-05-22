"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * The first-meeting screen used to be a separate ceremonial moment —
 * the model said one opening line, the user typed back, the app
 * rolled into /chat.
 *
 * That was theater. The user couldn't tell whether the line was
 * actually their model speaking, and the model had nothing at stake.
 *
 * The first meeting is now the verification flow: the model
 * demonstrates voice across multiple situations, the user
 * approves / edits / rejects each, and the relationship only opens
 * once the user has actually verified the model can speak as them.
 * That work lives on /eval.
 *
 * This page exists as a redirect so any stale link or saved bookmark
 * doesn't dead-end the user.
 */
export default function FirstMeetingPage() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/eval");
  }, [router]);
  return <main className="min-h-screen w-full bg-background" />;
}
