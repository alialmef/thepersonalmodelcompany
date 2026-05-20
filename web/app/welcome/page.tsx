"use client";

import { useRouter } from "next/navigation";
import FirstLaunchScreen from "@/components/app/first-launch-screen";

/**
 * Screen 1 — First Launch.
 *
 * The Mac app opens here. Pure white. Three seconds of held silence between
 * "Hello." and the second line. Do not shorten.
 *
 * On Begin → /connect (Step 1 of 3).
 */
export default function WelcomePage() {
  const router = useRouter();
  return <FirstLaunchScreen onBegin={() => router.push("/connect")} />;
}
