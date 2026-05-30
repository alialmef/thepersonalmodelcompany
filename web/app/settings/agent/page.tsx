"use client";

import AgentSettingsScreen from "@/components/app/agent-settings-screen";

/**
 * /settings/agent — BYOM provider configuration screen.
 *
 * Reachable from the settings drawer's "Configure agent" row. The
 * underlying screen handles its own loading + save + disconnect.
 */
export default function AgentSettingsPage() {
  return <AgentSettingsScreen />;
}
