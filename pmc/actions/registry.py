"""Registry for local and MCP action adapters."""

from __future__ import annotations

from pathlib import Path

from pmc.actions.adapters.base import ActionAdapter, ActionAdapterCapability
from pmc.actions.adapters.local_files import LocalFilesAdapter


class ActionAdapterRegistry:
    """In-process adapter registry keyed by surface + operation."""

    def __init__(self) -> None:
        self._adapters: list[ActionAdapter] = []

    def register(self, adapter: ActionAdapter) -> None:
        self._adapters.append(adapter)

    def capabilities(self) -> list[ActionAdapterCapability]:
        out: list[ActionAdapterCapability] = []
        for adapter in self._adapters:
            out.extend(adapter.capabilities)
        return out

    def find(self, surface: str, operation: str) -> ActionAdapter | None:
        for adapter in self._adapters:
            if adapter.can_handle(surface, operation):
                return adapter
        return None


def build_default_action_registry(storage_root: Path | str) -> ActionAdapterRegistry:
    registry = ActionAdapterRegistry()
    registry.register(LocalFilesAdapter(storage_root=storage_root))
    return registry
