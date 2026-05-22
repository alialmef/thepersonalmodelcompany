"""Full-laptop file adapter with simulate/stage/execute/undo support."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pmc.actions.adapters.base import (
    ActionAdapterCapability,
    ActionExecutionMode,
    ActionExecutionReceipt,
    ActionExecutionRequest,
)
from pmc.actions.schema import ActionRisk
from pmc.storage.paths import StoragePaths, safe_id


class LocalFilesAdapter:
    """Adapter for local file and note mutations.

    It intentionally accepts absolute paths: the product model is allowed to
    understand the laptop as a whole. Writes remain explicit, receipt-backed,
    and undoable when the OS grants access.
    """

    def __init__(self, storage_root: Path | str) -> None:
        self.paths = StoragePaths(storage_root)

    @property
    def capabilities(self) -> list[ActionAdapterCapability]:
        return [
            ActionAdapterCapability(
                surface="files",
                operation="read",
                risk_level=ActionRisk.LOW,
                supports_simulate=True,
                supports_stage=False,
                supports_execute=False,
                supports_undo=False,
                requires_confirmation=False,
                description="Read and preview a local file.",
            ),
            ActionAdapterCapability(
                surface="files",
                operation="write_text",
                risk_level=ActionRisk.MEDIUM,
                supports_simulate=True,
                supports_stage=True,
                supports_execute=True,
                supports_undo=True,
                description="Write text to a local file with an undo snapshot.",
            ),
            ActionAdapterCapability(
                surface="files",
                operation="append_text",
                risk_level=ActionRisk.MEDIUM,
                supports_simulate=True,
                supports_stage=True,
                supports_execute=True,
                supports_undo=True,
                description="Append text to a local file with an undo snapshot.",
            ),
            ActionAdapterCapability(
                surface="files",
                operation="replace_text",
                risk_level=ActionRisk.MEDIUM,
                supports_simulate=True,
                supports_stage=True,
                supports_execute=True,
                supports_undo=True,
                description="Replace text in a local file with an undo snapshot.",
            ),
            ActionAdapterCapability(
                surface="notes",
                operation="create",
                risk_level=ActionRisk.MEDIUM,
                supports_simulate=True,
                supports_stage=True,
                supports_execute=True,
                supports_undo=True,
                description="Create a local markdown note with an undo snapshot.",
            ),
        ]

    def can_handle(self, surface: str, operation: str) -> bool:
        return any(c.surface == surface and c.operation == operation for c in self.capabilities)

    def run(self, request: ActionExecutionRequest) -> ActionExecutionReceipt:
        payload = {**request.proposal.proposed_payload, **request.payload}
        if request.mode == ActionExecutionMode.UNDO:
            return self._undo(request, payload)
        if request.proposal.surface == "files" and request.proposal.operation == "read":
            return self._read(request, payload)
        if request.proposal.surface == "files" and request.proposal.operation == "write_text":
            return self._write_text(request, payload)
        if request.proposal.surface == "files" and request.proposal.operation == "append_text":
            return self._append_text(request, payload)
        if request.proposal.surface == "files" and request.proposal.operation == "replace_text":
            return self._replace_text(request, payload)
        if request.proposal.surface == "notes" and request.proposal.operation == "create":
            return self._create_note(request, payload)
        return self._receipt(request, ok=False, error="Unsupported local file operation")

    def _read(self, request: ActionExecutionRequest, payload: dict[str, Any]) -> ActionExecutionReceipt:
        path = self._required_path(payload)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return self._receipt(request, ok=False, error=str(e), evidence={"path": str(path)})
        return self._receipt(
            request,
            ok=True,
            preview=text[:4_000],
            evidence={
                "path": str(path),
                "size_bytes": path.stat().st_size,
            },
        )

    def _write_text(
        self,
        request: ActionExecutionRequest,
        payload: dict[str, Any],
    ) -> ActionExecutionReceipt:
        path = self._required_path(payload)
        content = str(payload.get("content") or payload.get("text") or "")
        if request.mode != ActionExecutionMode.EXECUTE:
            return self._receipt(
                request,
                ok=True,
                preview=content[:4_000],
                evidence=self._file_evidence(path, intended_size=len(content.encode())),
            )
        token = self._backup(request.user_id, path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            return self._receipt(request, ok=False, error=str(e), evidence={"path": str(path)})
        return self._receipt(
            request,
            ok=True,
            preview=content[:4_000],
            side_effects=[f"wrote {path}"],
            undo_token=token,
            evidence=self._file_evidence(path, intended_size=len(content.encode())),
        )

    def _append_text(
        self,
        request: ActionExecutionRequest,
        payload: dict[str, Any],
    ) -> ActionExecutionReceipt:
        path = self._required_path(payload)
        content = str(payload.get("content") or payload.get("text") or "")
        preview = f"{self._safe_preview(path)}{content}"[-4_000:]
        if request.mode != ActionExecutionMode.EXECUTE:
            return self._receipt(
                request,
                ok=True,
                preview=preview,
                evidence=self._file_evidence(path, intended_size=len(content.encode())),
            )
        token = self._backup(request.user_id, path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            return self._receipt(request, ok=False, error=str(e), evidence={"path": str(path)})
        return self._receipt(
            request,
            ok=True,
            preview=preview,
            side_effects=[f"appended {path}"],
            undo_token=token,
            evidence=self._file_evidence(path, intended_size=len(content.encode())),
        )

    def _replace_text(
        self,
        request: ActionExecutionRequest,
        payload: dict[str, Any],
    ) -> ActionExecutionReceipt:
        path = self._required_path(payload)
        old_text = str(payload.get("old_text") or "")
        new_text = str(payload.get("new_text") or "")
        current = self._safe_text(path)
        if old_text not in current:
            return self._receipt(
                request,
                ok=False,
                error="old_text was not found",
                evidence={"path": str(path)},
            )
        updated = current.replace(old_text, new_text, 1)
        if request.mode != ActionExecutionMode.EXECUTE:
            return self._receipt(
                request,
                ok=True,
                preview=updated[:4_000],
                evidence={
                    **self._file_evidence(path, intended_size=len(updated.encode())),
                    "old_text_found": True,
                },
            )
        token = self._backup(request.user_id, path)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as e:
            return self._receipt(request, ok=False, error=str(e), evidence={"path": str(path)})
        return self._receipt(
            request,
            ok=True,
            preview=updated[:4_000],
            side_effects=[f"edited {path}"],
            undo_token=token,
            evidence=self._file_evidence(path, intended_size=len(updated.encode())),
        )

    def _create_note(
        self,
        request: ActionExecutionRequest,
        payload: dict[str, Any],
    ) -> ActionExecutionReceipt:
        directory = Path(
            str(payload.get("directory") or Path.home() / "Documents" / "PMC Notes")
        ).expanduser()
        title = str(payload.get("title") or "Untitled").strip() or "Untitled"
        content = str(payload.get("content") or payload.get("text") or "")
        path = Path(str(payload.get("path") or directory / f"{safe_id(title, max_len=80)}.md"))
        payload = {**payload, "path": str(path), "content": content}
        return self._write_text(request, payload)

    def _undo(
        self,
        request: ActionExecutionRequest,
        payload: dict[str, Any],
    ) -> ActionExecutionReceipt:
        token = str(payload.get("undo_token") or payload.get("token") or "")
        if not token:
            return self._receipt(request, ok=False, error="undo_token is required")
        backup_path = self._undo_path(request.user_id, token)
        if not backup_path.is_file():
            return self._receipt(request, ok=False, error=f"No undo snapshot {token!r}")
        record = json.loads(backup_path.read_text(encoding="utf-8"))
        target = Path(record["path"])
        existed = bool(record["existed"])
        try:
            if existed:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(record.get("content") or ""), encoding="utf-8")
                side_effects = [f"restored {target}"]
            else:
                target.unlink(missing_ok=True)
                side_effects = [f"removed {target}"]
        except OSError as e:
            return self._receipt(request, ok=False, error=str(e), evidence={"path": str(target)})
        return self._receipt(
            request,
            ok=True,
            preview=self._safe_preview(target),
            side_effects=side_effects,
            evidence={"path": str(target), "undo_token": token},
        )

    def _backup(self, user_id: str, path: Path) -> str:
        token = f"undo-{uuid.uuid4().hex[:16]}"
        undo_path = self._undo_path(user_id, token)
        undo_path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        content = self._safe_text(path) if existed else ""
        undo_path.write_text(
            json.dumps(
                {
                    "path": str(path),
                    "existed": existed,
                    "content": content,
                }
            ),
            encoding="utf-8",
        )
        return token

    def _undo_path(self, user_id: str, token: str) -> Path:
        return self.paths.action_undo_dir(user_id) / f"{safe_id(token)}.json"

    @staticmethod
    def _required_path(payload: dict[str, Any]) -> Path:
        raw = str(payload.get("path") or "").strip()
        if not raw:
            raise ValueError("path is required")
        return Path(raw).expanduser()

    @staticmethod
    def _safe_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _safe_preview(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:4_000]
        except OSError:
            return ""

    @staticmethod
    def _file_evidence(path: Path, *, intended_size: int) -> dict[str, Any]:
        return {
            "path": str(path),
            "exists": path.exists(),
            "current_size_bytes": path.stat().st_size if path.exists() else 0,
            "intended_size_bytes": intended_size,
        }

    @staticmethod
    def _receipt(
        request: ActionExecutionRequest,
        *,
        ok: bool,
        preview: str = "",
        evidence: dict[str, Any] | None = None,
        side_effects: list[str] | None = None,
        undo_token: str | None = None,
        error: str | None = None,
    ) -> ActionExecutionReceipt:
        return ActionExecutionReceipt(
            user_id=request.user_id,
            proposal_id=request.proposal.id,
            surface=request.proposal.surface,
            operation=request.proposal.operation,
            mode=request.mode,
            ok=ok,
            preview=preview,
            evidence=evidence or {},
            side_effects=side_effects or [],
            undo_token=undo_token,
            error=error,
        )
