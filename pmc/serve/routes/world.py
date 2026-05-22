"""HTTP routes for the local laptop-world index."""

from __future__ import annotations

from typing import Any

from pmc.storage.audit import AuditLog
from pmc.world import LaptopWorldScanner, WorldScanConfig, WorldStore


def build_world_router(world_store: WorldStore, audit_log: AuditLog) -> Any:
    from fastapi import APIRouter, HTTPException, Query

    router = APIRouter(prefix="/v1/users/{user_id}/world", tags=["world"])
    scanner = LaptopWorldScanner()

    @router.post("/scan")
    def scan_world(user_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            config = WorldScanConfig.model_validate(payload or {})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid world scan config: {e}") from e
        report, entries = scanner.scan(user_id, config)
        world_store.save_scan(user_id, report, entries)
        audit_log.log(
            user_id,
            stage="memory",
            event="world_scan_completed",
            data={
                "scan_id": report.id,
                "roots": report.roots,
                "full_disk_requested": report.full_disk_requested,
                "files_indexed": report.files_indexed,
                "bytes_indexed": report.bytes_indexed,
                "errors": len(report.errors),
            },
        )
        return {
            "ok": True,
            "scan": report.model_dump(mode="json"),
            "indexed": len(entries),
        }

    @router.get("/files")
    def list_world_files(
        user_id: str,
        query: str | None = Query(default=None),
        limit: int = Query(default=100),
    ) -> dict[str, Any]:
        entries = world_store.list_entries(user_id, query=query, limit=limit)
        return {
            "files": [entry.model_dump(mode="json") for entry in entries],
            "latest_scan": (
                latest.model_dump(mode="json")
                if (latest := world_store.latest_scan(user_id)) is not None
                else None
            ),
        }

    @router.get("/scans/latest")
    def latest_world_scan(user_id: str) -> dict[str, Any]:
        latest = world_store.latest_scan(user_id)
        return {
            "scan": latest.model_dump(mode="json") if latest is not None else None,
        }

    return router
