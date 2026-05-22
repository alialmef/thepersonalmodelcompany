"""Durable laptop-world index storage."""

from __future__ import annotations

from pathlib import Path

from pmc.storage.paths import StoragePaths
from pmc.world.schema import WorldFile, WorldScanReport


class WorldStore:
    """Stores the latest file index plus append-only scan reports."""

    def __init__(self, root: Path | str) -> None:
        self.paths = StoragePaths(root)

    def save_scan(
        self,
        user_id: str,
        report: WorldScanReport,
        entries: list[WorldFile],
        *,
        replace_index: bool = True,
    ) -> None:
        self.append_scan_report(user_id, report)
        self.save_entries(user_id, entries, append=not replace_index)

    def append_scan_report(self, user_id: str, report: WorldScanReport) -> None:
        path = self.paths.world_scans_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(report.model_dump_json() + "\n")

    def save_entries(
        self,
        user_id: str,
        entries: list[WorldFile],
        *,
        append: bool = False,
    ) -> int:
        path = self.paths.world_files_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append and path.is_file() else "w"
        with path.open(mode, encoding="utf-8") as f:
            for entry in entries:
                f.write(entry.model_dump_json() + "\n")
        return len(entries)

    def list_entries(
        self,
        user_id: str,
        *,
        query: str | None = None,
        limit: int | None = 100,
    ) -> list[WorldFile]:
        entries = self._read_jsonl(self.paths.world_files_file(user_id), WorldFile)
        if query:
            needle = query.lower()
            entries = [
                entry for entry in entries
                if needle in entry.path.lower()
                or needle in entry.name.lower()
                or needle in entry.content_preview.lower()
            ]
        if limit is not None and limit >= 0:
            entries = entries[-limit:]
        return entries

    def list_scans(self, user_id: str, *, limit: int | None = None) -> list[WorldScanReport]:
        scans = self._read_jsonl(self.paths.world_scans_file(user_id), WorldScanReport)
        if limit is not None and limit >= 0:
            scans = scans[-limit:]
        return scans

    def latest_scan(self, user_id: str) -> WorldScanReport | None:
        scans = self.list_scans(user_id, limit=1)
        return scans[0] if scans else None

    @staticmethod
    def _read_jsonl(path: Path, model: type[WorldFile] | type[WorldScanReport]) -> list:
        if not path.exists():
            return []
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(model.model_validate_json(line))
        return out
