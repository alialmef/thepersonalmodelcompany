"""Local laptop-world scanner.

This layer is intentionally broad-read and side-effect free. It walks the
roots the OS grants us, records provenance, and tolerates permission failures.
"""

from __future__ import annotations

import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pmc.world.schema import WorldFile, WorldFileKind, WorldScanConfig, WorldScanReport


EXCLUDED_DIR_NAMES = {
    ".cache",
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "Caches",
    "DerivedData",
    "Library/Caches",
    "node_modules",
}

TEXT_EXTENSIONS = {
    ".csv",
    ".css",
    ".html",
    ".ipynb",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".log",
    ".md",
    ".mdx",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

DOCUMENT_EXTENSIONS = {
    ".doc",
    ".docx",
    ".md",
    ".pdf",
    ".rtf",
    ".txt",
}
IMAGE_EXTENSIONS = {".gif", ".heic", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
AUDIO_EXTENSIONS = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".wav"}
VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".mkv", ".webm"}
ARCHIVE_EXTENSIONS = {".7z", ".dmg", ".gz", ".rar", ".tar", ".tgz", ".zip"}
DATA_EXTENSIONS = {".db", ".parquet", ".sqlite", ".sqlite3"}


def default_laptop_roots(*, full_disk: bool = True) -> list[Path]:
    """Return broad local roots without requiring callers to know the OS."""

    home = Path.home()
    if not full_disk:
        return [home]
    roots = [home]
    if platform.system() == "Darwin":
        roots.extend(Path(p) for p in ("/Users", "/Applications", "/Volumes"))
    else:
        roots.append(Path("/"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen and root.exists():
            deduped.append(root)
            seen.add(key)
    return deduped


class LaptopWorldScanner:
    """Broad-read scanner for building the model's local-world map."""

    def scan(self, user_id: str, config: WorldScanConfig) -> tuple[WorldScanReport, list[WorldFile]]:
        roots = [Path(p).expanduser() for p in config.roots]
        if not roots:
            roots = default_laptop_roots(full_disk=config.full_disk)

        report = WorldScanReport(
            user_id=user_id,
            roots=[str(r) for r in roots],
            full_disk_requested=config.full_disk,
        )
        entries: list[WorldFile] = []

        for path in self._iter_files(roots, config, report):
            if len(entries) >= config.max_files:
                break
            report.files_seen += 1
            try:
                entry = self._entry_for(user_id, path, config)
            except OSError as e:
                self._record_error(report, f"{path}: {e}")
                continue
            if entry is None:
                continue
            report.files_indexed += 1
            report.bytes_indexed += entry.size_bytes
            entries.append(entry)

        report.finished_at = datetime.now()
        return report, entries

    def _iter_files(
        self,
        roots: Iterable[Path],
        config: WorldScanConfig,
        report: WorldScanReport,
    ) -> Iterable[Path]:
        for root in roots:
            if not root.exists():
                self._record_error(report, f"{root}: does not exist")
                continue
            for current, dirs, files in os.walk(root, followlinks=config.follow_symlinks):
                kept_dirs = []
                for name in dirs:
                    if self._should_skip_dir(Path(current) / name):
                        report.dirs_skipped += 1
                    else:
                        kept_dirs.append(name)
                dirs[:] = kept_dirs
                for name in files:
                    yield Path(current) / name

    def _entry_for(
        self,
        user_id: str,
        path: Path,
        config: WorldScanConfig,
    ) -> WorldFile | None:
        stat = path.stat()
        if not path.is_file():
            return None
        extension = path.suffix.lower()
        preview = ""
        if (
            config.include_text_preview
            and extension in TEXT_EXTENSIONS
            and stat.st_size <= config.max_file_bytes
        ):
            preview = self._preview(path)
        return WorldFile(
            user_id=user_id,
            path=str(path),
            name=path.name,
            extension=extension,
            kind=self._kind_for(extension),
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            content_preview=preview,
            metadata={
                "parent": str(path.parent),
            },
        )

    @staticmethod
    def _preview(path: Path, *, max_chars: int = 4_000) -> str:
        raw = path.read_bytes()[: max_chars * 2]
        if b"\x00" in raw:
            return ""
        return raw.decode("utf-8", errors="replace")[:max_chars]

    @staticmethod
    def _kind_for(extension: str) -> WorldFileKind:
        if extension in TEXT_EXTENSIONS and extension not in DOCUMENT_EXTENSIONS:
            return WorldFileKind.CODE
        if extension in DOCUMENT_EXTENSIONS:
            return WorldFileKind.DOCUMENT
        if extension in IMAGE_EXTENSIONS:
            return WorldFileKind.IMAGE
        if extension in AUDIO_EXTENSIONS:
            return WorldFileKind.AUDIO
        if extension in VIDEO_EXTENSIONS:
            return WorldFileKind.VIDEO
        if extension in ARCHIVE_EXTENSIONS:
            return WorldFileKind.ARCHIVE
        if extension in DATA_EXTENSIONS:
            return WorldFileKind.DATA
        return WorldFileKind.OTHER

    @staticmethod
    def _should_skip_dir(path: Path) -> bool:
        parts = set(path.parts)
        if path.name in EXCLUDED_DIR_NAMES:
            return True
        return any(name in parts for name in EXCLUDED_DIR_NAMES if "/" not in name)

    @staticmethod
    def _record_error(report: WorldScanReport, message: str) -> None:
        if len(report.errors) < 50:
            report.errors.append(message)
