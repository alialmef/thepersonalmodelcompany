"""Plain text and markdown file ingest.

Simplest possible source: a file or directory of text files. Each file becomes
a single RawItem with no thread or author — treated as standalone user writing.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pmc.ingest.base import Ingestor, RawItem
from pmc.schema.conversation import SourceType

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".text"}


class TextFileIngestor(Ingestor):
    """Read .txt / .md files from a path. The path can be a file or directory."""

    source_type = SourceType.NOTES

    def __init__(self, extensions: set[str] | None = None) -> None:
        self.extensions = extensions or TEXT_EXTENSIONS

    def ingest(self, source: Path | str) -> Iterator[RawItem]:
        path = Path(source)
        files = self._collect_files(path)
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            content = content.strip()
            if not content:
                continue
            stat = file_path.stat()
            yield RawItem(
                source_type=self.source_type,
                source_id=str(file_path.resolve()),
                content=content,
                timestamp=datetime.fromtimestamp(stat.st_mtime),
                is_user=True,
                subject=file_path.stem,
                metadata={
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "size_bytes": str(stat.st_size),
                },
            )

    def _collect_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in self.extensions else []
        if path.is_dir():
            return sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in self.extensions
            )
        return []
