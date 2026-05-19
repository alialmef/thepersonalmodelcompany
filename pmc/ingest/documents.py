"""Document ingest: PDF and DOCX.

Heavy parsers are lazy-imported so the rest of the package works without them.
Each document becomes a single RawItem — the full text of the document.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pmc.ingest.base import Ingestor, RawItem
from pmc.schema.conversation import SourceType


class DocumentIngestor(Ingestor):
    """Read PDF and DOCX files. Use the [ingest] extras to install parsers."""

    source_type = SourceType.DOCUMENT

    def ingest(self, source: Path | str) -> Iterator[RawItem]:
        path = Path(source)
        files = self._collect_files(path)
        for file_path in files:
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".pdf":
                    content = self._read_pdf(file_path)
                elif suffix == ".docx":
                    content = self._read_docx(file_path)
                else:
                    continue
            except Exception:
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
                    "format": suffix.lstrip("."),
                    "size_bytes": str(stat.st_size),
                },
            )

    def _collect_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in {".pdf", ".docx"} else []
        if path.is_dir():
            return sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}
            )
        return []

    def _read_pdf(self, path: Path) -> str:
        try:
            from PyPDF2 import PdfReader
        except ImportError as e:
            raise ImportError("Install pmc[ingest] to read PDFs") from e
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    def _read_docx(self, path: Path) -> str:
        try:
            from docx import Document
        except ImportError as e:
            raise ImportError("Install pmc[ingest] to read DOCX files") from e
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
