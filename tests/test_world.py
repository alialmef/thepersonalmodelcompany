"""Tests for laptop-world scanning and storage."""

from __future__ import annotations

from pathlib import Path

from pmc.world import LaptopWorldScanner, WorldScanConfig, WorldStore


def test_laptop_world_scanner_indexes_text_files(tmp_path: Path):
    root = tmp_path / "world"
    root.mkdir()
    (root / "notes.md").write_text("remember the tone", encoding="utf-8")
    (root / "image.bin").write_bytes(b"\x00\x01")

    report, entries = LaptopWorldScanner().scan(
        "alex",
        WorldScanConfig(
            roots=[str(root)],
            full_disk=False,
            max_files=10,
        ),
    )

    assert report.files_indexed == 2
    note = next(entry for entry in entries if entry.name == "notes.md")
    assert note.kind == "document"
    assert note.content_preview == "remember the tone"


def test_world_store_saves_latest_index_and_reports(tmp_path: Path):
    root = tmp_path / "world"
    root.mkdir()
    (root / "project.py").write_text("print('hi')", encoding="utf-8")
    report, entries = LaptopWorldScanner().scan(
        "alex",
        WorldScanConfig(roots=[str(root)], full_disk=False),
    )

    store = WorldStore(tmp_path / "storage")
    store.save_scan("alex", report, entries)

    assert store.latest_scan("alex") is not None
    files = store.list_entries("alex", query="project")
    assert len(files) == 1
    assert files[0].name == "project.py"
