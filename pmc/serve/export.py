"""Adapter export — package a user's bundle for download.

Two flavors:
- `export_bundle()`: produce the full ArtifactBundle zip (weights + style
  profile + manifest + eval + audit + README) — the analysis-doc deliverable.
- `export_adapter_only()`: just the LoRA safetensors + config — small file for
  users who only want the weights.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from pmc.serve.registry import AdapterRecord
from pmc.train.bundle import ArtifactBundle, BundleMetadata


def export_bundle(record: AdapterRecord, output_zip: Path | str) -> Path:
    """Export the full ArtifactBundle (including weights, manifest, eval, README) as zip."""
    output = Path(output_zip)
    if record.bundle_dir and Path(record.bundle_dir).is_dir():
        bundle = ArtifactBundle.load(record.bundle_dir)
    else:
        # No persisted bundle — build a minimal one wrapping just the adapter.
        bundle = ArtifactBundle(
            metadata=BundleMetadata(
                user_id=record.user_id,
                base_model=record.base_model,
                job_type="sft",
                notes="Exported on-demand — no persisted bundle was registered.",
            ),
            adapter_dir=Path(record.adapter_dir),
        )
    return bundle.to_zip(output)


def export_adapter_only(record: AdapterRecord, output_zip: Path | str) -> Path:
    """Export just the LoRA adapter (config + safetensors) as zip. Smaller download."""
    output = Path(output_zip)
    adapter_dir = Path(record.adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / f"{record.user_id}_adapter"
        shutil.copytree(adapter_dir, staging)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in staging.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(staging.parent))
    return output
