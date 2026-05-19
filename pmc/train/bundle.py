"""ArtifactBundle — the "you own it" deliverable.

Per the analysis doc, every user model run produces an artifact bundle the user
fully owns and can host, export, or walk away with:

    my_personal_model/
      bundle.json          # top-level metadata (this file's schema version, IDs, timestamps)
      adapter/             # LoRA weights + adapter_config.json (written by PEFT)
      style_profile.json   # extracted from their writing
      training_manifest.json  # what data trained this model
      eval_report.json     # how it scored (style match, factual, privacy, etc.)
      audit_log.json       # every pipeline event that produced this model
      README.md            # human-readable export instructions

The bundle is pure JSON + the adapter directory. No torch required to read,
serialize, or pack it. That's deliberate: the user shouldn't need PMC's stack
just to inspect what they own.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pmc.schema.user import DataManifest, StyleProfile

BUNDLE_VERSION = "1"

_TOP_LEVEL_FILES = {
    "bundle.json",
    "style_profile.json",
    "training_manifest.json",
    "eval_report.json",
    "audit_log.json",
    "README.md",
}


class AuditEvent(BaseModel):
    """A single event in the model's audit log."""

    timestamp: datetime = Field(default_factory=datetime.now)
    stage: str  # "ingest" | "curate" | "train" | "eval" | "deploy"
    event: str  # short descriptor, e.g. "sft_completed"
    data: dict[str, Any] = Field(default_factory=dict)


class BundleMetadata(BaseModel):
    """Top-level metadata describing what's in this bundle."""

    bundle_version: str = BUNDLE_VERSION
    user_id: str
    user_name: str | None = None
    user_email: str | None = None
    base_model: str
    job_type: str  # "sft" | "dpo" | "sft+dpo"
    created_at: datetime = Field(default_factory=datetime.now)
    adapter_checksum: str | None = None
    notes: str = ""


class ArtifactBundle:
    """Bundle = adapter directory + JSON sidecars + README.

    Build with `ArtifactBundle(...)`, write to disk with `.write(path)`, read
    back with `ArtifactBundle.load(path)`, pack into a zip with `.to_zip(path)`.
    """

    def __init__(
        self,
        metadata: BundleMetadata,
        adapter_dir: Path,
        style_profile: StyleProfile | None = None,
        training_manifest: DataManifest | None = None,
        eval_report: dict[str, Any] | None = None,
        audit_log: list[AuditEvent] | None = None,
    ) -> None:
        self.metadata = metadata
        self.adapter_dir = Path(adapter_dir)
        self.style_profile = style_profile
        self.training_manifest = training_manifest
        self.eval_report = eval_report or {}
        self.audit_log = audit_log or []

    def append_audit(self, stage: str, event: str, data: dict[str, Any] | None = None) -> None:
        self.audit_log.append(AuditEvent(stage=stage, event=event, data=data or {}))

    def write(self, output_dir: Path | str, *, copy_adapter: bool = True) -> Path:
        """Materialize the bundle on disk.

        If `copy_adapter` is True, copies adapter files into `output_dir/adapter/`.
        If False, the adapter is assumed to already be at `output_dir/adapter/` (or
        wherever `self.adapter_dir` points). Either way the checksum is recorded.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        target_adapter = out / "adapter"
        if copy_adapter and self.adapter_dir.exists() and self.adapter_dir.resolve() != target_adapter.resolve():
            _copy_adapter_dir(self.adapter_dir, target_adapter)

        # Compute checksum from whichever adapter dir is canonical now
        checksum_source = target_adapter if target_adapter.is_dir() else self.adapter_dir
        if checksum_source.is_dir():
            self.metadata.adapter_checksum = _checksum_dir(checksum_source)

        (out / "bundle.json").write_text(self.metadata.model_dump_json(indent=2))

        if self.style_profile is not None:
            (out / "style_profile.json").write_text(
                self.style_profile.model_dump_json(indent=2)
            )

        if self.training_manifest is not None:
            (out / "training_manifest.json").write_text(
                self.training_manifest.model_dump_json(indent=2)
            )

        (out / "eval_report.json").write_text(json.dumps(self.eval_report, indent=2, default=str))

        audit_data = [e.model_dump(mode="json") for e in self.audit_log]
        (out / "audit_log.json").write_text(json.dumps(audit_data, indent=2, default=str))

        (out / "README.md").write_text(self._render_readme())

        return out

    def to_zip(self, output_zip: Path | str) -> Path:
        """Package the bundle into a single .zip for download."""
        import tempfile

        zip_path = Path(output_zip)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp) / "bundle"
            self.write(tmp_dir, copy_adapter=True)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in tmp_dir.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(tmp_dir))
        return zip_path

    @classmethod
    def load(cls, bundle_dir: Path | str) -> ArtifactBundle:
        path = Path(bundle_dir)
        metadata = BundleMetadata.model_validate_json((path / "bundle.json").read_text())

        style_profile = None
        if (path / "style_profile.json").is_file():
            style_profile = StyleProfile.model_validate_json(
                (path / "style_profile.json").read_text()
            )

        manifest = None
        if (path / "training_manifest.json").is_file():
            manifest = DataManifest.model_validate_json(
                (path / "training_manifest.json").read_text()
            )

        eval_report: dict[str, Any] = {}
        if (path / "eval_report.json").is_file():
            eval_report = json.loads((path / "eval_report.json").read_text())

        audit_log: list[AuditEvent] = []
        if (path / "audit_log.json").is_file():
            raw = json.loads((path / "audit_log.json").read_text())
            audit_log = [AuditEvent.model_validate(e) for e in raw]

        adapter_dir = path / "adapter"
        return cls(
            metadata=metadata,
            adapter_dir=adapter_dir,
            style_profile=style_profile,
            training_manifest=manifest,
            eval_report=eval_report,
            audit_log=audit_log,
        )

    def _render_readme(self) -> str:
        m = self.metadata
        return f"""# Personal Model Bundle

This directory contains your personal AI model and everything that produced it.
You own all of it. You can host it, export it, or walk away with it.

## What's inside

| File | What it is |
|------|-----------|
| `bundle.json` | Top-level metadata about this model |
| `adapter/` | LoRA adapter weights (`adapter_model.safetensors`) |
| `style_profile.json` | Your writing style as extracted by curation |
| `training_manifest.json` | Which data sources trained this model |
| `eval_report.json` | How the model scored on style/factual/privacy evals |
| `audit_log.json` | Full event log of how this model was produced |

## How to use it

### Load locally with HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{m.base_model}")
tokenizer = AutoTokenizer.from_pretrained("{m.base_model}")
model = PeftModel.from_pretrained(base, "./adapter")

messages = [{{"role": "user", "content": "Draft a reply to this email..."}}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
output = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Merge into a standalone model

If you want a self-contained model (no PEFT runtime needed), merge the adapter:

```python
from pmc.train.checkpoint import merge_adapter_into_base
merge_adapter_into_base("{m.base_model}", "./adapter", "./merged_model")
```

### Export to GGUF (for Ollama / llama.cpp)

After merging, convert with llama.cpp's `convert_hf_to_gguf.py`.

## Provenance

- **Created**: {m.created_at.isoformat()}
- **Base model**: `{m.base_model}`
- **Training type**: {m.job_type}
- **User**: {m.user_name or m.user_id}
- **Bundle version**: {m.bundle_version}

See `audit_log.json` for the full event timeline of this model's creation.
"""


def _copy_adapter_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _checksum_dir(directory: Path) -> str:
    """Stable SHA-256 over the directory contents (sorted by relative path)."""
    h = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(directory)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()
