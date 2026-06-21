"""Voice memo transcription.

The Rust `voice_memos` extractor enumerates audio files but doesn't
transcribe them — Whisper has to live outside the Tauri binary
because shipping a 150MB model + native deps is hostile to first
launch. This module fills the gap on the Python side.

Pipeline:
    1. Read FileSignal entries with kind="voice_memo" from the graph
    2. For each audio file that doesn't yet have a transcript on disk,
       run whisper-cli (via ffmpeg → wav conversion)
    3. Persist transcripts to
       <storage_root>/users/<uid>/graph/synth/transcripts/<file_id>.txt
    4. Return a manifest: per-memo file_id, audio_path, transcript_path,
       text excerpt, duration estimate, modified-at

The transcripts can then be fed into the `build_threads` pass or any
future text-aware synthesis.

System requirements (the caller's environment, not bundled):
    whisper-cli (brew install whisper-cpp)
    ffmpeg      (brew install ffmpeg)
    whisper model (downloaded once; see WHISPER_MODEL_PATH)

If either binary is missing this module is a no-op that emits a clear
status message.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pmc.storage.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


WHISPER_MODEL_PATH = Path(
    os.environ.get(
        "PMC_WHISPER_MODEL",
        str(Path.home() / ".cache/pmc/whisper-models/ggml-base.en.bin"),
    )
)
WHISPER_CLI = os.environ.get("PMC_WHISPER_CLI", "whisper-cli")
FFMPEG_CLI = os.environ.get("PMC_FFMPEG_CLI", "ffmpeg")


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class Transcript:
    file_id: str
    audio_path: str
    transcript_path: str
    text_excerpt: str
    text_chars: int


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _transcripts_dir(storage_root: Path | str, user_id: str) -> Path:
    p = Path(storage_root) / "users" / user_id / "graph" / "synth" / "transcripts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _manifest_path(storage_root: Path | str, user_id: str) -> Path:
    return _transcripts_dir(storage_root, user_id) / "_manifest.jsonl"


# ---------------------------------------------------------------------------
# Tool availability
# ---------------------------------------------------------------------------


@dataclass
class ToolStatus:
    ok: bool
    reason: str = ""


def check_tools() -> ToolStatus:
    if shutil.which(WHISPER_CLI) is None:
        return ToolStatus(False, f"{WHISPER_CLI!r} not on PATH (brew install whisper-cpp)")
    if shutil.which(FFMPEG_CLI) is None:
        return ToolStatus(False, f"{FFMPEG_CLI!r} not on PATH (brew install ffmpeg)")
    if not WHISPER_MODEL_PATH.is_file():
        return ToolStatus(
            False,
            f"whisper model missing at {WHISPER_MODEL_PATH} "
            f"(download ggml-base.en.bin from huggingface.co/ggerganov/whisper.cpp)",
        )
    return ToolStatus(True)


# ---------------------------------------------------------------------------
# One-file transcription
# ---------------------------------------------------------------------------


def _audio_to_wav(audio_path: Path, dst_wav: Path) -> bool:
    """Convert audio to 16kHz mono PCM-16 WAV for whisper.cpp."""
    try:
        result = subprocess.run(
            [
                FFMPEG_CLI, "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le",
                str(dst_wav),
            ],
            capture_output=True, timeout=120,
        )
        return result.returncode == 0 and dst_wav.is_file() and dst_wav.stat().st_size > 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _whisper_transcribe(wav_path: Path) -> str:
    """Run whisper-cli on a WAV file. Returns the transcript text or empty."""
    try:
        result = subprocess.run(
            [
                WHISPER_CLI,
                "-m", str(WHISPER_MODEL_PATH),
                "-nt", "-np",
                "-l", "en",
                "-f", str(wav_path),
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return ""
        # whisper-cli prints the transcript to stdout (with -nt -np
        # suppressing timestamps and progress). Trim noise lines.
        text_lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.strip().startswith(("[", "system_info"))
        ]
        return "\n".join(text_lines).strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""


def transcribe_one(
    audio_path: Path,
    *,
    out_dir: Path,
    file_id: str,
    force: bool = False,
) -> Transcript | None:
    """Transcribe one audio file. Skips if a transcript already exists
    at `<out_dir>/<file_id>.txt` unless `force` is set."""
    transcript_path = out_dir / f"{file_id}.txt"
    if transcript_path.is_file() and not force:
        text = transcript_path.read_text(errors="replace")
        return Transcript(
            file_id=file_id,
            audio_path=str(audio_path),
            transcript_path=str(transcript_path),
            text_excerpt=text[:300],
            text_chars=len(text),
        )

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "input.wav"
        if not _audio_to_wav(audio_path, wav):
            return None
        text = _whisper_transcribe(wav)
        if not text:
            return None

    transcript_path.write_text(text)
    return Transcript(
        file_id=file_id,
        audio_path=str(audio_path),
        transcript_path=str(transcript_path),
        text_excerpt=text[:300],
        text_chars=len(text),
    )


# ---------------------------------------------------------------------------
# Batch over the graph
# ---------------------------------------------------------------------------


def _voice_memo_signals(graph_store: GraphStore, user_id: str) -> Iterable[dict]:
    """Yield FileSignal entries for voice memos from the graph."""
    for v in graph_store.iter_entities(user_id, "file"):
        if v.get("kind") == "voice_memo":
            yield v


def transcribe_voice_memos(
    *,
    graph_store: GraphStore,
    storage_root: Path | str,
    user_id: str,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Run Whisper across every voice-memo FileSignal in the user's
    graph. Returns a summary dict the caller can render or log.

    The first call after install is slow (Metal init + first decode).
    Subsequent calls only process new memos.
    """
    status = check_tools()
    if not status.ok:
        return {"ok": False, "reason": status.reason, "transcripts": [], "stats": {}}

    out_dir = _transcripts_dir(storage_root, user_id)
    manifest_path = _manifest_path(storage_root, user_id)

    transcripts: list[Transcript] = []
    skipped = 0
    failed = 0
    new_count = 0

    signals = list(_voice_memo_signals(graph_store, user_id))
    if limit is not None:
        signals = signals[:limit]

    for sig in signals:
        file_id = sig.get("id") or ""
        path_str = sig.get("path") or ""
        if not file_id or not path_str:
            skipped += 1
            continue
        audio_path = Path(path_str)
        if not audio_path.is_file():
            skipped += 1
            continue

        existed = (out_dir / f"{file_id}.txt").is_file()
        t = transcribe_one(audio_path, out_dir=out_dir, file_id=file_id, force=force)
        if t is None:
            failed += 1
            continue
        transcripts.append(t)
        if not existed:
            new_count += 1

    # Write manifest
    with manifest_path.open("w") as f:
        for t in transcripts:
            f.write(json.dumps(asdict(t)) + "\n")

    return {
        "ok": True,
        "reason": "",
        "stats": {
            "total_voice_memos": len(signals),
            "transcripts_present": len(transcripts),
            "new_this_run": new_count,
            "skipped_missing_file_or_id": skipped,
            "failed_transcription": failed,
        },
        "transcripts": [asdict(t) for t in transcripts],
    }


def load_transcripts(storage_root: Path | str, user_id: str) -> list[Transcript]:
    p = _manifest_path(storage_root, user_id)
    if not p.is_file():
        return []
    out: list[Transcript] = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(Transcript(**d))
        except Exception:
            continue
    return out
