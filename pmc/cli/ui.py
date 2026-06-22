"""Layout primitives for `pmc` terminal surfaces.

Design language: restraint, generous whitespace, a single ice-cyan
accent, reading-column wrapping, speakers-as-labels (not prompts).
The terminal is the canvas, not the layout — we ignore terminal
width and frame everything into a fixed ~72-char column with a
3-char left margin.

Public surface:
    column_width()                  - the reading column width
    margin()                        - left margin string
    rule()                          - dim horizontal rule
    speaker(label)                  - print a speaker label
    say(text)                       - print one message body
    StreamColumn(console)           - context manager for streaming
                                      assistant text into the column
    pause(label, duration_s)        - animated "⋯ label" beat
    glyph_pending / running / done  - state glyphs for checklists
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import time
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.text import Text


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

COLUMN_WIDTH = 72              # visible content width
LEFT_MARGIN = 3                # spaces of left padding
TOTAL_WIDTH = COLUMN_WIDTH + LEFT_MARGIN  # for `Console(width=...)` callers


def margin() -> str:
    return " " * LEFT_MARGIN


def column_width() -> int:
    return COLUMN_WIDTH


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

ACCENT = "cyan"
ACCENT_BOLD = "bold cyan"
DIM = "dim"
WHITE = "white"
WARN = "yellow"
OK = "green"


# Geometric glyphs — used consistently across the product.
GLYPH_PENDING = "○"            # not yet started
GLYPH_RUNNING = "◐"            # in progress (animated frames below)
GLYPH_DONE    = "●"            # finished
GLYPH_SKIPPED = "·"            # skipped, intentional
GLYPH_ERROR   = "×"            # errored

# Frames for animated "thinking" / "running" beats.
SPIN_FRAMES = ["⋯", "⋱", "⋮", "⋰"]


# ---------------------------------------------------------------------------
# Static primitives
# ---------------------------------------------------------------------------


def rule(console: Console) -> None:
    """A thin dim horizontal rule that spans the reading column."""
    console.print(f"{margin()}[{DIM}]{'─' * COLUMN_WIDTH}[/]")


def banner_top(console: Console, *, title: str, subtitle: str = "") -> None:
    """Three lines: rule, title (+ optional dim subtitle on the same row,
    separated by a dot), rule. No box. Used at the top of pmc chat and
    pmc connect to mark the start of a session."""
    console.print()
    rule(console)
    if subtitle:
        line = Text()
        line.append(margin())
        line.append(title, style=ACCENT_BOLD)
        line.append("  ·  ", style=DIM)
        line.append(subtitle, style=DIM)
        console.print(line)
    else:
        console.print(f"{margin()}[{ACCENT_BOLD}]{title}[/]")
    rule(console)
    console.print()


def speaker(console: Console, label: str) -> None:
    """Print a speaker label on its own line, in dim, with a blank line
    above. The message body that follows should be one blank line below."""
    console.print()
    console.print(f"{margin()}[{DIM}]{label}[/]")
    console.print()


def turn_break(console: Console) -> None:
    """Print a single dim character on its own line — the subtle visual
    rhythm between turns."""
    console.print()
    console.print(f"{margin()}[{DIM}]─[/]")


# ---------------------------------------------------------------------------
# Body text
# ---------------------------------------------------------------------------


def say(console: Console, text: str, *, style: str = WHITE) -> None:
    """Print a paragraph of body text, wrapped to the reading column.
    `highlight=False` so Rich doesn't auto-color numbers / paths / etc."""
    wrapped = textwrap.fill(
        text.strip(),
        width=COLUMN_WIDTH,
        initial_indent=margin(),
        subsequent_indent=margin(),
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=False,
    )
    if style:
        console.print(Text(wrapped, style=style), highlight=False)
    else:
        console.print(wrapped, highlight=False)


def say_dim(console: Console, text: str) -> None:
    say(console, text, style=DIM)


# ---------------------------------------------------------------------------
# Streaming column — wraps streamed chunks to the reading column
# ---------------------------------------------------------------------------


class StreamColumn:
    """A streaming wrapper that buffers chunks at word boundaries and
    emits column-wrapped lines as they cross the right edge.

    Use as a context manager around an `async for chunk in stream`:

        col = StreamColumn(console)
        async for chunk in provider.stream_chat(...):
            col.feed(chunk)
        col.close()

    Maintains the left margin on every line and respects newlines the
    model emits (paragraph breaks).
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._buf = ""        # text since last flush
        self._line = ""       # current visible line (after margin), no margin
        self._first_line = True  # need to emit margin on first write

    def feed(self, chunk: str) -> None:
        """Stream one chunk in. May trigger zero, one, or many line flushes."""
        if not chunk:
            return
        for ch in chunk:
            if ch == "\n":
                self._flush_line()
                continue
            self._line += ch
            # Wrap when the visible line gets long enough that the *next*
            # word would overflow. We break at the last space.
            if len(self._line) >= COLUMN_WIDTH:
                self._wrap_one_line()

    def _wrap_one_line(self) -> None:
        """Break the current line at the last space and flush."""
        # Find a wrap point — last space within the column.
        wrap_at = self._line.rfind(" ", 0, COLUMN_WIDTH)
        if wrap_at <= 0:
            # No space to break on — hard break at the column edge.
            wrap_at = COLUMN_WIDTH
        head = self._line[:wrap_at]
        tail = self._line[wrap_at:].lstrip(" ")
        self._emit_visible_line(head)
        self._line = tail

    def _flush_line(self) -> None:
        """Emit whatever's in self._line and start fresh (paragraph break)."""
        self._emit_visible_line(self._line)
        self._line = ""

    def _emit_visible_line(self, text: str) -> None:
        # Always include the left margin.
        sys.stdout.write(margin() + text + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        """Flush any remaining partial line."""
        if self._line:
            self._flush_line()


# ---------------------------------------------------------------------------
# Pause / "thinking" beat
# ---------------------------------------------------------------------------


async def pause(
    console: Console,
    label: str = "reading",
    *,
    duration_s: float = 0.6,
    fps: int = 8,
) -> None:
    """Animated `⋯ label` line for ~duration_s, then clear itself."""
    line = ""
    n_frames = int(duration_s * fps)
    for i in range(n_frames):
        frame = SPIN_FRAMES[i % len(SPIN_FRAMES)]
        line = f"{margin()}\033[2m{frame} {label}\033[0m"
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        await asyncio.sleep(1.0 / fps)
    # Clear the line.
    sys.stdout.write("\r" + " " * (len(line) + 4) + "\r")
    sys.stdout.flush()


def pause_sync(
    console: Console,
    label: str = "thinking",
    *,
    duration_s: float = 0.6,
    fps: int = 8,
) -> None:
    """Synchronous version for non-async call sites (e.g. /threads opening)."""
    line = ""
    n_frames = int(duration_s * fps)
    for i in range(n_frames):
        frame = SPIN_FRAMES[i % len(SPIN_FRAMES)]
        line = f"{margin()}\033[2m{frame} {label}\033[0m"
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        time.sleep(1.0 / fps)
    sys.stdout.write("\r" + " " * (len(line) + 4) + "\r")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Card — used by /threads, /patterns, /drift
# ---------------------------------------------------------------------------


def card_title(console: Console, label: str) -> None:
    """Section title: dim, lowercase, on its own line, blank line below."""
    console.print()
    console.print(f"{margin()}[{ACCENT}]{label}[/]")
    console.print()


def card_item(console: Console, tag: str, body: str) -> None:
    """One item in a card: small accent tag on its own line, then body
    text below in the reading column. Blank line after each."""
    # `highlight=False` so Rich doesn't auto-color the slash in "/threads"
    # or bold numbers in the body.
    console.print(
        Text(f"{margin()}{tag}", style=DIM),
        highlight=False,
    )
    say(console, body)
    console.print()


__all__ = [
    "ACCENT", "ACCENT_BOLD", "DIM", "WHITE", "WARN", "OK",
    "COLUMN_WIDTH", "LEFT_MARGIN", "TOTAL_WIDTH",
    "GLYPH_PENDING", "GLYPH_RUNNING", "GLYPH_DONE",
    "GLYPH_SKIPPED", "GLYPH_ERROR", "SPIN_FRAMES",
    "margin", "column_width", "rule", "banner_top",
    "speaker", "turn_break", "say", "say_dim",
    "StreamColumn", "pause", "pause_sync",
    "card_title", "card_item",
]
