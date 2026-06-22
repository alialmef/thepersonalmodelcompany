"""Reading — what was actually on screen.

Time tells you when. Apps tell you which surface. URLs tell you where.
But none of that tells you what the user was *thinking* about. For
that we read the page itself.

Pipeline:
  1. Pull recent URLs from Safari's History.db (precise timestamps,
     already on disk, no network)
  2. Filter: skip auth flows, redirects, search-engine result pages
     of the form ".../search?q=..." (we want the destinations, not
     the SERPs), and any of the "private" domain fragments we already
     mask in categorize.py.
  3. For each surviving URL, fetch the page over HTTP, extract the
     readable main text (BeautifulSoup, falls back to raw HTML strip).
  4. Cache the extracted text at:
        graph/raw/page_content/<sha1(url)>.txt
     so re-runs don't re-fetch.
  5. Cluster the corpus into topics with one LLM call. Each topic
     gets a short name + the URLs that contribute + total dwell
     estimate.
  6. Write `portrait/reading.md` with the topic list. The portrait
     synthesis pass folds it into time.md / self.md.

The agent now knows not just "you were on docs.anthropic.com for 9
hours" but "you were studying parallel function calling, got stuck on
tool_choice, then looked at three example repos under anthropic-cookbook
on GitHub."
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from pmc.agent.providers.base import Message, ProviderConfig
from pmc.agent.providers.registry import get_provider
from pmc.cli.local_config import LocalConfig
from pmc.consolidator.categorize import (
    _PRIVATE_DOMAIN_FRAGMENTS,
    categorize_domain,
)


log = logging.getLogger("pmc.consolidator.reading")


# Defaults — tunable via env vars.
DAYS_BACK = int(os.environ.get("PMC_READING_DAYS", "7"))
MAX_URLS_TO_FETCH = int(os.environ.get("PMC_READING_MAX_FETCH", "60"))
FETCH_TIMEOUT_S = 8.0
USER_AGENT = (
    "PMC/0.1 (personal-memory-substrate; +https://thepersonalmodelcompany.com)"
)


# URLs we never fetch — auth flows, redirects, OAuth, asset URLs.
SKIP_URL_PATTERNS = (
    re.compile(r"^https?://accounts\.google\.com"),
    re.compile(r"^https?://login\."),
    re.compile(r"^https?://auth\."),
    re.compile(r"^https?://oauth\."),
    re.compile(r"/oauth/"),
    re.compile(r"/signin"),
    re.compile(r"/login"),
    re.compile(r"/auth/callback"),
    re.compile(r"/_next/"),
    re.compile(r"\.(png|jpg|jpeg|gif|webp|svg|ico|css|js|woff|ttf)(\?|$)", re.I),
    # Google + DuckDuckGo result pages — we want destinations, not SERPs
    re.compile(r"^https?://(www\.)?google\.[^/]+/search\?"),
    re.compile(r"^https?://duckduckgo\.com/\?q="),
)


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


@dataclass
class VisitedPage:
    url: str
    title: str
    domain: str
    subcat: str                            # from categorize_domain
    visits_in_window: int                  # times visited in the window
    first_visit: str                       # ISO local
    last_visit: str
    text_excerpt: Optional[str] = None     # first ~2000 chars of cleaned text
    text_full_path: Optional[str] = None   # disk path to full text


@dataclass
class ReadingTopic:
    label: str                             # short topic name
    summary: str                           # 1-2 sentences
    page_urls: list[str] = field(default_factory=list)
    visits_total: int = 0


# ---------------------------------------------------------------------------
# Step 1 — read Safari History.db
# ---------------------------------------------------------------------------


def read_safari_history(*, days: int = DAYS_BACK) -> list[VisitedPage]:
    """Pull recent URLs from Safari's History.db. Each `VisitedPage`
    aggregates multiple visits to the same URL within the window."""
    db = Path("~/Library/Safari/History.db").expanduser()
    if not db.is_file():
        log.info("reading: Safari History.db not found at %s", db)
        return []

    # CFAbsoluteTime epoch offset.
    CF_OFFSET = 978307200
    cutoff_cf = datetime.now(timezone.utc).timestamp() - days * 86400 - CF_OFFSET

    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error as e:
        log.warning("reading: can't open History.db (FDA?): %s", e)
        return []

    pages: dict[str, VisitedPage] = {}
    try:
        for url, title, vtime in conn.execute("""
            SELECT i.url, v.title, v.visit_time
            FROM history_items i
            JOIN history_visits v ON v.history_item = i.id
            WHERE v.visit_time > ?
            ORDER BY v.visit_time DESC
        """, (cutoff_cf,)):
            if not url or _should_skip_url(url):
                continue
            domain = urlparse(url).netloc.lower()
            subcat = categorize_domain(domain)
            if subcat == "private":
                continue
            ts_iso = datetime.fromtimestamp(
                (vtime or 0) + CF_OFFSET, tz=timezone.utc
            ).astimezone().isoformat(timespec="seconds")
            if url in pages:
                pages[url].visits_in_window += 1
                pages[url].first_visit = min(pages[url].first_visit, ts_iso)
                pages[url].last_visit = max(pages[url].last_visit, ts_iso)
            else:
                pages[url] = VisitedPage(
                    url=url,
                    title=(title or "").strip(),
                    domain=domain,
                    subcat=subcat,
                    visits_in_window=1,
                    first_visit=ts_iso,
                    last_visit=ts_iso,
                )
    finally:
        conn.close()

    # Sort by visits desc, then recency
    out = sorted(
        pages.values(),
        key=lambda p: (p.visits_in_window, p.last_visit),
        reverse=True,
    )
    log.info("reading: collected %d unique URLs across %d days", len(out), days)
    return out


def _should_skip_url(url: str) -> bool:
    for pat in SKIP_URL_PATTERNS:
        if pat.search(url):
            return True
    return False


# ---------------------------------------------------------------------------
# Step 2/3/4 — fetch + extract + cache
# ---------------------------------------------------------------------------


def _content_dir(cfg: LocalConfig, user_id: str) -> Path:
    root = cfg.effective_storage_root()
    p = root / "users" / user_id / "graph" / "raw" / "page_content"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _url_cache_path(cfg: LocalConfig, user_id: str, url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return _content_dir(cfg, user_id) / f"{digest}.txt"


def fetch_pages(
    cfg: LocalConfig,
    user_id: str,
    pages: list[VisitedPage],
    *,
    max_fetch: int = MAX_URLS_TO_FETCH,
) -> list[VisitedPage]:
    """For each URL, fetch + extract text, cache to disk, attach to the
    VisitedPage. Skips URLs that already have cached content."""
    out: list[VisitedPage] = []
    fetched_count = 0
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=FETCH_TIMEOUT_S,
        follow_redirects=True,
        verify=True,
    ) as client:
        for page in pages:
            cache = _url_cache_path(cfg, user_id, page.url)
            if cache.is_file():
                # Use cached content.
                try:
                    text = cache.read_text(errors="ignore")
                    page.text_full_path = str(cache)
                    page.text_excerpt = text[:2000]
                except OSError:
                    pass
                out.append(page)
                continue

            if fetched_count >= max_fetch:
                # Past budget; keep the page in the list but no content.
                out.append(page)
                continue

            text = _fetch_one(client, page.url)
            if text:
                try:
                    cache.write_text(text)
                    page.text_full_path = str(cache)
                    page.text_excerpt = text[:2000]
                except OSError:
                    pass
                fetched_count += 1
            out.append(page)
    log.info("reading: fetched %d pages (cap %d)", fetched_count, max_fetch)
    return out


def _fetch_one(client: httpx.Client, url: str) -> Optional[str]:
    try:
        resp = client.get(url)
    except (httpx.HTTPError, httpx.InvalidURL):
        return None
    if resp.status_code >= 400:
        return None
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype.lower() and "text" not in ctype.lower():
        return None
    try:
        text = _extract_readable_text(resp.text)
    except Exception:  # noqa: BLE001
        text = None
    if not text:
        return None
    # Trim
    return text[:8000]


def _extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Strip non-content nodes
    for tag in soup(["script", "style", "noscript", "nav", "footer",
                     "aside", "header", "form", "iframe"]):
        tag.decompose()
    # Prefer <main> or <article> if present
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    # Collapse whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 5 — cluster topics via LLM
# ---------------------------------------------------------------------------


def cluster_topics(
    cfg: LocalConfig,
    pages: list[VisitedPage],
    *,
    max_topics: int = 8,
) -> list[ReadingTopic]:
    """One LLM call. Reads the corpus, returns named topics."""
    pages_with_text = [p for p in pages if p.text_excerpt]
    if not pages_with_text:
        return []

    # Build a compact corpus — each page as a one-line summary.
    lines: list[str] = []
    for i, p in enumerate(pages_with_text[:60]):
        title = p.title or "(no title)"
        excerpt = (p.text_excerpt or "").replace("\n", " ")[:400]
        lines.append(
            f"[{i+1}] {p.domain}{urlparse(p.url).path}  "
            f"({p.visits_in_window} visits)\n"
            f"    title: {title}\n"
            f"    excerpt: {excerpt}"
        )
    user_msg = (
        f"Below are {len(pages_with_text)} pages the user actually visited "
        f"in the last {DAYS_BACK} days, with the readable excerpts I "
        "fetched from each. Cluster them into the topics the user has "
        "been thinking about / learning / working on.\n\n"
        + "\n\n".join(lines)
    )

    text = _ask_llm(cfg, system=_TOPIC_SYSTEM.format(max=max_topics),
                    user=user_msg, max_tokens=4000)

    parsed = _parse_topics(text, pages_with_text)
    if not parsed:
        # One retry with a tighter constraint and even more headroom.
        text2 = _ask_llm(
            cfg,
            system=_TOPIC_SYSTEM.format(max=max_topics)
                   + "\n\nIMPORTANT: respond with VALID, COMPLETE JSON only.\n"
                   "Keep `summary` under 200 characters per topic.",
            user=user_msg, max_tokens=4000,
        )
        parsed = _parse_topics(text2, pages_with_text)
    return parsed


_TOPIC_SYSTEM = """\
You are clustering the user's recent browsing into topics so an agent
that talks to them can know what they've been thinking about.

Rules:
  - Return JSON only — no markdown, no code fences, no prose around it.
  - Up to {max} topics. Fewer if the user is focused; more only if
    they're genuinely working on multiple unrelated threads.
  - Each topic has:
       label    short, concrete — "wiring MCP servers into Claude
                Desktop", not "AI tools". Specific enough that the user
                would say "yeah that's what I was doing".
       summary  1-2 sentences. Cite something from the actual excerpts
                if possible — what they were specifically trying to
                figure out.
       indices  the [N] numbers from the input that belong to this topic.
  - Combine: if multiple pages are about the same thing, one topic.
  - Skip topics that are just navigation / re-auth / social media doom-
    scrolling — those are noise.

Output exactly this shape:
{{
  "topics": [
    {{ "label": "...", "summary": "...", "indices": [1, 3, 7] }}
  ]
}}
"""


def _parse_topics(text: str, pages: list[VisitedPage]) -> list[ReadingTopic]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("reading: topic JSON parse failed: %s", e)
        return []
    out: list[ReadingTopic] = []
    for t in data.get("topics", []):
        label = (t.get("label") or "").strip()
        summary = (t.get("summary") or "").strip()
        idxs = t.get("indices") or []
        urls = []
        visits = 0
        for i in idxs:
            try:
                page = pages[int(i) - 1]
            except (ValueError, IndexError):
                continue
            urls.append(page.url)
            visits += page.visits_in_window
        if not label:
            continue
        out.append(ReadingTopic(
            label=label, summary=summary,
            page_urls=urls, visits_total=visits,
        ))
    out.sort(key=lambda r: r.visits_total, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Step 6 — write reading.md + return for portrait synthesis
# ---------------------------------------------------------------------------


def write_reading_md(
    cfg: LocalConfig, user_id: str, topics: list[ReadingTopic],
    pages: list[VisitedPage],
) -> str:
    lines: list[str] = []
    lines.append(f"# What you've been reading (last {DAYS_BACK} days)")
    lines.append("")
    lines.append(
        f"From {len(pages)} unique URLs, "
        f"{sum(1 for p in pages if p.text_excerpt)} of which I read."
    )
    lines.append("")
    if not topics:
        lines.append("(no topic clusters extracted)")
    else:
        for t in topics:
            lines.append(f"## {t.label}")
            lines.append("")
            lines.append(t.summary)
            lines.append("")
            if t.page_urls:
                lines.append("Pages:")
                for u in t.page_urls[:6]:
                    lines.append(f"  - {u}")
                if len(t.page_urls) > 6:
                    lines.append(f"  - …and {len(t.page_urls) - 6} more")
            lines.append("")
    text = "\n".join(lines).strip() + "\n"
    root = cfg.effective_storage_root()
    p = root / "users" / user_id / "graph" / "synth" / "portrait" / "reading.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    log.info("reading: wrote reading.md (%d topics)", len(topics))
    return text


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_reading_layer(
    cfg: LocalConfig,
    *,
    user_id: str,
    days: int = DAYS_BACK,
    max_fetch: int = MAX_URLS_TO_FETCH,
) -> tuple[list[VisitedPage], list[ReadingTopic]]:
    """One full reading pass — history → fetch → cluster → markdown."""
    pages = read_safari_history(days=days)
    pages = fetch_pages(cfg, user_id, pages, max_fetch=max_fetch)
    topics = cluster_topics(cfg, pages)
    write_reading_md(cfg, user_id, topics, pages)
    return pages, topics


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------


def _ask_llm(cfg: LocalConfig, *, system: str, user: str, max_tokens: int) -> str:
    provider = get_provider(cfg.provider)
    if provider is None:
        raise RuntimeError(f"unknown provider {cfg.provider!r}")
    pcfg = ProviderConfig(
        provider=cfg.provider, model=cfg.model, api_key=cfg.api_key,
    )
    async def _call() -> str:
        resp = await provider.chat(
            messages=[Message(role="user", content=user)],
            config=pcfg, system=system, max_tokens=max_tokens,
        )
        return resp.text
    return asyncio.run(_call())


__all__ = [
    "VisitedPage", "ReadingTopic",
    "read_safari_history", "fetch_pages",
    "cluster_topics", "write_reading_md", "build_reading_layer",
    "DAYS_BACK", "MAX_URLS_TO_FETCH",
]
