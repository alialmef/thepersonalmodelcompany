"""Categorization maps — how to bucket apps, URLs, and shell commands.

The graph's per-extractor `category` field is too coarse and misclassifies
the apps the user actually spends time in (Cursor → "other", Notes →
"other", Premiere → "other"). This module re-categorizes by bundle_id
and domain to produce time aggregates that mean something.

Categories used across the time model:

  coding         — IDEs, terminals, the dev surfaces where work happens
  comms          — messaging + mail
  reading        — browsers (default); URL-level refinement layered on
  ai-assist      — Claude / GPT desktop apps, Cursor-as-AI
  creating       — notes, voice memos, doc writers
  video-editing  — DaVinci, Premiere, FCPX
  media          — Photos, Music, QuickTime, YouTube
  navigation     — Maps
  system         — Finder, settings, utilities
  reference      — bookmarks, reading lists, ebooks
  other          — fallback for unknown

Bundle IDs and domains can be added freely — every miss falls into
"other" with the original display_name preserved for debugging.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# App categorization by bundle id (exact match) or display-name fragment
# ---------------------------------------------------------------------------


# Exact bundle-id → category (most reliable)
BUNDLE_TO_CATEGORY: dict[str, str] = {
    # IDEs / coding
    "com.todesktop.230313mzl4w4u92":            "coding",       # Cursor
    "com.microsoft.VSCode":                      "coding",
    "com.apple.dt.Xcode":                        "coding",
    "com.jetbrains.intellij":                    "coding",
    "com.jetbrains.pycharm":                     "coding",
    "com.jetbrains.WebStorm":                    "coding",
    "com.apple.Terminal":                        "coding",
    "com.googlecode.iterm2":                     "coding",
    "co.zeit.hyper":                             "coding",
    "dev.warp.Warp-Stable":                      "coding",
    "com.github.atom":                           "coding",
    "io.neovim.MacVim":                          "coding",
    "com.sublimetext.4":                         "coding",
    "com.thepersonalmodelcompany.app":           "coding",       # dev'ing PMC itself

    # Comms — messages + mail
    "com.apple.MobileSMS":                       "comms",        # iMessage
    "com.apple.mail":                            "comms",
    "org.whispersystems.signal-desktop":         "comms",
    "com.tinyspeck.slackmacgap":                 "comms",
    "com.hnc.Discord":                           "comms",
    "us.zoom.xos":                               "comms",
    "com.microsoft.teams2":                      "comms",
    "com.microsoft.Outlook":                     "comms",
    "com.spikenow.macapp":                       "comms",
    "com.readdle.smartemail-Mac":                "comms",
    "com.apple.facetime":                        "comms",

    # AI assist
    "com.anthropic.claudefordesktop":            "ai-assist",
    "com.openai.chat":                           "ai-assist",
    "com.perplexityai.PerplexityForMac":         "ai-assist",
    "com.x.grok":                                "ai-assist",

    # Reading / browsers (URL-level breakdown layered on top)
    "com.apple.Safari":                          "reading",
    "com.google.Chrome":                         "reading",
    "company.thebrowser.Browser":                "reading",      # Arc
    "org.mozilla.firefox":                       "reading",
    "com.microsoft.edgemac":                     "reading",
    "com.brave.Browser":                         "reading",

    # Creating
    "com.apple.Notes":                           "creating",
    "com.apple.VoiceMemos":                      "creating",
    "notion.id":                                 "creating",
    "md.obsidian":                               "creating",
    "com.apple.iWork.Pages":                     "creating",
    "com.apple.iWork.Numbers":                   "creating",
    "com.apple.iWork.Keynote":                   "creating",
    "com.microsoft.Word":                        "creating",

    # Video editing
    "com.blackmagic-design.DaVinciResolve":      "video-editing",
    "com.adobe.PremierePro.26":                  "video-editing",
    "com.adobe.PremierePro":                     "video-editing",
    "com.apple.FinalCut":                        "video-editing",
    "com.adobe.AfterEffects":                    "video-editing",

    # Media consumption
    "com.apple.Photos":                          "media",
    "com.apple.Music":                           "media",
    "com.spotify.client":                        "media",
    "com.apple.QuickTimePlayerX":                "media",
    "com.apple.TV":                              "media",
    "com.google.android.youtube":                "media",
    "tv.netflix.macosx":                         "media",

    # Navigation
    "com.apple.Maps":                            "navigation",

    # System / utilities
    "com.apple.finder":                          "system",
    "com.apple.systempreferences":               "system",
    "com.apple.ActivityMonitor":                 "system",
    "com.apple.Spotlight":                       "system",
    "com.apple.calculator":                      "system",
    "com.apple.iCal":                            "system",       # Calendar
    "com.apple.AddressBook":                     "system",
    "com.apple.Preview":                         "system",
}


# Display-name fragments (lowercased substring match). Last-resort
# fallback for apps we don't know by bundle id.
NAME_FRAGMENTS: list[tuple[str, str]] = [
    ("cursor",       "coding"),
    ("vscode",       "coding"),
    ("xcode",        "coding"),
    ("intellij",     "coding"),
    ("pycharm",      "coding"),
    ("terminal",     "coding"),
    ("iterm",        "coding"),
    ("warp",         "coding"),
    ("ghostty",      "coding"),
    ("zed",          "coding"),
    ("davinci",      "video-editing"),
    ("premiere",     "video-editing"),
    ("final cut",    "video-editing"),
    ("after effects","video-editing"),
    ("safari",       "reading"),
    ("chrome",       "reading"),
    ("firefox",      "reading"),
    ("arc",          "reading"),
    ("edge",         "reading"),
    ("notes",        "creating"),
    ("notion",       "creating"),
    ("obsidian",     "creating"),
    ("voice memos",  "creating"),
    ("signal",       "comms"),
    ("messages",     "comms"),
    ("mobilesms",    "comms"),
    ("mail",         "comms"),
    ("slack",        "comms"),
    ("discord",      "comms"),
    ("zoom",         "comms"),
    ("claude",       "ai-assist"),
    ("chat",         "ai-assist"),       # OpenAI ChatGPT
    ("perplexity",   "ai-assist"),
    ("photos",       "media"),
    ("music",        "media"),
    ("spotify",      "media"),
    ("youtube",      "media"),
    ("maps",         "navigation"),
    ("finder",       "system"),
]


def categorize_app(
    *, bundle_id: Optional[str], display_name: Optional[str],
    fallback: Optional[str] = None,
) -> str:
    """Best-effort categorization. Bundle id first, then display name
    fragments, then the extractor's own category, then `other`."""
    if bundle_id:
        cat = BUNDLE_TO_CATEGORY.get(bundle_id)
        if cat:
            return cat
    if display_name:
        lower = display_name.lower()
        for frag, cat in NAME_FRAGMENTS:
            if frag in lower:
                return cat
    # The extractor's own category is sometimes ok — keep "social" as comms
    # but otherwise treat as unknown.
    if fallback == "social":
        return "comms"
    if fallback == "developer":
        return "coding"
    if fallback == "reference":
        return "reading"
    return "other"


# ---------------------------------------------------------------------------
# Pretty display name resolution for bundle ids we know are masked
# ---------------------------------------------------------------------------


# Bundle ids → friendly app name (when display_name is unhelpful, e.g.
# "230313mzl4w4u92" — the truncated ToDesktop bundle).
BUNDLE_TO_LABEL: dict[str, str] = {
    "com.todesktop.230313mzl4w4u92":      "Cursor",
    "com.adobe.PremierePro.26":            "Premiere Pro",
    "org.whispersystems.signal-desktop":   "Signal",
    "com.anthropic.claudefordesktop":      "Claude",
    "com.openai.chat":                     "ChatGPT",
    "com.apple.MobileSMS":                 "Messages",
    "com.apple.Notes":                     "Notes",
    "com.apple.Safari":                    "Safari",
    "com.google.Chrome":                   "Chrome",
    "com.apple.Mail":                      "Mail",
    "com.apple.Photos":                    "Photos",
    "com.apple.Maps":                      "Maps",
    "com.apple.finder":                    "Finder",
    "com.apple.QuickTimePlayerX":          "QuickTime",
    "com.blackmagic-design.DaVinciResolve":"DaVinci Resolve",
    "com.thepersonalmodelcompany.app":     "PMC (dev)",
}


def app_label(*, bundle_id: Optional[str], display_name: Optional[str]) -> str:
    """Pretty name. Falls back to display_name, then bundle tail, then '?'."""
    if bundle_id and bundle_id in BUNDLE_TO_LABEL:
        return BUNDLE_TO_LABEL[bundle_id]
    if display_name:
        # If the display_name looks like a truncated bundle hash (e.g.
        # "230313mzl4w4u92"), don't show it.
        if not display_name.isdigit() and not _looks_like_hash(display_name):
            return display_name
    if bundle_id:
        # Take the last segment of the reverse-DNS bundle id.
        tail = bundle_id.split(".")[-1]
        if not _looks_like_hash(tail):
            return tail.capitalize()
    return display_name or "(unknown)"


def _looks_like_hash(s: str) -> bool:
    """Heuristic: alphanumeric, length 6+, mixed digits and letters."""
    if len(s) < 6:
        return False
    has_digit = any(c.isdigit() for c in s)
    has_alpha = any(c.isalpha() for c in s)
    return has_digit and has_alpha and not s[0].isupper()


# ---------------------------------------------------------------------------
# URL classification — INTENTIONALLY MINIMAL.
#
# An earlier version of this file had a sprawling DOMAIN_TO_SUBCAT map
# ("airbnb=travel, sohohouse=lifestyle, github=code-host, …") that
# encoded one author's taxonomy of the web. That was the wrong shape
# for PMC. The whole thesis of the product is that the agent constructs
# per-user meaning from actual content — not from a static domain list.
#
# What stays in this module:
#   - PRIVATE filter (privacy requirement, not interpretation)
#   - JUNK URL skip patterns (auth flows, asset URLs — identity, not
#     categorization)
#   - A small set of FUNCTIONAL subcategories the time aggregator needs
#     to sum across surfaces: search, social, video, ai-tool. These are
#     roles, not interpretations.
#
# Everything else — "what does this user do on the open web" — is the
# reading layer's job. It fetches the actual page content and lets the
# LLM cluster topics per-user. The previous "travel" / "lifestyle" /
# "finance" buckets were brittle for anyone whose habits look different
# from mine.
# ---------------------------------------------------------------------------


DOMAIN_TO_SUBCAT: dict[str, str] = {
    "google.com":     "search",
    "bing.com":       "search",
    "duckduckgo.com": "search",
    "twitter.com":    "social",
    "x.com":          "social",
    "linkedin.com":   "social",
    "instagram.com":  "social",
    "facebook.com":   "social",
    "threads.net":    "social",
    "bsky.app":       "social",
    "reddit.com":     "social",
    "youtube.com":    "video",
    "youtu.be":       "video",
    "vimeo.com":      "video",
    "claude.ai":      "ai-tool",
    "chatgpt.com":    "ai-tool",
    "perplexity.ai":  "ai-tool",
}

# (legacy heavy DOMAIN_TO_SUBCAT removed — see the comment block above
#  the new DOMAIN_TO_SUBCAT for the architectural reasoning. Per-user
#  meaning is now the LLM reading-layer's job, not a static taxonomy.)


# Domains tagged "private" are visible to the user in `pmc time` but
# filtered out of the LLM-facing portrait (self.md, time.md, whoami).
# The data still lives in the graph; we just don't include it when
# composing the agent-readable substrate. This is local-first, but
# the agent doesn't need to characterize you by it.
_PRIVATE_DOMAIN_FRAGMENTS = (
    "pornhub", "xvideos", "xnxx", "redtube", "youporn",
    "onlyfans", "fansly",
    "/health/", "webmd", "drugs.com",
)


def categorize_domain(domain: str) -> str:
    if not domain:
        return "other"
    d = domain.lower().lstrip(".")

    # Private filter — applied first, before any other categorization.
    if any(frag in d for frag in _PRIVATE_DOMAIN_FRAGMENTS):
        return "private"

    if d in DOMAIN_TO_SUBCAT:
        return DOMAIN_TO_SUBCAT[d]
    # Try suffix matches (e.g. www.github.com → github.com)
    for known, cat in DOMAIN_TO_SUBCAT.items():
        if d.endswith("." + known) or d == known:
            return cat
    # Catch-all heuristics
    if d.endswith(".gov") or d.endswith(".gov.us"):
        return "civic"
    if d.endswith(".edu"):
        return "academic"
    if "docs" in d or "developer" in d:
        return "platform-docs"
    return "other"


def is_private_subcat(subcat: str) -> bool:
    """Subcats we drop from the LLM-facing portrait but keep in the
    user's own time view. Add new categories here if needed."""
    return subcat == "private"


__all__ = [
    "categorize_app",
    "app_label",
    "categorize_domain",
    "BUNDLE_TO_CATEGORY",
    "BUNDLE_TO_LABEL",
    "DOMAIN_TO_SUBCAT",
]
