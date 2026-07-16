"""
Discovers trending rock-music stories by polling competitor RSS feeds.

Important: this module only ever reads titles, summaries, and links from
RSS feeds (the same public data an RSS reader app would show you). It does
not scrape or store full competitor article bodies.
"""

import time
import hashlib
from datetime import datetime, timezone, timedelta

import feedparser

from config import COMPETITOR_FEEDS, LOOKBACK_HOURS, MIN_SOURCE_COUNT


def _entry_timestamp(entry):
    """Best-effort parse of an RSS entry's published time -> aware datetime."""
    for key in ("published_parsed", "updated_parsed"):
        val = entry.get(key)
        if val:
            return datetime.fromtimestamp(time.mktime(val), tz=timezone.utc)
    return None


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def fetch_recent_entries():
    """Pull recent entries from every competitor feed."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    all_entries = []

    for feed in COMPETITOR_FEEDS:
        parsed = feedparser.parse(feed["url"])
        if parsed.bozo and not parsed.entries:
            print(f"[warn] could not parse feed for {feed['name']}: {parsed.bozo_exception}")
            continue

        for entry in parsed.entries:
            ts = _entry_timestamp(entry)
            if ts and ts < cutoff:
                continue
            all_entries.append({
                "source": feed["name"],
                "title": entry.get("title", "").strip(),
                "summary": entry.get("summary", "").strip(),
                "link": entry.get("link", "").strip(),
                "published": ts.isoformat() if ts else None,
            })

    return all_entries


def cluster_topics(entries):
    """
    Groups entries that likely refer to the same story across outlets.
    Uses simple keyword-overlap clustering (no full-text scraping needed) --
    good enough for "same band/event mentioned by multiple outlets."
    """
    clusters = []

    def keywords(title):
        stopwords = {
            "the", "a", "an", "and", "of", "in", "on", "for", "to", "with",
            "at", "is", "new", "his", "her", "their", "album", "song", "tour",
        }
        return {w for w in _normalize_title(title).split() if len(w) > 3 and w not in stopwords}

    for entry in entries:
        entry_kw = keywords(entry["title"])
        placed = False
        for cluster in clusters:
            cluster_kw = keywords(cluster["items"][0]["title"])
            overlap = entry_kw & cluster_kw
            if len(overlap) >= 2:  # at least 2 shared meaningful words
                cluster["items"].append(entry)
                placed = True
                break
        if not placed:
            clusters.append({"items": [entry]})

    # Attach a stable id and filter by MIN_SOURCE_COUNT
    result = []
    for cluster in clusters:
        sources = {item["source"] for item in cluster["items"]}
        if len(sources) < MIN_SOURCE_COUNT:
            continue
        topic_id = hashlib.sha1(cluster["items"][0]["title"].encode()).hexdigest()[:10]
        result.append({
            "topic_id": topic_id,
            "items": cluster["items"],
            "source_count": len(sources),
        })

    # Most-covered stories first
    result.sort(key=lambda c: c["source_count"], reverse=True)
    return result


def get_trending_topics():
    entries = fetch_recent_entries()
    return cluster_topics(entries)
