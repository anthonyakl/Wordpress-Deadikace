"""
Discovers trending rock-music stories by polling competitor RSS feeds.

Important: this module only ever reads titles, summaries, and links from
RSS feeds (the same public data an RSS reader app would show you). It does
not scrape or store full competitor article bodies.

There is no reliable public way to see competitors' actual engagement
(views, shares) -- that data isn't published, and "most popular" widgets
(where they exist at all) are inconsistent and fragile to scrape. Instead,
topics are ranked using two honest signals we do have:
  1. How many outlets are covering the story (cross-outlet coverage is a
     real signal that something is genuinely newsworthy).
  2. How recent it is (freshness decays the score the older a story gets).
An optional list of priority keywords (favorite bands/artists/genres) can
also boost a story's ranking -- see PRIORITY_KEYWORDS in config.py.
"""

import time
import hashlib
from datetime import datetime, timezone, timedelta

import feedparser

from config import (
    COMPETITOR_FEEDS, LOOKBACK_HOURS, MIN_SOURCE_COUNT,
    SOURCE_COUNT_WEIGHT, PRIORITY_KEYWORDS, PRIORITY_KEYWORD_BONUS,
    EXCLUDE_KEYWORDS, SOURCE_COVERAGE_WEIGHT,
)


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

            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            combined_lower = f"{title} {summary}".lower()
            if any(kw.lower() in combined_lower for kw in EXCLUDE_KEYWORDS):
                continue

            all_entries.append({
                "source": feed["name"],
                "title": title,
                "summary": summary,
                "link": entry.get("link", "").strip(),
                "published": ts.isoformat() if ts else None,
            })

    return all_entries


def cluster_topics(entries):
    """
    Groups entries that likely refer to the same story across outlets.
    Uses simple keyword-overlap clustering (no full-text scraping needed) --
    good enough for "same band/event mentioned by multiple outlets."

    Compares each new entry against the UNION of keywords across ALL items
    already in a cluster (not just the first one added). Comparing only
    against the founding item was too brittle: two outlets can word the
    same story differently enough that neither matches the founding
    headline directly (e.g. one leads with "nearly died", another leads
    with a specific quote), landing the same story in two separate
    clusters -- which then get drafted as two duplicate articles, since
    downstream duplicate checks only compare against already-published
    posts, not against sibling topics in the same batch.
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
            cluster_kw = set()
            for item in cluster["items"]:
                cluster_kw |= keywords(item["title"])
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

        timestamps = [item["published"] for item in cluster["items"] if item["published"]]
        most_recent = max((datetime.fromisoformat(ts) for ts in timestamps), default=None)

        result.append({
            "topic_id": topic_id,
            "items": cluster["items"],
            "source_count": len(sources),
            "most_recent": most_recent.isoformat() if most_recent else None,
        })

    return result


def _score_topic(topic, now):
    """
    Higher score = higher publishing priority. Combines:
      - coverage: more outlets covering it -> more newsworthy
      - freshness: newer stories score higher, decaying linearly to 0
        over LOOKBACK_HOURS
      - optional keyword boost for favorite bands/artists/genres
    """
    # Weighted coverage: counts each distinct outlet covering this story,
    # but a metal-heavy outlet (see SOURCE_COVERAGE_WEIGHT in config.py)
    # contributes less than a full point so it can't single-handedly push
    # a borderline topic's ranking up. topic["source_count"] itself is
    # left untouched (MIN_SOURCE_COUNT filtering and logging elsewhere
    # still see the true outlet count) -- only this local ranking score
    # is weighted.
    distinct_sources = {item["source"] for item in topic["items"]}
    weighted_coverage = sum(SOURCE_COVERAGE_WEIGHT.get(s, 1.0) for s in distinct_sources)
    coverage_score = weighted_coverage * SOURCE_COUNT_WEIGHT

    if topic["most_recent"]:
        age_hours = (now - datetime.fromisoformat(topic["most_recent"])).total_seconds() / 3600
        freshness_score = max(0.0, LOOKBACK_HOURS - age_hours)
    else:
        freshness_score = 0.0  # unknown publish time -> treat as stale

    keyword_bonus = 0
    if PRIORITY_KEYWORDS:
        title_lower = topic["items"][0]["title"].lower()
        if any(kw.lower() in title_lower for kw in PRIORITY_KEYWORDS):
            keyword_bonus = PRIORITY_KEYWORD_BONUS

    return coverage_score + freshness_score + keyword_bonus


def get_trending_topics():
    entries = fetch_recent_entries()
    topics = cluster_topics(entries)

    now = datetime.now(timezone.utc)
    for topic in topics:
        topic["score"] = round(_score_topic(topic, now), 1)

    topics.sort(key=lambda t: t["score"], reverse=True)
    return topics
