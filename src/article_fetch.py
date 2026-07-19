"""
Fetches the full text of competitor articles (not just the RSS summary)
so the drafting model has real facts to synthesize from multiple sources,
instead of inventing specifics to fill gaps left by a one-line RSS blurb.

This is only used for topics the agent is actually about to write about
(a handful per run), not the full candidate list, to keep fetch volume
reasonable and respectful.

Only used as factual grounding: the drafting system prompt explicitly
forbids copying phrasing/structure from any fetched source. Facts
themselves aren't copyrightable -- only the specific expression of them
is -- which is the same principle any journalist relies on when writing
a story informed by other outlets' reporting.
"""

import requests
import trafilatura

USER_AGENT = "DeadikaceAgent/1.0 (https://www.deadikace.com)"


def fetch_article_text(url, max_chars=5000):
    """Returns extracted main article text, or None if fetching/extraction fails."""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[warn] Failed to fetch {url}: {e}")
        return None

    try:
        text = trafilatura.extract(resp.text, include_comments=False, favor_recall=True)
    except Exception as e:
        print(f"[warn] Failed to extract article text from {url}: {e}")
        return None

    if not text:
        return None
    return text.strip()[:max_chars]


def enrich_topic_with_full_text(topic, max_sources=4):
    """
    Mutates topic['items'] in place, adding a 'full_text' key to up to
    max_sources of them (the top outlets covering the story). Items that
    fail to fetch simply keep full_text=None, and drafting falls back to
    the RSS summary for those.
    """
    fetched_count = 0
    for item in topic["items"]:
        if fetched_count >= max_sources:
            item["full_text"] = None
            continue
        text = fetch_article_text(item["link"])
        item["full_text"] = text
        if text:
            fetched_count += 1
    return topic
