"""
Entry point: run the full discover -> draft -> publish pipeline once.
Invoked on a schedule by the GitHub Actions workflow.
"""

import sys
import time

from config import MAX_ARTICLES_PER_RUN, POST_STATUS
from discover import get_trending_topics
from draft import draft_article
from wordpress import get_recent_post_titles, create_post, search_related_posts


def _title_already_covered(title, existing_titles):
    """Very simple duplicate guard based on shared significant words."""
    title_words = {w.lower() for w in title.split() if len(w) > 4}
    for existing in existing_titles:
        existing_words = {w.lower() for w in existing.split() if len(w) > 4}
        if title_words and len(title_words & existing_words) / len(title_words) > 0.6:
            return True
    return False


def run():
    print("Fetching trending topics from competitor feeds...")
    topics = get_trending_topics()
    print(f"Found {len(topics)} candidate topic cluster(s).")

    if not topics:
        print("No topics found this run. Exiting.")
        return

    print("Fetching recent Deadikace posts to avoid duplicates...")
    existing_titles = get_recent_post_titles()

    published_count = 0
    for topic in topics:
        if published_count >= MAX_ARTICLES_PER_RUN:
            break

        headline = topic["items"][0]["title"]
        if _title_already_covered(headline, existing_titles):
            print(f"Skipping (likely already covered): {headline}")
            continue

        print(f"Drafting article for topic: {headline} "
              f"(covered by {topic['source_count']} outlet(s))")

        try:
            article = draft_article(topic)
        except Exception as e:
            print(f"[error] Failed to draft article for '{headline}': {e}")
            continue

        # Add a couple of internal links for SEO if related posts exist
        related = search_related_posts(article.get("focus_keyword", headline))
        if related:
            links_html = "<p>Related reading: " + ", ".join(
                f'<a href="{r["link"]}">{r["title"]}</a>' for r in related
            ) + "</p>"
            article["content_html"] += "\n" + links_html

        try:
            result = create_post(article)
        except Exception as e:
            print(f"[error] Failed to publish article for '{headline}': {e}")
            continue

        print(f"Created post (status={POST_STATUS}): {result.get('link', result.get('id'))}")
        published_count += 1
        time.sleep(2)  # be polite to the WP API between requests

    print(f"Done. Published/drafted {published_count} article(s) this run.")


if __name__ == "__main__":
    try:
        run()
    except KeyError as e:
        print(f"[fatal] Missing required environment variable: {e}")
        sys.exit(1)
