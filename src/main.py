"""
Entry point: run the full discover -> draft -> publish pipeline once.
Invoked on a schedule by the GitHub Actions workflow.
"""

import sys
import time
import html

from config import (
    MAX_ARTICLES_PER_RUN, POST_STATUS, TARGET_CATEGORY, LATEST_POSTS_COUNT,
    ARTICLE_FONT_SIZE_PX,
)
from discover import get_trending_topics
from draft import draft_article
from wordpress import (
    get_recent_post_titles, get_latest_posts, get_or_create_category,
    create_post, search_related_posts, upload_media,
)
from images import search_image, download_image_bytes


def _title_already_covered(title, existing_titles):
    """Very simple duplicate guard based on shared significant words."""
    title_words = {w.lower() for w in title.split() if len(w) > 4}
    for existing in existing_titles:
        existing_words = {w.lower() for w in existing.split() if len(w) > 4}
        if title_words and len(title_words & existing_words) / len(title_words) > 0.6:
            return True
    return False


def _process_images(article):
    """
    Finds, downloads, and uploads an image for each IMAGE placeholder in
    content_html, then swaps the placeholders for real <figure> markup.
    Returns the id of the first successfully uploaded image (for use as
    the featured/thumbnail image), or None if none succeeded.
    """
    queries = article.get("image_queries", [])
    content = article["content_html"]
    featured_media_id = None

    for i, query in enumerate(queries, start=1):
        placeholder = f"<!--IMAGE_{i}-->"
        if placeholder not in content:
            continue

        photo = search_image(query)
        if not photo:
            print(f"[warn] No stock image found for query: '{query}'")
            content = content.replace(placeholder, "")
            continue

        try:
            image_bytes = download_image_bytes(photo["url"])
            media = upload_media(
                image_bytes,
                filename=f"image-{i}.jpg",
                alt_text=photo["alt"],
            )
        except Exception as e:
            print(f"[warn] Failed to download/upload image for '{query}': {e}")
            content = content.replace(placeholder, "")
            continue

        if not media:
            content = content.replace(placeholder, "")
            continue

        if featured_media_id is None:
            featured_media_id = media["id"]

        alt = html.escape(photo["alt"])
        figure_html = (
            f'<figure class="wp-block-image size-large">'
            f'<img src="{media["source_url"]}" alt="{alt}"/>'
            f'<figcaption>Photo by {html.escape(photo["photographer"])} '
            f'via Pexels</figcaption></figure>'
        )
        content = content.replace(placeholder, figure_html)

    # Clean up any unused placeholders (e.g. if model included more than it used)
    for i in range(1, 10):
        content = content.replace(f"<!--IMAGE_{i}-->", "")

    article["content_html"] = content
    return featured_media_id


def _append_latest_posts_block(article, latest_posts):
    if not latest_posts:
        return
    items = "".join(f'<li><a href="{p["link"]}">{p["title"]}</a></li>' for p in latest_posts)
    block = f'<h2>Latest Posts</h2><ul class="deadikace-latest-posts">{items}</ul>'
    article["content_html"] += "\n" + block


def _apply_font_size(article):
    """Wraps the article body so it renders at a comfortable, non-tiny
    size regardless of the theme's default post-content font size."""
    article["content_html"] = (
        f'<div class="deadikace-article-body" '
        f'style="font-size:{ARTICLE_FONT_SIZE_PX}px; line-height:1.7;">'
        f'{article["content_html"]}</div>'
    )


def run():
    print("Fetching trending topics from competitor feeds...")
    topics = get_trending_topics()
    print(f"Found {len(topics)} candidate topic cluster(s).")

    if not topics:
        print("No topics found this run. Exiting.")
        return

    print("Fetching recent Deadikace posts to avoid duplicates...")
    existing_titles = get_recent_post_titles()

    print(f"Resolving target category '{TARGET_CATEGORY}'...")
    category_id = get_or_create_category(TARGET_CATEGORY)

    latest_posts = get_latest_posts(LATEST_POSTS_COUNT)

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

        if published_count > 0:
            time.sleep(10)  # stay comfortably under free-tier requests-per-minute limits

        try:
            article = draft_article(topic)
        except Exception as e:
            error_text = str(e)
            print(f"[error] Failed to draft article for '{headline}': {e}")
            # If the LLM provider is out of quota/credits, retrying on the next
            # topic will just fail the same way -- stop the whole run instead
            # of burning through every remaining topic with the same error.
            if ("RESOURCE_EXHAUSTED" in error_text or "429" in error_text
                    or "insufficient_quota" in error_text
                    or "NOT_FOUND" in error_text or "404" in error_text):
                print("[fatal] LLM provider error looks systemic (quota/billing/model "
                      "config issue), not specific to this topic. Stopping this run "
                      "early instead of retrying every topic with the same failure. "
                      "Check your provider's dashboard and model name before the next run.")
                break
            continue

        # Add a couple of internal links for SEO if related posts exist
        related = search_related_posts(article.get("focus_keyword", headline))
        if related:
            links_html = "<p>Related reading: " + ", ".join(
                f'<a href="{r["link"]}">{r["title"]}</a>' for r in related
            ) + "</p>"
            article["content_html"] += "\n" + links_html

        # Find/download/upload images and swap in the placeholders
        featured_media_id = _process_images(article)

        # Append the "Latest Posts" block
        _append_latest_posts_block(article, latest_posts)

        # Ensure the article renders at a comfortable, medium font size
        _apply_font_size(article)

        try:
            result = create_post(article, category_id=category_id, featured_media_id=featured_media_id)
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
