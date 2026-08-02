"""
Entry point: run the full discover -> draft -> publish pipeline once.
Invoked on a schedule by the GitHub Actions workflow.
"""

import sys
import time
import re

from config import (
    MAX_ARTICLES_PER_RUN, POST_STATUS, TARGET_CATEGORY, LATEST_POSTS_COUNT,
    ARTICLE_FONT_SIZE_PX, ENABLE_SOURCE_IMAGES,
)
from discover import get_trending_topics
from draft import draft_article, filter_rock_relevant_topics, filter_duplicate_topics, verify_and_refine
from article_fetch import enrich_topic_with_full_text, download_image_bytes
from wordpress import (
    get_recent_post_titles, get_latest_posts, get_or_create_category,
    create_post, search_related_posts, check_connectivity, upload_media,
)

def _title_already_covered(title, existing_titles):
    """Very simple duplicate guard based on shared significant words."""
    title_words = {w.lower() for w in title.split() if len(w) > 4}
    for existing in existing_titles:
        existing_words = {w.lower() for w in existing.split() if len(w) > 4}
        if title_words and len(title_words & existing_words) / len(title_words) > 0.6:
            return True
    return False

def _strip_image_placeholders(content_html):
    """
    Image sourcing (Wikimedia Commons search + Pexels stock-photo fallback)
    has been removed -- it was matching the wrong subject often enough to
    be worse than no image at all (e.g. returning a photo of an actual
    eagle for an article about the band Eagles). Rather than re-attempt
    automatic image sourcing, articles are simply published without
    inline images for now, so any \`<!--IMAGE_N-->\` placeholder the model
    still emits is just stripped out.
    """
    for i in range(1, 10):
        content_html = content_html.replace(f"<!--IMAGE_{i}-->", "")
    return content_html

def _get_featured_media_for_topic(topic, article_title):
    """
    Downloads the source article's own preview image (og:image) and
    re-hosts it on Deadikace as the featured image, since automatic
    image search/download was removed after repeatedly matching the
    wrong subject. See the IMAGE SOURCING NOTE in article_fetch.py for
    the copyright tradeoff this involves. Returns a media ID, or None
    if no source item had a usable image or the download/upload failed.
    """
    for item in topic.get("items", []):
        image_url = item.get("image_url")
        if not image_url:
            continue
        image_bytes, content_type = download_image_bytes(image_url)
        if not image_bytes:
            continue
        try:
            media = upload_media(
                image_bytes,
                filename=article_title,
                alt_text=f"Photo via {item.get('source', 'source article')}",
                content_type=content_type,
            )
        except Exception as e:
            print(f"[warn] Failed to upload source image to WordPress media library: {e}")
            continue
        if media:
            print(f"[info] Using source image from {item.get('source', 'source article')} as featured image.")
            return media["id"]
    return None

def _append_latest_posts_block(article, latest_posts):
    if not latest_posts:
        return
    items = "".join(f'<li><a href="{p["link"]}">{p["title"]}</a></li>' for p in latest_posts)
    block = f'<h2>Latest Posts</h2><ul class="deadikace-latest-posts">{items}</ul>'
    article["content_html"] += "\n" + block

_BLOCK_PATTERN = re.compile(
    r"(?P<p><p>.*?</p>)"
    r"|(?P<h3><h3>.*?</h3>)"
    r"|(?P<h2><h2>.*?</h2>)"
    r"|(?P<figure><figure[^>]*>.*?</figure>)"
    r"|(?P<ol><ol[^>]*>.*?</ol>)"
    r"|(?P<ul><ul[^>]*>.*?</ul>)",
    re.DOTALL,
)

def _blockify(content_html, font_size_px):
    """
    Converts plain HTML into real Gutenberg block markup (wp:paragraph,
    wp:heading, wp:image, wp:list) instead of one raw HTML blob. This is
    what makes the post open as normal, individually-editable blocks in
    the WordPress block editor -- a plain wrapped <div> of raw HTML gets
    treated as a single uneditable Custom HTML block instead. Paragraph
    text gets the medium font size as a real, editable block attribute
    (not just an inline style baked into unstructured HTML).
    """
    parts = []
    for m in _BLOCK_PATTERN.finditer(content_html):
        if m.group("p"):
            inner = m.group("p").replace("<p>", f'<p style="font-size:{font_size_px}px">', 1)
            parts.append(
                f'<!-- wp:paragraph {{"style":{{"typography":{{"fontSize":"{font_size_px}px"}}}}}} -->\n'
                f'{inner}\n<!-- /wp:paragraph -->'
            )
        elif m.group("h3"):
            parts.append(f'<!-- wp:heading {{"level":3}} -->\n{m.group("h3")}\n<!-- /wp:heading -->')
        elif m.group("h2"):
            parts.append(f'<!-- wp:heading -->\n{m.group("h2")}\n<!-- /wp:heading -->')
        elif m.group("figure"):
            parts.append(f'<!-- wp:image {{"sizeSlug":"large"}} -->\n{m.group("figure")}\n<!-- /wp:image -->')
        elif m.group("ol"):
            parts.append(f'<!-- wp:list {{"ordered":true}} -->\n{m.group("ol")}\n<!-- /wp:list -->')
        elif m.group("ul"):
            parts.append(f'<!-- wp:list -->\n{m.group("ul")}\n<!-- /wp:list -->')

    return "\n\n".join(parts)

def run():
    print("Checking connectivity to WordPress...")
    ok, error = check_connectivity()
    if not ok:
        print(f"[fatal] Could not reach the WordPress site before doing any other work: {error}")
        print("[fatal] A connection that times out (rather than an explicit error "
              "response) often means something between GitHub Actions and your "
              "host is silently blocking the request -- e.g. host-level bot/DDoS "
              "protection rejecting datacenter IP ranges. This has been reported "
              "on some hosts (including Hostinger) for exactly this kind of "
              "automated traffic. Retries already happened automatically before "
              "this message; if it keeps failing across multiple scheduled runs, "
              "contact your host's support with this specific symptom.")
        sys.exit(1)

    print("Fetching trending topics from competitor feeds...")
    topics = get_trending_topics()
    print(f"Found {len(topics)} candidate topic cluster(s).")

    if not topics:
        print("No topics found this run. Exiting.")
        return

    print("Filtering for rock-relevant topics (some feeds cover all genres)...")
    topics = filter_rock_relevant_topics(topics)

    if not topics:
        print("No rock-relevant topics found this run. Exiting.")
        return

    print("Fetching recent Deadikace posts to avoid duplicates...")
    existing_titles = get_recent_post_titles()

    print("Checking candidate topics against recently published posts to avoid duplicate stories, even if they were covered by a different outlet...")
    topics = filter_duplicate_topics(topics, existing_titles)

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

        print("Fetching full source article text for factual grounding...")
        enrich_topic_with_full_text(topic)

        if published_count > 0:
            time.sleep(10)  # stay comfortably under free-tier requests-per-minute limits

        try:
            articles = draft_article(topic)
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

        if len(articles) > 1:
            print(f"[info] Topic split into {len(articles)} separate articles "
                  f"(distinct newsworthy stories detected within one topic cluster).")

        for article in articles:
            if published_count >= MAX_ARTICLES_PER_RUN:
                break

            print("Fact-checking the draft against the source material...")
            try:
                article = verify_and_refine(article, topic)
            except Exception as e:
                print(f"[warn] Fact-check pass raised an unexpected error ({e}); "
                      "publishing the original draft unchanged.")

            # Add a couple of internal links for SEO if related posts exist
            related = search_related_posts(article.get("focus_keyword", headline))
            if related:
                links_html = "<p>Related reading: " + ", ".join(
                    f'<a href="{r["link"]}">{r["title"]}</a>' for r in related
                ) + "</p>"
                article["content_html"] += "\n" + links_html

            # Strip any leftover image placeholders the model might still
            # emit, then try to use the source article's own preview image
            # as the featured image (see IMAGE SOURCING NOTE in
            # article_fetch.py for the copyright tradeoff this involves).
            article["content_html"] = _strip_image_placeholders(article["content_html"])
            featured_media_id = (
                _get_featured_media_for_topic(topic, article["title"])
                if ENABLE_SOURCE_IMAGES else None
            )

            # Append the "Latest Posts" block
            _append_latest_posts_block(article, latest_posts)

            # Convert to real Gutenberg blocks (keeps the post editable in the
            # block editor) with the medium font size applied per-paragraph
            article["content_html"] = _blockify(article["content_html"], ARTICLE_FONT_SIZE_PX)

            try:
                result = create_post(article, category_id=category_id, featured_media_id=featured_media_id)
            except Exception as e:
                print(f"[error] Failed to publish article for '{article.get('title', headline)}': {e}")
                continue

            print(f"Created post (status={POST_STATUS}): {result.get('link', result.get('id'))}")
            published_count += 1
            time.sleep(2)  # be polite to the WP API between requests, and between multiple articles from one split topic

    print(f"Done. Published/drafted {published_count} article(s) this run.")

if __name__ == "__main__":
    try:
        run()
    except KeyError as e:
        print(f"[fatal] Missing required environment variable: {e}")
        sys.exit(1)
