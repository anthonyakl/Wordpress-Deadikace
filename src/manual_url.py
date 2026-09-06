"""Create one WordPress draft from a user-supplied source article URL.

This entry point deliberately bypasses RSS discovery, rock-relevance filtering,
source-count thresholds, ranking, and the scheduled agent's per-run selection.
After constructing one topic from the requested URL, it reuses the scheduled
agent's research, drafting, verification, media, linking, Gutenberg, dedupe,
and WordPress helpers.
"""

import hashlib
import os
import sys
from difflib import SequenceMatcher
from urllib.parse import urlparse

import trafilatura

from article_fetch import (
    _fetch_html,
    extract_body_images,
    extract_video_embeds,
    fetch_source_image_url,
)
from config import (
    ARTICLE_FONT_SIZE_PX,
    ENABLE_SOURCE_IMAGES,
    LATEST_POSTS_COUNT,
    TARGET_CATEGORY,
)
from draft import (
    draft_article,
    research_additional_context,
    verify_and_refine,
)
from main import (
    _append_latest_posts_block,
    _blockify,
    _get_featured_media_for_topic,
    _insert_illustrative_images,
    _insert_source_images,
    _insert_video_embeds,
    _normalize_story_text,
    _story_tokens,
    _strip_image_placeholders,
)
from wordpress import (
    check_connectivity,
    create_post,
    get_latest_posts,
    get_or_create_category,
    get_recent_posts_for_dedup,
    search_related_posts,
)


def _validate_source_url(raw_url):
    url = (raw_url or "").strip()
    if not url:
        raise ValueError("source_url is required and cannot be empty.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("source_url must be a valid absolute http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("source_url must not contain embedded credentials.")
    return url


def _source_name(parsed_url, metadata):
    return (
        (getattr(metadata, "sitename", None) if metadata else None)
        or parsed_url.hostname
        or "Requested source"
    )


def _build_topic(source_url):
    """Fetch source_url once and construct the topic/item shape used by draft.py."""
    html = _fetch_html(source_url)
    if not html:
        raise RuntimeError(f"The requested URL could not be fetched: {source_url}")

    try:
        full_text = trafilatura.extract(html, include_comments=False, favor_recall=True)
        metadata = trafilatura.extract_metadata(html)
    except Exception as exc:
        raise RuntimeError(f"The requested URL could not be parsed as an article: {source_url}") from exc

    full_text = full_text.strip()[:5000] if full_text else None
    if not full_text:
        raise RuntimeError(
            f"The requested URL was fetched, but no readable article text could be extracted: {source_url}"
        )

    parsed = urlparse(source_url)
    title = (getattr(metadata, "title", None) if metadata else None) or ""
    title = title.strip()
    if not title:
        raise RuntimeError(
            f"The requested URL was fetched, but no article title could be extracted: {source_url}"
        )

    image_url, image_caption = fetch_source_image_url(source_url, html=html)
    item = {
        "source": _source_name(parsed, metadata),
        "title": title,
        "summary": full_text[:500],
        "link": source_url,
        "published": getattr(metadata, "date", None) if metadata else None,
        "full_text": full_text,
        "image_url": image_url,
        "image_caption": image_caption,
        "video_embeds": extract_video_embeds(html),
        "body_images": extract_body_images(html, exclude_url_fragment=image_url),
    }
    return {
        "topic_id": hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:10],
        "items": [item],
        "source_count": 1,
        "most_recent": item["published"],
        "score": 0,
    }


def _find_clear_duplicate(candidate_title, existing_posts):
    """Return the matching post only when headlines clearly describe one event.

    The scheduled pipeline intentionally uses an aggressive duplicate policy.
    That is inappropriate for an explicit manual request: comparing hundreds
    of words of source/background text can mistake two different stories about
    the same artist for one story. The manual path therefore compares the
    event-bearing headline against existing titles/excerpts with deliberately
    high thresholds. This still catches exact and strongly reworded duplicates
    without treating a shared artist, band, instrument, or career detail as a
    duplicate by itself.
    """
    candidate = _normalize_story_text(candidate_title)
    candidate_tokens = _story_tokens(candidate_title)
    if not candidate or not candidate_tokens:
        return None

    for post in existing_posts:
        post_title = post.get("title", "")
        normalized_post_title = _normalize_story_text(post_title)
        if not normalized_post_title:
            continue

        if candidate == normalized_post_title:
            return post

        title_sequence = SequenceMatcher(None, candidate, normalized_post_title).ratio()
        post_title_tokens = _story_tokens(post_title)
        title_overlap = len(candidate_tokens & post_title_tokens)
        title_smaller = min(len(candidate_tokens), len(post_title_tokens))
        title_union = len(candidate_tokens | post_title_tokens)
        title_containment = title_overlap / title_smaller if title_smaller else 0
        title_jaccard = title_overlap / title_union if title_union else 0

        if title_sequence >= 0.86:
            return post
        if title_overlap >= 4 and title_containment >= 0.75 and title_jaccard >= 0.50:
            return post

        # A differently-worded existing headline may omit a key detail that is
        # present in its excerpt. Require nearly the whole requested headline
        # to appear across the existing title+excerpt before calling it the
        # same story; a couple of shared artist/background terms cannot pass.
        post_story_tokens = _story_tokens(f"{post_title} {post.get('excerpt', '')}")
        story_overlap = len(candidate_tokens & post_story_tokens)
        candidate_containment = story_overlap / len(candidate_tokens)
        if story_overlap >= 5 and candidate_containment >= 0.75:
            return post

    return None


def _duplicate_message(prefix, duplicate):
    location = duplicate.get("link") or "existing WordPress post"
    return f"{prefix}: {duplicate.get('title', 'Untitled')} ({location})"


def _write_job_summary(message):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(f"## Deadikace manual URL result\n\n{message}\n")


def run(source_url):
    source_url = _validate_source_url(source_url)
    print(f"[manual] Requested source URL: {source_url}")

    print("Checking connectivity to WordPress...")
    ok, error = check_connectivity()
    if not ok:
        raise RuntimeError(f"Could not reach WordPress: {error}")

    print("Fetching and parsing the requested source article...")
    topic = _build_topic(source_url)
    headline = topic["items"][0]["title"]
    print(f"[manual] Source article title: {headline}")

    print("Fetching existing published, draft, pending, and scheduled posts "
          "for duplicate checking...")
    existing_posts = get_recent_posts_for_dedup(per_page=100)
    duplicate = _find_clear_duplicate(headline, existing_posts)
    if duplicate:
        message = _duplicate_message(
            f"No draft was created because the requested story clearly matches '{headline}'",
            duplicate,
        )
        print(f"[manual] {message}")
        _write_job_summary(message)
        return "skipped"

    print("Researching additional factual context...")
    research_additional_context(topic)

    print("Drafting article with the existing editorial and SEO rules...")
    articles = draft_article(topic)
    if not articles:
        raise RuntimeError("The drafting step returned no article.")

    category_id = get_or_create_category(TARGET_CATEGORY)
    latest_posts = get_latest_posts(LATEST_POSTS_COUNT)
    created_count = 0

    for article in articles:
        article_title = article.get("title", headline)
        duplicate = _find_clear_duplicate(article_title, existing_posts)
        if duplicate:
            print(f"[manual] {_duplicate_message('Skipped generated duplicate', duplicate)}")
            continue

        print(f"Fact-checking and refining: {article_title}")
        try:
            article = verify_and_refine(article, topic)
        except Exception as exc:
            print(f"[warn] Fact-check pass raised an unexpected error ({exc}); "
                  "using the original draft unchanged.")

        article_title = article.get("title", headline)
        duplicate = _find_clear_duplicate(article_title, existing_posts)
        if duplicate:
            print(f"[manual] {_duplicate_message('Skipped refined duplicate', duplicate)}")
            continue

        related = search_related_posts(article.get("focus_keyword", headline))
        if related:
            links_html = "<p>Related reading: " + ", ".join(
                f'<a href="{post["link"]}">{post["title"]}</a>' for post in related
            ) + "</p>"
            article["content_html"] += "\n" + links_html

        article["content_html"] = _strip_image_placeholders(article["content_html"])
        featured_media_id, featured_hash, featured_phash = (
            _get_featured_media_for_topic(topic, article["title"], article.get("source_item_indices"))
            if ENABLE_SOURCE_IMAGES else (None, None, None)
        )
        article = _insert_illustrative_images(article, featured_hash, featured_phash)
        article = _insert_source_images(article, featured_hash, featured_phash)
        _append_latest_posts_block(article, latest_posts)
        article["content_html"] = _blockify(article["content_html"], ARTICLE_FONT_SIZE_PX)
        article = _insert_video_embeds(article)

        # Never inherit POST_STATUS here. A manual URL request is always a
        # reviewable WordPress draft, even when scheduled runs auto-publish.
        result = create_post(
            article,
            category_id=category_id,
            featured_media_id=featured_media_id,
            post_status="draft",
        )
        print(f"[manual] Created WordPress draft: {result.get('link', result.get('id'))}")
        created_count += 1
        existing_posts.append({
            "title": article.get("title", headline),
            "excerpt": article.get("excerpt", ""),
        })

    if not created_count:
        message = "No draft was created because every generated article clearly matched an existing story."
        print(f"[manual] {message}")
        _write_job_summary(message)
        return "skipped"

    message = f"Created {created_count} WordPress draft(s) from {source_url}."
    print(f"[manual] Done. {message}")
    _write_job_summary(message)
    return "created"


if __name__ == "__main__":
    try:
        run(os.environ.get("SOURCE_URL", ""))
    except (ValueError, RuntimeError, KeyError) as exc:
        print(f"[fatal] {exc}")
        sys.exit(1)
