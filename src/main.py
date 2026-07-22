"""
Entry point: run the full discover -> draft -> publish pipeline once.
Invoked on a schedule by the GitHub Actions workflow.
"""

import sys
import time
import html
import re
import io

from PIL import Image, ImageStat

from config import (
    MAX_ARTICLES_PER_RUN, POST_STATUS, TARGET_CATEGORY, LATEST_POSTS_COUNT,
    ARTICLE_FONT_SIZE_PX,
)
from discover import get_trending_topics
from draft import draft_article, filter_rock_relevant_topics, verify_and_refine
from article_fetch import enrich_topic_with_full_text
from wordpress import (
    get_recent_post_titles, get_latest_posts, get_or_create_category,
    create_post, search_related_posts, upload_media, check_connectivity,
)
from images import search_image as search_pexels_image, download_image as download_pexels_image
from wiki_images import find_real_photo, download_image as download_commons_image, extract_primary_subject


def _title_already_covered(title, existing_titles):
    """Very simple duplicate guard based on shared significant words."""
    title_words = {w.lower() for w in title.split() if len(w) > 4}
    for existing in existing_titles:
        existing_words = {w.lower() for w in existing.split() if len(w) > 4}
        if title_words and len(title_words & existing_words) / len(title_words) > 0.6:
            return True
    return False


def _build_collage(bytes_a, bytes_b, target_height=900):
    """Combines two portrait images side-by-side into one landscape image,
    for use as a featured image when the article has no landscape photo."""
    img_a = Image.open(io.BytesIO(bytes_a)).convert("RGB")
    img_b = Image.open(io.BytesIO(bytes_b)).convert("RGB")

    def _resize_to_height(img, height):
        w, h = img.size
        new_w = max(1, round(w * (height / h)))
        return img.resize((new_w, height))

    img_a = _resize_to_height(img_a, target_height)
    img_b = _resize_to_height(img_b, target_height)

    collage = Image.new("RGB", (img_a.width + img_b.width, target_height), color=(20, 20, 20))
    collage.paste(img_a, (0, 0))
    collage.paste(img_b, (img_a.width, 0))

    buf = io.BytesIO()
    collage.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _select_featured_image(processed_images, article_title):
    """
    Picks the featured/thumbnail image from the images already processed
    for this article. Prefers a landscape image (matches typical theme
    thumbnail crops much better than portrait); if none exist, builds a
    horizontal collage from two portrait images instead of using a single
    portrait image awkwardly cropped by the theme.
    """
    landscape = [img for img in processed_images if img["width"] and img["height"]
                 and img["width"] >= img["height"]]
    if landscape:
        return landscape[0]["media_id"]

    portraits = [img for img in processed_images if img["width"] and img["height"]]
    if len(portraits) >= 2:
        try:
            collage_bytes = _build_collage(portraits[0]["image_bytes"], portraits[1]["image_bytes"])
            collage_media = upload_media(
                collage_bytes, filename="featured-collage",
                alt_text=article_title, content_type="image/jpeg",
            )
            if collage_media:
                return collage_media["id"]
        except Exception as e:
            print(f"[warn] Failed to build featured-image collage from portrait images: {e}")

    if processed_images:
        return processed_images[0]["media_id"]
    return None


def _image_quality_ok(image_bytes, min_median_brightness=48, max_dark_fraction=0.55, dark_pixel_threshold=40):
    """
    Rejects images that are technically large enough but visually poor --
    e.g. a dark, distant concert photo where the subject is barely
    visible. Deliberately uses MEDIAN brightness and dark-pixel fraction,
    not the average: a concert photo with bright stage lights or a video
    screen against an otherwise pitch-black crowd/silhouette can have a
    perfectly normal AVERAGE brightness (the few very bright pixels pull
    the mean up) while still being mostly unreadable -- median and
    dark-fraction are robust to that because they look at how much of the
    frame is dark, not diluted by a handful of bright highlights.
    Fails open (returns True) if the image can't be analyzed, since a
    check that can't run shouldn't block an otherwise-fine image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img.thumbnail((300, 300))
        stat = ImageStat.Stat(img)
        median_brightness = stat.median[0]
        histogram = img.histogram()  # 256 buckets, grayscale
        total_pixels = sum(histogram) or 1
        dark_pixels = sum(histogram[:dark_pixel_threshold])
        dark_fraction = dark_pixels / total_pixels
    except Exception:
        return True, None

    if median_brightness < min_median_brightness:
        return False, f"too dark (median brightness {median_brightness:.0f}/255)"
    if dark_fraction > max_dark_fraction:
        return False, (f"majority of frame is dark/in shadow "
                        f"({dark_fraction * 100:.0f}% of pixels below brightness {dark_pixel_threshold})")
    return True, None


def _process_images(article):
    """
    Finds, downloads, and uploads an image for each IMAGE placeholder in
    content_html, then swaps the placeholders for real <figure> markup.
    Tries Wikimedia Commons first (real, properly-licensed photos of the
    actual bands/artists/events), and falls back to a generic Pexels stock
    photo only if no suitable Commons image is found. Returns the id of
    the selected featured image (see _select_featured_image), or None if
    no images succeeded.
    """
    queries = article.get("image_queries", [])
    content = article["content_html"]
    any_image_added = False
    used_urls = set()  # prevents the same photo being used twice in one article
    processed_images = []  # for featured-image selection after the loop

    for i, query in enumerate(queries, start=1):
        placeholder = f"<!--IMAGE_{i}-->"
        if placeholder not in content:
            continue

        # Gather candidates in preference order: Commons (full query),
        # Commons (simplified subject-only query), then Pexels. Each is
        # downloaded and quality-checked in turn -- a poor-quality image
        # (e.g. dark/blurry) is rejected in favor of the next candidate
        # rather than accepted just because it was the first match.
        candidates = []
        commons_photo = find_real_photo(query, exclude_urls=used_urls)
        if commons_photo:
            candidates.append(("commons", commons_photo))

        simplified = extract_primary_subject(query)
        if simplified and simplified.lower() != query.strip().lower():
            already_seen = {p["url"] for _, p in candidates}
            commons_photo_2 = find_real_photo(simplified, exclude_urls=used_urls | already_seen)
            if commons_photo_2:
                candidates.append(("commons", commons_photo_2))

        pexels_photo = search_pexels_image(query)
        if pexels_photo and pexels_photo["url"] not in used_urls:
            candidates.append(("pexels", pexels_photo))

        accepted = None
        for source, photo in candidates:
            if photo["url"] in used_urls:
                continue
            try:
                if source == "commons":
                    image_bytes, content_type = download_commons_image(photo["url"])
                else:
                    image_bytes, content_type = download_pexels_image(photo["url"])
            except Exception as e:
                print(f"[warn] Failed to download candidate image for '{query}' ({source}): {e}")
                continue

            ok, reason = _image_quality_ok(image_bytes)
            if not ok:
                print(f"[warn] Rejected low-quality image candidate for '{query}' ({source}): {reason}")
                continue

            accepted = (source, photo, image_bytes, content_type)
            break

        if not accepted:
            print(f"[warn] No usable, good-quality image found for query: '{query}'")
            content = content.replace(placeholder, "")
            continue

        source, photo, image_bytes, content_type = accepted
        used_urls.add(photo["url"])
        alt_text = query if source == "commons" else photo["alt"]

        try:
            media = upload_media(
                image_bytes,
                filename=f"image-{i}",
                alt_text=alt_text,
                content_type=content_type,
            )
        except Exception as e:
            print(f"[warn] Failed to upload image for '{query}' ({source}): {e}")
            content = content.replace(placeholder, "")
            continue

        if not media:
            content = content.replace(placeholder, "")
            continue

        any_image_added = True
        processed_images.append({
            "media_id": media["id"],
            "width": photo.get("width"),
            "height": photo.get("height"),
            "image_bytes": image_bytes,
        })

        if source == "commons":
            # Prefix the caption with the actual subject's name (e.g. "John
            # Coltrane, Photo: ...") when we can confidently extract one --
            # only done for Commons images since those are verified real
            # photos of the named subject, unlike the generic Pexels fallback.
            subject = extract_primary_subject(query)
            subject_prefix = f"{html.escape(subject)}, " if subject else ""
            credit_html = (
                f'{subject_prefix}Photo: <a href="{photo["page_url"]}" target="_blank" rel="nofollow noopener">'
                f'{html.escape(photo["artist"])}</a>, licensed '
                f'<a href="{photo["license_url"]}" target="_blank" rel="nofollow noopener">'
                f'{html.escape(photo["license_name"])}</a>, via Wikimedia Commons'
            )
        else:
            credit_html = f'Photo by {html.escape(photo["photographer"])} via Pexels'

        figure_html = (
            f'<figure class="wp-block-image size-large">'
            f'<img src="{media["source_url"]}" alt="{html.escape(alt_text)}"/>'
            f'<figcaption>{credit_html}</figcaption></figure>'
        )
        content = content.replace(placeholder, figure_html)

    # Clean up any unused placeholders (e.g. if model included more than it used)
    for i in range(1, 10):
        content = content.replace(f"<!--IMAGE_{i}-->", "")

    if not any_image_added:
        print("[warn] No images were added to this article at all. Check that "
              "PEXELS_API_KEY is set (Wikimedia Commons alone won't always have "
              "a match), and check the warnings above for the specific reason.")

    article["content_html"] = content
    return _select_featured_image(processed_images, article.get("title", "Deadikace"))


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

        # Find/download/upload images and swap in the placeholders
        featured_media_id = _process_images(article)

        # Append the "Latest Posts" block
        _append_latest_posts_block(article, latest_posts)

        # Convert to real Gutenberg blocks (keeps the post editable in the
        # block editor) with the medium font size applied per-paragraph
        article["content_html"] = _blockify(article["content_html"], ARTICLE_FONT_SIZE_PX)

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
