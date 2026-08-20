"""
Entry point: run the full discover -> draft -> publish pipeline once.
Invoked on a schedule by the GitHub Actions workflow.
"""

import sys
import time
import re
from difflib import SequenceMatcher

from config import (
    MAX_ARTICLES_PER_RUN, POST_STATUS, TARGET_CATEGORY, LATEST_POSTS_COUNT,
    ARTICLE_FONT_SIZE_PX, ENABLE_SOURCE_IMAGES,
)
from discover import get_trending_topics
from draft import (
    draft_article, filter_rock_relevant_topics, filter_duplicate_topics,
    dedupe_topics_within_batch, verify_and_refine, research_additional_context,
)
from article_fetch import enrich_topic_with_full_text, download_image_bytes
from wikimedia import search_commons_image, download_commons_image
from wordpress import (
    get_recent_posts_for_dedup, get_latest_posts, get_or_create_category,
    create_post, search_related_posts, check_connectivity, upload_media,
)

# Core Deadikace artists/bands. A matching recent topic receives a strong
# priority boost before drafting, but it STILL has to pass the normal rock-
# relevance and duplicate filters. Keeping this list here makes the editorial
# priority explicit in the entry-point workflow instead of relying on a GitHub
# secret that can silently be empty.
LEGENDARY_ARTISTS = [
    "led zeppelin", "pink floyd", "eric clapton", "mark knopfler",
    "neil young", "jimmy page", "robert plant", "david gilmour",
    "roger waters", "keith richards", "mick jagger", "the rolling stones",
    "paul mccartney", "bob dylan", "bruce springsteen", "ac/dc", "metallica",
    "ozzy osbourne", "black sabbath", "deep purple", "dire straits", "the who",
    "fleetwood mac", "eagles", "aerosmith", "jimi hendrix", "cream", "the doors",
    "janis joplin", "lynyrd skynyrd", "santana", "stevie ray vaughan",
    "chuck berry", "elvis presley", "van halen", "guns n' roses", "bon jovi",
    "iron maiden", "judas priest", "motorhead", "rush", "yes", "genesis",
    "thin lizzy", "gary moore", "joe bonamassa", "jeff beck", "tina turner",
    "the kinks", "the animals", "grateful dead", "steely dan", "zz top",
    "queen", "the beatles", "john lennon", "george harrison", "ringo starr",
    "david bowie", "elton john", "rod stewart", "tom petty", "alice cooper",
    "kiss", "scorpions", "def leppard", "rainbow", "jethro tull", "king crimson",
    "the clash", "ramones", "nirvana", "pearl jam", "soundgarden", "foo fighters",
    "red hot chili peppers", "oasis", "the cure", "radiohead", "heart", "journey",
    "foreigner", "kansas", "chicago", "creedence clearwater revival",
]

LEGENDARY_PRIORITY_BONUS = 80
LEGENDARY_MAJOR_NEWS_BONUS = 40
_MAJOR_NEWS_TERMS = (
    " dies", " died", " dead", " death", " passes away", " passed away",
    " obituary", " hospitalized", " hospitalised", " cancer", " health crisis",
    " reunion", " reunites", " reunited", " breakup", " break-up", " retires",
    " retirement",
)

# Deterministic duplicate guard. The LLM duplicate checker remains useful for
# semantically different headlines, but this layer cannot disappear because of
# a quota/API/JSON failure. It catches obvious same-story variants across
# outlets, drafts, published posts, and different runs.
_STORY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "at",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "from", "after",
    "before", "into", "over", "about", "this", "that", "these", "those", "their",
    "his", "her", "its", "new", "latest", "report", "reports", "says", "say",
    "reveals", "revealed", "announces", "announced", "update", "news",
}

_OBVIOUS_METAL_PATTERNS = (
    "metal band", "metal bands", "metal scene", "metal festival", "metal festivals",
    "metal tour", "metal tours", "metal act", "metal acts", "metal album",
    "death metal", "black metal", "doom metal", "sludge metal", "thrash metal",
    "power metal", "symphonic metal", "progressive metal", "metalcore", "deathcore",
    "djent", "nu-metal", "nu metal",
)


def _contains_artist(text, artist):
    return re.search(r"(?<!\w)" + re.escape(artist) + r"(?!\w)", text, flags=re.IGNORECASE) is not None


def _topic_text(topic):
    parts = []
    for item in topic.get("items", []):
        parts.append(item.get("title", ""))
        parts.append(item.get("summary", ""))
    return " ".join(parts)


def _apply_legendary_priority(topics):
    """Boost and re-sort topics involving Deadikace's core artists."""
    for topic in topics:
        text = _topic_text(topic).lower()
        matched = [artist for artist in LEGENDARY_ARTISTS if _contains_artist(text, artist)]
        if not matched:
            continue

        bonus = LEGENDARY_PRIORITY_BONUS
        if any(term in f" {text}" for term in _MAJOR_NEWS_TERMS):
            bonus += LEGENDARY_MAJOR_NEWS_BONUS
        topic["score"] = round(float(topic.get("score", 0)) + bonus, 1)
        topic["legendary_priority"] = matched

    topics.sort(key=lambda t: t.get("score", 0), reverse=True)
    top_priority = [t for t in topics if t.get("legendary_priority")]
    if top_priority:
        preview = ", ".join(
            f"{t['items'][0]['title']} (+{LEGENDARY_PRIORITY_BONUS} priority)"
            for t in top_priority[:5]
        )
        print(f"[info] Legendary-artist priority applied: {preview}")
    return topics


def _drop_obvious_metal_topics(topics):
    """
    Hard safety net for clearly metal-specific stories.

    Crossover legendary artists are left to the semantic relevance classifier,
    so a broad Black Sabbath/Metallica/Ozzy story is not rejected just because
    a source calls the artist metal. Generic stories like 'Indonesian Metal
    Bands ...' are removed before they can consume an LLM call or draft slot.
    """
    kept = []
    for topic in topics:
        text = _topic_text(topic).lower()
        legendary_match = any(_contains_artist(text, artist) for artist in LEGENDARY_ARTISTS)
        if not legendary_match and any(pattern in text for pattern in _OBVIOUS_METAL_PATTERNS):
            print(f"[info] Skipping obvious metal-specific topic: {topic['items'][0]['title']}")
            continue
        kept.append(topic)
    return kept


def _normalize_story_text(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9'&/+.-]+", " ", text)
    return " ".join(text.split())


def _story_tokens(text):
    normalized = _normalize_story_text(text)
    return {
        token.strip("'./+-")
        for token in normalized.split()
        if len(token.strip("'./+-")) >= 3
        and token.strip("'./+-") not in _STORY_STOPWORDS
    }


def _is_probable_same_story(text_a, text_b):
    """
    Conservative deterministic same-story test.

    It intentionally requires several meaningful shared terms, so two stories
    about the same artist but different events (e.g. a tour vs. an obituary)
    are not treated as duplicates merely because the artist name matches.
    """
    a = _normalize_story_text(text_a)
    b = _normalize_story_text(text_b)
    if not a or not b:
        return False
    if a == b:
        return True

    ta = _story_tokens(a)
    tb = _story_tokens(b)
    if not ta or not tb:
        return False

    overlap = len(ta & tb)
    smaller = min(len(ta), len(tb))
    union = len(ta | tb)
    containment = overlap / smaller if smaller else 0
    jaccard = overlap / union if union else 0
    sequence = SequenceMatcher(None, a, b).ratio()

    return (
        (overlap >= 5 and containment >= 0.50)
        or (overlap >= 4 and containment >= 0.62)
        or (overlap >= 3 and jaccard >= 0.50)
        or sequence >= 0.78
    )


def _post_text(post):
    return f"{post.get('title', '')} {post.get('excerpt', '')}"


def _topic_already_covered(topic, existing_posts):
    candidate = _topic_text(topic)
    return any(_is_probable_same_story(candidate, _post_text(post)) for post in existing_posts)


def _article_already_covered(article, existing_posts):
    candidate = f"{article.get('title', '')} {article.get('excerpt', '')}"
    return any(_is_probable_same_story(candidate, _post_text(post)) for post in existing_posts)


def _dedupe_against_posts_deterministic(topics, existing_posts):
    kept = []
    for topic in topics:
        if _topic_already_covered(topic, existing_posts):
            print(f"[info] Deterministic dedupe skipped existing story: {topic['items'][0]['title']}")
            continue
        kept.append(topic)
    return kept


def _dedupe_topics_deterministic(topics):
    kept = []
    kept_texts = []
    for topic in topics:
        text = _topic_text(topic)
        if any(_is_probable_same_story(text, previous) for previous in kept_texts):
            print(f"[info] Deterministic within-run dedupe skipped: {topic['items'][0]['title']}")
            continue
        kept.append(topic)
        kept_texts.append(text)
    return kept


def _title_already_covered(title, existing_titles):
    """Backward-compatible title guard using the stronger story matcher."""
    return any(_is_probable_same_story(title, existing) for existing in existing_titles)


def _strip_image_placeholders(content_html):
    """
    Image sourcing (Wikimedia Commons search + Pexels stock-photo fallback)
    has been removed -- it was matching the wrong subject often enough to
    be worse than no image at all (e.g. returning a photo of an actual
    eagle for an article about the band Eagles). Rather than re-attempt
    automatic image sourcing, articles are simply published without
    inline images for now, so any `<!--IMAGE_N-->` placeholder the model
    still emits is just stripped out.
    """
    for i in range(1, 10):
        content_html = content_html.replace(f"<!--IMAGE_{i}-->", "")
    return content_html


def _get_featured_media_for_topic(topic, article_title, source_item_indices=None):
    """
    Downloads a source article's own preview image (og:image) and
    re-hosts it on Deadikace as the featured image, since automatic
    image search/download was removed after repeatedly matching the
    wrong subject. See the IMAGE SOURCING NOTE in article_fetch.py for
    the copyright tradeoff this involves. Returns a media ID, or None
    if no source item had a usable image or the download/upload failed.

    source_item_indices (1-based, matching the "Source #N" numbering the
    drafting model saw) restricts which of the topic's source items are
    considered. This matters when one topic produced MULTIPLE articles
    (rule 0 in DRAFT_SYSTEM_PROMPT splitting unrelated stories) -- without
    it, an article could end up with another split article's image, since
    topic["items"] contains every source across all of them. If not
    provided, falls back to considering every item in the topic (the
    original, pre-split behavior).
    """
    items = topic.get("items", [])
    if source_item_indices:
        selected = {i - 1 for i in source_item_indices}
        items = [item for i, item in enumerate(items) if i in selected]

    for item in items:
        image_url = item.get("image_url")
        if not image_url:
            continue
        image_bytes, content_type = download_image_bytes(image_url)
        if not image_bytes:
            continue
        alt_text = item.get("image_caption") or f"Credit: {item.get('source', 'source article')}"
        try:
            media = upload_media(
                image_bytes,
                filename=article_title,
                alt_text=alt_text,
                content_type=content_type,
            )
        except Exception as e:
            print(f"[warn] Failed to upload source image to WordPress media library: {e}")
            continue
        if media:
            print(f"[info] Using source image from {item.get('source', 'source article')} as featured image "
                  f"(caption: {alt_text!r}).")
            return media["id"]
    return None


def _insert_illustrative_images(article):
    """
    Processes article["illustrative_images"] (proposed by the drafting
    model -- see rule 4c in DRAFT_SYSTEM_PROMPT): searches Wikimedia
    Commons for each specific query, uploads any match found to the
    WordPress media library, and splices a <figure> with the image and
    caption into content_html at the requested position (right after the
    named H2 heading, or roughly in the middle if no heading was given).
    Silently skips any image that fails to find a match, download, or
    upload -- a missing illustrative image is not worth failing the
    whole article over.
    """
    requests_list = article.get("illustrative_images") or []
    if not requests_list:
        return article

    content_html = article["content_html"]

    for req in requests_list:
        query = (req.get("query") or "").strip()
        if not query:
            continue

        result = search_commons_image(query)
        if not result:
            print(f"[info] No Wikimedia Commons match found for illustrative image query: {query!r}")
            continue

        image_bytes, content_type = download_commons_image(result["url"])
        if not image_bytes:
            continue

        caption_text = req.get("caption") or result["credit"]
        try:
            media = upload_media(
                image_bytes,
                filename=result["title"] or query,
                alt_text=caption_text,
                content_type=content_type,
            )
        except Exception as e:
            print(f"[warn] Failed to upload illustrative image to WordPress media library: {e}")
            continue
        if not media:
            continue

        figure_html = (
            f'<figure class="wp-block-image size-large">'
            f'<img src="{media["source_url"]}" alt="{caption_text}" />'
            f'<figcaption>{caption_text} ({result["credit"]})</figcaption>'
            f'</figure>'
        )

        heading = (req.get("placement_after_heading") or "").strip()
        inserted = False
        if heading:
            heading_match = re.search(
                r"(<h2[^>]*>\s*" + re.escape(heading) + r"\s*</h2>)",
                content_html, re.IGNORECASE,
            )
            if heading_match:
                insert_pos = heading_match.end()
                content_html = content_html[:insert_pos] + figure_html + content_html[insert_pos:]
                inserted = True
        if not inserted:
            paragraphs = list(re.finditer(r"</p>", content_html, re.IGNORECASE))
            if paragraphs:
                mid = paragraphs[len(paragraphs) // 2]
                insert_pos = mid.end()
                content_html = content_html[:insert_pos] + figure_html + content_html[insert_pos:]
            else:
                content_html += figure_html

        print(f"[info] Inserted illustrative image for query {query!r} "
              f"({'after heading' if inserted else 'mid-article fallback'}).")

    article["content_html"] = content_html
    return article


def _insert_video_embeds(article):
    """
    Processes article["video_embeds"] (proposed by the drafting model --
    see rule 4d in DRAFT_SYSTEM_PROMPT): splices a real YouTube <iframe>
    embed into content_html at the requested position (right after the
    named H2 heading's block, or mid-article as a fallback).

    IMPORTANT: this must run AFTER _blockify(), not before. Two reasons:
    1. A hand-built iframe is used instead of WordPress's bare-URL
       oEmbed autoembed, because that depends on a live oEmbed API call
       succeeding at render time and can silently leave a plain
       unclickable link behind if a caching layer serves the page
       before that resolves -- a direct iframe always works.
    2. _blockify()'s regex only recognizes <p>/<h2>/<h3>/<figure>/<ol>/
       <ul> and would wrap any <figure>-based embed as a "wp:image"
       block, which is semantically wrong for a video and can trigger a
       block-validation warning in the editor. Running this after
       blockify and wrapping the embed in its own proper "wp:html"
       block avoids that entirely.
    """
    video_list = article.get("video_embeds") or []
    if not video_list:
        return article

    content_html = article["content_html"]

    for video in video_list:
        url = (video.get("url") or "").strip()
        if not url:
            continue

        video_id_match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]+)", url)
        if not video_id_match:
            continue
        video_id = video_id_match.group(1)

        iframe_html = (
            '<figure style="position:relative;padding-bottom:56.25%;height:0;'
            'overflow:hidden;max-width:100%;margin:20px 0;">'
            f'<iframe src="https://www.youtube.com/embed/{video_id}" '
            'title="YouTube video player" '
            'style="position:absolute;top:0;left:0;width:100%;height:100%;" '
            'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
            'encrypted-media; gyroscope; picture-in-picture; web-share" '
            'referrerpolicy="strict-origin-when-cross-origin" allowfullscreen>'
            '</iframe></figure>'
        )
        embed_html = f'<!-- wp:html -->\n{iframe_html}\n<!-- /wp:html -->'

        heading = (video.get("placement_after_heading") or "").strip()
        inserted = False
        if heading:
            heading_match = re.search(
                r"(<!-- wp:heading -->\s*<h2[^>]*>\s*" + re.escape(heading)
                + r"\s*</h2>\s*<!-- /wp:heading -->)",
                content_html, re.IGNORECASE,
            )
            if heading_match:
                insert_pos = heading_match.end()
                content_html = content_html[:insert_pos] + "\n\n" + embed_html + content_html[insert_pos:]
                inserted = True
        if not inserted:
            para_blocks = list(re.finditer(r"<!-- /wp:paragraph -->", content_html, re.IGNORECASE))
            if para_blocks:
                mid = para_blocks[len(para_blocks) // 2]
                insert_pos = mid.end()
                content_html = content_html[:insert_pos] + "\n\n" + embed_html + content_html[insert_pos:]
            else:
                content_html += "\n\n" + embed_html

        print(f"[info] Embedded video {url} "
              f"({'after heading' if inserted else 'mid-article fallback'}).")

    article["content_html"] = content_html
    return article


def _insert_source_images(article):
    """
    Processes article["source_images"] (proposed by the drafting model --
    see rule 4f in DRAFT_SYSTEM_PROMPT): downloads each image directly
    from the source article's own body and inserts it into content_html
    at the requested position. Used for things like tour posters or
    promotional artwork that Wikimedia Commons usually has no
    free-licensed equivalent for -- same copyright tradeoff already
    accepted for the featured image (see the IMAGE SOURCING NOTE in this
    file). Mirrors _insert_illustrative_images()'s insertion logic, just
    sourcing the image bytes directly from the source URL instead of a
    Wikimedia Commons search.
    """
    source_image_list = article.get("source_images") or []
    if not source_image_list:
        return article

    content_html = article["content_html"]

    for img_entry in source_image_list:
        url = (img_entry.get("url") or "").strip()
        if not url:
            continue

        image_bytes, content_type = download_image_bytes(url)
        if not image_bytes:
            print(f"[warn] Failed to download source-body image {url}; skipping.")
            continue

        caption = (img_entry.get("caption") or "").strip()
        try:
            media = upload_media(
                image_bytes,
                filename=article["title"],
                alt_text=caption or "Photo via source article",
                content_type=content_type,
            )
        except Exception as e:
            print(f"[warn] Failed to upload source-body image to WordPress media library: {e}")
            continue
        if not media:
            continue

        figure_html = f'<figure class="wp-block-image size-large"><img src="{media["source_url"]}" alt="{caption}"/>'
        if caption:
            figure_html += f'<figcaption>{caption}</figcaption>'
        figure_html += '</figure>'

        heading = (img_entry.get("placement_after_heading") or "").strip()
        inserted = False
        if heading:
            heading_match = re.search(
                r"(<h2[^>]*>\s*" + re.escape(heading) + r"\s*</h2>)",
                content_html, re.IGNORECASE,
            )
            if heading_match:
                insert_pos = heading_match.end()
                content_html = content_html[:insert_pos] + figure_html + content_html[insert_pos:]
                inserted = True
        if not inserted:
            paragraphs = list(re.finditer(r"</p>", content_html, re.IGNORECASE))
            if paragraphs:
                mid = paragraphs[len(paragraphs) // 2]
                insert_pos = mid.end()
                content_html = content_html[:insert_pos] + figure_html + content_html[insert_pos:]
            else:
                content_html += figure_html

        print(f"[info] Inserted source-body image {url} "
              f"({'after heading' if inserted else 'mid-article fallback'}).")

    article["content_html"] = content_html
    return article


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

    # Apply Deadikace's own editorial priorities BEFORE the final selection.
    # This ensures major news about ZZ Top, Pink Floyd, Led Zeppelin, etc. is
    # not outranked by generic single-source listicles merely because of RSS
    # timing. A death/major-health/reunion story gets an additional boost.
    topics = _apply_legendary_priority(topics)

    # Cheap deterministic safety net before spending an LLM call. This catches
    # obvious off-topic cases such as "Indonesian Metal Bands..." even if the
    # semantic relevance classifier later hits quota or returns malformed JSON.
    topics = _drop_obvious_metal_topics(topics)

    print("Filtering for rock-relevant topics (some feeds cover all genres)...")
    topics = filter_rock_relevant_topics(topics)

    if not topics:
        print("No rock-relevant topics found this run. Exiting.")
        return

    print("Fetching recent Deadikace posts/drafts to avoid duplicates...")
    existing_posts = get_recent_posts_for_dedup(per_page=100)
    existing_titles = [p["title"] for p in existing_posts]

    # First deterministic pass: this works even if Gemini/Claude duplicate
    # classification is unavailable. Then keep the existing semantic LLM pass
    # for harder differently-worded cases.
    topics = _dedupe_against_posts_deterministic(topics, existing_posts)

    print("Checking candidate topics against recently published posts/drafts to avoid duplicate stories, even if they were covered by a different outlet...")
    topics = filter_duplicate_topics(topics, existing_posts)

    print("Checking candidate topics against each other, in case topic clustering "
          "missed that two of them are actually the same underlying story...")
    topics = dedupe_topics_within_batch(topics)
    topics = _dedupe_topics_deterministic(topics)

    print(f"Resolving target category '{TARGET_CATEGORY}'...")
    category_id = get_or_create_category(TARGET_CATEGORY)

    latest_posts = get_latest_posts(LATEST_POSTS_COUNT)

    published_count = 0
    for topic in topics:
        if published_count >= MAX_ARTICLES_PER_RUN:
            break

        headline = topic["items"][0]["title"]
        if _topic_already_covered(topic, existing_posts) or _title_already_covered(headline, existing_titles):
            print(f"Skipping (likely already covered): {headline}")
            continue

        print(f"Drafting article for topic: {headline} "
              f"(covered by {topic['source_count']} outlet(s))")

        print("Fetching full source article text for factual grounding...")
        enrich_topic_with_full_text(topic)

        print("Researching additional factual context beyond the immediate "
              "competitor coverage (chart/catalog data, documented prior "
              "statements, historical background)...")
        research_additional_context(topic)

        if published_count > 0:
            time.sleep(10)  # stay comfortably under free-tier requests-per-minute limits

        try:
            articles = draft_article(topic)
        except Exception as e:
            error_text = str(e)
            print(f"[error] Failed to draft article for '{headline}': {e}")
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

            # Final in-memory duplicate guard before doing any media upload or
            # WordPress write. This is especially important when one topic was
            # split into multiple drafts or another post was created earlier in
            # this same workflow run.
            if _article_already_covered(article, existing_posts):
                print(f"[info] Final duplicate guard skipped: {article.get('title', headline)}")
                continue

            print("Fact-checking the draft against the source material...")
            try:
                article = verify_and_refine(article, topic)
            except Exception as e:
                print(f"[warn] Fact-check pass raised an unexpected error ({e}); "
                      "publishing the original draft unchanged.")

            # Check once more after refinement in case the final title/excerpt
            # became closer to an existing post than the initial draft was.
            if _article_already_covered(article, existing_posts):
                print(f"[info] Final duplicate guard skipped after refinement: {article.get('title', headline)}")
                continue

            related = search_related_posts(article.get("focus_keyword", headline))
            if related:
                links_html = "<p>Related reading: " + ", ".join(
                    f'<a href="{r["link"]}">{r["title"]}</a>' for r in related
                ) + "</p>"
                article["content_html"] += "\n" + links_html

            article["content_html"] = _strip_image_placeholders(article["content_html"])
            featured_media_id = (
                _get_featured_media_for_topic(
                    topic, article["title"], article.get("source_item_indices")
                )
                if ENABLE_SOURCE_IMAGES else None
            )

            article = _insert_illustrative_images(article)
            article = _insert_source_images(article)
            _append_latest_posts_block(article, latest_posts)
            article["content_html"] = _blockify(article["content_html"], ARTICLE_FONT_SIZE_PX)
            article = _insert_video_embeds(article)

            try:
                result = create_post(article, category_id=category_id, featured_media_id=featured_media_id)
            except Exception as e:
                print(f"[error] Failed to publish article for '{article.get('title', headline)}': {e}")
                continue

            print(f"Created post (status={POST_STATUS}): {result.get('link', result.get('id'))}")
            published_count += 1

            # CRITICAL: immediately add the new draft/published post to the
            # in-memory dedupe pool. Without this, a later candidate in the SAME
            # run can cover the same event because WordPress was only queried at
            # the beginning of the run.
            new_post = {
                "title": article.get("title", headline),
                "excerpt": article.get("excerpt", ""),
            }
            existing_posts.append(new_post)
            existing_titles.append(new_post["title"])

            time.sleep(2)

    print(f"Done. Published/drafted {published_count} article(s) this run.")


if __name__ == "__main__":
    try:
        run()
    except KeyError as e:
        print(f"[fatal] Missing required environment variable: {e}")
        sys.exit(1)
