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

IMAGE SOURCING NOTE: automatic image search/download (Wikimedia/Pexels)
was removed after it repeatedly matched the wrong subject (e.g. a photo
of an actual eagle for an article about the band Eagles). As a safer
fallback, this module can pull the source article's own og:image (the
same "preview" photo used when that article gets shared on social media)
and Deadikace re-hosts it as the featured image, attributed to the
original outlet in the alt text. This is at least topically correct,
since it's the outlet's own chosen photo for that specific story -- but
it is NOT a copyright-clear image the way a CC-licensed Wikimedia Commons
photo would be. Most competitor news photos are copyrighted, not freely
licensed, so re-hosting them carries real legal risk; this is a deliberate
tradeoff the site owner has chosen (image relevance over image licensing
certainty) after weighing both options, not a default best practice.
"""

import hashlib
import re
from urllib.parse import urlsplit

import requests
import trafilatura

USER_AGENT = "DeadikaceAgent/1.0 (https://www.deadikace.com)"


def _normalize_image_key(url):
    """
    Reduces an image URL down to a comparable "same underlying photo" key,
    ignoring things that differ between two URLs that are really just
    different renditions of the same source image:
      - scheme/host (a CDN can serve the same photo from several hosts)
      - query string (cache-busting/resize params)
      - a trailing WordPress-style resize suffix like "-1024x683" or a
        retina suffix like "@2x" right before the extension

    This matters because the og:image URL (used for the featured image)
    and the <img src> for that same photo inside the article body are very
    often two different-sized renditions of one image, e.g.
    ".../photo-1200x630.jpg" (og:image) vs ".../photo-780x520.jpg" (body).
    A plain substring/equality check on the raw URLs misses this and lets
    the same photo get pulled in twice -- once as the featured image, once
    again as an inline "source image" -- which is the main image-duplication
    bug this key exists to prevent. Returns "" if no usable filename can be
    extracted (comparisons against "" never match).
    """
    if not url:
        return ""
    path = urlsplit(url).path
    filename = path.rsplit("/", 1)[-1]
    if not filename:
        return ""
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        stem, ext = filename, ""
    stem = re.sub(r"-\d{2,5}x\d{2,5}$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"@\d+x$", "", stem, flags=re.IGNORECASE)
    return f"{stem.lower()}.{ext.lower()}" if dot else stem.lower()


def image_bytes_hash(image_bytes):
    """
    SHA-256 hex digest of raw image bytes, or None. Used as a last-resort,
    belt-and-suspenders duplicate check (in addition to _normalize_image_key)
    right before an image actually gets inserted, since two different URLs
    can still resolve to byte-for-byte the same file in ways no URL-based
    heuristic can anticipate.
    """
    if not image_bytes:
        return None
    return hashlib.sha256(image_bytes).hexdigest()

def fetch_article_text(url, max_chars=5000):
    """Returns extracted main article text, or None if fetching/extraction fails."""
    html = _fetch_html(url)
    if not html:
        return None
    try:
        text = trafilatura.extract(html, include_comments=False, favor_recall=True)
    except Exception as e:
        print(f"[warn] Failed to extract article text from {url}: {e}")
        return None

    if not text:
        return None
    return text.strip()[:max_chars]

def _fetch_html(url):
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"[warn] Failed to fetch {url}: {e}")
        return None

def fetch_source_image_url(url, html=None):
    """
    Returns (image_url, caption_text) for the source article's og:image
    (the same preview photo used when the article is shared on social
    media), or (None, None) if unavailable. caption_text is a best-effort
    extraction of any figcaption/photo-credit text near that image in the
    source HTML (e.g. "Photo: Jane Smith, licensed CC BY 2.0") -- None if
    nothing usable was found, in which case a generic "Photo via {source}"
    credit should be used as a fallback by the caller.
    See the IMAGE SOURCING NOTE at the top of this file for the copyright
    tradeoff this involves -- this is a deliberate, explicitly-requested
    fallback, not a default recommendation.
    """
    if html is None:
        html = _fetch_html(url)
    if not html:
        return None, None
    try:
        metadata = trafilatura.extract_metadata(html)
    except Exception as e:
        print(f"[warn] Failed to extract image metadata from {url}: {e}")
        return None, None
    image_url = metadata.image if metadata else None
    if not image_url:
        return None, None
    # Prefer the caption/alt text tied to this SPECIFIC image (the one that
    # will be reused as the featured image), so the featured image's caption
    # matches what a reader sees on that same photo inside the article body
    # -- not a generic "Credit: {source}" line. Falls back to a page-wide
    # credit-line scan only if that specific image tag can't be found (e.g.
    # lazy-loaded via data-src/srcset instead of a plain src attribute).
    caption_text = _find_caption_for_image_url(html, image_url) or _extract_image_caption(html)
    return image_url, caption_text

def _find_caption_for_image_url(html, image_url):
    """
    Finds the caption text actually attached to the <img> tag in html whose
    src matches image_url (via _normalize_image_key, so a resized/CDN
    variant of the same photo still matches). Prefers a <figcaption> inside
    the nearest enclosing <figure>, falling back to the img's own alt text.
    Returns None if no matching <img> tag is found.
    """
    target_key = _normalize_image_key(image_url)
    if not target_key:
        return None

    for m in re.finditer(r"<img[^>]+>", html, re.IGNORECASE):
        tag = m.group(0)
        src_match = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        if not src_match or _normalize_image_key(src_match.group(1)) != target_key:
            continue

        figure_start = html.rfind("<figure", 0, m.start())
        if figure_start != -1:
            figure_end = html.find("</figure>", m.end())
            if figure_end != -1:
                fc_match = re.search(
                    r"<figcaption[^>]*>(.*?)</figcaption>",
                    html[figure_start:figure_end],
                    re.IGNORECASE | re.DOTALL,
                )
                if fc_match:
                    text = re.sub(r"<[^>]+>", " ", fc_match.group(1))
                    text = " ".join(text.split())
                    if text and len(text) < 300:
                        return text

        alt_match = re.search(r'alt=["\']([^"\']*)["\']', tag, re.IGNORECASE)
        if alt_match:
            alt_text = " ".join(alt_match.group(1).split())
            if alt_text and len(alt_text) < 300:
                return alt_text

    return None

def _extract_image_caption(html):
    """
    Best-effort extraction of a SHORT photo credit line near the
    article's lead image.

    IMPORTANT: credit detection must never scan CSS or JavaScript as if it
    were human-readable page text. A selector such as ``.image:after`` can
    otherwise be mistaken for an ``Image:`` credit and produce garbage like
    ``Credit: after{content...``.
    """
    figcaption_match = re.search(
        r"<figcaption[^>]*>(.*?)</figcaption>",
        html,
        re.IGNORECASE | re.DOTALL,
    )
    full_caption = None
    if figcaption_match:
        text = re.sub(r"<[^>]+>", " ", figcaption_match.group(1))
        text = " ".join(text.split())
        if text and len(text) < 300:
            full_caption = text

    if not full_caption:
        # Remove code/style regions before looking for visible-text credit
        # labels. Searching raw HTML caused CSS such as `.image:after` to be
        # interpreted as a photo credit.
        visible_html = re.sub(
            r"<(?:style|script|noscript)[^>]*>.*?</(?:style|script|noscript)>",
            " ",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        visible_text = re.sub(r"<[^>]+>", " ", visible_html)
        visible_text = re.sub(r"\s+", " ", visible_text)

        credit_match = re.search(
            r"\b((?:Photo|Image|Credit)\s*:\s*[^|<>\n]{3,150})",
            visible_text,
            re.IGNORECASE,
        )
        if credit_match:
            candidate = " ".join(credit_match.group(1).split())
            # Final sanity filter for anything that still looks like CSS/JS.
            lowered = candidate.lower()
            code_markers = (
                "{", "}", ";", "content:", "display:", "padding-", "margin-",
                "position:", "block;", "none;", "::before", "::after", ":after",
                ":before", "function(", "var(",
            )
            if not any(marker in lowered for marker in code_markers):
                full_caption = candidate

    if not full_caption:
        return None

    return _shorten_to_credit(full_caption)

def _shorten_to_credit(full_caption):
    """
    Reduces a full scraped caption down to just a short "Credit: X" line,
    since the corner overlay only has room for a short credit, not a
    full description. Tries a few common patterns before giving up and
    returning None (caller falls back to a generic "Photo via {source}"
    line in that case).
    """
    # Never turn stylesheet/script fragments into a public-facing credit.
    lowered = full_caption.lower()
    if any(marker in lowered for marker in (
        "{", "}", ";", "content:", "display:", "padding-", "position:",
        "::before", "::after", ":before", ":after", "function(", "var(",
    )):
        return None

    m = re.search(r"\bvia\s+([A-Z][\w.&'-]{2,40}(?:\s+[A-Z][\w.&'-]{1,40}){0,3})\s*\.?$", full_caption)
    if m:
        return f"Credit: {m.group(1).strip().rstrip('.')}"

    m = re.match(r"^(?:Photo|Image|Credit)\s*:\s*(.{2,60}?)(?:\.|,|$)", full_caption, re.IGNORECASE)
    if m:
        credit = m.group(1).strip()
        if credit and not any(ch in credit for ch in "{};"):
            return f"Credit: {credit}"

    known_agencies = [
        "Getty Images", "Associated Press", "Reuters", "Shutterstock",
        "WireImage", "AFP", "Redferns", "FilmMagic",
    ]
    for agency in known_agencies:
        if agency.lower() in full_caption.lower():
            return f"Credit: {agency}"

    return None

def download_image_bytes(image_url, max_bytes=8_000_000):
    """Downloads an image and returns (bytes, content_type), or (None, None) on failure."""
    try:
        resp = requests.get(image_url, headers={"User-Agent": USER_AGENT}, timeout=20, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith("image/"):
            print(f"[warn] URL did not return an image content-type ({content_type}): {image_url}")
            return None, None
        data = resp.raw.read(max_bytes + 1, decode_content=True)
        if len(data) > max_bytes:
            print(f"[warn] Image at {image_url} exceeds {max_bytes} bytes; skipping.")
            return None, None
        return data, content_type
    except requests.RequestException as e:
        print(f"[warn] Failed to download image {image_url}: {e}")
        return None, None

def enrich_topic_with_full_text(topic, max_sources=4):
    """
    Mutates topic['items'] in place, adding 'full_text', 'image_url', and
    'video_embeds' keys to up to max_sources of them (the top outlets
    covering the story). Items that fail to fetch simply keep these as
    None/empty, and drafting falls back to the RSS summary for those.
    """
    fetched_count = 0
    for item in topic["items"]:
        if fetched_count >= max_sources:
            item["full_text"] = None
            item["image_url"] = None
            item["image_caption"] = None
            item["video_embeds"] = []
            item["body_images"] = []
            continue
        html = _fetch_html(item["link"])
        if html:
            try:
                text = trafilatura.extract(html, include_comments=False, favor_recall=True)
            except Exception as e:
                print(f"[warn] Failed to extract article text from {item['link']}: {e}")
                text = None
            item["full_text"] = text.strip()[:5000] if text else None
            item["image_url"], item["image_caption"] = fetch_source_image_url(item["link"], html=html)
            item["video_embeds"] = extract_video_embeds(html)
            item["body_images"] = extract_body_images(html, exclude_url_fragment=item["image_url"])
        else:
            item["full_text"] = None
            item["image_url"] = None
            item["image_caption"] = None
            item["video_embeds"] = []
            item["body_images"] = []
        if item["full_text"]:
            fetched_count += 1
    return topic

def _extract_article_body_html(html):
    """
    Best-effort isolation of just the article body HTML, so video/heading
    extraction doesn't wander into sidebars, "related articles", or
    "you might also like" widgets elsewhere on the page. Those sections
often contain embedded videos for a COMPLETELY different story (this
    is exactly how an unrelated video ended up attached to an article
    about a different artist entirely) -- scoping to <article> avoids
    picking those up. Falls back to the full page if no <article> tag is
    found, since that's still better than extracting nothing.
    """
    match = re.search(r"<article[^>]*>(.*?)</article>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else html

def extract_video_embeds(html):
    """
    Extracts YouTube video links embedded in the source article, along
    with the nearest preceding <h2>/<h3> heading text (used to match each
    video to the right section when the drafting model restructures the
    article under its own headings). Returns a list of
    {"url": ..., "heading": ...} dicts, deduplicated by video ID. Only
    YouTube is handled for now, since it's by far the most common
    platform for the kind of archival live-performance clips this blog
    covers. Scoped to the article body only (see
    _extract_article_body_html) so sidebar/recommended-video widgets for
    unrelated stories don't get swept in.
    """
    html = _extract_article_body_html(html)
    results = []
    seen_ids = set()

    for m in re.finditer(
        r'<iframe[^>]+src=["\']https?://(?:www\.)?youtube(?:-nocookie)?\.com/embed/([A-Za-z0-9_-]{6,})[^"\']*["\']',
        html, re.IGNORECASE,
    ):
        video_id = m.group(1)
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        results.append({
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "heading": _nearest_preceding_heading(html, m.start()),
        })

    for m in re.finditer(
        r'https?://(?:www\.)?(?:youtube\.com/watch\?v=([A-Za-z0-9_-]{6,})|youtu\.be/([A-Za-z0-9_-]{6,}))',
        html, re.IGNORECASE,
    ):
        video_id = m.group(1) or m.group(2)
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        results.append({
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "heading": _nearest_preceding_heading(html, m.start()),
        })

    return results

def extract_body_images(html, exclude_url_fragment=None, max_images=4):
    """
    Extracts additional <img> candidates from within the article body
    (see _extract_article_body_html), paired with the nearest preceding
    heading and any alt/caption text. Wikimedia Commons frequently has
    no free-licensed equivalent for things like an official tour poster
    or promotional artwork -- this gives the drafting model a
    second, source-backed option for that case (same copyright tradeoff
    already accepted for the featured image; see the IMAGE SOURCING NOTE
    in main.py). exclude_url_fragment can be used to skip an image
    already used elsewhere (e.g. the featured image), to avoid offering
    it twice. Skips obviously-decorative tiny images (icons, spacers)
    via a minimum width/height check where that's available in the
    markup, and stops once max_images plausible candidates are found.
    """
    body_html = _extract_article_body_html(html)
    results = []
    seen_keys = set()
    exclude_key = _normalize_image_key(exclude_url_fragment) if exclude_url_fragment else None

    for m in re.finditer(r"<img[^>]+>", body_html, re.IGNORECASE):
        if len(results) >= max_images:
            break
        tag = m.group(0)

        src_match = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        if not src_match:
            continue
        src = src_match.group(1)
        src_key = _normalize_image_key(src)
        # Compare by normalized key, not raw URL: the featured image (from
        # og:image) and this same photo's <img src> inside the body are
        # frequently different-sized renditions of one file (see
        # _normalize_image_key), so a plain substring/equality check on the
        # raw URLs was letting the same photo through as a second "body
        # image" candidate -- which is how it ended up duplicated in both
        # the hero featured image and again inline in the post.
        if src_key and src_key in seen_keys:
            continue
        if exclude_key and src_key == exclude_key:
            continue

        width_match = re.search(r'width=["\']?(\d+)', tag, re.IGNORECASE)
        height_match = re.search(r'height=["\']?(\d+)', tag, re.IGNORECASE)
        if width_match and int(width_match.group(1)) < 200:
            continue
        if height_match and int(height_match.group(1)) < 150:
            continue

        alt_match = re.search(r'alt=["\']([^"\']*)["\']', tag, re.IGNORECASE)
        alt_text = alt_match.group(1).strip() if alt_match else ""

        if src_key:
            seen_keys.add(src_key)
        results.append({
            "url": src,
            "alt": alt_text,
            "heading": _nearest_preceding_heading(body_html, m.start()),
        })

    return results

def _nearest_preceding_heading(html, pos):
    """Finds the text of the closest <h2>/<h3> heading before position pos in html."""
    preceding = html[:pos]
    matches = list(re.finditer(r"<h[23][^>]*>(.*?)</h[23]>", preceding, re.IGNORECASE | re.DOTALL))
    if not matches:
        return None
    text = re.sub(r"<[^>]+>", " ", matches[-1].group(1))
    return " ".join(text.split())
