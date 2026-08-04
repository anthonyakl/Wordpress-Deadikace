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

import re
import requests
import trafilatura

USER_AGENT = "DeadikaceAgent/1.0 (https://www.deadikace.com)"

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
    caption_text = _extract_image_caption(html)
    return image_url, caption_text

def _extract_image_caption(html):
    """
    Best-effort extraction of a SHORT photo credit line near the
    article's lead image, e.g. "Credit: Getty Images" rather than a full
    descriptive caption. Earlier versions returned the whole figcaption
    text (description + credit combined), but that's too long to display
    as a corner overlay on the frontend without overlapping the title/
    date on mobile. Looks for the first <figcaption>, then tries to pull
    just the trailing credit/agency portion out of it; falls back to a
    "Photo:" / "Credit:" style text pattern if no figcaption exists.
    Returns None if nothing plausible is found -- callers should fall
    back to a generic credit line in that case, not leave the field
    empty in a way that looks broken.
    """
    figcaption_match = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", html, re.IGNORECASE | re.DOTALL)
    full_caption = None
    if figcaption_match:
        text = re.sub(r"<[^>]+>", " ", figcaption_match.group(1))
        text = " ".join(text.split())
        if text and len(text) < 300:
            full_caption = text

    if not full_caption:
        credit_match = re.search(r"((?:Photo|Image|Credit)\s*:\s*[^<\n]{3,150})", html, re.IGNORECASE)
        if credit_match:
            full_caption = " ".join(credit_match.group(1).split())

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
    m = re.search(r"\bvia\s+([A-Z][\w.&'-]{2,40}(?:\s+[A-Z][\w.&'-]{1,40}){0,3})\s*\.?$", full_caption)
    if m:
        return f"Credit: {m.group(1).strip().rstrip('.')}"

    m = re.match(r"^(?:Photo|Image|Credit)\s*:\s*(.{2,60}?)(?:\.|,|$)", full_caption, re.IGNORECASE)
    if m:
        return f"Credit: {m.group(1).strip()}"

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
        else:
            item["full_text"] = None
            item["image_url"] = None
            item["image_caption"] = None
            item["video_embeds"] = []
        if item["full_text"]:
            fetched_count += 1
    return topic

def extract_video_embeds(html):
    """
    Extracts YouTube video links embedded in the source article, along
    with the nearest preceding <h2>/<h3> heading text (used to match each
    video to the right section when the drafting model restructures the
    article under its own headings). Returns a list of
    {"url": ..., "heading": ...} dicts, deduplicated by video ID. Only
    YouTube is handled for now, since it's by far the most common
    platform for the kind of archival live-performance clips this blog
    covers.
    """
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

def _nearest_preceding_heading(html, pos):
    """Finds the text of the closest <h2>/<h3> heading before position pos in html."""
    preceding = html[:pos]
    matches = list(re.finditer(r"<h[23][^>]*>(.*?)</h[23]>", preceding, re.IGNORECASE | re.DOTALL))
    if not matches:
        return None
    text = re.sub(r"<[^>]+>", " ", matches[-1].group(1))
    return " ".join(text.split())
