"""
Searches Wikimedia Commons for copyright-free/CC-licensed images to
illustrate specific things mentioned in an article (e.g. a named album,
a named venue, a specific event photo). Used only for ADDITIONAL
illustrative images beyond the featured image -- see the IMAGE SOURCING
NOTE in article_fetch.py for the featured-image approach, which is
different (reuses the source outlet's own photo, not from Wikimedia).

Unlike a previous version of this pipeline that searched Wikimedia by a
bare band/artist name and sometimes matched an unrelated same-named
subject (e.g. an actual eagle for the band Eagles), search queries here
are expected to be SPECIFIC, multi-word phrases naming exactly what
should appear in the image (e.g. "Similitude of a Dream album cover Neal
Morse Band", not just "Neal Morse Band") -- see the illustrative-image
query-writing rule in draft.py's DRAFT_SYSTEM_PROMPT.
"""

import re
import requests

USER_AGENT = "DeadikaceAgent/1.0 (https://www.deadikace.com)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def search_commons_image(query):
    """
    Searches Wikimedia Commons for an image matching the query. Returns a
    dict with url, credit (artist + license text), and title, or None if
    nothing suitable was found.
    """
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": 5,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "format": "json",
    }
    try:
        resp = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[warn] Wikimedia Commons search failed for {query!r}: {e}")
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    candidates = []
    for page in pages.values():
        imageinfo = page.get("imageinfo")
        if not imageinfo:
            continue
        info = imageinfo[0]
        width = info.get("width", 0)
        height = info.get("height", 0)
        if width < 300 or height < 200:
            continue
        candidates.append((page, info))

    if not candidates:
        return None

    page, info = candidates[0]
    extmeta = info.get("extmetadata", {})
    artist = _clean_html(extmeta.get("Artist", {}).get("value", ""))
    license_name = extmeta.get("LicenseShortName", {}).get("value", "")
    credit_parts = [p for p in [artist, license_name] if p]
    credit = ", ".join(credit_parts) if credit_parts else "via Wikimedia Commons"
    if "Wikimedia Commons" not in credit:
        credit += ", via Wikimedia Commons"

    return {
        "url": info["url"],
        "credit": credit,
        "title": page.get("title", "").replace("File:", ""),
    }


def _clean_html(text):
    """Strip HTML tags from Commons' extmetadata fields (often contain <a> links)."""
    text = re.sub(r"<[^>]+>", "", text or "")
    return " ".join(text.split())


def download_commons_image(image_url, max_bytes=8_000_000):
    """Downloads a Commons image and returns (bytes, content_type), or (None, None) on failure."""
    try:
        resp = requests.get(image_url, headers={"User-Agent": USER_AGENT}, timeout=20, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith("image/"):
            return None, None
        data = resp.raw.read(max_bytes + 1, decode_content=True)
        if len(data) > max_bytes:
            print(f"[warn] Commons image at {image_url} exceeds {max_bytes} bytes; skipping.")
            return None, None
        return data, content_type
    except requests.RequestException as e:
        print(f"[warn] Failed to download Commons image {image_url}: {e}")
        return None, None

