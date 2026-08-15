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

    Commons' own search ranking isn't always reliable for this use case
    -- e.g. a query like "Bruce Springsteen performing live" once matched
    a museum photo of one of his guitars on display, just because the
    file's title/description happened to mention his name and "live"
    (as in a "live music" exhibit) without the image actually showing a
    performance. To reduce this, candidates are re-ranked by how many
    query words actually appear in the candidate's own title, rather
    than trusting Commons' result order at face value. Candidates with
    an extreme aspect ratio (very tall/narrow, like a vertical museum
    display shot) are also deprioritized, since they render awkwardly as
    an in-article illustrative image regardless of subject match.
    """
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": 10,
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

    query_words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}

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

        title_words = set(re.findall(r"[a-z0-9]+", page.get("title", "").lower()))
        relevance = len(query_words & title_words)

        aspect_ratio = max(width, height) / max(1, min(width, height))
        extreme_shape = aspect_ratio > 1.8

        candidates.append({
            "page": page,
            "info": info,
            "relevance": relevance,
            "extreme_shape": extreme_shape,
        })

    if not candidates:
        return None

    # Require a MEANINGFUL overlap between the query and the candidate's
    # title, not just any single shared word. A single-word overlap (e.g.
    # a shared surname) is not enough to confirm the right subject -- it
    # previously let a query like "Shane Hawkins Chevy Metal drummer"
    # match an unrelated "Taylor Hawkins memorial" photo, since both
    # happen to contain the word "Hawkins". Requiring at least two
    # overlapping words (or a full match for genuinely one-word queries)
    # makes it much harder for an incidental word overlap alone to pass
    # as a real subject match. Better to skip the image entirely than
    # risk an unrelated one (e.g. a museum photo of an artist's guitar
    # when the query asked for a performance shot, or a memorial photo
    # of a different family member entirely). Among relevant candidates,
    # prefer a normal (non-extreme) aspect ratio.
    min_relevance = min(2, len(query_words))
    relevant = [c for c in candidates if c["relevance"] >= min_relevance]
    if not relevant:
        return None

    # Hard-exclude extreme-shape (unusually tall/narrow) candidates
    # entirely, rather than just deprioritizing them. An earlier version
    # of this still picked an extreme-shape image if it was the only
    # relevant match, relying on a display-layer CSS crop to make it look
    # reasonable -- but cropping a bad image often hides the actual
    # subject (e.g. a tall vertical photo cropped down to a sliver of
    # sky). Better to have no illustrative image for this query than a
    # badly-shaped one.
    non_extreme = [c for c in relevant if not c["extreme_shape"]]
    if not non_extreme:
        return None

    non_extreme.sort(key=lambda c: c["relevance"], reverse=True)
    best = non_extreme[0]

    page, info = best["page"], best["info"]
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
