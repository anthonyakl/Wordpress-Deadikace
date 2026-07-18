"""
Finds real, properly-licensed photos of the actual band/artist/album
named in an article. Two strategies, tried in order:

1. Wikipedia page image: resolves the query to a Wikipedia article (e.g.
   "ZZ Top") and takes that page's own main image -- the one Wikipedia's
   editors have already curated as the representative photo for that
   exact subject. This is far more reliable than blind keyword search.
2. Commons keyword search (fallback): only used if step 1 finds nothing.
   Results are checked for actual title relevance to the query before
   being accepted, to avoid the kind of unrelated match (e.g. a scanned
   government report) that plain full-text search can return.

Every accepted image is checked against an allow-list of reusable
licenses (CC0, public domain, or CC-BY variants; never non-commercial-only
licenses) before use, and credited with photographer + license + link.

Google Image Search results are NOT used anywhere in this project: an
image showing up in a Google search is not thereby licensed for reuse,
and scraping it for a commercial blog would be a straightforward
copyright problem regardless of adding a credit line.
"""

import re
import html
import urllib.parse

import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = "DeadikaceAgent/1.0 (https://www.deadikace.com)"

_ALLOWED_PREFIXES = ("CC0", "PUBLIC DOMAIN", "PD", "CC BY", "CC-BY")


def _license_is_allowed(license_short_name):
    if not license_short_name:
        return False
    name = license_short_name.strip().upper()
    if "NC" in name.replace("-", " ").split():
        return False
    return any(name.startswith(prefix) for prefix in _ALLOWED_PREFIXES)


def _strip_html(raw):
    if not raw:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def _query_words(text):
    return {w.lower() for w in re.findall(r"[a-zA-Z']+", text) if len(w) > 2}


def _file_info_from_commons(file_title):
    """Looks up license/artist metadata for a specific File: page on Commons."""
    try:
        resp = requests.get(
            COMMONS_API,
            params={
                "action": "query", "titles": file_title, "prop": "imageinfo",
                "iiprop": "url|extmetadata|size", "iiurlwidth": 1200, "format": "json",
            },
            headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            continue
        imageinfo_list = page.get("imageinfo") or []
        if not imageinfo_list:
            continue
        info = imageinfo_list[0]
        extmeta = info.get("extmetadata", {})
        license_short = extmeta.get("LicenseShortName", {}).get("value", "")
        if not _license_is_allowed(license_short):
            return None

        artist = _strip_html(extmeta.get("Artist", {}).get("value", "")) or "Unknown"
        license_url = extmeta.get("LicenseUrl", {}).get("value", "")
        page_url = "https://commons.wikimedia.org/wiki/" + page.get("title", "").replace(" ", "_")
        return {
            "url": info.get("thumburl") or info.get("url"),
            "artist": artist,
            "license_name": license_short,
            "license_url": license_url,
            "page_url": page_url,
        }
    return None


def _wikipedia_page_image(query):
    """Tier 1: the subject's own Wikipedia page's main image."""
    try:
        resp = requests.get(
            WIKIPEDIA_API,
            params={"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"},
            headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        resp.raise_for_status()
        results = resp.json()
        titles = results[1] if len(results) > 1 else []
        if not titles:
            return None
        title = titles[0]

        summary_resp = requests.get(
            WIKIPEDIA_SUMMARY + urllib.parse.quote(title),
            headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        if summary_resp.status_code != 200:
            return None
        summary = summary_resp.json()
    except (requests.RequestException, ValueError):
        return None

    image_info = summary.get("originalimage") or summary.get("thumbnail")
    if not image_info:
        return None
    image_url = image_info["source"]
    if image_url.lower().endswith(".svg"):
        return None  # skip logos/icons

    # Derive the Commons "File:" title from the image URL so we can look
    # up its real license -- handles both direct and /thumb/ URLs.
    if "/thumb/" in image_url:
        filename = urllib.parse.unquote(image_url.split("/thumb/")[1].split("/")[-2])
    else:
        filename = urllib.parse.unquote(image_url.rsplit("/", 1)[-1])

    return _file_info_from_commons(f"File:{filename}")


def _commons_search(query, min_width=800):
    """Tier 2 fallback: keyword search, filtered for actual title relevance."""
    try:
        resp = requests.get(
            COMMONS_API,
            params={
                "action": "query", "generator": "search", "gsrsearch": query,
                "gsrnamespace": 6, "gsrlimit": 8, "prop": "imageinfo",
                "iiprop": "url|extmetadata|size", "iiurlwidth": 1200, "format": "json",
            },
            headers={"User-Agent": USER_AGENT}, timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[warn] Wikimedia Commons search failed for '{query}': {e}")
        return None

    pages = resp.json().get("query", {}).get("pages", {})
    if not pages:
        return None

    query_kw = _query_words(query)

    for page in sorted(pages.values(), key=lambda p: p.get("index", 999)):
        imageinfo_list = page.get("imageinfo") or []
        if not imageinfo_list:
            continue
        info = imageinfo_list[0]
        if info.get("width", 0) < min_width:
            continue

        extmeta = info.get("extmetadata", {})
        license_short = extmeta.get("LicenseShortName", {}).get("value", "")
        if not _license_is_allowed(license_short):
            continue

        page_title = page.get("title", "")
        description = _strip_html(extmeta.get("ImageDescription", {}).get("value", ""))
        # Relevance sanity check -- reject matches that don't actually
        # share meaningful words with the query (this is what filters out
        # things like an unrelated scanned document matching on noise).
        combined = f"{page_title} {description}".lower()
        hits = sum(1 for w in query_kw if w in combined)
        if query_kw and hits < max(1, len(query_kw) // 2):
            continue

        artist = _strip_html(extmeta.get("Artist", {}).get("value", "")) or "Unknown"
        license_url = extmeta.get("LicenseUrl", {}).get("value", "")
        page_url = "https://commons.wikimedia.org/wiki/" + page_title.replace(" ", "_")
        thumb_url = info.get("thumburl") or info.get("url")
        if not thumb_url:
            continue

        return {
            "url": thumb_url, "artist": artist, "license_name": license_short,
            "license_url": license_url, "page_url": page_url,
        }

    return None


def find_real_photo(query):
    """
    Main entry point. Returns a dict with url/artist/license_name/
    license_url/page_url for the best real, properly-licensed match, or
    None if nothing suitable was found by either strategy.
    """
    photo = _wikipedia_page_image(query)
    if photo:
        return photo
    return _commons_search(query)


def download_image(url):
    """Returns (bytes, content_type)."""
    resp = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    return resp.content, content_type
