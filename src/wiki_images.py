"""
Searches Wikimedia Commons for real, properly-licensed photos (bands,
artists, concerts, gear) instead of relying solely on generic stock
images. Commons hosts a large volume of photography that photographers
have deliberately released under Creative Commons or public domain
licenses -- meaning it's actually legal to reuse (including commercially),
as long as it's credited, which this module does automatically.

Google Image Search results are NOT used anywhere in this project: images
that show up in a Google search are not thereby licensed for reuse, and
scraping them for a commercial blog would be a straightforward copyright
problem regardless of adding a credit line. Commons is different because
every file carries explicit, checkable license metadata, and this module
only accepts files under licenses that permit reuse.
"""

import re
import html

import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "DeadikaceAgent/1.0 (https://www.deadikace.com)"

# Only accept clearly free/reusable licenses. Explicitly excludes
# non-commercial-only (NC) licenses, since this is a commercial site.
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


def search_commons_image(query, min_width=800):
    """
    Returns a dict with url/artist/license_name/license_url/page_url for
    the best properly-licensed match, or None if nothing suitable found.
    """
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,  # File namespace
        "gsrlimit": 8,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": 1200,
        "format": "json",
    }

    try:
        resp = requests.get(
            COMMONS_API, params=params, timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[warn] Wikimedia Commons search failed for '{query}': {e}")
        return None

    pages = resp.json().get("query", {}).get("pages", {})
    if not pages:
        return None

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

        artist = _strip_html(extmeta.get("Artist", {}).get("value", "")) or "Unknown"
        license_url = extmeta.get("LicenseUrl", {}).get("value", "")
        page_title = page.get("title", "")
        page_url = "https://commons.wikimedia.org/wiki/" + page_title.replace(" ", "_")
        thumb_url = info.get("thumburl") or info.get("url")

        if not thumb_url:
            continue

        return {
            "url": thumb_url,
            "artist": artist,
            "license_name": license_short,
            "license_url": license_url,
            "page_url": page_url,
        }

    return None


def download_image(url):
    """Returns (bytes, content_type)."""
    resp = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    return resp.content, content_type
