"""
Fallback image source: Pexels (https://www.pexels.com/api/), royalty-free
stock photos, used only when Wikimedia Commons (wiki_images.py) doesn't
have a real, properly-licensed photo of the actual subject. See
wiki_images.py for why Google Image Search results aren't used here.
"""

import requests

from config import PEXELS_API_KEY

SEARCH_URL = "https://api.pexels.com/v1/search"

_warned_missing_key = False


def search_image(query):
    """Returns a dict with url/photographer/alt for the top match, or None."""
    global _warned_missing_key
    if not PEXELS_API_KEY:
        if not _warned_missing_key:
            print("[warn] PEXELS_API_KEY is not set -- stock-photo fallback images "
                  "will be skipped for this entire run. Add the PEXELS_API_KEY "
                  "GitHub Secret to enable images (free key at pexels.com/api).")
            _warned_missing_key = True
        return None

    try:
        resp = requests.get(
            SEARCH_URL,
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[warn] Pexels search failed for '{query}': {e}")
        return None

    photos = resp.json().get("photos", [])
    if not photos:
        return None

    photo = photos[0]
    return {
        "url": photo["src"]["large"],
        "photographer": photo.get("photographer", "Pexels"),
        "photographer_url": photo.get("photographer_url", "https://www.pexels.com"),
        "alt": photo.get("alt") or query,
        "width": photo.get("width"),
        "height": photo.get("height"),
    }


def download_image(url):
    """Returns (bytes, content_type)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    return resp.content, content_type
