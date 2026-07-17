"""
Finds and downloads royalty-free, commercially-usable stock photos via the
Pexels API (https://www.pexels.com/api/) based on search queries the LLM
generates for each article. Pexels photos are free to use commercially with
no attribution legally required, but we credit the photographer in a
caption anyway as good practice.
"""

import requests

from config import PEXELS_API_KEY

SEARCH_URL = "https://api.pexels.com/v1/search"


def search_image(query):
    """Returns a dict with url/photographer/alt for the top match, or None."""
    if not PEXELS_API_KEY:
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
    }


def download_image_bytes(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content
