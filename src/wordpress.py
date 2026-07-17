"""
Thin WordPress REST API client using an Application Password.
Docs: https://developer.wordpress.org/rest-api/
"""

import requests
from requests.auth import HTTPBasicAuth

from config import (
    WP_BASE_URL, WP_USERNAME, WP_APP_PASSWORD, POST_STATUS,
    YOAST_TITLE_FIELD, YOAST_META_DESC_FIELD, YOAST_FOCUS_KEYWORD_FIELD,
)

API_ROOT = f"{WP_BASE_URL.rstrip('/')}/wp-json/wp/v2"
AUTH = HTTPBasicAuth(WP_USERNAME, WP_APP_PASSWORD)


def get_recent_post_titles(per_page=50):
    """Fetch recent post titles to avoid writing duplicate articles."""
    resp = requests.get(
        f"{API_ROOT}/posts",
        params={"per_page": per_page, "_fields": "title,link"},
        auth=AUTH,
        timeout=30,
    )
    resp.raise_for_status()
    return [p["title"]["rendered"] for p in resp.json()]


def get_or_create_tag(name):
    resp = requests.get(f"{API_ROOT}/tags", params={"search": name}, auth=AUTH, timeout=30)
    resp.raise_for_status()
    matches = resp.json()
    if matches:
        return matches[0]["id"]

    create = requests.post(f"{API_ROOT}/tags", json={"name": name}, auth=AUTH, timeout=30)
    create.raise_for_status()
    return create.json()["id"]


def search_related_posts(keyword, limit=3):
    """Used for internal linking suggestions."""
    resp = requests.get(
        f"{API_ROOT}/posts",
        params={"search": keyword, "per_page": limit, "_fields": "title,link"},
        auth=AUTH,
        timeout=30,
    )
    resp.raise_for_status()
    return [{"title": p["title"]["rendered"], "link": p["link"]} for p in resp.json()]


def create_post(article):
    """
    article: dict with keys:
      title, content_html, excerpt, tags (list[str]),
      seo_title, meta_description, focus_keyword
    """
    tag_ids = [get_or_create_tag(t) for t in article.get("tags", [])]

    payload = {
        "title": article["title"],
        "content": article["content_html"],
        "excerpt": article.get("excerpt", ""),
        "status": POST_STATUS,  # "publish" or "draft"
        "tags": tag_ids,
        "meta": {
            YOAST_TITLE_FIELD: article.get("seo_title", article["title"]),
            YOAST_META_DESC_FIELD: article.get("meta_description", ""),
            YOAST_FOCUS_KEYWORD_FIELD: article.get("focus_keyword", ""),
        },
    }

    resp = requests.post(f"{API_ROOT}/posts", json=payload, auth=AUTH, timeout=60)
    resp.raise_for_status()
    return resp.json()
