"""
Thin WordPress REST API client using an Application Password.
Docs: https://developer.wordpress.org/rest-api/
"""

import re

import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from config import (
    WP_BASE_URL, WP_USERNAME, WP_APP_PASSWORD, POST_STATUS,
    YOAST_TITLE_FIELD, YOAST_META_DESC_FIELD, YOAST_FOCUS_KEYWORD_FIELD,
)

API_ROOT = f"{WP_BASE_URL.rstrip('/')}/wp-json/wp/v2"
AUTH = HTTPBasicAuth(WP_USERNAME, WP_APP_PASSWORD)
REQUEST_TIMEOUT = 45

# Some hosts (Hostinger included) have bot/DDoS protection layers that
# occasionally drop a connection attempt from a datacenter IP like GitHub
# Actions' without it being a persistent block -- retrying with backoff
# lets a transient blip resolve itself instead of failing the whole run.
_retry = Retry(
    total=4, connect=4, read=2,
    backoff_factor=8,  # 8s, 16s, 32s, 64s between attempts
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=("GET", "POST"),
)
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.mount("http://", HTTPAdapter(max_retries=_retry))


def check_connectivity():
    """
    Quick reachability check against the WP REST API root, meant to be
    called first thing in a run so a persistent connectivity problem
    (e.g. host-level bot protection blocking GitHub Actions' IPs) fails
    fast with a clear message, rather than after burning LLM API quota on
    the discovery/relevance steps first.
    """
    try:
        resp = _session.get(f"{WP_BASE_URL.rstrip('/')}/wp-json/", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return True, None
    except requests.RequestException as e:
        return False, str(e)


def get_recent_post_titles(per_page=50):
    """
    Fetch recent post titles to avoid writing duplicate articles.
    Kept for backwards compatibility -- prefer get_recent_posts_for_dedup()
    for the duplicate-checking flow, which also returns excerpts so the
    LLM-based check has more than a bare headline to compare against.
    """
    return [p["title"] for p in get_recent_posts_for_dedup(per_page)]

def get_recent_posts_for_dedup(per_page=50):
    """
    Fetch recent posts' titles AND excerpts for duplicate-checking.
    Comparing candidate topics against bare titles alone is unreliable --
    two outlets (or two of our own runs) can cover the exact same story
    with completely different headlines, and a title-only comparison
    gives the LLM nothing to catch that with. The excerpt (first
    sentence or two of the actual article) is usually enough to confirm
    or rule out a match even when the headlines don't overlap at all.
    """
    resp = _session.get(
        f"{API_ROOT}/posts",
        params={"per_page": per_page, "_fields": "title,excerpt,link"},
        auth=AUTH,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    results = []
    for p in resp.json():
        title = p["title"]["rendered"]
        excerpt = re.sub(r"<[^>]+>", " ", p.get("excerpt", {}).get("rendered", ""))
        excerpt = " ".join(excerpt.split())[:200]
        results.append({"title": title, "excerpt": excerpt})
    return results


def get_latest_posts(limit=5):
    """Used to build the 'Latest Posts' block appended to new articles."""
    resp = _session.get(
        f"{API_ROOT}/posts",
        params={"per_page": limit, "_fields": "title,link", "orderby": "date", "order": "desc"},
        auth=AUTH,
        timeout=30,
    )
    resp.raise_for_status()
    return [{"title": p["title"]["rendered"], "link": p["link"]} for p in resp.json()]


def get_or_create_tag(name):
    resp = _session.get(f"{API_ROOT}/tags", params={"search": name}, auth=AUTH, timeout=30)
    resp.raise_for_status()
    matches = resp.json()
    if matches:
        return matches[0]["id"]

    create = _session.post(f"{API_ROOT}/tags", json={"name": name}, auth=AUTH, timeout=30)
    create.raise_for_status()
    return create.json()["id"]


def get_or_create_category(name):
    if not name or not name.strip():
        raise ValueError("get_or_create_category() called with an empty category name -- "
                          "refusing to guess a category rather than risk picking the wrong one.")

    resp = _session.get(f"{API_ROOT}/categories", params={"search": name}, auth=AUTH, timeout=30)
    resp.raise_for_status()
    matches = resp.json()

    # Only accept an EXACT case-insensitive name match. WP's search param
    # does a loose partial match, so falling back to "the first result"
    # when there's no exact match risks silently filing posts under an
    # unrelated category -- better to create the intended one instead.
    for m in matches:
        if m["name"].strip().lower() == name.strip().lower():
            return m["id"]

    create = _session.post(f"{API_ROOT}/categories", json={"name": name}, auth=AUTH, timeout=30)
    create.raise_for_status()
    return create.json()["id"]


def search_related_posts(keyword, limit=3):
    """Used for internal linking suggestions."""
    resp = _session.get(
        f"{API_ROOT}/posts",
        params={"search": keyword, "per_page": limit, "_fields": "title,link"},
        auth=AUTH,
        timeout=30,
    )
    resp.raise_for_status()
    return [{"title": p["title"]["rendered"], "link": p["link"]} for p in resp.json()]


def upload_media(image_bytes, filename, alt_text="", content_type="image/jpeg"):
    """
    Uploads an image to the WordPress media library.
    Returns dict with id and source_url, or None on failure.
    """
    ext_by_type = {
        "image/jpeg": ".jpg", "image/jpg": ".jpg",
        "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif",
    }
    ext = ext_by_type.get(content_type.lower(), ".jpg")

    base_name = re.sub(r"[^a-zA-Z0-9._-]", "-", filename)
    base_name = re.sub(r"\.(jpg|jpeg|png|webp|gif)$", "", base_name, flags=re.IGNORECASE)
    safe_filename = base_name + ext

    headers = {
        "Content-Disposition": f'attachment; filename="{safe_filename}"',
        "Content-Type": content_type,
    }

    resp = _session.post(
        f"{API_ROOT}/media",
        headers=headers,
        data=image_bytes,
        auth=AUTH,
        timeout=60,
    )
    resp.raise_for_status()
    media = resp.json()

    if alt_text:
        # Alt text has to be set via a follow-up PATCH -- the upload
        # endpoint doesn't accept it directly.
        try:
            _session.post(
                f"{API_ROOT}/media/{media['id']}",
                json={"alt_text": alt_text[:250]},
                auth=AUTH,
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"[warn] Could not set alt text for media {media['id']}: {e}")

    return {"id": media["id"], "source_url": media.get("source_url", "")}


def create_post(article, category_id=None, featured_media_id=None):
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

    if category_id:
        payload["categories"] = [category_id]
    if featured_media_id:
        payload["featured_media"] = featured_media_id

    resp = _session.post(f"{API_ROOT}/posts", json=payload, auth=AUTH, timeout=60)
    resp.raise_for_status()
    return resp.json()
