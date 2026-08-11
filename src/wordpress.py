"""
Thin WordPress REST API client using an Application Password.
Docs: https://developer.wordpress.org/rest-api/
"""

import re
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from config import (
    WP_BASE_URL, WP_USERNAME, WP_APP_PASSWORD, POST_STATUS,
    YOAST_TITLE_FIELD, YOAST_META_DESC_FIELD, YOAST_FOCUS_KEYWORD_FIELD,
)

# Hostinger/CDN setups can expose the public site through both the apex
# domain and www, while a WAF/proxy may treat traffic to one hostname
# differently from the other. Keep the configured URL first, then try the
# equivalent hostname as a safe fallback. This does not bypass authentication
# or any firewall; it simply avoids failing because the configured alias is
# the one being dropped.
def _candidate_base_urls(configured_url):
    configured = configured_url.rstrip("/")
    parsed = urlparse(configured)
    host = (parsed.hostname or "").lower()
    scheme = parsed.scheme or "https"

    candidates = [configured]
    if host.startswith("www."):
        alternate_host = host[4:]
    else:
        alternate_host = f"www.{host}" if host else ""

    if alternate_host:
        alternate = f"{scheme}://{alternate_host}"
        if alternate not in candidates:
            candidates.append(alternate)

    return candidates

ACTIVE_BASE_URL = WP_BASE_URL.rstrip("/")
API_ROOT = f"{ACTIVE_BASE_URL}/wp-json/wp/v2"
AUTH = HTTPBasicAuth(WP_USERNAME, WP_APP_PASSWORD)

# Keep ordinary API requests reasonably quick. The old connectivity check used
# the same long-retry session as write requests, which could turn one blocked
# host into a 5+ minute failure before the agent even started. Connectivity is
# now tested separately and briefly; normal requests retain modest retries.
REQUEST_TIMEOUT = 30
CONNECTIVITY_TIMEOUT = 8
_retry = Retry(
    total=2, connect=2, read=1,
    backoff_factor=2,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=("GET", "POST"),
)
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.mount("http://", HTTPAdapter(max_retries=_retry))


def _activate_base_url(base_url):
    global ACTIVE_BASE_URL, API_ROOT
    ACTIVE_BASE_URL = base_url.rstrip("/")
    API_ROOT = f"{ACTIVE_BASE_URL}/wp-json/wp/v2"


def check_connectivity():
    """
    Verify the WordPress REST API is reachable before spending LLM quota.

    First tries the configured hostname, then the equivalent www/apex alias.
    The probe uses a no-retry request so a blocked alias fails quickly and the
    fallback can be tested. The selected working base URL is then used by all
    subsequent WordPress API calls in this process.
    """
    errors = []
    for base_url in _candidate_base_urls(WP_BASE_URL):
        try:
            resp = requests.get(
                f"{base_url}/wp-json/",
                timeout=CONNECTIVITY_TIMEOUT,
                headers={"User-Agent": "DeadikacePublisher/1.0"},
            )
            resp.raise_for_status()
            _activate_base_url(base_url)
            print(f"[info] WordPress REST API reachable via {ACTIVE_BASE_URL}")
            return True, None
        except requests.RequestException as e:
            errors.append(f"{base_url}: {e}")

    return False, " | ".join(errors)


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
    gives the LLM nothing to catch that with. The excerpt (first sentence
    or two of the actual article) is usually enough to confirm or rule out
    a match even when the headlines don't overlap at all.
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
        raise ValueError("get_or_create_category() called with an empty category name -- refusing to guess a category rather than risk picking the wrong one.")

    resp = _session.get(f"{API_ROOT}/categories", params={"search": name}, auth=AUTH, timeout=30)
    resp.raise_for_status()
    matches = resp.json()

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
    """Uploads an image to the WordPress media library."""
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
        "status": POST_STATUS,
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
