"""
Thin WordPress REST API client using an Application Password.
Docs: https://developer.wordpress.org/rest-api/
"""

import re
from urllib.parse import urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from config import (
    WP_BASE_URL, WP_API_BASE_URL, WP_PROXY_TOKEN,
    WP_USERNAME, WP_APP_PASSWORD, POST_STATUS,
    YOAST_TITLE_FIELD, YOAST_META_DESC_FIELD, YOAST_FOCUS_KEYWORD_FIELD,
)

AUTH = HTTPBasicAuth(WP_USERNAME, WP_APP_PASSWORD)
REQUEST_TIMEOUT = 45
_ACTIVE_BASE_URL = WP_API_BASE_URL.rstrip("/")

# Some hosts (Hostinger included) can intermittently drop connections from
# datacenter IPs such as GitHub Actions runners. Keep retries for normal API
# traffic, but the initial connectivity probe also tries the equivalent
# www/non-www hostname so one bad DNS/proxy path cannot kill the whole run.
_retry = Retry(
    total=4, connect=4, read=2,
    backoff_factor=8,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=("GET", "POST"),
)
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Deadikace-WordPress-Agent/1.0 (+https://deadikace.com)",
    "Accept": "application/json, */*;q=0.8",
})
if WP_PROXY_TOKEN:
    _session.headers["X-Deadikace-Proxy-Token"] = WP_PROXY_TOKEN
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.mount("http://", HTTPAdapter(max_retries=_retry))


def _api_root():
    return f"{_ACTIVE_BASE_URL}/wp-json/wp/v2"


def _base_url_candidates():
    """Return configured URL plus the equivalent www/non-www hostname."""
    configured = WP_API_BASE_URL.rstrip("/")
    candidates = [configured]

    # A proxy URL is a distinct service, not an alternate spelling of the
    # public WordPress hostname. Never synthesize www/non-www proxy hosts.
    if configured != WP_BASE_URL.rstrip("/"):
        return candidates

    parsed = urlsplit(configured)
    hostname = parsed.hostname
    if not hostname:
        return candidates

    alternate_host = hostname[4:] if hostname.startswith("www.") else f"www.{hostname}"
    netloc = alternate_host
    if parsed.port:
        netloc += f":{parsed.port}"

    alternate = urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
    if alternate not in candidates:
        candidates.append(alternate)
    return candidates


def check_connectivity():
    """
    Check the REST API before spending LLM quota.

    The probe intentionally uses short attempts and tries both the configured
    hostname and its www/non-www equivalent. If one works, all subsequent
    WordPress API calls use that working base URL for the rest of the run.
    """
    global _ACTIVE_BASE_URL

    errors = []
    for base_url in _base_url_candidates():
        try:
            # Use a fresh session here so the normal long retry policy does not
            # spend several minutes retrying a dead hostname before failover.
            resp = requests.get(
                f"{base_url}/wp-json/",
                headers={
                    "User-Agent": "Deadikace-WordPress-Agent/1.0 (+https://deadikace.com)",
                    "Accept": "application/json, */*;q=0.8",
                    **({"X-Deadikace-Proxy-Token": WP_PROXY_TOKEN} if WP_PROXY_TOKEN else {}),
                },
                timeout=(12, 20),
                allow_redirects=True,
            )
            resp.raise_for_status()

            # Prefer the final hostname after redirects (for example www ->
            # apex) so later API calls avoid an unnecessary proxy hop.
            final = urlsplit(resp.url)
            original = urlsplit(base_url)
            if final.scheme and final.netloc:
                final_path = original.path.rstrip("/")
                _ACTIVE_BASE_URL = urlunsplit(
                    (final.scheme, final.netloc, final_path, "", "")
                ).rstrip("/")
            else:
                _ACTIVE_BASE_URL = base_url

            if _ACTIVE_BASE_URL != WP_API_BASE_URL.rstrip("/"):
                print(
                    f"[info] WordPress connectivity succeeded via {_ACTIVE_BASE_URL}; "
                    "using this hostname for the rest of the run."
                )
            return True, None
        except requests.RequestException as e:
            errors.append(f"{base_url}: {e}")
            print(f"[warn] WordPress connectivity probe failed via {base_url}: {e}")

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
    gives the LLM nothing to catch that with. The excerpt (first
    sentence or two of the actual article) is usually enough to confirm
    or rule out a match even when the headlines don't overlap at all.
    """
    resp = _session.get(
        f"{_api_root()}/posts",
        params={
            "per_page": per_page,
            "_fields": "title,excerpt,link",
            # Include drafts/pending/scheduled posts in the dedup pool, not
            # just published ones -- a topic already drafted (but not yet
            # published) in an earlier run must still be caught as a
            # duplicate, otherwise the same story can get drafted twice.
            "status": "publish,future,draft,pending",
        },
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
        f"{_api_root()}/posts",
        params={"per_page": limit, "_fields": "title,link", "orderby": "date", "order": "desc"},
        auth=AUTH,
        timeout=30,
    )
    resp.raise_for_status()
    return [{"title": p["title"]["rendered"], "link": p["link"]} for p in resp.json()]


def get_or_create_tag(name):
    resp = _session.get(f"{_api_root()}/tags", params={"search": name}, auth=AUTH, timeout=30)
    resp.raise_for_status()
    matches = resp.json()
    if matches:
        return matches[0]["id"]

    create = _session.post(f"{_api_root()}/tags", json={"name": name}, auth=AUTH, timeout=30)
    create.raise_for_status()
    return create.json()["id"]


def get_or_create_category(name):
    if not name or not name.strip():
        raise ValueError("get_or_create_category() called with an empty category name -- "
                          "refusing to guess a category rather than risk picking the wrong one.")

    resp = _session.get(f"{_api_root()}/categories", params={"search": name}, auth=AUTH, timeout=30)
    resp.raise_for_status()
    matches = resp.json()

    # Only accept an EXACT case-insensitive name match. WP's search param
    # does a loose partial match, so falling back to "the first result"
    # when there's no exact match risks silently filing posts under an
    # unrelated category -- better to create the intended one instead.
    for m in matches:
        if m["name"].strip().lower() == name.strip().lower():
            return m["id"]

    create = _session.post(f"{_api_root()}/categories", json={"name": name}, auth=AUTH, timeout=30)
    create.raise_for_status()
    return create.json()["id"]


def search_related_posts(keyword, limit=3):
    """Used for internal linking suggestions."""
    resp = _session.get(
        f"{_api_root()}/posts",
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
        f"{_api_root()}/media",
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
                f"{_api_root()}/media/{media['id']}",
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

    resp = _session.post(f"{_api_root()}/posts", json=payload, auth=AUTH, timeout=60)
    resp.raise_for_status()
    return resp.json()
