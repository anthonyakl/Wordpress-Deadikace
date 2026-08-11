"""Deadikace editorial safety/idempotency layer.

This module is auto-loaded because src/ is on PYTHONPATH. It wraps the existing
pipeline without replacing the established article generation logic.
"""
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone, timedelta

from requests.auth import HTTPBasicAuth

import config
import draft
import wordpress

API_ROOT = f"{config.WP_BASE_URL.rstrip('/')}/wp-json/wp/v2"
AUTH = HTTPBasicAuth(config.WP_USERNAME, config.WP_APP_PASSWORD)


# Gemini's free tier used by the current configuration is limited to 15
# generate_content requests/minute. The previous safety layer added multiple
# extra LLM calls per candidate and immediately exhausted that quota. Keep all
# LLM calls in this process below the provider's limit instead of hammering it.
_LLM_LOCK = threading.Lock()
_LAST_LLM_CALL = 0.0
_LLM_MIN_INTERVAL = 4.25


def _rate_limit_llm_call():
    global _LAST_LLM_CALL
    with _LLM_LOCK:
        now = time.monotonic()
        wait = _LLM_MIN_INTERVAL - (now - _LAST_LLM_CALL)
        if wait > 0:
            time.sleep(wait)
        _LAST_LLM_CALL = time.monotonic()


_original_call_llm = draft._call_llm
_original_call_llm_with_search = draft._call_llm_with_search


def _paced_call_llm(*args, **kwargs):
    _rate_limit_llm_call()
    return _original_call_llm(*args, **kwargs)


def _paced_call_llm_with_search(*args, **kwargs):
    _rate_limit_llm_call()
    return _original_call_llm_with_search(*args, **kwargs)


draft._call_llm = _paced_call_llm
draft._call_llm_with_search = _paced_call_llm_with_search


def _call_editor(system, prompt, max_tokens=1200):
    raw = draft._call_llm(system, prompt, max_tokens=max_tokens)
    return draft._clean_json_text(raw)


GENRE_SYSTEM = """You are the final editorial genre gate for Deadikace.
Deadikace covers rock and its rock-family variants: rock and roll, classic rock,
hard rock, heavy rock, blues rock, folk rock, progressive rock, psychedelic rock,
southern rock, garage rock, alternative/indie rock, punk rock, glam rock, roots rock,
country rock, funk rock, art rock, grunge, post-punk and post-rock.

Reject a story when its PRIMARY subject is a metal genre, metal scene, or metal-only
artist/release/tour: heavy metal, death metal, black metal, doom, sludge, thrash,
power metal, symphonic metal, progressive metal, metalcore, deathcore, djent,
nu-metal, industrial/groove/technical/melodic death metal, etc.

Do not reject a genuinely rock-focused story merely because a rock/metal crossover
artist is involved. Judge the story's primary subject, not one keyword. A mainstream
rock artist discussing a rock career, a classic-rock band announcement, or a
rock-oriented collaboration can pass even if the artist has metal history.

Also reject pop, rap/hip-hop, K-pop, R&B, EDM/dance, country-only, and unrelated
celebrity stories. When uncertain between rock and metal, reject.

Return ONLY a JSON array of the candidate numbers that PASS. No explanation."""


def _genre_gate(topics):
    if not topics:
        return topics
    lines = []
    for i, t in enumerate(topics, 1):
        item = t["items"][0]
        lines.append(f"{i}. {item['title']} -- {item.get('summary','')[:500]}")
    try:
        raw = _call_editor(
            GENRE_SYSTEM,
            "Candidates:\n" + "\n".join(lines) + "\n\nReturn passing numbers.",
            1200,
        )
        keep = {int(x) - 1 for x in json.loads(raw)}
    except Exception as exc:
        print(f"[fatal] Genre gate failed; rejecting all candidates rather than allowing off-topic content: {exc}")
        return []
    return [t for i, t in enumerate(topics) if i in keep]


draft.filter_rock_relevant_topics = _genre_gate


STORY_SYSTEM = """You are Deadikace's strict story-identity editor.
Determine whether each candidate describes the SAME underlying news event/story as
any existing Deadikace post, including drafts. Different headlines, different
outlets, and different wording do not make a new story.

Compare the concrete event: principal person/band, what happened, specific release,
announcement, interview, tour/date, incident, quote or development. Two stories about
the same artist are NOT automatically duplicates: they are duplicates only when the
underlying event/development is the same.

If uncertain, mark it duplicate. A missed story is preferable to two versions of the
same story. Return ONLY a JSON array of duplicate candidate numbers."""


def _wp_recent_all(limit=150):
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.DEDUP_LOOKBACK_DAYS)
    out = []
    for status in ("publish", "draft", "pending", "future", "private"):
        try:
            r = wordpress._session.get(
                f"{API_ROOT}/posts",
                params={
                    "per_page": min(limit, 100),
                    "status": status,
                    "orderby": "date",
                    "order": "desc",
                    "_fields": "id,date,status,title,excerpt,content,link",
                },
                auth=AUTH,
                timeout=wordpress.REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            for p in r.json():
                date_raw = p.get("date") or ""
                try:
                    dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)
                if dt < cutoff:
                    continue
                title = p.get("title", {}).get("rendered", "")
                excerpt = re.sub(r"<[^>]+>", " ", p.get("excerpt", {}).get("rendered", ""))
                content = p.get("content", {}).get("rendered", "")
                marker = re.search(r"<!-- deadikace-story:([a-f0-9]{12,40}) -->", content)
                out.append({
                    "id": p.get("id"),
                    "title": title,
                    "excerpt": " ".join(excerpt.split())[:500],
                    "story_id": marker.group(1) if marker else None,
                    "status": status,
                })
        except Exception as exc:
            print(f"[warn] Could not fetch WordPress {status} posts for dedup: {exc}")
    seen = set()
    unique = []
    for p in sorted(out, key=lambda x: x.get("id") or 0, reverse=True):
        if p.get("id") in seen:
            continue
        seen.add(p.get("id"))
        unique.append(p)
    return unique[:config.DEDUP_MAX_POSTS]


def _candidate_text(t):
    return " / ".join(x["title"] for x in t["items"][:4]) + " -- " + " ".join(
        (x.get("summary") or "") for x in t["items"][:2]
    )[:800]


def _semantic_dedupe(topics, existing):
    if not topics or not existing:
        return topics
    lines = "\n".join(f"{i}. {_candidate_text(t)}" for i, t in enumerate(topics, 1))
    existing_lines = "\n".join(f"- {p['title']} -- {p['excerpt']}" for p in existing)
    try:
        raw = _call_editor(
            STORY_SYSTEM,
            f"Candidates:\n{lines}\n\nExisting Deadikace posts/drafts:\n{existing_lines}\n\nReturn duplicates.",
            1500,
        )
        dup = {int(x) - 1 for x in json.loads(raw)}
    except Exception as exc:
        print(f"[fatal] Persistent duplicate check failed; rejecting all candidates for safety: {exc}")
        return []
    return [t for i, t in enumerate(topics) if i not in dup]


def _dedupe_with_existing(topics, existing_posts):
    return _semantic_dedupe(topics, _wp_recent_all())


draft.filter_duplicate_topics = _dedupe_with_existing


def _within_batch(topics):
    if len(topics) < 2:
        return topics
    peer_lines = "\n".join(f"{i}. {_candidate_text(t)}" for i, t in enumerate(topics, 1))
    system = STORY_SYSTEM + "\nHere the comparison set is the other candidates in this same run. Return numbers to remove, keeping the most detailed candidate."
    try:
        raw = _call_editor(system, f"Candidates:\n{peer_lines}\n\nReturn duplicate candidate numbers.", 1200)
        remove = {int(x) - 1 for x in json.loads(raw)}
        return [t for i, t in enumerate(topics) if i not in remove]
    except Exception as exc:
        print(f"[fatal] Within-run duplicate check failed; rejecting the batch for safety: {exc}")
        return []


draft.dedupe_topics_within_batch = _within_batch


# Do NOT add a second research call here. The existing research_additional_context()
# already performs the intended independent research pass. The previous safety layer
# called it and then performed another live-search LLM pass for every candidate,
# doubling API usage and exhausting Gemini's free-tier quota.


draft.DRAFT_SYSTEM_PROMPT += """

EDITORIAL CONTEXT AND CONNECTIONS:
This is still a news report, not an opinion column. However, when the evidence makes
it useful, add editorial context and concise analysis that helps the reader understand
why the development matters. Make specific, relevant connections to related stories,
previous releases, career developments, band relationships, tours, albums, songs,
or historical events when those connections are genuinely relevant and grounded in
the supplied research. Do not manufacture a connection merely to make the article
longer.

A safe editorial observation may occasionally be included when it follows directly
from the documented facts and is not controversial, speculative, or presented as a
quote or consensus. Never invent a reader reaction, public consensus, personal
opinion, or unsupported judgment.

ORIGINALITY GATE:
The finished article must provide value beyond the immediate competitor report. Do
not simply reorder or paraphrase source facts. Use the independent research to add
context, connections, chronology, implications, or documented background. If the
available material does not support meaningful added value, do not pad the article.
"""
draft.RELEVANCE_SYSTEM_PROMPT += "\n\nFINAL RULE: if the primary story is metal-scene content rather than rock, exclude it. When uncertain, exclude it."


_orig_create = wordpress.create_post


def _story_id_for_article(article):
    text = re.sub(
        r"\s+",
        " ",
        (article.get("title", "") + " " + re.sub(r"<[^>]+>", " ", article.get("content_html", ""))).lower(),
    ).strip()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def _final_create(article, category_id=None, featured_media_id=None):
    existing = _wp_recent_all()
    if existing:
        probe = [{
            "title": article.get("title", ""),
            "excerpt": re.sub(r"<[^>]+>", " ", article.get("content_html", ""))[:1200],
        }]
        if _semantic_dedupe(probe, existing) == []:
            raise RuntimeError(f"Final idempotency guard rejected likely duplicate: {article.get('title','')}")

    result = _orig_create(article, category_id=category_id, featured_media_id=featured_media_id)
    try:
        marker = f"<!-- deadikace-story:{_story_id_for_article(article)} -->"
        post_id = result.get("id")
        if post_id:
            current = article.get("content_html", "")
            wordpress._session.post(
                f"{API_ROOT}/posts/{post_id}",
                json={"content": current + "\n" + marker},
                auth=AUTH,
                timeout=wordpress.REQUEST_TIMEOUT,
            )
    except Exception as exc:
        print(f"[warn] Could not persist story marker after creation: {exc}")
    return result


wordpress.create_post = _final_create
