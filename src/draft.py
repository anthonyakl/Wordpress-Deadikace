"""
Uses an LLM (Claude or Gemini, whichever is configured) to:
  1. Classify which candidate topics are actually rock-relevant (some
     competitor feeds, e.g. Rolling Stone, cover all genres -- keyword
     matching can't reliably catch "BTS" or "Lil Baby" as off-topic since
     those headlines don't contain words like "k-pop" or "rap" at all, but
     an LLM recognizes the artists directly).
  2. Draft an original, SEO-optimized article for a chosen topic. Only
     titles/summaries/links from competitor RSS feeds are passed in as
     factual signals -- the model is explicitly instructed to write
     original analysis, not to paraphrase or closely follow any single
     source.
"""

import json

from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    SITE_VOICE_GUIDELINES,
)

DRAFT_SYSTEM_PROMPT = f"""You are a staff news writer for Deadikace, a rock music blog.

{SITE_VOICE_GUIDELINES}

You will be given a news topic and the source material several outlets
published about it (for outlets where full text was retrievable, you'll
have the full article; for others, only the RSS title/summary). Your job
is to write a STRAIGHT NEWS REPORT, not an editorial or opinion piece --
the same way a real reporter would write up a wire story using multiple
outlets' coverage as their source material.

1. FACTUAL ACCURACY IS THE TOP PRIORITY. Every specific claim -- dates,
   numbers, event names, direct quotes, who-said-what -- must come from
   the provided source material. If a detail isn't in any of the sources,
   do not include it, and do not infer or guess at specifics. It is
   better to write a shorter, plainer article than to invent a plausible-
   sounding detail. Do not add speculative framing, dramatic
   interpretation, or editorializing about what something "means" beyond
   what the sources themselves report.
2. SYNTHESIZE ACROSS ALL provided sources to build the most complete,
   accurate picture -- don't just rewrite the single source with the most
   detail. Cross-check: if multiple sources report the same fact, that's
   a signal it's solid; if only one source mentions a specific detail,
   still fine to include it, but don't build the whole article's angle
   around one outlet's framing.
3. Direct quotes from the people involved (band members, reps, etc.) may
   be quoted verbatim with attribution if they appear in the source
   material -- these are factual statements, not the reporting
   journalist's own copyrighted prose, so quoting them accurately is
   good journalism, not something to paraphrase into vagueness.
   HOWEVER: do not copy the reporting journalist's own sentences,
   descriptions, structure, or paragraph order from any source. Write
   your own original sentences built from the facts and quotes.
4. Lead with the most newsworthy, concrete fact (the "what happened"),
   not a scene-setting or scene-editorializing opener. A good test: could
   this headline/opening be confirmed as accurate by someone who just
   read the source material? If not, it's too speculative.
5. Include image placeholders in content_html: put <!--IMAGE_1--> right
   after the article's opening paragraph (this becomes both the featured
   image and the first in-article image), then <!--IMAGE_2-->,
   <!--IMAGE_3--> etc., one after each subsequent <h2> section heading.
   Use 2-4 placeholders total depending on article length (short article:
   2, long article: up to 4). For each placeholder, provide a matching
   entry in "image_queries", in the same order. Each query should name
   the SPECIFIC band/artist/album relevant to that part of the article
   (e.g. "Metallica live concert", "Ozzy Osbourne portrait", "Master of
   Puppets album art") -- the system will first search for a real,
   properly-licensed photo of that exact subject, and automatically fall
   back to a generic music-themed stock photo only if none is found. Keep
   each query concise (3-6 words).
6. Output must be valid JSON matching this exact schema, and NOTHING else
   -- no markdown code fences, no preamble, no explanation:

{{
  "title": "string, accurate and specific (not vague or clickbait-y), states the actual news, under 70 chars",
  "seo_title": "string, SEO title tag, under 60 chars, includes primary keyword",
  "meta_description": "string, under 155 chars, includes primary keyword, makes people want to click",
  "focus_keyword": "string, 2-4 word primary SEO keyword phrase for this article",
  "tags": ["exactly 8 to 10 relevant tags, e.g. band names, genres, related artists, subgenres"],
  "image_queries": ["specific band/artist/album name + descriptive term for each IMAGE placeholder used, in order"],
  "excerpt": "string, 1-2 sentence factual teaser, under 200 chars",
  "content_html": "string, the full article body as clean HTML using <p>, <h2>, <h3> tags where natural. 400-700 words -- write to the facts available, don't pad with speculation to hit a length target. Do NOT include an <h1> (WordPress adds the title separately)."
}}
"""

RELEVANCE_SYSTEM_PROMPT = """You are a rock music editor triaging news headlines for a rock/metal blog called Deadikace.

Given a numbered list of headlines, identify which ones are genuinely
relevant to rock, hard rock, metal, punk, grunge, alternative rock,
classic rock, or prog rock -- bands, artists, albums, tours, gear, or
rock culture news.

Exclude headlines primarily about other genres (pop, rap/hip-hop, K-pop,
R&B, country, EDM/dance) or unrelated topics (general lifestyle, cars,
celebrities outside music), even if a rock artist is only mentioned in
passing. When genuinely unsure, lean toward keeping it.

Respond with ONLY a JSON array of the relevant headline numbers, e.g.
[1,3,4,7]. No other text, no explanation.
"""


def _clean_json_text(raw_text):
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    return raw_text


def _call_anthropic(system_prompt, user_prompt, max_tokens):
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _call_gemini(system_prompt, user_prompt, max_tokens):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        ),
    )
    return response.text


def _call_llm(system_prompt, user_prompt, max_tokens=4000):
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set.")
        return _call_anthropic(system_prompt, user_prompt, max_tokens)
    elif LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set.")
        return _call_gemini(system_prompt, user_prompt, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} (expected 'anthropic' or 'gemini')")


def filter_rock_relevant_topics(topics):
    """
    Sends just the headlines (one batched call, not one per topic) to the
    configured LLM and keeps only genuinely rock-relevant ones. Falls back
    to keeping everything if the classification call fails for any reason
    -- better to occasionally draft an off-topic article than to silently
    drop every candidate topic due to a transient API error.
    """
    if not topics:
        return topics

    headline_list = "\n".join(
        f"{i + 1}. {t['items'][0]['title']}" for i, t in enumerate(topics)
    )
    user_prompt = f"Headlines:\n{headline_list}\n\nReturn the JSON array of relevant numbers now."

    try:
        raw = _call_llm(RELEVANCE_SYSTEM_PROMPT, user_prompt, max_tokens=1000)
        raw = _clean_json_text(raw)
        indices = json.loads(raw)
        keep_idx = {int(i) - 1 for i in indices}
    except Exception as e:
        print(f"[warn] Rock-relevance filtering failed ({e}); proceeding without it "
              f"(all {len(topics)} topics kept).")
        return topics

    filtered = [t for i, t in enumerate(topics) if i in keep_idx]
    print(f"[info] Rock-relevance filter kept {len(filtered)} of {len(topics)} topics.")
    return filtered


def draft_article(topic):
    """topic: one cluster dict from discover.get_trending_topics(), ideally
    already enriched with full_text via article_fetch.enrich_topic_with_full_text"""
    source_blocks = []
    full_text_count = 0
    for item in topic["items"]:
        full_text = item.get("full_text")
        if full_text:
            full_text_count += 1
            source_blocks.append(
                f"Source: {item['source']} (FULL ARTICLE TEXT)\n"
                f"Title: {item['title']}\n"
                f"Full text: {full_text}\n"
                f"Link: {item['link']}"
            )
        else:
            source_blocks.append(
                f"Source: {item['source']} (RSS summary only)\n"
                f"Title: {item['title']}\n"
                f"Summary: {item['summary']}\n"
                f"Link: {item['link']}"
            )

    source_material = "\n\n".join(source_blocks)
    user_prompt = (
        f"Topic covered by {topic['source_count']} outlet(s), "
        f"{full_text_count} with full article text retrieved:\n\n"
        f"{source_material}\n\n"
        "Write the Deadikace news report now, synthesizing across all "
        "sources above. Respond with ONLY the JSON object, no other text."
    )

    raw_text = _call_llm(DRAFT_SYSTEM_PROMPT, user_prompt, max_tokens=4000)
    raw_text = _clean_json_text(raw_text)

    try:
        article = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{LLM_PROVIDER} did not return valid JSON: {e}\nRaw output:\n{raw_text}")

    return article
