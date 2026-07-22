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
import re

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
1a. NEUTRAL FRAMING -- do not upgrade a neutral fact into a more dramatic
   or interpretive claim. Concretely:
   - Don't imply an audience's reaction or sentiment ("fans were
     disappointed by...") unless a source explicitly says the audience
     reacted that way. If a source says fans were surprised by a specific
     change (e.g. omitted songs), keep that specific framing rather than
     substituting a different implied reaction (e.g. don't turn "fans
     were surprised by omitted songs" into "fans experienced a shorter
     show," which shifts the claim and implies a negative reaction the
     source didn't state).
   - Don't frame a neutral numeric comparison as a deliberate decision
     unless the source says it was one. "The show ran 90 minutes, down
     from 120 at the previous stop" is neutral; "marking a deliberate
     reduction in length" implies intent the source may not support --
     only use language like that if a source explicitly frames it as a
     choice.
   - When a specific stat or ranking is attributed to a named source in
     the article (e.g. "according to setlist.fm," a chart, a specific
     report), KEEP that attribution in your own sentence rather than
     generalizing it into an unqualified claim. "Third-most-played song
     in the band's catalog, per setlist.fm" is accurate; "third-most-
     performed song in the band's history" overstates it as an official,
     universally-tracked ranking.
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
4a. LISTS: if the article includes a setlist, ranking, or any other
   enumerated set of items (song titles, album tracklist, etc.), format
   it as a proper HTML list -- <ol><li>Song One</li><li>Song Two</li>...
   </ol> for a setlist (order matters) or <ul> for an unordered list --
   never as a run-together comma-separated list inside a <p>.
4b. READABILITY -- SUBHEADINGS: if the article is longer than ~300 words,
   it MUST include at least one <h2> subheading within the first ~300
   words, and roughly one <h2> every 250-350 words after that. Do not let
   a long, unbroken run of paragraphs sit under only the title. Subheads
   should be genuine section breaks (a new angle, a new fact cluster, a
   quote section) not decorative -- and skip them entirely for short
   articles (under ~300 words), which don't need any.
4c. READABILITY -- SENTENCE LENGTH: keep the large majority of sentences
   at 20 words or fewer. If a sentence runs longer, check whether it's
   really two ideas joined together -- if so, split it into two
   sentences. A few longer sentences are fine for natural variety, but
   they should be a small minority (well under a quarter of all
   sentences), not close to half.
5. Include image placeholders in content_html: put <!--IMAGE_1--> right
   after the article's opening paragraph (this becomes both the featured
   image and the first in-article image), then <!--IMAGE_2-->,
   <!--IMAGE_3--> etc., one after each subsequent <h2> section heading.
   Use 2-4 placeholders total depending on article length (short article:
   2, long article: up to 4). For each placeholder, provide a matching
   entry in "image_queries", in the same order. Each query should name
   the SPECIFIC band/artist relevant to that part of the article (e.g.
   "Metallica live concert", "Ozzy Osbourne portrait", "Master of
   Puppets album art") -- PREFER the band/artist over a venue name even
   in sections about a specific show or venue (e.g. use "Bon Jovi live
   concert" rather than "Madison Square Garden," since venue photos tend
   to be architecture/exterior shots that are a weaker fit than a band
   photo) -- the system will first search for a real,
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
  "content_html": "string, the full article body as clean HTML using <p>, <h2>, <h3> tags where natural. Do NOT include an <h1> (WordPress adds the title separately). LENGTH: when full article text was retrieved for the sources (not just RSS summaries), extract and include the genuinely reported facts, direct quotes, and specific details actually present in that full text -- aim for 600-900 words in that case, matching the depth of a real news report rather than a condensed summary of one. When only RSS summaries are available (no full text), 300-500 words is appropriate since there's less real material to draw from -- don't pad with speculation just to hit a length target either way. The test is always: does the length match how much genuine source material exists, not a fixed target."
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


def _build_source_material(topic):
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
    return "\n\n".join(source_blocks), full_text_count


def draft_article(topic):
    """topic: one cluster dict from discover.get_trending_topics(), ideally
    already enriched with full_text via article_fetch.enrich_topic_with_full_text"""
    source_material, full_text_count = _build_source_material(topic)
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


VERIFY_SYSTEM_PROMPT = """You are a copy editor fact-checking AND
readability-editing a draft news article against its source material,
for a rock music blog called Deadikace.

You will be given the draft article's HTML content and the original
source material it was based on. Do two passes over it:

PASS 1 -- FACTUAL ACCURACY. Check every specific claim, framing choice,
and implied reaction in the draft against the sources:
- Flag and fix any claim (date, number, quote, attribution, or implied
  reaction/sentiment) that isn't clearly supported by the sources.
- Flag and fix any place where a neutral fact from the sources was
  reframed as more dramatic, interpretive, or emotionally loaded than the
  sources themselves state (e.g. a neutral numeric comparison reframed as
  a deliberate decision; a specific stated reaction swapped for a
  different implied one).
- Flag and fix any stat/ranking that dropped its source attribution
  (e.g. "third-most-played... per setlist.fm" turned into an unqualified
  claim).

PASS 2 -- READABILITY. Check the draft against these rules and fix any
that are violated:
- If the article is longer than ~300 words, it must have at least one
  <h2> subheading within the first ~300 words, and roughly one <h2>
  every 250-350 words after that. If a long article has few or no
  subheadings, insert natural section breaks (<h2>) at sensible points
  -- don't force one into a short article under ~300 words.
- The large majority of sentences should be 20 words or fewer. If you
  find many sentences over 20 words (more than roughly a quarter of all
  sentences), split some of the longest/most complex ones into two
  clearer sentences without changing their meaning or dropping content.

GENERAL RULES FOR BOTH PASSES:
- Do NOT rewrite anything that isn't actually a problem -- keep sentences
  that are already accurate and well-formed exactly as they are. Make the
  smallest edit that fixes each issue, not a full rewrite.
- Preserve every <!--IMAGE_N--> placeholder EXACTLY where it is in the
  content (inserting a new <h2> near one is fine, just don't remove or
  move the placeholder itself).

Respond with ONLY this JSON, nothing else:
{
  "content_html": "the corrected HTML (or unchanged, if no issues found)",
  "issues_found": ["short plain-English description of each fix made -- empty array if none"]
}
"""


def verify_and_refine(article, topic):
    """
    Second pass: re-checks the drafted article against the same source
    material and tightens up any embellishment/over-interpretation before
    publishing. Falls back to the original article unchanged if the
    verification call fails, or if it would have removed/altered the
    image placeholders (a sign something went wrong with the edit).
    """
    source_material, _ = _build_source_material(topic)
    placeholders_before = set(re.findall(r"<!--IMAGE_\d+-->", article["content_html"]))

    user_prompt = (
        f"Source material:\n\n{source_material}\n\n"
        f"Draft article content_html:\n\n{article['content_html']}\n\n"
        "Fact-check and return the corrected JSON now."
    )

    try:
        raw_text = _call_llm(VERIFY_SYSTEM_PROMPT, user_prompt, max_tokens=4000)
        raw_text = _clean_json_text(raw_text)
        result = json.loads(raw_text)
        corrected_html = result["content_html"]
    except Exception as e:
        print(f"[warn] Fact-check pass failed ({e}); publishing the original draft unchanged.")
        return article

    placeholders_after = set(re.findall(r"<!--IMAGE_\d+-->", corrected_html))
    if placeholders_after != placeholders_before:
        print("[warn] Fact-check pass altered the image placeholders unexpectedly; "
              "keeping the original draft instead to avoid breaking image insertion.")
        return article

    issues = result.get("issues_found") or []
    if issues:
        print(f"[info] Fact-check pass made {len(issues)} correction(s):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[info] Fact-check pass found no issues.")

    article["content_html"] = corrected_html
    return article
