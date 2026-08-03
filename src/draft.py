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

0. MULTIPLE DISTINCT STORIES -- SPLIT, DON'T MERGE: before writing, check
   whether the source material actually covers two or more clearly
   separate, independently newsworthy stories rather than one connected
   story. This commonly happens with festival coverage, where multiple
   artists each have their own unrelated news angle (e.g. one artist's
   onstage mishap or surprise guest, a different artist's setlist or
   album news) and only share an event/venue/date in common. Two artists
   appearing at the same festival are NOT one story just because they
   share an event -- if each has independently newsworthy content (its
   own quotes, its own noteworthy moment), write them as SEPARATE
   articles, not one combined piece. A single combined article targets
   two unrelated search intents at once and weakens both articles' SEO
   and CTR compared to two focused ones. Only keep multiple
   artists/subjects in ONE article when they are genuinely part of the
   SAME story -- a collaboration, a joint announcement, a feud, or one
   artist's guest appearance as the specific narrative focus of the other
   artist's news (in that case the guest appearance is a detail within
   one story, not a second story).
   When you do split, base EACH article only on the subset of source
   material that's actually about that story -- do not pad a thin split
   with details that belong to the other story.
1a. NEVER ADD A FACT THAT ISN'T IN THE SOURCE MATERIAL, even if it sounds
    plausible, even if you're confident it's true from general knowledge,
    and even if it would make the story better. This includes: final
    scores/results of an event, awards or honors given, specific dollar/
    pound amounts, dates or years for anything not explicitly dated in
    the source, and any future event, plan, or outcome not mentioned in
    the source. If the source describes an anecdote without saying how it
    ended, don't invent an ending -- either leave it open the way the
    source does, or note that the source doesn't say. A vivid, specific
    detail that sounds like it completes the story (a final score, a
    trophy, a dollar figure) is exactly the kind of thing that's tempting
    to add and easy for a reader to disprove -- treat any urge to add one
    as a signal to stop and check the source again, not a sign it's safe.
1b. BEFORE finalizing the article, re-read it against the source material
    one claim at a time and ask: is every date, number, named event,
    award, quote, and outcome in this draft actually present in the
    source? Cut anything that isn't, rather than leaving it in and hoping
    it's close enough.
1d. STAY ON THE MAIN STORY'S ANGLE. Even a true, verifiable fact about the
    subject can weaken an article if it's unrelated to the specific story
    being told -- e.g. tacking on an unconnected career trivia item at
    the end just to add length. Before including a fact, ask: does this
    directly serve the story I'm telling, or is it just trivia about the
    same person/band? If it's the latter, cut it.
1c. DON'T STRIP OUT THE SOURCE'S OWN SCENE-SETTING AND CONTEXT to make the
    article shorter or more "news-brief" -- concrete, specific details
    the source already provides (what a room looked like, what someone
    was doing right before/after, related career context establishing
    why the moment mattered) are what make a piece feel like a real story
    instead of a bare recap, and they carry zero hallucination risk since
    they're already confirmed by the source. Cutting them for brevity is
    a worse tradeoff than keeping the article a bit longer.
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
2a. DON'T BURY OR DROP THE BIGGEST FACT: before finalizing the article,
    scan all the source material for the single most newsworthy element
    -- the thing most sources emphasize, or the one with the most news
    value (an injury or safety incident, a surprise guest, a public
    dispute, a record broken, a cancellation). If that fact exists in
    the sources, it must be in the article, and prominently -- not
    mentioned in passing or left out in favor of less significant
    details like a full setlist. A setlist is supporting detail; an
    incident during the show is the story.
3. Direct quotes from the people involved (band members, reps, etc.) may
   be quoted verbatim with attribution if they appear in the source
   material -- these are factual statements, not the reporting
   journalist's own copyrighted prose, so quoting them accurately is
   good journalism, not something to paraphrase into vagueness.
   HOWEVER: do not copy the reporting journalist's own sentences,
   descriptions, structure, or paragraph order from any source. Write
   your own original sentences built from the facts and quotes.
3a. DON'T INVENT NEAR-QUOTES: if you're describing what someone said or
    the gist of a remark, make it clearly indirect ("joked that the
    delay put the show's quality in question") rather than phrasing it
    as a specific, quotable-sounding turn of phrase that reads like an
    actual quote. Only use quotation marks around text that is the
    verbatim wording given in the source material -- never construct a
    plausible-sounding quote or tighten/punch up a paraphrase until it
    reads like one.
4. Lead with the most newsworthy, concrete fact (the "what happened"),
   not a scene-setting or scene-editorializing opener. A good test: could
   this headline/opening be confirmed as accurate by someone who just
   read the source material? If not, it's too speculative.
4a. LISTS: if the article includes a setlist, ranking, or any other
    enumerated set of items (song titles, album tracklist, etc.), format
    it as a proper HTML list -- <ol><li>Song One</li><li>Song Two</li>...
    </ol> for a setlist (order matters) or <ul> for an unordered list --
    never as a run-together comma-separated list inside a <p>.
4a-i. SETLIST ACCURACY: never construct, complete, or infer a setlist --
    a specific, ordered list of songs performed is exactly the kind of
    detail readers expect to be 100% accurate, and getting even one
    song or the order wrong is a real credibility problem. Only include
    a setlist (full or partial) if the source material explicitly
    provides one (e.g. citing Setlist.fm or listing the songs played in
    order). If the sources only mention a handful of songs performed
    without a full confirmed list, name those specific songs in prose
    instead of presenting them as a complete setlist. If no source
    gives any song-by-song detail, don't include a setlist section at
    all.
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
5. TITLE STYLE: the title must be specific, accurate, AND genuinely
   eye-catching / SEO-optimized -- not a dry, academic-sounding
   description of the article's topic. Avoid vague, report-style
   phrasing like "Examining the Overlooked Solo Albums From Members of
   the Eagles" -- prefer a direct, punchy phrasing a reader would
   actually want to click and that matches how people search, e.g. "The
   Best Eagles Solo Albums You May Have Missed" for that same article.
   Where it genuinely fits the content, lean on proven high-CTR patterns
   -- "The Best X You May Have Missed", "Why X Still Matters", "X Songs
   You Forgot Were Written By Y", direct numbers ("5 Deep Cuts..."),
   or a strong direct claim -- without turning into misleading clickbait
   or overstating what the article actually says. Every word in the
   title must still be something the article backs up.
6. Output must be a valid JSON ARRAY, and NOTHING else -- no markdown code
   fences, no preamble, no explanation. In the normal case (one story),
   the array has exactly ONE element. If rule 0 applies (the source
   material covers multiple distinct newsworthy stories), the array has
   one element per story, each a fully independent article following
   every rule above on its own. Each element must match this exact
   schema:

[
{{
  "title": "string, accurate and specific, states the actual news or angle, written to be eye-catching and SEO-friendly (see rule 5), under 70 chars",
  "seo_title": "string, SEO title tag, under 60 chars, includes primary keyword",
  "meta_description": "string, under 155 chars, includes primary keyword, makes people want to click",
  "focus_keyword": "string, 2-4 word primary SEO keyword phrase for this article",
  "tags": ["exactly 8 to 10 relevant tags, e.g. band names, genres, related artists, subgenres"],
  "excerpt": "string, 1-2 sentence factual teaser, under 200 chars",
  "content_html": "string, the full article body as clean HTML using <p>, <h2>, <h3> tags where natural. Do NOT include an <h1> (WordPress adds the title separately). LENGTH: when full article text was retrieved for the sources (not just RSS summaries), extract and include the genuinely reported facts, direct quotes, and specific details actually present in that full text -- aim for 600-900 words in that case, matching the depth of a real news report rather than a condensed summary of one. When only RSS summaries are available (no full text), 300-500 words is appropriate since there's less real material to draw from -- don't pad with speculation just to hit a length target either way. The test is always: does the length match how much genuine source material exists, not a fixed target."
}}
]
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

DUPLICATE_SYSTEM_PROMPT = """You are an editor for Deadikace, a rock music
blog, checking new candidate news topics against a list of the blog's own
recent post titles.

For each numbered candidate topic below, decide whether it is about the
SAME underlying news story/event as any of the recent post titles -- even
if the wording is completely different, or the candidate topic was picked
up from a different outlet than whatever the earlier post was based on.
The same announcement, event, or quote described in different words by a
different publication still counts as a duplicate. A different piece of
news about the same band/artist is NOT a duplicate (e.g. two separate
posts about two different Metallica news items are both fine).

Respond with ONLY a JSON array of the candidate numbers that ARE
duplicates of an existing recent post (i.e. should be skipped), e.g.
[2,5]. Respond with [] if none are duplicates. No other text.
"""

def filter_duplicate_topics(topics, existing_titles):
    """
    Removes candidate topics that are about the same underlying story as
    a recently-published Deadikace post, even when the wording is
    completely different or the story was picked up from a different
    outlet than whichever one the earlier post was drafted from.

    This exists because the cheaper per-topic guard in main.py
    (_title_already_covered) compares the source RSS headline against the
    post's own published title -- but that published title is an
    original, LLM-authored headline, not a copy of the source headline,
    so genuinely duplicate stories can still slip through if the wording
    diverges enough. This is a batched LLM call (like
    filter_rock_relevant_topics above) so it costs one request per run,
    not one per topic. Falls back to keeping everything if the
    classification call fails for any reason.
    """
    if not topics or not existing_titles:
        return topics

    candidate_list = "\n".join(
        f"{i + 1}. " + " / ".join(item["title"] for item in t["items"][:3])
        for i, t in enumerate(topics)
    )
    existing_list = "\n".join(f"- {t}" for t in existing_titles)
    user_prompt = (
        f"Recent Deadikace post titles:\n{existing_list}\n\n"
        f"Candidate topics (each line may list headlines from multiple "
        f"outlets covering what might be the same candidate story):\n"
        f"{candidate_list}\n\n"
        "Return the JSON array of duplicate candidate numbers now."
    )

    try:
        raw = _call_llm(DUPLICATE_SYSTEM_PROMPT, user_prompt, max_tokens=1000)
        raw = _clean_json_text(raw)
        dup_idx = {int(i) - 1 for i in json.loads(raw)}
    except Exception as e:
        print(f"[warn] Duplicate-story filtering failed ({e}); proceeding without it.")
        return topics

    for i in sorted(dup_idx):
        if 0 <= i < len(topics):
            print(f"[info] Skipping likely duplicate of an existing post: "
                  f"{topics[i]['items'][0]['title']}")

    return [t for i, t in enumerate(topics) if i not in dup_idx]

WITHIN_BATCH_DUPLICATE_SYSTEM_PROMPT = """You are an editor for Deadikace,
a rock music blog, checking a batch of candidate news topics against EACH
OTHER (not against older posts) to catch cases where the same underlying
story was picked up by the topic clustering step as two or more separate
candidates -- this happens when different outlets word the same story
differently enough (e.g. one outlet leads with "artist reveals health
scare", another leads with a specific quote from the same interview) that
simple keyword-overlap clustering treats them as unrelated.

For each numbered candidate topic below, decide whether it covers the SAME
underlying news story/event as any EARLIER-numbered candidate in this same
list. Two candidates about the same band/artist but genuinely different
news (e.g. a tour announcement and, separately, an album review) are NOT
duplicates -- only flag candidates that are actually the same story.

Respond with ONLY a JSON array of the candidate numbers that are
duplicates of an earlier candidate in this list (i.e. should be dropped,
keeping only the earlier one), e.g. [3,5]. Respond with [] if none are
duplicates. No other text.
"""

def dedupe_topics_within_batch(topics):
    """
    Catches the case where topic clustering (discover.py) failed to merge
    two RSS entries about the same underlying story into one cluster --
    e.g. two different outlets covering the same interview with different
    enough headlines that keyword-overlap clustering missed the match.
    Without this, both would be drafted as separate topics and published
    as duplicate articles in the same run, since the other duplicate
    guards (_title_already_covered, filter_duplicate_topics) only compare
    against ALREADY-PUBLISHED posts, not against sibling candidates in the
    same batch. Falls back to keeping everything if the LLM call fails.
    """
    if not topics or len(topics) < 2:
        return topics

    candidate_list = "\n".join(
        f"{i + 1}. " + " / ".join(item["title"] for item in t["items"][:3])
        for i, t in enumerate(topics)
    )
    user_prompt = f"Candidate topics:\n{candidate_list}\n\nReturn the JSON array of duplicate candidate numbers now."

    try:
        raw = _call_llm(WITHIN_BATCH_DUPLICATE_SYSTEM_PROMPT, user_prompt, max_tokens=1000)
        raw = _clean_json_text(raw)
        dup_idx = {int(i) - 1 for i in json.loads(raw)}
    except Exception as e:
        print(f"[warn] Within-batch duplicate check failed ({e}); proceeding without it.")
        return topics

    for i in sorted(dup_idx):
        if 0 <= i < len(topics):
            print(f"[info] Dropping likely duplicate within this batch (same story as an "
                  f"earlier candidate): {topics[i]['items'][0]['title']}")

    return [t for i, t in enumerate(topics) if i not in dup_idx]

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
    already enriched with full_text via article_fetch.enrich_topic_with_full_text

    Returns a LIST of article dicts -- normally just one, but more than
    one if the model determined the source material actually covers
    multiple distinct newsworthy stories (see rule 0 in
    DRAFT_SYSTEM_PROMPT), e.g. two different artists at the same
    festival each with their own unrelated news angle."""
    source_material, full_text_count = _build_source_material(topic)
    user_prompt = (
        f"Topic covered by {topic['source_count']} outlet(s), "
        f"{full_text_count} with full article text retrieved:\n\n"
        f"{source_material}\n\n"
        "Write the Deadikace news report(s) now, synthesizing across all "
        "sources above. Respond with ONLY the JSON array, no other text."
    )

    raw_text = _call_llm(DRAFT_SYSTEM_PROMPT, user_prompt, max_tokens=4000)
    raw_text = _clean_json_text(raw_text)

    try:
        articles = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{LLM_PROVIDER} did not return valid JSON: {e}\nRaw output:\n{raw_text}")

    if isinstance(articles, dict):
        articles = [articles]

    if not isinstance(articles, list) or not articles:
        raise ValueError(f"{LLM_PROVIDER} did not return a non-empty JSON array: {raw_text}")

    return articles

VERIFY_SYSTEM_PROMPT = """You are a copy editor fact-checking AND
readability-editing a draft news article against its source material,
for a rock music blog called Deadikace.

Pay special attention to fabricated specifics that read as plausible but
aren't actually in the source: an invented final score or result, an
invented award/honor, an invented dollar/pound amount, an invented date,
or an invented future event/plan. These are the most common and most
damaging kind of error -- if you find one, remove it (don't just soften
the wording) rather than leaving a fabricated-but-vague version in.

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
- Do NOT shorten the article or remove factual content while fixing
  readability issues. Adding subheadings and splitting long sentences
  should reorganize/reformat the existing content, not condense it -- the
  corrected version should be roughly the same length as the draft (or
  slightly longer, since splitting a sentence adds a few words), never
  noticeably shorter.

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
    verification call fails.
    """
    source_material, _ = _build_source_material(topic)

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

    def _word_count(html_str):
        text_only = re.sub(r"<[^>]+>", " ", html_str)
        return len(text_only.split())

    words_before = _word_count(article["content_html"])
    words_after = _word_count(corrected_html)
    if words_before > 0 and words_after < words_before * 0.8:
        print(f"[warn] Fact-check pass shrank the article significantly "
              f"({words_before} -> {words_after} words); keeping the original "
              f"draft instead, since readability fixes should reorganize "
              f"content, not cut it.")
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
