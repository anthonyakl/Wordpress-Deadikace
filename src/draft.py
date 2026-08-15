"""
Uses an LLM (Claude or Gemini, whichever is configured) to:
1. Classify which candidate topics are actually rock-relevant (some
   competitor feeds, e.g. Rolling Stone, cover all genres -- keyword
   matching can't reliably catch "BTS" or "Lil Baby" as off-topic since
   those headlines don't contain words like "k-pop" or "rap" at all, but
   an LLM recognizes the artists directly).
2. Research additional factual grounding for a chosen topic beyond the
   immediate competitor RSS coverage, using a live web-search tool (see
   research_additional_context() below) -- background, catalog/chart
   data, prior interviews, etc., each tied to a real source URL.
3. Draft an original, SEO-optimized article for a chosen topic. Only
   titles/summaries/links from competitor RSS feeds -- plus, when
   available, the independently-researched notes from step 2 -- are
   passed in as factual signals; the model is explicitly instructed to
   write original analysis, not to paraphrase or closely follow any
   single source, and to add real value beyond a straight recap of that
   source material (see DRAFT_SYSTEM_PROMPT rule 2b).
"""

import json
import re

from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    SITE_VOICE_GUIDELINES,
    ENABLE_DEEP_RESEARCH, RESEARCH_MAX_SEARCHES,
)

DRAFT_SYSTEM_PROMPT = f"""You are a staff news writer for Deadikace, a rock music blog.

{SITE_VOICE_GUIDELINES}

You will be given a news topic and the source material several outlets
published about it (for outlets where full text was retrievable, you'll
have the full article; for others, only the RSS title/summary). You may
also be given an "Additional Research Notes" block -- independently
researched facts, gathered via live web search specifically for this
topic, that go beyond what the competitor RSS coverage alone contains
(see rule 2b for how to use it). Your job is to write a STRAIGHT NEWS
REPORT, not an editorial or opinion piece -- the same way a real reporter
would write up a wire story using multiple outlets' coverage (and their
own further research) as their source material.

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
   Each source block below is labeled "Source #N". For every article you
   output (whether the array has one element or several), set
   source_item_indices to the list of source numbers that article is
   actually based on. This matters even when there's only one story --
   it's how the featured image and other per-source assets get matched
   to the right article instead of picked from the wrong source when a
   topic contains multiple unrelated items.
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
    This rule applies with EQUAL force to the Additional Research Notes
    block, if one is provided -- it is source material, not license to
    speculate. See rule 2b for how it's meant to be used.
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
    same person/band? If it's the latter, cut it. This applies to facts
    drawn from Additional Research Notes too -- added depth should serve
    the story, not just prove that research happened.
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
1e. DON'T EDITORIALIZE CERTAINTY THE SOURCE DOESN'T HAVE. It's tempting to
    frame a plain announcement as more dramatic than it is -- e.g.
    writing that a band is "keeping something going past previous
    expectations" when the source only reports a new announcement, with
    no official prior end-date ever stated. If the sense of a prior
    "expectation" comes from someone's informal comment (not an official
    announcement), say so explicitly and attribute it, rather than
    writing as if a firm plan existed and was later changed. Prefer
    wording that states what was announced and, separately, what was
    previously suggested and by whom -- don't merge the two into a single
    implied narrative arc ("was expected to end, but now...") unless the
    source itself frames it that way. Example: source says only "the
    band announced 2027 tour dates"; a band member had separately mused
    that this year might be their last. WRONG: "The band is continuing
    the tour well past earlier expectations." RIGHT: "The band announced
    new 2027 tour dates, despite [member]'s earlier comments suggesting
    this year could be their last."

2. SYNTHESIZE ACROSS ALL provided sources to build the most complete,
   accurate picture -- don't just rewrite the single source with the most
   detail. Cross-check: if multiple sources report the same fact, that's
   a signal it's solid; if only one source mentions a specific detail,
   still fine to include it, but don't build the whole article's angle
   around one outlet's framing. "All provided sources" includes the
   Additional Research Notes block, when present, on equal footing with
   the numbered Source blocks.
2a. DON'T BURY OR DROP THE BIGGEST FACT: before finalizing the article,
    scan all the source material for the single most newsworthy element
    -- the thing most sources emphasize, or the one with the most news
    value (an injury or safety incident, a surprise guest, a public
    dispute, a record broken, a cancellation). If that fact exists in
    the sources, it must be in the article, and prominently -- not
    mentioned in passing or left out in favor of less significant
    details like a full setlist. A setlist is supporting detail; an
    incident during the show is the story.
2b. ADD GENUINE VALUE BEYOND A STRAIGHT RECAP -- THE ORIGINALITY TEST.
    Before finalizing, ask: if a reader already read the competitor
    coverage this topic is based on, would they still learn something
    from this article? Restructuring the same handful of competitor
    articles into new sentences is NOT enough on its own -- rewriting the
    same facts in different words is still a close paraphrase even when
    no single sentence is copied verbatim.
    If an Additional Research Notes block is provided below the source
    material, use the verifiable facts in it to add real depth the
    competitor coverage alone doesn't have -- catalog/discography
    context, chart or certification data (with its source), relevant
    historical background, or how this specific story fits into the
    artist's or band's broader, already-documented history. Every fact
    drawn from Additional Research Notes is bound by the exact same
    rules as rule 1: never state anything not actually present in that
    block, and never treat "it wasn't in the RSS feed" as license to
    speculate just because it feels like background knowledge you
    already have.
    If NO Additional Research Notes block is provided, or it genuinely
    contains nothing usable for this specific story, do not force in
    unrelated trivia to compensate (see rule 1d) -- a shorter, purely
    factual straight report is the right and acceptable output in that
    case, exactly as rule 1 already prefers. The goal is a smaller
    number of genuinely valuable articles, not padding every article to
    a fixed length just to look more original.

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
4c. ILLUSTRATIVE IMAGES: propose 1-2 additional images for a short article
    (under ~500 words), or up to 3-4 for a longer one -- scale to length,
    don't force more images than the content naturally supports. Each one
    should illustrate something SPECIFIC actually mentioned in the
    article (a named album's cover, a named venue, a specific person or
    event), not a generic mood shot. Write the search query as a
    specific, disambiguating phrase (see the illustrative_images schema
    field below for the exact reasoning and an example) -- this is
    searched against Wikimedia Commons, so a vague or bare-name query
    risks matching the wrong, unrelated subject. For placement, prefer
    right after the H2 heading whose section that image best supports;
    only use the "middle of the article" fallback if there's truly no
    heading it fits better under. If the article has no natural subject
    for extra images beyond the featured image (e.g. a very short news
    brief), return an empty illustrative_images array rather than forcing
    irrelevant ones in.
4d. EMBED VIDEOS FROM THE SOURCE: if any source block above lists
    "Videos embedded in this source", include EVERY one of those videos
    in video_embeds, using the exact URL given -- never invent a URL and
    never skip a video just to save space. Set placement_after_heading to
    whichever heading in YOUR OWN article structure covers that same
    item; use the video's listed "near heading" text as a hint for which
    item it belongs to, since the source's heading wording may differ
    from yours.
4e. COVER EVERY ITEM IN A ROUNDUP/LIST SOURCE: if the source material is
    a roundup-style article covering multiple distinct items (e.g. "12
    live performances just uploaded", "10 albums you need to hear"),
    your article must cover EVERY item from the source, not a subset --
    dropping items because there are many of them is not acceptable, and
    is different from rule 0's story-splitting (which is about
    separating UNRELATED stories, not trimming a single list). A shorter
    per-item treatment is fine if needed to fit them all in, but every
    item must appear.
4f. PREFER SOURCE-BODY IMAGES FOR PROMOTIONAL/COVER ART: if a source
    lists "Other images in this source's article body", check whether
    any of them is what the article is actually about showing (e.g. an
    official tour poster, an album cover, promotional artwork for an
    event). Wikimedia Commons frequently has no free-licensed version of
    this kind of copyrighted promotional material, so illustrative_images
    (Wikimedia search) will often come up empty for exactly this case --
    when that's what the article calls for, use source_images instead of
    (or alongside) a Wikimedia search. Match the image to what its alt
    text or nearby heading suggests it depicts; never guess.
4g. SOURCES & FURTHER READING SECTION (complementary sources only): the
    numbered Source blocks are the outlets (e.g. Louder Sound, Loudwire,
    Rolling Stone) that already reported this same core story -- they
    are the article's primary source material, not "further reading,"
    so NEVER cite them in this section. Only include a "Sources &
    Further Reading" section at all if the Additional Research Notes
    block was actually drawn upon for a fact, quote, or piece of
    context in the article -- i.e. you used genuinely complementary
    material (background, catalog/chart data, a prior interview, a
    historical connection) that goes beyond what the primary coverage
    already reported. If no Additional Research Notes were used (the
    block is absent, or present but nothing from it was actually
    used), omit the section entirely -- do not add a sources list just
    to have one, and never list the primary outlets in it. When the
    section is included, end content_html with a final <h2>Sources &
    Further Reading</h2> section followed by a <ul> list, with one
    <li> per complementary reference actually drawn upon from the
    Additional Research Notes block (nothing else). Each <li> must be
    an <a href> using the EXACT URL cited in Additional Research Notes
    for that item -- never invent, guess, shorten, or paraphrase a
    URL. Label each link with the outlet or publication's name, e.g.
    <li><a href="https://www.rollingstone.com/...">Rolling Stone</a></li>.

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
    "source_item_indices": [1, 2],
    "tags": ["exactly 8 to 10 relevant tags, e.g. band names, genres, related artists, subgenres"],
    "excerpt": "string, 1-2 sentence factual teaser, under 200 chars",
    "content_html": "string, the full article body as clean HTML using <p>, <h2>, <h3> tags where natural, ending with the Sources & Further Reading section ONLY if applicable, per rule 4g (most articles will NOT have one -- see that rule). Do NOT include an <h1> (WordPress adds the title separately). LENGTH: when full article text was retrieved for the sources (not just RSS summaries), extract and include the genuinely reported facts, direct quotes, and specific details actually present in that full text -- aim for 600-900 words in that case (up to ~1100 when Additional Research Notes materially added new grounded facts per rule 2b), matching the depth of a real news report rather than a condensed summary of one. When only RSS summaries are available (no full text) and no Additional Research Notes were usable, 300-500 words is appropriate since there's less real material to draw from -- don't pad with speculation just to hit a length target either way. The test is always: does the length match how much genuine, verifiable source material exists, not a fixed target."
    ,
    "illustrative_images": [
      {{
        "query": "string, a SPECIFIC multi-word Wikimedia Commons search naming exactly what the image should show -- e.g. 'Similitude of a Dream album cover Neal Morse Band', not just 'Neal Morse Band' or 'Similitude of a Dream'. A bare band/artist name is too ambiguous (it can match an unrelated same-named subject, like an actual eagle for the band Eagles) -- always combine the specific thing (album, venue, event, person) with enough context words to disambiguate it.",
        "placement_after_heading": "string, the exact text of the <h2> heading in content_html after which this image should be inserted, or \\"\\" to place it roughly in the middle of the article if there's no natural heading to anchor to",
        "caption": "string, a short factual caption for the image, e.g. the album title and artist, or what's shown"
      }}
    ],
    "video_embeds": [
      {{
        "url": "string, one of the exact video URLs listed under a source's \\"Videos embedded in this source\\" block above -- never invent or guess a video URL",
        "placement_after_heading": "string, the exact text of the <h2> heading in content_html after which this video should be embedded -- match it to whichever heading in YOUR article covers the same item/performance the video is of"
      }}
    ],
    "source_images": [
      {{
        "url": "string, one of the exact image URLs listed under a source's \\"Other images in this source's article body\\" block above -- never invent or guess a URL, and never reuse the same URL already used as the featured image",
        "placement_after_heading": "string, the exact text of the <h2> heading in content_html after which this image should be inserted",
        "caption": "string, a short factual caption for the image"
      }}
    ]
  }}
]
"""

RELEVANCE_SYSTEM_PROMPT = """You are a rock music editor triaging news headlines for Deadikace, a classic and mainstream rock blog.
CORE ROCK GENRES in scope: rock, rock and roll, classic rock, hard rock, heavy rock, blues rock, folk rock, progressive rock, psychedelic rock, southern rock, garage rock, alternative rock, indie rock, punk rock, glam rock, roots rock, country rock, funk rock, art rock, grunge, post-punk, and closely related rock styles, plus general rock culture news for artists working primarily in these styles.
METAL SUBGENRES out of scope: death metal, black metal, doom metal, sludge metal, thrash metal, power metal, symphonic metal, progressive metal, metalcore, deathcore, djent, nu-metal, and other extreme metal-scene subgenres, releases, or festivals.
CLASSIFY THE STORY, NOT JUST THE ARTIST. Crossover artists like Iron Maiden, Ozzy Osbourne, Metallica, and Black Sabbath have both a metal identity and broad classic or hard-rock relevance. Decide based on what the headline is actually about, not the artist name alone. A story passes only if its primary subject is genuinely rock-relevant, and should be excluded if its primary subject is metal-scene specific even if it mentions a crossover artist.
Example: a broad arena-tour or classic-rock retrospective story about a crossover artist should be kept. A story whose primary subject is a niche metal-scene release, festival, or band that only mentions a crossover artist in passing should be excluded.
Also exclude headlines primarily about other unrelated genres (pop, rap/hip-hop, K-pop, R&B, country, EDM/dance) or unrelated topics (general lifestyle, cars, celebrities outside music), even if a rock artist is only mentioned in passing. When in doubt between excluding and keeping a genuinely rock-focused headline, lean toward keeping it. When in doubt about whether a story's primary subject is metal-scene content rather than mainstream/classic/hard rock, lean toward excluding it. Respond with ONLY a JSON array of the relevant headline numbers, for example a two-item array like [1,3]. No other text, no explanation.
"""
RESEARCH_SYSTEM_PROMPT = """You are a research assistant for Deadikace, a
rock music blog, helping a staff writer add real depth to a news article
before it's written -- not writing the article itself.

You will be given a news topic and the competitor RSS/article coverage
already gathered for it. Use web search to find ADDITIONAL, VERIFIABLE
factual grounding a knowledgeable editor would want before writing a
deeper, more original piece than a straight recap of that competitor
coverage. Good things to look for, when genuinely relevant to this
specific story:
- Discography/catalog context (album/song release history, prior
  related releases or events).
- Chart positions, certifications (RIAA/BPI/etc.), award history --
  always from an identifiable, checkable source (Billboard, RIAA,
  Grammy.com, Wikipedia citing one of those, etc.), never a vague
  "reportedly."
- Documented prior statements by the people involved (interviews,
  books, official band history) that add context to the current story,
  as long as you can point to where they were actually said/published.
- Relevant, well-established historical background that helps a reader
  understand why the current story matters.

HARD RULES:
1. Only include a fact if you can cite the specific URL where you found
   it. Never state something as fact based on general impression without
   a source you can point to.
2. Do not speculate, estimate, or round up from partial information.
   If a number or date is unclear or contested across sources, say so
   explicitly rather than picking one version.
3. Do not try to write article prose, headlines, or ledes -- this is a
   research brief, not a draft. Plain factual notes only.
4. It is completely fine, and often correct, to return few findings or
   none at all -- not every news story has meaningful additional
   background to add. Do not pad the brief with generic trivia just to
   have something to report.
5. Stay on-topic: only research things that would plausibly help THIS
   specific story, not general trivia about the artist/band that isn't
   connected to it.

OUTPUT FORMAT: a short plain-text bulleted list. Each bullet is one
verifiable fact, followed by its source in parentheses, e.g.:
- "Hotel California" won the Grammy for Record of the Year in 1978 and
  is RIAA-certified 28x Platinum (source: https://www.grammy.com/... )

If nothing genuinely useful and verifiable was found, respond with
exactly the single word: NONE
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

def _call_anthropic_with_search(system_prompt, user_prompt, max_tokens, max_searches):
    """
    Like _call_anthropic, but grants Claude a live, hosted web-search
    tool for the duration of this one call, so it can look up real facts
    instead of answering from memory alone. Returns the concatenation of
    every plain-text block in the response, in order -- tool-use/tool-
    result blocks (the searches themselves and their raw results) are
    skipped, since we only want Claude's own written notes, not the raw
    search payloads.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": max_searches,
        }],
    )
    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "\n".join(text_parts).strip()

def _call_gemini_with_search(system_prompt, user_prompt, max_tokens, max_searches):
    """
    Like _call_gemini, but grants Gemini its built-in Google Search
    grounding tool for this one call. Google Search grounding and
    forced JSON output aren't a reliable combination, so this always
    returns plain text (the research brief format), never JSON --
    unlike _call_gemini, which is only ever used for JSON-schema calls.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return (response.text or "").strip()

def _call_llm_with_search(system_prompt, user_prompt, max_tokens, max_searches):
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set.")
        return _call_anthropic_with_search(system_prompt, user_prompt, max_tokens, max_searches)
    elif LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set.")
        return _call_gemini_with_search(system_prompt, user_prompt, max_tokens, max_searches)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} (expected 'anthropic' or 'gemini')")

def research_additional_context(topic):
    """
    Runs a live web-search research pass for a topic that's about to be
    drafted, looking for verifiable factual grounding beyond the
    competitor RSS coverage already gathered (see article_fetch.py) --
    catalog/chart data, documented prior statements, established
    historical background, each tied to a real source URL. This is what
    lets the drafting step (draft_article(), via DRAFT_SYSTEM_PROMPT
    rule 2b) add genuine depth instead of just re-synthesizing the same
    handful of competitor articles in different words.

    Mutates topic in place, setting topic["research_notes"] to the
    research brief text, or None if research is disabled, found nothing
    usable, or the call failed for any reason. Never raises -- an
    optional enhancement failing shouldn't fail the whole run, matching
    the defensive pattern used throughout this module (see
    filter_rock_relevant_topics, filter_duplicate_topics, and
    verify_and_refine for the same fallback approach.

    Note this is a separate step from image sourcing (article_fetch.py /
    wikimedia.py / main.py's _get_featured_media_for_topic and
    _insert_*_images helpers), which is untouched by this function and
    continues to work exactly as before.
    """
    if not ENABLE_DEEP_RESEARCH:
        topic["research_notes"] = None
        return topic

    source_material, _ = _build_source_material(topic)
    headline = topic["items"][0]["title"]
    user_prompt = (
        f"Topic: {headline}\n\n"
        f"Competitor coverage already gathered:\n\n{source_material}\n\n"
        "Research additional verifiable context for this story now, following "
        "the rules above. Respond with the plain-text bulleted brief (or NONE), "
        "nothing else."
    )

    try:
        raw = _call_llm_with_search(
            RESEARCH_SYSTEM_PROMPT, user_prompt,
            max_tokens=2000, max_searches=RESEARCH_MAX_SEARCHES,
        )
    except Exception as e:
        print(f"[warn] Additional-research pass failed ({e}); drafting from "
              f"competitor coverage only, same as before this feature existed.")
        topic["research_notes"] = None
        return topic

    if not raw or raw.strip().upper() == "NONE":
        print("[info] Research pass found nothing additional worth adding for this topic.")
        topic["research_notes"] = None
    else:
        print(f"[info] Research pass found additional grounding for this topic "
              f"({len(raw.split(chr(10)))} line(s)).")
        topic["research_notes"] = raw

    return topic

DUPLICATE_SYSTEM_PROMPT = """You are an editor for Deadikace, a rock music
blog, checking new candidate news topics against a list of the blog's own
recent posts (title + a short excerpt of each).

For each numbered candidate topic below, decide whether it is about the
SAME underlying news story/event as any of the existing posts, even if
the headline wording is completely different, the story was covered by
a different outlet, or it was picked up in a separate run of this same
pipeline hours apart (e.g. two differently-worded headlines about the
same interview, announcement, or event are still the same story). Use
the excerpt, not just the title, to judge this -- headlines for the same
story are often worded very differently by different outlets, but the
actual content described will match.

If you are genuinely unsure whether a candidate is the same story as an
existing post, treat it as a duplicate and skip it -- publishing a
near-duplicate is a worse outcome than missing one topic that had
another angle, since the missed topic may well resurface as its own
distinct story later.

Respond with ONLY a JSON array of the candidate numbers that ARE
duplicates of an existing recent post (i.e. should be skipped), e.g.
[2,5]. Respond with [] if none are duplicates. No other text.
"""

def filter_duplicate_topics(topics, existing_posts):
    """
    Removes candidate topics that are about the same underlying story as
    a recently-published Deadikace post, even when the wording is
    completely different or the story was picked up from a different
    outlet, or in a separate run of this pipeline hours apart.

    existing_posts: list of {"title": ..., "excerpt": ...} dicts (see
    wordpress.get_recent_posts_for_dedup) -- the excerpt is what lets
    this catch same-story-different-headline cases that a title-only
    comparison would miss.
    """
    if not topics or not existing_posts:
        return topics

    def _topic_snippet(t):
        first = t["items"][0]
        snippet = (first.get("full_text") or first.get("summary") or "")
        snippet = " ".join(snippet.split())[:200]
        return snippet

    candidate_list = "\n".join(
        f"{i + 1}. " + " / ".join(item["title"] for item in t["items"][:3])
        + (f" -- {_topic_snippet(t)}" if _topic_snippet(t) else "")
        for i, t in enumerate(topics)
    )
    existing_list = "\n".join(
        f"- {p['title']}" + (f" -- {p['excerpt']}" if p.get("excerpt") else "")
        for p in existing_posts
    )
    user_prompt = (
        f"Candidate topics:\n{candidate_list}\n\n"
        f"Existing recent posts (title -- excerpt):\n{existing_list}\n\n"
        "Return the JSON array of duplicate candidate numbers now."
    )

    try:
        raw = _call_llm(DUPLICATE_SYSTEM_PROMPT, user_prompt, max_tokens=1000)
        raw = _clean_json_text(raw)
        dup_idx = {int(i) - 1 for i in json.loads(raw)}
    except Exception as e:
        print(f"[warn] Duplicate-vs-existing-posts check failed ({e}); proceeding without it.")
        return topics

    for i in sorted(dup_idx):
        if 0 <= i < len(topics):
            print(f"[info] Skipping likely duplicate of an existing post: "
                  f"{topics[i]['items'][0]['title']}")

    return [t for i, t in enumerate(topics) if i not in dup_idx]

DRAFTPY_FALLBACK_MARKER_START
_METAL_SUBGENRE_FALLBACK_TERMS = (
   "death metal", "black metal", "doom metal", "sludge metal",
   "thrash metal", "power metal", "symphonic metal", "progressive metal",
   "metalcore", "deathcore", "djent", "nu-metal", "nu metal",
)
def _fallback_exclude_obvious_metal(topics):
   """
   Deterministic fallback used only when the LLM relevance call itself
   fails. Excludes a headline only if it contains an unambiguous metal
   subgenre keyword; everything else (including borderline or ambiguous
   headlines) is kept, matching the "when in doubt, keep it" bias of the
   real LLM-based filter.
   """
   kept = []
   dropped = 0
   for t in topics:
      title = t["items"][0]["title"].lower()
      if any(term in title for term in _METAL_SUBGENRE_FALLBACK_TERMS):
         dropped += 1
         continue
      kept.append(t)
   print(f"[warn] Fallback keyword filter kept {len(kept)} of {len(topics)} "
         f"topics ({dropped} dropped for obvious metal-subgenre keywords).")
   return kept


def filter_rock_relevant_topics(topics):
    """
    Sends just the headlines (one batched call, not one per topic) to the
    configured LLM and keeps only genuinely rock-relevant ones. If the
    classification call fails, falls back to a deterministic keyword
    filter (see _fallback_exclude_obvious_metal) that drops only obvious
    metal-subgenre headlines and keeps everything else if ambiguous.
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
        print(f"[warn] Rock-relevance filtering failed ({e}); falling back to "
              f"keyword-based metal filter.")
        return _fallback_exclude_obvious_metal(topics)

    filtered = [t for i, t in enumerate(topics) if i in keep_idx]
    print(f"[info] Rock-relevance filter kept {len(filtered)} of {len(topics)} topics.")
    return filtered

WITHIN_BATCH_DUPLICATE_SYSTEM_PROMPT = """You are an editor for Deadikace,
a rock music blog, checking a batch of candidate news topics against EACH
OTHER (not against older posts) to catch cases where the same underlying
story was picked up by the topic clustering step as two or more separate
topics -- e.g. two different outlets covering the same news with
different headlines, or one leads with "artist reveals health scare",
another leads with a specific quote from the same interview -- that
simple keyword-overlap clustering treats them as unrelated.

For each numbered candidate topic below, decide whether it covers the
SAME underlying news story/event as any OTHER candidate in this same
list. Use the short snippet given after each title, not just the title
-- headlines for the same story are often worded completely differently,
but the actual content described will match. If you are genuinely
unsure whether two candidates are the same story, treat them as
duplicates -- publishing two near-identical articles in the same run is
a worse outcome than merging two topics that turn out to have been
slightly different angles on the same news.

When you find a group of duplicates, keep only the one with the most
detail (prefer the one with a longer snippet / more source items) and
mark the others as duplicates to remove.

Respond with ONLY a JSON array of the candidate numbers to REMOVE (i.e.
the duplicates, not the one being kept), e.g. [2,5]. Respond with []
if there are no duplicates. No other text.
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

    def _within_batch_snippet(t):
        first = t["items"][0]
        snippet = (first.get("full_text") or first.get("summary") or "")
        snippet = " ".join(snippet.split())[:200]
        return snippet

    candidate_list = "\n".join(
        f"{i + 1}. " + " / ".join(item["title"] for item in t["items"][:3])
        + (f" -- {_within_batch_snippet(t)}" if _within_batch_snippet(t) else "")
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
    for i, item in enumerate(topic["items"], start=1):
        full_text = item.get("full_text")
        video_embeds = item.get("video_embeds") or []
        video_lines = ""
        if video_embeds:
            video_list = "\n".join(
                f"  - {v['url']}" + (f" (near heading: \"{v['heading']}\")" if v.get("heading") else "")
                for v in video_embeds
            )
            video_lines = f"\nVideos embedded in this source:\n{video_list}"
        body_images = item.get("body_images") or []
        image_lines = ""
        if body_images:
            image_list = "\n".join(
                f"  - {img['url']}"
                + (f" (alt text: \"{img['alt']}\")" if img.get("alt") else "")
                + (f" (near heading: \"{img['heading']}\")" if img.get("heading") else "")
                for img in body_images
            )
            image_lines = f"\nOther images in this source's article body:\n{image_list}"
        if full_text:
            full_text_count += 1
            source_blocks.append(
                f"Source #{i}: {item['source']} (FULL ARTICLE TEXT)\n"
                f"Title: {item['title']}\n"
                f"Full text: {full_text}\n"
                f"Link: {item['link']}"
                f"{video_lines}"
                f"{image_lines}"
            )
        else:
            source_blocks.append(
                f"Source #{i}: {item['source']} (RSS summary only)\n"
                f"Title: {item['title']}\n"
                f"Summary: {item['summary']}\n"
                f"Link: {item['link']}"
                f"{video_lines}"
                f"{image_lines}"
            )

    research_notes = topic.get("research_notes")
    if research_notes:
        source_blocks.append(
            "Additional Research Notes (independently gathered via live web "
            "search, specifically for this topic -- treat exactly like the "
            "numbered Source blocks above: every fact must be verifiable and "
            "attributed to the URL given, never invent beyond what's stated "
            "here; see DRAFT_SYSTEM_PROMPT rule 2b for how to use this):\n"
            f"{research_notes}"
        )

    return "\n\n".join(source_blocks), full_text_count

def draft_article(topic):
    """topic: one cluster dict from discover.get_trending_topics(), ideally
    already enriched with full_text via article_fetch.enrich_topic_with_full_text
    and, when available, research notes via research_additional_context().

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

    raw_text = _call_llm(DRAFT_SYSTEM_PROMPT, user_prompt, max_tokens=8000)
    raw_text = _clean_json_text(raw_text)

    try:
        articles = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{LLM_PROVIDER} did not return valid JSON: {e}\nRaw output:\n{raw_text}")

    if isinstance(articles, dict):
        articles = [articles]

    if not isinstance(articles, list) or not articles:
        raise ValueError(f"{LLM_PROVIDER} did not return a non-empty JSON array: {raw_text}")

    # Defensive default: if the model omitted source_item_indices (or the
    # whole array has just one article covering everything), fall back to
    # "all source items belong to this article" rather than none, so
    # image sourcing still has something to work with.
    for article in articles:
        if not article.get("source_item_indices"):
            article["source_item_indices"] = list(range(1, len(topic["items"]) + 1))

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
- If a Sources & Further Reading section is present, confirm every link
  in it corresponds to an actual URL given in the source material or
  research notes -- flag and remove any link that doesn't (an invented
  or mismatched source link is exactly the kind of fabricated-but-
  plausible detail this pass exists to catch).
- If a Sources & Further Reading section is present, flag and remove
  any entry that links to one of the primary/numbered source outlets
  (e.g. Louder Sound, Loudwire, Rolling Stone) rather than to
  Additional Research Notes material -- that section exists only for
  complementary research, never for the primary coverage. If, after
  removing those, the section would be empty, delete the entire
  <h2>Sources & Further Reading</h2> heading and list rather than
  leaving an empty or primary-sources-only section.

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
    material (including any Additional Research Notes) and tightens up
    any embellishment/over-interpretation before publishing. Falls back
    to the original article unchanged if the verification call fails.
    """
    source_material, _ = _build_source_material(topic)

    user_prompt = (
        f"Source material:\n\n{source_material}\n\n"
        f"Draft article content_html:\n\n{article['content_html']}\n\n"
        "Fact-check and return the corrected JSON now."
    )

    try:
        raw_text = _call_llm(VERIFY_SYSTEM_PROMPT, user_prompt, max_tokens=8000)
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


