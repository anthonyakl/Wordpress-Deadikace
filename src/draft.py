"""
Uses the Claude API to draft an original, SEO-optimized article based on a
trending topic. Only titles/summaries/links from competitor RSS feeds are
passed in as factual signals -- Claude is explicitly instructed to write
original analysis, not to paraphrase or closely follow any single source.
"""

import json
import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, SITE_VOICE_GUIDELINES

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = f"""You are a staff writer for Deadikace, a rock music blog.

{SITE_VOICE_GUIDELINES}

You will be given a news topic and short summaries of how several outlets
covered it (titles, snippets, and links -- not full articles). Your job:

1. Write a wholly ORIGINAL article about the same underlying news/topic.
   Do not paraphrase, closely follow the structure of, or lift phrasing
   from any of the provided source summaries. Treat them only as pointers
   to the underlying facts and as a way to know this story is currently
   newsworthy.
2. Add genuine value: context, a clear angle or opinion, connections to
   the band/artist's history, or why this matters to rock fans -- not
   just a restatement of "X happened."
3. If specific facts (dates, quotes, numbers) are needed, only use ones
   present in the provided summaries, and attribute them naturally
   (e.g., "according to a statement shared with outlets this week").
   Do not invent quotes or facts.
4. Output must be valid JSON matching this exact schema, nothing else:

{{
  "title": "string, compelling but not clickbait, under 70 chars",
  "seo_title": "string, SEO title tag, under 60 chars, includes primary keyword",
  "meta_description": "string, under 155 chars, includes primary keyword, makes people want to click",
  "focus_keyword": "string, 2-4 word primary SEO keyword phrase for this article",
  "tags": ["3-6 relevant tags, e.g. band names, genres"],
  "excerpt": "string, 1-2 sentence teaser, under 200 chars",
  "content_html": "string, the full article body as clean HTML using <p>, <h2>, <h3> tags where natural. 500-800 words. Do NOT include an <h1> (WordPress adds the title separately)."
}}
"""


def draft_article(topic):
    """topic: one cluster dict from discover.get_trending_topics()"""
    source_summaries = "\n\n".join(
        f"Source: {item['source']}\nTitle: {item['title']}\nSummary: {item['summary']}\nLink: {item['link']}"
        for item in topic["items"]
    )

    user_prompt = f"""Topic covered by {topic['source_count']} outlet(s):

{source_summaries}

Write the original Deadikace article now. Respond with ONLY the JSON object, no other text."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = response.content[0].text.strip()

    # Defensive cleanup in case the model wraps output in markdown fences
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        article = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude did not return valid JSON: {e}\nRaw output:\n{raw_text}")

    return article
