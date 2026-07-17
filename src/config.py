"""
Central configuration for the Deadikace auto-publishing agent.
Secrets (API keys, WP credentials) are read from environment variables —
never hardcode them here. See .env.example / GitHub Secrets setup in README.
"""

import os

# --- WordPress site ---
WP_BASE_URL = os.environ.get("WP_BASE_URL", "https://www.deadikace.com")
WP_USERNAME = os.environ["WP_USERNAME"]          # your WP username
WP_APP_PASSWORD = os.environ["WP_APP_PASSWORD"]  # WP Application Password (not your login password)

# --- LLM provider selection ---
# "anthropic" (Claude, paid) or "gemini" (Google, has a free tier)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

# --- Anthropic (Claude) API --- only required if LLM_PROVIDER == "anthropic"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# --- Google Gemini API --- only required if LLM_PROVIDER == "gemini"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Behavior ---
# "publish" = goes live immediately. "draft" = saved in WP for you to review first.
# Recommended: start with "draft" for a week or two, then switch to "publish".
POST_STATUS = os.environ.get("POST_STATUS", "publish")  # "publish" or "draft"

# Max number of new articles to generate per run
MAX_ARTICLES_PER_RUN = int(os.environ.get("MAX_ARTICLES_PER_RUN", "2"))

# A story must appear in at least this many competitor feeds to be
# considered "trending" enough to write about. Set to 1 to write about
# anything, 2+ to be more selective and reduce noise.
MIN_SOURCE_COUNT = int(os.environ.get("MIN_SOURCE_COUNT", "1"))

# How far back (in hours) to look at competitor feeds for "recent" news
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "48"))

# --- Competitor RSS feeds ---
# These are used ONLY to detect trending topics (titles/summaries/links).
# We never scrape or reproduce full competitor article text.
COMPETITOR_FEEDS = [
    {"name": "Rolling Stone", "url": "https://www.rollingstone.com/music/music-news/feed/"},
    {"name": "Louder Sound", "url": "https://www.loudersound.com/feeds.xml"},
    {"name": "Ultimate Classic Rock", "url": "https://ultimateclassicrock.com/feed/"},
    {"name": "Loudwire", "url": "https://loudwire.com/feed/"},
]

# --- Site voice / editorial guidelines given to Claude ---
SITE_VOICE_GUIDELINES = """
Deadikace is a rock music blog for passionate, knowledgeable fans.
Voice: enthusiastic but credible, conversational, avoids corporate/listicle
cliches ("in today's fast-paced music world..."). Assume the reader already
loves rock music -- don't over-explain basics. Add genuine perspective or
context, not just a rehash of the news.
"""

# --- Yoast SEO field names (WP REST API) ---
# These are exposed by the Yoast SEO plugin's REST support once enabled.
YOAST_TITLE_FIELD = "_yoast_wpseo_title"
YOAST_META_DESC_FIELD = "_yoast_wpseo_metadesc"
YOAST_FOCUS_KEYWORD_FIELD = "_yoast_wpseo_focuskw"
