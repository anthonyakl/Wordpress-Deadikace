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
# "gemini-flash-lite-latest" is Google's auto-updating alias for their
# current lite model -- it has the most generous free-tier quota (as of
# writing: 15 requests/min, 1,000/day) and avoids breaking again when
# Google retires a specific dated model version. Override via the
# GEMINI_MODEL secret if you want a specific model instead.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

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

# --- WordPress category every agent-generated post is filed under ---
# Set to exactly match one of your existing category names (case-insensitive).
# The agent will create it if it somehow doesn't exist yet, but it's meant
# to match one you already have (e.g. "Latest News").
TARGET_CATEGORY = os.environ.get("TARGET_CATEGORY", "Latest News")

# --- Images ---
# Pexels (https://www.pexels.com/api/) has a generous free tier and a
# straightforward API -- used to find royalty-free, commercially-usable
# stock photos relevant to each article. Get a free key at
# https://www.pexels.com/api/
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

# How many images to request per article (in addition to the featured
# image). The model is instructed to include this many section images
# scaled to article length -- this is an upper bound.
MAX_IMAGES_PER_ARTICLE = int(os.environ.get("MAX_IMAGES_PER_ARTICLE", "4"))

# --- "Latest posts" block appended to the end of every article ---
LATEST_POSTS_COUNT = int(os.environ.get("LATEST_POSTS_COUNT", "5"))

# --- Article body font size ---
# Many WP themes default post-body text to a small size. This wraps each
# article's content in a container with an explicit font size so it reads
# comfortably regardless of the theme's default. Adjust if it looks too
# big/small on your theme.
ARTICLE_FONT_SIZE_PX = int(os.environ.get("ARTICLE_FONT_SIZE_PX", "18"))

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
