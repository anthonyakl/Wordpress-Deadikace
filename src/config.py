"""
Central configuration for the Deadikace auto-publishing agent.

Secrets (API keys, WP credentials) are read from environment variables --
never hardcode them here. See .env.example / GitHub Secrets setup in README.
"""

import os


def _env(key, default=""):
    val = os.environ.get(key)
    return val if val not in (None, "") else default


def _env_int(key, default):
    val = _env(key, "")
    if val == "":
        return default
    try:
        return int(val)
    except ValueError:
        print(f"[warn] Env var {key}={val!r} is not a valid integer; using default {default}.")
        return default


def _env_bool(key, default):
    val = _env(key, "")
    if val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_list(key, default_csv):
    val = _env(key, default_csv)
    return [item.strip() for item in val.split(",") if item.strip()]


# --- WordPress site ---
WP_BASE_URL = _env("WP_BASE_URL", "https://www.deadikace.com")
WP_USERNAME = os.environ["WP_USERNAME"]
WP_APP_PASSWORD = os.environ["WP_APP_PASSWORD"]

# --- LLM provider selection ---
LLM_PROVIDER = _env("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
CLAUDE_MODEL = _env("CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-flash-lite-latest")

# --- Behavior ---
POST_STATUS = _env("POST_STATUS", "publish")
MAX_ARTICLES_PER_RUN = _env_int("MAX_ARTICLES_PER_RUN", 2)
ENABLE_SOURCE_IMAGES = _env_bool("ENABLE_SOURCE_IMAGES", True)
ENABLE_DEEP_RESEARCH = _env_bool("ENABLE_DEEP_RESEARCH", True)
RESEARCH_MAX_SEARCHES = _env_int("RESEARCH_MAX_SEARCHES", 6)

# A single-source feed can still surface a valid story, but it must pass the
# hard editorial genre gate and the later originality/value checks. Cross-
# outlet coverage remains a ranking signal rather than a requirement.
MIN_SOURCE_COUNT = _env_int("MIN_SOURCE_COUNT", 1)
LOOKBACK_HOURS = _env_int("LOOKBACK_HOURS", 48)

# --- Topic ranking ---
SOURCE_COUNT_WEIGHT = _env_int("SOURCE_COUNT_WEIGHT", 10)
PRIORITY_KEYWORDS = _env_list("PRIORITY_KEYWORDS", "")
PRIORITY_KEYWORD_BONUS = _env_int("PRIORITY_KEYWORD_BONUS", 15)

# Fast deterministic exclusion before clustering. This is deliberately broad:
# the LLM classifier remains the semantic gate, but an obvious metal-only story
# should never reach drafting merely because an LLM call failed or a headline is
# ambiguous. Keep famous rock/metal crossover artists out of this keyword list;
# their story-level classification is handled separately.
EXCLUDE_KEYWORDS = _env_list(
    "EXCLUDE_KEYWORDS",
    "rap,hip-hop,hip hop,k-pop,kpop,r&b,trap,reggaeton,boy band,girl group,country music,"
    "death metal,black metal,doom metal,sludge metal,thrash metal,power metal,symphonic metal,"
    "progressive metal,metalcore,deathcore,djent,nu-metal,nu metal,technical death metal,"
    "brutal death metal,melodic death metal,funeral doom,folk metal,viking metal,"
    "industrial metal,groove metal,metal festival,metal scene"
)

# Sources that are overwhelmingly metal-oriented are not allowed to be the sole
# reason a story is discovered. They can still contribute to a story when other
# trusted rock-oriented sources independently cover it.
METAL_HEAVY_SOURCE_NAMES = _env_list("METAL_HEAVY_SOURCE_NAMES", "Loudwire")

# Explicit rock-family taxonomy used by prompts and deterministic safeguards.
ROCK_GENRES = _env_list(
    "ROCK_GENRES",
    "rock,rock and roll,classic rock,hard rock,heavy rock,blues rock,folk rock,"
    "progressive rock,psychedelic rock,southern rock,garage rock,alternative rock,"
    "indie rock,punk rock,glam rock,roots rock,country rock,funk rock,art rock,"
    "grunge,post-punk,post-rock"
)

METAL_GENRES = _env_list(
    "METAL_GENRES",
    "metal,heavy metal,death metal,black metal,doom metal,sludge metal,thrash metal,"
    "power metal,symphonic metal,progressive metal,metalcore,deathcore,djent,nu-metal,"
    "industrial metal,groove metal,technical death metal,melodic death metal"
)

# Existing WP content to compare against. Drafts matter just as much as published
# posts because the workflow normally creates drafts for review and can be run again.
DEDUP_LOOKBACK_DAYS = _env_int("DEDUP_LOOKBACK_DAYS", 30)
DEDUP_MAX_POSTS = _env_int("DEDUP_MAX_POSTS", 150)

# A candidate should not be turned into a substantial article unless the research
# pass finds enough genuinely new/contextual material. The exact scoring is done by
# the editorial prompt; these thresholds make the policy explicit and tunable.
MIN_EDITORIAL_VALUE_SCORE = _env_int("MIN_EDITORIAL_VALUE_SCORE", 5)
MIN_RESEARCH_VALUE_SCORE = _env_int("MIN_RESEARCH_VALUE_SCORE", 3)

TARGET_CATEGORY = _env("TARGET_CATEGORY", "Latest News")
LATEST_POSTS_COUNT = _env_int("LATEST_POSTS_COUNT", 5)
ARTICLE_FONT_SIZE_PX = _env_int("ARTICLE_FONT_SIZE_PX", 22)

# --- Competitor RSS feeds ---
COMPETITOR_FEEDS = [
    {"name": "Rolling Stone", "url": "https://www.rollingstone.com/music/music-news/feed/"},
    {"name": "Louder Sound", "url": "https://www.loudersound.com/feeds.xml"},
    {"name": "Ultimate Classic Rock", "url": "https://ultimateclassicrock.com/feed/"},
    {"name": "Loudwire", "url": "https://loudwire.com/feed/"},
    {"name": "Far Out Magazine", "url": "https://faroutmagazine.co.uk/articles/music/music-news/feed/"},
    {"name": "Rock Cellar Magazine", "url": "https://rockcellarmagazine.com/category/latest-news/feed/"},
]

SITE_VOICE_GUIDELINES = """
Deadikace is a rock music blog for passionate, knowledgeable fans.
Voice: enthusiastic but credible, conversational, never corporate or generic.
Assume the reader already loves rock music -- don't over-explain basics.
The goal is not merely to summarize a source. Add genuinely useful editorial
context, connections to relevant related developments, historical or career
context, and analysis when the evidence supports it. A safe, non-controversial
editorial observation is acceptable occasionally, but never invent an opinion,
experience, reporting, quote, or consensus. Do not add filler to reach a word count.
"""

YOAST_TITLE_FIELD = "_yoast_wpseo_title"
YOAST_META_DESC_FIELD = "_yoast_wpseo_metadesc"
YOAST_FOCUS_KEYWORD_FIELD = "_yoast_wpseo_focuskw"
