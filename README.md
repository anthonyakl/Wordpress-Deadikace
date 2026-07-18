# Deadikace Auto-Publish Agent

Watches rock-music news via competitor RSS feeds (Rolling Stone, Louder
Sound, Ultimate Classic Rock, Loudwire), detects trending stories, and uses
Claude to draft **original, SEO-optimized** articles that get published to
your WordPress blog automatically via GitHub Actions.

**Important note on originality:** this agent is designed to use competitor
coverage only as a *signal for what's currently newsworthy* (titles/links/
summaries from RSS — the same public info an RSS reader shows). Claude is
explicitly instructed to write wholly original analysis, not to paraphrase
or mirror any single source's structure or wording. This matters both
legally (avoiding copyright issues) and for SEO (Google penalizes
duplicate/spun content). Please read the generated articles for the first
week or two before trusting it fully unattended — see Step 6.

---

## 1. Get the code onto GitHub

1. Go to https://github.com/new and create a new **private** repository
   (e.g. `deadikace-agent`).
2. Download all the files I've given you in this chat, preserving the folder
   structure:
   ```
   deadikace-agent/
     .github/workflows/publish.yml
     src/config.py
     src/discover.py
     src/draft.py
     src/wordpress.py
     src/main.py
     requirements.txt
     README.md
   ```
3. On the new repo's GitHub page, click **"uploading an existing file"**
   and drag the whole folder in (GitHub preserves the folder structure),
   or if you're comfortable with git:
   ```bash
   git init
   git add .
   git commit -m "Initial agent setup"
   git remote add origin https://github.com/YOUR_USERNAME/deadikace-agent.git
   git push -u origin main
   ```

## 2. Create a WordPress Application Password

This lets the agent post to your site without using your real login
password.

1. Log into `www.deadikace.com/wp-admin`
2. Go to **Users → Profile** (your own user)
3. Scroll to **Application Passwords**
4. Enter a name like `github-agent` and click **Add New Application Password**
5. Copy the generated password immediately (spaces included) — you won't
   see it again.

> If you don't see this section: this is a known issue on Hostinger and a
> few other hosts. Fix it by creating a file at
> `wp-content/mu-plugins/force-app-passwords.php` (via File Manager or
> FTP, creating the `mu-plugins` folder if needed) containing:
> ```php
> <?php
> add_filter('wp_is_application_passwords_available', '__return_true');
> ```
> Files in `mu-plugins` load automatically, no activation needed. Refresh
> Users → Profile afterward. If it's still missing, check for a security
> plugin (Wordfence, iThemes Security) with a toggle hiding it.

## 3. Expose Yoast SEO fields to the REST API (one-time)

By default, Yoast's SEO title/meta description fields aren't writable via
the REST API. Add this snippet via a small custom plugin, or your theme's
`functions.php` (a child theme is safest):

```php
add_action('init', function () {
    register_meta('post', '_yoast_wpseo_title', [
        'show_in_rest' => true, 'single' => true, 'type' => 'string',
        'auth_callback' => '__return_true',
    ]);
    register_meta('post', '_yoast_wpseo_metadesc', [
        'show_in_rest' => true, 'single' => true, 'type' => 'string',
        'auth_callback' => '__return_true',
    ]);
    register_meta('post', '_yoast_wpseo_focuskw', [
        'show_in_rest' => true, 'single' => true, 'type' => 'string',
        'auth_callback' => '__return_true',
    ]);
});
```

(I can generate this as a tiny standalone plugin zip for you if you'd
rather not touch `functions.php` — just ask.)

## 4. Get an LLM API key (choose Claude or Gemini)

The agent supports either provider — pick one via the `LLM_PROVIDER` secret
in Step 5.

**Option A — Claude (paid, pay-as-you-go):**
1. Go to https://console.anthropic.com
2. Create an API key under **Settings → API Keys**
3. Billed separately from any Claude.ai subscription. Drafting 2 articles
   a day is a small cost (roughly a few dollars a month), but check
   current pricing at https://docs.claude.com.

**Option B — Gemini (has a free tier):**
1. Go to https://aistudio.google.com/apikey and create an API key
2. Before relying on the free tier long-term for a commercial site, read
   Google's current terms yourself at https://ai.google.dev/gemini-api/terms
   — free-tier usage terms (commercial use, data-training policy) can
   change, and it's worth confirming they fit your situation.
3. The free tier is generous enough for 2 articles/day with plenty of
   headroom (roughly 1,500 requests/day as of writing, but check
   https://ai.google.dev/gemini-api/docs/rate-limits for current numbers).

**Never paste an API key into a chat with an AI assistant, including this
one — add it directly into GitHub Secrets yourself in the next step.**

## 5. Add your secrets to GitHub

In your repo: **Settings → Secrets and variables → Actions → New repository secret**.
Add each of these:

| Secret name          | Value                                                                 |
|-----------------------|------------------------------------------------------------------------|
| `WP_BASE_URL`         | `https://www.deadikace.com`                                           |
| `WP_USERNAME`         | your WordPress username                                                |
| `WP_APP_PASSWORD`     | the Application Password from Step 2 (with spaces)                     |
| `LLM_PROVIDER`        | `anthropic` or `gemini`                                                |
| `ANTHROPIC_API_KEY`   | your Claude API key (only needed if `LLM_PROVIDER` is `anthropic`)     |
| `GEMINI_API_KEY`      | your Gemini API key (only needed if `LLM_PROVIDER` is `gemini`)        |
| `PEXELS_API_KEY`      | free key from https://www.pexels.com/api/ — used to find article images |
| `TARGET_CATEGORY`     | exact name of the WP category to file posts under, e.g. `Latest News` |
| `POST_STATUS`         | `draft` (recommended to start) or `publish`                            |

You only need to fill in the API key secret matching whichever
`LLM_PROVIDER` you chose — the other one can be left blank or omitted.
Switching providers later is just changing the `LLM_PROVIDER` secret value
and making sure the matching key is set; no code changes needed.

**Getting a Pexels key**: go to https://www.pexels.com/api/, click
"Get Started", sign up (free), and copy the API key shown on your
dashboard. No credit card required, and the free tier (200 requests/hour)
is far more than this agent needs.

## 6. Test it before trusting the schedule

1. Go to the **Actions** tab in your repo
2. Click **"Deadikace Auto-Publish Agent"** in the sidebar
3. Click **"Run workflow"** → **Run workflow** (this is the manual trigger,
   `workflow_dispatch`, already built into the workflow file)
4. Watch the run logs. Check your WordPress site for the new draft/post.
5. Read the article. If anything reads too close to a specific competitor
   piece, or the voice feels off, tell me and I'll refine the prompt in
   `src/draft.py`.

Once you're happy with several runs, flip the `POST_STATUS` secret to
`publish` for full automation.

## 7. How the schedule works

`.github/workflows/publish.yml` runs automatically twice a day (9am and
5pm UTC) via GitHub's cron scheduler. Edit the `cron:` line to change
frequency — https://crontab.guru is useful for building cron expressions.
Each run drafts at most `MAX_ARTICLES_PER_RUN` (default 2) new articles,
skipping topics that look like duplicates of your existing posts.

## 8. How images work

Every article gets 2-4 images. For each one, the agent:
1. Searches **Wikimedia Commons** first for a real, properly-licensed photo
   of the actual band/artist/album named in the article. Commons hosts
   photography that photographers have deliberately released under
   Creative Commons or public-domain licenses -- meaning it's genuinely
   legal to reuse (including commercially), and the agent only accepts
   files with an allowed license (CC0, public domain, or CC-BY variants;
   never non-commercial-only licenses). Every Commons image gets an
   automatic credit linking the photographer and exact license.
2. Falls back to a generic **Pexels** stock photo only if no suitable
   Commons match exists for that specific subject.

**Note on Google Images**: this project deliberately does not scrape
Google Image Search results. An image showing up in a Google search isn't
thereby licensed for reuse -- most band/concert photos there belong to
photographers or agencies who haven't licensed them for republishing, and
adding a photo credit doesn't fix that; a credit states whose work it is,
it doesn't grant permission to use it. Wikimedia Commons is the legal
equivalent: real photos, but restricted to ones explicitly released for
reuse.

**If no images show up at all**, check the Action run's logs for a
`[warn]` line — the most common cause is `PEXELS_API_KEY` not being set
as a GitHub Secret, in which case Commons alone won't always have a match
for every topic.

## 9. How off-genre topics get filtered out

Some competitor feeds (notably Rolling Stone) cover all music genres, not
just rock — so without filtering, pop/rap/K-pop stories can slip in.
Simple keyword matching can't reliably catch this (a headline about "Lil
Baby" or "BTS" doesn't contain the word "rap" or "K-pop" anywhere), so
the agent sends just the batch of candidate headlines (one call per run,
not per article) to the configured LLM and asks it to identify which are
genuinely rock-relevant before drafting begins.


## Tuning knobs (all in `src/config.py` or as GitHub Secrets)

- `MAX_ARTICLES_PER_RUN` — how many articles to generate per run
- `MIN_SOURCE_COUNT` — require a story to appear on 2+ competitor sites
  before writing about it (reduces noise, increases "this is really
  trending" confidence)
- `LOOKBACK_HOURS` — how recent a competitor story must be
- `SITE_VOICE_GUIDELINES` in `config.py` — edit this to sharpen Deadikace's
  voice; the more specific, the better the output

### How topics are prioritized

There's no reliable public way to see competitors' actual article
performance (view counts, shares) — that data isn't published anywhere,
and scraping "most read" widgets (on sites that even have one) breaks
constantly and isn't something to build a pipeline on. Instead, topics are
scored using two honest signals from RSS:

1. **Cross-outlet coverage** — a story covered by 3 outlets is more likely
   to be a big deal than one covered by 1. Weighted by `SOURCE_COUNT_WEIGHT`
   (default 10 points per outlet).
2. **Freshness** — score decays linearly the older a story gets, down to 0
   at the edge of `LOOKBACK_HOURS`. This means a story from 1 hour ago on a
   single site can outrank one from 40 hours ago on three sites — tune
   `SOURCE_COUNT_WEIGHT` up or down to shift that balance.

Optionally, add a `PRIORITY_KEYWORDS` secret — a comma-separated list of
bands/artists/genres you want prioritized whenever they show up in a
headline, e.g. `Metallica,Iron Maiden,thrash metal`. Matching topics get a
scoring boost (`PRIORITY_KEYWORD_BONUS`, default 15 points). Leave it
blank/unset to disable.

There's also an `EXCLUDE_KEYWORDS` secret (comma-separated, defaults to a
sensible list of off-genre terms) that drops obviously off-topic entries
before they're even clustered, as a cheap first-pass filter alongside the
LLM-based relevance check described below.

## Costs to expect

- **GitHub Actions**: free tier covers this easily (a couple minutes of
  runtime per run).
- **Anthropic API**: pay-as-you-go, small (see Step 4).
- **Nothing else** — no other paid services required.
