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

> If you don't see this section, your host may have it disabled, or your
> WordPress version is older than 5.6. Ask your host to enable
> Application Passwords, or install the free "Application Passwords" plugin.

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

## 4. Get an Anthropic API key

1. Go to https://console.anthropic.com
2. Create an API key under **Settings → API Keys**
3. Note: this is billed separately from your Claude.ai subscription — API
   usage is pay-as-you-go. Drafting 2 articles a day is a small cost
   (roughly a few dollars a month at typical article lengths), but check
   current pricing at https://docs.claude.com.

## 5. Add your secrets to GitHub

In your repo: **Settings → Secrets and variables → Actions → New repository secret**.
Add each of these:

| Secret name          | Value                                              |
|-----------------------|----------------------------------------------------|
| `WP_BASE_URL`         | `https://www.deadikace.com`                        |
| `WP_USERNAME`         | your WordPress username                             |
| `WP_APP_PASSWORD`     | the Application Password from Step 2 (with spaces)  |
| `ANTHROPIC_API_KEY`   | your Claude API key from Step 4                     |
| `POST_STATUS`         | `draft` (recommended to start) or `publish`         |

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

## Tuning knobs (all in `src/config.py` or as GitHub Secrets)

- `MAX_ARTICLES_PER_RUN` — how many articles to generate per run
- `MIN_SOURCE_COUNT` — require a story to appear on 2+ competitor sites
  before writing about it (reduces noise, increases "this is really
  trending" confidence)
- `LOOKBACK_HOURS` — how recent a competitor story must be
- `SITE_VOICE_GUIDELINES` in `config.py` — edit this to sharpen Deadikace's
  voice; the more specific, the better the output

## Costs to expect

- **GitHub Actions**: free tier covers this easily (a couple minutes of
  runtime per run).
- **Anthropic API**: pay-as-you-go, small (see Step 4).
- **Nothing else** — no other paid services required.
