# Deadikace WordPress API proxy

This Worker forwards only the read-only `/wp-json/` probe and requests under
`/wp-json/wp/v2/` to `https://www.deadikace.com`. All requests require the
`X-Deadikace-Proxy-Token` header. The header is removed before forwarding;
the WordPress `Authorization` header is preserved unchanged.

## Cloudflare setup

1. Create a Worker on Cloudflare's free plan using `src/index.js`.
2. Add an encrypted Worker secret named `PROXY_TOKEN` containing a long,
   randomly generated value. Do not put the value in `wrangler.toml`.
3. Deploy the Worker and copy its `https://...workers.dev` URL.

## GitHub setup

Create these repository Actions secrets:

- `WP_API_BASE_URL`: the Worker URL, without a trailing slash.
- `WP_PROXY_TOKEN`: exactly the same value as Cloudflare `PROXY_TOKEN`.

Keep `WP_BASE_URL=https://www.deadikace.com`; it remains the canonical public
site URL. Existing `WP_USERNAME` and `WP_APP_PASSWORD` secrets are unchanged.

Run the **WordPress Connectivity Check** workflow before the publishing
workflow. It performs only a REST discovery request and an authenticated
one-post read; it never creates or changes WordPress content.
