const WORDPRESS_ORIGIN = "https://www.deadikace.com";
const TOKEN_HEADER = "X-Deadikace-Proxy-Token";
const FORWARDED_HEADERS = [
  "accept",
  "authorization",
  "content-disposition",
  "content-md5",
  "content-type",
  "if-modified-since",
  "if-none-match",
];

async function tokenMatches(provided, expected) {
  if (!provided || !expected) return false;
  const encoder = new TextEncoder();
  const left = encoder.encode(provided);
  const right = encoder.encode(expected);
  if (left.byteLength !== right.byteLength) return false;
  return crypto.subtle.timingSafeEqual(left, right);
}

function upstreamHeaders(request) {
  const headers = new Headers();
  for (const name of FORWARDED_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set(
    "user-agent",
    "Mozilla/5.0 (compatible; DeadikaceWordPressAgent/1.0; +https://deadikace.com)",
  );
  return headers;
}

export default {
  async fetch(request, env) {
    if (!(await tokenMatches(request.headers.get(TOKEN_HEADER), env.PROXY_TOKEN))) {
      return new Response("Unauthorized", { status: 401 });
    }

    const incoming = new URL(request.url);
    const isRootProbe = incoming.pathname === "/wp-json/";
    const allowed = (isRootProbe && (request.method === "GET" || request.method === "HEAD")) ||
      incoming.pathname.startsWith("/wp-json/wp/v2/");
    if (!allowed) return new Response("Not found", { status: 404 });

    const target = new URL(incoming.pathname + incoming.search, WORDPRESS_ORIGIN);
    return fetch(target, {
      method: request.method,
      headers: upstreamHeaders(request),
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    });
  },
};
